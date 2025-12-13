#!/usr/bin/env python3
# runES_hparam_search.py

import math
from pathlib import Path

import numpy as np
import torch
import wandb
import lovely_tensors as lt

from peft import LoraConfig, get_peft_model

from models.SanaSprint import SanaOneStep, SanaPipelineES
from rewards import (
    load_clip_model_and_processor,
    load_pickscore_model_and_processor,
    compute_all_rewards,
)

from utills import *  # standardize_fitness, get_trainable_params_and_shapes, flatten_params, unflatten_to_params, make_prompt_strip

lt.monkey_patch()

torch.set_float32_matmul_precision("high")
print("[init] Enabled TF32 tensor cores for float32 matmuls (matmul_precision='high').")

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"

# We do hyper-param search for the ONE-STEP backend
BACKEND = "one_step"

ENCODED_PROMPT_PATH = "encoded_prompts_pipeline_train_v3.pt"
TILE_SIZE = 512  # size of each per-prompt tile in the strip


# -------------------------
# EGGROLL-style low-rank noiser
# -------------------------
class EggRollNoiser:
    """
    Low-rank matrix perturbations E = (1/sqrt(r)) * A B^T per parameter matrix,
    following the EGGROLL paper.

    We generate low-rank noise for each trainable parameter (assumed 2D matrices),
    then flatten and concatenate everything into a single vector per candidate.

    Extras:
    - use_antithetic: if True, sample pairs (eps, -eps) to reduce gradient variance.
    """

    def __init__(
            self,
            param_shapes,
            sigma: float,
            lr_scale: float,
            rank: int = 1,
            use_antithetic: bool = False,
    ):
        self.param_shapes = param_shapes  # list of torch.Size
        self.sigma = sigma  # noise std in parameter space
        self.lr_scale = lr_scale  # base LR multiplier
        self.rank = rank  # low-rank rank r
        self.use_antithetic = use_antithetic

        # Total number of parameters (must match flatten_params)
        self.num_params = int(sum(int(np.prod(s)) for s in param_shapes))

    def _sample_low_rank_block(self, pop_size: int, device: str):
        """
        Internal helper: sample low-rank E for all param matrices.

        Returns:
            chunks: list[Tensor] where each Tensor is [pop_size, numel_for_that_param]
        """
        chunks = []
        r = self.rank

        for shape in self.param_shapes:
            numel = int(np.prod(shape))

            if len(shape) == 2:
                # Matrix-shaped parameter: use low-rank E = (1/sqrt(r)) * A B^T
                m, n = shape
                A = torch.randn(pop_size, m, r, device=device)
                B = torch.randn(pop_size, n, r, device=device)
                E = (A @ B.transpose(1, 2)) / math.sqrt(r)  # [pop_size, m, n]
                chunks.append(E.view(pop_size, numel))
            else:
                # Fallback for 1D / other shapes: full Gaussian
                E = torch.randn(pop_size, numel, device=device)
                chunks.append(E)

        return chunks

    def sample_eps(self, pop_size: int, device: str) -> torch.Tensor:
        """
        Sample low-rank perturbations for the whole parameter vector.

        Returns:
            eps: [pop_size, num_params]

        If use_antithetic=True, we generate pairs (e, -e) to reduce variance:
          - For even pop_size: [e_0..e_{K-1}, -e_0..-e_{K-1}]
          - For odd pop_size: same, plus one extra positive sample.
        """
        if not self.use_antithetic:
            # ---- Original non-antithetic path ----
            chunks = self._sample_low_rank_block(pop_size, device)
            eps = torch.cat(chunks, dim=1)  # [pop_size, D]
            return eps

        # ---- Antithetic path ----
        half = pop_size // 2
        base_pop = half if pop_size % 2 == 0 else half + 1  # number of base positive eps

        # Positive half
        chunks_pos = self._sample_low_rank_block(base_pop, device)
        eps_pos = torch.cat(chunks_pos, dim=1)  # [base_pop, D]

        # Negative half
        eps_neg = -eps_pos.clone()  # [base_pop, D]

        # Build [e_0..e_{half-1}, -e_0..-e_{half-1}]
        eps = torch.cat([eps_pos[:half], eps_neg[:half]], dim=0)  # [2*half, D]

        # If odd population size, append one extra positive sample
        if pop_size % 2 == 1:
            eps = torch.cat([eps, eps_pos[half:half + 1]], dim=0)  # [2*half+1, D]

        assert eps.shape[0] == pop_size
        return eps

    def convert_fitnesses(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """
        Map raw scalar rewards -> standardized fitness values.
        Here we just do (r - mean) / std, similar to ES fitness shaping.
        """
        return standardize_fitness(raw_scores)

    def do_update(self, theta: torch.Tensor, eps: torch.Tensor, fitnesses: torch.Tensor):
        """
        EGGROLL-style ES update in parameter space:

            θ_{t+1} = θ_t + α * E[ f * ε ]

        where ε is our low-rank noise (flattened) and
        α is lr_scale / σ (closer to the paper's notation).

        Args:
            theta:     [D] current parameter vector
            eps:       [pop_size, D] sampled noise
            fitnesses: [pop_size] standardized fitness
        """
        sigma = self.sigma

        lr = self.lr_scale * self.sigma
        # eps: [pop_size, D]
        # fitnesses: [pop_size]
        grad_est = (fitnesses.unsqueeze(1) * eps).mean(dim=0)  # [D]
        theta_new = theta + lr * grad_est
        return theta_new


# -------------------------
# Helper: save latest LoRA checkpoint
# -------------------------
def save_lora_checkpoint(
        theta: torch.Tensor,
        es_model,
        lora_params,
        lora_shapes,
        save_dir: Path,
        meta_path: Path,
        model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules,
        backend: str,
        epoch: int,
        stats: dict,
):
    """
    Overwrite save_dir with the *latest* LoRA weights.
    Called every N epochs (e.g., every 10) and at the very end.
    """
    print(f"[ckpt] Saving latest LoRA at epoch {epoch} -> {save_dir}")
    theta_cpu = theta.detach().cpu()

    # Load theta back into the transformer and save PEFT adapter
    unflatten_to_params(theta.to(DEVICE), lora_params, lora_shapes)
    es_model.transformer.save_pretrained(save_dir)

    torch.save(
        {
            "theta_latest": theta_cpu,
            "epoch": epoch,
            "summary_mean_reward": stats.get("summary/mean_reward", float("nan")),
            "MODEL_NAME": model_name,
            "LORA_R": lora_r,
            "LORA_ALPHA": lora_alpha,
            "LORA_DROPOUT": lora_dropout,
            "LORA_TARGET_MODULES": lora_target_modules,
            "BACKEND": backend,
        },
        meta_path,
    )


# -------------------------
# ES step (LoRA-only, EGGROLL-style low-rank noise)
# -------------------------
@torch.no_grad()
def es_step(
        theta: torch.Tensor,
        es_model,  # SanaOneStep or SanaPipelineES
        lora_params,
        lora_shapes,
        base_prompt_embeds: torch.Tensor,  # [P, seq, dim]
        base_attention_mask: torch.Tensor,  # [P, seq]
        prompts_list,  # List[str] length P
        clip_model,
        clip_processor,
        pick_model,
        pickscore_processor,
        noiser: EggRollNoiser,
        epoch: int,
        save_dir: Path,
        mix_weights,
        batch_size: int,
        pop_size: int,
        seed: int,
        guidance_scale: float,
        theta_max_norm: float,
):
    """
    seed: shared seed for the *whole* ES step.
    All individuals and prompts in this step use the same seed, matching
    the "fixed noise per iteration" idea in the EGGROLL paper.
    """
    torch.manual_seed(seed)

    num_prompts = base_prompt_embeds.shape[0]

    # Put all embeddings on device once
    prompt_embeds_all = base_prompt_embeds.to(DEVICE)
    prompt_attention_all = base_attention_mask.to(DEVICE)

    # Low-rank noise for population (EGGROLL-style)
    eps = noiser.sample_eps(pop_size, DEVICE)  # [pop_size, D]

    rewards_combined = []
    rewards_aes = []
    rewards_text = []
    rewards_noart = []
    rewards_pick = []
    all_images = []  # list[pop_size] of list[num_prompts] of PIL.Image

    # For per-prompt logging: list[P] of list[pop_size] of scalars
    per_prompt_combined = [[] for _ in range(num_prompts)]

    use_batch_pipeline = isinstance(es_model, SanaPipelineES) or hasattr(
        es_model, "generate_one_batch"
    )

    for k in range(pop_size):
        # θ_k = θ + σ * E_k
        theta_k = theta + noiser.sigma * eps[k]
        unflatten_to_params(theta_k, lora_params, lora_shapes)

        cand_comb = []
        cand_aes = []
        cand_txt = []
        cand_noart = []
        cand_pick = []
        cand_images_for_logging = []

        if use_batch_pipeline:
            # ---------- PIPELINE BACKEND: one call over ALL prompts ----------
            images_all, _ = es_model.generate_one_batch(
                prompt_embeds=prompt_embeds_all,  # [P, seq, dim]
                prompt_attention_mask=prompt_attention_all,  # [P, seq]
                seed=seed,  # SAME seed for all indiv in this step
                guidance_scale=guidance_scale,
                width_latent=32,
                height_latent=32,
            )
            # images_all is a list of length == num_prompts

        # Evaluate candidate k on ALL prompts, average reward over prompts
        for p_idx in range(num_prompts):
            prompt_text = (
                prompts_list[p_idx] if prompts_list is not None else f"prompt_{p_idx}"
            )

            if use_batch_pipeline:
                # One image per prompt for now (BATCH_SIZE=1 case)
                images = [images_all[p_idx]]
            else:
                # ---------- ONE-STEP BACKEND: per-prompt call ----------
                prompt_embeds = prompt_embeds_all[p_idx: p_idx + 1]  # [1, seq, dim]
                prompt_attention_mask = prompt_attention_all[p_idx: p_idx + 1]  # [1, seq]

                b = batch_size
                if b > 1:
                    prompt_embeds_b = prompt_embeds.expand(b, -1, -1).contiguous()
                    prompt_attention_mask_b = prompt_attention_mask.expand(b, -1).contiguous()
                else:
                    prompt_embeds_b = prompt_embeds
                    prompt_attention_mask_b = prompt_attention_mask

                images, _latents = es_model.generate(
                    prompt_embeds=prompt_embeds_b,
                    prompt_attention_mask=prompt_attention_mask_b,
                    latents=None,
                    seed=seed,  # SAME seed for all indiv & prompts in this step
                    guidance_scale=guidance_scale,
                    width_latent=32,
                    height_latent=32,
                )

            reward_dict = compute_all_rewards(
                images,
                prompt_text=prompt_text,
                clip_model=clip_model,
                clip_processor=clip_processor,
                mix_weights=mix_weights,
                pickscore_model=pick_model,
                pickscore_processor=pickscore_processor,
            )

            # Per-(candidate, prompt) scalar reward
            r_p_comb = reward_dict["combined"]

            cand_comb.append(r_p_comb)
            cand_aes.append(reward_dict["clip_aesthetic"])
            cand_txt.append(reward_dict["clip_text"])
            cand_noart.append(reward_dict["no_artifacts"])
            cand_pick.append(reward_dict["pickscore"])

            per_prompt_combined[p_idx].append(r_p_comb)  # for per-prompt stats

            # For logging, keep one representative image per prompt
            if use_batch_pipeline:
                cand_images_for_logging.append(images[0])  # images[0] is images_all[p_idx]
            else:
                if len(images) > 0:
                    cand_images_for_logging.append(images[0])
                else:
                    cand_images_for_logging.append(None)

        # Aggregate candidate reward across prompts (mean)
        r_comb = torch.stack(cand_comb).mean()
        r_aes = torch.stack(cand_aes).mean()
        r_txt = torch.stack(cand_txt).mean()
        r_noart = torch.stack(cand_noart).mean()
        r_pick = torch.stack(cand_pick).mean()

        rewards_combined.append(r_comb)
        rewards_aes.append(r_aes)
        rewards_text.append(r_txt)
        rewards_noart.append(r_noart)
        rewards_pick.append(r_pick)
        all_images.append(cand_images_for_logging)

        print(
            f"  indiv {k:02d} | "
            f"R_comb={r_comb.item():.4f}, "
            f"aesthetic={r_aes.item():.4f}, "
            f"text_align={r_txt.item():.4f}, "
            f"no_artifacts={r_noart.item():.4f}, "
            f"pickscore={r_pick.item():.4f} "
            f"(averaged over {num_prompts} prompts)"
        )

    rewards_combined = torch.stack(rewards_combined, dim=0)  # [pop_size]
    rewards_aes = torch.stack(rewards_aes, dim=0)
    rewards_text = torch.stack(rewards_text, dim=0)
    rewards_noart = torch.stack(rewards_noart, dim=0)
    rewards_pick = torch.stack(rewards_pick, dim=0)

    # Filter NaNs based on combined reward
    finite_mask = torch.isfinite(rewards_combined)
    if not finite_mask.any():
        print("⚠ All rewards NaN/Inf, skipping update.")
        stats = {
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "max_reward": float("nan"),
            "min_reward": float("nan"),
            "aesthetic_mean": float("nan"),
            "aesthetic_std": float("nan"),
            "clip_text_mean": float("nan"),
            "clip_text_std": float("nan"),
            "no_artifacts_mean": float("nan"),
            "no_artifacts_std": float("nan"),
            "pickscore_mean": float("nan"),
            "pickscore_std": float("nan"),
            "fitness_mean": float("nan"),
            "fitness_std": float("nan"),
            "summary/mean_reward": float("nan"),
            "summary/max_reward": float("nan"),
            "summary/min_reward": float("nan"),
        }
        for p_idx in range(num_prompts):
            stats[f"prompt_{p_idx}/mean_reward"] = float("nan")
            stats[f"prompt_{p_idx}/max_reward"] = float("nan")
            stats[f"prompt_{p_idx}/min_reward"] = float("nan")
        return theta, stats, {"best": None, "median": None, "worst": None}, None

    # Save best/median/worst according to combined reward (for visualization only)
    best_img = median_img = worst_img = None
    if finite_mask.all():
        sorted_vals, sorted_idx = torch.sort(rewards_combined)
        worst_idx = sorted_idx[0].item()
        best_idx = sorted_idx[-1].item()
        median_idx = sorted_idx[len(sorted_idx) // 2].item()

        epoch_dir = save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        best_strip = make_prompt_strip(all_images[best_idx], num_prompts)
        median_strip = make_prompt_strip(all_images[median_idx], num_prompts)
        worst_strip = make_prompt_strip(all_images[worst_idx], num_prompts)

        if best_strip is not None:
            best_strip.save(epoch_dir / "best.png")
            best_img = best_strip
        if median_strip is not None:
            median_strip.save(epoch_dir / "median.png")
            median_img = median_strip
        if worst_strip is not None:
            worst_strip.save(epoch_dir / "worst.png")
            worst_img = worst_strip

    # Restrict tensors to finite indices
    rewards_combined = rewards_combined[finite_mask]
    rewards_aes = rewards_aes[finite_mask]
    rewards_text = rewards_text[finite_mask]
    rewards_noart = rewards_noart[finite_mask]
    rewards_pick = rewards_pick[finite_mask]
    eps = eps[finite_mask]

    # ES update
    fitnesses = noiser.convert_fitnesses(rewards_combined)
    theta = noiser.do_update(theta, eps, fitnesses)

    # ---- Parameter-norm projection (cap θ-norm) ----
    if theta_max_norm is not None and theta_max_norm > 0:
        with torch.no_grad():
            tnorm = theta.norm()
            if tnorm > theta_max_norm:
                scale = theta_max_norm / (tnorm + 1e-8)
                theta = theta * scale

    # --------- Per-prompt stats for W&B ---------
    per_prompt_stats = {}
    for p_idx in range(num_prompts):
        vals = torch.stack(per_prompt_combined[p_idx])
        vals = vals[finite_mask]

        if vals.numel() == 0:
            mean_p = max_p = min_p = float("nan")
        else:
            mean_p = vals.mean().item()
            max_p = vals.max().item()
            min_p = vals.min().item()

        per_prompt_stats[f"prompt_{p_idx}/mean_reward"] = mean_p
        per_prompt_stats[f"prompt_{p_idx}/max_reward"] = max_p
        per_prompt_stats[f"prompt_{p_idx}/min_reward"] = min_p

    # --------- Summary stats ---------
    comb_mean = rewards_combined.mean().item()
    comb_std = rewards_combined.std().item() if rewards_combined.numel() > 1 else 0.0
    comb_max = rewards_combined.max().item()
    comb_min = rewards_combined.min().item()

    aes_mean = rewards_aes.mean().item()
    aes_std = rewards_aes.std().item() if rewards_aes.numel() > 1 else 0.0
    txt_mean = rewards_text.mean().item()
    txt_std = rewards_text.std().item() if rewards_text.numel() > 1 else 0.0
    noart_mean = rewards_noart.mean().item()
    noart_std = rewards_noart.std().item() if rewards_noart.numel() > 1 else 0.0
    pick_mean = rewards_pick.mean().item()
    pick_std = rewards_pick.numel() > 1 and rewards_pick.std().item() or 0.0
    if isinstance(pick_std, bool):
        pick_std = 0.0

    fit_mean = fitnesses.mean().item()
    fit_std = fitnesses.std().item() if fitnesses.numel() > 1 else 0.0

    print(
        f"ES step: R_comb_mean={comb_mean:.4f}, std={comb_std:.4f}, "
        f"max={comb_max:.4f}, min={comb_min:.4f}, "
        f"aesthetic_mean={aes_mean:.4f}, "
        f"text_align_mean={txt_mean:.4f}, "
        f"no_artifacts_mean={noart_mean:.4f}, "
        f"pickscore_mean={pick_mean:.4f} "
        f"(averaged over {num_prompts} prompts)"
    )

    stats = {
        "mean_reward": comb_mean,
        "std_reward": comb_std,
        "max_reward": comb_max,
        "min_reward": comb_min,
        "aesthetic_mean": aes_mean,
        "aesthetic_std": aes_std,
        "clip_text_mean": txt_mean,
        "clip_text_std": txt_std,
        "no_artifacts_mean": noart_mean,
        "no_artifacts_std": noart_std,
        "pickscore_mean": pick_mean,
        "pickscore_std": pick_std,
        "fitness_mean": fit_mean,
        "fitness_std": fit_std,
        "summary/mean_reward": comb_mean,
        "summary/max_reward": comb_max,
        "summary/min_reward": comb_min,
    }
    stats.update(per_prompt_stats)

    img_dict = {"best": best_img, "median": median_img, "worst": worst_img}

    # Return the finite combined rewards so we can log a histogram in main()
    rewards_for_hist = rewards_combined.detach().cpu()

    return theta, stats, img_dict, rewards_for_hist


# -------------------------
# One full ES run for a single hyper-param config
# -------------------------
def run_single_config(config_id: int, cfg: dict, prompt_data: dict):
    """
    cfg keys:
      - sigma
      - lr_scale
      - use_antithetic
    """
    # =========================
    # Fixed hyperparameters for the search
    # =========================
    MODEL_NAME = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"

    # LoRA
    LORA_R = 2
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.0
    LORA_TARGET_MODULES = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "linear_1",
        "linear_2",
        "proj_out",
        "linear",
    ]

    # ES / EGGROLL (search dimensions below)
    SIGMA = cfg["sigma"]
    LR_SCALE = cfg["lr_scale"]
    USE_ANTITHETIC = cfg["use_antithetic"]

    POP_SIZE = 128  # fixed as requested
    NUM_EPOCHS = 500  # 100 ES steps
    BATCH_SIZE = 1
    THETA_MAX_NORM = 40.0  # cap LoRA norm (starts ~10)
    EGG_RANK = 1

    # Reward mixing (still PickScore-only)
    W_AESTHETIC = 0.0
    W_TEXT_ALIGN = 0.0
    W_NO_ARTIFACTS = 0.0
    W_PICKSCORE = 1.0
    MIX_WEIGHTS = (W_AESTHETIC, W_TEXT_ALIGN, W_NO_ARTIFACTS, W_PICKSCORE)

    GUIDANCE_SCALE = 4.5

    WANDB_PROJECT = "SanaSprintOneStep-ES-Search-v2_1"
    run_name = (
        f"cfg{config_id}_backend={BACKEND}"
        f"_sigma={SIGMA:.0e}_lr={LR_SCALE:.0e}_ant={int(USE_ANTITHETIC)}"
    )

    # Root save dir for this config
    BASE_SAVE_DIR = Path(f"es_search_{BACKEND}_1")
    SAVE_DIR = BASE_SAVE_DIR / f"cfg{config_id}_sigma{SIGMA:.0e}_lr{LR_SCALE:.0e}_ant{int(USE_ANTITHETIC)}"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    best_lora_dir = SAVE_DIR / "latest_lora"
    best_lora_dir.mkdir(parents=True, exist_ok=True)
    meta_path = SAVE_DIR / "latest_lora_meta.pt"

    # =========================
    # Unpack encoded prompts (shared across configs)
    # =========================
    base_prompt_embeds = prompt_data["prompt_embeds"]  # [P, seq, dim]
    base_attention_mask = prompt_data["prompt_attention_mask"]  # [P, seq]
    prompts_list = prompt_data.get("prompts", None)
    num_prompts = base_prompt_embeds.shape[0]

    print(f"\n[cfg {config_id}] num_prompts: {num_prompts}")
    print(f"[cfg {config_id}] prompt_embeds: {base_prompt_embeds.shape}, {base_prompt_embeds.dtype}")
    print(f"[cfg {config_id}] attention_mask: {base_attention_mask.shape}, {base_attention_mask.dtype}")

    # =========================
    # Init ES-compatible Sana model (ONE-STEP)
    # =========================
    print(f"[cfg {config_id}] Using SanaOneStep backend.")
    es_model = SanaOneStep(
        model_name=MODEL_NAME,
        device=DEVICE,
        DTYPE=torch.float16,  # latents dtype; transformer/vae are float32 inside
        sigma_data=0.5,
    )

    # =========================
    # Compile transformer (if supported, *before* LoRA)
    # =========================
    if torch.cuda.is_available():
        print(f"[cfg {config_id}] Compiling transformer with torch.compile(mode='max-autotune')...")
        try:
            es_model.transformer = torch.compile(
                es_model.transformer,
                mode="max-autotune",
                fullgraph=True,
            )
            print(f"[cfg {config_id}] torch.compile SUCCESS – compiled transformer will be used.")
            es_model.vae = torch.compile(
                es_model.vae,
                mode="max-autotune",
                fullgraph=True,
            )
            print(f"[cfg {config_id}] torch.compile SUCCESS – compiled vae will be used.")

        except Exception as e:
            print(f"[cfg {config_id}] torch.compile FAILED ({e}). Using eager mode.")
    else:
        print(f"[cfg {config_id}] torch.compile skipped (DEVICE={DEVICE} is not CUDA).")

    # =========================
    # Attach LoRA
    # =========================
    print(f"[cfg {config_id}] Attaching LoRA adapters via PEFT...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    es_model.transformer = get_peft_model(es_model.transformer, lora_config)
    es_model.transformer.to(DEVICE)
    es_model.transformer.eval()

    # =========================
    # Collect LoRA params
    # =========================
    lora_params, lora_shapes = get_trainable_params_and_shapes(es_model.transformer)
    theta = flatten_params(lora_params).to(DEVICE)
    theta_init = theta.clone().detach()
    print(f"[cfg {config_id}] LoRA trainable parameters: {theta.numel():,}")
    print(
        f"[cfg {config_id}] EGGROLL noiser param count: {int(sum(np.prod(s) for s in lora_shapes)):,}"
    )
    print(f"[cfg {config_id}] theta_init_norm = {theta_init.norm().item():.4f}")

    # =========================
    # Init CLIP + PickScore
    # =========================
    clip_model, clip_processor = load_clip_model_and_processor(DEVICE)
    pick_model, pickscore_processor = load_pickscore_model_and_processor(DEVICE)

    # =========================
    # Init EGGROLL noiser + W&B
    # =========================
    noiser = EggRollNoiser(
        param_shapes=lora_shapes,
        sigma=SIGMA,
        lr_scale=LR_SCALE,
        rank=EGG_RANK,
        use_antithetic=USE_ANTITHETIC,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "config_id": config_id,
            "model_name": MODEL_NAME,
            "backend": BACKEND,
            "sigma": SIGMA,
            "lr_scale": LR_SCALE,
            "use_antithetic": USE_ANTITHETIC,
            "pop_size": POP_SIZE,
            "num_epochs": NUM_EPOCHS,
            "guidance_scale": GUIDANCE_SCALE,
            "device": DEVICE,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "mix_weights": MIX_WEIGHTS,
            "egg_rank": EGG_RANK,
            "num_params": noiser.num_params,
            "num_prompts": num_prompts,
            "theta_max_norm": THETA_MAX_NORM,
        },
    )

    # =========================
    # ES loop
    # =========================
    last_epoch_stats = None

    for epoch in range(NUM_EPOCHS):
        print(f"\n[cfg {config_id}] === Epoch {epoch} ({BACKEND}) ===")
        theta, stats, img_dict, rewards_vec = es_step(
            theta=theta,
            es_model=es_model,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            base_prompt_embeds=base_prompt_embeds,
            base_attention_mask=base_attention_mask,
            prompts_list=prompts_list,
            clip_model=clip_model,
            clip_processor=clip_processor,
            pick_model=pick_model,
            pickscore_processor=pickscore_processor,
            noiser=noiser,
            epoch=epoch,
            save_dir=SAVE_DIR,
            mix_weights=MIX_WEIGHTS,
            batch_size=BATCH_SIZE,
            pop_size=POP_SIZE,
            seed=epoch,  # shared seed per ES step
            guidance_scale=GUIDANCE_SCALE,
            theta_max_norm=THETA_MAX_NORM,
        )

        last_epoch_stats = stats

        # ===== LoRA parameter stats & distributions =====
        with torch.no_grad():
            delta_theta = theta - theta_init
            theta_norm = theta.norm().item()

            delta_theta_norm = delta_theta.norm().item()

            # Flatten to CPU for histogram logging
            theta_flat = theta.detach().cpu().view(-1)
            delta_flat = delta_theta.detach().cpu().view(-1)

            # Subsample for histograms to avoid huge logs
            max_hist_params = 50_000
            if theta_flat.numel() > max_hist_params:
                idx = torch.randperm(theta_flat.numel())[:max_hist_params]
                theta_hist_vals = theta_flat[idx]
                delta_hist_vals = delta_flat[idx]
            else:
                theta_hist_vals = theta_flat
                delta_hist_vals = delta_flat

            # Simple scalar stats
            lora_mean_abs = float(theta_flat.abs().mean().item())
            lora_delta_mean_abs = float(delta_flat.abs().mean().item())

        stats["theta_norm"] = theta_norm
        stats["delta_theta_norm"] = delta_theta_norm
        stats["lora/mean_abs"] = lora_mean_abs
        stats["lora/delta_mean_abs"] = lora_delta_mean_abs

        # ---- Save latest LoRA every 10 epochs ----
        if (epoch + 1) % 10 == 0:
            save_lora_checkpoint(
                theta=theta,
                es_model=es_model,
                lora_params=lora_params,
                lora_shapes=lora_shapes,
                save_dir=best_lora_dir,
                meta_path=meta_path,
                model_name=MODEL_NAME,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                lora_target_modules=LORA_TARGET_MODULES,
                backend=BACKEND,
                epoch=epoch + 1,
                stats=stats,
            )

        # ---- W&B images ----
        wandb_imgs = {}
        for key, img in img_dict.items():
            if img is not None:
                wandb_imgs[f"images/{key}"] = wandb.Image(img, caption=f"{key} epoch {epoch}")

        # ---- Build W&B log payload ----
        log_payload = {
            "epoch": epoch,
            **stats,
            **wandb_imgs,
        }

        # 🔍 Log full reward distribution as a histogram
        if rewards_vec is not None and rewards_vec.numel() > 0:
            log_payload["reward_hist"] = wandb.Histogram(
                rewards_vec.detach().cpu().numpy()
            )

        # 🔍 Log LoRA parameter distributions
        log_payload["lora/weights_hist"] = wandb.Histogram(
            theta_hist_vals.numpy()
        )
        log_payload["lora/delta_hist"] = wandb.Histogram(
            delta_hist_vals.numpy()
        )

        wandb.log(log_payload, step=epoch)

    # =========================
    # Final save (if last epoch wasn't a multiple of 10)
    # =========================
    if NUM_EPOCHS % 10 != 0 and last_epoch_stats is not None:
        save_lora_checkpoint(
            theta=theta,
            es_model=es_model,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            save_dir=best_lora_dir,
            meta_path=meta_path,
            model_name=MODEL_NAME,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            lora_target_modules=LORA_TARGET_MODULES,
            backend=BACKEND,
            epoch=NUM_EPOCHS,
            stats=last_epoch_stats,
        )

    # =========================
    # Finish
    # =========================
    print(
        f"\n[cfg {config_id}] Training done. Latest LoRA checkpoint (saved every 10 epochs / at end) is at: {best_lora_dir}"
    )
    print(f"[cfg {config_id}] Meta checkpoint at: {meta_path}")

    run.finish()


# -------------------------
# Hyper-parameter search driver
# -------------------------
def main():
    # Small but sensible search space; all use POP_SIZE=16 and 100 steps
    SEARCH_CONFIGS = [
        # Each config: sigma, lr_scale, use_antithetic
        # Effective lr = sigma * lr_scale  (shown in comments)

        # Around your old stable setting (sigma=1e-2, lr≈1e-2)
        {"sigma": 1e-2, "lr_scale": 1.0, "use_antithetic": True},  # lr = 1e-2  (old-style baseline)
    ]

    # Load encoded prompts once and reuse
    print(f"[init] loading encoded prompts from {ENCODED_PROMPT_PATH}...")
    prompt_data = torch.load(ENCODED_PROMPT_PATH, map_location="cpu")

    for cfg_id, cfg in enumerate(SEARCH_CONFIGS):
        print("\n" + "=" * 80)
        print(f"[SEARCH] Starting config {cfg_id}: {cfg}")
        print("=" * 80)
        run_single_config(cfg_id, cfg, prompt_data)


if __name__ == "__main__":
    main()
