#!/usr/bin/env python3
# runES.py

import os
from pathlib import Path
import math

import torch
from torch import nn
import numpy as np
import wandb
import lovely_tensors as lt

from peft import LoraConfig, get_peft_model
from PIL import Image  # ✅ for resizing & composing strips

from models.SanaSprintOneStep import SanaTransformerOneStepES  # your Sana wrapper
from rewards import (
    load_clip_model_and_processor,
    load_pickscore_model_and_processor,
    compute_all_rewards,
)

from utills import *

lt.monkey_patch()


torch.set_float32_matmul_precision("high")
print("[init] Enabled TF32 tensor cores for float32 matmuls (matmul_precision='high').")

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"
ENCODED_PROMPT_PATH = "encoded_prompts.pt"

TILE_SIZE = 256  # size of each per-prompt tile in the strip

# -------------------------
# EGGROLL-style low-rank noiser
# -------------------------
class EggRollNoiser:
    """
    Low-rank matrix perturbations E = (1/sqrt(r)) * A B^T per parameter matrix,
    following the EGGROLL paper.

    We generate low-rank noise for each trainable parameter (assumed 2D matrices),
    then flatten and concatenate everything into a single vector per candidate.
    """

    def __init__(
        self,
        param_shapes,
        sigma: float,
        lr_scale: float,
        rank: int = 1,
    ):
        self.param_shapes = param_shapes  # list of torch.Size
        self.sigma = sigma
        self.lr_scale = lr_scale
        self.rank = rank

        # Total number of parameters (must match flatten_params)
        self.num_params = int(sum(int(np.prod(s)) for s in param_shapes))

    def sample_eps(self, pop_size: int, device: str) -> torch.Tensor:
        """
        Sample low-rank perturbations for the whole parameter vector.

        Returns:
            eps: [pop_size, num_params]
        """
        chunks = []
        r = self.rank

        for shape in self.param_shapes:
            numel = int(np.prod(shape))
            if len(shape) == 2:
                m, n = shape

                # A: [pop_size, m, r]
                # B: [pop_size, n, r]
                A = torch.randn(pop_size, m, r, device=device)
                B = torch.randn(pop_size, n, r, device=device)

                # E: [pop_size, m, n] = (1/sqrt(r)) * A @ B^T
                E = (A @ B.transpose(1, 2)) / math.sqrt(r)
                chunks.append(E.view(pop_size, numel))
            else:
                # Fallback: full Gaussian (for non-2D params, if any)
                E = torch.randn(pop_size, numel, device=device)
                chunks.append(E)

        eps = torch.cat(chunks, dim=1)  # [pop_size, D]
        return eps

    def convert_fitnesses(self, raw_scores: torch.Tensor) -> torch.Tensor:
        # Standardized fitness (zero mean, unit std)
        return standardize_fitness(raw_scores)

    def do_update(self, theta: torch.Tensor, eps: torch.Tensor, fitnesses: torch.Tensor):
        """
        EGGROLL-style ES update in parameter space:

            θ_{t+1} = θ_t + α * E[ f * E ]

        where E is our low-rank noise flattened, and α is lr_scale * σ.
        """
        sigma = self.sigma
        # Effective learning rate; you can tune lr_scale in config
        lr = self.lr_scale * sigma

        # eps: [pop_size, D]
        # fitnesses: [pop_size]
        grad_est = (fitnesses.unsqueeze(1) * eps).mean(dim=0)  # [D]
        theta_new = theta + lr * grad_est
        return theta_new


# -------------------------
# ES step (LoRA-only, EGGROLL-style low-rank noise)
# -------------------------
@torch.no_grad()
def es_step(
    theta: torch.Tensor,
    sana_es: SanaTransformerOneStepES,
    lora_params,
    lora_shapes,
    base_prompt_embeds: torch.Tensor,       # [P, seq, dim]
    base_attention_mask: torch.Tensor,      # [P, seq]
    prompts_list,                           # List[str] length P
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

        # Evaluate candidate k on ALL prompts, average reward over prompts
        for p_idx in range(num_prompts):
            prompt_embeds = prompt_embeds_all[p_idx: p_idx + 1]       # [1, seq, dim]
            prompt_attention_mask = prompt_attention_all[p_idx: p_idx + 1]  # [1, seq]
            prompt_text = (
                prompts_list[p_idx] if prompts_list is not None else f"prompt_{p_idx}"
            )

            # If you want >1 images per prompt, expand here
            b = batch_size
            if b > 1:
                prompt_embeds_b = prompt_embeds.expand(b, -1, -1).contiguous()
                prompt_attention_mask_b = prompt_attention_mask.expand(b, -1).contiguous()
            else:
                prompt_embeds_b = prompt_embeds
                prompt_attention_mask_b = prompt_attention_mask

            images, _latents = sana_es.sana_one_step_trigflow(
                prompt_embeds=prompt_embeds_b,
                prompt_attention_mask=prompt_attention_mask_b,
                latents=None,
                seed=seed * 1000 + k * 10 + p_idx,  # different seed per (candidate, prompt)
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
        }
        # Also add empty per-prompt / summary sections
        stats.update(
            {
                "summary/mean_reward": float("nan"),
                "summary/max_reward": float("nan"),
                "summary/min_reward": float("nan"),
            }
        )
        for p_idx in range(num_prompts):
            stats[f"prompt_{p_idx}/mean_reward"] = float("nan")
            stats[f"prompt_{p_idx}/max_reward"] = float("nan")
            stats[f"prompt_{p_idx}/min_reward"] = float("nan")
        return theta, stats, {"best": None, "median": None, "worst": None}

    # Save best/median/worst according to combined reward
    best_img = median_img = worst_img = None
    if finite_mask.all():
        sorted_vals, sorted_idx = torch.sort(rewards_combined)
        worst_idx = sorted_idx[0].item()
        best_idx = sorted_idx[-1].item()
        median_idx = sorted_idx[len(sorted_idx) // 2].item()

        epoch_dir = save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # For each candidate, build a horizontal strip of per-prompt images
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

    # Restrict tensors to finite indices (candidates with valid aggregated reward)
    rewards_combined = rewards_combined[finite_mask]
    rewards_aes = rewards_aes[finite_mask]
    rewards_text = rewards_text[finite_mask]
    rewards_noart = rewards_noart[finite_mask]
    rewards_pick = rewards_pick[finite_mask]
    eps = eps[finite_mask]

    # ES update (EGGROLL-style, with low-rank noise)
    fitnesses = noiser.convert_fitnesses(rewards_combined)
    theta = noiser.do_update(theta, eps, fitnesses)

    # --------- Per-prompt stats for W&B ---------
    per_prompt_stats = {}
    for p_idx in range(num_prompts):
        # tensor of shape [pop_size] in candidate order
        vals = torch.stack(per_prompt_combined[p_idx])  # [pop_size]
        # apply same finite_mask to be consistent with ES update
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

    # --------- Summary stats (over candidates, with aggregated reward) ---------
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
    pick_std = rewards_pick.std().item() if rewards_pick.numel() > 1 else 0.0

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
        # original summary (backwards compatible)
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
        # explicit "summary" section
        "summary/mean_reward": comb_mean,
        "summary/max_reward": comb_max,
        "summary/min_reward": comb_min,
    }

    # Add per-prompt stats (these become separate sections in W&B)
    stats.update(per_prompt_stats)

    img_dict = {"best": best_img, "median": median_img, "worst": worst_img}
    return theta, stats, img_dict


def main():
    # =========================
    # Hyperparameters
    # =========================
    MODEL_NAME = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"

    # LoRA
    LORA_R = 2
    LORA_ALPHA = 16
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

    # ES / EGGROLL
    SIGMA = 1e-2
    LR_SCALE = 1.0
    POP_SIZE = 32
    NUM_EPOCHS = 300
    BATCH_SIZE = 1
    THETA_MAX_NORM = 5.0
    EGG_RANK = 1  # rank r for low-rank noise

    # Reward mixing: (aesthetic, alignment, no_artifacts, pickscore)
    W_AESTHETIC = 0.0
    W_TEXT_ALIGN = 0.0
    W_NO_ARTIFACTS = 0.0
    W_PICKSCORE = 1.0
    MIX_WEIGHTS = (W_AESTHETIC, W_TEXT_ALIGN, W_NO_ARTIFACTS, W_PICKSCORE)

    GUIDANCE_SCALE = 4.5

    WANDB_PROJECT = "SanaSprint-ES-multiprompt"
    WANDB_RUN_NAME = "sana_es_eggroll_pickscore"

    SAVE_DIR = Path("es_runs_eggroll_pickscore_multiprompt")

    # =========================
    # Load encoded prompts
    # =========================
    print(f"[init] loading encoded prompts from {ENCODED_PROMPT_PATH}...")
    prompt_data = torch.load(ENCODED_PROMPT_PATH, map_location="cpu")
    base_prompt_embeds = prompt_data["prompt_embeds"]           # [P, seq, dim]
    base_attention_mask = prompt_data["prompt_attention_mask"]  # [P, seq]
    prompts_list = prompt_data.get("prompts", None)
    num_prompts = base_prompt_embeds.shape[0]

    print(f"  num_prompts: {num_prompts}")
    print(f"  prompt_embeds: {base_prompt_embeds.shape}, {base_prompt_embeds.dtype}")
    print(f"  attention_mask: {base_attention_mask.shape}, {base_attention_mask.dtype}")
    if prompts_list is not None:
        for i, p in enumerate(prompts_list):
            print(f"    [{i:02d}] {p}")

    # =========================
    # Init Sana wrapper
    # =========================
    sana_es = SanaTransformerOneStepES(
        model_name=MODEL_NAME,
        device=DEVICE,
        DTYPE=torch.float16,
        sigma_data=0.5,
    )

    # =========================
    # Compile transformer (if supported, *before* LoRA)
    # =========================
    if torch.cuda.is_available():
        print("[compile] Compiling Sana transformer with torch.compile(mode='max-autotune')...")
        try:
            sana_es.transformer = torch.compile(
                sana_es.transformer,
                mode="max-autotune",
                fullgraph=True,  # you can change to False if this causes issues
            )
            print("[compile] torch.compile SUCCESS – compiled transformer will be used.")
        except Exception as e:
            print(f"[compile] torch.compile FAILED ({e}). Falling back to eager mode.")
    else:
        print(f"[compile] torch.compile skipped (DEVICE={DEVICE} is not CUDA).")

    # =========================
    # Attach LoRA
    # =========================
    print("[init] attaching LoRA adapters via PEFT...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    sana_es.transformer = get_peft_model(sana_es.transformer, lora_config)
    sana_es.transformer.to(DEVICE)
    sana_es.transformer.eval()

    # =========================
    # Collect LoRA params
    # =========================
    lora_params, lora_shapes = get_trainable_params_and_shapes(sana_es.transformer)
    theta = flatten_params(lora_params).to(DEVICE)
    theta_init = theta.clone().detach()
    print(f"LoRA trainable parameters: {theta.numel():,}")
    print(
        f"[init] EGGROLL noiser param count: {int(sum(np.prod(s) for s in lora_shapes)):,}"
    )

    # =========================
    # Init CLIP + PickScore
    # =========================
    clip_model, clip_processor = load_clip_model_and_processor(DEVICE)
    pick_model, pick_processor = load_pickscore_model_and_processor(DEVICE)

    # =========================
    # Init EGGROLL noiser + W&B
    # =========================
    noiser = EggRollNoiser(
        param_shapes=lora_shapes,
        sigma=SIGMA,
        lr_scale=LR_SCALE,
        rank=EGG_RANK,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "sigma": SIGMA,
            "lr_scale": LR_SCALE,
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
        },
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # Track best LoRA across training
    # =========================
    best_mean_reward = -float("inf")
    best_theta = None
    best_epoch = None

    # =========================
    # ES loop
    # =========================
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch} ===")
        theta, stats, img_dict = es_step(
            theta=theta,
            sana_es=sana_es,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            base_prompt_embeds=base_prompt_embeds,
            base_attention_mask=base_attention_mask,
            prompts_list=prompts_list,
            clip_model=clip_model,
            clip_processor=clip_processor,
            pick_model=pick_model,
            pickscore_processor=pick_processor,
            noiser=noiser,
            epoch=epoch,
            save_dir=SAVE_DIR,
            mix_weights=MIX_WEIGHTS,
            batch_size=BATCH_SIZE,
            pop_size=POP_SIZE,
            seed=epoch,
            guidance_scale=GUIDANCE_SCALE,
            theta_max_norm=THETA_MAX_NORM,
        )

        with torch.no_grad():
            theta_norm = theta.norm().item()
            delta_theta_norm = (theta - theta_init).norm().item()

        stats["theta_norm"] = theta_norm
        stats["delta_theta_norm"] = delta_theta_norm

        # ---- Track best LoRA (by summary/mean_reward) ----
        current_mean = stats.get(
            "summary/mean_reward", stats.get("mean_reward", float("-inf"))
        )
        if current_mean > best_mean_reward:
            best_mean_reward = current_mean
            best_theta = theta.detach().clone().cpu()
            best_epoch = epoch
            print(f"[best] New best mean reward {best_mean_reward:.4f} at epoch {epoch}")

        # ---- W&B images ----
        wandb_imgs = {}
        for key, img in img_dict.items():
            if img is not None:
                wandb_imgs[f"images/{key}"] = wandb.Image(img, caption=f"{key} epoch {epoch}")

        wandb.log(
            {
                "epoch": epoch,
                **stats,
                **wandb_imgs,
            },
            step=epoch,
        )

    # =========================
    # Finish: restore best theta & save best LoRA
    # =========================
    if best_theta is not None:
        print(
            f"\n[finish] Restoring best theta from epoch {best_epoch} "
            f"(mean_reward={best_mean_reward:.4f}) and saving LoRA..."
        )
        unflatten_to_params(best_theta.to(DEVICE), lora_params, lora_shapes)
        theta_to_save = best_theta
    else:
        print("\n[finish] No best_theta stored (using final theta).")
        unflatten_to_params(theta, lora_params, lora_shapes)
        theta_to_save = theta.detach().cpu()

    # Directory for best LoRA adapter
    best_lora_dir = SAVE_DIR / "best_lora"
    best_lora_dir.mkdir(parents=True, exist_ok=True)

    # Save PEFT LoRA weights (easy to load later)
    sana_es.transformer.save_pretrained(best_lora_dir)

    # Optionally also save raw theta + some metadata
    meta_path = SAVE_DIR / "best_lora_meta.pt"
    torch.save(
        {
            "theta_best": theta_to_save,
            "best_mean_reward": best_mean_reward,
            "best_epoch": best_epoch,
            "MODEL_NAME": MODEL_NAME,
            "LORA_R": LORA_R,
            "LORA_ALPHA": LORA_ALPHA,
            "LORA_DROPOUT": LORA_DROPOUT,
            "LORA_TARGET_MODULES": LORA_TARGET_MODULES,
        },
        meta_path,
    )

    print(f"[finish] Saved best LoRA weights to: {best_lora_dir}")
    print(f"[finish] Meta checkpoint saved to: {meta_path}")

    # Hint on how to load later
    print(
        "\n[hint] To load later:\n"
        "  from diffusers import SanaTransformer2DModel\n"
        "  from peft import PeftModel\n"
        f"  base = SanaTransformer2DModel.from_pretrained('{MODEL_NAME}', subfolder='transformer')\n"
        f"  lora = PeftModel.from_pretrained(base, '{best_lora_dir.as_posix()}')\n"
    )

    run.finish()


if __name__ == "__main__":
    main()