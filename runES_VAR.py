#!/usr/bin/env python3

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
import wandb
import lovely_tensors as lt
from peft import LoraConfig, get_peft_model

from models.VAR import VARClassGenerator
from rewards import (
    load_clip_model_and_processor,
    load_pickscore_model_and_processor,
    compute_all_rewards,
)
from utills import (
    EggRollNoiser,
    get_trainable_params_and_shapes,
    flatten_params,
    unflatten_to_params,
    make_prompt_strip,
    imagenet_prompt_text,
    is_all_classes
)

lt.monkey_patch()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# log at most this many repeats (each repeat has CLASSES_PER_GEN images)
MAX_LOG_BATCHES = 2

# -------------------------
# Speed / compile knobs
# -------------------------
USE_TORCH_COMPILE = True
COMPILE_MODE = "max-autotune"  # or "reduce-overhead"
COMPILE_FULLGRAPH = True  # safer: False

USE_TF32 = True
MATMUL_PRECISION = "high"  # "high" or "highest"

DETERMINISTIC = True
CUDNN_BENCHMARK = False

torch.set_float32_matmul_precision(MATMUL_PRECISION)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = bool(USE_TF32)
    torch.backends.cudnn.allow_tf32 = bool(USE_TF32)
    torch.backends.cudnn.deterministic = bool(DETERMINISTIC)
    torch.backends.cudnn.benchmark = bool(CUDNN_BENCHMARK) and (not bool(DETERMINISTIC))


def compile_modules(es_model, use_compile: bool = USE_TORCH_COMPILE):
    """
    IMPORTANT:
      - Keep es_model.transformer as the *PEFT* module so save_pretrained() works.
      - VARClassGenerator.generate() uses self.var.autoregressive_infer_cfg(...).
      - So we set es_model.var to the LoRA-wrapped transformer (or its compiled version).
    """
    es_model.var = es_model.transformer

    if not use_compile:
        return es_model
    if not torch.cuda.is_available():
        print("[compile] CUDA not available -> skipping torch.compile")
        return es_model

    print(f"[compile] Compiling VAR(+VAE) (mode={COMPILE_MODE}, fullgraph={COMPILE_FULLGRAPH}) ...")
    try:
        es_model.var = torch.compile(
            es_model.transformer,
            mode=COMPILE_MODE,
            fullgraph=COMPILE_FULLGRAPH,
        )
        if hasattr(es_model, "vae") and es_model.vae is not None:
            es_model.vae = torch.compile(
                es_model.vae,
                mode=COMPILE_MODE,
                fullgraph=COMPILE_FULLGRAPH,
            )
        print("[compile] SUCCESS.")
    except Exception as e:
        print(f"[compile] FAILED: {e}  -> continuing in eager mode.")
        es_model.var = es_model.transformer

    return es_model


# -------------------------
# Class sampling utilities
# -------------------------
def sample_classes_for_step(
        seed: int,
        allowed_classes: Optional[Union[str, Sequence[int]]],
        classes_per_gen: int,
        num_classes_total: int = 1000,
) -> List[int]:
    rng = np.random.RandomState(seed)

    if allowed_classes is None or allowed_classes == "all":
        pool = np.arange(num_classes_total, dtype=np.int64)
    else:
        pool = np.array(list(allowed_classes), dtype=np.int64)
        pool = np.unique(pool)
        pool = pool[(pool >= 0) & (pool < num_classes_total)]

    if pool.size == 0:
        pool = np.arange(num_classes_total, dtype=np.int64)

    if classes_per_gen <= 0:
        raise ValueError("classes_per_gen must be >= 1")
    if classes_per_gen > pool.size:
        raise ValueError(
            f"classes_per_gen ({classes_per_gen}) > pool size ({pool.size}). "
            f"(If you want to include ALL allowed classes each step, set CLASSES_PER_GEN=len(ALLOWED_CLASSES).)"
        )

    chosen = rng.choice(pool, size=classes_per_gen, replace=False)
    return chosen.tolist()


def sample_batches_for_step(
        seed: int,
        allowed_classes: Optional[Union[str, Sequence[int]]],
        classes_per_gen: int,
        batches_per_gen: int,
        num_classes_total: int = 1000,
) -> List[List[int]]:
    """
    Semantics:
      - CLASSES_PER_GEN = number of unique classes used in this step
      - BATCHES_PER_GEN = how many times to repeat them

    Output: [batches_per_gen][classes_per_gen] where each row is the same class list.
    """
    if batches_per_gen <= 0:
        raise ValueError("batches_per_gen must be >= 1")

    classes_this_step = sample_classes_for_step(
        seed=seed,
        allowed_classes=allowed_classes,
        classes_per_gen=classes_per_gen,
        num_classes_total=num_classes_total,
    )
    return [list(classes_this_step) for _ in range(batches_per_gen)]


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
        backend: str,
        epoch: int,
        stats: dict,
):
    print(f"[ckpt] Saving latest LoRA at epoch {epoch} -> {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    unflatten_to_params(theta.to(DEVICE), lora_params, lora_shapes)
    es_model.transformer.save_pretrained(save_dir)

    torch.save(
        {
            "theta_latest": theta.detach().cpu(),
            "epoch": epoch,
            "summary_mean_reward": stats.get("summary/mean_reward", float("nan")),
            "BACKEND": backend,
        },
        meta_path,
    )


# -------------------------
# Paper scoring: per-prompt z-score with global variance
# -------------------------
def paper_prompt_normalized_scores(
        S: torch.Tensor,
        eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implements paper Section 6.3 scoring:
      - S is [n, m] where:
          n = population size
          m = prompts/classes per step (CLASSES_PER_GEN)
      - mu_q is per-prompt mean over population: [m]
      - sigma_bar is GLOBAL std over all centered entries
      - score_i = mean_j ( (S_ij - mu_qj) / sigma_bar )

    Returns:
      scores:    [n]
      mu_q:      [m]
      sigma_bar: scalar tensor
    """
    if S.ndim != 2:
        raise ValueError(f"S must be [n, m], got shape={tuple(S.shape)}")

    mu_q = S.mean(dim=0)  # [m]
    centered = S - mu_q[None, :]  # [n, m]
    sigma_bar = torch.sqrt((centered ** 2).mean()).clamp_min(eps)  # scalar
    Z = centered / sigma_bar  # [n, m]
    scores = Z.mean(dim=1)  # [n]
    return scores, mu_q, sigma_bar


# -------------------------
# ES step with paper prompt-normalization
# -------------------------
@torch.no_grad()
def es_step(
        theta: torch.Tensor,
        es_model: VARClassGenerator,
        lora_params,
        lora_shapes,
        clip_model,
        clip_processor,
        pick_model,
        pickscore_processor,
        noiser: EggRollNoiser,
        epoch: int,
        save_dir: Path,
        mix_weights,
        pop_size: int,
        seed: int,
        guidance_scale: float,
        theta_max_norm: float,
        allowed_classes: Optional[Union[str, Sequence[int]]],
        classes_per_gen: int,
        batches_per_gen: int,
        num_classes_total: int = 1000,
        max_log_batches: int = 2,
):
    torch.manual_seed(seed)

    class_id_batches = sample_batches_for_step(
        seed=seed,
        allowed_classes=allowed_classes,
        classes_per_gen=classes_per_gen,
        batches_per_gen=batches_per_gen,
        num_classes_total=num_classes_total,
    )
    classes_this_step = class_id_batches[0] if len(class_id_batches) else []
    log_per_class = not is_all_classes(allowed_classes)
    m = len(classes_this_step)
    if m != classes_per_gen:
        raise RuntimeError(f"internal mismatch: m={m}, classes_per_gen={classes_per_gen}")

    total_imgs_per_indiv = batches_per_gen * classes_per_gen
    log_batches = int(max(0, min(max_log_batches, batches_per_gen)))
    total_imgs_for_logging = log_batches * classes_per_gen

    print(f"[epoch {epoch}] allowed_classes={allowed_classes}")
    print(f"[epoch {epoch}] CLASSES_PER_GEN={classes_per_gen}, BATCHES_PER_GEN(repeats)={batches_per_gen}")
    print(f"[epoch {epoch}] classes_this_step={classes_this_step} (each repeated {batches_per_gen}x)")
    print(f"[epoch {epoch}] logging {log_batches} repeats -> {total_imgs_for_logging} imgs/indiv")

    # ES noise
    eps = noiser.sample_eps(pop_size, DEVICE)  # [n, D]

    # Score matrix for paper normalization (COMBINED only)
    S_comb = torch.empty((pop_size, m), device=DEVICE, dtype=torch.float32)

    # Raw per-individual means over ALL images (for logging & histogram)
    raw_comb_mean = torch.empty((pop_size,), device=DEVICE, dtype=torch.float32)
    raw_aes_mean = torch.empty((pop_size,), device=DEVICE, dtype=torch.float32)
    raw_txt_mean = torch.empty((pop_size,), device=DEVICE, dtype=torch.float32)
    raw_noart_mean = torch.empty((pop_size,), device=DEVICE, dtype=torch.float32)
    raw_pick_mean = torch.empty((pop_size,), device=DEVICE, dtype=torch.float32)

    # For strips
    all_images_flat: List[List] = []  # pop_size lists of PIL images (limited)

    for k in range(pop_size):
        theta_k = theta + noiser.sigma * eps[k]
        unflatten_to_params(theta_k, lora_params, lora_shapes)

        imgs_grouped, _ = es_model.generate(
            seed=seed,
            guidance_scale=guidance_scale,
            class_ids=class_id_batches,  # [repeats][m]
            return_grouped=True,
        )
        # imgs_grouped: [batches_per_gen][m]

        # Per-prompt combined rewards (accumulate over repeats)
        per_prompt_comb: List[List[torch.Tensor]] = [[] for _ in range(m)]

        # Per-image lists for raw means
        all_comb: List[torch.Tensor] = []
        all_aes: List[torch.Tensor] = []
        all_txt: List[torch.Tensor] = []
        all_noart: List[torch.Tensor] = []
        all_pick: List[torch.Tensor] = []

        images_for_logging: List = []

        for b in range(batches_per_gen):
            for j, cls_id in enumerate(class_id_batches[b]):
                img = imgs_grouped[b][j]
                if b < log_batches:
                    images_for_logging.append(img)

                prompt_text = imagenet_prompt_text(int(cls_id))
                reward_dict = compute_all_rewards(
                    [img],
                    prompt_text=prompt_text,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    mix_weights=mix_weights,
                    pickscore_model=pick_model,
                    pickscore_processor=pickscore_processor,
                )

                r_comb = reward_dict["combined"].float()
                r_aes = reward_dict["clip_aesthetic"].float()
                r_txt = reward_dict["clip_text"].float()
                r_noart = reward_dict["no_artifacts"].float()
                r_pick = reward_dict["pickscore"].float()

                per_prompt_comb[j].append(r_comb)

                all_comb.append(r_comb)
                all_aes.append(r_aes)
                all_txt.append(r_txt)
                all_noart.append(r_noart)
                all_pick.append(r_pick)

        # Fill S_comb[k, j] = mean over repeats for prompt j
        for j in range(m):
            vals_j = torch.stack(per_prompt_comb[j])
            S_comb[k, j] = vals_j.mean()

        # Raw means over ALL images for this individual
        raw_comb_mean[k] = torch.stack(all_comb).mean()
        raw_aes_mean[k] = torch.stack(all_aes).mean()
        raw_txt_mean[k] = torch.stack(all_txt).mean()
        raw_noart_mean[k] = torch.stack(all_noart).mean()
        raw_pick_mean[k] = torch.stack(all_pick).mean()

        all_images_flat.append(images_for_logging)

        per_prompt_means_str = ", ".join([f"{S_comb[k, j].item():.4f}" for j in range(m)])
        print(
            f"  indiv {k:02d} | raw_prompt_means=[{per_prompt_means_str}] | "
            f"raw_comb_mean={raw_comb_mean[k].item():.4f} | "
            f"aes={raw_aes_mean[k].item():.4f} txt={raw_txt_mean[k].item():.4f} "
            f"noart={raw_noart_mean[k].item():.4f} pick={raw_pick_mean[k].item():.4f} "
            f"(over {total_imgs_per_indiv} imgs)"
        )

    # Paper scoring (normalized) for ES update
    norm_scores, mu_q, sigma_bar = paper_prompt_normalized_scores(S_comb)  # [n]

    finite_mask = torch.isfinite(norm_scores)
    if not finite_mask.any():
        print("⚠ All normalized scores NaN/Inf, skipping update.")
        stats = {"summary/mean_reward": float("nan")}
        return theta, stats, {"best": None, "median": None, "worst": None}, None

    # Best/median/worst strips selected by NORMALIZED score (the thing you optimize)
    best_img = median_img = worst_img = None
    if finite_mask.all() and total_imgs_for_logging > 0:
        _, sorted_idx = torch.sort(norm_scores)
        worst_idx = sorted_idx[0].item()
        best_idx = sorted_idx[-1].item()
        median_idx = sorted_idx[len(sorted_idx) // 2].item()

        epoch_dir = save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        best_strip = make_prompt_strip(all_images_flat[best_idx], total_imgs_for_logging)
        median_strip = make_prompt_strip(all_images_flat[median_idx], total_imgs_for_logging)
        worst_strip = make_prompt_strip(all_images_flat[worst_idx], total_imgs_for_logging)

        if best_strip is not None:
            best_strip.save(epoch_dir / "best.png")
            best_img = best_strip
        if median_strip is not None:
            median_strip.save(epoch_dir / "median.png")
            median_img = median_strip
        if worst_strip is not None:
            worst_strip.save(epoch_dir / "worst.png")
            worst_img = worst_strip

    # ES update uses normalized scores
    norm_scores_f = norm_scores[finite_mask]
    eps_f = eps[finite_mask]
    fitnesses = noiser.convert_fitnesses(norm_scores_f)
    theta = noiser.do_update(theta, eps_f, fitnesses)

    # Norm cap
    if theta_max_norm is not None and theta_max_norm > 0:
        tnorm = theta.norm()
        if tnorm > theta_max_norm:
            theta = theta * (theta_max_norm / (tnorm + 1e-8))

    # ---- Logging stats ----
    raw_comb_f = raw_comb_mean[finite_mask]
    raw_aes_f = raw_aes_mean[finite_mask]
    raw_txt_f = raw_txt_mean[finite_mask]
    raw_noart_f = raw_noart_mean[finite_mask]
    raw_pick_f = raw_pick_mean[finite_mask]

    stats: Dict[str, float] = {
        # normalized (paper) score summary
        "summary/mean_reward": float(norm_scores_f.mean().item()),
        "summary/max_reward": float(norm_scores_f.max().item()),
        "summary/min_reward": float(norm_scores_f.min().item()),
        "std_reward": float(norm_scores_f.std().item() if norm_scores_f.numel() > 1 else 0.0),

        # raw combined & components (mean over individuals, where each individual is mean over ALL images)
        "raw/combined_mean": float(raw_comb_f.mean().item()),
        "raw/combined_std": float(raw_comb_f.std().item() if raw_comb_f.numel() > 1 else 0.0),

        "aesthetic_mean": float(raw_aes_f.mean().item()),
        "clip_text_mean": float(raw_txt_f.mean().item()),
        "no_artifacts_mean": float(raw_noart_f.mean().item()),
        "pickscore_mean": float(raw_pick_f.mean().item()),

        "aesthetic_std": float(raw_aes_f.std().item() if raw_aes_f.numel() > 1 else 0.0),
        "clip_text_std": float(raw_txt_f.std().item() if raw_txt_f.numel() > 1 else 0.0),
        "no_artifacts_std": float(raw_noart_f.std().item() if raw_noart_f.numel() > 1 else 0.0),
        "pickscore_std": float(raw_pick_f.std().item() if raw_pick_f.numel() > 1 else 0.0),

        # promptnorm diagnostics
        "promptnorm/sigma_bar": float(sigma_bar.item()),

        # epoch info
        "epoch/seed": int(seed),
        "epoch/classes_per_gen": int(classes_per_gen),
        "epoch/repeats_per_class": int(batches_per_gen),
        "epoch/classes_this_step": str(classes_this_step),
        "epoch/allowed_classes": str(allowed_classes),
        "epoch/max_log_batches": int(max_log_batches),
        "epoch/logged_repeats": int(log_batches),
        "epoch/logged_imgs_per_indiv": int(total_imgs_for_logging),
        "epoch/total_imgs_per_indiv": int(total_imgs_per_indiv),
    }

    # Log per-prompt stats (indexed) AND per-class-id stats (so you see class "2" etc clearly)
    # Only create per-class/prompt sections if user gave an explicit list (NOT "all")
    if log_per_class:
        for j in range(m):
            cls_id = int(classes_this_step[j])

            # prompt-indexed (since m is small)
            stats[f"prompt_{j}/class_id"] = cls_id
            stats[f"prompt_{j}/mu_over_pop"] = float(mu_q[j].item())
            stats[f"prompt_{j}/raw_mean_over_pop"] = float(S_comb[:, j].mean().item())
            stats[f"prompt_{j}/raw_std_over_pop"] = float(S_comb[:, j].std().item() if pop_size > 1 else 0.0)

            # class-id keyed
            stats[f"class_{cls_id}/mu_over_pop"] = float(mu_q[j].item())
            stats[f"class_{cls_id}/raw_mean_over_pop"] = float(S_comb[:, j].mean().item())
            stats[f"class_{cls_id}/raw_std_over_pop"] = float(S_comb[:, j].std().item() if pop_size > 1 else 0.0)
    else:
        # keep a single compact string so you can still debug what got sampled
        stats["promptnorm/classes_this_step"] = str(classes_this_step)

    img_dict = {"best": best_img, "median": median_img, "worst": worst_img}

    rewards_for_hist = raw_comb_f.detach().cpu()

    return theta, stats, img_dict, rewards_for_hist


# -------------------------
# Run one config
# -------------------------
def run_single_config():
    # =========================
    # VAR settings
    # =========================
    MODEL_DEPTH = 16
    NUM_CLASSES_TOTAL = 1000

    # You can change these freely.
    # If you set ALLOWED_CLASSES=list(range(3)) AND CLASSES_PER_GEN=2,
    # each epoch uses 2 unique classes sampled from {0,1,2}.
    ALLOWED_CLASSES = "all"  # train on classes 0..5
    CLASSES_PER_GEN = 2  # 2 prompts per individual (per epoch)
    BATCHES_PER_GEN = 32  # repeats per prompt

    # =========================
    # LoRA settings
    # =========================
    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.0

    # =========================
    # ES settings
    # =========================
    SIGMA = 1e-2
    LR_SCALE = 5e-1
    USE_ANTITHETIC = True

    POP_SIZE = 8
    NUM_EPOCHS = 1000
    THETA_MAX_NORM = 40.0
    EGG_RANK = 1

    MIX_WEIGHTS = (0.0, 0.0, 0.0, 1.0)  # PickScore only (as you had)
    GUIDANCE_SCALE = 4.0

    WANDB_PROJECT = "VAR-ES-Class-Search-v0"
    BACKEND = "var_class_promptnorm"

    SAVE_DIR = Path("es_var_class_promptnorm_all_4")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    latest_lora_dir = SAVE_DIR / "latest_lora"
    meta_path = SAVE_DIR / "latest_lora_meta.pt"

    # =========================
    # Init VAR model
    # =========================
    es_model = VARClassGenerator(
        model_depth=MODEL_DEPTH,
        device=DEVICE.split(":")[0] if DEVICE.startswith("cuda") else DEVICE,
        ckpt_dir="checkpoints_var",
        download_if_missing=True,
    )

    # =========================
    # Attach LoRA
    # =========================
    target_modules = [
        "mat_qkv",
        "proj",
        "fc1",
        "fc2",
        "ada_lin.1",
        "head_nm.ada_lin.1",
        "head",
    ]
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
    )

    es_model.transformer = get_peft_model(es_model.transformer, lora_config)
    es_model.transformer.to(DEVICE)
    es_model.transformer.eval()

    es_model = compile_modules(es_model, USE_TORCH_COMPILE)

    # =========================
    # Collect LoRA params
    # =========================
    lora_params, lora_shapes = get_trainable_params_and_shapes(es_model.transformer)
    theta = flatten_params(lora_params).to(DEVICE, dtype=torch.float32)
    theta_init = theta.clone().detach()
    print(f"[init] LoRA trainable parameters: {theta.numel():,}")
    print(f"[init] theta_init_norm = {theta_init.norm().item():.4f}")

    # =========================
    # Init CLIP + PickScore
    # =========================
    clip_model, clip_processor = load_clip_model_and_processor(DEVICE)
    pick_model, pickscore_processor = load_pickscore_model_and_processor(DEVICE)

    # =========================
    # Init noiser + W&B
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
        name=f"{BACKEND}_d{MODEL_DEPTH}_sigma{SIGMA:.0e}_lr{LR_SCALE:.0e}_ant{int(USE_ANTITHETIC)}_cls={str(ALLOWED_CLASSES)}_m{CLASSES_PER_GEN}_rep{BATCHES_PER_GEN}",
        config={
            "backend": BACKEND,
            "model_depth": MODEL_DEPTH,
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
            "theta_max_norm": THETA_MAX_NORM,
            "allowed_classes": str(ALLOWED_CLASSES),
            "classes_per_gen": CLASSES_PER_GEN,
            "batches_per_gen": BATCHES_PER_GEN,
            "num_classes_total": NUM_CLASSES_TOTAL,
            "use_torch_compile": USE_TORCH_COMPILE,
            "compile_mode": COMPILE_MODE,
            "compile_fullgraph": COMPILE_FULLGRAPH,
            "tf32": USE_TF32,
            "matmul_precision": MATMUL_PRECISION,
            "deterministic": DETERMINISTIC,
            "max_log_batches": MAX_LOG_BATCHES,
        },
    )

    last_epoch_stats = None
    prev_norm = None
    prev_raw = None

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch} ({BACKEND}) ===")

        theta, stats, img_dict, rewards_vec_raw = es_step(
            theta=theta,
            es_model=es_model,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            clip_model=clip_model,
            clip_processor=clip_processor,
            pick_model=pick_model,
            pickscore_processor=pickscore_processor,
            noiser=noiser,
            epoch=epoch,
            save_dir=SAVE_DIR,
            mix_weights=MIX_WEIGHTS,
            pop_size=POP_SIZE,
            seed=epoch,
            guidance_scale=GUIDANCE_SCALE,
            theta_max_norm=THETA_MAX_NORM,
            allowed_classes=ALLOWED_CLASSES,
            classes_per_gen=CLASSES_PER_GEN,
            batches_per_gen=BATCHES_PER_GEN,
            num_classes_total=NUM_CLASSES_TOTAL,
            max_log_batches=MAX_LOG_BATCHES,
        )
        last_epoch_stats = stats

        # -------------------------
        # Trend prints (like before)
        # -------------------------
        curr_norm = float(stats.get("summary/mean_reward", float("nan")))
        curr_raw = float(stats.get("raw/combined_mean", float("nan")))

        def _trend_line(curr, prev, label: str) -> str:
            if prev is None or not np.isfinite(prev):
                return f"[trend] {label}={curr:.6f} (first)"
            delta = curr - prev
            if not np.isfinite(delta):
                arrow = "→"
            elif delta > 0:
                arrow = "↑"
            elif delta < 0:
                arrow = "↓"
            else:
                arrow = "→"
            return f"[trend] {label} {arrow} {curr:.6f}  Δ={delta:+.6f} vs prev"

        print(_trend_line(curr_norm, prev_norm, "norm_mean_reward"))
        print(_trend_line(curr_raw, prev_raw, "raw_combined_mean"))

        prev_norm = curr_norm
        prev_raw = curr_raw

        # -------------------------
        # LoRA stats + hist sampling
        # -------------------------
        with torch.no_grad():
            delta_theta = theta - theta_init
            theta_flat = theta.detach().cpu().view(-1)
            delta_flat = delta_theta.detach().cpu().view(-1)

            stats["theta_norm"] = float(theta.norm().item())
            stats["delta_theta_norm"] = float(delta_theta.norm().item())
            stats["lora/mean_abs"] = float(theta_flat.abs().mean().item())
            stats["lora/delta_mean_abs"] = float(delta_flat.abs().mean().item())

            max_hist_params = 50_000
            if theta_flat.numel() > max_hist_params:
                idx = torch.randperm(theta_flat.numel())[:max_hist_params]
                theta_hist_vals = theta_flat[idx]
                delta_hist_vals = delta_flat[idx]
            else:
                theta_hist_vals = theta_flat
                delta_hist_vals = delta_flat

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_lora_checkpoint(
                theta=theta,
                es_model=es_model,
                lora_params=lora_params,
                lora_shapes=lora_shapes,
                save_dir=latest_lora_dir,
                meta_path=meta_path,
                backend=BACKEND,
                epoch=epoch + 1,
                stats=stats,
            )

        # W&B images
        wandb_imgs = {}
        for key, img in img_dict.items():
            if img is not None:
                wandb_imgs[f"images/{key}"] = wandb.Image(img, caption=f"{key} epoch {epoch}")

        log_payload = {"epoch": epoch, **stats, **wandb_imgs}

        # IMPORTANT: reward_hist should be RAW reward, not normalized
        if rewards_vec_raw is not None and getattr(rewards_vec_raw, "numel", lambda: 0)() > 0:
            log_payload["reward_hist"] = wandb.Histogram(rewards_vec_raw.detach().cpu().numpy())

        log_payload["lora/weights_hist"] = wandb.Histogram(theta_hist_vals.numpy())
        log_payload["lora/delta_hist"] = wandb.Histogram(delta_hist_vals.numpy())

        wandb.log(log_payload, step=epoch)

    # Final save
    if NUM_EPOCHS % 10 != 0 and last_epoch_stats is not None:
        save_lora_checkpoint(
            theta=theta,
            es_model=es_model,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            save_dir=latest_lora_dir,
            meta_path=meta_path,
            backend=BACKEND,
            epoch=NUM_EPOCHS,
            stats=last_epoch_stats,
        )

    print(f"\nDone. Latest LoRA checkpoint at: {latest_lora_dir}")
    print(f"Meta checkpoint at: {meta_path}")
    run.finish()


def main():
    run_single_config()


if __name__ == "__main__":
    main()