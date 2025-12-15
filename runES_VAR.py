#!/usr/bin/env python3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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
)

lt.monkey_patch()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_LOG_BATCHES = 2  # log at most this many batches (each batch has CLASSES_PER_GEN images)

# -------------------------
# Speed / compile knobs
# -------------------------
USE_TORCH_COMPILE = True
COMPILE_MODE = "max-autotune"       # or "reduce-overhead"
COMPILE_FULLGRAPH = True            # safer: False

USE_TF32 = True
MATMUL_PRECISION = "high"           # "high" or "highest"

DETERMINISTIC = True
CUDNN_BENCHMARK = False

# matmul precision
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
      - generate() in your VARClassGenerator uses self.var.autoregressive_infer_cfg(...).
      - So we set es_model.var to the LoRA-wrapped transformer (or its compiled version).
    """
    # Always ensure generate() uses LoRA weights
    es_model.var = es_model.transformer

    if not use_compile:
        return es_model

    if not torch.cuda.is_available():
        print("[compile] CUDA not available -> skipping torch.compile")
        return es_model

    print(f"[compile] Compiling VAR(+VAE) (mode={COMPILE_MODE}, fullgraph={COMPILE_FULLGRAPH}) ...")
    try:
        # Compile the LoRA-wrapped module, but keep es_model.transformer uncompiled for saving
        es_model.var = torch.compile(
            es_model.transformer,
            mode=COMPILE_MODE,
            fullgraph=COMPILE_FULLGRAPH,
        )

        # VAE not used in current generate(), optional compile
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
        pool = np.arange(num_classes_total, dtype=np.int64)

    if classes_per_gen > pool.size:
        raise ValueError(f"classes_per_gen ({classes_per_gen}) > pool size ({pool.size})")

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
    Deterministically sample class_ids as a list-of-lists:
      shape = [batches_per_gen][classes_per_gen]
    """
    if batches_per_gen <= 0:
        raise ValueError("batches_per_gen must be >= 1")

    out: List[List[int]] = []
    for b in range(batches_per_gen):
        # distinct deterministic seed per batch
        batch_seed = seed * 10_000 + b
        out.append(
            sample_classes_for_step(
                seed=batch_seed,
                allowed_classes=allowed_classes,
                classes_per_gen=classes_per_gen,
                num_classes_total=num_classes_total,
            )
        )
    return out


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
    theta_cpu = theta.detach().cpu()

    unflatten_to_params(theta.to(DEVICE), lora_params, lora_shapes)
    # save the PEFT module (NOT the compiled wrapper)
    es_model.transformer.save_pretrained(save_dir)

    torch.save(
        {
            "theta_latest": theta_cpu,
            "epoch": epoch,
            "summary_mean_reward": stats.get("summary/mean_reward", float("nan")),
            "BACKEND": backend,
        },
        meta_path,
    )


# -------------------------
# ES step (LoRA-only) using TRUE batched generation in the class
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
    max_log_batches: int = 2,   # <-- NEW: log only first N batches
):
    """
    Efficient batching strategy:
      - Pre-sample class_ids as 2D list: [batches_per_gen][classes_per_gen]
      - For EACH individual:
          - do ONE call: es_model.generate(class_ids=class_id_batches, return_grouped=True)
            => returns list[list[PIL]] with same shape
      - Reward is averaged over ALL images (all batches).
      - Logging/strips use ONLY first `max_log_batches` batches.
    """
    torch.manual_seed(seed)

    class_id_batches = sample_batches_for_step(
        seed=seed,
        allowed_classes=allowed_classes,
        classes_per_gen=classes_per_gen,
        batches_per_gen=batches_per_gen,
        num_classes_total=num_classes_total,
    )

    total_imgs_per_indiv = batches_per_gen * classes_per_gen
    log_batches = int(max(0, min(max_log_batches, batches_per_gen)))
    total_imgs_for_logging = log_batches * classes_per_gen

    print(f"[epoch {epoch}] batches_per_gen={batches_per_gen}, classes_per_gen={classes_per_gen}")
    print(f"[epoch {epoch}] class_id_batches={class_id_batches}")
    print(f"[epoch {epoch}] max_log_batches={max_log_batches} -> logging {log_batches} batches ({total_imgs_for_logging} imgs)")

    eps = noiser.sample_eps(pop_size, DEVICE)  # [pop_size, D]

    rewards_combined = []
    rewards_aes = []
    rewards_text = []
    rewards_noart = []
    rewards_pick = []

    # list[pop_size] of list[<= total_imgs_for_logging] PILs (flattened)
    all_images_flat = []
    per_class_combined: Dict[int, List[torch.Tensor]] = {}

    for k in range(pop_size):
        theta_k = theta + noiser.sigma * eps[k]
        unflatten_to_params(theta_k, lora_params, lora_shapes)

        # ONE generation call (batched) per individual
        imgs_grouped, _ = es_model.generate(
            seed=seed,
            guidance_scale=guidance_scale,
            class_ids=class_id_batches,  # 2D list-of-lists
            return_grouped=True,
        )
        # imgs_grouped: shape [batches_per_gen][classes_per_gen]

        cand_comb = []
        cand_aes = []
        cand_txt = []
        cand_noart = []
        cand_pick = []
        cand_images_for_logging = []

        for b in range(batches_per_gen):
            for j, cls_id in enumerate(class_id_batches[b]):
                img = imgs_grouped[b][j]

                # keep ONLY first log_batches for logging/strips
                if b < log_batches:
                    cand_images_for_logging.append(img)

                prompt_text = imagenet_prompt_text(cls_id)
                reward_dict = compute_all_rewards(
                    [img],
                    prompt_text=prompt_text,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    mix_weights=mix_weights,
                    pickscore_model=pick_model,
                    pickscore_processor=pickscore_processor,
                )

                r_comb = reward_dict["combined"]
                cand_comb.append(r_comb)
                cand_aes.append(reward_dict["clip_aesthetic"])
                cand_txt.append(reward_dict["clip_text"])
                cand_noart.append(reward_dict["no_artifacts"])
                cand_pick.append(reward_dict["pickscore"])

                per_class_combined.setdefault(int(cls_id), []).append(r_comb)

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

        all_images_flat.append(cand_images_for_logging)

        print(
            f"  indiv {k:02d} | "
            f"R_comb={r_comb.item():.4f}, "
            f"aesthetic={r_aes.item():.4f}, "
            f"text_align={r_txt.item():.4f}, "
            f"no_artifacts={r_noart.item():.4f}, "
            f"pickscore={r_pick.item():.4f} "
            f"(reward avg over {total_imgs_per_indiv} imgs; logging {total_imgs_for_logging})"
        )

    rewards_combined = torch.stack(rewards_combined, dim=0)
    rewards_aes = torch.stack(rewards_aes, dim=0)
    rewards_text = torch.stack(rewards_text, dim=0)
    rewards_noart = torch.stack(rewards_noart, dim=0)
    rewards_pick = torch.stack(rewards_pick, dim=0)

    finite_mask = torch.isfinite(rewards_combined)
    if not finite_mask.any():
        print("⚠ All rewards NaN/Inf, skipping update.")
        stats = {
            "summary/mean_reward": float("nan"),
            "summary/max_reward": float("nan"),
            "summary/min_reward": float("nan"),
        }
        return theta, stats, {"best": None, "median": None, "worst": None}, None

    # best/median/worst strips (over LOGGED imgs only)
    best_img = median_img = worst_img = None
    if finite_mask.all() and total_imgs_for_logging > 0:
        _, sorted_idx = torch.sort(rewards_combined)
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

    # restrict finite
    rewards_combined_f = rewards_combined[finite_mask]
    rewards_aes_f = rewards_aes[finite_mask]
    rewards_text_f = rewards_text[finite_mask]
    rewards_noart_f = rewards_noart[finite_mask]
    rewards_pick_f = rewards_pick[finite_mask]
    eps_f = eps[finite_mask]

    # ES update
    fitnesses = noiser.convert_fitnesses(rewards_combined_f)
    theta = noiser.do_update(theta, eps_f, fitnesses)

    # norm cap
    if theta_max_norm is not None and theta_max_norm > 0:
        tnorm = theta.norm()
        if tnorm > theta_max_norm:
            theta = theta * (theta_max_norm / (tnorm + 1e-8))

    # per-class stats
    per_class_stats = {}
    for cls_id, vals_list in per_class_combined.items():
        vals = torch.stack(vals_list)
        vals = vals[vals.isfinite()]
        if vals.numel() == 0:
            mean_v = max_v = min_v = float("nan")
        else:
            mean_v = vals.mean().item()
            max_v = vals.max().item()
            min_v = vals.min().item()

        per_class_stats[f"class_{cls_id}/mean_reward"] = mean_v
        per_class_stats[f"class_{cls_id}/max_reward"] = max_v
        per_class_stats[f"class_{cls_id}/min_reward"] = min_v

    comb_mean = rewards_combined_f.mean().item()
    comb_std = rewards_combined_f.std().item() if rewards_combined_f.numel() > 1 else 0.0
    comb_max = rewards_combined_f.max().item()
    comb_min = rewards_combined_f.min().item()

    stats = {
        "summary/mean_reward": comb_mean,
        "summary/max_reward": comb_max,
        "summary/min_reward": comb_min,
        "std_reward": comb_std,
        "aesthetic_mean": rewards_aes_f.mean().item(),
        "clip_text_mean": rewards_text_f.mean().item(),
        "no_artifacts_mean": rewards_noart_f.mean().item(),
        "pickscore_mean": rewards_pick_f.mean().item(),
        "epoch/seed": seed,
        "epoch/batches_per_gen": int(batches_per_gen),
        "epoch/classes_per_gen": int(classes_per_gen),
        "epoch/classes": str(class_id_batches),
        "epoch/max_log_batches": int(max_log_batches),
        "epoch/logged_batches": int(log_batches),
        "epoch/logged_imgs_per_indiv": int(total_imgs_for_logging),
    }
    stats.update(per_class_stats)

    img_dict = {"best": best_img, "median": median_img, "worst": worst_img}
    rewards_for_hist = rewards_combined_f.detach().cpu()
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

    # USER PARAM #1: allowed classes (list) or "all"/None
    ALLOWED_CLASSES = [0]     # example: only one class

    # USER PARAM #2: unique classes per batch
    CLASSES_PER_GEN = 1

    # USER PARAM #3: how many batches per individual
    BATCHES_PER_GEN = 32

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
    NUM_EPOCHS = 500
    THETA_MAX_NORM = 40.0
    EGG_RANK = 1

    MIX_WEIGHTS = (0.0, 0.0, 0.0, 1.0)  # PickScore only
    GUIDANCE_SCALE = 4.0

    WANDB_PROJECT = "VAR-ES-Class-Search-v0"
    BACKEND = "var_class"

    SAVE_DIR = Path("es_var_class_run")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_lora_dir = SAVE_DIR / "latest_lora"
    best_lora_dir.mkdir(parents=True, exist_ok=True)
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

    # Ensure generate() uses LoRA weights + optionally compile
    es_model = compile_modules(es_model, USE_TORCH_COMPILE)

    # =========================
    # Collect LoRA params
    # =========================
    lora_params, lora_shapes = get_trainable_params_and_shapes(es_model.transformer)
    theta = flatten_params(lora_params).to(DEVICE)
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
        name=f"{BACKEND}_d{MODEL_DEPTH}_sigma{SIGMA:.0e}_lr{LR_SCALE:.0e}_ant{int(USE_ANTITHETIC)}",
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
        },
    )

    # =========================
    # ES loop
    # =========================
    last_epoch_stats = None
    prev_mean_reward = None

    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch} ({BACKEND}) ===")

        theta, stats, img_dict, rewards_vec = es_step(
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
        )

        last_epoch_stats = stats
        # -------------------------
        # Print training trend summary
        # -------------------------
        curr = float(stats.get("summary/mean_reward", float("nan")))

        if prev_mean_reward is None or not np.isfinite(prev_mean_reward):
            arrow = "→"
            delta = float("nan")
        else:
            delta = curr - prev_mean_reward
            if not np.isfinite(delta):
                arrow = "→"
            elif delta > 0:
                arrow = "↑"
            elif delta < 0:
                arrow = "↓"
            else:
                arrow = "→"

        # Choose what "good" means:
        # Here: up is good. You can also add a small threshold like abs(delta) < 1e-4 => "→"
        if prev_mean_reward is None:
            msg = f"[trend] mean_reward={curr:.6f} (first epoch)"
        else:
            msg = f"[trend] {arrow} mean_reward={curr:.6f}  Δ={delta:+.6f} vs prev"

        print(msg)

        prev_mean_reward = curr

        # LoRA stats
        with torch.no_grad():
            delta_theta = theta - theta_init
            theta_norm = theta.norm().item()
            delta_theta_norm = delta_theta.norm().item()

            theta_flat = theta.detach().cpu().view(-1)
            delta_flat = delta_theta.detach().cpu().view(-1)

            max_hist_params = 50_000
            if theta_flat.numel() > max_hist_params:
                idx = torch.randperm(theta_flat.numel())[:max_hist_params]
                theta_hist_vals = theta_flat[idx]
                delta_hist_vals = delta_flat[idx]
            else:
                theta_hist_vals = theta_flat
                delta_hist_vals = delta_flat

        stats["theta_norm"] = theta_norm
        stats["delta_theta_norm"] = delta_theta_norm
        stats["lora/mean_abs"] = float(theta_flat.abs().mean().item())
        stats["lora/delta_mean_abs"] = float(delta_flat.abs().mean().item())

        # Save latest LoRA every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_lora_checkpoint(
                theta=theta,
                es_model=es_model,
                lora_params=lora_params,
                lora_shapes=lora_shapes,
                save_dir=best_lora_dir,
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

        if rewards_vec is not None and rewards_vec.numel() > 0:
            log_payload["reward_hist"] = wandb.Histogram(rewards_vec.detach().cpu().numpy())

        log_payload["lora/weights_hist"] = wandb.Histogram(theta_hist_vals.numpy())
        log_payload["lora/delta_hist"] = wandb.Histogram(delta_hist_vals.numpy())

        wandb.log(log_payload, step=epoch)

    # Final save if needed
    if NUM_EPOCHS % 10 != 0 and last_epoch_stats is not None:
        save_lora_checkpoint(
            theta=theta,
            es_model=es_model,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            save_dir=best_lora_dir,
            meta_path=meta_path,
            backend=BACKEND,
            epoch=NUM_EPOCHS,
            stats=last_epoch_stats,
        )

    print(f"\nDone. Latest LoRA checkpoint at: {best_lora_dir}")
    print(f"Meta checkpoint at: {meta_path}")
    run.finish()


def main():
    run_single_config()


if __name__ == "__main__":
    main()