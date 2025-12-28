#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
import lovely_tensors as lt

from rewards import (
    load_clip_model_and_processor,
    load_pickscore_model_and_processor,
    compute_all_rewards,
)
from es_backend import *

lt.monkey_patch()


# ============================================================
# Infinity variants (edit this dict later to add more)
# ============================================================
INFINITY_VARIANTS: Dict[str, Dict[str, Any]] = {
    # "what it's now" (your current defaults)
    "2b_reg": dict(
        inf_model_path="Infinity/weights/infinity_2b_reg.pth",
        inf_vae_path="Infinity/weights/infinity_vae_d32_reg.pth",
        inf_text_encoder_ckpt="google/flan-t5-xl",
        inf_vae_type=32,
        inf_pn="0.60M",
        inf_model_type="infinity_2b",
        inf_apply_spatial_patchify=0,
        inf_checkpoint_type="torch",
        inf_encoded_prompts="encoded_prompts_infinity.pt",
    ),

    # small fast variant (your snippet)
    "125m_256": dict(
        inf_model_path="Infinity/weights/infinity_125M_256x256.pth",
        inf_vae_path="Infinity/weights/infinity_vae_d16.pth",
        inf_text_encoder_ckpt="google/flan-t5-xl",
        inf_vae_type=16,
        inf_pn="0.06M",
        inf_model_type="infinity_layer12",
        inf_apply_spatial_patchify=0,
        inf_checkpoint_type="torch",
        inf_encoded_prompts="encoded_prompts_infinity_125m_256.pt",
    ),
    "8b_512": dict(
        inf_model_path="Infinity/weights/infinity_8b_512x512_weights",
        inf_vae_path="Infinity/weights/infinity_vae_d56_f8_14_patchify.pth",
        inf_text_encoder_ckpt="google/flan-t5-xl",
        inf_vae_type=14,
        inf_pn="0.25M",
        inf_model_type="infinity_8b",
        inf_apply_spatial_patchify=1,
        inf_checkpoint_type="torch_shard",
        inf_encoded_prompts="encoded_prompts_infinity_8b_512.pt",
    )
}



def apply_infinity_variant(args: argparse.Namespace) -> argparse.Namespace:
    """
    Convenience override: set the Infinity paths/architecture knobs from a small preset list.
    Add more variants by editing INFINITY_VARIANTS above.
    """
    v = getattr(args, "inf_variant", "2b_reg")
    if v not in INFINITY_VARIANTS:
        raise ValueError(f"Unknown --inf_variant {v!r}. Available: {list(INFINITY_VARIANTS.keys())}")

    overrides = INFINITY_VARIANTS[v]
    for k, val in overrides.items():
        setattr(args, k, val)

    print(f"[infinity] variant={v} applied:")
    for k, val in overrides.items():
        print(f"  - {k} = {val}")
    return args


# ============================================================
# Unified ES step
# ============================================================

@torch.no_grad()
def es_step_unified(
    *,
    theta: torch.Tensor,
    backend: ESBackend,
    lora_params: List[torch.nn.Parameter],
    lora_shapes: List[Tuple[int, ...]],
    clip_model,
    clip_processor,
    pick_model,
    pickscore_processor,
    noiser: EggRollNoiser,
    mix_weights: Tuple[float, float, float, float],
    seed: int,
    guidance_scale: float,
    pop_size: int,
    promptnorm_enabled: bool,
    theta_max_norm: float,
    max_step_norm: float,
    max_log_batches: int,
    save_dir: Path,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any], Optional[torch.Tensor], List[str]]:
    """
    Returns:
      theta_after,
      stats,
      img_dict (best/median/worst strips),
      rewards_for_hist (RAW combined mean per indiv),
      unique_texts (for captions)
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    info = backend.step_sampling_info(seed=seed)
    unique_ids: List[int] = info["unique_ids"]
    flat_ids: List[int] = info["flat_ids"]
    unique_texts: List[str] = info["unique_texts"]
    flat_texts: List[str] = info["flat_texts"]
    pid_to_j: Dict[int, int] = info["pid_to_j"]
    m: int = info["m"]
    total_imgs_per_indiv: int = info["total_imgs_per_indiv"]
    repeats = int(len(flat_ids) // max(1, m))

    log_batches = int(max(0, min(int(max_log_batches), repeats)))
    total_imgs_for_logging = int(log_batches * m)

    # ES noise
    eps = noiser.sample_eps(pop_size, theta.device)  # [n, D]

    # S_comb: [n, m] mean reward for that unique prompt/class over repeats
    S_comb = torch.empty((pop_size, m), device=theta.device, dtype=torch.float32)

    # Raw per-individual means over ALL images
    raw_comb_mean = torch.empty((pop_size,), device=theta.device, dtype=torch.float32)
    raw_aes_mean  = torch.empty((pop_size,), device=theta.device, dtype=torch.float32)
    raw_txt_mean  = torch.empty((pop_size,), device=theta.device, dtype=torch.float32)
    raw_noart_mean= torch.empty((pop_size,), device=theta.device, dtype=torch.float32)
    raw_pick_mean = torch.empty((pop_size,), device=theta.device, dtype=torch.float32)

    all_images_flat: List[List[Any]] = []  # pop_size lists of images (limited)

    print(f"[epoch {epoch}] backend={backend.name}")
    print(f"[epoch {epoch}] unique_count(m)={m}, repeats={repeats}, total_imgs/indiv={total_imgs_per_indiv}")
    print(f"[epoch {epoch}] logging first {log_batches} repeats -> {total_imgs_for_logging} imgs/indiv")
    print(f"[epoch {epoch}] unique prompts/classes:")
    for j, t in enumerate(unique_texts):
        print(f"  [{j}] {t}")

    for k in range(pop_size):
        theta_k = theta + noiser.sigma * eps[k]
        unflatten_to_params(theta_k, lora_params, lora_shapes)

        images_all = backend.generate_flat(flat_ids=flat_ids, seed=seed, guidance_scale=guidance_scale)

        per_prompt_comb: List[List[torch.Tensor]] = [[] for _ in range(m)]

        all_comb: List[torch.Tensor] = []
        all_aes: List[torch.Tensor] = []
        all_txt: List[torch.Tensor] = []
        all_noart: List[torch.Tensor] = []
        all_pick: List[torch.Tensor] = []

        cand_images_for_logging: List[Any] = []

        for idx in range(total_imgs_per_indiv):
            pid = flat_ids[idx]
            prompt_text = flat_texts[idx]
            img = images_all[idx]

            if idx < total_imgs_for_logging:
                cand_images_for_logging.append(img)

            reward_dict = compute_all_rewards(
                [img],
                prompt_text=prompt_text,
                clip_model=clip_model,
                clip_processor=clip_processor,
                mix_weights=mix_weights,
                pickscore_model=pick_model,
                pickscore_processor=pickscore_processor,
            )

            r_comb = reward_dict["combined"].float().to(theta.device)
            r_aes  = reward_dict["clip_aesthetic"].float().to(theta.device)
            r_txt  = reward_dict["clip_text"].float().to(theta.device)
            r_no   = reward_dict["no_artifacts"].float().to(theta.device)
            r_pick = reward_dict["pickscore"].float().to(theta.device)

            j = pid_to_j[int(pid)]
            per_prompt_comb[j].append(r_comb)

            all_comb.append(r_comb)
            all_aes.append(r_aes)
            all_txt.append(r_txt)
            all_noart.append(r_no)
            all_pick.append(r_pick)

        for j in range(m):
            S_comb[k, j] = torch.stack(per_prompt_comb[j]).mean()

        raw_comb_mean[k]  = torch.stack(all_comb).mean()
        raw_aes_mean[k]   = torch.stack(all_aes).mean()
        raw_txt_mean[k]   = torch.stack(all_txt).mean()
        raw_noart_mean[k] = torch.stack(all_noart).mean()
        raw_pick_mean[k]  = torch.stack(all_pick).mean()

        all_images_flat.append(cand_images_for_logging)

        per_prompt_str = ", ".join([f"{S_comb[k, j].item():.4f}" for j in range(m)])
        print(
            f"  indiv {k:02d} | raw_prompt_means=[{per_prompt_str}] | "
            f"raw_comb_mean={raw_comb_mean[k].item():.4f} | "
            f"aes={raw_aes_mean[k].item():.4f} txt={raw_txt_mean[k].item():.4f} "
            f"noart={raw_noart_mean[k].item():.4f} pick={raw_pick_mean[k].item():.4f} "
        )

    promptnorm_sigma_bar = torch.tensor(float("nan"), device=theta.device)
    mu_q = torch.empty((m,), device=theta.device, dtype=torch.float32)

    if promptnorm_enabled:
        scores, mu_q, promptnorm_sigma_bar = paper_prompt_normalized_scores(S_comb)  # [n]
        opt_scores = scores
    else:
        opt_scores = S_comb.mean(dim=1)  # [n]

    finite_mask = torch.isfinite(opt_scores)
    if not finite_mask.any():
        stats = {"summary/mean_reward": float("nan")}
        img_dict = {"best": None, "median": None, "worst": None}
        return theta, stats, img_dict, None, unique_texts

    best_img = median_img = worst_img = None
    if finite_mask.all() and total_imgs_for_logging > 0:
        _, sorted_idx = torch.sort(opt_scores)
        worst_idx = sorted_idx[0].item()
        best_idx = sorted_idx[-1].item()
        median_idx = sorted_idx[len(sorted_idx) // 2].item()

        epoch_dir = save_dir / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        best_strip   = make_prompt_strip(all_images_flat[best_idx], total_imgs_for_logging)
        median_strip = make_prompt_strip(all_images_flat[median_idx], total_imgs_for_logging)
        worst_strip  = make_prompt_strip(all_images_flat[worst_idx], total_imgs_for_logging)

        if best_strip is not None:
            best_strip.save(epoch_dir / "best.png")
            best_img = best_strip
        if median_strip is not None:
            median_strip.save(epoch_dir / "median.png")
            median_img = median_strip
        if worst_strip is not None:
            worst_strip.save(epoch_dir / "worst.png")
            worst_img = worst_strip

    opt_scores_f = opt_scores[finite_mask]
    eps_f = eps[finite_mask]

    raw_comb_f  = raw_comb_mean[finite_mask]
    raw_aes_f   = raw_aes_mean[finite_mask]
    raw_txt_f   = raw_txt_mean[finite_mask]
    raw_noart_f = raw_noart_mean[finite_mask]
    raw_pick_f  = raw_pick_mean[finite_mask]

    fitnesses = noiser.convert_fitnesses(opt_scores_f)

    theta_before = theta
    theta_after = noiser.do_update(theta, eps_f, fitnesses)

    theta_after = cap_step_norm(theta_before, theta_after, max_step_norm)
    theta_after = cap_theta_norm(theta_after, theta_max_norm)

    stats: Dict[str, float] = {
        "summary/mean_reward": float(opt_scores_f.mean().item()),
        "summary/max_reward":  float(opt_scores_f.max().item()),
        "summary/min_reward":  float(opt_scores_f.min().item()),
        "std_reward": float(opt_scores_f.std().item() if opt_scores_f.numel() > 1 else 0.0),

        "raw/combined_mean": float(raw_comb_f.mean().item()),
        "raw/combined_std":  float(raw_comb_f.std().item() if raw_comb_f.numel() > 1 else 0.0),
        "aesthetic_mean":    float(raw_aes_f.mean().item()),
        "clip_text_mean":    float(raw_txt_f.mean().item()),
        "no_artifacts_mean": float(raw_noart_f.mean().item()),
        "pickscore_mean":    float(raw_pick_f.mean().item()),

        "promptnorm/enabled": float(1.0 if promptnorm_enabled else 0.0),
        "promptnorm/sigma_bar": float(promptnorm_sigma_bar.item()) if promptnorm_enabled else float("nan"),

        "epoch/seed": int(seed),
        "epoch/m_unique": int(m),
        "epoch/repeats": int(len(flat_ids) // max(1, m)),
        "epoch/logged_repeats": int(log_batches),
        "epoch/logged_imgs_per_indiv": int(total_imgs_for_logging),
        "epoch/total_imgs_per_indiv": int(total_imgs_per_indiv),
    }

    for j in range(m):
        stats[f"prompt_{j}/mu_over_pop"] = float(mu_q[j].item()) if promptnorm_enabled else float(S_comb[:, j].mean().item())
        stats[f"prompt_{j}/raw_mean_over_pop"] = float(S_comb[:, j].mean().item())
        stats[f"prompt_{j}/raw_std_over_pop"]  = float(S_comb[:, j].std().item() if pop_size > 1 else 0.0)

    img_dict = {"best": best_img, "median": median_img, "worst": worst_img}
    rewards_for_hist = raw_comb_f.detach().cpu()  # histogram always RAW
    return theta_after, stats, img_dict, rewards_for_hist, unique_texts


# ============================================================
# Main
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v!r}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Unified ES/EGGROLL trainer")

    # --- core ---
    p.add_argument("--backend", type=str, default="infinity",
                   choices=["sana_one_step", "sana_pipeline", "var", "zimage", "infinity"])
    p.add_argument("--device", type=str, default="auto")

    # --- wandb/save ---
    p.add_argument("--wandb_project", type=str, default="ES-Unified")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--save_root", type=str, default="es_runs_unified")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--pop_size", type=int, default=8)

    # --- ES / EGGROLL ---
    p.add_argument("--sigma", type=float, default=1e-2)
    p.add_argument("--lr_scale", type=float, default=1e-1)
    p.add_argument("--egg_rank", type=int, default=1)
    p.add_argument("--use_antithetic", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--promptnorm", type=str2bool, nargs="?", const=True, default=True)

    # stabilizers
    p.add_argument("--theta_max_norm", type=float, default=40.0)
    p.add_argument("--max_step_norm", type=float, default=0.0)

    # --- reward weights (combined) ---
    p.add_argument("--w_aesthetic", type=float, default=0.0)
    p.add_argument("--w_text", type=float, default=0.0)
    p.add_argument("--w_noart", type=float, default=0.0)
    p.add_argument("--w_pick", type=float, default=1.0)

    # --- perf knobs (global) ---
    p.add_argument("--tf32", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--matmul_precision", type=str, default="high", choices=["high", "highest"])
    p.add_argument("--deterministic", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--cudnn_benchmark", type=str2bool, nargs="?", const=True, default=False)

    # --- torch.compile knobs (where used) ---
    p.add_argument("--torch_compile", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--compile_mode", type=str, default="max-autotune")
    p.add_argument("--compile_fullgraph", type=str2bool, nargs="?", const=True, default=True)

    # --- sampling semantics ---
    p.add_argument("--prompts_per_gen", type=int, default=4)
    p.add_argument("--batches_per_gen", type=int, default=4)
    p.add_argument("--max_log_batches", type=int, default=1)

    # --- Sana-specific ---
    p.add_argument("--sana_model", type=str, default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    p.add_argument("--sana_encoded_prompts", type=str, default="encoded_prompts_sana_unifed.pt")
    p.add_argument("--sana_guidance_scale", type=float, default=4.5)
    p.add_argument("--sana_width_latent", type=int, default=16)
    p.add_argument("--sana_height_latent", type=int, default=16)
    p.add_argument("--sana_dtype_latents", type=str, default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--sana_lora_r", type=int, default=2)
    p.add_argument("--sana_lora_alpha", type=int, default=8)
    p.add_argument("--sana_lora_dropout", type=float, default=0.0)
    p.add_argument("--sana_lora_targets", type=str, default="to_q,to_k,to_v,to_out.0,linear_1,linear_2,proj_out,linear")
    p.add_argument("--sana_prompts_txt", type=str, default="prompts_train_v3.txt")
    p.add_argument("--sana_auto_encode", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--sana_encode_bs", type=int, default=8)

    # --- VAR-specific ---
    p.add_argument("--var_depth", type=int, default=16)
    p.add_argument("--var_ckpt_dir", type=str, default="checkpoints_var")
    p.add_argument("--var_download_if_missing", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--var_guidance_scale", type=float, default=4.0)
    p.add_argument("--var_allowed_classes", type=str, default="all")
    p.add_argument("--var_classes_per_gen", type=int, default=2)
    p.add_argument("--var_lora_r", type=int, default=4)
    p.add_argument("--var_lora_alpha", type=int, default=16)
    p.add_argument("--var_lora_dropout", type=float, default=0.0)
    p.add_argument("--var_lora_targets", type=str, default="mat_qkv,proj,fc1,fc2,ada_lin.1,head_nm.ada_lin.1,head")

    # --- ZImage-specific ---
    p.add_argument("--zimage_model", type=str, default="Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--zimage_prompts_txt", type=str, default="untitled.txt")
    p.add_argument("--zimage_encoded_prompts", type=str, default="encoded_zimage_untitledtxt_big.pt")
    p.add_argument("--zimage_auto_encode", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--zimage_width", type=int, default=384)
    p.add_argument("--zimage_height", type=int, default=384)
    p.add_argument("--zimage_steps", type=int, default=7)
    p.add_argument("--zimage_guidance_scale", type=float, default=0.0)
    p.add_argument("--zimage_micro_batch", type=int, default=1)
    p.add_argument("--zimage_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--zimage_attention_backend", type=str, default="flash")
    p.add_argument("--zimage_compile_transformer", type=str2bool, nargs="?", const=True, default=False)

    # --- Infinity-specific ---
    p.add_argument(
        "--max_batch",
        type=int,
        default=2,
        help="Global cap on generation batch size. 0 disables. For Infinity this overrides --inf_micro_batch.",
    )

    # NEW: pick between your current model and the small one
    p.add_argument(
        "--inf_variant",
        type=str,
        default="8b_512",
        choices=list(INFINITY_VARIANTS.keys()),
        help="Select an Infinity preset variant (edit INFINITY_VARIANTS to add more).",
    )

    # These get overridden by --inf_variant (still kept for transparency / future use)
    p.add_argument("--inf_model_path", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_model_path"])
    p.add_argument("--inf_text_encoder_ckpt", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_text_encoder_ckpt"])
    p.add_argument("--inf_vae_path", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_vae_path"])
    p.add_argument("--inf_vae_type", type=int, default=INFINITY_VARIANTS["2b_reg"]["inf_vae_type"])
    p.add_argument("--inf_pn", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_pn"], choices=["0.06M", "0.25M", "0.60M", "1M"])
    p.add_argument("--inf_model_type", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_model_type"])
    p.add_argument("--inf_h_div_w_template", type=float, default=1.0)
    p.add_argument("--inf_text_channels", type=int, default=2048)
    p.add_argument("--inf_apply_spatial_patchify", type=int, default=INFINITY_VARIANTS["2b_reg"]["inf_apply_spatial_patchify"], choices=[0, 1])
    p.add_argument("--inf_use_flex_attn", type=int, default=0, choices=[0, 1])
    p.add_argument("--inf_bf16", type=int, default=1, choices=[0, 1])
    p.add_argument("--inf_checkpoint_type", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_checkpoint_type"], choices=["torch", "torch_shard"])
    p.add_argument("--inf_guidance_scale", type=float, default=3.0)

    p.add_argument("--inf_prompts_txt", type=str, default="untitled1.txt")
    p.add_argument("--inf_encoded_prompts", type=str, default=INFINITY_VARIANTS["2b_reg"]["inf_encoded_prompts"])
    p.add_argument("--inf_auto_encode", type=str2bool, nargs="?", const=True, default=True)
    p.add_argument("--inf_encode_bs", type=int, default=16)
    p.add_argument("--inf_drop_text_encoder_after_encode", type=str2bool, nargs="?", const=True, default=True)

    p.add_argument("--inf_cfg_list", type=str, default="3")
    p.add_argument("--inf_tau_list", type=str, default="1")
    p.add_argument("--inf_cfg_insertion_layer", type=int, default=0)
    p.add_argument("--inf_sampling_per_bits", type=int, default=1)
    p.add_argument("--inf_enable_positive_prompt", type=int, default=0, choices=[0, 1])
    p.add_argument("--inf_top_k", type=int, default=900)
    p.add_argument("--inf_top_p", type=float, default=0.97)
    p.add_argument("--inf_micro_batch", type=int, default=0)

    p.add_argument("--inf_lora_r", type=int, default=2)
    p.add_argument("--inf_lora_alpha", type=int, default=8)
    p.add_argument("--inf_lora_dropout", type=float, default=0.0)
    p.add_argument("--inf_lora_targets", type=str, default="fc1")

    # GGUF
    p.add_argument("--zimage_use_gguf", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--zimage_gguf_repo_id", type=str, default="jayn7/Z-Image-Turbo-GGUF")
    p.add_argument("--zimage_gguf_filename", type=str, default="z_image_turbo-Q3_K_S.gguf")
    p.add_argument("--zimage_gguf_local_dir", type=str, default="gguf_cache")
    p.add_argument("--zimage_gguf_local_path", type=str, default="")

    # ZImage LoRA
    p.add_argument("--zimage_lora_r", type=int, default=2)
    p.add_argument("--zimage_lora_alpha", type=int, default=8)
    p.add_argument("--zimage_lora_dropout", type=float, default=0.0)
    p.add_argument("--zimage_lora_targets", type=str, default="to_q,to_k,to_v,linear,w1,w2,w3")

    # optional VAE decoder LoRA
    p.add_argument("--zimage_use_vae_lora", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--zimage_vae_lora_r", type=int, default=2)
    p.add_argument("--zimage_vae_lora_alpha", type=int, default=8)
    p.add_argument("--zimage_vae_lora_dropout", type=float, default=0.0)
    p.add_argument("--zimage_vae_lora_targets", type=str, default="to_q,to_k,to_v,to_out.0")

    return p


def main():
    args = build_argparser().parse_args()

    # Apply Infinity preset overrides (only affects Infinity backend)
    if args.backend == "infinity":
        args = apply_infinity_variant(args)

    device = pick_device(args.device)
    set_global_perf_knobs(
        device=device,
        use_tf32=args.tf32,
        matmul_precision=args.matmul_precision,
        deterministic=args.deterministic,
        cudnn_benchmark=args.cudnn_benchmark,
    )

    torch.set_float32_matmul_precision(args.matmul_precision)
    print(f"[init] device={device}  tf32={args.tf32}  matmul_precision={args.matmul_precision}")
    print(f"[init] deterministic={args.deterministic} cudnn_benchmark={args.cudnn_benchmark}")

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    backend_name = args.backend
    run_name = args.run_name or (
        f"{backend_name}"
        f"_sigma{args.sigma:.0e}_lr{args.lr_scale:.0e}_ant{int(args.use_antithetic)}"
        f"_pop{args.pop_size}"
        f"_pgen{args.prompts_per_gen}_bgen{args.batches_per_gen}"
        f"_promptnorm{int(args.promptnorm)}"
    )

    run_dir = save_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_lora_dir = run_dir / "latest_lora"
    latest_lora_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "latest_lora_meta.pt"

    mix_weights = (args.w_aesthetic, args.w_text, args.w_noart, args.w_pick)

    # -------------------------
    # Build backend
    # -------------------------
    if args.backend in ("sana_one_step", "sana_pipeline"):
        sana_cfg = SanaConfig(
            model_name=args.sana_model,
            backend_mode=("one_step" if args.backend == "sana_one_step" else "pipeline"),

            prompts_txt_path=args.sana_prompts_txt,
            encoded_prompt_path=args.sana_encoded_prompts,
            auto_encode_if_missing=bool(args.sana_auto_encode),
            encode_batch_size=int(args.sana_encode_bs),

            guidance_scale=args.sana_guidance_scale,
            width_latent=args.sana_width_latent,
            height_latent=args.sana_height_latent,
            batch_size=1,
            prompts_per_gen=args.prompts_per_gen,
            batches_per_gen=args.batches_per_gen,
            max_log_batches=args.max_log_batches,
            torch_compile=args.torch_compile,
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            dtype_latents=args.sana_dtype_latents,

            lora_r=args.sana_lora_r,
            lora_alpha=args.sana_lora_alpha,
            lora_dropout=args.sana_lora_dropout,
            lora_target_modules=[x.strip() for x in args.sana_lora_targets.split(",") if x.strip()],
        )
        backend: ESBackend = SanaBackend(device=device, cfg=sana_cfg)

    elif args.backend == "var":
        allowed = parse_int_list(args.var_allowed_classes)
        var_cfg = VarConfig(
            model_depth=args.var_depth,
            ckpt_dir=args.var_ckpt_dir,
            download_if_missing=bool(args.var_download_if_missing),
            guidance_scale=args.var_guidance_scale,
            allowed_classes=allowed,
            classes_per_gen=args.var_classes_per_gen,
            batches_per_gen=args.batches_per_gen,
            max_log_batches=args.max_log_batches,
            torch_compile=args.torch_compile,
            compile_mode=args.compile_mode,
            compile_fullgraph=args.compile_fullgraph,
            lora_r=args.var_lora_r,
            lora_alpha=args.var_lora_alpha,
            lora_dropout=args.var_lora_dropout,
            lora_target_modules=[x.strip() for x in args.var_lora_targets.split(",") if x.strip()],
        )
        backend = VarBackend(device=device, cfg=var_cfg)

    elif args.backend == "zimage":
        if not device.startswith("cuda"):
            raise RuntimeError("zimage backend requires CUDA in your setup.")
        z_cfg = ZImageConfig(
            model_name=args.zimage_model,
            prompts_txt_path=args.zimage_prompts_txt,
            encoded_prompt_path=args.zimage_encoded_prompts,
            auto_encode_if_missing=bool(args.zimage_auto_encode),

            width_px=args.zimage_width,
            height_px=args.zimage_height,
            num_inference_steps=args.zimage_steps,
            guidance_scale=args.zimage_guidance_scale,
            micro_batch=args.zimage_micro_batch,

            prompts_per_gen=args.prompts_per_gen,
            batches_per_gen=args.batches_per_gen,
            max_log_batches=args.max_log_batches,

            compile_transformer=bool(args.zimage_compile_transformer),
            attention_backend=args.zimage_attention_backend,

            use_gguf=bool(args.zimage_use_gguf),
            gguf_repo_id=args.zimage_gguf_repo_id,
            gguf_filename=args.zimage_gguf_filename,
            gguf_local_dir=args.zimage_gguf_local_dir,
            gguf_local_path=(args.zimage_gguf_local_path if args.zimage_gguf_local_path else None),

            lora_r=args.zimage_lora_r,
            lora_alpha=args.zimage_lora_alpha,
            lora_dropout=args.zimage_lora_dropout,
            lora_target_modules=[x.strip() for x in args.zimage_lora_targets.split(",") if x.strip()],

            use_vae_decoder_lora=bool(args.zimage_use_vae_lora),
            vae_lora_r=args.zimage_vae_lora_r,
            vae_lora_alpha=args.zimage_vae_lora_alpha,
            vae_lora_dropout=args.zimage_vae_lora_dropout,
            vae_lora_target_modules=[x.strip() for x in args.zimage_vae_lora_targets.split(",") if x.strip()],

            dtype=args.zimage_dtype,
        )
        backend = ZImageBackend(device=device, cfg=z_cfg)

    elif args.backend == "infinity":
        def parse_floats(s: str):
            xs = [float(x) for x in str(s).split(",") if str(x).strip() != ""]
            return xs[0] if len(xs) == 1 else xs

        inf_cfg = InfinityConfig(
            model_path=args.inf_model_path,
            text_encoder_ckpt=args.inf_text_encoder_ckpt,
            vae_path=args.inf_vae_path,

            pn=args.inf_pn,
            model_type=args.inf_model_type,
            vae_type=int(args.inf_vae_type),
            h_div_w_template=float(args.inf_h_div_w_template),
            text_channels=int(args.inf_text_channels),
            apply_spatial_patchify=int(args.inf_apply_spatial_patchify),
            use_flex_attn=int(args.inf_use_flex_attn),
            bf16=int(args.inf_bf16),
            checkpoint_type=args.inf_checkpoint_type,

            prompts_txt_path=args.inf_prompts_txt,
            encoded_prompt_path=args.inf_encoded_prompts,
            auto_encode_if_missing=bool(args.inf_auto_encode),
            encode_batch_size=int(args.inf_encode_bs),
            drop_text_encoder_after_encode=bool(args.inf_drop_text_encoder_after_encode),

            prompts_per_gen=int(args.prompts_per_gen),
            batches_per_gen=int(args.batches_per_gen),
            max_log_batches=int(args.max_log_batches),

            cfg_list=parse_floats(args.inf_cfg_list),
            tau_list=parse_floats(args.inf_tau_list),
            cfg_insertion_layer=int(args.inf_cfg_insertion_layer),
            sampling_per_bits=int(args.inf_sampling_per_bits),
            enable_positive_prompt=int(args.inf_enable_positive_prompt),
            top_k=int(args.inf_top_k),
            top_p=float(args.inf_top_p),

            # torch.compile (reuse global args; InfinityBackend.compile_if_requested should consume these)
            torch_compile=bool(args.torch_compile),
            compile_mode=str(args.compile_mode),
            compile_fullgraph=bool(args.compile_fullgraph),
            compile_vae=bool(args.torch_compile),

            micro_batch=int(args.max_batch) if int(args.max_batch) > 0 else int(args.inf_micro_batch),

            lora_r=int(args.inf_lora_r),
            lora_alpha=int(args.inf_lora_alpha),
            lora_dropout=float(args.inf_lora_dropout),
            lora_target_modules=[x.strip() for x in args.inf_lora_targets.split(",") if x.strip()],
        )
        backend = InfinityBackend(device=device, cfg=inf_cfg)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------------
    # Init models
    # -------------------------
    backend.init_and_attach_lora()
    backend.compile_if_requested()

    lora_params, lora_shapes = backend.collect_lora_params()
    theta = flatten_params(lora_params).to(device=device, dtype=torch.float32)
    theta_init = theta.clone().detach()

    print(f"[init] backend={backend.name}  trainable_params={theta.numel():,}  theta_init_norm={theta_init.norm().item():.4f}")

    clip_model, clip_processor = load_clip_model_and_processor(device)
    pick_model, pickscore_processor = load_pickscore_model_and_processor(device)

    noiser = EggRollNoiser(
        param_shapes=lora_shapes,
        sigma=args.sigma,
        lr_scale=args.lr_scale,
        rank=args.egg_rank,
        use_antithetic=bool(args.use_antithetic),
    )

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "backend": backend.name,
            "pop_size": args.pop_size,
            "num_epochs": args.num_epochs,
            "sigma": args.sigma,
            "lr_scale": args.lr_scale,
            "egg_rank": args.egg_rank,
            "use_antithetic": bool(args.use_antithetic),
            "promptnorm": bool(args.promptnorm),
            "theta_max_norm": args.theta_max_norm,
            "max_step_norm": args.max_step_norm,
            "mix_weights": mix_weights,
            "device": device,
            "torch_compile": bool(args.torch_compile),
            "compile_mode": args.compile_mode,
            "compile_fullgraph": bool(args.compile_fullgraph),
            "tf32": bool(args.tf32),
            "matmul_precision": args.matmul_precision,
            "deterministic": bool(args.deterministic),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "prompts_per_gen": args.prompts_per_gen,
            "batches_per_gen": args.batches_per_gen,
            "max_log_batches": args.max_log_batches,
            "num_params": int(noiser.num_params),
            "max_batch": int(getattr(args, "max_batch", 0)),
            "inf_variant": getattr(args, "inf_variant", ""),
            "inf_guidance_scale": float(getattr(args, "inf_guidance_scale", 0.0)),
        },
    )

    last_stats = None
    extra_meta: Dict[str, Any] = {"run_name": run_name, "wandb_project": args.wandb_project}

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch} ({backend.name}) ===")

        theta, stats, img_dict, rewards_vec_raw, unique_texts = es_step_unified(
            theta=theta,
            backend=backend,
            lora_params=lora_params,
            lora_shapes=lora_shapes,
            clip_model=clip_model,
            clip_processor=clip_processor,
            pick_model=pick_model,
            pickscore_processor=pickscore_processor,
            noiser=noiser,
            mix_weights=mix_weights,
            seed=epoch,
            guidance_scale=(
                args.sana_guidance_scale if backend.name.startswith("sana") else
                args.var_guidance_scale if backend.name == "var_class" else
                args.inf_guidance_scale if backend.name == "infinity" else
                args.zimage_guidance_scale
            ),
            pop_size=args.pop_size,
            promptnorm_enabled=bool(args.promptnorm),
            theta_max_norm=args.theta_max_norm,
            max_step_norm=args.max_step_norm,
            max_log_batches=args.max_log_batches,
            save_dir=run_dir,
            epoch=epoch,
        )
        last_stats = stats

        with torch.no_grad():
            delta_theta = theta - theta_init
            stats["theta_norm"] = float(theta.norm().item())
            stats["delta_theta_norm"] = float(delta_theta.norm().item())

            theta_hist_vals = subsample_for_hist(theta)
            delta_hist_vals = subsample_for_hist(delta_theta)

            stats["lora/mean_abs"] = float(theta.detach().abs().mean().item())
            stats["lora/delta_mean_abs"] = float(delta_theta.detach().abs().mean().item())

        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            save_latest_checkpoint(
                theta=theta,
                backend=backend,
                lora_params=lora_params,
                lora_shapes=lora_shapes,
                save_dir=latest_lora_dir,
                meta_path=meta_path,
                epoch=epoch + 1,
                stats=stats,
                extra_meta=extra_meta,
            )

        wandb_imgs = {}
        prompts_caption = "\n".join([f"- {t}" for t in unique_texts])
        for key, img in img_dict.items():
            if img is not None:
                wandb_imgs[f"images/{key}"] = wandb.Image(img, caption=f"{key} epoch {epoch}\n{prompts_caption}")

        log_payload = {"epoch": epoch, **stats, **wandb_imgs}

        if rewards_vec_raw is not None and rewards_vec_raw.numel() > 0:
            log_payload["reward_hist"] = wandb.Histogram(rewards_vec_raw.numpy())

        log_payload["lora/weights_hist"] = wandb.Histogram(theta_hist_vals.numpy())
        log_payload["lora/delta_hist"] = wandb.Histogram(delta_hist_vals.numpy())

        wandb.log(log_payload, step=epoch)

    if args.save_every <= 0 or (args.num_epochs % args.save_every != 0):
        if last_stats is not None:
            save_latest_checkpoint(
                theta=theta,
                backend=backend,
                lora_params=lora_params,
                lora_shapes=lora_shapes,
                save_dir=latest_lora_dir,
                meta_path=meta_path,
                epoch=args.num_epochs,
                stats=last_stats,
                extra_meta=extra_meta,
            )

    print(f"\nDone. Latest LoRA checkpoint at: {latest_lora_dir}")
    print(f"Meta checkpoint at: {meta_path}")
    run.finish()


if __name__ == "__main__":
    main()