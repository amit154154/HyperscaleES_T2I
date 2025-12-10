#!/usr/bin/env python
"""
Generate images for PartiPrompts encodings using ES-SanaSprint-OneStep.

- Loads encoded prompts from a .pt file (created by encode_benchmark_prompts.py)
- Uses either:
    * base SanaSprintOneStepES ("base" mode), or
    * ES-trained LoRA ("lora" mode)
- Generates one image per prompt.
- Processes prompts in batches for speed.
- In batch mode, all prompts in a batch share the same base seed; if you need
  strict seed == prompt_idx, set --batch_size 1.

Usage:

  python generate_partiprompts_es.py \
      --encoded_path encoded_parti_prompts.pt \
      --mode base \
      --out_dir outputs_partiprompts_base

  python generate_partiprompts_es.py \
      --encoded_path encoded_parti_prompts.pt \
      --mode lora \
      --lora_dir es_runs_eggroll_pickscore_multiprompt/best_lora \
      --out_dir outputs_partiprompts_lora
"""

import argparse
import re
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm

from SanaSprintOneStep import SanaTransformerOneStepES  # your wrapper


# ---------------------------
# Config / device
# ---------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"
MODEL_NAME = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"


def slugify(text: str, max_len: int = 80) -> str:
    """
    Turn a prompt into a filesystem-friendly slug.
    """
    text = text.strip().lower()
    # Replace non-word chars with underscores
    text = re.sub(r"[^\w]+", "_", text)
    text = text.strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    if not text:
        text = "prompt"
    return text


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoded_path",
        type=str,
        default="encoded_parti_prompts.pt",
        help="Path to .pt file with encoded PartiPrompts (prompts, embeds, masks).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_partiprompts_base",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "lora"],
        default="base",
        help="Whether to use base Sana or ES-trained LoRA.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="es_runs_eggroll_pickscore_multiprompt/best_lora",
        help="Directory containing LoRA adapter weights (for mode='lora').",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="CFG guidance scale.",
    )
    parser.add_argument(
        "--width_latent",
        type=int,
        default=32,
        help="Latent width (32 -> 32 * 8 = 256px if vae scale factor is 8).",
    )
    parser.add_argument(
        "--height_latent",
        type=int,
        default=32,
        help="Latent height.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional limit on number of prompts to generate (for quick tests).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of prompts to process per forward pass. "
             "Set to 1 if you need strict seed == prompt_idx.",
    )
    args = parser.parse_args()

    encoded_path = Path(args.encoded_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] Loading encoded prompts from {encoded_path} ...")
    data = torch.load(encoded_path, map_location="cpu")

    prompt_embeds_all = data["prompt_embeds"]           # [P, seq, dim]
    attention_mask_all = data["prompt_attention_mask"]  # [P, seq]
    prompts_list = data.get("prompts", None)

    num_prompts = prompt_embeds_all.shape[0]
    print(f"[init] num_prompts: {num_prompts}")
    if prompts_list is not None:
        print("  Example prompts:")
        for i, p in enumerate(prompts_list[:5]):
            print(f"   [{i:02d}] {p}")

    if args.max_prompts is not None:
        num_to_gen = min(num_prompts, args.max_prompts)
    else:
        num_to_gen = num_prompts

    print(f"[init] Will generate for {num_to_gen} prompts.")
    print(f"[init] Using batch_size={args.batch_size}")

    # ---------------------------
    # Load model (base or LoRA)
    # ---------------------------
    print(f"[init] Loading SanaSprintOneStepES (mode={args.mode}) ...")
    sana = SanaTransformerOneStepES(
        model_name=MODEL_NAME,
        device=DEVICE,
        DTYPE=torch.float16,
        sigma_data=0.5,
    )
    sana.transformer.to(DEVICE).eval()

    if args.mode == "lora":
        lora_dir = Path(args.lora_dir)
        print(f"[init] Loading LoRA adapter from {lora_dir} ...")
        sana.transformer = PeftModel.from_pretrained(
            sana.transformer,
            lora_dir,
        )
        sana.transformer.to(DEVICE).eval()
        print("[init] LoRA adapter loaded.")

    print("[init] Model ready.")

    # ---------------------------
    # Generation loop (batched)
    # ---------------------------
    batch_size = max(1, args.batch_size)

    for start in tqdm(
        range(0, num_to_gen, batch_size),
        desc=f"Generating ({args.mode})",
        total=(num_to_gen + batch_size - 1) // batch_size,
    ):
        end = min(start + batch_size, num_to_gen)
        current_bs = end - start

        # [B, seq, dim] / [B, seq]
        batch_embeds = prompt_embeds_all[start:end].to(DEVICE)
        batch_masks = attention_mask_all[start:end].to(DEVICE)

        # Base seed for this batch.
        # If you need exact reproducibility with seed == prompt_idx, use batch_size=1.
        seed = start

        # Optional: print first prompt in batch
        # if prompts_list is not None:
        #     print(f"\n[gen] batch {start}:{end}, seed={seed}")
        #     print("      first prompt:", prompts_list[start])

        images, _latents = sana.sana_one_step_trigflow(
            prompt_embeds=batch_embeds,
            prompt_attention_mask=batch_masks,
            latents=None,
            seed=seed,
            guidance_scale=args.guidance_scale,
            width_latent=args.width_latent,
            height_latent=args.height_latent,
        )

        if len(images) < current_bs:
            print(
                f"  ⚠ Expected {current_bs} images, "
                f"got {len(images)}. Truncating to min."
            )

        for j in range(min(current_bs, len(images))):
            idx = start + j
            img = images[j]

            if prompts_list is not None:
                prompt_text = prompts_list[idx]
            else:
                prompt_text = f"prompt_{idx}"

            slug = slugify(prompt_text)
            filename = f"{idx:04d}_{slug}.png"
            save_path = out_dir / filename
            img.save(save_path)

        # free a bit of VRAM between batches
        del batch_embeds, batch_masks, images, _latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[done] Generation finished.")


if __name__ == "__main__":
    main()



