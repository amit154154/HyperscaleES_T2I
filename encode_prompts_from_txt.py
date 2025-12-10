#!/usr/bin/env python
from pathlib import Path
import torch

from diffusers import SanaSprintPipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"

PROMPTS_TXT_PATH = Path("prompts_test.txt")              # 1 prompt per line
ENCODED_SAVE_PATH = Path("encoded_prompts_multi_test.pt")

complex_human_instruction = None

# how many prompts to encode per batch
BATCH_SIZE = 4


def load_prompts_from_txt(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip()]
    return prompts


@torch.no_grad()
def main():
    print(f"[encode] Reading prompts from {PROMPTS_TXT_PATH} ...")
    prompts = load_prompts_from_txt(PROMPTS_TXT_PATH)
    num_prompts = len(prompts)
    print(f"[encode] Loaded {num_prompts} prompts.")
    for i, p in enumerate(prompts):
        print(f"  {i:02d}: {p}")

    # --- build pipeline with text encoder ---
    print("[encode] Loading Sana pipeline with text encoder...")
    pipe = SanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        torch_dtype=torch.float16,
    ).to(DEVICE)

    print("[encode] Encoding prompts with text encoder on GPU in batches...")
    all_embeds = []
    all_masks = []

    for start in range(0, num_prompts, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_prompts)
        batch_prompts = prompts[start:end]
        print(f"[encode] Batch {start}:{end} / {num_prompts}")

        prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
            batch_prompts,
            num_images_per_prompt=1,
            device=DEVICE,
            prompt_embeds=None,
            prompt_attention_mask=None,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=complex_human_instruction,
        )

        # move to CPU immediately to free VRAM
        all_embeds.append(prompt_embeds.cpu())
        all_masks.append(prompt_attention_mask.cpu())

        # extra cleanup just in case
        del prompt_embeds, prompt_attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # concatenate along batch dimension
    prompt_embeds = torch.cat(all_embeds, dim=0)
    prompt_attention_mask = torch.cat(all_masks, dim=0)

    print("[encode] Encoded prompts:")
    print(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}, device: {prompt_embeds.device}")
    print(f"  attention_mask shape: {prompt_attention_mask.shape}, dtype: {prompt_attention_mask.dtype}, device: {prompt_attention_mask.device}")

    print(f"[encode] Saving encoded prompts to: {ENCODED_SAVE_PATH}")
    to_save = {
        "prompts": prompts,
        "prompt_embeds": prompt_embeds,                 # [num_prompts, seq, dim]
        "prompt_attention_mask": prompt_attention_mask, # [num_prompts, seq]
    }
    torch.save(to_save, ENCODED_SAVE_PATH)
    print("[encode] Saved encoded prompts.")

    # free GPU
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        print("[encode] Moving text encoder to CPU and clearing CUDA cache...")
        pipe.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[encode] Text encoder moved to CPU.")
    else:
        print("[encode] No text encoder found on pipeline (or already moved).")


if __name__ == "__main__":
    main()