import torch
from typing import Dict, Tuple, List, Optional

from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
)

# -------------------------------------------------------------------
# Default texts (global, but not confusing)
# -------------------------------------------------------------------
AESTHETIC_TEXT = "a high quality, professional, beautiful, aesthetically pleasing image"
NEGATIVE_TEXT = (
    "blurry, low resolution, noisy, pixelated, washed out colors, oversaturated "
)


# -------------------------------------------------------------------
# CLIP loader
# -------------------------------------------------------------------
def load_clip_model_and_processor(device: str = "cuda:0") -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model + processor once and reuse.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    model.to(device).eval()
    return model, processor


# -------------------------------------------------------------------
# PickScore loader
# -------------------------------------------------------------------
def load_pickscore_model_and_processor(
    device: str = "cuda:0",
) -> Tuple[AutoModel, AutoProcessor]:
    """
    Load PickScore_v1 + its CLIP-H processor.

      - model:    yuvalkirstain/PickScore_v1
      - encoder:  laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    """
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path, use_fast=True)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path)
    model.to(device).eval()
    return model, processor


# -------------------------------------------------------------------
# CLIP helper
# -------------------------------------------------------------------
@torch.no_grad()
def _clip_image_text_sims(
    images,
    texts: List[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> torch.Tensor:
    """
    images: list of PIL.Image
    texts: list[str] (M texts)

    Returns:
        sims_mean: tensor of shape [M], each in [0,1]
    """
    if len(images) == 0:
        device = next(clip_model.parameters()).device
        return torch.zeros(len(texts), device=device)

    device = next(clip_model.parameters()).device

    img_inputs = clip_processor(images=images, return_tensors="pt")
    txt_inputs = clip_processor(text=texts, return_tensors="pt", padding=True)

    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}

    img_embeds = clip_model.get_image_features(pixel_values=img_inputs["pixel_values"])
    txt_embeds = clip_model.get_text_features(
        input_ids=txt_inputs["input_ids"],
        attention_mask=txt_inputs["attention_mask"],
    )

    # Normalize
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    sims = img_embeds @ txt_embeds.T          # [B, M], ~[-1,1]
    sims = (sims + 1.0) / 2.0                 # -> [0,1]

    sims_mean = sims.mean(dim=0)              # [M]
    return sims_mean


# -------------------------------------------------------------------
# PickScore helper
# -------------------------------------------------------------------
@torch.no_grad()
def compute_pickscore(
    images,
    prompt: str,
    pick_model: AutoModel,
    pick_processor: AutoProcessor,
) -> torch.Tensor:
    """
    Compute a scalar PickScore-style reward for a batch of PIL images.

    We follow the official example:
      - embed text + image with CLIP-H
      - take logit_scale * dot(text, image)
      - average across images -> scalar reward
    """
    if len(images) == 0:
        device = next(pick_model.parameters()).device
        return torch.tensor(0.0, device=device)

    device = next(pick_model.parameters()).device

    image_inputs = pick_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = pick_processor(
        text=[prompt],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    image_embs = pick_model.get_image_features(**image_inputs)
    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    text_embs = pick_model.get_text_features(**text_inputs)
    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    # scores: [num_text(=1), num_images]
    scores = pick_model.logit_scale.exp() * (text_embs @ image_embs.T)
    score = scores.mean()  # scalar
    return score


# -------------------------------------------------------------------
# Combined rewards (multi-prompt clean version)
# -------------------------------------------------------------------
@torch.no_grad()
def compute_all_rewards(
    images,
    *,
    prompt_text: str,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    mix_weights: Tuple[float, ...] = (0.3, 0.3, 0.2, 0.2),
    pickscore_model: Optional[AutoModel] = None,
    pickscore_processor: Optional[AutoProcessor] = None,
    pickscore_prompt: Optional[str] = None,
    aesthetic_text: str = AESTHETIC_TEXT,
    negative_text: str = NEGATIVE_TEXT,
) -> Dict[str, torch.Tensor]:
    """
    Compute scalar rewards for a batch of PIL images given *their* prompt.

    Args:
        images:
            list of PIL.Image, usually generated from `prompt_text`.
        prompt_text:
            The actual text prompt used for these images.
        clip_model / clip_processor:
            CLIP model + processor.
        mix_weights:
            (w_aesthetic, w_align, w_no_artifacts)
            or (w_aesthetic, w_align, w_no_artifacts, w_pickscore).
        pickscore_model / pickscore_processor:
            If provided, compute PickScore; otherwise 0.
        pickscore_prompt:
            Text for PickScore; if None, defaults to `prompt_text`.
        aesthetic_text:
            Text describing an aesthetically pleasing image.
        negative_text:
            Text describing bad artifacts; we use (1 - CLIP_sim).

    Returns:
        dict with:
            - clip_aesthetic  (scalar tensor)
            - clip_text       (scalar tensor)
            - no_artifacts    (scalar tensor)
            - pickscore       (scalar tensor)
            - combined        (scalar tensor)
    """
    device = next(clip_model.parameters()).device

    if len(images) == 0:
        zero = torch.tensor(0.0, device=device)
        return {
            "clip_aesthetic": zero,
            "clip_text": zero,
            "no_artifacts": zero,
            "pickscore": zero,
            "combined": zero,
        }

    if pickscore_prompt is None:
        pickscore_prompt = prompt_text

    # --- CLIP-based rewards ---
    sims_mean = _clip_image_text_sims(
        images,
        [aesthetic_text, prompt_text, negative_text],
        clip_model,
        clip_processor,
    )
    # sims_mean: [3], each in [0,1]
    clip_aesthetic = sims_mean[0]
    clip_align = sims_mean[1]
    no_artifacts = 1.0 - sims_mean[2]  # high reward when dissimilar to NEGATIVE_TEXT

    # --- PickScore reward (optional) ---
    if pickscore_model is not None and pickscore_processor is not None:
        pickscore = compute_pickscore(
            images,
            prompt=pickscore_prompt,
            pick_model=pickscore_model,
            pick_processor=pickscore_processor,
        ).to(device)
    else:
        pickscore = torch.tensor(0.0, device=device)

    # --- Combine ---
    if len(mix_weights) == 3:
        w_aes, w_align, w_noart = mix_weights
        w_pick = 0.0
    elif len(mix_weights) == 4:
        w_aes, w_align, w_noart, w_pick = mix_weights
    else:
        raise ValueError(f"mix_weights must have length 3 or 4, got {len(mix_weights)}")

    combined = (
        w_aes * clip_aesthetic
        + w_align * clip_align
        + w_noart * no_artifacts
        + w_pick * pickscore
    )

    return {
        "clip_aesthetic": clip_aesthetic.detach(),
        "clip_text": clip_align.detach(),
        "no_artifacts": no_artifacts.detach(),
        "pickscore": pickscore.detach(),
        "combined": combined.detach(),
    }


if __name__ == "__main__":
    """
    Quick sanity check for rewards.py

    - Loads CLIP + PickScore models
    - Creates a dummy PIL image
    - Runs compute_all_rewards
    - Prints the resulting scores
    """
    import sys
    from PIL import Image

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {DEVICE}", file=sys.stderr)

    # 1) Load CLIP
    print("[main] Loading CLIP model + processor...", file=sys.stderr)
    clip_model, clip_processor = load_clip_model_and_processor(device=DEVICE)
    print("    CLIP loaded.", file=sys.stderr)

    # 2) Load PickScore
    print("[main] Loading PickScore model + processor...", file=sys.stderr)
    pick_model, pick_processor = load_pickscore_model_and_processor(device=DEVICE)
    print("    PickScore loaded.", file=sys.stderr)

    # 3) Dummy image
    print("[main] Creating dummy test image...", file=sys.stderr)
    test_img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    images = [test_img]

    # 4) Mix weights
    mix_weights = (0.3, 0.3, 0.2, 0.2)

    # 5) Compute rewards for a test prompt
    test_prompt = "a plain white square on a white background"
    print("[main] Computing rewards...", file=sys.stderr)
    rewards = compute_all_rewards(
        images,
        prompt_text=test_prompt,
        clip_model=clip_model,
        clip_processor=clip_processor,
        mix_weights=mix_weights,
        pickscore_model=pick_model,
        pickscore_processor=pick_processor,
    )

    print("\n=== Reward debug output ===")
    for k, v in rewards.items():
        print(f"{k:20s}: {float(v):.6f}")