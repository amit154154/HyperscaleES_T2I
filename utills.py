from pathlib import Path
import torch
from torch import nn
from PIL import Image



# -------------------------
# Helpers for trainable (LoRA) params
# -------------------------
def get_trainable_params_and_shapes(module: nn.Module):
    params = []
    shapes = []
    for p in module.parameters():
        if p.requires_grad:
            params.append(p)
            shapes.append(p.shape)
    return params, shapes


def flatten_params(params):
    return torch.cat([p.data.view(-1) for p in params])


def unflatten_to_params(flat: torch.Tensor, params, shapes):
    assert flat.numel() == sum(p.numel() for p in params)
    idx = 0
    for p, shape in zip(params, shapes):
        numel = p.numel()
        chunk = flat[idx: idx + numel].view(shape)
        p.data.copy_(chunk)
        idx += numel


# -------------------------
# Simple fitness shaping (zero mean, unit std)
# -------------------------
def standardize_fitness(rewards: torch.Tensor) -> torch.Tensor:
    """
    Center and scale rewards: (r - mean) / std.
    This is similar in spirit to fitness shaping used in ES/EGGROLL.
    """
    r = rewards.detach()
    mean = r.mean()
    std = r.std()
    if std < 1e-8:
        return torch.zeros_like(r)
    return (r - mean) / (std + 1e-8)

def save_candidate_image(images, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(images) > 0:
        images[0].save(path)

# -------------------------
# Compose strip of per-prompt images
# -------------------------
def make_prompt_strip(
    images_for_candidate,
    num_prompts: int,
    tile_size: int = 256,
    bg_color=(0, 0, 0),
):
    """
    images_for_candidate: list of PIL.Image, one per prompt (length ≤ num_prompts)
    Returns a single PIL.Image of size (tile_size * num_prompts, tile_size)
    where each prompt's image is resized to tile_size×tile_size and pasted
    horizontally.
    """
    if num_prompts <= 0:
        return None

    strip = Image.new("RGB", (tile_size * num_prompts, tile_size), color=bg_color)

    for i in range(num_prompts):
        if i < len(images_for_candidate) and images_for_candidate[i] is not None:
            img = images_for_candidate[i]
            img = img.convert("RGB")
            img = img.resize((tile_size, tile_size), Image.LANCZOS)
            strip.paste(img, (i * tile_size, 0))

    return strip
