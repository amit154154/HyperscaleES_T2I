from __future__ import annotations

import torch
from torch import nn
from PIL import Image
import numpy as np
import math

from pathlib import Path
from typing import List, Optional, Union
import urllib.request


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

_IMAGENET_LABELS_CACHE: Optional[List[str]] = None

IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def get_imagenet_labels(
    labels_path: Union[str, Path] = "imagenet_classes.txt",
    download_if_missing: bool = True,
    url: str = IMAGENET_LABELS_URL,
    use_cache: bool = True,
) -> List[str]:
    """
    Returns a list of 1000 ImageNet class names in index order [0..999].

    If labels_path doesn't exist and download_if_missing=True, downloads it from `url`.
    """
    global _IMAGENET_LABELS_CACHE

    if use_cache and _IMAGENET_LABELS_CACHE is not None:
        return _IMAGENET_LABELS_CACHE

    labels_path = Path(labels_path)

    if not labels_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"ImageNet labels file not found: {labels_path}")
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[imagenet] Downloading labels -> {labels_path}")
        urllib.request.urlretrieve(url, str(labels_path))

    labels = labels_path.read_text(encoding="utf-8").splitlines()
    labels = [l.strip() for l in labels if l.strip()]

    if len(labels) != 1000:
        print(f"[imagenet] WARNING: expected 1000 labels, got {len(labels)} from {labels_path}")

    if use_cache:
        _IMAGENET_LABELS_CACHE = labels

    return labels


def imagenet_class_name(
    class_id: int,
    labels_path: Union[str, Path] = "imagenet_classes.txt",
    download_if_missing: bool = True,
) -> str:
    labels = get_imagenet_labels(labels_path=labels_path, download_if_missing=download_if_missing)
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"class_{class_id}"


def imagenet_prompt_text(
    class_id: int,
    labels_path: Union[str, Path] = "imagenet_classes.txt",
    download_if_missing: bool = True,
) -> str:
    name = imagenet_class_name(class_id, labels_path=labels_path, download_if_missing=download_if_missing)
    return f"a photo of {name}"

def is_all_classes(allowed_classes) -> bool:
    return allowed_classes is None or (isinstance(allowed_classes, str) and allowed_classes.lower() == "all")

def pick_device(device: str) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_perf_knobs(
    device: str,
    use_tf32: bool,
    matmul_precision: str,
    deterministic: bool,
    cudnn_benchmark: bool,
):
    torch.set_float32_matmul_precision(matmul_precision)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_tf32)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark) and (not bool(deterministic))



def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def paper_prompt_normalized_scores(S: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Paper Section 6.3 scoring:
      S: [n, m] = pop_size x (#unique prompts/classes this step)
      mu_q: [m] per-prompt mean across population
      sigma_bar: scalar GLOBAL std over all centered entries
      score_i = mean_j ( (S_ij - mu_qj) / sigma_bar )

    Returns:
      scores: [n]
      mu_q:   [m]
      sigma_bar: scalar
    """
    if S.ndim != 2:
        raise ValueError(f"S must be [n, m], got {tuple(S.shape)}")
    mu_q = S.mean(dim=0)                  # [m]
    centered = S - mu_q[None, :]          # [n, m]
    sigma_bar = torch.sqrt((centered ** 2).mean()).clamp_min(eps)  # scalar
    Z = centered / sigma_bar              # [n, m]
    scores = Z.mean(dim=1)                # [n]
    return scores, mu_q, sigma_bar


def cap_theta_norm(theta: torch.Tensor, theta_max_norm: float) -> torch.Tensor:
    if theta_max_norm is None or theta_max_norm <= 0:
        return theta
    n = theta.norm()
    if n > theta_max_norm:
        theta = theta * (theta_max_norm / (n + 1e-8))
    return theta


def cap_step_norm(theta_before: torch.Tensor, theta_after: torch.Tensor, max_step_norm: float) -> torch.Tensor:
    if max_step_norm is None or max_step_norm <= 0:
        return theta_after
    d = theta_after - theta_before
    dn = d.norm()
    if dn > max_step_norm:
        theta_after = theta_before + d * (max_step_norm / (dn + 1e-8))
    return theta_after


def subsample_for_hist(x: torch.Tensor, max_hist_params: int = 50_000) -> torch.Tensor:
    x = x.detach().cpu().view(-1)
    if x.numel() <= max_hist_params:
        return x
    idx = torch.randperm(x.numel())[:max_hist_params]
    return x[idx]


# ============================================================
# Prompt/class sampling (VAR-like semantics)
# ============================================================

def sample_indices_unique(seed: int, total: int, k: int) -> List[int]:
    if total <= 0:
        raise ValueError("total must be >= 1")
    if k <= 0:
        raise ValueError("k must be >= 1")
    rng = np.random.RandomState(int(seed))
    if k >= total:
        return list(range(total))
    idx = rng.choice(np.arange(total, dtype=np.int64), size=k, replace=False)
    return idx.tolist()


def repeat_batches(ids_unique: List[int], repeats: int) -> List[int]:
    if repeats <= 0:
        raise ValueError("repeats must be >= 1")
    return [i for _ in range(repeats) for i in ids_unique]


def parse_int_list(s: str) -> Union[str, List[int]]:
    s = (s or "").strip()
    if s.lower() == "all" or s == "":
        return "all"
    parts = [x.strip() for x in s.split(",") if x.strip() != ""]
    out = []
    for x in parts:
        out.append(int(x))
    return out
