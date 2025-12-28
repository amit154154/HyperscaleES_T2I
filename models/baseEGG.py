from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class ESBaseModel:
    """
    Base class for ES-compatible image generators.

    Contract:
      - self.transformer: torch.nn.Module (the trainable / LoRA target module)
      - generate(...): returns (images: list[PIL.Image], latents_out: Any)
      - encode_prompts(...): optional (text models), not required for class-conditional
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.float32,
        sigma_data: float = 0.5,
    ):
        self.model_name = str(model_name)
        self.device = str(device)
        self.DTYPE = DTYPE
        self.sigma_data = float(sigma_data)

        # Subclasses must set this to the module where LoRA / ES updates apply.
        self.transformer: Optional[torch.nn.Module] = None

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        seed: int = 0,
        guidance_scale: float = 1.0,
        width_latent: int = 32,
        height_latent: int = 32,
        **kwargs: Any,
    ):
        """
        Generic ES interface.

        NOTE:
          We keep the familiar signature for text2image pipelines,
          but allow subclasses to accept extra conditioning via **kwargs
          (e.g., class_ids=..., negative_prompt=..., etc).

        Must return:
          images: list[PIL.Image] OR list[list[PIL.Image]] if return_grouped=True
          latents_out: Any (optional)
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts_txt_path: Union[str, Path],
        encoded_save_path: Union[str, Path],
        batch_size: int = 4,
        complex_human_instruction=None,
        overwrite: bool = False,
    ):
        """
        Optional. Only needed for text-conditional models.
        """
        raise NotImplementedError("Subclasses must implement encode_prompts()")

# models/base_class_conditional.py



ClassIds1D = Union[int, Iterable[int], torch.Tensor]
ClassIds2D = Union[Sequence[Sequence[int]], torch.Tensor]
ClassIdsAny = Union[ClassIds1D, ClassIds2D]


class ESClassConditionalModel(ESBaseModel):
    """
    Reusable base for class-conditional generators.

    Provides:
      - class id validation
      - caching for 1D label tensors
      - flattening 1D / 2D class_id inputs
      - seed helper
      - tensor->PIL conversion helper

    Subclasses implement:
      - _infer_images_from_labels(label_flat, seed, guidance_scale, **kwargs) -> Tensor[B,3,H,W]
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        DTYPE: torch.dtype,
        num_classes: int,
        sigma_data: float = 0.5,
        tf32: bool = True,
        deterministic: bool = True,
    ):
        super().__init__(model_name=model_name, device=device, DTYPE=DTYPE, sigma_data=sigma_data)
        self.num_classes = int(num_classes)

        # cache for 1D python tuples -> tensor on device
        self._label_cache_1d: Dict[Tuple[int, ...], torch.LongTensor] = {}

        # TF32 / determinism knobs (generic)
        if str(device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.set_float32_matmul_precision("high" if tf32 else "highest")
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)

    # -------------------------
    # Helpers
    # -------------------------
    def _seed_all(self, seed: int):
        seed = int(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _validate_labels(self, label: torch.LongTensor) -> torch.LongTensor:
        if label.ndim != 1:
            raise ValueError("class_ids must be 1D or 2D (flattened internally to 1D)")
        if (label < 0).any() or (label >= self.num_classes).any():
            raise ValueError(f"class_ids must be in [0, {self.num_classes - 1}]")
        return label

    def _to_label_tensor_1d_cached(self, ids_1d: Union[int, Iterable[int], torch.Tensor]) -> torch.LongTensor:
        device = self.device

        if isinstance(ids_1d, torch.Tensor):
            label = ids_1d.to(device=device, dtype=torch.long)
            return self._validate_labels(label)

        if isinstance(ids_1d, int):
            label = torch.tensor([int(ids_1d)], device=device, dtype=torch.long)
            return self._validate_labels(label)

        ids_tuple = tuple(int(x) for x in ids_1d)
        cached = self._label_cache_1d.get(ids_tuple, None)
        if cached is None or cached.device.type != torch.device(device).type:
            cached = torch.tensor(list(ids_tuple), device=device, dtype=torch.long)
            self._label_cache_1d[ids_tuple] = cached
        return self._validate_labels(cached)

    def _flatten_class_ids(
        self,
        class_ids: ClassIdsAny,
    ) -> Tuple[torch.LongTensor, Optional[Tuple[int, int]]]:
        """
        Returns:
          label_flat: [B_total]
          shape: (num_batches, batch_size) if input was 2D, else None
        """
        if isinstance(class_ids, torch.Tensor):
            if class_ids.ndim == 1:
                return self._to_label_tensor_1d_cached(class_ids), None
            if class_ids.ndim == 2:
                nb, bs = class_ids.shape
                flat = class_ids.reshape(-1)
                return self._to_label_tensor_1d_cached(flat), (int(nb), int(bs))
            raise ValueError("class_ids tensor must be 1D or 2D")

        if not isinstance(class_ids, int):
            if isinstance(class_ids, (list, tuple)) and len(class_ids) > 0 and isinstance(class_ids[0], (list, tuple)):
                batches = [list(map(int, b)) for b in class_ids]  # type: ignore[arg-type]
                if len(batches) == 0:
                    raise ValueError("empty class_ids batches")
                bs = len(batches[0])
                if bs == 0:
                    raise ValueError("empty inner batch in class_ids")
                if any(len(b) != bs for b in batches):
                    raise ValueError("All inner batches must have the same length")
                flat_list = [x for b in batches for x in b]
                label = self._to_label_tensor_1d_cached(flat_list)
                return label, (len(batches), bs)

        return self._to_label_tensor_1d_cached(class_ids), None

    @staticmethod
    def _tensor_to_pil_list(x_B3HW: torch.Tensor) -> List[Image.Image]:
        x = x_B3HW.detach().cpu()

        # If it looks like [-1,1], map to [0,1]
        if x.numel() > 0 and x.min() < -0.01:
            x = (x + 1.0) * 0.5

        x = x.clamp(0, 1)

        imgs: List[Image.Image] = []
        for i in range(x.shape[0]):
            chw = x[i]
            hwc = (chw.permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
            imgs.append(Image.fromarray(hwc))
        return imgs

    # -------------------------
    # Subclass hook
    # -------------------------
    @torch.no_grad()
    def _infer_images_from_labels(
        self,
        label_flat: torch.LongTensor,
        *,
        seed: int,
        guidance_scale: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Subclasses must implement model inference and return tensor [B,3,H,W] in [0,1] or [-1,1].
        """
        raise NotImplementedError

    # -------------------------
    # ESBaseModel API
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: Optional[torch.Tensor] = None,          # UNUSED for class-conditional
        prompt_attention_mask: Optional[torch.Tensor] = None,  # UNUSED for class-conditional
        latents: Optional[torch.Tensor] = None,                # UNUSED for class-conditional
        seed: int = 0,
        guidance_scale: float = 4.0,
        width_latent: int = 32,   # UNUSED
        height_latent: int = 32,  # UNUSED
        *,
        class_ids: ClassIdsAny,
        return_grouped: bool = False,
        **kwargs: Any,
    ):
        label_flat, shape = self._flatten_class_ids(class_ids)
        B_total = int(label_flat.shape[0])

        self._seed_all(seed)

        recon = self._infer_images_from_labels(
            label_flat=label_flat,
            seed=int(seed),
            guidance_scale=float(guidance_scale),
            **kwargs,
        )
        if recon.ndim != 4 or recon.shape[0] != B_total:
            raise RuntimeError(f"_infer_images_from_labels must return [B,3,H,W], got {tuple(recon.shape)}")

        images = self._tensor_to_pil_list(recon)

        if return_grouped:
            if shape is None:
                raise ValueError("return_grouped=True requires 2D class_ids input")
            nb, bs = shape
            grouped = [images[i * bs : (i + 1) * bs] for i in range(nb)]
            return grouped, None

        return images, None