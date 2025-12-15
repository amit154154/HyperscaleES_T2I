# models/VAROneStep.py
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from models.baseEGG import ESBaseModel
from VAR_models import build_vae_var


# ---------------------------------------------------------------------
# Speed trick: disable default parameter init (we always load ckpt)
# ---------------------------------------------------------------------
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


ClassIds1D = Union[int, Iterable[int], torch.Tensor]
ClassIds2D = Union[Sequence[Sequence[int]], torch.Tensor]  # 2D tensor OR list-of-lists
ClassIdsAny = Union[ClassIds1D, ClassIds2D]


class VARClassGenerator(ESBaseModel):
    """
    ES-compatible wrapper for VAR (classifier-conditional).

    Key performance feature:
      - generate() supports batched class_ids (2D), so you can do ONE model call for many batches.

    Behavior:
      - if class_ids is 1D -> returns list[PIL.Image] length B
      - if class_ids is 2D and return_grouped=True -> returns list[list[PIL.Image]] shape [num_batches][batch_size]
      - else -> returns flat list[PIL.Image] length (num_batches * batch_size)
    """

    def __init__(
        self,
        model_depth: int = 16,  # must be in {16,20,24,30}
        device: Optional[str] = None,
        DTYPE: torch.dtype = torch.float16,
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes: int = 1000,
        ckpt_dir: Union[str, Path] = ".",
        download_if_missing: bool = True,
        # perf / determinism knobs
        tf32: bool = True,
        deterministic: bool = True,
    ):
        if model_depth not in {16, 20, 24, 30}:
            raise ValueError("model_depth must be one of {16, 20, 24, 30}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            model_name=f"FoundationVision/var (depth={model_depth})",
            device=device,
            DTYPE=DTYPE,
            sigma_data=0.5,  # not used for VAR
        )

        self.num_classes = int(num_classes)
        self.model_depth = int(model_depth)

        # cache for 1D python tuples -> tensor on device
        self._label_cache_1d: Dict[Tuple[int, ...], torch.LongTensor] = {}

        # -------------------------------------------------------------
        # TF32 / matmul precision
        # -------------------------------------------------------------
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
            torch.backends.cudnn.allow_tf32 = bool(tf32)
            torch.set_float32_matmul_precision("high" if tf32 else "highest")

            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = not bool(deterministic)

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt = ckpt_dir / "vae_ch160v4096z32.pth"
        var_ckpt = ckpt_dir / f"var_d{model_depth}.pth"

        if download_if_missing:
            if not vae_ckpt.exists():
                os.system(f"wget {hf_home}/{vae_ckpt.name} -O {str(vae_ckpt)}")
            if not var_ckpt.exists():
                os.system(f"wget {hf_home}/{var_ckpt.name} -O {str(var_ckpt)}")

        if not vae_ckpt.exists():
            raise FileNotFoundError(f"Missing VAE checkpoint: {vae_ckpt}")
        if not var_ckpt.exists():
            raise FileNotFoundError(f"Missing VAR checkpoint: {var_ckpt}")

        # Build models (official hyperparams)
        vae, var = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device=device,
            patch_nums=patch_nums,
            num_classes=num_classes,
            depth=model_depth,
            shared_aln=False,
        )

        # Load checkpoints
        vae.load_state_dict(torch.load(str(vae_ckpt), map_location="cpu"), strict=True)
        var.load_state_dict(torch.load(str(var_ckpt), map_location="cpu"), strict=True)

        # Inference defaults
        vae.eval()
        var.eval()

        for p in vae.parameters():
            p.requires_grad_(False)
        for p in var.parameters():
            p.requires_grad_(False)

        self.vae = vae
        self.var = var

        # ES / LoRA attaches here (and PEFT injects into modules in-place)
        self.transformer = self.var

        print(
            f"[VARClassGenerator] Ready. depth={model_depth}, device={device}, "
            f"DTYPE={DTYPE}, tf32={tf32}, deterministic={deterministic}"
        )

    # -------------------------
    # Optional: compile helpers
    # -------------------------
    def compile_models(
        self,
        compile_var: bool = True,
        compile_vae: bool = False,
        *,
        mode: str = "max-autotune",
        fullgraph: bool = False,
    ):
        """
        IMPORTANT:
          If you use LoRA (PEFT), call this AFTER you attach LoRA,
          because compilation before LoRA can be invalidated.
        """
        if not hasattr(torch, "compile"):
            print("[compile] torch.compile not available in this torch build.")
            return

        if not self.device.startswith("cuda"):
            print("[compile] Skipping compile (not on CUDA).")
            return

        try:
            if compile_var:
                self.var = torch.compile(self.var, mode=mode, fullgraph=fullgraph)
                self.transformer = self.var
            if compile_vae:
                self.vae = torch.compile(self.vae, mode=mode, fullgraph=fullgraph)
            print(f"[compile] SUCCESS (mode={mode}, fullgraph={fullgraph}).")
        except Exception as e:
            print(f"[compile] FAILED: {e} -> continuing in eager mode.")

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
        # tensor path
        if isinstance(class_ids, torch.Tensor):
            if class_ids.ndim == 1:
                return self._to_label_tensor_1d_cached(class_ids), None
            if class_ids.ndim == 2:
                nb, bs = class_ids.shape
                flat = class_ids.reshape(-1)
                return self._to_label_tensor_1d_cached(flat), (int(nb), int(bs))
            raise ValueError("class_ids tensor must be 1D or 2D")

        # python list-of-lists path
        if not isinstance(class_ids, int):
            # detect nested (2D)
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

        # fallback: 1D
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
    # ESBaseModel API
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor = None,          # UNUSED
        prompt_attention_mask: torch.Tensor = None,  # UNUSED
        latents: torch.Tensor = None,                # UNUSED
        seed: int = 0,
        guidance_scale: float = 4.0,                 # cfg
        width_latent: int = 32,                      # UNUSED
        height_latent: int = 32,                     # UNUSED
        *,
        class_ids: ClassIdsAny,
        top_k: int = 900,
        top_p: float = 0.95,
        more_smooth: bool = False,
        use_autocast: bool = True,
        autocast_cache: bool = True,
        return_grouped: bool = False,
    ):
        """
        Efficient generation.

        - class_ids 1D: [B] -> returns images length B
        - class_ids 2D: [num_batches, batch_size] -> ONE model call with B_total
          - return_grouped=True -> returns list[list[PIL]] grouped by batch
        """
        device = self.device
        label_flat, shape = self._flatten_class_ids(class_ids)
        B_total = int(label_flat.shape[0])

        # seed torch + numpy + python like the official snippet
        self._seed_all(seed)

        with torch.inference_mode():
            if use_autocast and device.startswith("cuda"):
                with torch.autocast(
                    device_type="cuda",
                    enabled=True,
                    dtype=self.DTYPE,
                    cache_enabled=bool(autocast_cache),
                ):
                    recon = self.var.autoregressive_infer_cfg(
                        B=B_total,
                        label_B=label_flat,
                        cfg=float(guidance_scale),
                        top_k=int(top_k),
                        top_p=float(top_p),
                        g_seed=int(seed),
                        more_smooth=bool(more_smooth),
                    )
            else:
                recon = self.var.autoregressive_infer_cfg(
                    B=B_total,
                    label_B=label_flat,
                    cfg=float(guidance_scale),
                    top_k=int(top_k),
                    top_p=float(top_p),
                    g_seed=int(seed),
                    more_smooth=bool(more_smooth),
                )

        images = self._tensor_to_pil_list(recon)

        if return_grouped:
            if shape is None:
                raise ValueError("return_grouped=True requires 2D class_ids input")
            nb, bs = shape
            grouped = [images[i * bs:(i + 1) * bs] for i in range(nb)]
            return grouped, None

        return images, None

    @torch.no_grad()
    def encode_prompts(self, *args, **kwargs):
        raise NotImplementedError("VARClassGenerator is class-conditional; no text prompt encoding.")