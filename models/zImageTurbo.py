# models/zImageTurbo.py

from __future__ import annotations

import gc
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
from diffusers import ZImagePipeline

from models.baseEGG import ESBaseModel


class ZImageTurboES(ESBaseModel):
    """
    ES-compatible wrapper for Tongyi-MAI/Z-Image-Turbo via diffusers.ZImagePipeline.

    Notes:
    - ZImagePipeline.encode_prompt() returns a *list* of embeddings (one tensor per prompt, variable length)
    - Turbo typically runs with guidance_scale=0.0
    - Batched generation (micro_batch>1) can crash on some installs due to SDPA attn_mask broadcasting.
      We handle this by trying micro_batch and falling back to 1 if needed.
    """

    def __init__(
        self,
        model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.bfloat16,
        sigma_data: float = 0.5,  # kept for API symmetry (not used here)
        num_inference_steps: int = 9,
        compile_transformer: bool = False,
        attention_backend: Optional[str] = None,  # e.g. "flash" or "_flash_3"
        low_cpu_mem_usage: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            DTYPE=DTYPE,
            sigma_data=sigma_data,
        )

        print(f"[ZImageTurboES] Loading ZImagePipeline: {model_name}")
        self.pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device)

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

        self.transformer = self.pipe.transformer

        if attention_backend is not None:
            # Example: "flash" or "_flash_3"
            try:
                self.transformer.set_attention_backend(attention_backend)
                print(f"[ZImageTurboES] attention_backend={attention_backend}")
            except Exception as e:
                print(f"[ZImageTurboES] Warning: couldn't set attention backend ({attention_backend}): {e}")

        if compile_transformer and hasattr(self.transformer, "compile"):
            print("[ZImageTurboES] Compiling transformer (first run will be slower)...")
            self.transformer.compile()

        self.num_inference_steps = int(num_inference_steps)

        self.vae = getattr(self.pipe, "vae", None)
        self.tokenizer = getattr(self.pipe, "tokenizer", None)
        self.text_encoder = getattr(self.pipe, "text_encoder", None)

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _load_prompts_from_txt(path: Path) -> List[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    def _cuda_empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _check_hw_divisible(self, height_px: int, width_px: int):
        # Pipeline requires divisible by vae_scale_factor*2
        vae_scale = getattr(self.pipe, "vae_scale_factor", None)
        if vae_scale is None:
            return
        div = int(vae_scale * 2)
        if height_px % div != 0 or width_px % div != 0:
            raise ValueError(
                f"height/width must be divisible by {div}. "
                f"Got height={height_px}, width={width_px}."
            )

    # -------------------------
    # Drop text encoder (after encodings exist)
    # -------------------------
    def drop_text_encoder(self):
        """
        Free VRAM by removing the text encoder from the pipeline.
        Safe ONLY if you will use prompt_embeds (prompt=None) going forward.
        """
        try:
            if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
                try:
                    self.pipe.text_encoder.to("cpu")
                except Exception:
                    pass
                self.pipe.text_encoder = None
            self.text_encoder = None
        finally:
            gc.collect()
            self._cuda_empty_cache()
        print("[ZImageTurboES] Dropped text encoder (prompt_embeds-only mode).")

    # -------------------------
    # Prompt encoding (saves a list of tensors)
    # -------------------------
    @torch.no_grad()
    def encode_prompts(
        self,
        prompts_txt_path: str | Path,
        encoded_save_path: str | Path,
        batch_size: int = 64,
        max_sequence_length: int = 512,
        overwrite: bool = False,
        drop_text_encoder_after: bool = True,
    ) -> Dict[str, Any]:
        """
        Encodes prompts using ZImagePipeline.encode_prompt().

        Saved dict keys:
          "prompts"
          "prompt_embeds"  # List[Tensor], len=N, each Tensor is [seq_i, hidden] on CPU
        """
        prompts_txt_path = Path(prompts_txt_path)
        encoded_save_path = Path(encoded_save_path)

        if encoded_save_path.is_file() and not overwrite:
            print(f"[encode] Found existing encoded file at {encoded_save_path}, loading...")
            data = torch.load(encoded_save_path, map_location="cpu")
            if drop_text_encoder_after:
                self.drop_text_encoder()
            return data

        prompts = self._load_prompts_from_txt(prompts_txt_path)
        print(f"[encode] Loaded {len(prompts)} prompts from {prompts_txt_path}")

        all_embeds: List[torch.Tensor] = []

        # encode_prompt accepts list[str] and returns List[Tensor] (variable lengths)
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            print(f"[encode] Batch {start}:{end} / {len(prompts)}")

            prompt_embeds_list, _neg = self.pipe.encode_prompt(
                prompt=batch_prompts,
                device=torch.device(self.device),
                do_classifier_free_guidance=False,  # turbo: no CFG
                negative_prompt=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=max_sequence_length,
            )

            for e in prompt_embeds_list:
                all_embeds.append(e.detach().cpu())

            del prompt_embeds_list, _neg
            self._cuda_empty_cache()

        to_save = {"prompts": prompts, "prompt_embeds": all_embeds}

        encoded_save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[encode] Saving encoded prompts to: {encoded_save_path}")
        torch.save(to_save, encoded_save_path)

        # now that encodings are safely on disk, we can drop the encoder
        if drop_text_encoder_after:
            self.drop_text_encoder()

        return to_save

    # -------------------------
    # Generation (ES API)
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: Union[List[torch.Tensor], torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor] = None,  # unused (kept for ES compatibility)
        latents: Optional[torch.Tensor] = None,
        seed: int = 0,
        guidance_scale: float = 0.0,
        width_latent: int = 32,       # unused
        height_latent: int = 32,      # unused
        width_px: int = 1024,
        height_px: int = 1024,
        num_inference_steps: Optional[int] = None,
    ):
        return self.generate_one_batch(
            prompt_embeds=prompt_embeds,
            seed=seed,
            guidance_scale=guidance_scale,
            width_px=width_px,
            height_px=height_px,
            num_inference_steps=num_inference_steps,
        )

    @torch.no_grad()
    def generate_one_batch(
        self,
        prompt_embeds: List[torch.Tensor],   # List[Tensor], each [T_i, D]
        seed: int = 0,
        guidance_scale: float = 0.0,
        width_px: int = 384,
        height_px: int = 384,
        num_inference_steps: int | None = None,
        micro_batch: int = 1,
        max_sequence_length: int = 512,
    ):
        """
        Generates images for a list of prompt embeddings.

        - Tries micro_batch>1 for speed.
        - If your install hits the SDPA broadcast bug, it auto-falls back to micro_batch=1.
        - Uses per-prompt generators so chunking doesn't change determinism.
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps

        self._check_hw_divisible(height_px, width_px)

        if micro_batch is None or micro_batch <= 0:
            micro_batch = 1

        device = self.device
        images: List[Any] = []

        def _run_chunk(chunk_embeds: List[torch.Tensor], global_start_idx: int):
            # determinism per prompt index
            gens = [
                torch.Generator(device=device).manual_seed(int(seed) + int(global_start_idx) + j)
                for j in range(len(chunk_embeds))
            ]

            out = self.pipe(
                prompt=None,
                prompt_embeds=chunk_embeds,
                height=height_px,
                width=width_px,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=gens,                 # list[Generator] length == batch
                output_type="pil",
                return_dict=True,
                max_sequence_length=int(max_sequence_length),
            )
            return out.images

        # Try requested micro_batch; fallback to 1 if SDPA mask error happens
        try_micro = micro_batch
        while True:
            try:
                images = []
                for i in range(0, len(prompt_embeds), try_micro):
                    chunk = prompt_embeds[i : i + try_micro]
                    images.extend(_run_chunk(chunk, global_start_idx=i))
                return images, None

            except RuntimeError as e:
                msg = str(e)
                # This is the exact family of error you posted.
                sdpa_mask_broadcast = ("scaled_dot_product_attention" in msg) and ("expanded size of the tensor" in msg)

                if try_micro > 1 and sdpa_mask_broadcast:
                    print(
                        f"[ZImageTurboES] Warning: SDPA mask broadcast crash with micro_batch={try_micro}. "
                        f"Falling back to micro_batch=1."
                    )
                    self._cuda_empty_cache()
                    try_micro = 1
                    continue

                # otherwise it's a real error
                raise