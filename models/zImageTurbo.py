# models/zImageTurbo.py
from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import torch
from diffusers import ZImagePipeline

from models.baseEGG import ESBaseModel


class ZImageTurboES(ESBaseModel):
    """
    ES-compatible wrapper for Tongyi-MAI/Z-Image-Turbo via diffusers.ZImagePipeline.

    Key goal:
    - When using GGUF, load EXACTLY like your working snippet:

        transformer = ZImageTransformer2DModel.from_single_file(
            local_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
        )

        pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            transformer=transformer,
            dtype=torch.bfloat16,
        ).to("cuda")

        pipeline.transformer.set_attention_backend("flash")
    """

    def __init__(
        self,
        model_name: str = "Tongyi-MAI/Z-Image-Turbo",
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.bfloat16,
        sigma_data: float = 0.5,  # kept for API symmetry (not used here)
        num_inference_steps: int = 9,
        compile_transformer: bool = False,
        attention_backend: Optional[str] = None,  # e.g. "flash" or "_flash_3" or "_sage_qk_int8_pv_fp16_triton"
        low_cpu_mem_usage: bool = False,
        # -------------------------
        # GGUF quantized transformer support
        # -------------------------
        use_quantize: bool = False,
        gguf_repo_id: str = "jayn7/Z-Image-Turbo-GGUF",
        gguf_filename: str = "z_image_turbo-Q4_K_S.gguf",
        gguf_local_dir: str | Path = "gguf_cache",
        gguf_compute_dtype: torch.dtype = torch.bfloat16,
        gguf_local_path: str | Path | None = None,  # ✅ NEW: exact local path (matches your snippet)
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            DTYPE=DTYPE,
            sigma_data=sigma_data,
        )

        # Z-Image Turbo expects bf16 (your build asserts internally)
        if DTYPE != torch.bfloat16:
            print(f"[ZImageTurboES] Warning: forcing DTYPE to bfloat16 (got {DTYPE}).")
            DTYPE = torch.bfloat16
            self.DTYPE = DTYPE

        self.use_quantize = bool(use_quantize)
        self.gguf_repo_id = gguf_repo_id
        self.gguf_filename = gguf_filename
        self.gguf_local_dir = Path(gguf_local_dir)
        self.gguf_local_path = gguf_local_path

        if self.use_quantize:
            # ✅ This will load EXACTLY like the snippet (local_path wins if provided)
            print("[ZImageTurboES] Loading QUANTIZED GGUF transformer...")
            if gguf_local_path is not None:
                print(f"[ZImageTurboES] GGUF local_path: {gguf_local_path}")
            else:
                print(f"[ZImageTurboES] GGUF repo: {gguf_repo_id}/{gguf_filename}")

            self.pipe = self._load_quantized_pipe_like_snippet(
                base_model_id=model_name,
                device=device,
                dtype=DTYPE,
                low_cpu_mem_usage=low_cpu_mem_usage,
                gguf_repo_id=gguf_repo_id,
                gguf_filename=gguf_filename,
                gguf_local_dir=self.gguf_local_dir,
                gguf_compute_dtype=gguf_compute_dtype,
                gguf_local_path=gguf_local_path,
            )
        else:
            print(f"[ZImageTurboES] Loading regular ZImagePipeline: {model_name}")
            self.pipe = ZImagePipeline.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                **self._dtype_kwargs_for_pipe(DTYPE),
            ).to(device)

        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

        self.transformer = self.pipe.transformer

        # Set attention backend (exactly like your snippet)
        if attention_backend is not None:
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
    # EXACT snippet-compatible GGUF loader
    # -------------------------
    @staticmethod
    def _dtype_kwargs_for_pipe(dtype: torch.dtype) -> dict:
        """
        Some diffusers builds use `dtype=...`, others use `torch_dtype=...`.
        Your GGUF examples use `dtype=...`, so we prefer that when available.
        """
        sig = inspect.signature(ZImagePipeline.from_pretrained)
        if "dtype" in sig.parameters:
            return {"dtype": dtype}
        return {"torch_dtype": dtype}

    @staticmethod
    def _load_quantized_pipe_like_snippet(
        base_model_id: str,
        device: str,
        dtype: torch.dtype,
        low_cpu_mem_usage: bool,
        gguf_repo_id: str,
        gguf_filename: str,
        gguf_local_dir: Path,
        gguf_compute_dtype: torch.dtype,
        gguf_local_path: str | Path | None,
    ) -> ZImagePipeline:
        """
        Loads EXACTLY like:

            transformer = ZImageTransformer2DModel.from_single_file(
                local_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                dtype=torch.bfloat16,
            )

            pipeline = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                transformer=transformer,
                dtype=torch.bfloat16,
            ).to("cuda")
        """
        from diffusers import ZImageTransformer2DModel, GGUFQuantizationConfig

        # Resolve GGUF path (local path wins)
        if gguf_local_path is not None:
            gguf_path = str(Path(gguf_local_path))
        else:
            from huggingface_hub import hf_hub_download
            gguf_local_dir.mkdir(parents=True, exist_ok=True)
            gguf_path = hf_hub_download(
                repo_id=gguf_repo_id,
                filename=gguf_filename,
                local_dir=str(gguf_local_dir),
                local_dir_use_symlinks=False,
            )

        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=gguf_compute_dtype),
            dtype=dtype,
        )

        pipe = ZImagePipeline.from_pretrained(
            base_model_id,
            transformer=transformer,
            low_cpu_mem_usage=low_cpu_mem_usage,
            **ZImageTurboES._dtype_kwargs_for_pipe(dtype),
        ).to(device)

        # Sanity: Z-Image Turbo expects bf16
        assert pipe.dtype == torch.bfloat16, f"pipe.dtype={pipe.dtype}, expected torch.bfloat16"
        return pipe

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

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            print(f"[encode] Batch {start}:{end} / {len(prompts)}")

            prompt_embeds_list, _neg = self.pipe.encode_prompt(
                prompt=batch_prompts,
                device=torch.device(self.device),
                do_classifier_free_guidance=False,
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
        prompt_attention_mask: Optional[torch.Tensor] = None,  # unused
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
                generator=gens,
                output_type="pil",
                return_dict=True,
                max_sequence_length=int(max_sequence_length),
            )
            return out.images

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
                sdpa_mask_broadcast = ("scaled_dot_product_attention" in msg) and ("expanded size of the tensor" in msg)
                if try_micro > 1 and sdpa_mask_broadcast:
                    print(
                        f"[ZImageTurboES] Warning: SDPA mask broadcast crash with micro_batch={try_micro}. "
                        f"Falling back to micro_batch=1."
                    )
                    self._cuda_empty_cache()
                    try_micro = 1
                    continue
                raise