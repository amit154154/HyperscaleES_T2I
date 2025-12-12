# models/SanaSprintOneStep.py

import torch
from diffusers import SanaTransformer2DModel, AutoencoderDC,SanaSprintPipeline
from diffusers.image_processor import PixArtImageProcessor
from models.baseEGG import ESBaseModel
from pathlib import Path

class SanaOneStep(ESBaseModel):
    """
    Sana Sprint one-step trigflow wrapper, ES-compatible.

    - Uses a SanaTransformer2DModel as `self.transformer`
    - Uses AutoencoderDC + PixArtImageProcessor for decoding
    - Implements `generate(...)` with the same API as the old
      `sana_one_step_trigflow`.
    """

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.float32,
        sigma_data: float = 0.5,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            DTYPE=DTYPE,
            sigma_data=sigma_data,
        )

        # Use full float32 for transformer for now (much more stable)
        self.transformer: SanaTransformer2DModel = SanaTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=torch.float32,
        ).to(device)

        # Save config (valid after PEFT wrapping)
        self.transformer_config = self.transformer.config

        # VAE decode in float32
        self.vae = AutoencoderDC.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        latents: torch.Tensor = None,
        seed: int = 0,
        guidance_scale: float = 1.0,
        width_latent: int = 32,
        height_latent: int = 32,
    ):
        """
        One-step trigflow / SCM + DDIM-like step, then VAE decode.

        This is the same math as your previous `sana_one_step_trigflow`,
        just renamed to `generate(...)` to match the ES base interface.
        """

        b = prompt_embeds.shape[0]
        latent_channels = self.transformer_config.in_channels

        # --------- Latents init (optional) ----------
        if latents is None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            latents = torch.randn(
                b,
                latent_channels,
                height_latent,
                width_latent,
                device=self.device,
                dtype=self.DTYPE,
                generator=g,
            )
            latents = latents * self.sigma_data  # N(0, sigma_data^2)

        # Normalize by sigma_data (like the pipeline)
        latent_model_input = latents / self.sigma_data

        # t ~ pi/2  (max timestep)
        t = torch.tensor(1.571, device=self.device, dtype=torch.float32)
        timestep = t.expand(b)  # shape [b]

        # SCM timestep & norm
        scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)

        # Guidance scalar
        guidance = torch.full(
            (b,),
            guidance_scale,
            device=self.device,
            dtype=self.DTYPE,
        )
        guidance = guidance * self.transformer_config.guidance_embeds_scale

        # ---------- Transformer forward (eps prediction) ----------
        # Run everything in float32 inside the transformer
        latent_in = latent_model_input.to(torch.float32)
        prompt_in = prompt_embeds.to(torch.float32)
        guidance_in = guidance.to(torch.float32)
        timestep_in = scm_timestep.to(torch.float32)

        noise_pred_eps = self.transformer(
            latent_in,
            timestep=timestep_in,
            encoder_hidden_states=prompt_in,
            encoder_attention_mask=prompt_attention_mask,
            guidance=guidance_in,
            return_dict=False,
            attention_kwargs=None,
        )[0]

        # 🔐 NaN / Inf guard – if ES ever blows up, we don't propagate NaNs
        noise_pred_eps = torch.nan_to_num(
            noise_pred_eps, nan=0.0, posinf=0.0, neginf=0.0
        )

        # SCM combination (same structure as in the pipeline)
        noise_pred = (
            (1 - 2 * scm_timestep_expanded) * latent_model_input
            + (1 - 2 * scm_timestep_expanded + 2 * scm_timestep_expanded**2)
            * noise_pred_eps.to(latent_model_input.dtype)
        ) / torch.sqrt(
            scm_timestep_expanded**2 + (1 - scm_timestep_expanded) ** 2
        )

        noise_pred = noise_pred.float() * self.sigma_data

        # --- "Scheduler one step" hack ---
        alpha_t = 0.267
        sigma_t = 0.964

        pred_x0 = alpha_t * latents - sigma_t * noise_pred
        pred_x0 = pred_x0 / self.sigma_data  # back to VAE latent scale
        pred_x0 = pred_x0.to(self.vae.dtype)

        # --------- Decode with VAE ----------
        image_tensor = self.vae.decode(
            pred_x0 / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]

        images = self.image_processor.postprocess(image_tensor, output_type="pil")
        # return images + latents in VAE scale (same as before)
        return images, pred_x0 / self.vae.config.scaling_factor

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts_txt_path: str | Path,
        encoded_save_path: str | Path,
        batch_size: int = 4,
        complex_human_instruction = None,
        overwrite: bool = False,
    ):
        """
        Encode prompts using SanaSprintPipeline's text encoder.

        - If encoded_save_path already exists and overwrite=False:
            -> loads and returns the saved dict.
        - Else:
            -> reads prompts from prompts_txt_path (1 per line),
               encodes them, saves to encoded_save_path, and returns the dict.

        Saved dict keys:
          "prompts"
          "prompt_embeds"          # [N, seq, dim], float16, on CPU
          "prompt_attention_mask"  # [N, seq], int64, on CPU
        """

        prompts_txt_path = Path(prompts_txt_path)
        encoded_save_path = Path(encoded_save_path)

        if encoded_save_path.is_file() and not overwrite:
            print(f"[encode] Found existing encoded file at {encoded_save_path}, loading...")
            return torch.load(encoded_save_path, map_location="cpu")

        print(f"[encode] Reading prompts from {prompts_txt_path} ...")
        prompts = self._load_prompts_from_txt(prompts_txt_path)
        num_prompts = len(prompts)
        print(f"[encode] Loaded {num_prompts} prompts.")
        for i, p in enumerate(prompts):
            print(f"  {i:02d}: {p}")

        # --- build pipeline with text encoder ---
        print(f"[encode] Loading SanaSprintPipeline for model: {self.model_name}")
        pipe = SanaSprintPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        print("[encode] Encoding prompts with text encoder in batches...")
        all_embeds = []
        all_masks = []

        for start in range(0, num_prompts, batch_size):
            end = min(start + batch_size, num_prompts)
            batch_prompts = prompts[start:end]
            print(f"[encode] Batch {start}:{end} / {num_prompts}")

            prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
                batch_prompts,
                num_images_per_prompt=1,
                device=self.device,
                prompt_embeds=None,
                prompt_attention_mask=None,
                clean_caption=False,
                max_sequence_length=300,
                complex_human_instruction=complex_human_instruction,
            )

            # move to CPU immediately to free VRAM
            all_embeds.append(prompt_embeds.cpu())
            all_masks.append(prompt_attention_mask.cpu())

            del prompt_embeds, prompt_attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # concatenate along batch dimension
        prompt_embeds = torch.cat(all_embeds, dim=0)
        prompt_attention_mask = torch.cat(all_masks, dim=0)

        print("[encode] Encoded prompts:")
        print(
            f"  prompt_embeds shape: {prompt_embeds.shape}, "
            f"dtype: {prompt_embeds.dtype}, device: {prompt_embeds.device}"
        )
        print(
            f"  attention_mask shape: {prompt_attention_mask.shape}, "
            f"dtype: {prompt_attention_mask.dtype}, device: {prompt_attention_mask.device}"
        )

        print(f"[encode] Saving encoded prompts to: {encoded_save_path}")
        to_save = {
            "prompts": prompts,
            "prompt_embeds": prompt_embeds,                 # [N, seq, dim]
            "prompt_attention_mask": prompt_attention_mask, # [N, seq]
        }
        torch.save(to_save, encoded_save_path)
        print("[encode] Saved encoded prompts.")

        # free GPU text encoder
        if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
            print("[encode] Moving text encoder to CPU and clearing CUDA cache...")
            pipe.text_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[encode] Text encoder moved to CPU.")
        else:
            print("[encode] No text encoder found on pipeline (or already moved).")

        return to_save
# models/SanaSprint.py

from pathlib import Path
import torch
from diffusers import SanaSprintPipeline

from .es_base import ESBaseModel  # whatever your base class is called


class SanaPipelineES(ESBaseModel):
    """
    ES-compatible wrapper that uses SanaSprintPipeline end-to-end.

    - Uses SanaSprintPipeline for generation (num_inference_steps > 1 if you want)
    - `self.transformer` is pipe.transformer (so LoRA attaches there)
    - `generate(...)` / `generate_one_batch(...)` delegate to the pipeline.
    """

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.float16,
        sigma_data: float = 0.5,   # not really used here, but kept for API symmetry
        num_inference_steps: int = 8,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            DTYPE=DTYPE,
            sigma_data=sigma_data,
        )

        # Full SanaSprintPipeline with text encoder, VAE, transformer, etc.
        print(f"[SanaPipelineES] Loading SanaSprintPipeline: {model_name}")
        self.pipe = SanaSprintPipeline.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        # The part ES/LoRA will modify:
        self.transformer = self.pipe.transformer

        # Cache VAE scale factor from VAE config
        self.vae = self.pipe.vae
        if hasattr(self.vae, "config") and hasattr(self.vae.config, "encoder_block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        else:
            # fallback; Sana Sprint 1024px uses 32
            self.vae_scale_factor = 32

        self.num_inference_steps = num_inference_steps

    # -------------------------
    # Prompt encoding helper
    # -------------------------
    @staticmethod
    def _load_prompts_from_txt(path: Path):
        lines = path.read_text(encoding="utf-8").splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip()]
        return prompts

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts_txt_path: str | Path,
        encoded_save_path: str | Path,
        batch_size: int = 4,
        complex_human_instruction=None,
        overwrite: bool = False,
    ):
        """
        Encode prompts using the *existing* self.pipe text encoder.

        - If encoded_save_path already exists and overwrite=False:
            -> loads and returns the saved dict.
        - Else:
            -> reads prompts from prompts_txt_path (1 per line),
               encodes them, saves to encoded_save_path, and returns the dict.
        """

        prompts_txt_path = Path(prompts_txt_path)
        encoded_save_path = Path(encoded_save_path)

        if encoded_save_path.is_file() and not overwrite:
            print(f"[encode] Found existing encoded file at {encoded_save_path}, loading...")
            return torch.load(encoded_save_path, map_location="cpu")

        print(f"[encode] Reading prompts from {prompts_txt_path} ...")
        prompts = self._load_prompts_from_txt(prompts_txt_path)
        num_prompts = len(prompts)
        print(f"[encode] Loaded {num_prompts} prompts.")
        for i, p in enumerate(prompts):
            print(f"  {i:02d}: {p}")

        print("[encode] Encoding prompts with pipeline text encoder in batches...")
        all_embeds = []
        all_masks = []

        for start in range(0, num_prompts, batch_size):
            end = min(start + batch_size, num_prompts)
            batch_prompts = prompts[start:end]
            print(f"[encode] Batch {start}:{end} / {num_prompts}")

            prompt_embeds, prompt_attention_mask = self.pipe.encode_prompt(
                batch_prompts,
                num_images_per_prompt=1,
                device=self.device,
                prompt_embeds=None,
                prompt_attention_mask=None,
                clean_caption=False,
                max_sequence_length=300,
                complex_human_instruction=complex_human_instruction,
            )

            all_embeds.append(prompt_embeds.cpu())
            all_masks.append(prompt_attention_mask.cpu())

            del prompt_embeds, prompt_attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prompt_embeds = torch.cat(all_embeds, dim=0)
        prompt_attention_mask = torch.cat(all_masks, dim=0)

        print("[encode] Encoded prompts:")
        print(
            f"  prompt_embeds shape: {prompt_embeds.shape}, "
            f"dtype: {prompt_embeds.dtype}, device: {prompt_embeds.device}"
        )
        print(
            f"  attention_mask shape: {prompt_attention_mask.shape}, "
            f"dtype: {prompt_attention_mask.dtype}, device: {prompt_attention_mask.device}"
        )

        print(f"[encode] Saving encoded prompts to: {encoded_save_path}")
        to_save = {
            "prompts": prompts,
            "prompt_embeds": prompt_embeds,                 # [N, seq, dim]
            "prompt_attention_mask": prompt_attention_mask, # [N, seq]
        }
        torch.save(to_save, encoded_save_path)
        print("[encode] Saved encoded prompts.")

        # Optional: drop text encoder after encoding to free VRAM
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            print("[encode] Moving text encoder to CPU and clearing CUDA cache...")
            self.pipe.text_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[encode] Text encoder moved to CPU.")
        else:
            print("[encode] No text encoder found on pipeline (or already moved).")

        return to_save

    # -------------------------
    # Single-batch generation (generic ES API)
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        latents: torch.Tensor = None,  # ignored for now (pipeline manages its own)
        seed: int = 0,
        guidance_scale: float = 1.0,
        width_latent: int = 32,
        height_latent: int = 32,
    ):
        """
        Generic generate used by ESBaseModel, but here it's just a thin wrapper
        around `generate_one_batch` for backward compatibility.
        """
        return self.generate_one_batch(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            seed=seed,
            guidance_scale=guidance_scale,
            width_latent=width_latent,
            height_latent=height_latent,
        )

    # -------------------------
    # NEW: generate_one_batch – run ALL prompts for one indiv in one call
    # -------------------------
    @torch.no_grad()
    def generate_one_batch(
        self,
        prompt_embeds: torch.Tensor,        # [P, seq, dim] or [B, seq, dim]
        prompt_attention_mask: torch.Tensor,# [P, seq]
        seed: int = 0,
        guidance_scale: float = 1.0,
        width_latent: int = 32,
        height_latent: int = 32,
    ):
        """
        Run SanaSprintPipeline once on the *entire* prompt batch.

        Used in ES so that, for a given individual (fixed LoRA weights), we
        evaluate all prompts in one shot.

        The `seed` here is shared across ALL individuals in a given ES step
        (set in runES.py), matching the EGGROLL paper behaviour.
        """
        device = self.device

        prompt_embeds = prompt_embeds.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        # Convert latent grid size to pixel image size
        width_px = 1024
        height_px = 1024

        generator = torch.Generator(device=device).manual_seed(seed)

        out = self.pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=guidance_scale,
            width=width_px,
            height=height_px,
            generator=generator,
            output_type="pil",
            use_resolution_binning=False,
        )

        images = out.images  # list of PIL.Image, len == batch size
        return images, None  # second return kept for ES API compatibility
