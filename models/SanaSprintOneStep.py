import torch
from diffusers import SanaTransformer2DModel, AutoencoderDC
from diffusers.image_processor import PixArtImageProcessor
import lovely_tensors as lt

lt.monkey_patch()


class SanaTransformerOneStepES:
    """
    Minimal wrapper around Sana Sprint transformer + VAE
    with a single trigflow-like step.

    DTYPE controls the working dtype for latents / prompts.
    The transformer & VAE are loaded in that dtype as well, but you can
    switch to float32 by passing DTYPE=torch.float32 if stability issues appear.
    """

    def __init__(
        self,
        model_name: str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.float16,
        sigma_data: float = 0.5,
    ):
        self.device = device
        self.DTYPE = DTYPE          # dtype for latents / prompts
        self.model_name = model_name
        self.sigma_data = sigma_data

        # -------------------------
        # Transformer (LoRA target)
        # -------------------------
        # Load in the requested dtype (can be fp16/bf16/fp32).
        self.transformer = SanaTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=DTYPE,
        ).to(device)

        # Save config & dtype (still valid after PEFT wrapping)
        self.transformer_config = self.transformer.config
        self.transformer_dtype = next(self.transformer.parameters()).dtype

        # -------------------------
        # VAE (decoder)
        # -------------------------
        self.vae = AutoencoderDC.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        ).to(device)

        self.vae_dtype = next(self.vae.parameters()).dtype

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

    @torch.no_grad()
    def sana_one_step_trigflow(
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

        Uses self.DTYPE / model dtypes instead of forcing float32.
        Still sanitizes NaNs/Infs so ES updates can't permanently break generation.
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
        latent_model_input = latents / self.sigma_data  # [B, C, H, W], self.DTYPE

        # t ~ pi/2  (max timestep)
        t = torch.tensor(1.571, device=self.device, dtype=torch.float32)
        timestep = t.expand(b)  # shape [B]

        # SCM timestep & norm
        scm_timestep = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_timestep_expanded = scm_timestep.view(-1, 1, 1, 1)  # [B,1,1,1]

        # Guidance scalar
        guidance = torch.full(
            (b,),
            guidance_scale,
            device=self.device,
            dtype=self.DTYPE,
        )
        guidance = guidance * self.transformer_config.guidance_embeds_scale

        # ---------- Transformer forward (eps prediction) ----------
        # Use whatever dtype the transformer is actually in.
        tr_dtype = self.transformer_dtype

        latent_in = latent_model_input.to(tr_dtype)
        prompt_in = prompt_embeds.to(tr_dtype)
        guidance_in = guidance.to(tr_dtype)
        timestep_in = scm_timestep.to(tr_dtype)

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
        scm_timestep_expanded_tr = scm_timestep_expanded.to(tr_dtype)
        latent_model_input_tr = latent_model_input.to(tr_dtype)

        noise_pred = (
            (1 - 2 * scm_timestep_expanded_tr) * latent_model_input_tr
            + (1 - 2 * scm_timestep_expanded_tr + 2 * scm_timestep_expanded_tr**2)
            * noise_pred_eps
        ) / torch.sqrt(
            scm_timestep_expanded_tr**2 + (1 - scm_timestep_expanded_tr) ** 2
        )

        # Back to the working latent dtype
        noise_pred = noise_pred.to(self.DTYPE) * self.sigma_data

        # --- "Scheduler one step" hack ---
        alpha_t = 0.267
        sigma_t = 0.964

        # Ensure latents + noise_pred are same dtype
        latents = latents.to(self.DTYPE)
        pred_x0 = alpha_t * latents - sigma_t * noise_pred
        pred_x0 = pred_x0 / self.sigma_data  # back to VAE latent scale

        # --------- Decode with VAE ----------
        pred_x0_vae = pred_x0.to(self.vae_dtype) / self.vae.config.scaling_factor
        image_tensor = self.vae.decode(
            pred_x0_vae,
            return_dict=False,
        )[0]

        images = self.image_processor.postprocess(image_tensor, output_type="pil")
        # Return images and the VAE-scaled latent (same as in original code)
        return images, pred_x0_vae