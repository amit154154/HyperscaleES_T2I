# models/SanaSprintOneStep.py

import torch
from pathlib import Path

import lovely_tensors as lt

class ESBaseModel:
    """
    Base class for ES-compatible image generators.

    Requirements:
      - must define self.transformer (trainable part for LoRA/ES)
      - must implement `generate(...)` with the same signature as below
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.float32,
        sigma_data: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device
        self.DTYPE = DTYPE
        self.sigma_data = sigma_data

        # Subclasses must set:
        #   self.transformer
        #   any extra modules (e.g., VAE, image_processor)
        self.transformer = None

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
        Abstract interface for ES:

        Args are intentionally identical to SanaOneStep.sana_one_step_trigflow.
        Must return:
            images: list of PIL.Image
            latents_out: torch.Tensor (whatever latent representation you want)
        """
        raise NotImplementedError("Subclasses must implement generate()")

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
        Abstract interface for encoding prompts.

        Subclasses should:
          - use self.model_name + their own text encoder
          - read raw prompts from `prompts_txt_path` (1 per line)
          - save a dict to `encoded_save_path` with at least:
              { "prompts", "prompt_embeds", "prompt_attention_mask" }

        This base implementation just defines the signature.
        """
        raise NotImplementedError("Subclasses must implement encode_prompts()")
