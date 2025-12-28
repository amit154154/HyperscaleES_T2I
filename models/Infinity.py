# models/Infinity.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast

from models.baseEGG import ESBaseModel

# Infinity repo imports
from Infinity.models.infinity import Infinity
from Infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from Infinity.models.bsq_vae.vae import vae_model


# We store per-prompt "compact" encoding:
#   kv_compact: [Li, C]  (Li = true token length, C=text_channels)
#   li: int
KVCompact = torch.Tensor
PromptCompact = Tuple[KVCompact, int]
class InfinityES(ESBaseModel):
    """
    Infinity ES wrapper:
      - encode_prompts(): encodes prompts once -> saves compact embeddings to .pt (CPU tensors)
      - drop_text_encoder(): optional VRAM cleanup after encoding
      - generate_one_batch_from_compacts(): batches ALL selected prompts in ONE call to Infinity.autoregressive_infer_cfg
    """

    def __init__(
        self,
        *,
        model_path: str,
        text_encoder_ckpt: str,
        vae_path: str,
        vae_type: int,
        pn: str,
        model_type: str = "infinity_2b",
        h_div_w_template: float = 1.0,
        text_channels: int = 2048,
        apply_spatial_patchify: int = 0,
        use_flex_attn: int = 0,
        bf16: bool = True,
        checkpoint_type: str = "torch",  # 'torch' | 'torch_shard'

        # Infinity knobs (defaults; can be overridden at generate time)
        top_k: int = 900,
        top_p: float = 0.97,
        cfg_exp_k: float = 0.0,
        gumbel: int = 0,
        softmax_merge_topk: int = -1,
        cfg_insertion_layer: int = 0,
        sampling_per_bits: int = 1,
        enable_positive_prompt: int = 0,
        gt_leak: int = 0,

        device: str = "cuda:0",
        DTYPE: torch.dtype = torch.bfloat16,
        sigma_data: float = 0.5,  # unused but kept for ESBaseModel symmetry
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        super().__init__(model_name="Infinity", device=device, DTYPE=DTYPE, sigma_data=sigma_data)
        self.device = device
        self.bf16 = bool(bf16)

        # defaults (override per call if needed)
        self.top_k_default = int(top_k)
        self.top_p_default = float(top_p)
        self.cfg_exp_k_default = float(cfg_exp_k)
        self.gumbel_default = int(gumbel)
        self.softmax_merge_topk_default = int(softmax_merge_topk)
        self.cfg_insertion_layer_default = int(cfg_insertion_layer)
        self.sampling_per_bits_default = int(sampling_per_bits)
        self.enable_positive_prompt_default = int(enable_positive_prompt)
        self.gt_leak_default = int(gt_leak)

        # scale schedule
        scales = dynamic_resolution_h_w[h_div_w_template][pn]["scales"]
        self.scale_schedule = [(1, h, w) for (_, h, w) in scales]

        # tokenizer + encoder
        self.text_tokenizer, self.text_encoder = self._load_tokenizer_and_encoder(text_encoder_ckpt)

        # VAE
        self.vae = self._load_visual_tokenizer(vae_type=vae_type, vae_path=vae_path, apply_spatial_patchify=apply_spatial_patchify)

        # Infinity transformer
        model_kwargs = self._kwargs_for_model_type(model_type)
        self.infinity = self._load_infinity(
            model_path=model_path,
            model_kwargs=model_kwargs,
            text_channels=text_channels,
            apply_spatial_patchify=apply_spatial_patchify,
            use_flex_attn=use_flex_attn,
            checkpoint_type=checkpoint_type,
            pn=pn,
        )


        # LoRA/ES target
        self.transformer = self.infinity

        # keep vae_type for generate call convenience
        self.vae_type = int(vae_type)
        dtype = torch.bfloat16 if self.bf16 else torch.float16
        self.vae = self.vae.to(dtype=dtype)
        self.infinity = self.infinity.to(dtype=dtype)
        self.transformer = self.infinity

    # -------------------------
    # Load helpers
    # -------------------------
    def _load_tokenizer_and_encoder(self, t5_path: str) -> Tuple[T5TokenizerFast, T5EncoderModel]:
        print("[InfinityES] Loading tokenizer + text encoder...")
        tok: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
        tok.model_max_length = 512

        enc: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
        enc.to(self.device).eval()
        enc.requires_grad_(False)
        return tok, enc

    def _load_visual_tokenizer(self, *, vae_type: int, vae_path: str, apply_spatial_patchify: int) -> torch.nn.Module:
        if vae_type not in [14, 16, 18, 20, 24, 32, 64]:
            raise ValueError(f"vae_type={vae_type} not supported")

        schedule_mode = "dynamic"
        codebook_dim = int(vae_type)
        codebook_size = 2 ** codebook_dim

        if int(apply_spatial_patchify) == 1:
            patch_size = 8
            encoder_ch_mult = [1, 2, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult = [1, 2, 4, 4, 4]
            decoder_ch_mult = [1, 2, 4, 4, 4]

        print("[InfinityES] Loading VAE...")
        v = vae_model(
            vae_path,
            schedule_mode,
            codebook_dim,
            codebook_size,
            patch_size=patch_size,
            encoder_ch_mult=encoder_ch_mult,
            decoder_ch_mult=decoder_ch_mult,
            test_mode=True,
        ).to(self.device)
        v.eval()
        v.requires_grad_(False)
        return v

    @staticmethod
    def _kwargs_for_model_type(model_type: str) -> Dict[str, Any]:
        if model_type == "infinity_2b":
            return dict(depth=32, embed_dim=2048, num_heads=2048 // 128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
        if model_type == "infinity_8b":
            return dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
        if model_type == "infinity_layer12":
            return dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        if model_type == "infinity_layer16":
            return dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        if model_type == "infinity_layer24":
            return dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        if model_type == "infinity_layer32":
            return dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        if model_type == "infinity_layer40":
            return dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        if model_type == "infinity_layer48":
            return dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
        raise ValueError(f"Unknown model_type={model_type}")

    def _load_infinity(
        self,
        *,
        model_path: str,
        model_kwargs: Dict[str, Any],
        text_channels: int,
        apply_spatial_patchify: int,
        use_flex_attn: int,
        checkpoint_type: str,
        pn: str,
    ) -> Infinity:
        print("[InfinityES] Building Infinity model...")
        text_maxlen = 512

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.bf16, dtype=torch.bfloat16, cache_enabled=True):
                inf: Infinity = Infinity(
                    vae_local=self.vae,
                    text_channels=int(text_channels),
                    text_maxlen=text_maxlen,
                    shared_aln=True,
                    raw_scale_schedule=None,
                    checkpointing="full-block",
                    customized_flash_attn=False,
                    fused_norm=True,
                    pad_to_multiplier=128,
                    use_flex_attn=bool(use_flex_attn),
                    add_lvl_embeding_only_first_block=0,
                    use_bit_label=1,
                    rope2d_each_sa_layer=1,
                    rope2d_normalized_by_hw=2,
                    pn=pn,
                    apply_spatial_patchify=int(apply_spatial_patchify),
                    inference_mode=True,
                    train_h_div_w_list=[1.0],
                    **model_kwargs,
                ).to(self.device)

        inf.eval()
        inf.requires_grad_(False)

        print("[InfinityES] Loading Infinity weights...")
        if checkpoint_type == "torch":
            state_dict = torch.load(model_path, map_location=self.device)
            print(inf.load_state_dict(state_dict))
        elif checkpoint_type == "torch_shard":
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(inf, model_path, strict=False)
        else:
            raise ValueError(f"checkpoint_type must be 'torch' or 'torch_shard', got {checkpoint_type}")
        inf.rng = torch.Generator(device=self.device)
        torch.cuda.empty_cache()
        return inf

    # -------------------------
    # Text encoding (encode once)
    # -------------------------
    @staticmethod
    def _load_prompts_from_txt(path: Path) -> List[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    @staticmethod
    def _aug_with_positive_prompt(prompt: str) -> str:
        for key in [
            "man", "woman", "men", "women", "boy", "girl", "child", "person", "human", "adult", "teenager",
            "employee", "employer", "worker", "mother", "father", "sister", "brother", "grandmother",
            "grandfather", "son", "daughter"
        ]:
            if key in prompt:
                prompt = prompt + ". very smooth faces, good looking faces, face to the camera, perfect facial features"
                break
        return prompt

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts_txt_path: Union[str, Path],
        encoded_save_path: Union[str, Path],
        batch_size: int = 8,
        complex_human_instruction=None,  # unused for Infinity
        overwrite: bool = False,
        store_dtype: str = "float16",  # "float16" | "float32"
    ):
        """
        Encodes prompts ONCE and saves CPU tensors.

        Saved dict:
          {
            "prompts": [str...],
            "kv_compact_list": [Tensor[Li,C] on CPU],
            "lens_list": [int...],
          }

        This is *exactly* what you want for ES: load once, then just index and batch-pack.
        """
        prompts_txt_path = Path(prompts_txt_path)
        encoded_save_path = Path(encoded_save_path)

        if encoded_save_path.is_file() and not overwrite:
            print(f"[InfinityES.encode] Found existing {encoded_save_path}, loading...")
            return torch.load(encoded_save_path, map_location="cpu")

        prompts = self._load_prompts_from_txt(prompts_txt_path)
        print(f"[InfinityES.encode] Loaded {len(prompts)} prompts")

        if int(self.enable_positive_prompt_default) == 1:
            prompts = [self._aug_with_positive_prompt(p) for p in prompts]

        kv_compact_list: List[torch.Tensor] = []
        lens_list: List[int] = []

        out_dtype = torch.float16 if store_dtype == "float16" else torch.float32

        for start in range(0, len(prompts), int(batch_size)):
            end = min(start + int(batch_size), len(prompts))
            batch_prompts = prompts[start:end]
            print(f"[InfinityES.encode] batch {start}:{end}/{len(prompts)}")

            tokens = self.text_tokenizer(
                text=batch_prompts,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device, non_blocking=True)
            mask = tokens.attention_mask.to(self.device, non_blocking=True)

            # encoder in fp16, then we slice true lengths; store compact
            feats = self.text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"]  # [B,512,C]
            feats = feats.to(torch.float32)  # stable slicing like your script

            lens = mask.sum(dim=-1).tolist()  # List[int], len=B
            for i, li in enumerate(lens):
                li = int(li)
                kv_i = feats[i, :li].detach().to(dtype=out_dtype).cpu()  # [Li,C] CPU
                kv_compact_list.append(kv_i)
                lens_list.append(li)

            del input_ids, mask, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "prompts": prompts,
            "kv_compact_list": kv_compact_list,
            "lens_list": lens_list,
        }
        encoded_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, encoded_save_path)
        print(f"[InfinityES.encode] Saved -> {encoded_save_path}")
        return payload

    def drop_text_encoder(self):
        """
        Optional VRAM cleanup after encoding prompts.
        """
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            print("[InfinityES] Dropping text encoder to CPU + freeing GPU...")
            try:
                self.text_encoder.to("cpu")
            except Exception:
                pass
            self.text_encoder = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------
    # Batch packing + inference
    # -------------------------
    @staticmethod
    def _seed_all(seed: int):
        seed = int(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _pack_compacts_for_infinity(
        kv_list: Sequence[torch.Tensor],
        lens_list: Sequence[int],
        device: str,
        kv_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
        """
        Builds Infinity's label_B_or_BLT tuple for arbitrary batch size B:

          kv_compact_cat: [sum(Li), C]
          lens: List[int] length B
          cu_seqlens_k: int32 [B+1] cumulative
          Ltext: max(Li)

        This matches your encode_prompt() structure but batched.
        """
        lens = [int(x) for x in lens_list]
        if len(kv_list) != len(lens):
            raise ValueError("kv_list and lens_list must have same length")
        if len(lens) == 0:
            raise ValueError("empty batch")

        Ltext = int(max(lens))
        cu = torch.tensor([0] + list(np.cumsum(lens).astype(np.int32)), device=device, dtype=torch.int32)  # [B+1]

        kv_cat = torch.cat([kv.to(device=device, dtype=kv_dtype, non_blocking=True) for kv in kv_list], dim=0)
        return kv_cat, lens, cu, Ltext

    @staticmethod
    def _to_pil_list(img_list: List[Any]) -> List[Image.Image]:
        out: List[Image.Image] = []
        for img in img_list:
            if isinstance(img, Image.Image):
                out.append(img)
            elif torch.is_tensor(img):
                x = img.detach().cpu()
                if x.ndim == 3 and x.shape[-1] == 3:  # HWC
                    arr = x.numpy()
                elif x.ndim == 3 and x.shape[0] == 3:  # CHW
                    arr = x.permute(1, 2, 0).numpy()
                else:
                    raise ValueError(f"Unsupported tensor image shape: {tuple(x.shape)}")
                if arr.dtype != np.uint8:
                    if arr.max() <= 1.5:
                        arr = (arr * 255.0).clip(0, 255)
                    arr = arr.astype(np.uint8)
                out.append(Image.fromarray(arr))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        return out

    @torch.no_grad()
    def generate_one_batch_from_compacts(
        self,
        *,
        kv_compact_list: Sequence[torch.Tensor],  # CPU tensors or already on GPU
        lens_list: Sequence[int],
        seed: int,
        guidance_scale: float,  # maps to cfg_sc in Infinity
        cfg_list: Union[float, List[float]],
        tau_list: Union[float, List[float]],
        negative_kv_compact_list: Optional[Sequence[torch.Tensor]] = None,
        negative_lens_list: Optional[Sequence[int]] = None,

        # optional overrides
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg_exp_k: Optional[float] = None,
        cfg_insertion_layer: Optional[int] = None,
        vae_type: Optional[int] = None,
        sampling_per_bits: Optional[int] = None,
        gumbel: Optional[int] = None,
        softmax_merge_topk: Optional[int] = None,
        gt_leak: Optional[int] = None,
        gt_ls_Bl: Any = None,
    ) -> List[Image.Image]:
        """
        One Infinity call for the whole batch (B = len(kv_compact_list)).
        """
        self._seed_all(seed)

        B = int(len(lens_list))
        if B == 0:
            return []

        # defaults
        top_k = self.top_k_default if top_k is None else int(top_k)
        top_p = self.top_p_default if top_p is None else float(top_p)
        cfg_exp_k = self.cfg_exp_k_default if cfg_exp_k is None else float(cfg_exp_k)
        cfg_insertion_layer = self.cfg_insertion_layer_default if cfg_insertion_layer is None else int(cfg_insertion_layer)
        vae_type = self.vae_type if vae_type is None else int(vae_type)
        sampling_per_bits = self.sampling_per_bits_default if sampling_per_bits is None else int(sampling_per_bits)
        gumbel = self.gumbel_default if gumbel is None else int(gumbel)
        softmax_merge_topk = self.softmax_merge_topk_default if softmax_merge_topk is None else int(softmax_merge_topk)
        gt_leak = self.gt_leak_default if gt_leak is None else int(gt_leak)
        # -------------------------
        # Infinity expects cfg_list/tau_list as a *list* (per-scale schedule).
        # Your backend may pass floats -> convert & pad/truncate to schedule length.
        # -------------------------
        T = len(self.scale_schedule)

        def _as_schedule_list(x, name: str) -> List[float]:
            if x is None:
                raise ValueError(f"{name} cannot be None for Infinity (need scalar or list).")

            # scalar -> repeat
            if isinstance(x, (float, int)):
                return [float(x)] * T

            # 0-d tensor -> repeat
            if torch.is_tensor(x):
                if x.ndim == 0:
                    return [float(x.item())] * T
                x = x.detach().cpu().tolist()

            # list/tuple -> normalize
            if isinstance(x, (list, tuple)):
                xs = [float(v) for v in x]
                if len(xs) < T:
                    xs = xs + [xs[-1]] * (T - len(xs))   # pad with last
                elif len(xs) > T:
                    xs = xs[:T]                          # truncate
                return xs

            raise TypeError(f"{name} must be float/int/tensor/list, got {type(x)}")

        cfg_list = _as_schedule_list(cfg_list, "cfg_list")
        tau_list = _as_schedule_list(tau_list, "tau_list")

        # Pack label tuple
        label = self._pack_compacts_for_infinity(
            kv_list=kv_compact_list,
            lens_list=lens_list,
            device=self.device,
            kv_dtype=torch.float16 if self.DTYPE == torch.float16 else torch.bfloat16,
        )

        if negative_kv_compact_list is not None and negative_lens_list is not None:
            neg_label = self._pack_compacts_for_infinity(
                kv_list=negative_kv_compact_list,
                lens_list=negative_lens_list,
                device=self.device,
                kv_dtype=torch.float16 if self.DTYPE == torch.float16 else torch.bfloat16,
            )
        else:
            neg_label = None

        with torch.cuda.amp.autocast(enabled=self.bf16, dtype=torch.bfloat16, cache_enabled=True):
            _, _, img_list = self.infinity.autoregressive_infer_cfg(
                vae=self.vae,
                scale_schedule=self.scale_schedule,
                label_B_or_BLT=label,
                g_seed=int(seed),
                B=B,
                negative_label_B_or_BLT=neg_label,
                force_gt_Bhw=None,
                cfg_sc=float(guidance_scale),
                cfg_list=cfg_list,
                tau_list=tau_list,
                top_k=int(top_k),
                top_p=float(top_p),
                returns_vemb=1,
                ratio_Bl1=None,
                gumbel=int(gumbel),
                norm_cfg=False,
                cfg_exp_k=float(cfg_exp_k),
                cfg_insertion_layer=[int(cfg_insertion_layer)],
                vae_type=int(vae_type),
                softmax_merge_topk=int(softmax_merge_topk),
                ret_img=True,
                trunk_scale=1000,
                gt_leak=int(gt_leak),
                gt_ls_Bl=gt_ls_Bl,
                inference_mode=True,
                sampling_per_bits=int(sampling_per_bits),
            )

        return self._to_pil_list(img_list)

    # ESBaseModel API (we won't use prompt_attention_mask here)
    @torch.no_grad()
    def generate(
        self,
        prompt_embeds: Optional[Any] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        seed: int = 0,
        guidance_scale: float = 3.0,
        width_latent: int = 32,
        height_latent: int = 32,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Use InfinityBackend.generate_flat(...) which indexes prompt cache and calls generate_one_batch_from_compacts()."
        )