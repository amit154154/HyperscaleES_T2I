from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from peft import LoraConfig, get_peft_model
from utills import *
from utills import _sync_cuda
# Backends
from models.SanaSprint import SanaOneStep, SanaPipelineES
from models.VAR import VARClassGenerator
from models.zImageTurbo import ZImageTurboES

class ESBackend:
    """
    Backend interface: "the same ES training" across different generators.
    """
    name: str

    def init_and_attach_lora(self) -> None:
        raise NotImplementedError

    def compile_if_requested(self) -> None:
        pass

    def collect_lora_params(self) -> Tuple[List[torch.nn.Parameter], List[Tuple[int, ...]]]:
        raise NotImplementedError

    def save_lora(self, save_dir: Path) -> None:
        """
        Must save PEFT adapters (save_pretrained).
        """
        raise NotImplementedError

    def step_sampling_info(self, seed: int) -> Dict[str, Any]:
        """
        Returns dict with:
          - unique_ids: List[int] (#m)
          - flat_ids:   List[int] (#m * repeats)
          - unique_texts: List[str] (#m) for captions
          - flat_texts:   List[str] (#flat)
          - m: int (#unique)
          - total_imgs_per_indiv: int
          - total_imgs_for_logging: int
          - pid_to_j: Dict[int,int] mapping unique id -> prompt index j
        """
        raise NotImplementedError

    def generate_flat(self, flat_ids: List[int], seed: int, guidance_scale: float) -> List[Any]:
        """
        Generate images for each flat_id in order. Returns list of PIL images.
        """
        raise NotImplementedError


# -------------------------
# Sana backend
# -------------------------

@dataclass
class SanaConfig:
    model_name: str
    backend_mode: str                 # "one_step" or "pipeline"

    # NEW: always provide both
    prompts_txt_path: str
    encoded_prompt_path: str
    auto_encode_if_missing: bool

    # encoding knobs
    encode_batch_size: int

    guidance_scale: float
    width_latent: int
    height_latent: int
    batch_size: int
    prompts_per_gen: int
    batches_per_gen: int
    max_log_batches: int
    torch_compile: bool
    compile_mode: str
    compile_fullgraph: bool
    dtype_latents: str

    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


class SanaBackend(ESBackend):
    def __init__(self, device: str, cfg: SanaConfig):
        self.name = f"sana_{cfg.backend_mode}"
        self.device = device
        self.cfg = cfg
        self.es_model = None
        self.prompt_data = None
        self.base_prompt_embeds = None
        self.base_attention_mask = None
        self.prompts_list = None

    def _dtype(self) -> torch.dtype:
        if self.cfg.dtype_latents.lower() == "bfloat16":
            return torch.bfloat16
        return torch.float16

    def _load_or_encode_prompts(self):
        enc = Path(self.cfg.encoded_prompt_path)

        if enc.is_file():
            self.prompt_data = torch.load(enc, map_location="cpu")
        else:
            if not self.cfg.auto_encode_if_missing:
                raise FileNotFoundError(f"encoded_prompt_path not found and auto_encode disabled: {enc}")

            txt = Path(self.cfg.prompts_txt_path)
            if not txt.is_file():
                raise FileNotFoundError(f"prompts_txt_path not found: {txt}")

            print(f"[prompts] Sana: {enc} missing -> encoding from {txt}")

            # build a temporary Sana model ONLY for encoding
            if self.cfg.backend_mode == "one_step":
                tmp = SanaOneStep(
                    model_name=self.cfg.model_name,
                    device=self.device,
                    DTYPE=self._dtype(),
                    sigma_data=0.5,
                )
            elif self.cfg.backend_mode == "pipeline":
                tmp = SanaPipelineES(
                    model_name=self.cfg.model_name,
                    device=self.device,
                    DTYPE=self._dtype(),
                    sigma_data=0.5,
                )
            else:
                raise ValueError(f"Unknown Sana backend_mode: {self.cfg.backend_mode}")

            _ = tmp.encode_prompts(
                prompts_txt_path=txt,
                encoded_save_path=enc,
                batch_size=int(self.cfg.encode_batch_size),
                overwrite=True,
            )

            # optional cleanup hook
            if hasattr(tmp, "drop_text_encoder"):
                tmp.drop_text_encoder()

            del tmp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _sync_cuda()

            self.prompt_data = torch.load(enc, map_location="cpu")

        # unpack
        self.base_prompt_embeds = self.prompt_data["prompt_embeds"]
        self.base_attention_mask = self.prompt_data["prompt_attention_mask"]
        self.prompts_list = self.prompt_data.get("prompts", None)

        if not torch.is_tensor(self.base_prompt_embeds):
            raise RuntimeError("Sana expects prompt_embeds as a Tensor [P, seq, dim].")
        if not torch.is_tensor(self.base_attention_mask):
            raise RuntimeError("Sana expects prompt_attention_mask as a Tensor [P, seq].")
    def init_and_attach_lora(self):
        self._load_or_encode_prompts()

        if self.cfg.backend_mode == "one_step":
            self.es_model = SanaOneStep(
                model_name=self.cfg.model_name,
                device=self.device,
                DTYPE=self._dtype(),
                sigma_data=0.5,
            )
        elif self.cfg.backend_mode == "pipeline":
            self.es_model = SanaPipelineES(
                model_name=self.cfg.model_name,
                device=self.device,
                DTYPE=self._dtype(),
                sigma_data=0.5,
            )
        else:
            raise ValueError(f"Unknown Sana backend_mode: {self.cfg.backend_mode}")

        # Attach LoRA to transformer
        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
        )
        self.es_model.transformer = get_peft_model(self.es_model.transformer, lora_config)
        self.es_model.transformer.to(self.device).eval()

    def compile_if_requested(self):
        if not self.cfg.torch_compile:
            return
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            print("[compile] Sana: CUDA not available -> skipping torch.compile")
            return
        try:
            print(f"[compile] Sana: compiling transformer (mode={self.cfg.compile_mode}, fullgraph={self.cfg.compile_fullgraph})")
            self.es_model.transformer = torch.compile(
                self.es_model.transformer,
                mode=self.cfg.compile_mode,
                fullgraph=self.cfg.compile_fullgraph,
            )
            # optional VAE compile if present
            if hasattr(self.es_model, "vae") and self.es_model.vae is not None:
                print(f"[compile] Sana: compiling VAE (mode={self.cfg.compile_mode}, fullgraph={self.cfg.compile_fullgraph})")
                self.es_model.vae = torch.compile(
                    self.es_model.vae,
                    mode=self.cfg.compile_mode,
                    fullgraph=self.cfg.compile_fullgraph,
                )
            print("[compile] Sana: SUCCESS")
        except Exception as e:
            print(f"[compile] Sana: FAILED: {e} -> eager mode")

    def collect_lora_params(self):
        return get_trainable_params_and_shapes(self.es_model.transformer)

    def save_lora(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        self.es_model.transformer.save_pretrained(save_dir)

    def step_sampling_info(self, seed: int) -> Dict[str, Any]:
        P = int(self.base_prompt_embeds.shape[0])
        unique_ids = sample_indices_unique(seed=seed, total=P, k=self.cfg.prompts_per_gen)
        flat_ids = repeat_batches(unique_ids, repeats=self.cfg.batches_per_gen)

        pid_to_j = {pid: j for j, pid in enumerate(unique_ids)}
        m = len(unique_ids)

        unique_texts = []
        for pid in unique_ids:
            unique_texts.append(self.prompts_list[pid] if self.prompts_list is not None else f"prompt_{pid}")

        flat_texts = [(self.prompts_list[pid] if self.prompts_list is not None else f"prompt_{pid}") for pid in flat_ids]

        # logging only first N repeats
        log_batches = int(max(0, min(self.cfg.max_log_batches, self.cfg.batches_per_gen)))
        total_imgs_for_logging = log_batches * m
        total_imgs_per_indiv = len(flat_ids)

        return dict(
            unique_ids=unique_ids,
            flat_ids=flat_ids,
            unique_texts=unique_texts,
            flat_texts=flat_texts,
            pid_to_j=pid_to_j,
            m=m,
            total_imgs_per_indiv=total_imgs_per_indiv,
            total_imgs_for_logging=total_imgs_for_logging,
            log_batches=log_batches,
        )

    def generate_flat(self, flat_ids: List[int], seed: int, guidance_scale: float) -> List[Any]:
        # gather embeds/masks for flat ids
        pe = self.base_prompt_embeds[flat_ids].to(self.device)
        am = self.base_attention_mask[flat_ids].to(self.device)

        if isinstance(self.es_model, SanaPipelineES) or hasattr(self.es_model, "generate_one_batch"):
            # pipeline can do one call over a batch of prompts
            images, _ = self.es_model.generate_one_batch(
                prompt_embeds=pe,
                prompt_attention_mask=am,
                seed=seed,
                guidance_scale=guidance_scale,
                width_latent=self.cfg.width_latent,
                height_latent=self.cfg.height_latent,
            )
            return images

        # one_step: single call too (it already accepts batch)
        images, _ = self.es_model.generate(
            prompt_embeds=pe,
            prompt_attention_mask=am,
            latents=None,
            seed=seed,
            guidance_scale=guidance_scale,
            width_latent=self.cfg.width_latent,
            height_latent=self.cfg.height_latent,
        )
        return images


# -------------------------
# VAR backend
# -------------------------

@dataclass
class VarConfig:
    model_depth: int
    ckpt_dir: str
    download_if_missing: bool
    guidance_scale: float
    allowed_classes: Union[str, Sequence[int]]  # "all" or list
    classes_per_gen: int
    batches_per_gen: int
    max_log_batches: int
    torch_compile: bool
    compile_mode: str
    compile_fullgraph: bool
    # LoRA
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


class VarBackend(ESBackend):
    def __init__(self, device: str, cfg: VarConfig):
        self.name = "var_class"
        self.device = device
        self.cfg = cfg
        self.es_model: Optional[VARClassGenerator] = None

    def init_and_attach_lora(self):
        self.es_model = VARClassGenerator(
            model_depth=self.cfg.model_depth,
            device=self.device.split(":")[0] if self.device.startswith("cuda") else self.device,
            ckpt_dir=self.cfg.ckpt_dir,
            download_if_missing=self.cfg.download_if_missing,
        )

        lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
        )
        self.es_model.transformer = get_peft_model(self.es_model.transformer, lora_config)
        self.es_model.transformer.to(self.device).eval()

        # IMPORTANT: VARClassGenerator.generate uses self.var.autoregressive_infer_cfg
        self.es_model.var = self.es_model.transformer

    def compile_if_requested(self):
        if not self.cfg.torch_compile:
            return
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            print("[compile] VAR: CUDA not available -> skipping torch.compile")
            return
        try:
            print(f"[compile] VAR: compiling var(+vae) (mode={self.cfg.compile_mode}, fullgraph={self.cfg.compile_fullgraph})")
            self.es_model.var = torch.compile(
                self.es_model.transformer,
                mode=self.cfg.compile_mode,
                fullgraph=self.cfg.compile_fullgraph,
            )
            if hasattr(self.es_model, "vae") and self.es_model.vae is not None:
                self.es_model.vae = torch.compile(
                    self.es_model.vae,
                    mode=self.cfg.compile_mode,
                    fullgraph=self.cfg.compile_fullgraph,
                )
            print("[compile] VAR: SUCCESS")
        except Exception as e:
            print(f"[compile] VAR: FAILED: {e} -> eager mode")
            self.es_model.var = self.es_model.transformer

    def collect_lora_params(self):
        return get_trainable_params_and_shapes(self.es_model.transformer)

    def save_lora(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        self.es_model.transformer.save_pretrained(save_dir)

    def _sample_classes_unique(self, seed: int, num_classes_total: int = 1000) -> List[int]:
        rng = np.random.RandomState(int(seed))
        allowed = self.cfg.allowed_classes
        if allowed is None or allowed == "all":
            pool = np.arange(num_classes_total, dtype=np.int64)
        else:
            pool = np.array(list(allowed), dtype=np.int64)
            pool = np.unique(pool)
            pool = pool[(pool >= 0) & (pool < num_classes_total)]
            if pool.size == 0:
                pool = np.arange(num_classes_total, dtype=np.int64)

        m = int(self.cfg.classes_per_gen)
        if m <= 0:
            raise ValueError("classes_per_gen must be >= 1")
        if m > pool.size:
            raise ValueError(f"classes_per_gen ({m}) > pool size ({pool.size})")

        chosen = rng.choice(pool, size=m, replace=False)
        return chosen.tolist()

    def step_sampling_info(self, seed: int) -> Dict[str, Any]:
        unique_ids = self._sample_classes_unique(seed=seed, num_classes_total=1000)
        flat_ids = repeat_batches(unique_ids, repeats=self.cfg.batches_per_gen)

        pid_to_j = {cid: j for j, cid in enumerate(unique_ids)}
        m = len(unique_ids)

        unique_texts = [imagenet_prompt_text(int(cid)) for cid in unique_ids]
        flat_texts = [imagenet_prompt_text(int(cid)) for cid in flat_ids]

        log_batches = int(max(0, min(self.cfg.max_log_batches, self.cfg.batches_per_gen)))
        total_imgs_for_logging = log_batches * m
        total_imgs_per_indiv = len(flat_ids)

        return dict(
            unique_ids=unique_ids,
            flat_ids=flat_ids,
            unique_texts=unique_texts,
            flat_texts=flat_texts,
            pid_to_j=pid_to_j,
            m=m,
            total_imgs_per_indiv=total_imgs_per_indiv,
            total_imgs_for_logging=total_imgs_for_logging,
            log_batches=log_batches,
        )

    def generate_flat(self, flat_ids: List[int], seed: int, guidance_scale: float) -> List[Any]:
        """
        VAR generator supports grouped classes as [repeats][m]. We rebuild that structure.
        """
        m = self.cfg.classes_per_gen
        r = self.cfg.batches_per_gen
        if len(flat_ids) != m * r:
            raise RuntimeError(f"VAR expected flat_ids length {m*r}, got {len(flat_ids)}")

        grouped: List[List[int]] = []
        idx = 0
        for _ in range(r):
            grouped.append([int(flat_ids[idx + j]) for j in range(m)])
            idx += m

        imgs_grouped, _ = self.es_model.generate(
            seed=seed,
            guidance_scale=guidance_scale,
            class_ids=grouped,
            return_grouped=True,
        )
        # flatten [r][m] to list
        out = []
        for b in range(r):
            for j in range(m):
                out.append(imgs_grouped[b][j])
        return out


# -------------------------
# Z-Image backend
# -------------------------

@dataclass
class ZImageConfig:
    model_name: str
    prompts_txt_path: str
    encoded_prompt_path: str
    auto_encode_if_missing: bool

    width_px: int
    height_px: int
    num_inference_steps: int
    guidance_scale: float
    micro_batch: int

    prompts_per_gen: int
    batches_per_gen: int
    max_log_batches: int

    compile_transformer: bool
    attention_backend: str

    # GGUF / quantization options
    use_gguf: bool
    gguf_repo_id: str
    gguf_filename: str
    gguf_local_dir: str
    gguf_local_path: Optional[str]

    # LoRA transformer
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

    # Optional VAE decoder LoRA
    use_vae_decoder_lora: bool
    vae_lora_r: int
    vae_lora_alpha: int
    vae_lora_dropout: float
    vae_lora_target_modules: List[str]

    dtype: str  # "bfloat16" etc


class ZImageBackend(ESBackend):
    def __init__(self, device: str, cfg: ZImageConfig):
        self.name = "zimage"
        self.device = device
        self.cfg = cfg
        self.es_model: Optional[ZImageTurboES] = None
        self.prompt_data = None
        self.base_prompt_embeds = None  # list[Tensor] or Tensor
        self.prompts_list = None

    def _dtype(self) -> torch.dtype:
        if self.cfg.dtype.lower() == "bfloat16":
            return torch.bfloat16
        if self.cfg.dtype.lower() == "float16":
            return torch.float16
        return torch.bfloat16

    def _load_or_encode_prompts(self):
        enc = Path(self.cfg.encoded_prompt_path)
        if enc.is_file():
            self.prompt_data = torch.load(enc, map_location="cpu")
            self.base_prompt_embeds = self.prompt_data["prompt_embeds"]
            self.prompts_list = self.prompt_data.get("prompts", None)
            return

        if not self.cfg.auto_encode_if_missing:
            raise FileNotFoundError(f"encoded_prompt_path not found and auto_encode disabled: {enc}")

        txt = Path(self.cfg.prompts_txt_path)
        if not txt.is_file():
            raise FileNotFoundError(f"prompts_txt_path not found: {txt}")

        # build temp model to encode
        print(f"[prompts] {enc} missing -> encoding prompts from {txt}")
        tmp = ZImageTurboES(
            model_name=self.cfg.model_name,
            device=self.device,
            DTYPE=self._dtype(),
            num_inference_steps=self.cfg.num_inference_steps,
            compile_transformer=self.cfg.compile_transformer,
            attention_backend=self.cfg.attention_backend,
            low_cpu_mem_usage=False,
            use_quantize=self.cfg.use_gguf,
            gguf_repo_id=self.cfg.gguf_repo_id,
            gguf_filename=self.cfg.gguf_filename,
            gguf_local_dir=self.cfg.gguf_local_dir,
            gguf_local_path=self.cfg.gguf_local_path,
        )
        _ = tmp.encode_prompts(
            prompts_txt_path=txt,
            encoded_save_path=enc,
            batch_size=64,
            max_sequence_length=512,
            overwrite=True,
        )
        if hasattr(tmp, "drop_text_encoder"):
            tmp.drop_text_encoder()
        del tmp
        torch.cuda.empty_cache()
        _sync_cuda()

        self.prompt_data = torch.load(enc, map_location="cpu")
        self.base_prompt_embeds = self.prompt_data["prompt_embeds"]
        self.prompts_list = self.prompt_data.get("prompts", None)

    def init_and_attach_lora(self):
        if not self.device.startswith("cuda"):
            raise RuntimeError("Z-Image-Turbo is intended for CUDA in your setup.")

        self._load_or_encode_prompts()

        self.es_model = ZImageTurboES(
            model_name=self.cfg.model_name,
            device=self.device,
            DTYPE=self._dtype(),
            num_inference_steps=self.cfg.num_inference_steps,
            compile_transformer=self.cfg.compile_transformer,
            attention_backend=self.cfg.attention_backend,
            low_cpu_mem_usage=False,
            use_quantize=self.cfg.use_gguf,
            gguf_repo_id=self.cfg.gguf_repo_id,
            gguf_filename=self.cfg.gguf_filename,
            gguf_local_dir=self.cfg.gguf_local_dir,
            gguf_local_path=self.cfg.gguf_local_path,
        )

        # Transformer LoRA
        tr_lora_config = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.lora_target_modules,
        )
        peft_transformer = get_peft_model(self.es_model.pipe.transformer, tr_lora_config)
        self.es_model.pipe.transformer = peft_transformer
        self.es_model.transformer = peft_transformer
        self.es_model.transformer.to(self.device).eval()

        # Optional VAE decoder LoRA
        if self.cfg.use_vae_decoder_lora:
            vae_lora_config = LoraConfig(
                r=self.cfg.vae_lora_r,
                lora_alpha=self.cfg.vae_lora_alpha,
                lora_dropout=self.cfg.vae_lora_dropout,
                target_modules=self.cfg.vae_lora_target_modules,
            )
            peft_dec = get_peft_model(self.es_model.pipe.vae.decoder, vae_lora_config)
            self.es_model.pipe.vae.decoder = peft_dec
            self.es_model.pipe.vae.decoder.to(self.device).eval()

    def compile_if_requested(self):
        # ZImageTurboES already handles internal compile_transformer; keep this no-op.
        return

    def collect_lora_params(self):
        tr_params, tr_shapes = get_trainable_params_and_shapes(self.es_model.transformer)
        if self.cfg.use_vae_decoder_lora:
            vae_params, vae_shapes = get_trainable_params_and_shapes(self.es_model.pipe.vae.decoder)
        else:
            vae_params, vae_shapes = [], []
        return (tr_params + vae_params, tr_shapes + vae_shapes)

    def save_lora(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # keep same folder layout you used
        (save_dir / "transformer").mkdir(parents=True, exist_ok=True)
        self.es_model.transformer.save_pretrained(save_dir / "transformer")
        if self.cfg.use_vae_decoder_lora:
            (save_dir / "vae_decoder").mkdir(parents=True, exist_ok=True)
            self.es_model.pipe.vae.decoder.save_pretrained(save_dir / "vae_decoder")

    def _total_prompts(self) -> int:
        if isinstance(self.base_prompt_embeds, list):
            return len(self.base_prompt_embeds)
        return int(self.base_prompt_embeds.shape[0])

    def _get_prompt_embed(self, pid: int) -> torch.Tensor:
        if isinstance(self.base_prompt_embeds, list):
            return self.base_prompt_embeds[pid].to(self.device)
        return self.base_prompt_embeds[pid].to(self.device)

    def step_sampling_info(self, seed: int) -> Dict[str, Any]:
        total = self._total_prompts()
        unique_ids = sample_indices_unique(seed=seed, total=total, k=self.cfg.prompts_per_gen)
        flat_ids = repeat_batches(unique_ids, repeats=self.cfg.batches_per_gen)
        pid_to_j = {pid: j for j, pid in enumerate(unique_ids)}
        m = len(unique_ids)

        unique_texts = [(self.prompts_list[pid] if self.prompts_list is not None else f"prompt_{pid}") for pid in unique_ids]
        flat_texts = [(self.prompts_list[pid] if self.prompts_list is not None else f"prompt_{pid}") for pid in flat_ids]

        log_batches = int(max(0, min(self.cfg.max_log_batches, self.cfg.batches_per_gen)))
        total_imgs_for_logging = log_batches * m
        total_imgs_per_indiv = len(flat_ids)

        return dict(
            unique_ids=unique_ids,
            flat_ids=flat_ids,
            unique_texts=unique_texts,
            flat_texts=flat_texts,
            pid_to_j=pid_to_j,
            m=m,
            total_imgs_per_indiv=total_imgs_per_indiv,
            total_imgs_for_logging=total_imgs_for_logging,
            log_batches=log_batches,
        )

    def generate_flat(self, flat_ids: List[int], seed: int, guidance_scale: float) -> List[Any]:
        flat_embeds: List[torch.Tensor] = [self._get_prompt_embed(pid) for pid in flat_ids]
        images, _ = self.es_model.generate_one_batch(
            prompt_embeds=flat_embeds,
            seed=seed,
            guidance_scale=guidance_scale,
            width_px=self.cfg.width_px,
            height_px=self.cfg.height_px,
            num_inference_steps=self.cfg.num_inference_steps,
            micro_batch=self.cfg.micro_batch,
        )
        return images

def save_latest_checkpoint(
    *,
    theta: torch.Tensor,
    backend: ESBackend,
    lora_params: List[torch.nn.Parameter],
    lora_shapes: List[Tuple[int, ...]],
    save_dir: Path,
    meta_path: Path,
    epoch: int,
    stats: Dict[str, float],
    extra_meta: Dict[str, Any],
):
    print(f"[ckpt] Saving latest LoRA at epoch {epoch} -> {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Apply theta
    unflatten_to_params(theta, lora_params, lora_shapes)

    # Save adapters
    backend.save_lora(save_dir)

    # Meta
    payload = {
        "theta_latest": theta.detach().cpu(),
        "epoch": epoch,
        "summary_mean_reward": stats.get("summary/mean_reward", float("nan")),
        "backend": backend.name,
        **extra_meta,
    }
    torch.save(payload, meta_path)