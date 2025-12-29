# ES-EGGROLL for Text-to-Image (T2I)

[![Log project Site](https://img.shields.io/badge/Project%20Site-ES--EGGROLL%20Log-ffbe00?style=for-the-badge)](https://amit154154.github.io/HyperscaleES_T2I/)
[![EggRoll Paper](https://img.shields.io/badge/Paper-EGGROLL%20(arXiv%202511.16652)-ffbe00?style=for-the-badge)](https://arxiv.org/abs/2511.16652)

research repo for **post-train a frozen T2I model with black-box rewards** using **EGGROLL-style Evolution Strategies** on **LoRA weights** (no diffusion backprop).


---

## Main result (PartiPrompts, overall)

One image per prompt, **shared seeds** across models.

| Model | aesthetic ↑ | CLIP text sim ↑ | no artifacts ↑ | PickScore ↑ |
|---|---:|---:|---:|---:|
| SanaOneStep_Base | 0.5978 | 0.6592 | 0.3859 | 22.3220 |
| **SanaOneStep_eggroll (ES-LoRA)** | 0.5975 | **0.6611** | **0.3899** | **22.5013** |
| SanaTwoStep_Base (more compute) | 0.5965 | 0.6614 | 0.3926 | 22.8059 |

[![W&B Run](https://img.shields.io/badge/W%26B-Training%20Run-ffbe00?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/amit154154/SanaSprintOneStep-ES-Search-v2_1/runs/hrkxbeyv?nw=nwuseramit154154)

---

## Qualitative (Base vs ES-LoRA)

| Prompt | Base                                                      | ES-LoRA                                                   |
|---|-----------------------------------------------------------|-----------------------------------------------------------|
| spaghetti cowboy | ![](assets/first_experimant_assets/cowboy_base.jpg)       | ![](assets/first_experimant_assets/cowboy_lora.jpg)       |
| Will Smith egg roll | ![](assets/first_experimant_assets/will_smith_base.jpg)   | ![](assets/first_experimant_assets/will_smith_lora.jpg)   |
| close-up eyes | ![](assets/first_experimant_assets/human_eyes_base.jpg)   | ![](assets/first_experimant_assets/human_eyes_lora.jpg)   |
| dragon + skyscraper | ![](assets/first_experimant_assets/dragon_before.jpg)     | ![](assets/first_experimant_assets/drageon_after.jpg)     |
| hand + crystal | ![](assets/first_experimant_assets/hand_crystel_base.jpg) | ![](assets/first_experimant_assets/hand_crystel_lora.jpg) |


**Full prompts**
- **Neon egg roll (Will Smith):** `Will Smith eating an egg roll on a neon-lit street in Tokyo at night, cinematic, shallow depth of field, 35mm photography`
- **Spaghetti cowboy:** `spaghetti sculpture of a lone cowboy riding into the sunset, entire cowboy and horse made out of spaghetti, cinematic wide shot`
- **Eyes close-up:** `a close-up of human eyes with detailed eyelashes and reflections, ultra realistic`
- **Dragon around skyscraper:** `a dragon curled around a skyscraper in a modern city, overcast sky, realistic`
- **Hand + crystal:** `a hyper-detailed shot of a hand holding a translucent crystal, complex light refractions on a dark background, studio photography style`
- **Tiny astronaut on coffee cup:** `a tiny astronaut sitting on the rim of a coffee cup on a wooden desk, shallow depth of field, soft morning window light, ultra realistic, high detail on textures and reflections`
- **Butterfly glass cat:** `a cat with butterfly wings, its whole body is made of translucent glass`
- **Submarine:** `a submarine`
---

## What’s trained

- Backbone: **Sana Sprint one-step** (`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`)
- Trainable params: **LoRA only**
- Optimized reward: **PickScore v1**
- Extras are logged as diagnostics (CLIP text sim, aesthetic, no-artifacts)


## Train with `unifed_es.py` (Unified ES/EGGROLL runner)

This repo includes a single training entrypoint: **`unifed_es.py`**. It runs **EGGROLL-style Evolution Strategies** to optimize **LoRA weights** on top of a *frozen* T2I model using **reward feedback** (PickScore / CLIP, etc.). No diffusion backprop required.

### 1) Prepare a prompts `.txt`
Create a plain text file where **each line is one prompt**.

### 2) Key arguments (what you’ll actually touch)
**Core**
- `--backend {sana_one_step,sana_pipeline,var,zimage,infinity}`: choose which model backend to train.
- `--device {auto,cuda:0,cpu}`: device selection.

**Training / ES**
- `--num_epochs`: number of ES iterations.
- `--pop_size`: ES population size (higher = more compute).
- `--sigma`: noise std for sampling LoRA perturbations.
- `--lr_scale`: ES step size / update scale.
- `--egg_rank`: EggRoll low-rank noise rank.
- `--use_antithetic`: reduces variance (recommended).
- `--promptnorm`: prompt-normalized scoring (recommended).

**Rewards**
- `--w_pick`, `--w_text`, `--w_aesthetic`, `--w_noart`: weights for the combined reward (I usually run PickScore-only with `--w_pick 1.0`), if you want you can implament and log/optemize diffrent rewwards in rewards.py 

**Sampling (controls how many images you generate per epoch)**
- `--prompts_per_gen`: number of unique prompts/classes used per “generation group”.
- `--batches_per_gen`: how many groups to generate per individual.
- `--max_log_batches`: how many groups are logged as image strips (W&B).

**Backend-specific prompt files**
- Sana: `--sana_prompts_txt`, `--sana_auto_encode`, `--sana_encoded_prompts`
- VAR:  `--var_allowed_classes`, `--var_classes_per_gen`
- Z-Image: `--zimage_prompts_txt`, `--zimage_auto_encode`, `--zimage_encoded_prompts`
- Infinity: `--inf_prompts_txt`, `--inf_auto_encode`, `--inf_encoded_prompts`, `--inf_variant`

### 3) Run training (examples)

**Sana Sprint (one-step)**
```bash
python unifed_es.py \
  --backend sana_one_step \
  --device cuda:0 \
  --wandb_project ES-Unified \
  --run_name sana_eggroll_test \
  --sana_prompts_txt prompts_train.txt \
  --sana_auto_encode true \
  --num_epochs 200 \
  --pop_size 8 \
  --prompts_per_gen 4 \
  --batches_per_gen 4 \
  --sigma 1e-2 \
  --lr_scale 1e-1 \
  --egg_rank 1 \
  --promptnorm true \
  --w_pick 1.0
```
**Infinity (autoregressive)**
```bash
python unifed_es.py \
  --backend infinity \
  --device cuda:0 \
  --wandb_project ES-Unified \
  --run_name inf_eggroll_test \
  --inf_variant 8b_512 \
  --inf_prompts_txt prompts_train.txt \
  --inf_auto_encode true \
  --inf_drop_text_encoder_after_encode true \
  --inf_guidance_scale 3.0 \
  --num_epochs 100 \
  --pop_size 8 \
  --prompts_per_gen 4 \
  --batches_per_gen 2 \
  --sigma 1e-2 \
  --lr_scale 1e-1 \
  --egg_rank 1 \
  --promptnorm true \
  --w_pick 1.0
```

