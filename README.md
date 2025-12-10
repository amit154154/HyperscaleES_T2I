# ES-EGGROLL for Text-to-Image

This repo is an **experimental playground** for implementing and exploring  
[**Evolution Strategies at the Hyperscale (Sarkar et al., 2025)**](https://arxiv.org/abs/2511.16652)  
in the **text-to-image** domain.

The core focus is **EGGROLL-style evolution strategies on LoRA parameters**, with a pluggable
“backend” text-to-image model. Right now the main backend is:

- **Sana Sprint** using a single step setting  
  (`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`)

…but the code is structured so that other T2I models can be dropped in later
with minimal changes.

Conceptually:

- We treat the **LoRA weights** of a T2I model as the parameter vector **θ**.
- We apply an **EGGROLL-style low-rank ES** update directly in parameter space.
- Image-level rewards are built from **CLIP**, **aesthetic scorers**, simple **“no-artifacts”**
  heuristics, and **PickScore** (and can be extended).

The medium-term goal is to compare **EGGROLL ES** against modern RL-style fine-tuning methods on the
*same prompts and reward signals*, including:

- **PPO**, **DPO**, **GRPO**
- **Pairwise preference GRPO**, e.g.  
  [*Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Models*](https://arxiv.org/abs/2508.20751)

This is **research code**, not a production library – expect rough edges, many knobs, and ongoing refactors
as I plug in more backends and reward models :)

## First experiment: ES vs base on PartiPrompts

[![W&B – first training run](https://img.shields.io/badge/W%26B-PartiPrompts%20ES%20run-ffbe00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/amit154154/SanaSprint-ES-multiprompt/runs/txgc6nc0?nw=nwuseramit154154)

All training logs, metrics, reward traces, and best/median/worst image strips for these runs are available in the W&B run linked above.

In this experiment I fine-tune **LoRA adapters** on top of a Sana-style single-step T2I model using **EGGROLL-style ES** and evaluate on **PartiPrompts (CompBench-T2I)**:

- Reward is **PickScore-only** (CLIP-H text–image score).
- ES acts directly in **LoRA parameter space**.
- Evaluation is done on **one image per prompt**, with **shared seeds** between:
  - the **base** model,
  - an earlier ES-LoRA run (**pop 32, 300 steps**), and
  - the newer/better ES-LoRA run  
    **`SanaSprintOneStep_LoRA_pop128_150steps`** (population **128**, **150 ES steps**).

For evaluation I report:

- `aesthetic` – CLIP similarity to a “beautiful, high-quality image” text.
- `text` – CLIP similarity to the actual prompt.
- `no_artifacts` – \(1 -\) CLIP similarity to a “bad / artifacty image” text.
- `pickscore` – Yuval Kirstain’s PickScore_v1 (higher is better).

### Overall score

| Model                                  | #images | aesthetic ↑ |    text ↑     | no_artifacts ↑ |     pickscore ↑     |
|----------------------------------------|:-------:|:-----------:|:-------------:|:--------------:|:-------------------:|
| SanaSprintOneStep_Base                 |  1631   | **0.5978**  |    0.6592     |     0.3859     |       22.3220       |
| SanaSprintOneStep_LoRA_pop32_300steps  |  1630   |   0.5969    |     0.6600    |     0.3880     |       22.3734       |
| SanaSprintOneStep_LoRA_pop128_150steps |  1631   |   0.5975    |  **0.6613**   |   **0.3881**   |     **22.4868**     |

For more detailed benchmark information (per-Category and per-Challenge breakdowns, including this `pop128_150steps` run), see `benchmark_results`.

**Quick takeaway:**  
The **`SanaSprintOneStep_LoRA_pop128_150steps`** model stays essentially **neutral on aesthetics** while giving small but **consistent gains** in:

- **text alignment** (`text_mean` 0.6592 → **0.6613**),
- **artifact suppression** (`no_artifacts_mean` 0.3859 → **0.3881**),
- and big **PickScore** improvment (22.32 → **22.49**),

with improvements showing up across several **PartiPrompts categories** (e.g. Animals, Arts, People) and **challenges** (Complex, Fine-grained Detail, Imagination, Properties & Positioning).
### Qualitative examples

| Prompt     | Base                                                    | EGGROL_LoRA                                             |
|-----------|---------------------------------------------------------|---------------------------------------------------------|
| "spaghetti sculpture of a lone cowboy riding into the sunset, entire cowboy and horse made out of spaghetti, cinematic wide shot"    | ![](assets/first_experimant_assets/cowboy_base.png)     | ![](assets/first_experimant_assets/cowboy_lora.png)     |
| "Will Smith eating an egg roll on a neon-lit street in Tokyo at night, cinematic, shallow depth of field, 35mm photography"| ![](assets/first_experimant_assets/will_smith_base.png) | ![](assets/first_experimant_assets/will_smith_lora.png) |
| "a close-up of human eyes with detailed eyelashes and reflections, ultra realistic"| ![](assets/first_experimant_assets/human_eyes_base.png) | ![](assets/first_experimant_assets/human_eyes_lora.png) |
| "a dragon curled around a skyscraper in a modern city, overcast sky, realistic"| ![](assets/first_experimant_assets/dragon_before.png)   | ![](assets/first_experimant_assets/drageon_after.png)    |

Overall, this suggests that ES-trained LoRA meaningfully fixes many of the typical one-step Sana Sprint failure modes – especially around text alignment, small details, and artifacty regions – and in some cases the improvements are quite strong.
In a small A/B test I ran with the Gradio demo (gradio_inference.py), the ES-LoRA variant achieved a noticeably higher win-rate than the base model in blind comparisons.
