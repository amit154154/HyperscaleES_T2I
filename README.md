# ES-EGGROLL for Text-to-Image

This repo is an **experimental playground** for implementing and exploring  
[**Evolution Strategies at the Hyperscale (Sarkar et al., 2025)**](https://arxiv.org/abs/2502.14575)  
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


## First experiment: ES vs base on PartiPrompts (CompBench‑T2I)

  
[![W&B – first training run](https://img.shields.io/badge/W%26B-PartiPrompts%20ES%20run-ffbe00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/amit154154/SanaSprint-ES-multiprompt/runs/txgc6nc0?nw=nwuseramit154154)

All training logs, metrics, reward traces, and best/median/worst image strips for this experiment are available in the W&B run linked above.

In the first experiment I fine‑tuned a LoRA on top of a Sana‑style single‑step T2I model using **300 ES steps**,  
population size **32**, and **4 prompts per candidate** and PickScore as a reward. To sanity‑check the effect of EGGROLL ES, I evaluated the  
**base model** and the **ES‑LoRA** on the **PartiPrompts** , **one image per prompt** with same seed and scoring with CLIP‑based metrics and PickScore.

Metrics:
- `aesthetic` – CLIP similarity to a “beautiful, high‑quality image” text.
- `text` – CLIP similarity to the actual prompt.
- `no_artifacts` – 1 − CLIP similarity to a “bad / artifacty image” text.
- `pickscore` – Yuval Kirstain’s PickScore_v1 (higher is better).

### Overall score

| Model                  | #images | aesthetic ↑ |      text ↑      |   no_artifacts ↑    |       pickscore↑       |
|------------------------|:-------:|:-----------:|:----------------:|:-------------------:|:----------------------:|
| SanaSprintOneStep_Base |  1631   | **0.5978**  |      0.6592      |       0.3859        |        22.3220         |
| SanaSprintOneStep_LoRA |  1630   |   0.5969    |    **0.660**     |     **0.3880**      |      **22.3734**       |


**Quick takeaway:** after 300 ES steps the LoRA is **very close to the base model** on all metrics, with tiny but  
consistent improvements in **text alignment** and **PickScore** on several categories/challenges (e.g. Animals, Arts,  
Imagination, Complex, Fine‑grained Detail). Aesthetics and artifact rates move only at the 0.001–0.003 level, which  
suggests this ES run behaves like a gentle “pre‑RL” nudging of the model rather than a strong personalization step.  
Future experiments will push more aggressive schedules, narrower prompt slices, and RL‑style baselines on the same setup.


