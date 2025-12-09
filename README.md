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