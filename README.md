# ES-SanaSprint-OneStep (EGGROLL-style ES for T2I)

This repo is an **experimental playground** for bringing ideas from  
[**“Evolution Strategies at the Hyperscale” (Sarkar et al., 2025)** ](https://arxiv.org/abs/2502.14575)
into the **text-to-image** world, using **Sana Sprint**’s **single-step** diffusion model  
(**`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`**) as the base generator.

The goal is to translate the **EGGROLL** evolution strategy from the LLM domain to the  
**personalized text-to-image** setting and compare it against modern RL-style fine-tuning  
methods such as **PPO**, **DPO**, **GRPO**, and particularly **pairwise preference GRPO**  
(e.g. [*Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Models*](https://arxiv.org/abs/2508.20751))

Concretely:

- We treat the **LoRA weights** of Sana Sprint as the parameter vector θ.
- We apply an **EGGROLL-style low-rank ES** update in parameter space using image-level rewards.
- Rewards are built from **CLIP**, **aesthetic models**, “no-artifacts” heuristics, and **PickScore**.
- The project is designed to be extensible so the same pipeline can later be compared with  
  **RL fine-tuning** style approaches (PPO/DPO/GRPO/Pref-GRPO) on the *same* prompts and rewards.

This is **research code**, not a production library – expect rough edges, lots of knobs, but i will clean it i promise (:
