# ES-SanaSprint-OneStep (EGGROLL-style ES for T2I)

This repo is an **experimental playground** for bringing ideas from  
**“Evolution Strategies at the Hyperscale” (Sarkar et al., 2025)**  [oai_citation:0‡LinkedIn](https://www.linkedin.com/posts/williamluciw_evolution-strategies-at-the-hyperscale-authors-activity-7399510428239814656-e_h5?utm_source=chatgpt.com)  
into the **text-to-image** world, using **Sana Sprint**’s **single-step** diffusion model  
(**`Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers`**) as the base generator.  [oai_citation:1‡OpenReview](https://openreview.net/pdf/161d90de290b235c9452997948179d7863f30336.pdf?utm_source=chatgpt.com)

The goal is to translate the **EGGROLL** evolution strategy from the LLM domain to the  
**personalized text-to-image** setting and compare it against modern RL-style fine-tuning  
methods such as **PPO**, **DPO**, **GRPO**, and particularly **pairwise preference GRPO**  
(e.g. *Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Models*).  [oai_citation:2‡Semantic Scholar](https://www.semanticscholar.org/paper/Pref-GRPO%3A-Pairwise-Preference-Reward-based-GRPO-Wang-Li/e7197f0ff2e60c94c8009e1c9b0885be6e2b1c2e?utm_source=chatgpt.com)

Concretely:

- We treat the **LoRA weights** of Sana Sprint as the parameter vector θ.
- We apply an **EGGROLL-style low-rank ES** update in parameter space using image-level rewards.
- Rewards are built from **CLIP**, **aesthetic models**, “no-artifacts” heuristics, and **PickScore**.
- The project is designed to be extensible so the same pipeline can later be compared with  
  **RL fine-tuning** style approaches (PPO/DPO/GRPO/Pref-GRPO) on the *same* prompts and rewards.

This is **research code**, not a production library – expect rough edges, lots of knobs, but i will clean it i promise (:
