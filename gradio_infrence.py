#!/usr/bin/env python
"""
Gradio Space for ES-SanaSprint-OneStep

- Loads text embeddings (prompt_embeds, attention_mask, prompts) from a .pt file.
- Loads base SanaSprintOneStepES, and optionally a LoRA adapter.
- Lets you interactively:
    * choose Base / LoRA / Both (manual mode)
    * pick which prompt embedding to use
    * pick seed & guidance scale
    * generate images (Base-only, LoRA-only, or side-by-side)

- Also includes a blind "Test it!" mode:
    * samples a random prompt + random seed
    * generates Base vs LoRA in random A/B order
    * lets you pick which is better
    * tracks a session score (LoRA vs Base wins)

Usage (local):

  python app_gradio_es.py \
      --encoded_path encoded_prompts.pt \
      --lora_dir /path/to/best_lora

On HF Spaces:
  - Put this as `app.py` (or similar).
  - Hardcode / env-var the paths instead of CLI args if needed.
"""

import argparse
import random
from pathlib import Path

import torch
from peft import PeftModel
import gradio as gr

from models.SanaSprint import SanaOneStep

# ---------------------------
# Device & model name
# ---------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "mps"
MODEL_NAME = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"


def load_embeddings(encoded_path: Path):
    print(f"[init] Loading encoded prompts from {encoded_path} ...")
    data = torch.load(encoded_path, map_location="cpu")

    prompt_embeds_all = data["prompt_embeds"]           # [P, seq, dim]
    attention_mask_all = data["prompt_attention_mask"]  # [P, seq]
    prompts_list = data.get("prompts", None)

    num_prompts = prompt_embeds_all.shape[0]
    print(f"[init] num_prompts: {num_prompts}")
    if prompts_list is not None:
        print("  Example prompts:")
        for i, p in enumerate(prompts_list[:5]):
            print(f"   [{i:02d}] {p}")

    return prompt_embeds_all, attention_mask_all, prompts_list


def build_prompt_choices(prompts_list, num_prompts):
    """
    Build dropdown choices: (label, value=index).
    """
    choices = []
    for i in range(num_prompts):
        if prompts_list is not None:
            text = prompts_list[i]
        else:
            text = f"prompt_{i}"

        short = text.replace("\n", " ")
        if len(short) > 80:
            short = short[:77] + "..."

        label = f"{i:04d} – {short}"
        choices.append((label, i))
    return choices


def create_models(lora_dir: Path | None):
    """
    Create base Sana model, and if lora_dir is provided, create a LoRA version.
    """
    print(f"[init] Creating base SanaSprintOneStepES on {DEVICE} ...")
    base_sana = SanaOneStep(
        model_name=MODEL_NAME,
        device=DEVICE,
        DTYPE=torch.float16,
        sigma_data=0.5,
    )
    base_sana.transformer.to(DEVICE).eval()
    print("[init] Base model ready.")

    lora_sana = None
    if lora_dir is not None:
        print(f"[init] Creating LoRA SanaSprintOneStepES from {lora_dir} ...")
        lora_sana = SanaOneStep(
            model_name=MODEL_NAME,
            device=DEVICE,
            DTYPE=torch.float16,
            sigma_data=0.5,
        )
        lora_sana.transformer = PeftModel.from_pretrained(
            lora_sana.transformer,
            lora_dir,
        )
        lora_sana.transformer.to(DEVICE).eval()
        print("[init] LoRA model ready.")
    else:
        print("[init] No LoRA directory provided; only base model will be available.")

    return base_sana, lora_sana


def format_score(scores: dict) -> str:
    n = scores.get("n_trials", 0)
    lw = scores.get("lora_wins", 0)
    bw = scores.get("base_wins", 0)
    if n > 0:
        winrate = 100.0 * lw / n
        return (
            f"Session score: {n} votes — "
            f"LoRA wins: {lw}, Base wins: {bw} "
            f"(LoRA win rate: {winrate:.1f}%)"
        )
    else:
        return "Session score: no votes yet. Hit **Test it!** and start choosing."


def build_interface(
    prompt_embeds_all,
    attention_mask_all,
    prompts_list,
    base_sana: SanaOneStep,
    lora_sana: SanaOneStep | None,
):
    num_prompts = prompt_embeds_all.shape[0]
    choices = build_prompt_choices(prompts_list, num_prompts)

    def get_prompt_text(idx: int) -> str:
        if prompts_list is None:
            return f"prompt_{idx}"
        return prompts_list[idx]

    # ------------- Manual generation (Base / LoRA / Both) -------------

    def generate_fn(
        mode: str,
        prompt_index: int,
        seed: int,
        guidance_scale: float,
        width_latent: int,
        height_latent: int,
    ):
        if mode in ("lora", "both") and lora_sana is None:
            raise gr.Error("LoRA mode selected but no LoRA directory was provided at startup.")

        idx = int(prompt_index)
        if idx < 0 or idx >= num_prompts:
            raise gr.Error(f"Invalid prompt index {idx} (0 <= idx < {num_prompts})")

        prompt_embeds = prompt_embeds_all[idx: idx + 1].to(DEVICE)
        attention_mask = attention_mask_all[idx: idx + 1].to(DEVICE)

        seed = int(seed)
        guidance_scale = float(guidance_scale)
        width_latent = int(width_latent)
        height_latent = int(height_latent)

        base_img = None
        lora_img = None

        # Base
        if mode in ("base", "both"):
            images_base, _ = base_sana.generate(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=attention_mask,
                latents=None,
                seed=seed,
                guidance_scale=guidance_scale,
                width_latent=width_latent,
                height_latent=height_latent,
            )
            if not images_base:
                raise gr.Error("Base model did not generate any images.")
            base_img = images_base[0]

        # LoRA
        if mode in ("lora", "both"):
            images_lora, _ = lora_sana.generate(
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=attention_mask,
                latents=None,
                seed=seed,
                guidance_scale=guidance_scale,
                width_latent=width_latent,
                height_latent=height_latent,
            )
            if not images_lora:
                raise gr.Error("LoRA model did not generate any images.")
            lora_img = images_lora[0]

        prompt_text = get_prompt_text(idx)
        return base_img, lora_img, prompt_text

    # ------------- Blind Test mode (A/B randomised) -------------

    def test_fn(
        guidance_scale: float,
        width_latent: int,
        height_latent: int,
        scores: dict,
    ):
        if lora_sana is None:
            raise gr.Error("Blind test requires a LoRA directory. Start the app with --lora_dir.")

        # random prompt index + seed
        idx = random.randrange(num_prompts)
        seed = random.randint(0, 10_000)

        prompt_embeds = prompt_embeds_all[idx: idx + 1].to(DEVICE)
        attention_mask = attention_mask_all[idx: idx + 1].to(DEVICE)

        guidance_scale = float(guidance_scale)
        width_latent = int(width_latent)
        height_latent = int(height_latent)

        # Generate base + LoRA with the same seed
        images_base, _ = base_sana.generate(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=attention_mask,
            latents=None,
            seed=seed,
            guidance_scale=guidance_scale,
            width_latent=width_latent,
            height_latent=height_latent,
        )
        images_lora, _ = lora_sana.generate(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=attention_mask,
            latents=None,
            seed=seed,
            guidance_scale=guidance_scale,
            width_latent=width_latent,
            height_latent=height_latent,
        )

        if not images_base or not images_lora:
            raise gr.Error("Failed to generate images for blind test.")

        base_img = images_base[0]
        lora_img = images_lora[0]

        # Randomise which side is which
        if random.random() < 0.5:
            img_a, img_b = base_img, lora_img
            mapping = {"A": "base", "B": "lora"}
        else:
            img_a, img_b = lora_img, base_img
            mapping = {"A": "lora", "B": "base"}

        # Also keep meta for debugging / future if needed
        mapping["prompt_index"] = idx
        mapping["seed"] = seed

        prompt_text = get_prompt_text(idx)
        score_text = format_score(scores)

        return img_a, img_b, prompt_text, mapping, scores, score_text

    def vote_internal(choice: str, mapping: dict | None, scores: dict):
        if mapping is None:
            raise gr.Error("No active test. Hit 'Test it!' first.")

        winner_model = mapping.get(choice)
        if winner_model not in ("base", "lora"):
            raise gr.Error("Invalid mapping state. Try 'Test it!' again.")

        # Update scores
        scores = dict(scores or {})
        scores.setdefault("n_trials", 0)
        scores.setdefault("lora_wins", 0)
        scores.setdefault("base_wins", 0)

        scores["n_trials"] += 1
        if winner_model == "lora":
            scores["lora_wins"] += 1
        else:
            scores["base_wins"] += 1

        score_text = format_score(scores)
        return scores, score_text

    def vote_A(mapping, scores):
        return vote_internal("A", mapping, scores)

    def vote_B(mapping, scores):
        return vote_internal("B", mapping, scores)

    # ----------- Build Gradio UI -----------

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # ES-EGGROLL × Sana Sprint – Demo

            ## Manual mode
            Generate images from **pre-encoded text embeddings** with:
            - **Base** SanaSprintOneStep,  
            - **ES-trained LoRA**, or  
            - **Both side-by-side** (same seed).
            """
        )

        # Shared state for blind test
        mapping_state = gr.State(None)  # holds {"A": "base"/"lora", "B": ..., "prompt_index": ..., "seed": ...}
        score_state = gr.State({"n_trials": 0, "lora_wins": 0, "base_wins": 0})

        # ----- Manual section -----
        with gr.Row():
            mode_radio = gr.Radio(
                choices=["base", "lora", "both"],
                value="lora" if lora_sana is not None else "base",
                label="Model",
                info="Use base Sana, ES-trained LoRA, or both.",
            )

            prompt_dropdown = gr.Dropdown(
                choices=choices,
                value=choices[0][1] if choices else 0,
                label="Prompt / embedding",
                info="These come from the loaded embedding .pt file",
            )

        with gr.Row():
            seed_slider = gr.Slider(
                minimum=0,
                maximum=10_000,
                value=0,
                step=1,
                label="Seed",
            )
            guidance_slider = gr.Slider(
                minimum=0.0,
                maximum=10.0,
                value=4.5,
                step=0.1,
                label="Guidance scale",
            )

        with gr.Row():
            width_latent = gr.Slider(
                minimum=8,
                maximum=64,
                value=32,
                step=1,
                label="Latent width",
                info="32 -> 1024px",
            )
            height_latent = gr.Slider(
                minimum=8,
                maximum=64,
                value=32,
                step=1,
                label="Latent height",
            )

        generate_btn = gr.Button("Generate")

        with gr.Row():
            base_image_out = gr.Image(label="Base", type="pil")
            lora_image_out = gr.Image(label="LoRA", type="pil")

        prompt_out = gr.Textbox(
            label="Prompt text",
            interactive=False,
        )

        generate_btn.click(
            fn=generate_fn,
            inputs=[
                mode_radio,
                prompt_dropdown,
                seed_slider,
                guidance_slider,
                width_latent,
                height_latent,
            ],
            outputs=[base_image_out, lora_image_out, prompt_out],
        )

        # ----- Blind test section -----
        gr.Markdown(
            """
            ---
            ## Blind A/B test

            Click **Test it!** to:
            - sample a random prompt + random seed  
            - generate **Base vs LoRA** in random A/B order  
            - choose which image is better and track who wins this session.
            """
        )

        test_btn = gr.Button("Test it! (random prompt & seed)")

        with gr.Row():
            test_image_a = gr.Image(label="Image A", type="pil")
            test_image_b = gr.Image(label="Image B", type="pil")

        test_prompt_out = gr.Textbox(
            label="Prompt text (for this test)",
            interactive=False,
        )

        with gr.Row():
            vote_a_btn = gr.Button("A is better")
            vote_b_btn = gr.Button("B is better")

        score_text = gr.Markdown(format_score({"n_trials": 0, "lora_wins": 0, "base_wins": 0}))

        # Test button: generate new pair + mapping, keep scores
        test_btn.click(
            fn=test_fn,
            inputs=[
                guidance_slider,
                width_latent,
                height_latent,
                score_state,
            ],
            outputs=[
                test_image_a,
                test_image_b,
                test_prompt_out,
                mapping_state,
                score_state,
                score_text,
            ],
        )

        # Voting buttons: update scores only
        vote_a_btn.click(
            fn=vote_A,
            inputs=[mapping_state, score_state],
            outputs=[score_state, score_text],
        )
        vote_b_btn.click(
            fn=vote_B,
            inputs=[mapping_state, score_state],
            outputs=[score_state, score_text],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoded_path",
        type=str,
        default="encoded_parti_prompts.pt",
        help="Path to .pt file with encoded prompts (prompt_embeds, attention_mask, prompts).",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="es_search_one_step_1/cfg0_sigma1e-02_lr1e+00_ant1/latest_lora",
        help="Directory with LoRA adapter weights (optional). If not set, only base model is available.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Launch Gradio with share=True (useful for quick demos).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    encoded_path = Path(args.encoded_path)
    if not encoded_path.is_file():
        raise SystemExit(f"encoded_path not found: {encoded_path}")

    lora_dir = Path(args.lora_dir) if args.lora_dir is not None else None
    if lora_dir is not None and not lora_dir.is_dir():
        raise SystemExit(f"lora_dir not found or not a directory: {lora_dir}")

    prompt_embeds_all, attention_mask_all, prompts_list = load_embeddings(encoded_path)
    base_sana, lora_sana = create_models(lora_dir)

    demo = build_interface(
        prompt_embeds_all=prompt_embeds_all,
        attention_mask_all=attention_mask_all,
        prompts_list=prompts_list,
        base_sana=base_sana,
        lora_sana=lora_sana,
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()