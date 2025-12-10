#!/usr/bin/env python
"""
Benchmark generated images against PartiPrompts using CLIP + PickScore rewards.

Inputs:
  - A folder of images generated from PartiPrompts, named like:
        0000_some_prompt_slug.png
        0001_other_prompt_slug.png
        ...
    where the leading zero-padded integer is the prompt index.
  - The PartiPrompts TSV file, which contains at least:
        Prompt, Category, Challenge
    (column names matched case-insensitively).

For each image:
  - Look up the corresponding prompt row by its index.
  - Compute:
      * clip_aesthetic
      * clip_text
      * no_artifacts
      * pickscore
      * combined  (weighted mix)
  - Aggregate:
      * Global totals
      * Per-Category averages
      * Per-Challenge averages

Usage example:

  python benchmark_partiprompts.py \
      --images_dir outputs_partiprompts_base \
      --tsv_path PartiPrompts.tsv

  python benchmark_partiprompts.py \
      --images_dir outputs_partiprompts_lora \
      --tsv_path PartiPrompts.tsv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import torch
from PIL import Image
from tqdm.auto import tqdm

from rewards import (
    load_clip_model_and_processor,
    load_pickscore_model_and_processor,
    compute_all_rewards,
)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Default mix weights for combined score: (aesthetic, text_align, no_artifacts, pickscore)
DEFAULT_MIX_WEIGHTS = (0.3, 0.3, 0.2, 0.2)


def find_col_name(fieldnames, target: str) -> str:
    """
    Find column name in TSV header matching target (case-insensitive).
    Raises ValueError if not found.
    """
    target_lower = target.lower()
    for name in fieldnames:
        if name.lower() == target_lower:
            return name
    raise ValueError(f"Could not find column '{target}' (case-insensitive) in TSV header: {fieldnames}")


def parse_index_from_filename(name: str) -> int:
    """
    Extract the leading integer index from a filename like '0001_something.png'.

    Returns:
        int index

    Raises:
        ValueError if no leading integer is found.
    """
    m = re.match(r"^(\d+)_", name)
    if not m:
        raise ValueError(f"Filename does not start with an index + underscore: {name}")
    return int(m.group(1))


def update_stats(agg: Dict[str, Dict[str, float]], key: str, metrics: Dict[str, float]):
    """
    Update aggregation dict:

        agg[key] = {
            "count": int,
            "sum_clip_aesthetic": float,
            "sum_clip_text": float,
            "sum_no_artifacts": float,
            "sum_pickscore": float,
            "sum_combined": float,
        }
    """
    if key not in agg:
        agg[key] = {
            "count": 0,
            "sum_clip_aesthetic": 0.0,
            "sum_clip_text": 0.0,
            "sum_no_artifacts": 0.0,
            "sum_pickscore": 0.0,
            "sum_combined": 0.0,
        }

    agg[key]["count"] += 1
    agg[key]["sum_clip_aesthetic"] += metrics["clip_aesthetic"]
    agg[key]["sum_clip_text"] += metrics["clip_text"]
    agg[key]["sum_no_artifacts"] += metrics["no_artifacts"]
    agg[key]["sum_pickscore"] += metrics["pickscore"]
    agg[key]["sum_combined"] += metrics["combined"]


def print_group_stats(title: str, agg: Dict[str, Dict[str, float]]):
    """
    Pretty-print averaged metrics per group (Category or Challenge).
    """
    print(f"\n=== {title} ===")
    keys_sorted = sorted(agg.keys())
    for key in keys_sorted:
        st = agg[key]
        c = st["count"]
        if c == 0:
            continue
        aes_mean = st["sum_clip_aesthetic"] / c
        txt_mean = st["sum_clip_text"] / c
        noart_mean = st["sum_no_artifacts"] / c
        pick_mean = st["sum_pickscore"] / c
        comb_mean = st["sum_combined"] / c
        print(
            f"- {key} (n={c:4d}): "
            f"aesthetic={aes_mean:.4f}, "
            f"text={txt_mean:.4f}, "
            f"no_artifacts={noart_mean:.4f}, "
            f"pickscore={pick_mean:.4f}, "
            f"combined={comb_mean:.4f}"
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default="outputs_partiprompts_base",
        help="Folder containing generated images (one per PartiPrompt).",
    )
    parser.add_argument(
        "--tsv_path",
        type=str,
        default="PartiPrompt.tsv",
        help="Path to PartiPrompts TSV file (with Prompt / Category / Challenge).",
    )
    parser.add_argument(
        "--mix_aes",
        type=float,
        default=DEFAULT_MIX_WEIGHTS[0],
        help="Weight for aesthetic CLIP similarity in combined score.",
    )
    parser.add_argument(
        "--mix_txt",
        type=float,
        default=DEFAULT_MIX_WEIGHTS[1],
        help="Weight for text alignment CLIP similarity in combined score.",
    )
    parser.add_argument(
        "--mix_noart",
        type=float,
        default=DEFAULT_MIX_WEIGHTS[2],
        help="Weight for no_artifacts score in combined score.",
    )
    parser.add_argument(
        "--mix_pick",
        type=float,
        default=DEFAULT_MIX_WEIGHTS[3],
        help="Weight for PickScore in combined score.",
    )
    args = parser.parse_args()

    mix_weights = (args.mix_aes, args.mix_txt, args.mix_noart, args.mix_pick)
    images_dir = Path(args.images_dir)
    tsv_path = Path(args.tsv_path)

    print(f"[init] Using device: {DEVICE}")
    print(f"[init] Images dir: {images_dir}")
    print(f"[init] TSV path:   {tsv_path}")
    print(f"[init] Mix weights (aes, txt, noart, pick): {mix_weights}")

    # ----------------------------------------------------
    # Load PartiPrompts TSV
    # ----------------------------------------------------
    print("[init] Loading PartiPrompts TSV...")
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("TSV file appears to have no header / fieldnames.")

        prompt_col = find_col_name(fieldnames, "prompt")
        category_col = find_col_name(fieldnames, "category")
        challenge_col = find_col_name(fieldnames, "challenge")

        rows = list(reader)

    num_rows = len(rows)
    print(f"[init] Loaded {num_rows} TSV rows.")
    print(f"       Columns: {fieldnames}")
    print(f"       Using columns -> Prompt: '{prompt_col}', Category: '{category_col}', Challenge: '{challenge_col}'")

    # ----------------------------------------------------
    # Index images by prompt index
    # ----------------------------------------------------
    print("[init] Scanning image folder...")
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    print(f"[init] Found {len(image_paths)} image files.")

    # dict: prompt_idx -> list[Path]
    images_by_idx = defaultdict(list)

    for p in image_paths:
        try:
            idx = parse_index_from_filename(p.name)
            images_by_idx[idx].append(p)
        except ValueError:
            # ignore files that don't match the pattern
            print(f"  [warn] Skipping file without index prefix: {p.name}")

    if not images_by_idx:
        print("[error] No valid images with index prefix found. Exiting.")
        return

    # ----------------------------------------------------
    # Load models once
    # ----------------------------------------------------
    print("[init] Loading CLIP model + processor...")
    clip_model, clip_processor = load_clip_model_and_processor(device=DEVICE)
    print("[init] Loading PickScore model + processor...")
    pick_model, pick_processor = load_pickscore_model_and_processor(device=DEVICE)

    # ----------------------------------------------------
    # Aggregation structures
    # ----------------------------------------------------
    overall = {
        "count": 0,
        "sum_clip_aesthetic": 0.0,
        "sum_clip_text": 0.0,
        "sum_no_artifacts": 0.0,
        "sum_pickscore": 0.0,
        "sum_combined": 0.0,
    }
    by_category: Dict[str, Dict[str, float]] = {}
    by_challenge: Dict[str, Dict[str, float]] = {}

    # ----------------------------------------------------
    # Main loop over images / prompts
    # ----------------------------------------------------
    print("[run] Evaluating images...")
    # We'll iterate over indices in sorted order for reproducibility
    valid_indices = sorted(images_by_idx.keys())

    for idx in tqdm(valid_indices, desc="Benchmarking", total=len(valid_indices)):
        if idx < 0 or idx >= num_rows:
            print(f"[warn] Image index {idx} exceeds TSV rows ({num_rows}), skipping.")
            continue

        row = rows[idx]
        prompt_text = row[prompt_col]
        category = row[category_col]
        challenge = row[challenge_col]

        # Load all images for this index (usually 1)
        img_paths = images_by_idx[idx]
        images = []
        for p in img_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[warn] Failed to open image {p}: {e}")

        if not images:
            print(f"[warn] No valid images for index {idx}, skipping.")
            continue

        rewards = compute_all_rewards(
            images,
            prompt_text=prompt_text,
            clip_model=clip_model,
            clip_processor=clip_processor,
            mix_weights=mix_weights,
            pickscore_model=pick_model,
            pickscore_processor=pick_processor,
        )

        # Convert tensors to plain floats
        metrics = {
            "clip_aesthetic": float(rewards["clip_aesthetic"]),
            "clip_text": float(rewards["clip_text"]),
            "no_artifacts": float(rewards["no_artifacts"]),
            "pickscore": float(rewards["pickscore"]),
            "combined": float(rewards["combined"]),
        }

        # Update global stats
        overall["count"] += 1
        overall["sum_clip_aesthetic"] += metrics["clip_aesthetic"]
        overall["sum_clip_text"] += metrics["clip_text"]
        overall["sum_no_artifacts"] += metrics["no_artifacts"]
        overall["sum_pickscore"] += metrics["pickscore"]
        overall["sum_combined"] += metrics["combined"]

        # Update Category / Challenge stats
        update_stats(by_category, category, metrics)
        update_stats(by_challenge, challenge, metrics)

    # ----------------------------------------------------
    # Print results
    # ----------------------------------------------------
    print("\n================= BENCHMARK RESULTS =================")

    # Overall
    c = overall["count"]
    if c == 0:
        print("No valid images were evaluated. Nothing to report.")
        return

    aes_mean = overall["sum_clip_aesthetic"] / c
    txt_mean = overall["sum_clip_text"] / c
    noart_mean = overall["sum_no_artifacts"] / c
    pick_mean = overall["sum_pickscore"] / c
    comb_mean = overall["sum_combined"] / c

    print("\n=== OVERALL ===")
    print(f"Total images evaluated: {c}")
    print(f"  aesthetic_mean   = {aes_mean:.4f}")
    print(f"  text_mean        = {txt_mean:.4f}")
    print(f"  no_artifacts_mean= {noart_mean:.4f}")
    print(f"  pickscore_mean   = {pick_mean:.4f}")
    print(f"  combined_mean    = {comb_mean:.4f}")

    # Per-category
    print_group_stats("Per Category", by_category)

    # Per-challenge
    print_group_stats("Per Challenge", by_challenge)

    print("\n[done] Benchmark completed.")


if __name__ == "__main__":
    main()