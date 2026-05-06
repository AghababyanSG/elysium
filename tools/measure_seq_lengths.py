"""Measure tokenized sequence-length distribution over the processed dataset.

Loads the same processor used in training (Qwen2.5-VL via Unsloth's FastVisionModel),
runs `apply_chat_template` + processor on a sample of records (text + image), and
prints percentile statistics so you can pick `training.max_seq_length`.

Usage:
    python tools/measure_seq_lengths.py [--config configs/train.yaml]
                                        [--split train] [--sample 500]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import yaml
from datasets import load_from_disk
from PIL import Image

from elysium.log import logger
from elysium.model.action_io import (
    action_conversation_messages,
    apply_action_chat_template,
)
from elysium.model.predict import apply_image_pixel_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--sample", type=int, default=500,
                        help="Random sample size; use 0 for full dataset.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    model_name = cfg["model"]["name"]
    dataset_path = Path(cfg["data"]["dataset_path"])
    current_max = cfg["training"]["max_seq_length"]
    horizon = cfg["data"]["action_horizon"]

    from unsloth import FastVisionModel
    logger.info("Loading processor for {}", model_name)
    _, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=cfg["model"]["load_in_4bit"],
    )
    apply_image_pixel_budget(processor, cfg["model"])

    logger.info("Loading dataset from {}", dataset_path)
    ds = load_from_disk(str(dataset_path))[args.split]
    logger.info("{} split has {} records", args.split, len(ds))

    indices = list(range(len(ds)))
    if args.sample and args.sample < len(indices):
        random.seed(args.seed)
        indices = random.sample(indices, args.sample)

    def _extract_instruction(messages: list[dict]) -> str:
        for m in messages:
            if m["role"] == "user":
                for part in m["content"]:
                    if part.get("type") == "text":
                        return part["text"]
        raise ValueError("No user text found in messages")

    lengths: list[int] = []
    completion_lengths: list[int] = []
    for i in indices:
        rec = ds[i]
        instruction = _extract_instruction(rec["messages"])
        image = Image.open(rec["image"]).convert("RGB")

        prompt_msgs = action_conversation_messages(instruction, horizon)
        completion_msgs = [
            {"role": "assistant", "content": [{"type": "text", "text": rec["gt_actions"]}]}
        ]
        text_full = apply_action_chat_template(
            processor,
            prompt_msgs + completion_msgs,
            add_generation_prompt=False,
            continue_final_message=False,
        )
        full = processor(text=[text_full], images=[image], return_tensors="pt", padding=False)
        lengths.append(int(full["input_ids"].shape[-1]))

        completion_lengths.append(len(processor.tokenizer(rec["gt_actions"]).input_ids))

    arr = np.array(lengths)
    comp = np.array(completion_lengths)

    def pcts(a: np.ndarray) -> str:
        return (
            f"min={a.min()}  p50={int(np.percentile(a, 50))}  "
            f"p90={int(np.percentile(a, 90))}  p95={int(np.percentile(a, 95))}  "
            f"p99={int(np.percentile(a, 99))}  max={a.max()}  mean={a.mean():.0f}"
        )

    logger.info("--- full sequence length (system+image+instruction+assistant) ---")
    logger.info("{}", pcts(arr))
    logger.info("--- assistant-only token count (gt_actions) ---")
    logger.info("{}", pcts(comp))

    over = (arr > current_max).sum()
    logger.info(
        "Records exceeding current max_seq_length={}: {} / {} ({:.1%})",
        current_max, over, len(arr), over / len(arr),
    )

    for cand in (1024, 1280, 1536, 1792, 2048, 2560):
        cov = (arr <= cand).sum() / len(arr)
        logger.info("max_seq_length={:>4}  covers {:.2%} of sample", cand, cov)


if __name__ == "__main__":
    main()
