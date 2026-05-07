"""Offline evaluation harness for the chunk-prediction policy.

Loads a checkpoint, samples N rows from the validation (or train) split, and
reports per-chunk metrics that catch the failure mode the training-reward
curve hides:

  - mean visual reward     (uses elysium.model.reward.visual_reward)
  - fraction of all-noop completions
  - mean completion length in tokens
  - per-row sample table (first K rows)

Run BEFORE and AFTER an RL training run. RL is only an improvement if it beats
the SFT checkpoint on mean reward without inflating noop fraction.

Usage:
    python scripts/eval_rl.py --checkpoint models/checkpoints/final
    python scripts/eval_rl.py --checkpoint models/checkpoints/rl_final --n 64
    python scripts/eval_rl.py --checkpoint models/checkpoints/final --split train --n 16
"""

from __future__ import annotations

import argparse
import os
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from PIL import Image
from pydantic import ValidationError
from transformers import StoppingCriteriaList
from unsloth import FastVisionModel

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.action_io import build_generation_processor_inputs, parse_action_chunk
from elysium.model.predict import (
    apply_image_pixel_budget,
    cached_repo_ids,
    ensure_rgb_canvas_size,
    model_compute_dtype,
)
from elysium.model.reward import visual_reward
from elysium.model.stop_on_json import JsonBalanceStoppingCriteria
from elysium.schemas.actions import ActionChunk


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _image_to_float32(pil: Image.Image) -> np.ndarray:
    return np.array(ensure_rgb_canvas_size(pil), dtype=np.float32) / 255.0


def _extract_instruction(row: dict[str, Any]) -> str:
    for msg in row["messages"]:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content.get("type") == "text" and content.get("text"):
                    return str(content["text"])
    raise AssertionError("Dataset sample is missing user instruction text")


@torch.inference_mode()
def _generate_chunk(
    model: Any,
    processor: Any,
    canvas_pil: Image.Image,
    instruction: str,
    horizon: int,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, int, ActionChunk | None]:
    """Generate one chunk; return (raw_text, completion_token_count, parsed_chunk_or_none)."""
    inputs = build_generation_processor_inputs(processor, canvas_pil, instruction, horizon)
    dtype = model_compute_dtype(model)
    inputs = {
        k: v.to(device=model.device, dtype=dtype if v.is_floating_point() else None)
        for k, v in inputs.items()
        if v is not None
    }

    tok = processor.tokenizer
    prompt_len = inputs["input_ids"].shape[1]
    stop_crit = JsonBalanceStoppingCriteria(tokenizer=tok, prompt_len=prompt_len)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.7 if do_sample else None,
        top_p=0.95 if do_sample else None,
        repetition_penalty=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop_crit]),
    )
    generated_ids = output_ids[0][prompt_len:]
    completion_tokens = int(generated_ids.shape[0])
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)

    try:
        chunk = parse_action_chunk(raw_text, horizon)
    except (JSONDecodeError, KeyError, ValueError, ValidationError, AssertionError):
        chunk = None
    return raw_text, completion_tokens, chunk


def _format_action_summary(chunk: ActionChunk | None) -> str:
    if chunk is None:
        return "PARSE_ERROR"
    types = [a.action_type for a in chunk.actions]
    return ",".join(types)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline eval for the chunk-prediction policy")
    p.add_argument("--checkpoint", type=Path, required=True, help="LoRA adapter dir")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--n", type=int, default=64, help="Number of rows to evaluate")
    p.add_argument("--seed", type=int, default=0, help="Subsample seed")
    p.add_argument("--show", type=int, default=8, help="Per-row rows to print")
    p.add_argument("--do-sample", action="store_true", help="Use sampling (default: greedy)")
    p.add_argument("--max-new-tokens", type=int, default=None, help="Override config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    horizon = cfg["data"]["action_horizon"]
    max_new_tokens = args.max_new_tokens or int(cfg.get("inference", {}).get("max_new_tokens", 384))

    logger.info("Loading checkpoint from {}", args.checkpoint)
    base_model = cfg["model"]["name"]
    local_only = base_model in cached_repo_ids()
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(args.checkpoint),
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        local_files_only=local_only,
    )
    apply_image_pixel_budget(processor, cfg["model"])
    FastVisionModel.for_inference(model)

    noop_json = ActionChunk.noop_chunk(horizon).to_json_str()
    noop_token_len = len(processor.tokenizer(noop_json, add_special_tokens=False)["input_ids"])
    logger.info("Noop-JSON token length: {}", noop_token_len)

    dataset_path = Path(cfg["data"]["dataset_path"])
    logger.info("Loading dataset from {} (split={})", dataset_path, args.split)
    raw = load_from_disk(str(dataset_path))[args.split]

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(raw))
    indices = rng.choice(len(raw), size=n, replace=False).tolist()
    logger.info("Evaluating {} rows (do_sample={})", n, args.do_sample)

    rewards: list[float] = []
    noop_flags: list[bool] = []
    parse_failures: int = 0
    completion_lens: list[int] = []
    sample_rows: list[tuple[int, str, str, str, float, int]] = []  # idx, instr, gt, pred, reward, len

    for i, idx in enumerate(indices):
        row = raw[int(idx)]
        instruction = _extract_instruction(row)
        image_field = row["image"]
        canvas_pil = image_field if isinstance(image_field, Image.Image) else Image.open(image_field)
        canvas_np = _image_to_float32(canvas_pil)

        raw_text, n_tok, pred_chunk = _generate_chunk(
            model, processor, canvas_pil, instruction, horizon, max_new_tokens, args.do_sample
        )
        completion_lens.append(n_tok)

        if pred_chunk is None:
            parse_failures += 1
            rewards.append(-1.0)
            noop_flags.append(False)
            continue

        noop_flags.append(pred_chunk.is_terminal)

        gt_chunk = ActionChunk.from_json_str(row["gt_actions"], horizon)
        if gt_chunk.is_terminal or not row.get("next_image"):
            # Terminal GT row — visual_reward asserts non-terminal, so skip scoring
            # but still record noop-fraction and length above.
            continue

        gt_target = execute_chunk(canvas_np, gt_chunk, original=canvas_np)
        diff = np.abs(gt_target - canvas_np).max(axis=2)
        if int((diff > 0.01).sum()) < 50:
            continue  # near-terminal — same reasoning

        predicted = execute_chunk(canvas_np, pred_chunk, original=canvas_np)
        r = visual_reward(predicted, gt_target, canvas_np)
        rewards.append(r)

        if i < args.show:
            sample_rows.append((
                int(idx), instruction[:60], _format_action_summary(gt_chunk),
                _format_action_summary(pred_chunk), r, n_tok,
            ))

    n_eval = len(rewards)
    n_noop = sum(noop_flags)
    logger.info("=" * 70)
    logger.info("Checkpoint:        {}", args.checkpoint)
    logger.info("Split / N:         {} / {}", args.split, n)
    logger.info("Mean reward:       {:.4f}  (over {} scoreable rows)", float(np.mean(rewards)) if rewards else float("nan"), n_eval)
    logger.info("Noop fraction:     {:.3f}  ({}/{})", n_noop / max(n, 1), n_noop, n)
    logger.info("Mean compl. len:   {:.1f} tokens  (noop-len={})", float(np.mean(completion_lens)) if completion_lens else 0.0, noop_token_len)
    logger.info("Parse failures:    {}/{}", parse_failures, n)
    logger.info("=" * 70)
    if sample_rows:
        logger.info("Sample completions (idx, instr, gt_types, pred_types, reward, tok):")
        for idx, instr, gt, pred, r, n_tok in sample_rows:
            logger.info("  [{:>5}] '{}' | gt={} | pred={} | r={:+.3f} | tok={}", idx, instr, gt, pred, r, n_tok)


if __name__ == "__main__":
    main()
