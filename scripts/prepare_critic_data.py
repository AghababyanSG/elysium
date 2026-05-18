"""Phase 7.1: build the SigLIP critic's training dataset.

Three classes of records, each ``{"image": PIL.Image, "instruction": str,
"label": int, "mode": str}``:

  * POSITIVE (label=1): canvas frames K=1..N from each annotation session,
    paired with the session's true instruction. K=0 (unedited input)
    is intentionally excluded — otherwise the critic learns "unedited
    inputs are plausible" and the model gets rewarded for not editing.

  * MISMATCHED negative (label=0): a positive frame paired with a
    randomly-drawn DIFFERENT instruction. Teaches the critic that
    (canvas, instruction) coupling matters.

  * NOISED negative (label=0): a positive frame with gaussian noise
    + random rectangular patches over it. Teaches "off-distribution
    pixels = implausible."

  * HARVESTED negative (label=0): intermediate frames from the §5.6
    sweep dirs ``outputs/comparison_v5p6_rl/<slug>_frames/`` for the
    prompts that hit ``max_chunks`` (doggy / girl_with_mole / mecedes /
    strawberry_heart). These are real model-failure outputs — high-
    value negatives because the critic will need to discriminate them
    from real human-edit trajectories.

Output: ``data/processed_critic/`` HuggingFace DatasetDict with
``{train, validation}`` splits, 80/20.

Usage:
    python scripts/prepare_critic_data.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from elysium.log import logger  # noqa: E402

SESSIONS_INSTR_PATH = ROOT / "configs/instructions.yaml"
FRAMES_DIR = ROOT / "data/raw/frames"
# Multiple sweep dirs harvested: each ran the §5.4.2 8-prompt set on a
# different checkpoint and the prompts that hit max_chunks left rich
# intermediate-frame trails of "model went off the rails."
HARVEST_DIRS = [
    ROOT / "outputs/comparison_v5p6_rl",
    ROOT / "outputs/comparison_v5p5p1_t05_max200",
]
OUT_DIR = ROOT / "data/processed_critic"

# Prompts where at least one sweep hit max_chunks (over-drew). Their
# intermediate frames are unambiguous failure-mode canvases for that
# instruction.
HARVEST_PROMPTS: dict[str, str] = {
    "doggy": "remove the dog from the image",
    "girl_with_mole": "remove the mole",
    "mecedes": "blur the car logo and plate",
    "strawberry_heart": "deform the strawberry to look like a heart, using warp tool",
    "cyan_clouds": "fill the canvas with cyan and draw clouds",
}

# Sample N evenly-spaced frames per (harvest_dir, prompt) pair (skip the
# first few where over-drawing hasn't kicked in yet; sample dense from
# mid-trajectory onward — those are the canvases that look most "broken").
HARVEST_PER_PROMPT = 200
HARVEST_SKIP_LEAD = 10  # first 10 frames are usually OK; skip them

NOISE_SIGMA = 0.10
NOISE_PATCH_FRACTION = 0.10  # 10% of area covered by random patches
NOISE_PATCH_COUNT_RANGE = (3, 8)

SEED = 42
TRAIN_SPLIT = 0.8


def _load_instructions() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(session -> instruction_text, task_name -> instruction_text)``."""
    with SESSIONS_INSTR_PATH.open() as f:
        cfg = yaml.safe_load(f)
    sess_to_inst: dict[str, str] = {}
    task_to_inst: dict[str, str] = {}
    for task_name, info in cfg.get("tasks", {}).items():
        instr = info["instruction"]
        task_to_inst[task_name] = instr
        for sess in info.get("sessions", []):
            sess_to_inst[sess] = instr
    return sess_to_inst, task_to_inst


def _collect_positives(sess_to_inst: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    for sess_dir in sorted(FRAMES_DIR.iterdir()):
        if not sess_dir.is_dir():
            continue
        sess_name = sess_dir.name
        if sess_name not in sess_to_inst:
            continue
        instr = sess_to_inst[sess_name]
        # Sort by integer stem so 10.jpg comes after 2.jpg, not after 1.jpg.
        frames = sorted(
            (p for p in sess_dir.iterdir() if p.suffix == ".jpg"),
            key=lambda p: int(p.stem),
        )
        # Skip frame 0 — the unedited input.
        for fp in frames[1:]:
            out.append({
                "image_path": str(fp),
                "instruction": instr,
                "label": 1,
                "mode": "positive",
                "source_session": sess_name,
            })
    logger.info("Collected {} positives from {} sessions",
                len(out), len({r["source_session"] for r in out}))
    return out


def _make_mismatched(positives: list[dict], task_to_inst: dict[str, str],
                     rng: random.Random) -> list[dict]:
    out: list[dict] = []
    all_instructions = list(set(task_to_inst.values()))
    for rec in positives:
        wrong = rec["instruction"]
        attempts = 0
        while wrong == rec["instruction"] and attempts < 10:
            wrong = rng.choice(all_instructions)
            attempts += 1
        if wrong == rec["instruction"]:
            continue  # only one instruction in the pool, can't mismatch
        out.append({
            "image_path": rec["image_path"],
            "instruction": wrong,
            "label": 0,
            "mode": "mismatched",
            "source_session": rec["source_session"],
        })
    logger.info("Generated {} mismatched negatives", len(out))
    return out


def _apply_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    noise = rng.gauss(0, NOISE_SIGMA) * np.ones_like(arr)
    arr = arr + np.random.RandomState(rng.randint(0, 2**31)).normal(
        0, NOISE_SIGMA, arr.shape
    )
    h, w = arr.shape[:2]
    n_patches = rng.randint(*NOISE_PATCH_COUNT_RANGE)
    target_area_per_patch = (h * w * NOISE_PATCH_FRACTION) / n_patches
    for _ in range(n_patches):
        side = int(target_area_per_patch ** 0.5)
        side = max(8, min(h // 3, side))
        x0 = rng.randint(0, w - side)
        y0 = rng.randint(0, h - side)
        colour = np.array(
            [rng.random(), rng.random(), rng.random()], dtype=np.float32
        )
        arr[y0:y0 + side, x0:x0 + side] = colour
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def _make_noised(positives: list[dict], rng: random.Random,
                 out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for i, rec in enumerate(positives):
        src = Image.open(rec["image_path"]).convert("RGB")
        noised = _apply_noise(src, rng)
        out_path = out_dir / f"noised_{i:05d}.jpg"
        noised.save(out_path, quality=88)
        out.append({
            "image_path": str(out_path),
            "instruction": rec["instruction"],
            "label": 0,
            "mode": "noised",
            "source_session": rec["source_session"],
        })
    logger.info("Generated {} noised negatives -> {}", len(out), out_dir)
    return out


def _make_harvested(rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for harvest_dir in HARVEST_DIRS:
        if not harvest_dir.exists():
            logger.warning("Harvest dir missing: {}", harvest_dir)
            continue
        for slug, instr in HARVEST_PROMPTS.items():
            frames_dir = harvest_dir / f"{slug}_frames"
            if not frames_dir.exists():
                continue
            frames = sorted(frames_dir.glob("frame_*.png"))
            if len(frames) <= HARVEST_SKIP_LEAD:
                continue
            candidates = frames[HARVEST_SKIP_LEAD:]
            if len(candidates) <= HARVEST_PER_PROMPT:
                picked = candidates
            else:
                idxs = np.linspace(0, len(candidates) - 1, HARVEST_PER_PROMPT,
                                   dtype=int)
                picked = [candidates[i] for i in idxs]
            for fp in picked:
                out.append({
                    "image_path": str(fp),
                    "instruction": instr,
                    "label": 0,
                    "mode": "harvested",
                    "source_session": f"harvest:{harvest_dir.name}/{slug}",
                })
    logger.info("Generated {} harvested negatives", len(out))
    return out


def main() -> None:
    rng = random.Random(SEED)
    sess_to_inst, task_to_inst = _load_instructions()

    positives = _collect_positives(sess_to_inst)
    mismatched = _make_mismatched(positives, task_to_inst, rng)
    noised_dir = OUT_DIR / "noised_imgs"
    noised = _make_noised(positives, rng, noised_dir)
    harvested = _make_harvested(rng)

    all_records = positives + mismatched + noised + harvested
    logger.info(
        "Total: {} ({} pos, {} mis, {} noi, {} har)",
        len(all_records), len(positives), len(mismatched),
        len(noised), len(harvested),
    )

    rng.shuffle(all_records)
    split_n = int(len(all_records) * TRAIN_SPLIT)
    train_records = all_records[:split_n]
    val_records = all_records[split_n:]

    from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value

    features = Features({
        "image": HFImage(),
        "instruction": Value("string"),
        "label": Value("int32"),
        "mode": Value("string"),
        "source_session": Value("string"),
    })

    def _to_hf(records: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "image": [r["image_path"] for r in records],
                "instruction": [r["instruction"] for r in records],
                "label": [r["label"] for r in records],
                "mode": [r["mode"] for r in records],
                "source_session": [r["source_session"] for r in records],
            },
            features=features,
        )

    ds = DatasetDict({"train": _to_hf(train_records), "validation": _to_hf(val_records)})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUT_DIR))
    logger.info("Saved DatasetDict to {}: {} train, {} val",
                OUT_DIR, len(train_records), len(val_records))

    # Per-mode label balance — sanity print.
    from collections import Counter
    train_by_mode = Counter(r["mode"] for r in train_records)
    val_by_mode = Counter(r["mode"] for r in val_records)
    logger.info("Train by mode: {}", dict(train_by_mode))
    logger.info("Val   by mode: {}", dict(val_by_mode))


if __name__ == "__main__":
    main()
