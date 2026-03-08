# Usage Guide

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop)
- CUDA 12.x drivers installed

---

## Setup

### 1. Create virtual environment

```bash
cd /home/spartak/Desktop/elysium
python3 -m venv .venv
source .venv/bin/activate
```

> Activate the environment before every session: `source .venv/bin/activate`

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True  12.4
```

### 3. Install dependencies

```bash
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets peft bitsandbytes transformers accelerate pyyaml rdp opencv-python Pillow
pip install -e .
```

---

## Workflow

### Step 1 — Collect annotation data

Run the annotation tool for each new training session:

```bash
python tools/annotate.py data/raw/images/pirlo.jpg
```

Each session saves:
- Canvas frames → `data/raw/frames/`
- Operation log → `data/raw/sessions/`

Repeat until you have **20–30 sessions per task** for meaningful training.

### Step 2 — Register sessions in config

Open `configs/instructions.yaml` and add the new session names under the correct task:

```yaml
tasks:
  mustache:
    instruction: "Draw a mustache on the face"
    sessions:
      - pirlo_20260302_012217
      - pirlo_20260302_012147
      - pirlo_20260302_XXXXXX   # add new sessions here
  glasses:
    instruction: "Draw glasses on the face"
    sessions:
      - pirlo_20260302_YYYYYY
```

### Step 3 — Build the training dataset

```bash
python scripts/prepare_data.py
```

This runs the full data pipeline:
1. Trajectory compression (RDP) → `data/interim/compressed/`
2. Action chunking (horizon=5, sliding window) → `data/interim/chunks/`
3. HuggingFace dataset assembly → `data/processed/`

### Step 4 — Download model and train

The model (`Qwen2.5-VL-3B-Instruct`) is downloaded automatically from HuggingFace on the first run (~6GB, cached to `~/.cache/huggingface/`).

To pre-download manually:

```bash
huggingface-cli download unsloth/Qwen2.5-VL-3B-Instruct
```

Launch training:

```bash
python scripts/train.py
```

| Metric | Expected value |
|--------|---------------|
| VRAM usage | ~6.5–7.5 GB |
| Speed | ~2–4 steps/min |
| Time (3 epochs, 50 samples) | ~10–20 min |

Checkpoints are saved to `models/checkpoints/`. The final model is at `models/checkpoints/final/`.

> **OOM fix:** If training crashes with an out-of-memory error, reduce `max_seq_length` in `configs/train.yaml` from `2048` to `1024`.

### Step 5 — Run inference

```bash
python scripts/infer.py data/raw/images/pirlo.jpg "Draw a mustache on the face"
python scripts/infer.py data/raw/images/pirlo.jpg "Draw glasses on the face"
```

Results are saved to `outputs/<image_stem>_result.jpg`.

Options:

```
--checkpoint  Path to model checkpoint (default: models/checkpoints/final)
--output      Path to save result image (default: outputs/<stem>_result.jpg)
--config      Path to config file      (default: configs/train.yaml)
```

---

## Quick Reference

```bash
# One-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install trl datasets peft bitsandbytes transformers accelerate pyyaml rdp opencv-python Pillow
pip install -e .

# Every training cycle
python tools/annotate.py data/raw/images/pirlo.jpg   # 1. collect data
# edit configs/instructions.yaml                      # 2. register sessions
python scripts/prepare_data.py                        # 3. build dataset
python scripts/train.py                               # 4. fine-tune
python scripts/infer.py data/raw/images/pirlo.jpg "Draw a mustache on the face"  # 5. infer
```

---

## Project Structure

```
elysium/
├── configs/
│   ├── train.yaml          # Training hyperparameters
│   └── instructions.yaml   # Task → session → instruction mapping
├── data/
│   ├── raw/                # Immutable source data
│   │   ├── images/         #   Input images
│   │   ├── sessions/       #   Raw annotation logs (JSON)
│   │   └── frames/         #   Canvas frame snapshots (JPG)
│   ├── interim/            # Pipeline intermediates
│   │   ├── compressed/     #   RDP-compressed strokes
│   │   └── chunks/         #   Sliding-window action chunks
│   └── processed/          # Final HuggingFace dataset
├── models/
│   └── checkpoints/        # Saved LoRA adapters + final model
├── outputs/                # Inference result images
├── scripts/
│   ├── prepare_data.py     # Run full data pipeline
│   ├── train.py            # Launch training
│   └── infer.py            # Run inference
├── src/elysium/            # Python package
│   ├── data/               #   compress, chunk, format
│   ├── engine/             #   Deterministic canvas executor
│   ├── model/              #   train, predict
│   └── schemas/            #   Pydantic action types
└── tools/
    ├── annotate.py         # Pygame annotation tool
    └── to_jpg.py           # Image resize utility
```
