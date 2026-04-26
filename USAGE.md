# Usage Guide

## Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop)
- CUDA 12.x drivers (match the PyTorch wheel you install, e.g. cu124)
- [Git LFS](https://git-lfs.com/) if the repo stores `data/` via LFS (see `.gitattributes`)

---

## Configuration files

| File | Role |
|------|------|
| **`configs/train.yaml`** | Main training config: base model (`model.name`), LoRA, optimizer, batching, `max_seq_length`, data paths (`data.*`), `train_split`, checkpoint dir, RL block (`rl`), and inference defaults used by `scripts/train.py`, `scripts/infer.py`, and `scripts/prepare_data.py` (via `--config`). |
| **`configs/instructions.yaml`** | Task definitions: natural-language **instruction** per task plus the **session** ids that link to JSON logs under `data/raw/sessions/`. Referenced from `train.yaml` (`data.instructions`). |

---

## Setup

### 1. Clone and enter the repo

```bash
git clone <repository-url> elysium
cd elysium
git lfs pull   # if data/ is tracked with Git LFS
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Activate this environment whenever you work on the project.

### 3. Install PyTorch with CUDA

Pick the wheel that matches your driver/CUDA from [PyTorch — Get Started](https://pytorch.org/get-started/locally/). Example for CUDA 12.4:

```bash
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True  and a CUDA version string (e.g. 12.4)
```

### 4. Install project dependencies

From the repo root (after PyTorch):

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the ML stack (Unsloth, TRL, Transformers, etc.) and tools (Pygame, OpenCV, `rdp`, Pydantic, …). If `unsloth` fails to resolve against your Torch/CUDA build, install a matching extra after reading the comment block at the top of `requirements.txt`, or follow [Unsloth’s install docs](https://github.com/unslothai/unsloth).

Optional (advanced): some setups use Unsloth from Git with a CUDA-torch extra, for example:

```bash
pip install "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

Only use that if the default `pip install -r requirements.txt` path does not work on your VM.

### 5. Hugging Face CLI (optional, for pre-downloading weights)

```bash
pip install "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-4B
```

The training and inference code also downloads the model on first use and caches it under `~/.cache/huggingface/`.

---

## Workflow

### Step 1 — Collect annotation data

Run the annotation tool for each new training session:

```bash
python tools/annotate.py data/raw/images/pirlo.jpg
```

More detail on the Pygame UI (shortcuts, tools) is in [`tools/README.md`](tools/README.md). The annotator and action schema evolve; if something is missing there, check `tools/annotate.py` and `src/elysium/schemas/actions.py`.

Each session saves:

- Canvas frames → `data/raw/frames/<session_stem>/<frame_id>.jpg` (one directory per session)
- Operation log → `data/raw/sessions/<session_stem>.json` with `session_stem = <image_stem>_<YYYYmmdd_HHMMSS>`

Repeat until you have **20–30 sessions per task** for meaningful training (rule of thumb).

### Step 2 — Register sessions in config

Open `configs/instructions.yaml` and add the new session ids under the correct task (each session id matches the JSON filename stem under `data/raw/sessions/`):

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

Optional flags: `--config <path>` (default `configs/train.yaml`), `--epsilon <float>` (RDP override), `--instructions <path>` (override `configs/instructions.yaml`).

This expects the full data pipeline implementation (`elysium.data.pipeline.run_pipeline`): compression → chunking → HuggingFace `DatasetDict` on disk. Intended layout:

1. Trajectory compression (RDP) → `data/interim/compressed/`
2. Action chunking (horizon from `configs/train.yaml`, sliding window) → `data/interim/chunks/`
3. Dataset assembly → `data/processed/`

If `prepare_data.py` fails with an import error for `elysium.data.pipeline`, your checkout is missing that package; align with the complete repo or restore `src/elysium/data/`.

Legacy frame layout (flat `data/raw/frames/<id>.jpg`) can be migrated with `python scripts/migrate_frames.py` if present in your tree.

### Step 4 — Download model and train

The base model id is set in `configs/train.yaml` under `model.name` (currently **`Qwen/Qwen3.5-4B`**). It is pulled from Hugging Face on first run when not already cached.

Launch SFT training (default when you run the script with no flags):

```bash
python scripts/train.py
```

REINFORCE RL (after an SFT checkpoint exists), optionally chained with SFT:

```bash
python scripts/train.py --rl                    # RL only (expects checkpoint, see script help)
python scripts/train.py --sft --rl              # SFT then RL
python scripts/train.py --help                  # --epochs, --batch-size, --sft-checkpoint, etc.
```

| Metric | Expected value |
|--------|----------------|
| VRAM usage | Depends on model and `max_seq_length`; reduce if OOM |
| Speed | Depends on GPU; order of minutes per epoch for small runs |
| Time (3 epochs, small dataset) | ~10–20 min (typical laptop GPU; varies) |

Checkpoints are written under `models/checkpoints/`. The SFT run saves a final adapter at `models/checkpoints/final/`.

> **OOM:** If training runs out of memory, lower `max_seq_length` in `configs/train.yaml` (e.g. from `2048` to `1024`).

### Step 5 — Run inference

```bash
python scripts/infer.py data/raw/images/pirlo.jpg "Draw a mustache on the face"
python scripts/infer.py data/raw/images/pirlo.jpg "Draw glasses on the face"
```

Results default to `outputs/<image_stem>_result.jpg`.

Options:

```
--checkpoint  Path to model checkpoint directory (default: models/checkpoints/final)
--output      Path to save result image          (default: outputs/<stem>_result.jpg)
--config      Path to config file                (default: configs/train.yaml)
--preview     Open a live preview window; close it to stop inference early
```

---

## Quick reference

```bash
# One-time setup (after clone + git lfs pull if needed)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .

# Every training cycle
python tools/annotate.py data/raw/images/pirlo.jpg   # 1. collect data
# edit configs/instructions.yaml                      # 2. register sessions
python scripts/prepare_data.py                        # 3. build dataset
python scripts/train.py                               # 4. fine-tune (add --rl as needed)
python scripts/infer.py data/raw/images/pirlo.jpg "Draw a mustache on the face"  # 5. infer
```

---

## Project structure

```
elysium/
├── configs/
│   ├── train.yaml          # Model, data paths, training + RL hyperparameters
│   └── instructions.yaml   # Task → session ids → instruction text
├── data/
│   ├── raw/                # Source data (often Git LFS)
│   │   ├── images/         #   Input images
│   │   ├── sessions/       #   Annotation logs (JSON)
│   │   └── frames/         #   Frames per session: frames/<session_stem>/*.jpg
│   ├── interim/            # Pipeline intermediates
│   │   ├── compressed/     #   RDP-compressed strokes
│   │   └── chunks/         #   Sliding-window action chunks
│   └── processed/          # HuggingFace DatasetDict on disk (train / validation)
├── models/
│   └── checkpoints/        # LoRA checkpoints + final/
├── outputs/                  # Inference outputs (gitignored by default)
├── scripts/
│   ├── prepare_data.py     # Full data pipeline (requires elysium.data.pipeline)
│   ├── train.py            # SFT and/or RL training
│   ├── infer.py            # Inference
│   ├── migrate_frames.py   # Migrate flat frames → per-session frame dirs
│   └── rescale_images.py   # Batch image rescale utility
├── src/elysium/
│   ├── engine/             # Deterministic canvas executor
│   ├── model/              # train, rl_train, predict, reward
│   └── schemas/            # Pydantic action types
└── tools/
    ├── annotate.py         # Pygame annotation tool
    └── to_jpg.py           # Image resize utility
```
