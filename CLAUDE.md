# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (after PyTorch with CUDA is installed separately)
pip install -r requirements.txt && pip install -e .

# Lint
ruff check src/ scripts/ tools/
ruff format src/ scripts/ tools/

# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_action_io.py

# Annotate training data
python tools/annotate.py data/raw/images/<image>.jpg

# Build training dataset
python scripts/prepare_data.py [--config configs/train.yaml] [--epsilon 2.0] [--instructions configs/instructions.yaml]

# Train. Defaults to SFT only. The data pipeline (prepare_data) is auto-run
# every launch unless --skip-prepare is passed.
python scripts/train.py                      # SFT only
python scripts/train.py --rl                 # RL only (requires existing SFT checkpoint)
python scripts/train.py --sft --rl           # SFT warmup then RL
python scripts/train.py --epochs N --batch-size N --skip-prepare

# Inference
python scripts/infer.py <image_path> "<instruction>" [--checkpoint models/checkpoints/final] [--output path.jpg] [--preview]
```

Ruff is configured in `pyproject.toml`: line length 100, rules E/F/I. Commit message format: `type(scope): description` (e.g. `feat(model): add RL checkpoint save`).

## Architecture

Elysium is a vision–language–action model (VLA) that edits images by predicting drawing commands rather than pixels. The model never touches pixels directly — it issues structured actions, and a deterministic executor renders them.

### Data pipeline (`src/elysium/data/`)

Three sequential stages called by `scripts/prepare_data.py` via `elysium.data.pipeline.run_pipeline`:

1. **Compress** (`compress.py`) — Groups raw mouse events from annotation sessions into logical strokes, then applies Ramer–Douglas–Peucker (RDP) to reduce each stroke's point count by 10–50x while preserving shape. Output: `data/interim/compressed/`.

2. **Chunk** (`chunk.py`) — Converts the flat list of compressed strokes into overlapping `(canvas_frame, actions[horizon])` pairs using a sliding window (stride 1, default horizon=5). Short sequences are noop-padded. Output: `data/interim/chunks/`.

3. **Format** (`format.py`) — Converts chunks into Qwen2.5-VL chat-template conversations and saves as a HuggingFace `DatasetDict` (80/20 split). Each record: user message = canvas image + instruction text; assistant message = JSON string of 5 actions. Output: `data/processed/`.

Sessions are registered in `configs/instructions.yaml` mapping task names → instruction strings → session IDs (each session ID matches a JSON file in `data/raw/sessions/`).

### Action schema (`src/elysium/schemas/actions.py`)

All actions are Pydantic models validated at construction. The union type `Action` covers: `brush`, `pencil`, `eraser`, `fill`, `color_adjust`, `text_overlay`, `gaussian_blur`, `clone_stamp`, `scatter_brush`, `pattern_brush`, `forward_warp`, `noop`. `ActionChunk` wraps a fixed-size list; `is_terminal` is true when all actions are noops (signals end of editing). `parse_action()` handles deserialization and coordinate clamping from model output. Canvas coordinates are in `[0, 511]` (CANVAS_SIZE=512); trajectories are capped at 64 points.

### Canvas executor (`src/elysium/engine/canvas.py`)

`execute_chunk(canvas, chunk, original)` applies each action in sequence to a `float32 (H, W, 3)` array in `[0, 1]`. Sparse control-point trajectories are interpolated with de Casteljau's Bézier algorithm before drawing. Eraser restores pixels from the `original` array. Returns a new array; never mutates the input.

### Training (`src/elysium/model/train.py`, `rl_train.py`)

- **SFT**: QLoRA fine-tuning of Qwen2.5-VL via Unsloth (`FastVisionModel`). Vision encoder is frozen; LoRA adapters (rank 16, alpha 32) attach to all language-model linear layers. Loss is masked to assistant tokens only. Saves a LoRA adapter to `models/checkpoints/final/`.
- **RL**: GRPO policy-gradient training (`GRPOTrainer`) using visual SSIM reward (`src/elysium/model/reward.py`). Loads the SFT checkpoint as the policy; reference policy uses `disable_adapter()` on the same PEFT model — no extra model copy needed. Config under `rl:` key in `configs/train.yaml`.

Key config in `configs/train.yaml`: `model.name`, `lora.*`, `training.*`, `data.action_horizon`, `inference.max_chunks`, `inference.ensemble_execute_k`. Reduce `training.max_seq_length` (e.g. 2048→1024) if OOM.

### Inference (`src/elysium/model/predict.py`)

`Predictor.run()` executes the observe–predict–execute loop:
1. Convert canvas ndarray to PIL.
2. Build processor inputs (`model/action_io.py`) and run model generation with `JsonBalanceStoppingCriteria` to stop cleanly after the JSON object closes.
3. Parse output into `ActionChunk` via `parse_action_chunk()`.
4. Execute with `execute_chunk()`.
5. Repeat until all-noop chunk or `max_chunks` reached.

Temporal ensembling: if `ensemble_execute_k < horizon`, only the first k actions execute before re-observing. Configured via `inference.ensemble_execute_k` in `configs/train.yaml`.

## Code rules (from `.cursorrules`)

**Fail fast** — Do not use `try/except` to swallow failures or return defaults. Let exceptions propagate; use `assert` for invariants. No broad `except Exception:` in project code.

**Logging** — Use `from elysium.log import logger` (Loguru) everywhere in `src/elysium/`, `scripts/`, and `tools/`. Never use stdlib `logging.getLogger` or `print()` for diagnostics. Use `{}` placeholders: `logger.info("Loading {}", path)`.

**Images** — Internal representation is `float32 (H, W, 3)` in `[0, 1]`, RGB. Do not mutate caller-owned arrays in-place. Use vectorized ops; avoid pixel-level Python loops.

**ML** — Wrap inference in `@torch.inference_mode()`. Do not reload large models on every call. Validate shape/dtype/range before model entry.

**Paths** — Use `pathlib.Path`; avoid `os.path` in new code.

Do not edit `unsloth_compiled_cache/` or other generated/vendor directories.
