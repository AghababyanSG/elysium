# Elysium — Architecture and Technical Design

Elysium is a vision-language action model (VLA) that learns to perform image editing operations from human demonstrations. A user paints on an image using drawing tools, and the system learns to reproduce that kind of edit from a natural language instruction alone. The core idea is to treat image editing as a sequential decision-making problem: given a canvas state and an instruction, the model predicts the next batch of drawing actions.

This document explains every stage of the system, from raw data collection through training to inference.

---

## Mental Model

Traditional image editing requires pixel-level manipulation by hand. Elysium replaces the human hand with a fine-tuned vision-language model. The model sees the current canvas, reads an instruction like "Draw a mustache on the face", and outputs a sequence of structured drawing commands (brush strokes, pencil lines, fills, eraser passes) that a deterministic executor renders onto the canvas. The model never touches pixels directly — it issues tool commands, and the executor handles rendering.

This separation between prediction and execution is fundamental. The model's job is spatial reasoning and planning. The executor's job is pixel-perfect rendering. This means the model's output is interpretable (you can read exactly what strokes it planned), reproducible (same commands always produce the same image), and auditable (you can replay or modify the action sequence).

---

## Data Collection

Training data comes from a Pygame-based annotation tool where users paint on 256×256 images. The tool supports four instruments: brush (configurable size and color), pencil (1px strokes), eraser (restores original pixels), and flood fill. Every mouse event is logged at ~30fps as a raw operation record containing timestamp, frame ID, tool type, size, color, and start/end positions. Canvas snapshots are saved as JPEG frames whenever the canvas changes.

A single annotation session produces two artifacts: a directory of canvas frame images and a JSON log of hundreds of raw operations. These form the supervised training signal — the frames are observations, and the operations are the actions the model must learn to predict.

Sessions are registered in `configs/instructions.yaml`, which maps each session name to a natural language instruction (e.g., "Draw a mustache on the face"). Multiple sessions can share the same instruction to provide varied demonstrations of the same task.

---

## Data Pipeline

Raw annotation logs are far too dense for training. Hundreds of near-duplicate mouse events per stroke would overwhelm the model's context window and teach it nothing about spatial planning. The data pipeline has three stages that progressively refine the raw data into model-ready training samples.

### Trajectory Compression

The first stage groups consecutive same-tool operations into logical strokes and applies Ramer-Douglas-Peucker (RDP) compression to each stroke's coordinate sequence. RDP removes points that lie close to a straight line between their neighbors, controlled by an epsilon tolerance (default 2.0 pixels). A stroke of 200 raw mouse events might compress to 5-10 control points that capture the same shape. This reduces data volume by 10-50x while preserving the geometric intent.

### Action Chunking

The second stage converts the flat list of compressed strokes into overlapping (observation, actions) training pairs using a sliding window. Each pair contains a canvas frame captured before the first action in the window and a fixed-size chunk of actions (default horizon = 5). If fewer than 5 actions remain, the chunk is padded with noop (no-operation) actions. The sliding window with stride 1 maximizes the number of training samples: N strokes produce N chunks.

Chunking is critical because the model predicts a fixed number of actions per inference step. A horizon of 5 means the model sees one canvas frame and must plan 5 steps ahead. This balances prediction complexity (longer horizons require more planning) against re-observation frequency (shorter horizons allow more course correction).

### Dataset Assembly

The final stage converts all chunks into Qwen2.5-VL chat-template conversations and packages them as a HuggingFace Dataset. Each record has a user message containing the canvas image and instruction text, and an assistant message containing a JSON string with 5 structured actions. The dataset is split 80/20 into train and validation sets.

---

## Training

Training uses QLoRA (Quantized Low-Rank Adaptation) to fine-tune Qwen2.5-VL-3B-Instruct, a 3.8-billion-parameter vision-language model. The base model is loaded in 4-bit quantization to fit within 8GB VRAM, and a LoRA adapter (rank 16, alpha 32) is attached to all linear layers in the language model. The vision encoder is frozen entirely — it already understands images well; only the language layers learn to map visual observations and instructions to drawing actions.

The training objective is standard causal language modeling, but loss is masked to the assistant tokens only. The model learns to generate the JSON action output, not to predict the instruction or image tokens. This focuses all gradient signal on the actual task.

Key hyperparameters from the default config: batch size 1, gradient accumulation 4 (effective batch size 4), learning rate 2e-4 with warmup, AdamW 8-bit optimizer, and gradient checkpointing via Unsloth for memory efficiency. The SFT (Supervised Fine-Tuning) trainer from TRL handles the training loop.

The checkpoint saved after training is a LoRA adapter (~157MB), not a full model copy. At inference time, the adapter is merged with the base model weights.

---

## Inference

Inference runs a closed-loop cycle: observe, predict, execute, repeat.

1. **Observe** — The current canvas (initially the input image) is captured as a PIL image.
2. **Predict** — The model receives the canvas image and instruction, and generates a JSON string containing 5 structured actions.
3. **Execute** — A deterministic canvas executor parses the actions and renders them onto the canvas using OpenCV. Sparse control-point trajectories from the model are interpolated into smooth Bézier curves before drawing.
4. **Repeat** — The loop returns to step 1 with the updated canvas. It terminates when the model outputs an all-noop chunk (signaling completion) or the maximum chunk limit is reached (default 20).

The executor supports the same four tools as the annotation interface. Brush and pencil actions define an RGB color and a trajectory of control points. The executor interpolates these through de Casteljau's algorithm to produce smooth curves, then draws anti-aliased lines with OpenCV. Eraser actions restore the original image's pixels along the stroke path. Fill actions perform flood-fill from a seed point.

### Temporal Ensembling

The system supports temporal ensembling: instead of executing all 5 predicted actions before re-observing, it can execute only the first k actions (configured via `ensemble_execute_k`), then re-observe and re-predict. This creates overlapping predictions where later actions are predicted multiple times from increasingly refined canvas states. In practice with k < horizon, early actions benefit from higher confidence (predicted when the canvas was closer to the observed state) and later actions are implicitly averaged across multiple predictions.

---

## Action Schema

All actions are defined as Pydantic models with validation. Each action type enforces constraints: colors must be in [0, 255], stroke sizes in [1, 50], and trajectories must contain at least one point. The five action types — brush, pencil, eraser, fill, noop — cover the full editing vocabulary. The noop action is the termination signal: when the model predicts a chunk of all noops, inference stops.

This typed schema creates a formal contract between the model (which generates JSON) and the executor (which renders pixels). If the model produces malformed output, the parser falls back to a noop chunk rather than crashing.

---

## Key Design Decisions

**Why action prediction over pixel generation?** Pixel-level generation (like diffusion models) produces photorealistic output but gives no control over the editing process. Action prediction produces interpretable, editable, and reproducible edit sequences. The trade-off is that output quality depends on the executor's rendering fidelity.

**Why Qwen2.5-VL?** It's a capable vision-language model small enough (3B parameters) to fine-tune and run inference on a single consumer GPU in 4-bit quantization. Its chat template naturally accommodates the image-instruction-response format.

**Why fixed-horizon chunks?** Variable-length output would require the model to decide when to stop mid-generation, adding complexity. Fixed chunks with noop padding simplify both training (uniform sequence lengths) and inference (predictable compute per step).

**Why QLoRA over full fine-tuning?** VRAM. Full fine-tuning of a 3B model requires 24GB+. QLoRA with 4-bit quantization and rank-16 adapters fits in under 8GB while training only 1.08% of parameters. The vision encoder stays frozen because spatial understanding transfers well; only the language layers need task-specific adaptation.
