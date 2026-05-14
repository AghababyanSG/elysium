"""Inference pipeline: chunk-execute-reobserve loop.

Loads the fine-tuned Qwen3.5-VL model (LoRA adapter, via HuggingFace + PEFT),
then runs the following loop until completion or max_chunks is reached:

  1. Observe  -- convert current canvas ndarray to PIL image
  2. Predict  -- model generates a JSON action chunk
  3. Execute  -- canvas executor applies the actions sequentially
  4. Repeat   -- until all-noop chunk or max_chunks exceeded

Optionally uses temporal ensembling: execute only k < horizon actions before
re-observing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image
from transformers import StoppingCriteriaList

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.action_io import build_generation_processor_inputs, parse_action_chunk
from elysium.model.stop_on_json import JsonBalanceStoppingCriteria
from elysium.schemas.actions import ActionChunk, CANVAS_SIZE

__all__ = [
    "Predictor",
    "apply_image_pixel_budget",
    "cached_repo_ids",
    "ensure_rgb_canvas_size",
    "model_compute_dtype",
    "run_inference",
]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def cached_repo_ids() -> set[str]:
    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.exists():
        return set()
    return {repo.repo_id for repo in scan_cache_dir(cache_dir=cache_dir).repos}


def apply_image_pixel_budget(processor: Any, model_cfg: dict[str, Any]) -> None:
    """Cap the vision processor's pixel budget to control image-token count.

    Qwen3.5-VL's image processor stores ``size = {longest_edge, shortest_edge}``
    in pixel-area units. NOTE: as of Unsloth 2026.4.x, the vision data collator
    bypasses these settings and uses the model's default size — leaving this
    helper as a no-op when used with Unsloth. Kept for non-Unsloth processors
    and forward-compatibility.
    """
    img_min = model_cfg.get("image_min_pixels")
    img_max = model_cfg.get("image_max_pixels")
    if img_min is None and img_max is None:
        return
    image_processor = getattr(processor, "image_processor", None)
    assert image_processor is not None, "Processor has no image_processor; cannot set pixel budget"
    size = dict(getattr(image_processor, "size", {}) or {})
    if img_min is not None:
        size["shortest_edge"] = int(img_min)
    if img_max is not None:
        size["longest_edge"] = int(img_max)
    image_processor.size = size
    logger.info("Image processor size set to {}", size)


def model_compute_dtype(model: Any) -> torch.dtype:
    dtype_names = {
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    for candidate in (model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)):
        config = getattr(candidate, "config", None)
        dtype = getattr(config, "torch_dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str) and dtype in dtype_names:
            return dtype_names[dtype]
    for name, param in model.named_parameters():
        if "visual" in name and param.is_floating_point():
            return param.dtype
    for param in model.parameters():
        if param.is_floating_point() and param.dtype != torch.float32:
            return param.dtype
    return next(model.parameters()).dtype


def ensure_rgb_canvas_size(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size == (CANVAS_SIZE, CANVAS_SIZE):
        return rgb
    return rgb.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)


def _image_to_float32(image: Image.Image) -> np.ndarray:
    """PIL RGB image -> float32 ndarray [0, 1]."""
    return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def _float32_to_pil(canvas: np.ndarray) -> Image.Image:
    """float32 ndarray [0, 1] -> PIL RGB image."""
    return Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8), mode="RGB")


class Predictor:
    """Wraps the loaded model and runs the chunk-execute-reobserve loop.

    Attributes:
        model: Loaded Qwen3.5 model.
        processor: Corresponding processor/tokenizer.
        horizon: Number of actions per chunk.
        max_chunks: Maximum inference iterations before stopping.
        ensemble_k: Number of actions to execute before re-observing.
        max_new_tokens: Upper bound on assistant tokens per chunk decode.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        horizon: int = 5,
        max_chunks: int = 20,
        ensemble_k: int = 5,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        history_length: int = 0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.horizon = horizon
        self.max_chunks = max_chunks
        self.ensemble_k = min(ensemble_k, horizon)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.history_length = history_length

    @torch.inference_mode()  # type: ignore[misc]
    def _predict_chunk(
        self,
        canvas_pil: Image.Image,
        instruction: str,
        history_actions: list[str] | None = None,
    ) -> ActionChunk:
        """Run one model forward pass and parse the output into an ActionChunk.

        Generation params mirror the RL training distribution (sample at
        temperature 1.0, no repetition penalty) so the inference-time policy
        matches the policy that was actually rewarded. Greedy decoding is
        catastrophic on this small SFT dataset — the argmax path collapses to
        the noop chunk on step 0 and produces a blank canvas.
        """
        inputs = build_generation_processor_inputs(
            self.processor,
            canvas_pil,
            instruction,
            self.horizon,
            history_actions=history_actions,
        )
        dtype = model_compute_dtype(self.model)
        inputs = {
            k: v.to(device=self.model.device, dtype=dtype if v.is_floating_point() else None)
            for k, v in inputs.items()
            if v is not None
        }

        tok = self.processor.tokenizer
        prompt_len = inputs["input_ids"].shape[1]
        stop_crit = JsonBalanceStoppingCriteria(tokenizer=tok, prompt_len=prompt_len)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stop_crit]),
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.processor.decode(generated, skip_special_tokens=True)

        logger.info("Raw model output: {}", raw_output[:500])
        try:
            return parse_action_chunk(raw_output, self.horizon)
        except Exception as exc:
            logger.warning(
                "Failed to parse model output ({}): {} — using noop chunk",
                type(exc).__name__, exc,
            )
            return ActionChunk.noop_chunk(self.horizon)

    def run(
        self,
        image: Image.Image,
        instruction: str,
        show_preview: bool = False,
        frames_dir: Path | None = None,
    ) -> tuple[Image.Image, list[ActionChunk]]:
        """Run the full chunk-execute-reobserve loop.

        Args:
            image: Input image to edit.
            instruction: Natural language instruction.
            show_preview: If True, open a live preview window. Closing the window
                          stops inference early.
            frames_dir: If set, save the canvas after each chunk to this directory
                        as frame_0000.png, frame_0001.png, …

        Returns:
            Tuple of (final edited image, list of all ActionChunks executed).
        """
        from collections import deque

        image = ensure_rgb_canvas_size(image)
        original = _image_to_float32(image)
        canvas = original.copy()
        executed_chunks: list[ActionChunk] = []
        history: deque[str] = deque(maxlen=max(0, self.history_length))

        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
            _float32_to_pil(original).save(frames_dir / "frame_0000.png")
            logger.info("Saving frames to {}", frames_dir)

        fig = None
        im_handle = None
        if show_preview:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis("off")
            fig.canvas.manager.set_window_title("Elysium preview — close window to stop")
            im_handle = ax.imshow(image)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.001)

        for step in range(self.max_chunks):
            if show_preview and not plt.fignum_exists(fig.number):
                logger.info("Preview window closed — stopping inference")
                break

            canvas_pil = _float32_to_pil(canvas)
            chunk = self._predict_chunk(
                canvas_pil, instruction, history_actions=list(history) if history else None
            )

            if chunk.is_terminal:
                logger.info("Step {}: model signalled completion (all-noop chunk)", step)
                break

            if self.ensemble_k < self.horizon:
                partial_actions = chunk.actions[: self.ensemble_k]
                partial_chunk = ActionChunk(actions=partial_actions, horizon=self.ensemble_k)
                canvas = execute_chunk(canvas, partial_chunk, original=original)
                if self.history_length > 0:
                    history.append(partial_chunk.to_json_str())
            else:
                canvas = execute_chunk(canvas, chunk, original=original)
                if self.history_length > 0:
                    history.append(chunk.to_json_str())

            executed_chunks.append(chunk)
            logger.info("Step {}: executed chunk ({} actions)", step, len(chunk.actions))

            if frames_dir is not None:
                _float32_to_pil(canvas).save(frames_dir / f"frame_{step + 1:04d}.png")

            if show_preview:
                im_handle.set_data(_float32_to_pil(canvas))
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
        else:
            logger.info("Reached max_chunks={}, stopping", self.max_chunks)

        if show_preview:
            plt.ioff()
            plt.show(block=False)

        return _float32_to_pil(canvas), executed_chunks


def run_inference(
    image_path: Path,
    instruction: str,
    checkpoint_dir: Path,
    output_path: Path,
    config_path: Path = Path("configs/train.yaml"),
    show_preview: bool = False,
    frames_dir: Path | None = None,
) -> None:
    """Load model from checkpoint and run inference on an image.

    Args:
        image_path: Path to the input image.
        instruction: Natural language instruction (e.g. "Draw a mustache on the face").
        checkpoint_dir: Path to the saved LoRA adapter or merged model directory.
        output_path: Path to save the resulting image.
        config_path: Path to train.yaml config.
        show_preview: If True, open a live preview window updated after each chunk.
    """
    cfg = _load_config(config_path)
    infer_cfg = cfg.get("inference", {})
    data_cfg = cfg.get("data", {})
    horizon = data_cfg["action_horizon"]
    history_length = int(data_cfg.get("history_length", 0))
    max_chunks = infer_cfg.get("max_chunks", 20)
    ensemble_k = infer_cfg.get("ensemble_execute_k", horizon)
    max_new_tokens = int(infer_cfg.get("max_new_tokens", 4096))
    do_sample = bool(infer_cfg.get("do_sample", True))
    temperature = float(infer_cfg.get("temperature", 1.0))
    top_p = float(infer_cfg.get("top_p", 1.0))
    repetition_penalty = float(infer_cfg.get("repetition_penalty", 1.0))

    base_model = cfg["model"]["name"]
    local_only = base_model in cached_repo_ids()
    use_unsloth = bool(cfg["model"].get("use_unsloth", False))
    logger.info(
        "Loading model from {} (local_files_only={}, use_unsloth={})",
        checkpoint_dir, local_only, use_unsloth,
    )

    if use_unsloth:
        import os as _os
        _os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastVisionModel  # type: ignore

        model, processor = FastVisionModel.from_pretrained(
            model_name=str(checkpoint_dir),
            load_in_4bit=cfg["model"].get("load_in_4bit", True),
            local_files_only=local_only,
        )
        # Idempotent: SFT/RL checkpoint should already contain coord tokens.
        from elysium.model.coord_tokens import (
            add_coord_tokens as _add_coord,
            init_coord_token_embeddings as _init_coord,
        )
        base_tokenizer = getattr(processor, "tokenizer", processor)
        added = _add_coord(base_tokenizer)
        if added:
            try:
                model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(base_tokenizer))
            _init_coord(model, base_tokenizer)
            logger.warning(
                "Inference added {} coord tokens at load time -- checkpoint did "
                "not contain them.", added,
            )
        FastVisionModel.for_inference(model)
    else:
        # Local import to avoid circular import (loading imports predict for helpers).
        from elysium.model.loading import load_adapter_for_inference

        model, processor = load_adapter_for_inference(
            checkpoint_dir=checkpoint_dir,
            base_model_name=base_model,
            load_in_4bit=bool(cfg["model"].get("load_in_4bit", False)),
            local_only=local_only,
        )

    apply_image_pixel_budget(processor, cfg["model"])

    predictor = Predictor(
        model=model,
        processor=processor,
        horizon=horizon,
        max_chunks=max_chunks,
        ensemble_k=ensemble_k,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        history_length=history_length,
    )

    image = ensure_rgb_canvas_size(Image.open(image_path).convert("RGB"))
    logger.info("Running inference: '{}'", instruction)

    result_image, chunks = predictor.run(image, instruction, show_preview=show_preview, frames_dir=frames_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(str(output_path))
    logger.info("Saved result to {} ({} chunks executed)", output_path, len(chunks))


