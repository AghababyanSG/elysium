"""Inference pipeline: chunk-execute-reobserve loop.

Loads the fine-tuned Qwen3.5 model (merged LoRA adapter), then runs the
following loop until completion or max_chunks is reached:

  1. Observe  -- convert current canvas ndarray to PIL image
  2. Predict  -- model generates a JSON 5-action chunk
  3. Execute  -- canvas executor applies all 5 actions sequentially
  4. Repeat   -- until all-noop chunk or max_chunks exceeded

Optionally uses temporal ensembling: execute only k < horizon actions before
re-observing, averaging overlapping predictions for smoother transitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import numpy as np
import torch
import yaml
from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from PIL import Image
from unsloth import FastVisionModel

from transformers import StoppingCriteriaList

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.action_io import build_generation_processor_inputs, parse_action_chunk
from elysium.model.stop_on_json import JsonBalanceStoppingCriteria
from elysium.schemas.actions import ActionChunk, CANVAS_SIZE

__all__ = ["Predictor", "cached_repo_ids", "ensure_rgb_canvas_size", "model_compute_dtype", "run_inference"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def cached_repo_ids() -> set[str]:
    cache_dir = Path(HF_HUB_CACHE)
    if not cache_dir.exists():
        return set()
    return {repo.repo_id for repo in scan_cache_dir(cache_dir=cache_dir).repos}


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
    ) -> None:
        self.model = model
        self.processor = processor
        self.horizon = horizon
        self.max_chunks = max_chunks
        self.ensemble_k = min(ensemble_k, horizon)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()  # type: ignore[misc]
    def _predict_chunk(
        self, canvas_pil: Image.Image, instruction: str, do_sample: bool = True
    ) -> ActionChunk:
        """Run one model forward pass and parse the output into an ActionChunk.

        Args:
            canvas_pil: Current canvas as a PIL image.
            instruction: Natural language editing instruction.
            do_sample: If True, use temperature sampling. Default True — greedy
                decoding is prone to falling into degenerate trajectory loops on
                a small SFT dataset.

        Returns:
            Parsed ActionChunk.
        """
        inputs = build_generation_processor_inputs(self.processor, canvas_pil, instruction)
        dtype = model_compute_dtype(self.model)
        inputs = {
            k: v.to(device=self.model.device, dtype=dtype if v.is_floating_point() else None)
            for k, v in inputs.items()
            if v is not None
        }

        tok = self.processor.tokenizer
        prompt_len = inputs["input_ids"].shape[1]
        stop_crit = JsonBalanceStoppingCriteria(tokenizer=tok, prompt_len=prompt_len)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=0.3 if do_sample else None,
            top_p=0.9 if do_sample else None,
            repetition_penalty=1.15,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stop_crit]),
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_output = self.processor.decode(generated, skip_special_tokens=True)

        logger.debug("Raw model output: {}", raw_output[:500])
        try:
            return parse_action_chunk(raw_output, self.horizon)
        except Exception as exc:
            logger.warning("Failed to parse model output ({}): {} — using noop chunk", type(exc).__name__, exc)
            return ActionChunk.noop_chunk(self.horizon)

    def run(
        self,
        image: Image.Image,
        instruction: str,
        show_preview: bool = False,
    ) -> tuple[Image.Image, list[ActionChunk]]:
        """Run the full chunk-execute-reobserve loop.

        Args:
            image: Input image to edit.
            instruction: Natural language instruction.
            show_preview: If True, open a live preview window. Closing the window
                          stops inference early.

        Returns:
            Tuple of (final edited image, list of all ActionChunks executed).
        """
        image = ensure_rgb_canvas_size(image)
        original = _image_to_float32(image)
        canvas = original.copy()
        executed_chunks: list[ActionChunk] = []

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
            chunk = self._predict_chunk(canvas_pil, instruction, do_sample=False)

            if chunk.is_terminal:
                logger.info("Step {}: model signalled completion (all-noop chunk)", step)
                break

            if self.ensemble_k < self.horizon:
                partial_actions = chunk.actions[: self.ensemble_k]
                partial_chunk = ActionChunk(actions=partial_actions, horizon=self.ensemble_k)
                canvas = execute_chunk(canvas, partial_chunk, original=original)
            else:
                canvas = execute_chunk(canvas, chunk, original=original)

            executed_chunks.append(chunk)
            logger.info("Step {}: executed chunk ({} actions)", step, len(chunk.actions))

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
    horizon = cfg["data"]["action_horizon"]
    max_chunks = infer_cfg.get("max_chunks", 20)
    ensemble_k = infer_cfg.get("ensemble_execute_k", horizon)
    max_new_tokens = int(infer_cfg.get("max_new_tokens", 4096))

    base_model = cfg["model"]["name"]
    local_only = base_model in cached_repo_ids()
    logger.info("Loading model from {} (local_files_only={})", checkpoint_dir, local_only)
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(checkpoint_dir),
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        local_files_only=local_only,
    )
    FastVisionModel.for_inference(model)

    predictor = Predictor(
        model=model,
        processor=processor,
        horizon=horizon,
        max_chunks=max_chunks,
        ensemble_k=ensemble_k,
        max_new_tokens=max_new_tokens,
    )

    image = ensure_rgb_canvas_size(Image.open(image_path).convert("RGB"))
    logger.info("Running inference: '{}'", instruction)

    result_image, chunks = predictor.run(image, instruction, show_preview=show_preview)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(str(output_path))
    logger.info("Saved result to {} ({} chunks executed)", output_path, len(chunks))


