"""GRPO policy-gradient training for the chunk-prediction model.

Uses TRL's GRPOTrainer with a pure visual (SSIM) reward.  The SFT checkpoint
is loaded with its LoRA adapters as the policy; the frozen reference policy is
obtained via disable_adapter() on the same PEFT model — no extra copy needed.

Config: configs/train.yaml  (rl section)
Data:   data/processed/     (HuggingFace DatasetDict; uses gt_actions to
                             synthesize the per-prompt reward target)
Output: models/checkpoints/rl_final/
"""

from __future__ import annotations

import datetime
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
from datasets import Dataset, Features
from datasets import Image as HFImage
from datasets import Value
from datasets import load_from_disk
from PIL import Image
from pydantic import ValidationError
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.action_io import (
    action_conversation_messages,
    apply_action_chat_template,
    parse_action_chunk,
)
from elysium.model.coord_tokens import add_coord_tokens, init_coord_token_embeddings
from elysium.model.predict import apply_image_pixel_budget, cached_repo_ids, ensure_rgb_canvas_size
from elysium.model.reward import visual_reward
from elysium.schemas.actions import ActionChunk

__all__ = ["run_rl_training"]


class _AssertParamsFinite(TrainerCallback):
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        for name, p in model.named_parameters():
            if p.requires_grad and not torch.isfinite(p).all():
                raise AssertionError(
                    f"Non-finite parameter at step {state.global_step}: {name}"
                )


class _DetectCollapse(TrainerCallback):
    """Abort training if the policy collapses to noop / wrong-answer output.

    True collapse with the coverage-weighted reward looks like:
      - very short completions, AND
      - sustained near-zero mean reward.

    A short completion with HIGH reward is fine — that's perfect imitation
    (e.g. `gaussian_blur radius=2` exactly matching GT). Don't false-alarm
    on those.
    """

    def __init__(self, noop_token_len: int, warmup_steps: int = 50, window: int = 20) -> None:
        # Trained completions include template suffix + EOS, so actual noop
        # completions run ~2x the bare-JSON token count (observed ~29 vs 14
        # measured for horizon=2). Use 2x with a +5 cushion as the threshold.
        self.noop_threshold = max(2 * noop_token_len + 5, noop_token_len + 15)
        self.noop_len = noop_token_len
        self.warmup = warmup_steps
        self.window = window
        self._collapsed_logs = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step < self.warmup:
            return
        mean_len = logs.get("completions/mean_length")
        mean_reward = logs.get("rewards/visual_reward_fn/mean", logs.get("reward"))
        if mean_len is None or mean_reward is None:
            return
        is_noop_length = mean_len <= self.noop_threshold
        is_low_reward = mean_reward < 0.05
        if is_noop_length and is_low_reward:
            self._collapsed_logs += 1
        else:
            self._collapsed_logs = 0
        if self._collapsed_logs >= self.window:
            raise RuntimeError(
                f"RL collapsed to degenerate short low-reward output "
                f"(step {state.global_step}): mean_len={mean_len:.1f} ≤ "
                f"{self.noop_threshold:.0f}, mean_reward={mean_reward:.3f} for "
                f"{self.window} consecutive logs."
            )


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _image_to_float32(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"), dtype=np.float32) / 255.0


def _extract_instruction(row: dict[str, Any]) -> str:
    for msg in row["messages"]:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content.get("type") == "text" and content.get("text"):
                    return str(content["text"])
    raise AssertionError("Dataset sample is missing user instruction text")


_MIN_GT_PIXELS_FOR_RL_DEFAULT = 50  # mirrors reward._MIN_GT_PIXELS lower bound


def _build_grpo_dataset(
    train_data: Any,
    processor: Any,
    horizon: int,
    min_gt_pixels: int = _MIN_GT_PIXELS_FOR_RL_DEFAULT,
) -> Dataset:
    """Build the GRPO dataset, dropping terminal and near-terminal rows.

    RL must not see rows where the optimal answer is "do nothing" — those create
    a degenerate noop attractor (see plans/merry-scribbling-acorn.md). Filter on
    three signals:
      1. `next_image == ""` — structural marker for the last chunk in a session
      2. `ActionChunk.is_terminal` — all-noop GT chunk
      3. executing GT changes < `min_gt_pixels` pixels — must be at least
         reward._MIN_GT_PIXELS (=50) so the reward assertion holds; setting it
         higher trims the saturated tail where SFT already nails the GT and the
         GRPO group collapses to zero reward variance.
    """
    assert min_gt_pixels >= _MIN_GT_PIXELS_FOR_RL_DEFAULT, (
        f"min_gt_pixels={min_gt_pixels} is below reward._MIN_GT_PIXELS={_MIN_GT_PIXELS_FOR_RL_DEFAULT}; "
        "the reward function will assert."
    )
    prompts: list[str] = []
    images: list[Any] = []
    gt_actions: list[str] = []
    dropped_terminal = 0
    dropped_near_terminal = 0

    for row in train_data:
        if not row.get("next_image"):
            dropped_terminal += 1
            continue
        gt_chunk = ActionChunk.from_json_str(row["gt_actions"], horizon)
        if gt_chunk.is_terminal:
            dropped_terminal += 1
            continue
        image_field = row["image"]
        pil = image_field if isinstance(image_field, Image.Image) else Image.open(image_field)
        # Must match the reward's image path exactly — the reward operates on
        # CANVAS_SIZE (256) not the raw 512x512, so a row with >50 changed
        # pixels at native res can drop below 50 after downsampling.
        canvas_np = _image_to_float32(ensure_rgb_canvas_size(pil))
        gt_target = execute_chunk(canvas_np, gt_chunk, original=canvas_np)
        diff = np.abs(gt_target - canvas_np).max(axis=2)
        if int((diff > 0.01).sum()) < min_gt_pixels:
            dropped_near_terminal += 1
            continue

        instruction = _extract_instruction(row)
        messages = action_conversation_messages(instruction, horizon)
        prompt_text = apply_action_chat_template(processor, messages, add_generation_prompt=True)
        prompts.append(prompt_text)
        images.append(row["image"])
        gt_actions.append(row["gt_actions"])

    total = len(train_data)
    kept = len(prompts)
    logger.info(
        "GRPO dataset filter: kept {}/{} ({:.1f}%); dropped {} terminal, {} near-terminal",
        kept, total, 100.0 * kept / max(total, 1), dropped_terminal, dropped_near_terminal,
    )
    assert kept > 0, "GRPO dataset is empty after filtering — check data/processed/"

    return Dataset.from_dict(
        {"prompt": prompts, "image": images, "gt_actions": gt_actions},
        features=Features({
            "prompt": Value("string"),
            "image": HFImage(),
            "gt_actions": Value("string"),
        }),
    )


def _make_reward_fn(horizon: int) -> Any:
    def visual_reward_fn(
        completions: list[Any],
        image: list[Image.Image],
        gt_actions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        # Within one reward call (one GRPO group), all `num_generations`
        # completions share a single (canvas, gt_chunk) pair — cache the
        # executed GT canvas to skip ~7/8 of the GT executions per step.
        gt_cache: dict[tuple[int, str], np.ndarray] = {}
        canvas_cache: dict[int, np.ndarray] = {}

        rewards: list[float] = []
        for completion, canvas_pil, gt_json in zip(completions, image, gt_actions):
            text = completion if isinstance(completion, str) else completion[-1]["content"]

            pil_id = id(canvas_pil)
            canvas_np = canvas_cache.get(pil_id)
            if canvas_np is None:
                canvas_np = _image_to_float32(ensure_rgb_canvas_size(canvas_pil))
                canvas_cache[pil_id] = canvas_np

            try:
                pred_chunk = parse_action_chunk(text, horizon)
            except (JSONDecodeError, KeyError, ValueError, ValidationError, AssertionError):
                rewards.append(-1.0)
                continue

            predicted = execute_chunk(canvas_np, pred_chunk, original=canvas_np)
            gt_key = (pil_id, gt_json)
            gt_target = gt_cache.get(gt_key)
            if gt_target is None:
                gt_chunk = ActionChunk.from_json_str(gt_json, horizon)
                gt_target = execute_chunk(canvas_np, gt_chunk, original=canvas_np)
                gt_cache[gt_key] = gt_target
            rewards.append(visual_reward(predicted, gt_target, canvas_np))

        return rewards

    return visual_reward_fn


def run_rl_training(
    config_path: Path = Path("configs/train.yaml"),
    checkpoint_dir: Path | None = None,
) -> None:
    """Load SFT checkpoint and run GRPO training with visual reward.

    Args:
        config_path: Path to train.yaml.
        checkpoint_dir: Optional override for the SFT checkpoint to start from.
                        Defaults to cfg["data"]["checkpoint_dir"] / "final".
    """
    cfg = _load_config(config_path)
    rl_cfg = cfg.get("rl", {})
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    sft_checkpoint = checkpoint_dir or Path(data_cfg["checkpoint_dir"]) / "final"
    output_dir = Path(data_cfg["checkpoint_dir"]) / "rl_final"
    horizon: int = data_cfg["action_horizon"]

    tb_root = Path(cfg.get("training", {}).get("tensorboard_dir", "logs/tensorboard"))
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = tb_root / f"rl_{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TensorBoard logs → {}", log_dir)

    base_model = model_cfg["name"]
    local_only = base_model in cached_repo_ids()
    use_unsloth = bool(model_cfg.get("use_unsloth", False))

    logger.info(
        "Loading model from {} (local_files_only={}, use_unsloth={})",
        sft_checkpoint, local_only, use_unsloth,
    )

    if use_unsloth:
        import os as _os
        _os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
        from unsloth import FastVisionModel  # type: ignore

        model, processor = FastVisionModel.from_pretrained(
            model_name=str(sft_checkpoint),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            local_files_only=local_only,
        )
        # Idempotent: SFT checkpoint should already contain coord tokens.
        base_tokenizer = getattr(processor, "tokenizer", processor)
        added = add_coord_tokens(base_tokenizer)
        if added:
            try:
                model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(base_tokenizer))
            init_coord_token_embeddings(model, base_tokenizer)
            logger.warning(
                "RL added {} coord tokens at load time -- the SFT checkpoint did "
                "not contain them. Did you forget to run SFT after Phase 1?", added,
            )
        FastVisionModel.for_training(model)
    else:
        from elysium.model.loading import load_adapter_for_training

        model, processor = load_adapter_for_training(
            checkpoint_dir=sft_checkpoint,
            base_model_name=base_model,
            load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
            local_only=local_only,
        )

    apply_image_pixel_budget(processor, model_cfg)

    dataset_path = Path(data_cfg["dataset_path"])
    logger.info("Loading dataset from {}", dataset_path)
    raw_dataset = load_from_disk(str(dataset_path))
    train_data = raw_dataset["train"]

    min_gt_pixels = int(rl_cfg.get("min_gt_pixels", _MIN_GT_PIXELS_FOR_RL_DEFAULT))
    logger.info(
        "Building GRPO dataset ({} samples, min_gt_pixels={})",
        len(train_data), min_gt_pixels,
    )
    grpo_dataset = _build_grpo_dataset(
        train_data, processor, horizon, min_gt_pixels=min_gt_pixels
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=rl_cfg.get("epochs", 3),
        learning_rate=rl_cfg.get("learning_rate", 2e-6),
        per_device_train_batch_size=rl_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=rl_cfg.get("gradient_accumulation_steps", 4),
        num_generations=rl_cfg.get("num_generations", 8),
        temperature=rl_cfg.get("temperature", 1.0),
        top_p=rl_cfg.get("top_p", 0.95),
        beta=rl_cfg.get("beta", 0.1),
        max_prompt_length=rl_cfg.get("max_prompt_length", 768),
        max_completion_length=rl_cfg.get("max_completion_length", 384),
        repetition_penalty=rl_cfg.get("repetition_penalty", 1.0),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="tensorboard",
        logging_dir=str(log_dir),
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=rl_cfg.get("save_steps", 100),
        save_total_limit=rl_cfg.get("save_total_limit", 3),
        log_completions=True,
        max_grad_norm=rl_cfg.get("max_grad_norm", 0.1),
        warmup_ratio=rl_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=rl_cfg.get("lr_scheduler_type", "cosine"),
    )
    if "max_steps" in rl_cfg and rl_cfg["max_steps"] is not None:
        grpo_kwargs["max_steps"] = int(rl_cfg["max_steps"])
    grpo_config = GRPOConfig(**grpo_kwargs)

    noop_json = ActionChunk.noop_chunk(horizon).to_json_str()
    noop_token_len = len(processor.tokenizer(noop_json, add_special_tokens=False)["input_ids"])
    logger.info("Noop-JSON token length for collapse detection: {}", noop_token_len)

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[_make_reward_fn(horizon)],
        callbacks=[
            _AssertParamsFinite(),
            _DetectCollapse(noop_token_len=noop_token_len),
        ],
    )

    logger.info(
        "Starting GRPO training: {} epochs, {} samples, {} generations/prompt",
        grpo_config.num_train_epochs,
        len(grpo_dataset),
        grpo_config.num_generations,
    )
    trainer.train()

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("GRPO training complete. Saved to {}", output_dir)
