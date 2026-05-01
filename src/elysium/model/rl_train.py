"""GRPO policy-gradient training for the chunk-prediction model.

Uses TRL's GRPOTrainer with a pure visual (SSIM) reward.  The SFT checkpoint
is loaded with its LoRA adapters as the policy; the frozen reference policy is
obtained via disable_adapter() on the same PEFT model — no extra copy needed.

Config: configs/train.yaml  (rl section)
Data:   data/processed/     (HuggingFace DatasetDict with next_image)
Output: models/checkpoints/rl_final/
"""

from __future__ import annotations

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
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.action_io import (
    action_conversation_messages,
    apply_action_chat_template,
    parse_action_chunk,
)
from elysium.model.predict import cached_repo_ids, ensure_rgb_canvas_size
from elysium.model.reward import visual_reward

__all__ = ["run_rl_training"]


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


def _build_grpo_dataset(train_data: Any, processor: Any) -> Dataset:
    prompts: list[str] = []
    images: list[str] = []
    next_images: list[str] = []

    for row in train_data:
        instruction = _extract_instruction(row)
        messages = action_conversation_messages(instruction)
        prompt_text = apply_action_chat_template(processor, messages, add_generation_prompt=True)
        prompts.append(prompt_text)
        images.append(row["image"])
        next_images.append(row.get("next_image", ""))

    return Dataset.from_dict(
        {"prompt": prompts, "image": images, "next_image": next_images},
        features=Features({
            "prompt": Value("string"),
            "image": HFImage(),
            "next_image": Value("string"),
        }),
    )


def _make_reward_fn(horizon: int) -> Any:
    def visual_reward_fn(
        completions: list[Any],
        image: list[Image.Image],
        next_image: list[str],
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, canvas_pil, gt_path in zip(completions, image, next_image):
            if not gt_path:
                rewards.append(0.0)
                continue

            text = completion if isinstance(completion, str) else completion[-1]["content"]
            canvas_np = _image_to_float32(ensure_rgb_canvas_size(canvas_pil))
            gt_pil = ensure_rgb_canvas_size(Image.open(gt_path).convert("RGB"))
            gt_np = _image_to_float32(gt_pil)

            try:
                pred_chunk = parse_action_chunk(text, horizon)
            except (JSONDecodeError, KeyError, ValueError, ValidationError, AssertionError):
                rewards.append(0.0)
                continue

            predicted = execute_chunk(canvas_np, pred_chunk, original=canvas_np)
            rewards.append(visual_reward(predicted, gt_np))

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

    base_model = model_cfg["name"]
    local_only = base_model in cached_repo_ids()

    logger.info("Loading model from {} (local_files_only={})", sft_checkpoint, local_only)
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(sft_checkpoint),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        local_files_only=local_only,
    )
    FastVisionModel.for_training(model)

    dataset_path = Path(data_cfg["dataset_path"])
    logger.info("Loading dataset from {}", dataset_path)
    raw_dataset = load_from_disk(str(dataset_path))
    train_data = raw_dataset["train"]

    logger.info("Building GRPO dataset ({} samples)", len(train_data))
    grpo_dataset = _build_grpo_dataset(train_data, processor)

    output_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=rl_cfg.get("epochs", 3),
        learning_rate=rl_cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size=rl_cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=rl_cfg.get("gradient_accumulation_steps", 8),
        num_generations=rl_cfg.get("num_generations", 4),
        temperature=rl_cfg.get("temperature", 0.7),
        top_p=rl_cfg.get("top_p", 0.9),
        beta=rl_cfg.get("beta", 0.04),
        max_prompt_length=rl_cfg.get("max_prompt_length", 2048),
        max_completion_length=rl_cfg.get("max_completion_length", 512),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=rl_cfg.get("save_steps", 50),
        log_completions=True,
        max_grad_norm=rl_cfg.get("max_grad_norm", 0.3),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        args=grpo_config,
        train_dataset=grpo_dataset,
        reward_funcs=[_make_reward_fn(horizon)],
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
