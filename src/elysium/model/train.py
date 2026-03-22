"""QLoRA fine-tuning for Qwen3.5 via Unsloth.

Trains the model to predict 5-action chunks from (image, instruction) inputs.
Vision encoder is frozen to fit within 8GB VRAM. Loss is masked to assistant
tokens only so gradients only flow through the JSON action output.

Config: configs/train.yaml
Data:   data/processed/  (HuggingFace DatasetDict)
Output: models/checkpoints/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
import yaml
from datasets import load_from_disk
from huggingface_hub import scan_cache_dir
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from elysium.log import logger

__all__ = ["run_training"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _build_conversation(sample: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Apply the model's chat template to a single dataset record.

    Args:
        sample: Record with "messages" and "image" fields.
        processor: Qwen3.5 processor.

    Returns:
        Dict with "input_ids", "attention_mask", "pixel_values", "labels".
    """
    from PIL import Image

    messages = sample["messages"]
    image_path = sample["image"]

    image = Image.open(image_path).convert("RGB")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=False,
    )

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values[0]

    labels = input_ids.clone()
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    label_mask_started = False
    for idx, tok in enumerate(input_ids):
        if not label_mask_started:
            labels[idx] = -100
        if tok.item() == assistant_token_id:
            label_mask_started = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


def run_training(
    config_path: Path = Path("configs/train.yaml"),
    epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Load data, build model, run QLoRA fine-tuning.

    Args:
        config_path: Path to train.yaml config file.
        epochs: Override epochs from config.
        batch_size: Override batch_size from config.
    """
    cfg = _load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size

    model_name = model_cfg["name"]
    cached_repos = {r.repo_id for r in scan_cache_dir().repos}
    local_only = model_name in cached_repos
    logger.info(
        "Loading model {} (4bit={}, local_files_only={})",
        model_name,
        model_cfg["load_in_4bit"],
        local_only,
    )
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=model_cfg["load_in_4bit"],
        use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
        local_files_only=local_only,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lora_cfg["finetune_vision_layers"],
        finetune_language_layers=lora_cfg["finetune_language_layers"],
        finetune_attention_modules=lora_cfg["finetune_attention_modules"],
        finetune_mlp_modules=lora_cfg["finetune_mlp_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        random_state=42,
        use_rslora=False,
    )

    dataset_path = Path(data_cfg["dataset_path"])
    logger.info("Loading dataset from {}", dataset_path)
    dataset = load_from_disk(str(dataset_path))
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation")

    checkpoint_dir = Path(data_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["epochs"],
        optim=train_cfg["optimizer"],
        max_seq_length=train_cfg["max_seq_length"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    logger.info("Starting training")
    trainer_stats = trainer.train()

    final_checkpoint = checkpoint_dir / "final"
    model.save_pretrained(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))

    logger.info(
        "Training complete. Steps: {}, Loss: {:.4f}",
        trainer_stats.global_step,
        trainer_stats.training_loss,
    )
    logger.info("Saved final checkpoint to {}", final_checkpoint)
