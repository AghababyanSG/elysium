"""QLoRA fine-tuning for Qwen3.5 via Unsloth.

Trains the model to predict 5-action chunks from (image, instruction) inputs.
Vision encoder is frozen to fit within 8GB VRAM. Loss is masked to assistant
tokens only so gradients only flow through the JSON action output.

Config: configs/train.yaml
Data:   data/processed/  (HuggingFace DatasetDict)
Output: models/checkpoints/
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

import torch
import yaml
from unsloth import FastVisionModel
from datasets import load_from_disk
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from elysium.log import logger
from elysium.model.predict import apply_image_pixel_budget, cached_repo_ids, ensure_rgb_canvas_size
from unsloth_zoo.vision_utils import UnslothVisionDataCollator

__all__ = ["run_training"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


class ElysiumActionVisionDataCollator(UnslothVisionDataCollator):
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        from PIL import Image

        pc: list[dict[str, Any]] = []
        for ex in examples:
            msgs = ex["messages"]
            assert msgs[-1]["role"] == "assistant"
            img = ensure_rgb_canvas_size(Image.open(ex["image"]).convert("RGB"))
            pc.append(
                {
                    "prompt": msgs[:-1],
                    "completion": [msgs[-1]],
                    "images": [img],
                }
            )
        batch = super().__call__(pc)
        if "mm_token_type_ids" in batch:
            input_len = batch["input_ids"].shape[1]
            mm_len = batch["mm_token_type_ids"].shape[1]
            if mm_len < input_len:
                pad = torch.zeros(
                    batch["mm_token_type_ids"].shape[0],
                    input_len - mm_len,
                    dtype=batch["mm_token_type_ids"].dtype,
                    device=batch["mm_token_type_ids"].device,
                )
                batch["mm_token_type_ids"] = torch.cat([batch["mm_token_type_ids"], pad], dim=1)
            elif mm_len > input_len:
                batch["mm_token_type_ids"] = batch["mm_token_type_ids"][:, :input_len]
        return batch

    def _render_chat(
        self,
        prompt_messages: list[Any],
        completion_messages: list[Any] | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        merged = prompt_messages + (completion_messages or [])
        return self.processor.apply_chat_template(
            merged,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=False,
        )


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
    local_only = model_name in cached_repo_ids()
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

    apply_image_pixel_budget(tokenizer, model_cfg)

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

    tb_root = Path(train_cfg.get("tensorboard_dir", "logs/tensorboard"))
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = tb_root / f"sft_{run_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("TensorBoard logs → {}", log_dir)

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg.get("eval_batch_size", train_cfg["batch_size"]),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["epochs"],
        optim=train_cfg["optimizer"],
        max_length=train_cfg["max_seq_length"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="tensorboard",
        logging_dir=str(log_dir),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    data_collator = ElysiumActionVisionDataCollator(
        model,
        tokenizer,
        max_seq_length=train_cfg["max_seq_length"],
    )

    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 2))
    early_stopping_threshold = float(train_cfg.get("early_stopping_threshold", 0.001))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            ),
        ],
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
