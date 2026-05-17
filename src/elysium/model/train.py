"""LoRA fine-tuning for Qwen3.5-VL.

Two parallel loading paths, selected by ``model.use_unsloth`` in
``configs/train.yaml``:

  * ``use_unsloth: true``  -- Unsloth's FastVisionModel + UnslothVisionDataCollator.
                              Best for small-VRAM laptops (QLoRA, 8-bit optim).
  * ``use_unsloth: false`` -- Plain HuggingFace + PEFT with a hand-rolled vision
                              collator. Use on bigger GPUs / when Unsloth's
                              vision quirks (forced 512x512 patch embed, vision
                              tower require_grad) get in the way.

Both paths apply the Phase 1 coord-token surgery before LoRA wrap and write
loss only on assistant tokens.

Config: configs/train.yaml
Data:   data/processed/  (HuggingFace DatasetDict)
Output: models/checkpoints/
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_from_disk
from PIL import Image
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from elysium.log import logger
from elysium.model.coord_tokens import add_coord_tokens, init_coord_token_embeddings
from elysium.model.predict import apply_image_pixel_budget, cached_repo_ids, ensure_rgb_canvas_size

__all__ = ["run_training"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


_NOOP_PATTERN = '{"action_type":"noop"}'


def _noop_token_positions(
    tokenizer: Any,
    full_text: str,
    prompt_text: str,
    n_prompt: int,
    full_seq_length: int,
) -> list[int]:
    """Absolute token positions inside any ``{"action_type":"noop"}`` span.

    Phase 5.5.1: assistant tokens that fall inside a noop action serialization
    get an upweighted loss. This helper finds those tokens.

    Strategy: strip the prompt prefix from ``full_text`` to get the assistant
    segment, tokenize it standalone with offset mapping, find noop substring
    spans in the segment, mark tokens whose char offsets overlap any span,
    map back to absolute positions by adding ``n_prompt``.

    Returns an empty list (and logs a warning) if the standalone segment
    tokenization length doesn't match ``full_seq_length - n_prompt`` — that
    indicates the chat-template boundary merged tokens differently between
    the standalone and concatenated tokenizations, and we'd rather skip
    weighting for that example than mark the wrong positions.
    """
    if not full_text.startswith(prompt_text):
        return []
    segment = full_text[len(prompt_text):]
    if _NOOP_PATTERN not in segment:
        return []

    encoding = tokenizer(
        segment,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    seg_token_count = len(encoding["input_ids"])
    expected = full_seq_length - n_prompt
    if seg_token_count != expected:
        logger.warning(
            "noop weight: segment retokenize {} != trailing {} (skipping example)",
            seg_token_count, expected,
        )
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = segment.find(_NOOP_PATTERN, start)
        if i < 0:
            break
        spans.append((i, i + len(_NOOP_PATTERN)))
        start = i + len(_NOOP_PATTERN)

    positions: list[int] = []
    for tok_idx, (a, b) in enumerate(offsets):
        if a == b:  # special tokens have (0, 0) — skip
            continue
        for s, e in spans:
            if a < e and b > s:
                positions.append(n_prompt + tok_idx)
                break
    return positions


def _strip_none_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # HF datasets unifies the arrow schema across all `content` items, so text
    # dicts end up with a stray `image: None` key (and image dicts with
    # `text: None`). The Qwen3.5 chat template checks key *presence* via
    # `'image' in item`, which then (a) raises "System message cannot contain
    # images" on system text and (b) would silently emit <|image_pad|> in place
    # of user instruction text. Strip None-valued keys from every content dict.
    clean: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = [
                {k: v for k, v in item.items() if v is not None}
                if isinstance(item, dict) else item
                for item in content
            ]
            clean.append({**msg, "content": new_content})
        else:
            clean.append(msg)
    return clean


# ---------------------------------------------------------------------------
# Unsloth path (laptop / QLoRA)
# ---------------------------------------------------------------------------

def _build_unsloth_model_and_collator(
    model_cfg: dict[str, Any],
    lora_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[Any, Any, Any]:
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    from unsloth import FastVisionModel  # type: ignore
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator  # type: ignore

    model_name = model_cfg["name"]
    local_only = model_name in cached_repo_ids()
    logger.info(
        "[unsloth] Loading model {} (4bit={}, local_files_only={})",
        model_name, model_cfg["load_in_4bit"], local_only,
    )
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=model_cfg["load_in_4bit"],
        use_gradient_checkpointing=train_cfg["gradient_checkpointing"],
        local_files_only=local_only,
    )
    apply_image_pixel_budget(tokenizer, model_cfg)

    base_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    added = add_coord_tokens(base_tokenizer)
    if added:
        try:
            model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(len(base_tokenizer))
        init_coord_token_embeddings(model, base_tokenizer)
        logger.info("[unsloth] Added {} coord tokens and initialized embeddings", added)

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

    class _UnslothCollator(UnslothVisionDataCollator):
        def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
            pc: list[dict[str, Any]] = []
            for ex in examples:
                msgs = _strip_none_content(ex["messages"])
                assert msgs[-1]["role"] == "assistant"
                img = ensure_rgb_canvas_size(Image.open(ex["image"]).convert("RGB"))
                pc.append({
                    "prompt": msgs[:-1],
                    "completion": [msgs[-1]],
                    "images": [img],
                })
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

    if float(train_cfg.get("noop_loss_weight", 1.0)) != 1.0:
        logger.warning(
            "[unsloth] noop_loss_weight={} ignored — Phase 5.5.1 weighting is "
            "only wired on the HF path (set use_unsloth=false to use it).",
            train_cfg.get("noop_loss_weight"),
        )
    collator = _UnslothCollator(
        model, tokenizer, max_seq_length=train_cfg["max_seq_length"]
    )
    return model, tokenizer, collator


# ---------------------------------------------------------------------------
# HuggingFace + PEFT path (bigger GPU / no Unsloth)
# ---------------------------------------------------------------------------

class _HFActionVisionDataCollator:
    """Plain HF/PEFT vision SFT collator masking loss to assistant tokens.

    With ``noop_loss_weight != 1.0`` (Phase 5.5.1), additionally attaches
    a ``loss_weights`` tensor to the batch — 1.0 default, ``noop_loss_weight``
    at tokens that fall inside any ``{"action_type":"noop"}`` action span.
    ``_WeightedSFTTrainer`` consumes the tensor to upweight noop-token loss.
    """

    def __init__(
        self,
        processor: Any,
        max_seq_length: int,
        noop_loss_weight: float = 1.0,
    ) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.noop_loss_weight = noop_loss_weight
        self._template_kwargs = {"enable_thinking": False}

    def _apply_template(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self._template_kwargs,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Image.Image] = []
        for ex in examples:
            msgs = _strip_none_content(ex["messages"])
            assert msgs[-1]["role"] == "assistant"
            img = ensure_rgb_canvas_size(Image.open(ex["image"]).convert("RGB"))
            full_texts.append(self._apply_template(msgs, add_generation_prompt=False))
            prompt_texts.append(self._apply_template(msgs[:-1], add_generation_prompt=True))
            images.append(img)

        full_batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        labels = full_batch["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        attn = full_batch.get("attention_mask")
        weight = self.noop_loss_weight
        loss_weights: torch.Tensor | None = None
        if weight != 1.0:
            loss_weights = torch.ones_like(labels, dtype=torch.float32)

        for i, prompt_text in enumerate(prompt_texts):
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[images[i]],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
            )
            n_prompt = int(prompt_inputs["input_ids"].shape[1])
            labels[i, :n_prompt] = -100

            if loss_weights is not None:
                # `full_batch` is padded to the longest sequence in the batch
                # — that's NOT this example's actual length. Subtracting
                # n_prompt from the padded length would over-count by the
                # pad token tail. Use attention_mask to get true content len.
                if attn is not None:
                    actual_full_len = int(attn[i].sum().item())
                else:
                    actual_full_len = int((labels[i] != pad_id).sum().item())
                positions = _noop_token_positions(
                    self.tokenizer,
                    full_text=full_texts[i],
                    prompt_text=prompt_text,
                    n_prompt=n_prompt,
                    full_seq_length=actual_full_len,
                )
                if positions:
                    idx = torch.tensor(positions, dtype=torch.long)
                    loss_weights[i].index_fill_(0, idx, weight)

        if pad_id is not None:
            labels[labels == pad_id] = -100
        full_batch["labels"] = labels
        if loss_weights is not None:
            full_batch["loss_weights"] = loss_weights
        return full_batch


def _freeze_vision_tower(model: Any) -> None:
    """Walk a (possibly PEFT-wrapped) model and disable grads on `visual`."""
    candidates: list[Any] = []
    base = getattr(model, "base_model", None)
    if base is not None:
        candidates.append(getattr(base, "model", base))
    candidates.append(model)
    for root in candidates:
        visual = getattr(root, "visual", None)
        if visual is None:
            inner = getattr(root, "model", None)
            visual = getattr(inner, "visual", None) if inner is not None else None
        if visual is not None:
            n = 0
            for p in visual.parameters():
                p.requires_grad = False
                n += 1
            logger.info("[hf] Vision tower frozen ({} params)", n)
            return
    logger.warning("[hf] Vision tower not located; skipping freeze")


class _WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer that consumes a per-token ``loss_weights`` tensor.

    When the batch carries ``loss_weights`` (Phase 5.5.1 noop upweighting),
    recompute the loss as a weighted mean of per-token cross-entropy over
    non-masked (`label != -100`) positions. When absent, fall back to the
    model's built-in loss.
    """

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        loss_weights = inputs.pop("loss_weights", None)
        outputs = model(**inputs)
        if loss_weights is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        labels = inputs["labels"]
        # Causal-LM shift: predict token t+1 from logits at t.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].to(shift_logits.dtype).contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)
        active = (shift_labels != -100).to(per_token.dtype)
        weighted = per_token * shift_weights * active
        loss = weighted.sum() / active.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def _build_hf_model_and_collator(
    model_cfg: dict[str, Any],
    lora_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> tuple[Any, Any, Any]:
    # Local import to avoid pulling Unsloth/peft for users on the Unsloth path.
    from elysium.model.coord_tokens import coord_token_ids
    from elysium.model.loading import (
        apply_lora,
        ensure_coord_tokens,
        load_base_model,
        load_processor,
    )

    model = load_base_model(
        model_cfg["name"],
        load_in_4bit=bool(model_cfg.get("load_in_4bit", False)),
        local_only=True,
    )
    processor = load_processor(model_cfg["name"], local_only=True)
    apply_image_pixel_budget(processor, model_cfg)

    # Phase 1 coord-token surgery must happen before LoRA wrap.
    ensure_coord_tokens(model, processor)

    if not lora_cfg.get("finetune_vision_layers", False):
        _freeze_vision_tower(model)

    # Without this the 768 new embed_tokens/lm_head rows stay frozen at their
    # digit-mean init and the LM head never learns to prefer <cN>/<xN>/<yN>
    # over plain decimal digits (greedy decode silently emits decimals).
    coord_ids = coord_token_ids(getattr(processor, "tokenizer", processor))
    model = apply_lora(model, lora_cfg, trainable_token_indices=coord_ids)
    noop_loss_weight = float(train_cfg.get("noop_loss_weight", 1.0))
    if noop_loss_weight != 1.0:
        logger.info("[hf] noop_loss_weight = {} (Phase 5.5.1)", noop_loss_weight)
    collator = _HFActionVisionDataCollator(
        processor=processor,
        max_seq_length=train_cfg["max_seq_length"],
        noop_loss_weight=noop_loss_weight,
    )
    return model, processor, collator


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training(
    config_path: Path = Path("configs/train.yaml"),
    epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Load data, build model, run LoRA fine-tuning."""
    cfg = _load_config(config_path)
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    if epochs is not None:
        train_cfg["epochs"] = epochs
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size

    use_unsloth = bool(model_cfg.get("use_unsloth", False))
    logger.info("Training path: use_unsloth={}", use_unsloth)
    if use_unsloth:
        model, processor, data_collator = _build_unsloth_model_and_collator(
            model_cfg, lora_cfg, train_cfg
        )
    else:
        model, processor, data_collator = _build_hf_model_and_collator(
            model_cfg, lora_cfg, train_cfg
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

    bf16 = torch.cuda.is_bf16_supported()
    # gradient_checkpointing config in train.yaml may be a string ("unsloth")
    # or a bool. TRL/HF Trainer only accepts the bool.
    gc_cfg = train_cfg.get("gradient_checkpointing", False)
    gc_bool = bool(gc_cfg) if not isinstance(gc_cfg, str) else (gc_cfg.lower() != "false")
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
        gradient_checkpointing=gc_bool,
        fp16=not bf16,
        bf16=bf16,
        report_to="tensorboard",
        logging_dir=str(log_dir),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    early_stopping_patience = int(train_cfg.get("early_stopping_patience", 2))
    early_stopping_threshold = float(train_cfg.get("early_stopping_threshold", 0.001))

    trainer_cls = _WeightedSFTTrainer if not use_unsloth else SFTTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
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
    processor.save_pretrained(str(final_checkpoint))

    logger.info(
        "Training complete. Steps: {}, Loss: {:.4f}",
        trainer_stats.global_step,
        trainer_stats.training_loss,
    )
    logger.info("Saved final checkpoint to {}", final_checkpoint)
