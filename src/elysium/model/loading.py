"""Model and processor loading utilities (HuggingFace + PEFT, no Unsloth).

Centralizes the slightly-fiddly logic for loading a Qwen3.5-VL model with
optional 4-bit quantization, attaching LoRA adapters, and resizing the
embedding table for the Phase-1 coord tokens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor

from elysium.log import logger
from elysium.model.coord_tokens import add_coord_tokens, init_coord_token_embeddings

__all__ = [
    "preferred_dtype",
    "auto_vlm_class",
    "build_quantization_config",
    "load_base_model",
    "load_processor",
    "apply_lora",
    "load_adapter_for_inference",
    "load_adapter_for_training",
    "ensure_coord_tokens",
]


def preferred_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def auto_vlm_class() -> type:
    """Return the auto-class for vision-language models, preferring the newer name.

    transformers >= 4.45 ships ``AutoModelForImageTextToText``; older versions
    expose ``AutoModelForVision2Seq``. Either works for our purposes.
    """
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore
        return AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq  # type: ignore
        return AutoModelForVision2Seq


def build_quantization_config(load_in_4bit: bool) -> Any | None:
    if not load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig  # local import; needs bitsandbytes
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=preferred_dtype(),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _move_to_cuda(model: Any) -> Any:
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def load_base_model(
    model_name: str,
    *,
    load_in_4bit: bool = False,
    local_only: bool = True,
) -> Any:
    """Load a vision-language base model in the preferred dtype."""
    cls = auto_vlm_class()
    kwargs: dict[str, Any] = dict(
        torch_dtype=preferred_dtype(),
        local_files_only=local_only,
    )
    qcfg = build_quantization_config(load_in_4bit)
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    logger.info(
        "Loading base model {} (4bit={}, dtype={}, local_only={})",
        model_name, load_in_4bit, kwargs["torch_dtype"], local_only,
    )
    model = cls.from_pretrained(model_name, **kwargs)
    if qcfg is None:
        # 4-bit models already place themselves; non-quantized ones we move.
        model = _move_to_cuda(model)
    return model


def load_processor(model_name_or_path: str, *, local_only: bool = True) -> Any:
    return AutoProcessor.from_pretrained(model_name_or_path, local_files_only=local_only)


def apply_lora(
    model: Any,
    lora_cfg: dict[str, Any],
    *,
    task_type: str = "CAUSAL_LM",
) -> Any:
    """Wrap ``model`` with PEFT LoRA adapters from a config dict."""
    target_modules = lora_cfg["target_modules"]
    config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg["bias"],
        target_modules=target_modules,
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def ensure_coord_tokens(model: Any, processor: Any) -> int:
    """Idempotently add coord sentinel tokens and initialize new embedding rows.

    Returns the number of newly added tokens (0 if already present).
    Must be called **before** ``apply_lora`` so the LoRA-targeted LM head
    picks up the new vocabulary rows.
    """
    base_tokenizer = getattr(processor, "tokenizer", processor)
    added = add_coord_tokens(base_tokenizer)
    if added:
        try:
            model.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
        except TypeError:
            # Older transformers without mean_resizing kwarg.
            model.resize_token_embeddings(len(base_tokenizer))
        init_coord_token_embeddings(model, base_tokenizer)
        logger.info("Added {} coord tokens and initialized embeddings", added)
    return added


def _is_adapter_checkpoint(checkpoint_dir: Path) -> bool:
    return (checkpoint_dir / "adapter_config.json").exists()


def _resolve_adapter_base_name(checkpoint_dir: Path, fallback: str) -> str:
    cfg_path = checkpoint_dir / "adapter_config.json"
    if not cfg_path.exists():
        return fallback
    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg.get("base_model_name_or_path") or fallback


def load_adapter_for_inference(
    checkpoint_dir: Path,
    base_model_name: str,
    *,
    load_in_4bit: bool = False,
    local_only: bool = True,
) -> tuple[Any, Any]:
    """Load a saved LoRA adapter (or merged model) for inference.

    Returns (model, processor) with the model already on cuda (if available)
    and in eval mode.
    """
    checkpoint_dir = Path(checkpoint_dir)
    processor = load_processor(str(checkpoint_dir), local_only=local_only)

    if _is_adapter_checkpoint(checkpoint_dir):
        base_name = _resolve_adapter_base_name(checkpoint_dir, base_model_name)
        base = load_base_model(base_name, load_in_4bit=load_in_4bit, local_only=local_only)
        base_tokenizer = getattr(processor, "tokenizer", processor)
        cur_rows = base.get_input_embeddings().weight.shape[0]
        if len(base_tokenizer) > cur_rows:
            try:
                base.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
            except TypeError:
                base.resize_token_embeddings(len(base_tokenizer))
            init_coord_token_embeddings(base, base_tokenizer)
            logger.warning(
                "Inference resized base embeddings from {} -> {} rows. "
                "The adapter at {} should already encode the coord tokens; "
                "the new rows are an init fallback only.",
                cur_rows, len(base_tokenizer), checkpoint_dir,
            )
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
    else:
        # Treat as a merged / full checkpoint.
        cls = auto_vlm_class()
        kwargs: dict[str, Any] = dict(
            torch_dtype=preferred_dtype(),
            local_files_only=local_only,
        )
        qcfg = build_quantization_config(load_in_4bit)
        if qcfg is not None:
            kwargs["quantization_config"] = qcfg
        model = cls.from_pretrained(str(checkpoint_dir), **kwargs)
        if qcfg is None:
            model = _move_to_cuda(model)

    model.eval()
    return model, processor


def load_adapter_for_training(
    checkpoint_dir: Path,
    base_model_name: str,
    *,
    load_in_4bit: bool = False,
    local_only: bool = True,
) -> tuple[Any, Any]:
    """Load a saved LoRA adapter as the trainable starting policy for RL.

    Returns (model, processor) with the adapter marked trainable.
    """
    checkpoint_dir = Path(checkpoint_dir)
    processor = load_processor(str(checkpoint_dir), local_only=local_only)
    assert _is_adapter_checkpoint(checkpoint_dir), (
        f"{checkpoint_dir} is not a LoRA adapter checkpoint; RL training "
        "expects adapter_config.json to be present."
    )
    base_name = _resolve_adapter_base_name(checkpoint_dir, base_model_name)
    base = load_base_model(base_name, load_in_4bit=load_in_4bit, local_only=local_only)
    base_tokenizer = getattr(processor, "tokenizer", processor)
    cur_rows = base.get_input_embeddings().weight.shape[0]
    if len(base_tokenizer) > cur_rows:
        try:
            base.resize_token_embeddings(len(base_tokenizer), mean_resizing=False)
        except TypeError:
            base.resize_token_embeddings(len(base_tokenizer))
        init_coord_token_embeddings(base, base_tokenizer)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir), is_trainable=True)
    return model, processor
