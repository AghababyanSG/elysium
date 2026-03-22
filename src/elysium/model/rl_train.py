"""REINFORCE policy gradient training for the chunk-prediction model.

Each training step:
  1. Generate a chunk autoregressively (with temperature sampling, no_grad).
  2. Execute the generated actions on the canvas.
  3. Compute reward: alpha * action_reward + beta * visual_reward.
  4. Forward pass with gradients to compute token log-probabilities.
  5. REINFORCE update: loss = -log_prob * (reward - baseline).

An exponential moving average baseline is maintained for variance reduction.

Config: configs/train.yaml  (rl section)
Data:   data/processed/     (HuggingFace DatasetDict with next_image + gt_actions)
Output: models/checkpoints/ (same directory as SFT, saved as rl_final/)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import os
import torch
import yaml

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

from datasets import load_from_disk
from PIL import Image
from unsloth import FastVisionModel

from elysium.engine.canvas import execute_chunk
from elysium.log import logger
from elysium.model.predict import _parse_chunk
from elysium.model.reward import compute_reward
from elysium.schemas.actions import SYSTEM_PROMPT

__all__ = ["run_rl_training"]


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _image_to_float32(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.float32) / 255.0


def _gather_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Extract per-token log-probabilities for the generated token ids.

    Args:
        logits: (1, seq_len, vocab_size) — logits from the model forward pass,
                shifted so logits[:, t] predicts token_ids[:, t].
        token_ids: (1, gen_len) — the generated token ids.

    Returns:
        (gen_len,) tensor of log-probabilities.
    """
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
    return log_probs.gather(1, token_ids[0].unsqueeze(1)).squeeze(1)


def _build_prompt_inputs(
    processor: Any,
    canvas_pil: Image.Image,
    instruction: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenise the user-side prompt (no assistant turn) for generation.

    Args:
        processor: Qwen3.5 processor.
        canvas_pil: Current canvas as a PIL image.
        instruction: Natural language editing instruction.
        device: Target device.

    Returns:
        Dict of tensors ready for model.generate / model.forward.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[canvas_pil], return_tensors="pt", padding=False)
    return {k: v.to(device) for k, v in inputs.items() if v is not None}


def _generate_chunk(
    model: Any,
    processor: Any,
    prompt_inputs: dict[str, torch.Tensor],
    temperature: float,
    max_new_tokens: int,
) -> tuple[torch.Tensor, str]:
    """Generate one action chunk autoregressively (no gradient).

    Args:
        model: The fine-tuned model.
        processor: Qwen3.5 processor.
        prompt_inputs: Tokenised prompt tensors.
        temperature: Sampling temperature (must be > 0 for REINFORCE).
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (generated_ids tensor of shape (1, gen_len), decoded string).
    """
    with torch.no_grad():
        output = model.generate(
            **prompt_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    input_len = prompt_inputs["input_ids"].shape[1]
    generated_ids = output[:, input_len:]
    raw_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    return generated_ids, raw_text


def _compute_log_prob(
    model: Any,
    prompt_inputs: dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with gradients to get the sum of log-probs of generated tokens.

    We concatenate prompt + generated tokens, run a single forward pass, then
    extract the logits corresponding to the generated portion.

    Args:
        model: The fine-tuned model.
        prompt_inputs: Dict of prompt tensors (input_ids, attention_mask, pixel_values).
        generated_ids: (1, gen_len) generated token ids.

    Returns:
        Scalar tensor: sum of log-probs over generated tokens (with gradient).
    """
    input_len = prompt_inputs["input_ids"].shape[1]
    full_ids = torch.cat([prompt_inputs["input_ids"], generated_ids], dim=1)
    full_mask = torch.cat(
        [
            prompt_inputs["attention_mask"],
            torch.ones(1, generated_ids.shape[1], device=full_ids.device, dtype=torch.long),
        ],
        dim=1,
    )

    forward_kwargs: dict[str, Any] = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
    }
    if "pixel_values" in prompt_inputs:
        forward_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
    if "image_grid_thw" in prompt_inputs:
        forward_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

    output = model(**forward_kwargs)

    gen_logits = output.logits[:, input_len - 1 : input_len - 1 + generated_ids.shape[1], :]
    log_probs = _gather_log_probs(gen_logits, generated_ids)
    return log_probs.sum()


def run_rl_training(
    config_path: Path = Path("configs/train.yaml"),
    checkpoint_dir: Path | None = None,
) -> None:
    """Load fine-tuned model and run REINFORCE training loop.

    Args:
        config_path: Path to train.yaml.
        checkpoint_dir: Optional override for the SFT checkpoint to start from.
                        Defaults to cfg["data"]["checkpoint_dir"] / "final".
    """
    cfg = _load_config(config_path)
    rl_cfg = cfg.get("rl", {})
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    num_epochs: int = rl_cfg.get("epochs", 3)
    lr: float = rl_cfg.get("learning_rate", 1e-5)
    temperature: float = rl_cfg.get("temperature", 0.7)
    alpha: float = rl_cfg.get("alpha", 0.5)
    beta: float = rl_cfg.get("beta", 0.5)
    baseline_decay: float = rl_cfg.get("baseline_decay", 0.99)
    max_new_tokens: int = rl_cfg.get("max_new_tokens", 1024)
    horizon: int = data_cfg["action_horizon"]

    sft_checkpoint = checkpoint_dir or Path(data_cfg["checkpoint_dir"]) / "final"
    output_dir = Path(data_cfg["checkpoint_dir"]) / "rl_final"

    from huggingface_hub import scan_cache_dir
    cached_repos = {r.repo_id for r in scan_cache_dir().repos}
    base_model = model_cfg["name"]
    local_only = base_model in cached_repos

    logger.info("Loading model from {} (local_files_only={})", sft_checkpoint, local_only)
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(sft_checkpoint),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        local_files_only=local_only,
    )
    FastVisionModel.for_inference(model)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    dataset_path = Path(data_cfg["dataset_path"])
    logger.info("Loading dataset from {}", dataset_path)
    dataset = load_from_disk(str(dataset_path))
    train_data = dataset["train"]

    device = next(model.parameters()).device
    baseline = 0.0
    global_step = 0

    logger.info("Starting REINFORCE training: {} epochs, {} samples", num_epochs, len(train_data))

    for epoch in range(num_epochs):
        epoch_rewards: list[float] = []

        for sample in train_data:
            instruction: str = sample["messages"][0]["content"][-1]["text"]
            image_path: str = sample["image"]
            gt_actions_json: str = sample.get("gt_actions", "")
            next_image_path: str = sample.get("next_image", "")

            canvas_pil = Image.open(image_path).convert("RGB")
            canvas_np = _image_to_float32(canvas_pil)

            gt_next_np: np.ndarray | None = None
            if next_image_path:
                gt_next_pil = Image.open(next_image_path).convert("RGB")
                gt_next_np = _image_to_float32(gt_next_pil)

            gt_actions: list[dict[str, Any]] = []
            if gt_actions_json:
                gt_actions = json.loads(gt_actions_json).get("actions", [])

            prompt_inputs = _build_prompt_inputs(processor, canvas_pil, instruction, device)

            generated_ids, raw_text = _generate_chunk(
                model, processor, prompt_inputs, temperature, max_new_tokens
            )

            pred_chunk = _parse_chunk(raw_text, horizon)
            predicted_canvas = execute_chunk(canvas_np, pred_chunk, original=canvas_np)

            reward = compute_reward(
                pred_chunk,
                gt_actions,
                predicted_canvas,
                gt_next_np,
                alpha=alpha,
                beta=beta,
            )

            log_prob = _compute_log_prob(model, prompt_inputs, generated_ids)
            advantage = reward - baseline
            loss = -log_prob * advantage

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            baseline = baseline_decay * baseline + (1 - baseline_decay) * reward
            epoch_rewards.append(reward)
            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "Epoch {} | step {} | reward={:.4f} | baseline={:.4f} | loss={:.4f}",
                    epoch + 1,
                    global_step,
                    reward,
                    baseline,
                    loss.item(),
                )

        mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        logger.info("Epoch {} complete | mean_reward={:.4f}", epoch + 1, mean_reward)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("RL training complete. Saved to {}", output_dir)
