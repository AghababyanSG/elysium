"""Visual reward for GRPO training.

Computes a masked region accuracy between the canvas produced by executing
generated actions and the ground-truth next canvas from the recording.

Rather than global SSIM (blind to small strokes that cover <1% of pixels),
the reward focuses on the pixels that actually changed in the ground truth:

- Non-terminal chunks: MAE inside the GT-changed region, with a small penalty
  for unexpected changes outside that region. Reward in [-1, 1].
- Terminal chunks (no gt_next) or near-end chunks with <50 changed pixels:
  penalise any change to the canvas — a correct noop scores ~1.0.
"""

from __future__ import annotations

import numpy as np

from elysium.schemas.actions import CANVAS_SIZE

__all__ = ["visual_reward"]

_CHANGE_THRESHOLD = 0.01   # per-channel delta considered "changed"
_MIN_CHANGED_PX = 50       # below this, treat the chunk as terminal
_TERMINAL_SCALE = 20.0     # multiplier that maps mean canvas change → penalty


def _region_reward(
    predicted: np.ndarray,
    gt_next: np.ndarray,
    current: np.ndarray,
) -> float:
    """MAE-based reward focused on the GT-changed region."""
    diff_gt = np.abs(gt_next - current)                   # (H, W, 3)
    mask = diff_gt.max(axis=2) > _CHANGE_THRESHOLD        # (H, W) bool

    n_changed = int(mask.sum())
    if n_changed < _MIN_CHANGED_PX:
        return _terminal_reward(predicted, current)

    region_mae = float(np.abs(predicted[mask] - gt_next[mask]).mean())
    region_acc = 1.0 - region_mae                          # in [-∞, 1]

    outside = ~mask
    if outside.any():
        unexpected = float(np.abs(predicted[outside] - current[outside]).mean())
        region_acc -= 0.1 * unexpected

    return float(np.clip(region_acc, -1.0, 1.0))


def _terminal_reward(predicted: np.ndarray, current: np.ndarray) -> float:
    """Reward for chunks where no change is expected (noop territory)."""
    pred_change = float(np.abs(predicted - current).mean())
    return float(max(-1.0, 1.0 - pred_change * _TERMINAL_SCALE))


def visual_reward(
    predicted_canvas: np.ndarray,
    gt_next_canvas: np.ndarray | None,
    current_canvas: np.ndarray,
) -> float:
    """Masked-region visual reward in [-1, 1].

    Args:
        predicted_canvas: float32 RGB [0, 1] canvas after executing generated
            actions. Shape: (H, W, 3).
        gt_next_canvas: float32 RGB [0, 1] ground-truth next-state canvas, or
            None for terminal chunks (model should predict noop).
        current_canvas: float32 RGB [0, 1] canvas before any actions were
            applied (the observation the model received).

    Returns:
        Scalar reward in [-1, 1].
    """
    assert predicted_canvas.shape == current_canvas.shape, (
        f"Shape mismatch: {predicted_canvas.shape} vs {current_canvas.shape}"
    )

    if gt_next_canvas is None:
        return _terminal_reward(predicted_canvas, current_canvas)

    assert predicted_canvas.shape == gt_next_canvas.shape, (
        f"Shape mismatch: {predicted_canvas.shape} vs {gt_next_canvas.shape}"
    )
    return _region_reward(predicted_canvas, gt_next_canvas, current_canvas)
