"""Visual reward for GRPO training.

Coverage-weighted reward that pushes the policy away from noop predictions.

For a non-terminal training row (the only kind RL training sees — the dataset
is filtered upstream in `rl_train._build_grpo_dataset`):

    coverage  = |pred_changed_mask ∩ gt_mask| / |gt_mask|     in [0, 1]
    accuracy  = 1 - mean|pred - gt_next| inside gt_mask        in [-∞, 1]
    spurious  = mean|pred - current| outside gt_mask           in [0, 1]
    reward    = coverage * accuracy - 0.1 * spurious           clipped to [-1, 1]

A noop prediction has `coverage = 0` ⇒ reward ≤ 0, so the policy can never
beat real strokes by doing nothing. A pixel-perfect prediction has
coverage = 1 and accuracy = 1 ⇒ reward = 1 (minus any spurious-change
penalty). The reward is monotone in "actually painted near where GT painted",
which gives GRPO a usable advantage signal.
"""

from __future__ import annotations

import numpy as np

__all__ = ["visual_reward"]

_CHANGE_THRESHOLD = 0.01   # per-channel delta considered "changed"
_MIN_GT_PIXELS = 50        # below this the row should not be in the RL dataset


def visual_reward(
    predicted_canvas: np.ndarray,
    gt_next_canvas: np.ndarray,
    current_canvas: np.ndarray,
) -> float:
    """Coverage-weighted visual reward in [-1, 1].

    Args:
        predicted_canvas: float32 RGB [0, 1] canvas after executing the
            generated actions. Shape: (H, W, 3).
        gt_next_canvas: float32 RGB [0, 1] ground-truth next-state canvas.
            Must not be None — terminal rows are filtered out of the RL
            dataset upstream.
        current_canvas: float32 RGB [0, 1] canvas before any actions were
            applied (the observation the model received).

    Returns:
        Scalar reward in [-1, 1].
    """
    assert gt_next_canvas is not None, (
        "visual_reward called with gt_next_canvas=None; terminal rows must be "
        "filtered from the RL dataset before training (see rl_train._build_grpo_dataset)."
    )
    assert predicted_canvas.shape == current_canvas.shape == gt_next_canvas.shape, (
        f"Shape mismatch: pred={predicted_canvas.shape} "
        f"current={current_canvas.shape} gt={gt_next_canvas.shape}"
    )

    gt_diff = np.abs(gt_next_canvas - current_canvas).max(axis=2)   # (H, W)
    gt_mask = gt_diff > _CHANGE_THRESHOLD                            # (H, W) bool
    n_gt = int(gt_mask.sum())
    assert n_gt >= _MIN_GT_PIXELS, (
        f"Row has only {n_gt} GT-changed pixels; should have been filtered out."
    )

    pred_diff = np.abs(predicted_canvas - current_canvas).max(axis=2)
    pred_changed = pred_diff > _CHANGE_THRESHOLD

    coverage = float((pred_changed & gt_mask).sum()) / float(n_gt)

    region_mae = float(np.abs(predicted_canvas[gt_mask] - gt_next_canvas[gt_mask]).mean())
    accuracy = 1.0 - region_mae

    outside = ~gt_mask
    if outside.any():
        spurious = float(np.abs(predicted_canvas[outside] - current_canvas[outside]).mean())
    else:
        spurious = 0.0

    reward = coverage * accuracy - 0.1 * spurious
    return float(np.clip(reward, -1.0, 1.0))
