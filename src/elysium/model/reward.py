"""Reward functions for REINFORCE training.

Two signals are combined into a scalar reward per generated chunk:

  action_reward  -- how closely the generated actions match the ground-truth
                    expert actions (type match + trajectory + color + size)
  visual_reward  -- SSIM between the canvas produced by executing the generated
                    actions and the ground-truth next canvas from the recording

Combined reward:
    r = alpha * action_reward + beta * visual_reward

All rewards are normalised to approximately [-1, 1].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from elysium.schemas.actions import (
    Action,
    ActionChunk,
    BrushAction,
    ColorAdjustAction,
    EraserAction,
    FillAction,
    NoopAction,
    PencilAction,
)

__all__ = ["compute_reward", "action_reward", "visual_reward"]

_CANVAS_SIZE = 256
_COLOR_MAX = math.sqrt(3 * 255**2)
_TRAJ_MAX = math.sqrt(2 * _CANVAS_SIZE**2)
_SIZE_MAX = 50.0


def _trajectory_distance(a: list[Any], b: list[Any]) -> float:
    """Mean L2 distance between two trajectories, normalised to [0, 1].

    Trajectories may have different lengths; the shorter one is padded
    with its last point before comparing.
    """
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    def to_arr(t: list[Any]) -> np.ndarray:
        return np.array([[p[0], p[1]] for p in t], dtype=float)

    arr_a, arr_b = to_arr(a), to_arr(b)
    n = max(len(arr_a), len(arr_b))

    def pad(arr: np.ndarray, length: int) -> np.ndarray:
        if len(arr) >= length:
            return arr[:length]
        pad_rows = np.tile(arr[-1], (length - len(arr), 1))
        return np.vstack([arr, pad_rows])

    arr_a, arr_b = pad(arr_a, n), pad(arr_b, n)
    dists = np.linalg.norm(arr_a - arr_b, axis=1)
    return float(np.mean(dists) / _TRAJ_MAX)


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Normalised Euclidean RGB distance in [0, 1]."""
    diff = np.array(c1, dtype=float) - np.array(c2, dtype=float)
    return float(np.linalg.norm(diff) / _COLOR_MAX)


def _size_distance(s1: int, s2: int) -> float:
    """Normalised absolute size difference in [0, 1]."""
    return abs(s1 - s2) / _SIZE_MAX


def _single_action_reward(pred: Action, gt: Action) -> float:
    """Reward for a single action pair in [-1, 1].

    +1.0 base for matching action_type, then penalties for mismatches in
    trajectory, color, and stroke_size when both actions have those fields.
    """
    type_match = pred.action_type == gt.action_type

    if not type_match:
        return -1.0

    if isinstance(pred, NoopAction):
        return 1.0

    if isinstance(pred, ColorAdjustAction) and isinstance(gt, ColorAdjustAction):
        b_diff = abs(pred.brightness - gt.brightness) / 200.0
        c_diff = abs(pred.contrast - gt.contrast) / 1.5
        s_diff = abs(pred.saturation - gt.saturation) / 2.0
        penalty = (b_diff + c_diff + s_diff) / 3.0
        return 1.0 - 2.0 * penalty

    penalty = 0.0
    terms = 0

    if isinstance(pred, (BrushAction, PencilAction, EraserAction)) and isinstance(
        gt, (BrushAction, PencilAction, EraserAction)
    ):
        penalty += _trajectory_distance(pred.trajectory, gt.trajectory)
        terms += 1

    if isinstance(pred, (BrushAction, PencilAction)) and isinstance(
        gt, (BrushAction, PencilAction)
    ):
        penalty += _color_distance(pred.color_rgb, gt.color_rgb)
        terms += 1

    if isinstance(pred, (BrushAction, EraserAction)) and isinstance(
        gt, (BrushAction, EraserAction)
    ):
        penalty += _size_distance(pred.stroke_size, gt.stroke_size)
        terms += 1

    if isinstance(pred, FillAction) and isinstance(gt, FillAction):
        penalty += _color_distance(pred.color_rgb, gt.color_rgb)
        penalty += _trajectory_distance([pred.position], [gt.position])
        terms += 2

    avg_penalty = (penalty / terms) if terms > 0 else 0.0
    return 1.0 - 2.0 * avg_penalty


def action_reward(pred_chunk: ActionChunk, gt_actions: list[dict[str, Any]]) -> float:
    """Mean per-action reward across the chunk, in approximately [-1, 1].

    Args:
        pred_chunk: Generated ActionChunk.
        gt_actions: List of ground-truth action dicts (raw, from dataset).

    Returns:
        Scalar reward in [-1, 1].
    """
    from elysium.schemas.actions import parse_action

    n = max(len(pred_chunk.actions), len(gt_actions))
    if n == 0:
        return 0.0

    noop = NoopAction(action_type="noop")
    gt_parsed = [parse_action(a) for a in gt_actions]

    rewards = []
    for i in range(n):
        pred = pred_chunk.actions[i] if i < len(pred_chunk.actions) else noop
        gt = gt_parsed[i] if i < len(gt_parsed) else noop
        rewards.append(_single_action_reward(pred, gt))

    return float(np.mean(rewards))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified single-channel SSIM in [-1, 1].

    Args:
        a: float32 array, values in [0, 1].
        b: float32 array, same shape as a.

    Returns:
        SSIM value in [-1, 1].
    """
    C1, C2 = 0.01**2, 0.03**2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = ((a - mu_a) ** 2).mean()
    sigma_b = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a**2 + mu_b**2 + C1) * (sigma_a + sigma_b + C2)
    return float(numerator / denominator)


def visual_reward(predicted_canvas: np.ndarray, gt_next_canvas: np.ndarray) -> float:
    """SSIM-based visual reward in [-1, 1].

    Args:
        predicted_canvas: float32 RGB [0,1] canvas after executing generated actions.
        gt_next_canvas: float32 RGB [0,1] ground-truth next-state canvas.

    Returns:
        SSIM score in [-1, 1].
    """
    assert predicted_canvas.shape == gt_next_canvas.shape, (
        f"Canvas shape mismatch: {predicted_canvas.shape} vs {gt_next_canvas.shape}"
    )
    scores = [
        _ssim(predicted_canvas[:, :, c], gt_next_canvas[:, :, c]) for c in range(3)
    ]
    return float(np.mean(scores))


def compute_reward(
    pred_chunk: ActionChunk,
    gt_actions: list[dict[str, Any]],
    predicted_canvas: np.ndarray | None,
    gt_next_canvas: np.ndarray | None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Compute combined reward for one generated chunk.

    Args:
        pred_chunk: Parsed generated ActionChunk.
        gt_actions: Ground-truth action dicts from the dataset.
        predicted_canvas: Canvas after executing pred_chunk, or None if skipped.
        gt_next_canvas: Ground-truth next-state canvas, or None if unavailable.
        alpha: Weight for action reward.
        beta: Weight for visual reward.

    Returns:
        Scalar reward.
    """
    r_action = action_reward(pred_chunk, gt_actions)

    if predicted_canvas is not None and gt_next_canvas is not None:
        r_visual = visual_reward(predicted_canvas, gt_next_canvas)
        return alpha * r_action + beta * r_visual

    return r_action
