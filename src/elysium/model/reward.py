"""Visual reward for GRPO training.

Computes SSIM between the canvas produced by executing generated actions and
the ground-truth next canvas from the recording.

Reward range: [-1, 1].
"""

from __future__ import annotations

import numpy as np

from elysium.schemas.actions import CANVAS_SIZE

__all__ = ["visual_reward"]


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified single-channel SSIM.

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
        predicted_canvas: float32 RGB [0, 1] canvas after executing generated actions.
            Shape: (H, W, 3).
        gt_next_canvas: float32 RGB [0, 1] ground-truth next-state canvas.
            Same shape as predicted_canvas.

    Returns:
        Mean per-channel SSIM in [-1, 1].
    """
    assert predicted_canvas.shape == gt_next_canvas.shape, (
        f"Canvas shape mismatch: {predicted_canvas.shape} vs {gt_next_canvas.shape}"
    )
    scores = [
        _ssim(predicted_canvas[:, :, c], gt_next_canvas[:, :, c]) for c in range(3)
    ]
    return float(np.mean(scores))
