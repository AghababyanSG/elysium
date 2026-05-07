from __future__ import annotations

import unittest

import numpy as np

from elysium.model.reward import visual_reward


def _blank(size: int = 64) -> np.ndarray:
    return np.ones((size, size, 3), dtype=np.float32)


def _paint_square(
    canvas: np.ndarray, y0: int, x0: int, side: int, color: tuple[float, float, float]
) -> np.ndarray:
    out = canvas.copy()
    out[y0:y0 + side, x0:x0 + side] = color
    return out


class TestVisualReward(unittest.TestCase):
    def setUp(self) -> None:
        self.current = _blank()
        self.gt = _paint_square(self.current, 10, 10, 16, (1.0, 0.0, 0.0))

    def test_pixel_perfect_prediction_scores_one(self) -> None:
        r = visual_reward(self.gt, self.gt, self.current)
        self.assertAlmostEqual(r, 1.0, places=4)

    def test_noop_prediction_scores_zero_or_less(self) -> None:
        r = visual_reward(self.current, self.gt, self.current)
        self.assertLessEqual(r, 0.0)

    def test_partial_overlap_partial_color_drift_in_mid_range(self) -> None:
        pred = _paint_square(self.current, 10, 10, 16, (0.5, 0.0, 0.0))
        r = visual_reward(pred, self.gt, self.current)
        self.assertGreater(r, 0.3)
        self.assertLess(r, 0.9)

    def test_wrong_region_scores_negative(self) -> None:
        pred = _paint_square(self.current, 40, 40, 16, (1.0, 0.0, 0.0))
        r = visual_reward(pred, self.gt, self.current)
        self.assertLess(r, 0.0)

    def test_spurious_changes_reduce_reward(self) -> None:
        clean = self.gt
        with_spurious = _paint_square(self.gt, 50, 50, 8, (0.0, 1.0, 0.0))
        r_clean = visual_reward(clean, self.gt, self.current)
        r_spurious = visual_reward(with_spurious, self.gt, self.current)
        self.assertLess(r_spurious, r_clean)

    def test_terminal_row_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            visual_reward(self.current, self.current, self.current)

    def test_none_gt_asserts(self) -> None:
        with self.assertRaises(AssertionError):
            visual_reward(self.current, None, self.current)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
