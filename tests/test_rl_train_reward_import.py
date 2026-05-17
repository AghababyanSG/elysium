from __future__ import annotations

import unittest
from pathlib import Path


class TestRlTrainRewardImport(unittest.TestCase):
    def test_rl_train_imports_visual_reward(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "src" / "elysium" / "model" / "rl_train.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn("from elysium.model.reward import visual_reward", text)
        self.assertIn("visual_reward(", text)


class TestFormatBonusGating(unittest.TestCase):
    """Phase 5.6: the +0.05 format-valid bonus is now config-gated and off by
    default. Regression test that pins the gating in source so the §5.1
    termination-breaker can't sneak back in via a stray hard-code.
    """

    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.rl_train_text = (
            root / "src" / "elysium" / "model" / "rl_train.py"
        ).read_text(encoding="utf-8")
        self.config_text = (
            root / "configs" / "train.yaml"
        ).read_text(encoding="utf-8")

    def test_make_reward_fn_accepts_format_bonus_param(self) -> None:
        self.assertIn(
            "def _make_reward_fn(horizon: int, format_bonus: float = 0.0)",
            self.rl_train_text,
        )

    def test_bonus_is_gated_by_format_bonus_value(self) -> None:
        # Must check the truthiness of format_bonus AND the non-terminal flag.
        # Pre-§5.6 code unconditionally added +0.05 on non-terminal.
        self.assertIn(
            "if format_bonus and not pred_chunk.is_terminal:",
            self.rl_train_text,
        )

    def test_no_hard_coded_005_add_remains(self) -> None:
        # Specifically guard against the pre-§5.6 line
        # `r = float(np.clip(r + 0.05, -1.0, 1.0))`.
        self.assertNotIn("r + 0.05", self.rl_train_text)

    def test_run_rl_training_reads_format_bonus_from_config(self) -> None:
        self.assertIn(
            'format_bonus = float(rl_cfg.get("format_bonus", 0.0))',
            self.rl_train_text,
        )

    def test_config_defaults_format_bonus_to_zero(self) -> None:
        self.assertIn("format_bonus: 0.0", self.config_text)


if __name__ == "__main__":
    unittest.main()
