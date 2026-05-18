"""Phase 7.4: regression tests for the critic-as-auxiliary-RL-reward wiring.

Static checks (no GPU, no SigLIP download) — pin the integration points
so future refactors don't silently drop the critic path.
"""
from __future__ import annotations

import unittest
from pathlib import Path


class TestCriticRewardWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.rl_train_text = (
            root / "src" / "elysium" / "model" / "rl_train.py"
        ).read_text(encoding="utf-8")
        cls.reward_text = (
            root / "src" / "elysium" / "model" / "reward.py"
        ).read_text(encoding="utf-8")
        cls.config_text = (
            root / "configs" / "train.yaml"
        ).read_text(encoding="utf-8")

    def test_critic_reward_public_api(self) -> None:
        # Public function name, batched signature.
        self.assertIn("def critic_reward(", self.reward_text)
        self.assertIn("canvases: list[np.ndarray]", self.reward_text)
        self.assertIn("instructions: list[str]", self.reward_text)
        # The rescale to [-1, 1] (matching visual_reward range).
        self.assertIn("2.0 * p - 1.0", self.reward_text)
        # Public export
        self.assertIn('"critic_reward"', self.reward_text)

    def test_make_reward_fn_accepts_critic(self) -> None:
        self.assertIn("critic: Any = None", self.rl_train_text)
        self.assertIn("critic_weight: float = 0.0", self.rl_train_text)

    def test_grpo_dataset_includes_instruction_column(self) -> None:
        # The reward fn needs the bare instruction to score-batch the critic.
        self.assertIn('"instruction": bare_instructions', self.rl_train_text)
        self.assertIn('"instruction": Value("string")', self.rl_train_text)

    def test_run_rl_training_reads_critic_config(self) -> None:
        self.assertIn(
            'critic_weight = float(rl_cfg.get("critic_weight", 0.0))',
            self.rl_train_text,
        )
        self.assertIn(
            'critic_path = rl_cfg.get("critic_path", "models/critic")',
            self.rl_train_text,
        )

    def test_config_exposes_critic_knobs(self) -> None:
        # Phase 8 GT-free defaults: visual_weight=0.0, critic_weight=1.0.
        self.assertIn("visual_weight: 0.0", self.config_text)
        self.assertIn("critic_weight: 1.0", self.config_text)
        self.assertIn('critic_path: "models/critic"', self.config_text)

    def test_make_reward_fn_accepts_visual_weight(self) -> None:
        self.assertIn("visual_weight: float = 1.0", self.rl_train_text)
        # The GT-execution skip branch — proves we don't render GT in Phase 8.
        self.assertIn("if visual_weight > 0.0:", self.rl_train_text)

    def test_run_rl_training_guards_against_no_reward(self) -> None:
        # Both weights = 0 is an obvious foot-gun (no gradient signal at all);
        # the trainer should refuse with a clear error rather than silently
        # train against a zero reward.
        self.assertIn(
            "visual_weight == 0.0 and critic_weight == 0.0",
            self.rl_train_text,
        )


if __name__ == "__main__":
    unittest.main()
