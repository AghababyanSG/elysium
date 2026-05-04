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


if __name__ == "__main__":
    unittest.main()
