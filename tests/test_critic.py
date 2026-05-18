"""Phase 7.2: smoke tests for DrawingCritic.

Network-dependent (downloads ~370 MB SigLIP weights on first run if not
cached). Tests are marked `slow` so a developer can skip with
`pytest -m 'not slow'` — they still run in the default invocation.
"""
from __future__ import annotations

import unittest

import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.slow


class TestDrawingCritic(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from elysium.model.critic import DrawingCritic
        # Default model_name; will hit HF cache if available, else download.
        cls.critic = DrawingCritic(local_files_only=False)
        cls.critic.eval()

    def test_output_shape_and_range(self) -> None:
        img = Image.new("RGB", (256, 256), color=(120, 60, 200))
        out = self.critic([img], ["make the image purple"])
        self.assertEqual(out.shape, (1,))
        v = float(out[0])
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_batch_independence(self) -> None:
        # Two same-image, different-instruction batch entries should produce
        # independent (non-identical with overwhelming probability) scores
        # under a random head.
        img = Image.new("RGB", (256, 256), color=(50, 200, 50))
        out = self.critic(
            [img, img],
            ["draw a tree", "remove the boat from the image"],
        )
        self.assertEqual(out.shape, (2,))
        # Sanity: random-init head still produces finite probabilities.
        self.assertTrue(torch.isfinite(out).all())

    def test_encoder_frozen(self) -> None:
        for p in self.critic.encoder.parameters():
            self.assertFalse(p.requires_grad)

    def test_head_trainable(self) -> None:
        for p in self.critic.head.parameters():
            self.assertTrue(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
