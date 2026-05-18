"""Phase 7.2: SigLIP-based drawing critic.

Frozen SigLIP image + text encoders, trainable 2-layer MLP head over the
concatenated embeddings. Produces ``P(canvas is plausibly human-drawn for
instruction X)``.

Used as auxiliary RL reward in Phase 7.4 (``critic_reward`` in
``reward.py``) and as the sole reward in Phase 8 (GT-free regime).
"""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch import nn

from elysium.log import logger

__all__ = ["DrawingCritic", "DEFAULT_SIGLIP_MODEL"]

DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"


class DrawingCritic(nn.Module):
    """Bi-encoder (SigLIP image + text) → concat → MLP → sigmoid.

    The encoders stay frozen; only the MLP head trains. For Qwen3.5-VL we
    have to keep ourselves on the small-params budget anyway — the head is
    ~400K params, the encoders are ~200M each but `requires_grad=False`.

    Args:
        model_name: SigLIP model id (resolved via ``transformers.AutoModel``).
        hidden_dim: width of the MLP middle layer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        hidden_dim: int = 256,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading SigLIP encoder: {}", model_name)
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.encoder = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # Probe embedding dims by running a single forward at init time.
        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            inputs = self.processor(
                text=["probe"],
                images=[dummy_img],
                return_tensors="pt",
                padding="max_length",
            )
            out = self.encoder(**inputs)
            img_dim = int(out.image_embeds.shape[-1])
            txt_dim = int(out.text_embeds.shape[-1])
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.model_name = model_name

        in_dim = img_dim + txt_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        n_trainable = sum(p.numel() for p in self.head.parameters())
        logger.info(
            "DrawingCritic head: in={} (img={} + txt={}), hidden={}, "
            "trainable_params={}",
            in_dim, img_dim, txt_dim, hidden_dim, n_trainable,
        )

    @torch.inference_mode()
    def _encode_inference(
        self, images: list[Image.Image], instructions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """No-grad encoder pass (for eval/inference)."""
        inputs = self.processor(
            text=instructions,
            images=images,
            return_tensors="pt",
            padding="max_length",
        ).to(self.head[0].weight.device)
        out = self.encoder(**inputs)
        return out.image_embeds, out.text_embeds

    def encode(
        self, images: list[Image.Image], instructions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder pass that respects no_grad on the frozen tower but lets
        the surrounding training loop's mixed precision / autograd context
        flow through normally. The encoders' ``requires_grad=False`` keeps
        gradients off the SigLIP params regardless.
        """
        inputs = self.processor(
            text=instructions,
            images=images,
            return_tensors="pt",
            padding="max_length",
        ).to(self.head[0].weight.device)
        with torch.no_grad():
            out = self.encoder(**inputs)
        return out.image_embeds, out.text_embeds

    def forward(
        self,
        images: list[Image.Image] | torch.Tensor,
        instructions: list[str] | None = None,
        *,
        precomputed: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return ``P(plausible)`` in [0, 1] of shape ``[B]``.

        Two call modes:
          * ``critic(images, instructions)`` — standard, encodes then heads.
          * ``critic(None, None, precomputed=(img_emb, txt_emb))`` — for
            ablation / fast-path where embeddings are cached.
        """
        if precomputed is not None:
            img_emb, txt_emb = precomputed
        else:
            assert isinstance(images, list) and instructions is not None
            img_emb, txt_emb = self.encode(images, instructions)
        z = torch.cat([img_emb, txt_emb], dim=-1)
        logit = self.head(z).squeeze(-1)
        return torch.sigmoid(logit)

    def save_head(self, path: str) -> None:
        """Save only the trainable head + metadata (encoders stay on HF hub)."""
        import json
        from pathlib import Path

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), p / "head.pt")
        meta = {
            "model_name": self.model_name,
            "img_dim": self.img_dim,
            "txt_dim": self.txt_dim,
            "hidden_dim": int(self.head[0].out_features),
        }
        (p / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("Saved critic head to {}", p)

    @classmethod
    def load(cls, path: str, local_files_only: bool = False) -> "DrawingCritic":
        import json
        from pathlib import Path

        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        critic = cls(
            model_name=meta["model_name"],
            hidden_dim=meta["hidden_dim"],
            local_files_only=local_files_only,
        )
        critic.head.load_state_dict(torch.load(p / "head.pt", map_location="cpu"))
        logger.info("Loaded critic head from {}", p)
        return critic
