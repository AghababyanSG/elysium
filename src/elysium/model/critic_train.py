"""Phase 7.3: train the SigLIP DrawingCritic on the harvest-augmented
positive/negative dataset built by ``scripts/prepare_critic_data.py``.

Loss: BCE on the sigmoid head output.
Sampler: mixed 1:3 positive:negative ratio per batch (the global dataset
is closer to 1:5 because of how many harvested negatives we have; without
the mixed sampler the gradient on positives would be tiny).

Saves only the trainable MLP head (~400 KB) + a meta.json that pins the
SigLIP model name. The frozen encoder weights stay on the HF cache.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from elysium.log import logger
from elysium.model.critic import DEFAULT_SIGLIP_MODEL, DrawingCritic

__all__ = ["run_training", "MixedRatioSampler"]


class MixedRatioSampler(Sampler[int]):
    """Yield ``batch_size`` indices per batch with a fixed positive ratio.

    The dataset is unbalanced (1:5 positive:negative), so a uniform sampler
    gives BCE gradients dominated by negatives. With this sampler each
    batch has ``int(round(pos_ratio * batch_size))`` positives and the
    rest negatives, drawn with replacement from their respective pools.
    """

    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        pos_ratio: float = 0.25,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.pos_idx = [i for i, l in enumerate(labels) if l == 1]
        self.neg_idx = [i for i, l in enumerate(labels) if l == 0]
        assert self.pos_idx and self.neg_idx, "need both positive and negative indices"
        self.batch_size = batch_size
        self.n_pos = max(1, int(round(pos_ratio * batch_size)))
        self.n_neg = batch_size - self.n_pos
        if num_batches is None:
            # One "epoch" = enough batches to see every record once on average.
            num_batches = max(1, (len(self.pos_idx) + len(self.neg_idx)) // batch_size)
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            pos = self.rng.choice(self.pos_idx, size=self.n_pos, replace=True)
            neg = self.rng.choice(self.neg_idx, size=self.n_neg, replace=True)
            batch = np.concatenate([pos, neg])
            self.rng.shuffle(batch)
            for i in batch:
                yield int(i)

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": [b["image"] for b in batch],
        "instruction": [b["instruction"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "mode": [b["mode"] for b in batch],
    }


def _per_mode_accuracy(
    preds: list[float], labels: list[int], modes: list[str], threshold: float = 0.5
) -> dict[str, float]:
    out: dict[str, list[int]] = {}
    for p, l, m in zip(preds, labels, modes):
        out.setdefault(m, []).append(int((p >= threshold) == bool(l)))
    return {m: float(np.mean(v)) for m, v in out.items()}


def _auc(preds: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, preds))


def run_training(
    dataset_dir: Path = Path("data/processed_critic"),
    output_dir: Path = Path("models/critic"),
    siglip_model: str = DEFAULT_SIGLIP_MODEL,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    pos_ratio: float = 0.5,
    pos_loss_weight: float = 1.5,
    hidden_dim: int = 256,
    tb_root: Path = Path("logs/tensorboard"),
) -> None:
    ds = load_from_disk(str(dataset_dir))
    train_ds = ds["train"]
    val_ds = ds["validation"]
    logger.info("Critic dataset: {} train, {} val", len(train_ds), len(val_ds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    critic = DrawingCritic(model_name=siglip_model, hidden_dim=hidden_dim).to(device)
    critic.head.train()

    train_labels = list(train_ds["label"])
    sampler = MixedRatioSampler(train_labels, batch_size=batch_size, pos_ratio=pos_ratio)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=None,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=_collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
    )

    optim = torch.optim.AdamW(critic.head.parameters(), lr=lr, weight_decay=0.01)
    # Class-weighted BCE: scale per-element loss by pos_loss_weight on
    # positives. Initial run (pos_ratio=0.25, plain BCE) trained the head
    # into a strong negative-bias state (positive scores capped at 0.47,
    # never crossing the 0.5 threshold). Weighting + 50/50 batches gives
    # symmetric gradient pressure.
    loss_fn = torch.nn.BCELoss(reduction="none")
    logger.info(
        "Loss: weighted BCE (pos_loss_weight={}), sampler pos_ratio={}",
        pos_loss_weight, pos_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = tb_root / f"critic_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(log_dir))
    logger.info("TensorBoard logs -> {}", log_dir)

    best_auc = -1.0
    global_step = 0
    for epoch in range(epochs):
        critic.head.train()
        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            preds = critic(batch["image"], batch["instruction"])
            labels = batch["label"].to(preds.device)
            per_elem = loss_fn(preds, labels)
            weights = torch.where(
                labels > 0.5,
                torch.full_like(per_elem, pos_loss_weight),
                torch.ones_like(per_elem),
            )
            loss = (per_elem * weights).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.head.parameters(), 1.0)
            optim.step()

            running_loss += float(loss.detach()) * len(labels)
            n_seen += len(labels)
            tb.add_scalar("train/loss", float(loss.detach()), global_step)
            global_step += 1
        avg_train_loss = running_loss / max(1, n_seen)
        tb.add_scalar("train/loss_epoch_mean", avg_train_loss, epoch + 1)

        # Eval.
        critic.head.eval()
        all_preds: list[float] = []
        all_labels: list[int] = []
        all_modes: list[str] = []
        with torch.no_grad():
            for batch in val_loader:
                preds = critic(batch["image"], batch["instruction"])
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(batch["label"].tolist())
                all_modes.extend(batch["mode"])
        auc = _auc(all_preds, all_labels)
        per_mode = _per_mode_accuracy(all_preds, all_labels, all_modes)
        val_loss = float(
            torch.nn.functional.binary_cross_entropy(
                torch.tensor(all_preds),
                torch.tensor(all_labels, dtype=torch.float32),
            )
        )
        tb.add_scalar("eval/loss", val_loss, epoch + 1)
        tb.add_scalar("eval/auc", auc, epoch + 1)
        for m, a in per_mode.items():
            tb.add_scalar(f"eval/acc_{m}", a, epoch + 1)
        logger.info(
            "Epoch {}/{}  train_loss={:.4f}  val_loss={:.4f}  AUC={:.4f}  "
            "per-mode acc: {}",
            epoch + 1, epochs, avg_train_loss, val_loss, auc,
            {k: f"{v:.3f}" for k, v in per_mode.items()},
        )

        if auc > best_auc:
            best_auc = auc
            critic.save_head(str(output_dir))
            logger.info("New best AUC {:.4f} - saved to {}", best_auc, output_dir)

    tb.close()
    logger.info("Training complete. Best AUC: {:.4f}", best_auc)
