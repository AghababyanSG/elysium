from __future__ import annotations

from typing import Any

__all__ = ["run_training", "Predictor", "run_inference"]


def __getattr__(name: str) -> Any:
    if name == "run_training":
        from elysium.model.train import run_training

        return run_training
    if name == "Predictor":
        from elysium.model.predict import Predictor

        return Predictor
    if name == "run_inference":
        from elysium.model.predict import run_inference

        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
