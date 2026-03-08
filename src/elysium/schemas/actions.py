from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, field_validator

__all__ = ["BrushAction", "PencilAction", "EraserAction", "FillAction", "NoopAction", "Action", "ActionChunk"]

_TOOLS = Literal["brush", "pencil", "eraser", "fill", "noop"]


class BrushAction(BaseModel):
    action_type: Literal["brush"]
    color_rgb: tuple[int, int, int]
    stroke_size: int
    trajectory: list[tuple[int, int]]

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("stroke_size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50, "Stroke size must be in [1, 50]"
        return v

    @field_validator("trajectory")
    @classmethod
    def _validate_trajectory(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        assert len(v) >= 1, "Trajectory must have at least one point"
        for x, y in v:
            assert 0 <= x <= 256 and 0 <= y <= 256, f"Coordinate ({x},{y}) out of canvas bounds"
        return v


class PencilAction(BaseModel):
    action_type: Literal["pencil"]
    color_rgb: tuple[int, int, int]
    trajectory: list[tuple[int, int]]

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("trajectory")
    @classmethod
    def _validate_trajectory(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        assert len(v) >= 1, "Trajectory must have at least one point"
        return v


class EraserAction(BaseModel):
    action_type: Literal["eraser"]
    stroke_size: int
    trajectory: list[tuple[int, int]]

    @field_validator("stroke_size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50, "Stroke size must be in [1, 50]"
        return v


class FillAction(BaseModel):
    action_type: Literal["fill"]
    color_rgb: tuple[int, int, int]
    position: tuple[int, int]

    @field_validator("color_rgb")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v


class NoopAction(BaseModel):
    action_type: Literal["noop"]


Action = BrushAction | PencilAction | EraserAction | FillAction | NoopAction


def parse_action(data: dict) -> Action:
    """Deserialize a raw dict into the correct Action subtype."""
    tool = data.get("action_type")
    dispatch: dict[str, type[Action]] = {
        "brush": BrushAction,
        "pencil": PencilAction,
        "eraser": EraserAction,
        "fill": FillAction,
        "noop": NoopAction,
    }
    if tool not in dispatch:
        raise ValueError(f"Unknown action_type '{tool}'. Valid: {list(dispatch)}")
    return dispatch[tool].model_validate(data)


class ActionChunk(BaseModel):
    """A fixed-length sequence of actions produced by the model per inference step."""

    actions: list[Action]
    horizon: int = 5

    @field_validator("actions")
    @classmethod
    def _validate_length(cls, v: list[Action]) -> list[Action]:
        assert len(v) > 0, "ActionChunk must contain at least one action"
        return v

    @property
    def is_terminal(self) -> bool:
        """True when all actions in the chunk are noops -- signals edit completion."""
        return all(a.action_type == "noop" for a in self.actions)

    def to_json_str(self) -> str:
        """Serialize to the JSON string format used in training targets."""
        return json.dumps({"actions": [a.model_dump() for a in self.actions]}, separators=(",", ":"))

    @classmethod
    def from_json_str(cls, raw: str, horizon: int = 5) -> "ActionChunk":
        """Parse model-generated JSON string into an ActionChunk."""
        data = json.loads(raw)
        actions = [parse_action(a) for a in data["actions"]]
        return cls(actions=actions, horizon=horizon)

    @classmethod
    def noop_chunk(cls, horizon: int = 5) -> "ActionChunk":
        """Return a chunk filled entirely with noop actions."""
        return cls(actions=[NoopAction(action_type="noop")] * horizon, horizon=horizon)
