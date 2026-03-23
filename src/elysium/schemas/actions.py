from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

__all__ = [
    "BrushAction",
    "PencilAction",
    "EraserAction",
    "FillAction",
    "ColorAdjustAction",
    "NoopAction",
    "Action",
    "ActionChunk",
    "SYSTEM_PROMPT",
]

SYSTEM_PROMPT = (
    "You are a canvas drawing assistant. "
    "Given an image of a canvas and a user instruction, respond with ONLY a JSON object "
    "specifying exactly 5 sequential drawing actions to apply to the canvas. "
    "Do not explain, describe, or add any text outside the JSON.\n\n"
    "Output format:\n"
    "{\"actions\":[{...},{...},{...},{...},{...}]}\n\n"
    "Available action types and their required fields:\n"
    "- \"brush\": color_rgba ([R,G,B,A] ints 0-255), stroke_size (int 1-50), "
    "hardness (int 0-100, 0 = invisible, 100 = hard edge like a stamp), "
    "trajectory ([[x,y],...] pixel coords 0-256)\n"
    "- \"pencil\": color_rgba ([R,G,B,A] ints 0-255), trajectory ([[x,y],...] pixel coords 0-256)\n"
    "- \"eraser\": stroke_size (int 1-50), trajectory ([[x,y],...] pixel coords 0-256)\n"
    "- \"fill\": color_rgba ([R,G,B,A] ints 0-255), position ([x,y] pixel coords 0-256)\n"
    "- \"color_adjust\": brightness (int -100 to 100), contrast (float 0.5-2.0), "
    "saturation (float 0.0-2.0), exposure (int -100 to 100, default 0), "
    "highlights (int -100 to 100, default 0), shadows (int -100 to 100, default 0), "
    "hue_shift (int -180 to 180, default 0), temperature (int -100 to 100, default 0)\n"
    "- \"noop\": no additional fields — use when no more drawing is needed\n\n"
    "Respond with valid JSON only."
)

class BrushAction(BaseModel):
    action_type: Literal["brush"]
    color_rgba: tuple[int, int, int, int]
    stroke_size: int
    trajectory: list[tuple[int, int]]
    hardness: int = 100

    @model_validator(mode="before")
    @classmethod
    def _legacy_color_rgb(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "color_rgba" not in data and "color_rgb" in data:
            cr = data["color_rgb"]
            if isinstance(cr, (list, tuple)) and len(cr) == 3:
                data = {k: v for k, v in data.items() if k != "color_rgb"}
                data["color_rgba"] = (int(cr[0]), int(cr[1]), int(cr[2]), 255)
        if "hardness" not in data:
            data = {**data, "hardness": 100}
        return data

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("stroke_size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50, "Stroke size must be in [1, 50]"
        return v

    @field_validator("hardness")
    @classmethod
    def _validate_hardness(cls, v: int) -> int:
        assert 0 <= v <= 100, "Hardness must be in [0, 100]"
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
    color_rgba: tuple[int, int, int, int]
    trajectory: list[tuple[int, int]]

    @model_validator(mode="before")
    @classmethod
    def _legacy_color_rgb(cls, data: Any) -> Any:
        if isinstance(data, dict) and "color_rgba" not in data and "color_rgb" in data:
            cr = data["color_rgb"]
            if isinstance(cr, (list, tuple)) and len(cr) == 3:
                data = {k: v for k, v in data.items() if k != "color_rgb"}
                data["color_rgba"] = (int(cr[0]), int(cr[1]), int(cr[2]), 255)
        return data

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
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
    color_rgba: tuple[int, int, int, int]
    position: tuple[int, int]

    @model_validator(mode="before")
    @classmethod
    def _legacy_color_rgb(cls, data: Any) -> Any:
        if isinstance(data, dict) and "color_rgba" not in data and "color_rgb" in data:
            cr = data["color_rgb"]
            if isinstance(cr, (list, tuple)) and len(cr) == 3:
                data = {k: v for k, v in data.items() if k != "color_rgb"}
                data["color_rgba"] = (int(cr[0]), int(cr[1]), int(cr[2]), 255)
        return data

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v


class ColorAdjustAction(BaseModel):
    action_type: Literal["color_adjust"]
    brightness: int
    contrast: float
    saturation: float
    exposure: int = 0
    highlights: int = 0
    shadows: int = 0
    hue_shift: int = 0
    temperature: int = 0

    @field_validator("brightness")
    @classmethod
    def _validate_brightness(cls, v: int) -> int:
        assert -100 <= v <= 100
        return v

    @field_validator("contrast")
    @classmethod
    def _validate_contrast(cls, v: float) -> float:
        assert 0.5 <= v <= 2.0
        return v

    @field_validator("saturation")
    @classmethod
    def _validate_saturation(cls, v: float) -> float:
        assert 0.0 <= v <= 2.0
        return v

    @field_validator("exposure")
    @classmethod
    def _validate_exposure(cls, v: int) -> int:
        assert -100 <= v <= 100
        return v

    @field_validator("highlights")
    @classmethod
    def _validate_highlights(cls, v: int) -> int:
        assert -100 <= v <= 100
        return v

    @field_validator("shadows")
    @classmethod
    def _validate_shadows(cls, v: int) -> int:
        assert -100 <= v <= 100
        return v

    @field_validator("hue_shift")
    @classmethod
    def _validate_hue_shift(cls, v: int) -> int:
        assert -180 <= v <= 180
        return v

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: int) -> int:
        assert -100 <= v <= 100
        return v


class NoopAction(BaseModel):
    action_type: Literal["noop"]


Action = BrushAction | PencilAction | EraserAction | FillAction | ColorAdjustAction | NoopAction


def parse_action(data: dict) -> Action:
    """Deserialize a raw dict into the correct Action subtype."""
    tool = data.get("action_type")
    dispatch: dict[str, type[Action]] = {
        "brush": BrushAction,
        "pencil": PencilAction,
        "eraser": EraserAction,
        "fill": FillAction,
        "color_adjust": ColorAdjustAction,
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
