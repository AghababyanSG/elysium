from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

__all__ = [
    "CANVAS_SIZE",
    "BrushAction",
    "PencilAction",
    "EraserAction",
    "FillAction",
    "ColorAdjustAction",
    "TextOverlayAction",
    "GaussianBlurAction",
    "CloneStampAction",
    "ScatterBrushAction",
    "PatternBrushAction",
    "NoopAction",
    "Action",
    "ActionChunk",
    "SYSTEM_PROMPT",
]

CANVAS_SIZE = 512

_STAMP_SHAPES = {"circle", "leaf", "star", "triangle", "dash"}

_SYSTEM_PROMPT_COORD = f"0-{CANVAS_SIZE}"
SYSTEM_PROMPT = (
    "You are a canvas drawing assistant. "
    "Given an image of a canvas and a user instruction, respond with ONLY a JSON object "
    "specifying exactly 5 sequential drawing actions to apply to the canvas. "
    "Do not explain, describe, or add any text outside the JSON.\n\n"
    "Output format:\n"
    '{"actions":[{...},{...},{...},{...},{...}]}\n\n'
    "Available action types and their required fields:\n"
    '- "brush": color_rgba ([R,G,B,A] ints 0-255), stroke_size (int 1-50), '
    "hardness (int 0-100, 0 = invisible, 100 = hard edge like a stamp), "
    f"trajectory ([[x,y],...] pixel coords {_SYSTEM_PROMPT_COORD})\n"
    f'- "pencil": color_rgba ([R,G,B,A] ints 0-255), trajectory ([[x,y],...] pixel coords {_SYSTEM_PROMPT_COORD})\n'
    f'- "eraser": stroke_size (int 1-50), trajectory ([[x,y],...] pixel coords {_SYSTEM_PROMPT_COORD})\n'
    f'- "fill": color_rgba ([R,G,B,A] ints 0-255), position ([x,y] pixel coords {_SYSTEM_PROMPT_COORD})\n'
    "- \"color_adjust\": brightness (int -100 to 100), contrast (float 0.5-2.0), "
    "saturation (float 0.0-2.0), exposure (int -100 to 100, default 0), "
    "highlights (int -100 to 100, default 0), shadows (int -100 to 100, default 0), "
    "hue_shift (int -180 to 180, default 0), temperature (int -100 to 100, default 0)\n"
    "- \"noop\": no additional fields — use when no more drawing is needed\n"
    f'- "text_overlay": text (str), position ([x,y] pixel coords {_SYSTEM_PROMPT_COORD}), '
    "font_name (str, one of: simplex, duplex, complex, triplex, script; default simplex), "
    "font_size (float 0.2-5.0, default 1.0), color_rgba ([R,G,B,A] ints 0-255), "
    "thickness (int 1-10, default 1)\n"
    "- \"gaussian_blur\": radius (int 1-31, kernel = 2*radius+1; default 5)\n"
    f'- "clone_stamp": source ([x,y] pixel coords {_SYSTEM_PROMPT_COORD}), destination ([x,y] pixel coords {_SYSTEM_PROMPT_COORD}), '
    "size (int 1-50 radius in pixels; default 10)\n"
    '- "scatter_brush": shape (str, one of: circle, leaf, star, triangle, dash; default circle), '
    f"color_rgba ([R,G,B,A] ints 0-255), trajectory ([[x,y],...] pixel coords {_SYSTEM_PROMPT_COORD}), "
    "size (int 1-50 base stamp size; default 8), density (int 1-20 stamps per step; default 5), "
    "scatter (int 0-100 scatter distance percent; default 30), "
    "size_jitter (int 0-100 size variation percent; default 50), "
    "angle_jitter (int 0-360: for shape dash, max degrees added to stroke tangent; 0 = follow stroke; "
    "for other shapes, max rotation in [0,angle_jitter] or full 360 if 0; default 0), seed (int; default 0), "
    "thickness (int 1-10 dash line width; default 1), length (int 0-100 dash half-length override; 0 = use size; default 0), "
    "base_angle (int -1 or 0-360; -1 = align dash to stroke tangent; else fixed base degrees; default -1)\n"
    '- "pattern_brush": shape (str, one of: circle, leaf, star, triangle, dash; default leaf), '
    f"color_rgba ([R,G,B,A] ints 0-255), trajectory ([[x,y],...] pixel coords {_SYSTEM_PROMPT_COORD}), "
    "size (int 1-50 stamp size; default 10), spacing (int 5-100 pixels between stamps; default 20), "
    "angle_jitter (int 0-90 rotation variation per stamp; default 15), "
    "thickness (int 1-10 dash line width; default 1), length (int 0-100 dash half-length override; 0 = use size; default 0)\n\n"
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
            assert 0 <= x <= CANVAS_SIZE and 0 <= y <= CANVAS_SIZE, f"Coordinate ({x},{y}) out of canvas bounds"
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


_FONT_NAMES = {"simplex", "duplex", "complex", "triplex", "script"}


class TextOverlayAction(BaseModel):
    action_type: Literal["text_overlay"]
    text: str
    position: tuple[int, int]
    font_name: str = "simplex"
    font_size: float = 1.0
    color_rgba: tuple[int, int, int, int]
    thickness: int = 1

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x <= CANVAS_SIZE and 0 <= y <= CANVAS_SIZE, f"Position ({x},{y}) out of canvas bounds"
        return v

    @field_validator("font_name")
    @classmethod
    def _validate_font_name(cls, v: str) -> str:
        assert v in _FONT_NAMES, f"font_name must be one of {sorted(_FONT_NAMES)}"
        return v

    @field_validator("font_size")
    @classmethod
    def _validate_font_size(cls, v: float) -> float:
        assert 0.2 <= v <= 5.0, "font_size must be in [0.2, 5.0]"
        return v

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("thickness")
    @classmethod
    def _validate_thickness(cls, v: int) -> int:
        assert 1 <= v <= 10, "thickness must be in [1, 10]"
        return v


class GaussianBlurAction(BaseModel):
    action_type: Literal["gaussian_blur"]
    radius: int = 5

    @field_validator("radius")
    @classmethod
    def _validate_radius(cls, v: int) -> int:
        assert 1 <= v <= 31, "radius must be in [1, 31]"
        return v


class CloneStampAction(BaseModel):
    action_type: Literal["clone_stamp"]
    source: tuple[int, int]
    destination: tuple[int, int]
    size: int = 10

    @field_validator("source", "destination")
    @classmethod
    def _validate_coords(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x <= CANVAS_SIZE and 0 <= y <= CANVAS_SIZE, f"Coordinate ({x},{y}) out of canvas bounds"
        return v

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50, "size must be in [1, 50]"
        return v


class ScatterBrushAction(BaseModel):
    action_type: Literal["scatter_brush"]
    shape: str = "circle"
    color_rgba: tuple[int, int, int, int]
    trajectory: list[tuple[int, int]]
    size: int = 8
    density: int = 5
    scatter: int = 30
    size_jitter: int = 50
    angle_jitter: int = 0
    seed: int = 0
    thickness: int = 1
    length: int = 0
    base_angle: int = -1

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, v: str) -> str:
        assert v in _STAMP_SHAPES, f"shape must be one of {sorted(_STAMP_SHAPES)}"
        return v

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

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50
        return v

    @field_validator("density")
    @classmethod
    def _validate_density(cls, v: int) -> int:
        assert 1 <= v <= 20
        return v

    @field_validator("scatter")
    @classmethod
    def _validate_scatter(cls, v: int) -> int:
        assert 0 <= v <= 100
        return v

    @field_validator("size_jitter")
    @classmethod
    def _validate_size_jitter(cls, v: int) -> int:
        assert 0 <= v <= 100
        return v

    @field_validator("angle_jitter")
    @classmethod
    def _validate_angle_jitter(cls, v: int) -> int:
        assert 0 <= v <= 360
        return v

    @field_validator("thickness")
    @classmethod
    def _validate_thickness(cls, v: int) -> int:
        assert 1 <= v <= 10
        return v

    @field_validator("length")
    @classmethod
    def _validate_length(cls, v: int) -> int:
        assert 0 <= v <= 100
        return v

    @field_validator("base_angle")
    @classmethod
    def _validate_base_angle(cls, v: int) -> int:
        assert v == -1 or 0 <= v <= 360
        return v


class PatternBrushAction(BaseModel):
    action_type: Literal["pattern_brush"]
    shape: str = "leaf"
    color_rgba: tuple[int, int, int, int]
    trajectory: list[tuple[int, int]]
    size: int = 10
    spacing: int = 20
    angle_jitter: int = 15
    thickness: int = 1
    length: int = 0

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, v: str) -> str:
        assert v in _STAMP_SHAPES, f"shape must be one of {sorted(_STAMP_SHAPES)}"
        return v

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

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50
        return v

    @field_validator("spacing")
    @classmethod
    def _validate_spacing(cls, v: int) -> int:
        assert 5 <= v <= 100
        return v

    @field_validator("angle_jitter")
    @classmethod
    def _validate_angle_jitter(cls, v: int) -> int:
        assert 0 <= v <= 90
        return v

    @field_validator("thickness")
    @classmethod
    def _validate_thickness(cls, v: int) -> int:
        assert 1 <= v <= 10
        return v

    @field_validator("length")
    @classmethod
    def _validate_length(cls, v: int) -> int:
        assert 0 <= v <= 100
        return v


class NoopAction(BaseModel):
    action_type: Literal["noop"]


Action = (
    BrushAction
    | PencilAction
    | EraserAction
    | FillAction
    | ColorAdjustAction
    | TextOverlayAction
    | GaussianBlurAction
    | CloneStampAction
    | ScatterBrushAction
    | PatternBrushAction
    | NoopAction
)


def parse_action(data: dict) -> Action:
    """Deserialize a raw dict into the correct Action subtype."""
    tool = data.get("action_type")
    dispatch: dict[str, type[Action]] = {
        "brush": BrushAction,
        "pencil": PencilAction,
        "eraser": EraserAction,
        "fill": FillAction,
        "color_adjust": ColorAdjustAction,
        "text_overlay": TextOverlayAction,
        "gaussian_blur": GaussianBlurAction,
        "clone_stamp": CloneStampAction,
        "scatter_brush": ScatterBrushAction,
        "pattern_brush": PatternBrushAction,
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
