from __future__ import annotations

import json
import re
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
    "ForwardWarpAction",
    "NoopAction",
    "Action",
    "ActionChunk",
    "build_system_prompt",
]

CANVAS_SIZE = 256

_STAMP_SHAPES = {"circle", "leaf", "star", "triangle", "dash"}

_ACTION_TYPE_ALIASES: dict[str, str] = {"blur": "gaussian_blur"}

def build_system_prompt(horizon: int) -> str:
    assert horizon >= 1, f"horizon must be >= 1, got {horizon}"
    return (
        "You are a canvas drawing assistant. Output ONLY a JSON object with exactly "
        f"{horizon} drawing actions: "
        '{"actions":[{"action_type":"...",...},...]}\n'
        "Available action_type: brush, pencil, eraser, fill, color_adjust, text_overlay, "
        "gaussian_blur, clone_stamp, scatter_brush, pattern_brush, forward_warp, noop.\n"
        f"Canvas coordinates use single tokens: <x0>..<x{CANVAS_SIZE - 1}> for column, "
        f"<y0>..<y{CANVAS_SIZE - 1}> for row. Color channels use <c0>..<c255>. "
        "Path actions draw a segment from start_point to end_point.\n"
        "Respond with the JSON object only."
    )


def _legacy_trajectory_to_endpoints(data: Any) -> Any:
    """If a legacy 'trajectory' list is present and start/end aren't, derive endpoints.

    Lets old checkpoints (trained before the segment refactor) still be parsed
    by treating the first and last trajectory points as the segment endpoints.
    Derived endpoints are clamped to canvas bounds so they pass validation.
    """
    if not isinstance(data, dict):
        return data
    if "start_point" in data and "end_point" in data:
        return data
    traj = data.get("trajectory")
    if not isinstance(traj, (list, tuple)) or len(traj) < 1:
        return data
    new_data = {k: v for k, v in data.items() if k != "trajectory"}
    if "start_point" not in new_data:
        new_data["start_point"] = list(_clamp_canvas_point(traj[0]))
    if "end_point" not in new_data:
        last = traj[-1] if len(traj) >= 2 else traj[0]
        new_data["end_point"] = list(_clamp_canvas_point(last))
    return new_data


def _legacy_color_rgb_to_rgba(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    if "color_rgba" in data or "color_rgb" not in data:
        return data
    cr = data["color_rgb"]
    if not isinstance(cr, (list, tuple)) or len(cr) not in (3, 4):
        return data
    new_data = {k: v for k, v in data.items() if k != "color_rgb"}
    if len(cr) == 3:
        new_data["color_rgba"] = (int(cr[0]), int(cr[1]), int(cr[2]), 255)
    else:
        new_data["color_rgba"] = (int(cr[0]), int(cr[1]), int(cr[2]), int(cr[3]))
    return new_data


class BrushAction(BaseModel):
    action_type: Literal["brush"]
    color_rgba: tuple[int, int, int, int]
    stroke_size: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    hardness: int = 100

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        data = _legacy_color_rgb_to_rgba(data)
        data = _legacy_trajectory_to_endpoints(data)
        if isinstance(data, dict) and "hardness" not in data:
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

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
        return v


class PencilAction(BaseModel):
    action_type: Literal["pencil"]
    color_rgba: tuple[int, int, int, int]
    start_point: tuple[int, int]
    end_point: tuple[int, int]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        data = _legacy_color_rgb_to_rgba(data)
        data = _legacy_trajectory_to_endpoints(data)
        return data

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
        return v


class EraserAction(BaseModel):
    action_type: Literal["eraser"]
    stroke_size: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return _legacy_trajectory_to_endpoints(data)

    @field_validator("stroke_size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 50, "Stroke size must be in [1, 50]"
        return v

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
        return v


class FillAction(BaseModel):
    action_type: Literal["fill"]
    color_rgba: tuple[int, int, int, int]
    position: tuple[int, int]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return _legacy_color_rgb_to_rgba(data)

    @field_validator("color_rgba")
    @classmethod
    def _validate_color(cls, v: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        assert all(0 <= c <= 255 for c in v), "Color channels must be in [0, 255]"
        return v

    @field_validator("position")
    @classmethod
    def _validate_position(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Position ({x},{y}) out of canvas bounds"
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
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Position ({x},{y}) out of canvas bounds"
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
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Coordinate ({x},{y}) out of canvas bounds"
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
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    size: int = 8
    density: int = 5
    scatter: int = 30
    size_jitter: int = 50
    angle_jitter: int = 0
    seed: int = 0
    thickness: int = 1
    length: int = 0
    base_angle: int = -1

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return _legacy_trajectory_to_endpoints(data)

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

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
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
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    size: int = 10
    spacing: int = 20
    angle_jitter: int = 15
    thickness: int = 1
    length: int = 0

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return _legacy_trajectory_to_endpoints(data)

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

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
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


class ForwardWarpAction(BaseModel):
    action_type: Literal["forward_warp"]
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    size: int = 20
    strength: int = 50

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return _legacy_trajectory_to_endpoints(data)

    @field_validator("start_point", "end_point")
    @classmethod
    def _validate_point(cls, v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        assert 0 <= x < CANVAS_SIZE and 0 <= y < CANVAS_SIZE, f"Point ({x},{y}) out of canvas bounds"
        return v

    @field_validator("size")
    @classmethod
    def _validate_size(cls, v: int) -> int:
        assert 1 <= v <= 100, "size must be in [1, 100]"
        return v

    @field_validator("strength")
    @classmethod
    def _validate_strength(cls, v: int) -> int:
        assert 1 <= v <= 100, "strength must be in [1, 100]"
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
    | ForwardWarpAction
    | NoopAction
)


def _clamp_canvas_scalar(v: Any) -> int:
    x = int(v)
    if x < 0:
        return 0
    if x >= CANVAS_SIZE:
        return CANVAS_SIZE - 1
    return x


def _clamp_canvas_point(pt: Any) -> tuple[int, int]:
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        raise ValueError(f"Expected [x,y] point, got {pt!r}")
    return (_clamp_canvas_scalar(pt[0]), _clamp_canvas_scalar(pt[1]))


_POINT_KEYS = ("position", "source", "destination", "start_point", "end_point")
_COLOR_KEYS = ("color_rgba",)


def _clamp_action_coords_for_model_output(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    for key in _POINT_KEYS:
        pt = out.get(key)
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            c = _clamp_canvas_point(pt)
            out[key] = [c[0], c[1]]
    return out


_SENTINEL_RE = re.compile(r"<([xyc])(\d+)>")


def _emit_value(key: str, v: Any) -> str:
    """JSON-encode ``v`` for action key ``key``.

    For point keys we emit ``[<xN>,<yN>]`` and for color-channel keys
    ``[<cR>,<cG>,<cB>,<cA>]``; everything else is normal ``json.dumps``.
    The emitted text is not strict JSON but the brace-balance extractor and
    :func:`_sentinels_to_ints` together restore strict JSON before parsing.
    """
    if key in _POINT_KEYS and isinstance(v, (list, tuple)) and len(v) >= 2:
        return f"[<x{int(v[0])}>,<y{int(v[1])}>]"
    if key in _COLOR_KEYS and isinstance(v, (list, tuple)) and len(v) == 4:
        return "[" + ",".join(f"<c{int(c)}>" for c in v) + "]"
    return json.dumps(v, separators=(",", ":"))


def _emit_action(action: dict[str, Any]) -> str:
    parts = [f"{json.dumps(k)}:{_emit_value(k, v)}" for k, v in action.items()]
    return "{" + ",".join(parts) + "}"


def _emit_chunk(actions: list[dict[str, Any]]) -> str:
    return '{"actions":[' + ",".join(_emit_action(a) for a in actions) + "]}"


def _sentinels_to_ints(text: str) -> str:
    """Resolve every ``<xN>``/``<yN>``/``<cN>`` to its bare integer text."""
    return _SENTINEL_RE.sub(lambda m: m.group(2), text)


def parse_action(data: dict) -> Action:
    """Deserialize a raw dict into the correct Action subtype."""
    raw_tool = data.get("action_type")
    tool = _ACTION_TYPE_ALIASES.get(raw_tool, raw_tool)
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
        "forward_warp": ForwardWarpAction,
        "noop": NoopAction,
    }
    if tool not in dispatch:
        raise ValueError(f"Unknown action_type '{raw_tool}'. Valid: {list(dispatch)}")
    normalized = dict(data)
    if tool != raw_tool:
        normalized["action_type"] = tool
    return dispatch[tool].model_validate(_clamp_action_coords_for_model_output(normalized))


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
        """Serialize to the sentinel-augmented JSON string used as training targets.

        Coordinate and color-channel integers are emitted as single-token
        sentinels (``<xN>``, ``<yN>``, ``<cN>``); see
        :mod:`elysium.model.coord_tokens` for the wire format.
        """
        return _emit_chunk([a.model_dump() for a in self.actions])

    @classmethod
    def from_json_str(cls, raw: str, horizon: int = 5) -> "ActionChunk":
        """Parse the sentinel-augmented JSON string into an ActionChunk.

        Strict JSON without sentinels is also accepted -- :func:`_sentinels_to_ints`
        is a no-op on text that contains no sentinels.
        """
        data = json.loads(_sentinels_to_ints(raw))
        actions = [parse_action(a) for a in data["actions"]]
        return cls(actions=actions, horizon=horizon)

    @classmethod
    def noop_chunk(cls, horizon: int = 5) -> "ActionChunk":
        """Return a chunk filled entirely with noop actions."""
        return cls(actions=[NoopAction(action_type="noop")] * horizon, horizon=horizon)
