from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pygame

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from elysium.engine.canvas import apply_color_adjust_rgb01


@dataclass(frozen=True)
class Theme:
    bg_canvas: tuple[int, int, int]
    sidebar: tuple[int, int, int]
    sidebar_border: tuple[int, int, int]
    text: tuple[int, int, int]
    text_muted: tuple[int, int, int]
    btn_idle: tuple[int, int, int]
    btn_hover: tuple[int, int, int]
    btn_active: tuple[int, int, int]
    accent: tuple[int, int, int]
    track: tuple[int, int, int]
    input_bg: tuple[int, int, int]
    input_border: tuple[int, int, int]
    danger: tuple[int, int, int]
    status_bar: tuple[int, int, int]


@dataclass(frozen=True)
class Layout:
    pad: int
    btn_h: int
    btn_gap: int
    sidebar_w: int
    status_h: int
    slider_w: int
    slider_track_h: int
    handle_r: int
    swatch: int
    swatch_gap: int


THEME = Theme(
    bg_canvas=(40, 40, 44),
    sidebar=(43, 43, 46),
    sidebar_border=(60, 60, 64),
    text=(230, 230, 235),
    text_muted=(140, 140, 150),
    btn_idle=(58, 58, 62),
    btn_hover=(72, 72, 78),
    btn_active=(10, 132, 255),
    accent=(10, 132, 255),
    track=(70, 70, 75),
    input_bg=(30, 30, 33),
    input_border=(90, 90, 95),
    danger=(200, 80, 80),
    status_bar=(32, 32, 36),
)

LAYOUT = Layout(
    pad=10,
    btn_h=30,
    btn_gap=6,
    sidebar_w=228,
    status_h=30,
    slider_w=130,
    slider_track_h=8,
    handle_r=7,
    swatch=36,
    swatch_gap=8,
)

TOOL_ROWS: list[tuple[str, str, int, str]] = [
    ("brush", "Brush", pygame.K_b, "B"),
    ("pencil", "Pencil", pygame.K_p, "P"),
    ("eraser", "Eraser", pygame.K_e, "E"),
    ("fill", "Fill", pygame.K_f, "F"),
    ("picker", "Picker", pygame.K_i, "I"),
    ("color_adjust", "Adjust", pygame.K_a, "A"),
]


class ImageEditor:
    def __init__(self, image_path: str) -> None:
        pygame.init()

        self.canvas_size = 256
        self.zoom_pct: int = 200
        self.layout = LAYOUT
        self.theme = THEME

        self.window_width = self.canvas_size * 2 + self.layout.sidebar_w
        self.window_height = self.canvas_size * 2 + self.layout.status_h

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Elysium Annotator")

        self.font_sm = pygame.font.Font(None, 17)
        self.font_md = pygame.font.Font(None, 20)

        self.load_image(image_path)

        self.drawing = False
        self.tool = "brush"
        self.brush_size = 5
        self.color = [255, 0, 0]
        self.prev_color = [255, 0, 0]
        self.prev_pos = None

        self.adj_brightness = 0
        self.adj_exposure: int = 0
        self.adj_contrast_int = 100
        self.adj_highlights: int = 0
        self.adj_shadows: int = 0
        self.adj_saturation_int = 100
        self.adj_hue_shift: int = 0
        self.adj_temperature: int = 0

        self.dragging_slider: str | None = None
        self.active_input: str | None = None
        self.input_text = ""
        self._input_range: tuple[int, int] | None = None
        self.sliders: dict[str, dict] = {}

        self.tool_button_rects: list[tuple[pygame.Rect, str]] = []
        self.discard_btn_rect = pygame.Rect(0, 0, 0, 0)
        self.apply_adjust_rect = pygame.Rect(0, 0, 0, 0)
        self.swatch_prev_rect = pygame.Rect(0, 0, 0, 0)
        self.swatch_cur_rect = pygame.Rect(0, 0, 0, 0)

        self.image_name = Path(image_path).stem
        self.image_path = str(Path(image_path).resolve())
        self.output_dir = Path("data/raw/frames")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path("data/raw/sessions")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.frame_id = 0
        self.canvas_dirty = False
        self.start_time = time.time()
        self.last_save_time = self.start_time
        self.fps = 15
        self.frame_interval = 1.0 / self.fps
        self.operations: list[dict] = []

        self.undo_stack: list[np.ndarray] = []
        self.redo_stack: list[np.ndarray] = []
        self._stroke_checkpoint_pending = False

        self.clock = pygame.time.Clock()
        self.running = True
        self.discard_session = False

    def load_image(self, image_path: str) -> None:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LANCZOS4)
        self.canvas_array = img.copy()
        self.original_array = img.copy()

    def array_to_surface(self, arr: np.ndarray) -> pygame.Surface:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    def _display_canvas_bgr(self) -> np.ndarray:
        if self.tool != "color_adjust":
            return self.canvas_array
        rgb01 = cv2.cvtColor(self.canvas_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = apply_color_adjust_rgb01(
            rgb01,
            self.adj_brightness,
            self.adj_contrast_int / 100.0,
            self.adj_saturation_int / 100.0,
            exposure=self.adj_exposure,
            highlights=self.adj_highlights,
            shadows=self.adj_shadows,
            hue_shift=self.adj_hue_shift,
            temperature=self.adj_temperature,
        )
        u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)

    def _get_slider_value(self, label: str) -> int:
        match label:
            case "R":
                return self.color[0]
            case "G":
                return self.color[1]
            case "B":
                return self.color[2]
            case "Size":
                return self.brush_size
            case "Zoom":
                return self.zoom_pct
            case "Bright":
                return self.adj_brightness
            case "Expose":
                return self.adj_exposure
            case "Contrast":
                return self.adj_contrast_int
            case "Highs":
                return self.adj_highlights
            case "Shads":
                return self.adj_shadows
            case "Sat":
                return self.adj_saturation_int
            case "Hue":
                return self.adj_hue_shift
            case "Temp":
                return self.adj_temperature
            case _:
                raise ValueError(label)

    def _set_slider_value(self, label: str, value: int) -> None:
        match label:
            case "R":
                self.color[0] = value
            case "G":
                self.color[1] = value
            case "B":
                self.color[2] = value
            case "Size":
                self.brush_size = value
            case "Zoom":
                self.zoom_pct = value
            case "Bright":
                self.adj_brightness = value
            case "Expose":
                self.adj_exposure = value
            case "Contrast":
                self.adj_contrast_int = value
            case "Highs":
                self.adj_highlights = value
            case "Shads":
                self.adj_shadows = value
            case "Sat":
                self.adj_saturation_int = value
            case "Hue":
                self.adj_hue_shift = value
            case "Temp":
                self.adj_temperature = value
            case _:
                raise ValueError(label)

    def log_operation(
        self,
        tool: str,
        size: int | None,
        color: list[int] | tuple[int, ...] | None,
        start_pos: tuple[int, int],
        end_pos: tuple[int, int],
        **extras: object,
    ) -> None:
        rec: dict = {
            "timestamp": round(time.time() - self.start_time, 4),
            "frame_id": self.frame_id,
            "tool": tool,
            "size": size,
            "color": list(color) if color else None,
            "start_pos": list(start_pos),
            "end_pos": list(end_pos),
        }
        rec.update(extras)
        self.operations.append(rec)

    def _commit_undo_checkpoint(self) -> None:
        self.undo_stack.append(self.canvas_array.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self.canvas_array.copy())
        self.canvas_array = self.undo_stack.pop()
        self.canvas_dirty = True
        self.log_operation("undo", None, None, (0, 0), (0, 0))

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self.canvas_array.copy())
        self.canvas_array = self.redo_stack.pop()
        self.canvas_dirty = True
        self.log_operation("redo", None, None, (0, 0), (0, 0))

    def save_log(self) -> None:
        log_data = {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "canvas_size": self.canvas_size,
            "total_frames": self.frame_id,
            "duration_s": round(time.time() - self.start_time, 2),
            "operations": self.operations,
        }
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{self.image_name}_{timestamp}.json"
        log_path.write_text(json.dumps(log_data, indent=2))
        print(f"Log saved: {log_path} ({len(self.operations)} operations)")

    def flood_fill(self, x: int, y: int) -> None:
        mask = np.zeros((self.canvas_size + 2, self.canvas_size + 2), np.uint8)
        cv2.floodFill(self.canvas_array, mask, (x, y), self.color[::-1], loDiff=(10, 10, 10), upDiff=(10, 10, 10))
        self.log_operation("fill", None, tuple(self.color), (x, y), (x, y))
        self.canvas_dirty = True

    def erase_to_original(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> None:
        x1, y1 = pos1
        x2, y2 = pos2
        mask = np.zeros(self.canvas_array.shape[:2], dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, max(1, 2 * self.brush_size))
        self.canvas_array[mask > 0] = self.original_array[mask > 0]
        self.log_operation("eraser", self.brush_size, None, (x1, y1), (x2, y2))
        self.canvas_dirty = True

    def draw_line(
        self,
        pos1: tuple[int, int],
        pos2: tuple[int, int],
        size: int,
        color: tuple[int, ...],
        *,
        pencil: bool = False,
    ) -> None:
        x1, y1 = pos1
        x2, y2 = pos2
        if pencil:
            thickness = max(1, size)
        else:
            thickness = max(1, 2 * size)
        cv2.line(self.canvas_array, (x1, y1), (x2, y2), color[::-1], thickness)
        self.log_operation(self.tool, size, color, (x1, y1), (x2, y2))
        self.canvas_dirty = True

    def draw_circle(self, pos: tuple[int, int], size: int, color: tuple[int, ...]) -> None:
        x, y = pos
        r = max(1, size)
        cv2.circle(self.canvas_array, (x, y), r, color[::-1], -1)
        self.log_operation(self.tool, size, color, (x, y), (x, y))
        self.canvas_dirty = True

    def _reset_color_adjust(self) -> None:
        self.adj_brightness = 0
        self.adj_exposure = 0
        self.adj_contrast_int = 100
        self.adj_highlights = 0
        self.adj_shadows = 0
        self.adj_saturation_int = 100
        self.adj_hue_shift = 0
        self.adj_temperature = 0

    def apply_color_adjustment(self) -> None:
        c = self.adj_contrast_int / 100.0
        s = self.adj_saturation_int / 100.0
        if (
            self.adj_brightness == 0
            and self.adj_exposure == 0
            and abs(c - 1.0) < 1e-6
            and self.adj_highlights == 0
            and self.adj_shadows == 0
            and abs(s - 1.0) < 1e-6
            and self.adj_hue_shift == 0
            and self.adj_temperature == 0
        ):
            return
        self._commit_undo_checkpoint()
        rgb01 = cv2.cvtColor(self.canvas_array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = apply_color_adjust_rgb01(
            rgb01,
            self.adj_brightness,
            c,
            s,
            exposure=self.adj_exposure,
            highlights=self.adj_highlights,
            shadows=self.adj_shadows,
            hue_shift=self.adj_hue_shift,
            temperature=self.adj_temperature,
        )
        u8 = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        self.canvas_array = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        self.log_operation(
            "color_adjust",
            None,
            None,
            (0, 0),
            (0, 0),
            brightness=self.adj_brightness,
            exposure=self.adj_exposure,
            contrast=c,
            highlights=self.adj_highlights,
            shadows=self.adj_shadows,
            saturation=s,
            hue_shift=self.adj_hue_shift,
            temperature=self.adj_temperature,
        )
        self._reset_color_adjust()
        self.canvas_dirty = True

    def draw_slider(
        self,
        x: int,
        y: int,
        width: int,
        label: str,
        handle_color: tuple[int, int, int],
        min_value: int,
        max_value: int,
    ) -> dict:
        t = self.theme
        value = self._get_slider_value(label)

        text = self.font_sm.render(label, True, t.text)
        self.screen.blit(text, (x, y))

        slider_y = y + 18
        pygame.draw.rect(self.screen, t.track, (x, slider_y, width, self.layout.slider_track_h), border_radius=4)

        span = max_value - min_value
        frac = (value - min_value) / span if span else 0.0
        frac = max(0.0, min(1.0, frac))
        handle_x = x + int(frac * width)
        cy = slider_y + self.layout.slider_track_h // 2
        pygame.draw.circle(self.screen, handle_color, (handle_x, cy), self.layout.handle_r)

        input_x = x + width + self.layout.pad
        input_width = 44
        input_rect = pygame.Rect(input_x, y, input_width, 22)

        is_active = self.active_input == label
        border_c = t.accent if is_active else t.input_border
        pygame.draw.rect(self.screen, t.input_bg, input_rect)
        pygame.draw.rect(self.screen, border_c, input_rect, 1)

        display_value = self.input_text if is_active else str(value)
        value_text = self.font_sm.render(display_value, True, t.text)
        self.screen.blit(value_text, (input_x + 4, y + 4))

        return {
            "slider": (x, slider_y, width, self.layout.slider_track_h),
            "input": input_rect,
            "label": label,
            "min": min_value,
            "max": max_value,
        }

    def draw_ui(self) -> None:
        t = self.theme
        ly = self.layout
        ui_x = self.window_width - ly.sidebar_w

        pygame.draw.line(
            self.screen,
            t.sidebar_border,
            (ui_x, 0),
            (ui_x, self.window_height),
            1,
        )
        pygame.draw.rect(self.screen, t.sidebar, (ui_x, 0, ly.sidebar_w, self.window_height))

        y = ly.pad
        title = self.font_md.render("TOOLS", True, t.text)
        self.screen.blit(title, (ui_x + ly.pad, y))
        y += 26

        mx, my = pygame.mouse.get_pos()
        self.tool_button_rects = []

        for tid, label, _key, letter in TOOL_ROWS:
            rect = pygame.Rect(ui_x + ly.pad, y, ly.sidebar_w - 2 * ly.pad, ly.btn_h)
            self.tool_button_rects.append((rect, tid))
            hover = rect.collidepoint(mx, my)
            active = tid == self.tool
            if active:
                bg = t.btn_active
            elif hover:
                bg = t.btn_hover
            else:
                bg = t.btn_idle
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            lbl = self.font_sm.render(f"[{letter}] {label}", True, t.text if not active else (255, 255, 255))
            self.screen.blit(lbl, (rect.x + 8, rect.y + 7))
            y += ly.btn_h + ly.btn_gap

        y += ly.pad
        title = self.font_md.render("COLOR", True, t.text)
        self.screen.blit(title, (ui_x + ly.pad, y))
        y += 26

        sw = ly.swatch
        cur_rect = pygame.Rect(ui_x + ly.pad, y, sw, sw)
        prev_rect = pygame.Rect(cur_rect.right + ly.swatch_gap, y, sw, sw)
        pygame.draw.rect(self.screen, tuple(self.prev_color), prev_rect)
        pygame.draw.rect(self.screen, t.input_border, prev_rect, 2)
        pygame.draw.rect(self.screen, tuple(self.color), cur_rect)
        pygame.draw.rect(self.screen, t.accent, cur_rect, 2)
        hint = self.font_sm.render("prev | cur", True, t.text_muted)
        self.screen.blit(hint, (prev_rect.right + ly.pad, y + sw // 2 - 6))
        self.swatch_prev_rect = prev_rect
        self.swatch_cur_rect = cur_rect
        y += sw + ly.pad + 4

        self.sliders.clear()
        if self.tool in ("brush", "pencil", "fill"):
            for cfg in (
                {"label": "R", "color": (200, 60, 60), "min": 0, "max": 255},
                {"label": "G", "color": (60, 200, 60), "min": 0, "max": 255},
                {"label": "B", "color": (60, 60, 200), "min": 0, "max": 255},
            ):
                self.sliders[cfg["label"]] = self.draw_slider(
                    ui_x + ly.pad,
                    y,
                    ly.slider_w,
                    cfg["label"],
                    cfg["color"],
                    cfg["min"],
                    cfg["max"],
                )
                y += 42
        if self.tool in ("brush", "eraser"):
            self.sliders["Size"] = self.draw_slider(
                ui_x + ly.pad,
                y,
                ly.slider_w,
                "Size",
                (120, 120, 120),
                1,
                50,
            )
            y += 42

        if self.tool == "color_adjust":
            for cfg in (
                {"label": "Bright",   "color": (200, 200, 80),  "min": -100, "max": 100},
                {"label": "Expose",   "color": (220, 180, 60),  "min": -100, "max": 100},
                {"label": "Contrast", "color": (140, 140, 200), "min":   50, "max": 200},
                {"label": "Highs",    "color": (200, 200, 200), "min": -100, "max": 100},
                {"label": "Shads",    "color": (100, 100, 120), "min": -100, "max": 100},
                {"label": "Sat",      "color": (200, 140, 200), "min":    0, "max": 200},
                {"label": "Hue",      "color": (200, 100, 200), "min": -180, "max": 180},
                {"label": "Temp",     "color": (220, 140, 60),  "min": -100, "max": 100},
            ):
                self.sliders[cfg["label"]] = self.draw_slider(
                    ui_x + ly.pad, y, ly.slider_w, cfg["label"], cfg["color"], cfg["min"], cfg["max"]
                )
                y += 42
            self.apply_adjust_rect = pygame.Rect(ui_x + ly.pad, y, ly.sidebar_w - 2 * ly.pad, ly.btn_h)
            pygame.draw.rect(self.screen, t.accent, self.apply_adjust_rect, border_radius=6)
            at = self.font_sm.render("Apply adjustment", True, (255, 255, 255))
            self.screen.blit(at, (self.apply_adjust_rect.x + 12, self.apply_adjust_rect.y + 8))
            y += ly.btn_h + ly.pad

        y += ly.pad
        view_title = self.font_md.render("VIEW", True, t.text)
        self.screen.blit(view_title, (ui_x + ly.pad, y))
        y += 26
        self.sliders["Zoom"] = self.draw_slider(
            ui_x + ly.pad, y, ly.slider_w, "Zoom", (120, 180, 120), 50, 400
        )
        y += 42

        discard_y = self.window_height - ly.btn_h - ly.pad
        self.discard_btn_rect = pygame.Rect(ui_x + ly.pad, discard_y, ly.sidebar_w - 2 * ly.pad, ly.btn_h)
        pygame.draw.rect(self.screen, t.danger, self.discard_btn_rect, border_radius=6)
        btn_text = self.font_sm.render("Discard session", True, (255, 255, 255))
        self.screen.blit(btn_text, (self.discard_btn_rect.x + 28, self.discard_btn_rect.y + 8))

    def draw_status_bar(self) -> None:
        t = self.theme
        elapsed = int(time.time() - self.start_time)
        hx = "#{:02x}{:02x}{:02x}".format(self.color[0], self.color[1], self.color[2])
        line = f"{self.tool}  |  size {self.brush_size}  |  {hx}  |  frames {self.frame_id}  |  {elapsed}s"
        surf = self.font_sm.render(line, True, t.text_muted)
        self.screen.blit(surf, (self.layout.pad, self.window_height - self.layout.status_h + 6))

    def handle_slider(self, pos: tuple[int, int]) -> str | None:
        x, y = pos
        for label, sd in self.sliders.items():
            sx, sy, sw, sh = sd["slider"]
            if sx <= x <= sx + sw and sy - 10 <= y <= sy + sh + 10:
                span = sd["max"] - sd["min"]
                raw = int((x - sx) / sw * span + sd["min"]) if sw else sd["min"]
                self._set_slider_value(label, max(sd["min"], min(sd["max"], raw)))
                return label
        return None

    def handle_input_click(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        for sd in self.sliders.values():
            if sd["input"].collidepoint(x, y):
                self.active_input = sd["label"]
                self._input_range = (sd["min"], sd["max"])
                self.input_text = str(self._get_slider_value(sd["label"]))
                return True
        return False

    def apply_input(self) -> None:
        if not self.active_input or not self.input_text:
            return
        raw = self.input_text.strip()
        if raw == "-" or not raw:
            return
        if self._input_range is not None:
            lo, hi = self._input_range
        elif self.active_input in self.sliders:
            lo = self.sliders[self.active_input]["min"]
            hi = self.sliders[self.active_input]["max"]
        else:
            return
        value = int(raw)
        self._set_slider_value(self.active_input, max(lo, min(hi, value)))
        self.active_input = None
        self.input_text = ""
        self._input_range = None

    def handle_ui_click(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        ui_x = self.window_width - self.layout.sidebar_w
        if x < ui_x:
            return False

        if self.discard_btn_rect.collidepoint(x, y):
            self.discard_session = True
            self.running = False
            return True

        for rect, tid in self.tool_button_rects:
            if rect.collidepoint(x, y):
                if self.tool == "color_adjust" and tid != "color_adjust":
                    self._reset_color_adjust()
                self.tool = tid
                self.active_input = None
                self.input_text = ""
                self._input_range = None
                return True

        if self.tool == "color_adjust" and self.apply_adjust_rect.collidepoint(x, y):
            self.apply_color_adjustment()
            return True

        if self.swatch_prev_rect.collidepoint(x, y):
            self.color, self.prev_color = self.prev_color, self.color
            return True
        if self.swatch_cur_rect.collidepoint(x, y):
            return True

        if self.handle_input_click((x, y)):
            return True

        slider = self.handle_slider((x, y))
        if slider:
            self.dragging_slider = slider
            self.active_input = None
            self.input_text = ""
            self._input_range = None
            return True

        self.active_input = None
        self.input_text = ""
        self._input_range = None
        return True

    def _sample_color_at(self, cx: int, cy: int) -> None:
        if 0 <= cx < self.canvas_size and 0 <= cy < self.canvas_size:
            self.prev_color = list(self.color)
            b, g, r = self.canvas_array[cy, cx]
            self.color = [int(r), int(g), int(b)]

    def handle_canvas_mouse(self, pos: tuple[int, int], event_type: str) -> None:
        zoom = self.zoom_pct / 100.0
        zoomed_size = int(self.canvas_size * zoom)
        canvas_w = self.window_width - self.layout.sidebar_w
        canvas_h = self.window_height - self.layout.status_h
        if pos[0] >= min(zoomed_size, canvas_w) or pos[1] >= min(zoomed_size, canvas_h):
            return

        if self.tool == "color_adjust":
            return

        cx = int(pos[0] / zoom)
        cy = int(pos[1] / zoom)
        canvas_pos = (cx, cy)

        if event_type == "down":
            if pygame.key.get_mods() & pygame.KMOD_ALT:
                self._sample_color_at(cx, cy)
                return

            if self.tool == "picker":
                self.drawing = True
                self.prev_pos = canvas_pos
                self._sample_color_at(cx, cy)
                return

            if not self._stroke_checkpoint_pending:
                self._commit_undo_checkpoint()
                self._stroke_checkpoint_pending = True

            self.drawing = True
            self.prev_pos = canvas_pos

            if self.tool == "fill":
                self.flood_fill(cx, cy)
                self._stroke_checkpoint_pending = False
            elif self.tool == "eraser":
                self.erase_to_original(canvas_pos, canvas_pos)
            else:
                size = 1 if self.tool == "pencil" else self.brush_size
                self.draw_circle(canvas_pos, size, tuple(self.color))

        elif event_type == "motion" and self.drawing:
            if self.prev_pos and self.tool != "picker":
                if self.tool == "eraser":
                    self.erase_to_original(self.prev_pos, canvas_pos)
                else:
                    size = 1 if self.tool == "pencil" else self.brush_size
                    self.draw_line(
                        self.prev_pos,
                        canvas_pos,
                        size,
                        tuple(self.color),
                        pencil=self.tool == "pencil",
                    )
            self.prev_pos = canvas_pos if self.tool != "picker" else None

        elif event_type == "up":
            self.drawing = False
            self.prev_pos = None
            self._stroke_checkpoint_pending = False

    def draw_canvas_overlay(self, mouse_xy: tuple[int, int]) -> None:
        x, y = mouse_xy
        zoom = self.zoom_pct / 100.0
        zoomed_size = int(self.canvas_size * zoom)
        canvas_w = self.window_width - self.layout.sidebar_w
        canvas_h = self.window_height - self.layout.status_h
        if x >= min(zoomed_size, canvas_w) or y >= min(zoomed_size, canvas_h):
            return
        if self.tool in ("fill", "picker"):
            pygame.draw.line(self.screen, (255, 255, 255), (x - 10, y), (x + 10, y), 1)
            pygame.draw.line(self.screen, (255, 255, 255), (x, y - 10), (x, y + 10), 1)
            return
        if self.tool == "color_adjust":
            return
        r = max(1, int((self.brush_size if self.tool != "pencil" else 1) * zoom))
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 255, 255, 70), (r + 2, r + 2), r, 1)
        self.screen.blit(s, (x - r - 2, y - r - 2))

    def auto_save(self) -> None:
        current_time = time.time()
        if current_time - self.last_save_time >= self.frame_interval:
            if self.canvas_dirty:
                output_path = self.output_dir / f"{self.image_name}{self.frame_id}.jpg"
                cv2.imwrite(str(output_path), self.canvas_array)
                self.canvas_dirty = False
                self.frame_id += 1
            self.last_save_time = current_time

    def run(self) -> None:
        print("\n=== Elysium Annotator ===")
        print("- Tools: brush, pencil, eraser, fill, picker, color adjust")
        print("- Shortcuts: B P E F I A; Ctrl+Z undo; Ctrl+Shift+Z redo")
        print("- Alt+click: sample color")
        print("- c: reset canvas; q / Esc: quit")
        print(f"- Auto-save ~{self.fps}/s on change -> {self.output_dir}/\n")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if not self.handle_ui_click(event.pos):
                            self.handle_canvas_mouse(event.pos, "down")

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        self.handle_slider(event.pos)
                    else:
                        self.handle_canvas_mouse(event.pos, "motion")

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = None
                        self.handle_canvas_mouse(event.pos, "up")

                elif event.type == pygame.KEYDOWN:
                    if self.active_input:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self.apply_input()
                        elif event.key == pygame.K_ESCAPE:
                            self.active_input = None
                            self.input_text = ""
                            self._input_range = None
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif (
                            event.unicode == "-"
                            and not self.input_text
                            and self._input_range is not None
                            and self._input_range[0] < 0
                        ):
                            self.input_text = "-"
                        elif event.unicode.lstrip("-").isdigit() and len(self.input_text) < 5:
                            _neg_ok = self._input_range is not None and self._input_range[0] < 0
                            if event.unicode == "-" and not _neg_ok:
                                pass
                            elif event.unicode != "-" or _neg_ok:
                                self.input_text += event.unicode
                    else:
                        mods = pygame.key.get_mods()
                        if mods & pygame.KMOD_CTRL and event.key == pygame.K_z:
                            if mods & pygame.KMOD_SHIFT:
                                self.redo()
                            else:
                                self.undo()
                            continue
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_c:
                            self._commit_undo_checkpoint()
                            self.canvas_array = self.original_array.copy()
                            self.log_operation("clear", None, None, (0, 0), (0, 0))
                            self.canvas_dirty = True
                        else:
                            for _tid, _label, key, _letter in TOOL_ROWS:
                                if event.key == key:
                                    if self.tool == "color_adjust" and _tid != "color_adjust":
                                        self._reset_color_adjust()
                                    self.tool = _tid
                                    break

            self.window_width, self.window_height = self.screen.get_size()

            self.auto_save()

            zoom = self.zoom_pct / 100.0
            zoomed_size = max(1, int(self.canvas_size * zoom))
            disp = self._display_canvas_bgr()
            canvas_surface = self.array_to_surface(disp)
            scaled_surface = pygame.transform.scale(canvas_surface, (zoomed_size, zoomed_size))
            self.screen.fill(self.theme.bg_canvas)
            self.screen.blit(scaled_surface, (0, 0))
            self.draw_canvas_overlay(pygame.mouse.get_pos())
            self.draw_ui()
            self.draw_status_bar()

            pygame.display.flip()
            self.clock.tick(60)

        if self.discard_session:
            for i in range(self.frame_id):
                p = self.output_dir / f"{self.image_name}{i}.jpg"
                if p.exists():
                    p.unlink()
            print("Session discarded, frames removed.")
        else:
            self.save_log()
            print(f"Total frames saved: {self.frame_id} in {self.output_dir}/")
        pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/annotate.py <image_path>")
        sys.exit(1)

    ImageEditor(sys.argv[1]).run()
