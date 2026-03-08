import pygame
import numpy as np
import sys
import json
from pathlib import Path
import time
import cv2

SLIDER_CONFIGS = [
    {"label": "R", "color": (255, 0, 0), "min": 0, "max": 255},
    {"label": "G", "color": (0, 255, 0), "min": 0, "max": 255},
    {"label": "B", "color": (0, 0, 255), "min": 0, "max": 255},
    {"label": "Size", "color": (0, 0, 0), "min": 1, "max": 50},
]


class ImageEditor:
    def __init__(self, image_path):
        pygame.init()

        self.canvas_size = 256
        self.display_scale = 2
        self.display_size = self.canvas_size * self.display_scale
        self.ui_width = 200
        self.window_width = self.display_size + self.ui_width
        self.window_height = self.display_size

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Image Editor")

        self.font_sm = pygame.font.Font(None, 18)
        self.font_md = pygame.font.Font(None, 20)

        self.load_image(image_path)

        self.drawing = False
        self.tool = "brush"
        self.brush_size = 5
        self.color = [255, 0, 0]
        self.prev_pos = None

        self.tools = ["brush", "pencil", "eraser", "fill", "picker"]

        self.dragging_slider = None
        self.active_input = None
        self.input_text = ""
        self.sliders: dict[str, dict] = {}

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

        self.clock = pygame.time.Clock()
        self.running = True
        self.discard_session = False
        self.discard_btn_rect = (0, 0, 0, 0)

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_LANCZOS4)
        self.canvas_array = img.copy()
        self.original_array = img.copy()

    def array_to_surface(self, arr):
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    def _get_slider_value(self, label: str) -> int:
        match label:
            case "R": return self.color[0]
            case "G": return self.color[1]
            case "B": return self.color[2]
            case "Size": return self.brush_size

    def _set_slider_value(self, label: str, value: int):
        match label:
            case "R": self.color[0] = value
            case "G": self.color[1] = value
            case "B": self.color[2] = value
            case "Size": self.brush_size = value

    def log_operation(self, tool: str, size: int | None, color: list[int] | tuple[int, ...] | None,
                      start_pos: tuple[int, int], end_pos: tuple[int, int]):
        self.operations.append({
            "timestamp": round(time.time() - self.start_time, 4),
            "frame_id": self.frame_id,
            "tool": tool,
            "size": size,
            "color": list(color) if color else None,
            "start_pos": list(start_pos),
            "end_pos": list(end_pos),
        })

    def save_log(self):
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

    def flood_fill(self, x, y):
        mask = np.zeros((self.canvas_size + 2, self.canvas_size + 2), np.uint8)
        cv2.floodFill(self.canvas_array, mask, (x, y), self.color[::-1],
                      loDiff=(10, 10, 10), upDiff=(10, 10, 10))
        self.log_operation("fill", None, tuple(self.color), (x, y), (x, y))
        self.canvas_dirty = True

    def erase_to_original(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        mask = np.zeros(self.canvas_array.shape[:2], dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, self.brush_size * 2)
        self.canvas_array[mask > 0] = self.original_array[mask > 0]
        self.log_operation("eraser", self.brush_size, None, (x1, y1), (x2, y2))
        self.canvas_dirty = True

    def draw_line(self, pos1, pos2, size, color):
        x1, y1 = pos1
        x2, y2 = pos2
        cv2.line(self.canvas_array, (x1, y1), (x2, y2), color[::-1], max(1, size * 2))
        self.log_operation(self.tool, size, color, (x1, y1), (x2, y2))
        self.canvas_dirty = True

    def draw_circle(self, pos, size, color):
        x, y = pos
        cv2.circle(self.canvas_array, (x, y), size, color[::-1], -1)
        self.log_operation(self.tool, size, color, (x, y), (x, y))
        self.canvas_dirty = True

    def draw_slider(self, x, y, width, label, handle_color, min_value, max_value):
        value = self._get_slider_value(label)

        text = self.font_sm.render(label, True, (0, 0, 0))
        self.screen.blit(text, (x, y))

        slider_y = y + 20
        pygame.draw.rect(self.screen, (100, 100, 100), (x, slider_y, width, 10))

        handle_x = x + int((value / max_value) * width)
        pygame.draw.circle(self.screen, handle_color, (handle_x, slider_y + 5), 8)

        input_x = x + width + 10
        input_width = 45
        input_rect = (input_x, y, input_width, 20)

        is_active = self.active_input == label
        border_color = (0, 0, 0) if is_active else (150, 150, 150)
        bg_color = (255, 255, 255) if is_active else (240, 240, 240)

        pygame.draw.rect(self.screen, bg_color, input_rect)
        pygame.draw.rect(self.screen, border_color, input_rect, 2)

        display_value = self.input_text if is_active else str(value)
        value_text = self.font_sm.render(display_value, True, (0, 0, 0))
        self.screen.blit(value_text, (input_x + 5, y + 3))

        return {
            'slider': (x, slider_y, width, 10),
            'input': input_rect,
            'label': label,
            'min': min_value,
            'max': max_value,
        }

    def draw_ui(self):
        ui_x = self.display_size

        pygame.draw.rect(self.screen, (220, 220, 220), (ui_x, 0, self.ui_width, self.window_height))

        y = 10

        title = self.font_md.render("TOOLS", True, (0, 0, 0))
        self.screen.blit(title, (ui_x + 10, y))
        y += 30

        for i, tool in enumerate(self.tools):
            tool_y = y + i * 30
            color = (100, 200, 100) if tool == self.tool else (180, 180, 180)
            pygame.draw.rect(self.screen, color, (ui_x + 10, tool_y, 180, 25), border_radius=5)

            text = self.font_md.render(tool.upper(), True, (0, 0, 0))
            self.screen.blit(text, (ui_x + 15, tool_y + 5))

        y += len(self.tools) * 30 + 20

        title = self.font_md.render("COLOR", True, (0, 0, 0))
        self.screen.blit(title, (ui_x + 10, y))
        y += 30

        pygame.draw.rect(self.screen, tuple(self.color), (ui_x + 10, y, 50, 50))
        pygame.draw.rect(self.screen, (0, 0, 0), (ui_x + 10, y, 50, 50), 2)
        picker_hint = self.font_sm.render("Alt+click canvas", True, (80, 80, 80))
        self.screen.blit(picker_hint, (ui_x + 65, y + 20))
        y += 60

        for cfg in SLIDER_CONFIGS:
            self.sliders[cfg["label"]] = self.draw_slider(
                ui_x + 10, y, 120, cfg["label"], cfg["color"], cfg["min"], cfg["max"]
            )
            y += 40

        y = self.window_height - 45
        pygame.draw.rect(self.screen, (220, 120, 120), (ui_x + 10, y, 180, 35), border_radius=5)
        btn_text = self.font_md.render("Don't save session", True, (0, 0, 0))
        self.screen.blit(btn_text, (ui_x + 25, y + 10))
        self.discard_btn_rect = (ui_x + 10, y, 180, 35)

    def handle_slider(self, pos):
        x, y = pos

        for label, sd in self.sliders.items():
            sx, sy, sw, sh = sd['slider']
            if sx <= x <= sx + sw and sy - 8 <= y <= sy + sh + 8:
                raw = int(((x - sx) / sw) * sd['max'])
                self._set_slider_value(label, max(sd['min'], min(sd['max'], raw)))
                return label

        return None

    def handle_input_click(self, pos):
        x, y = pos

        for sd in self.sliders.values():
            ix, iy, iw, ih = sd['input']
            if ix <= x <= ix + iw and iy <= y <= iy + ih:
                self.active_input = sd['label']
                self.input_text = str(self._get_slider_value(sd['label']))
                return True

        return False

    def apply_input(self):
        if not self.active_input or not self.input_text:
            return

        try:
            value = int(self.input_text)
            sd = self.sliders[self.active_input]
            self._set_slider_value(self.active_input, max(sd['min'], min(sd['max'], value)))
        except ValueError:
            pass

        self.active_input = None
        self.input_text = ""

    def handle_ui_click(self, pos):
        x, y = pos
        ui_x = self.display_size

        if x < ui_x:
            return False

        bx, by, bw, bh = self.discard_btn_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self.discard_session = True
            self.running = False
            return True

        y_offset = 40
        for i, tool in enumerate(self.tools):
            tool_y = y_offset + i * 30
            if ui_x + 10 <= x <= ui_x + 190 and tool_y <= y <= tool_y + 25:
                self.tool = tool
                self.active_input = None
                return True

        if self.handle_input_click(pos):
            return True

        slider = self.handle_slider(pos)
        if slider:
            self.dragging_slider = slider
            self.active_input = None
            return True

        self.active_input = None
        return True

    def _sample_color_at(self, cx: int, cy: int) -> None:
        if 0 <= cx < self.canvas_size and 0 <= cy < self.canvas_size:
            b, g, r = self.canvas_array[cy, cx]
            self.color = [int(r), int(g), int(b)]

    def handle_canvas_mouse(self, pos, event_type):
        if pos[0] >= self.display_size:
            return

        cx = pos[0] // self.display_scale
        cy = pos[1] // self.display_scale
        canvas_pos = (cx, cy)

        if event_type == "down":
            if pygame.key.get_mods() & pygame.KMOD_ALT:
                self._sample_color_at(cx, cy)
                return

            self.drawing = True
            self.prev_pos = canvas_pos

            if self.tool == "picker":
                self._sample_color_at(cx, cy)
            elif self.tool == "fill":
                self.flood_fill(cx, cy)
            elif self.tool == "eraser":
                self.erase_to_original(canvas_pos, canvas_pos)
            else:
                size = 0 if self.tool == "pencil" else self.brush_size
                self.draw_circle(canvas_pos, size, tuple(self.color))

        elif event_type == "motion" and self.drawing:
            if self.prev_pos and self.tool != "picker":
                if self.tool == "eraser":
                    self.erase_to_original(self.prev_pos, canvas_pos)
                else:
                    size = 0 if self.tool == "pencil" else self.brush_size
                    self.draw_line(self.prev_pos, canvas_pos, size, tuple(self.color))
            self.prev_pos = canvas_pos if self.tool != "picker" else None

        elif event_type == "up":
            self.drawing = False
            self.prev_pos = None

    def auto_save(self):
        current_time = time.time()
        if current_time - self.last_save_time >= self.frame_interval:
            if self.canvas_dirty:
                output_path = self.output_dir / f"{self.image_name}{self.frame_id}.jpg"
                cv2.imwrite(str(output_path), self.canvas_array)
                self.canvas_dirty = False
                self.frame_id += 1
            self.last_save_time = current_time

    def run(self):
        print("\n=== Image Editor ===")
        print("- Tools: brush, pencil, eraser, fill, picker")
        print("- Alt+click on canvas to sample color (no tool switch)")
        print("- Adjust color/size with sliders or text input")
        print("- [c] clear canvas, [q/ESC] quit, 'Don't save session' discards")
        print(f"- Auto-saving at {self.fps}fps to {self.output_dir}/\n")

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
                        if event.key == pygame.K_RETURN:
                            self.apply_input()
                        elif event.key == pygame.K_ESCAPE:
                            self.active_input = None
                            self.input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif event.unicode.isdigit() and len(self.input_text) < 3:
                            self.input_text += event.unicode
                    else:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_c:
                            self.canvas_array = self.original_array.copy()
                            self.log_operation("clear", None, None, (0, 0), (0, 0))
                            self.canvas_dirty = True

            self.auto_save()

            canvas_surface = self.array_to_surface(self.canvas_array)
            scaled_surface = pygame.transform.scale(canvas_surface, (self.display_size, self.display_size))
            self.screen.blit(scaled_surface, (0, 0))
            self.draw_ui()

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
        print("Usage: python data_annotation/annotate.py <image_path>")
        print("Example: python data_annotation/annotate.py data/images/pirlo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    editor = ImageEditor(image_path)
    editor.run()
