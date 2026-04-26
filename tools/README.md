# Data annotation

Pygame editor for collecting training data: paint and global color adjustments; frames and operations are logged for supervision.

## Usage

From the repository root (with `src` on `PYTHONPATH`, e.g. `pip install -e .`):

```bash
python tools/annotate.py <image_path>
```

Images are resized to 512×512 on load.

## Tools

| Tool | Behavior |
|------|----------|
| Brush | Current color and stroke size |
| Pencil | 1px strokes |
| Eraser | Restores original pixels under stroke |
| Fill | Flood fill at click |
| Picker | Sample color from canvas (updates foreground swatch) |
| Adjust | Brightness (-100..100), contrast (0.5..2.0), saturation (0..2.0); live preview; **Apply** commits |

## UI

- Dark sidebar: tool buttons with hover and active state, shortcut letter on each label.
- Two swatches: previous (left) and current (right); click previous to swap.
- RGB sliders for brush, pencil, and fill. Size slider for brush and eraser only.
- Canvas: semi-transparent ring shows brush/eraser size at cursor; crosshair for fill and picker.
- Status bar under the canvas: active tool, size, hex color, frame count, session seconds.

## Controls

| Input | Effect |
|--------|--------|
| `B` `P` `E` `F` `I` `A` | Select brush / pencil / eraser / fill / picker / adjust |
| Click + drag on canvas | Draw with selected tool |
| Slider / numeric field | Adjust parameters; Enter to apply typed value |
| Alt + click canvas | Sample color |
| Ctrl+Z | Undo (canvas state, last 50 steps) |
| Ctrl+Shift+Z | Redo |
| `c` | Reset canvas to original image |
| `q` / Esc | Quit |

## Output

| Path | Content |
|------|---------|
| `data/raw/frames/{session_stem}/{frame_id}.jpg` | Canvas snapshots at ~10 fps when the canvas changes; each session gets its own subdirectory so the same image can be annotated multiple times without collisions |
| `data/raw/sessions/{session_stem}.json` | All operations for the session; `session_stem = {image_name}_{YYYYmmdd_HHMMSS}` |

### Log format

`tool` is one of `brush`, `pencil`, `eraser`, `fill`, `picker`, `clear`, `undo`, `redo`, `color_adjust`.

`color_adjust` entries include `brightness`, `contrast`, and `saturation` (floats for contrast/saturation). Other fields follow the existing pattern (`start_pos`, `end_pos`, `size`, `color` as applicable).

## Utilities

**to_jpg.py** — convert to RGB JPEG and resize to 512×512:

```bash
python tools/to_jpg.py <input> <output.jpg>
```

## Dependencies

```
pygame
numpy
opencv-python
Pillow
```
