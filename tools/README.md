# Data Annotation

Pygame-based image editor for collecting training data. Paint on images; frames and tool operations are logged for model supervision.

## Usage

```bash
python data_annotation/annotate.py <image_path>
```

Images are resized to 256×256 on load.

## Tools

| Tool | Behavior |
|------|----------|
| Brush | Draws with current color and size |
| Pencil | 1px strokes |
| Eraser | Restores original pixels |
| Fill | Flood-fill at click position |
| Picker | Click on canvas to sample color (updates current color) |

## Controls

| Key / Action | Effect |
|---|---|
| Click + drag on canvas | Draw with selected tool |
| Slider drag | Adjust R, G, B (0–255) or Size (1–50) |
| Click input box → type → Enter | Set exact value |
| Esc (in input) | Cancel text input |
| `c` | Reset canvas to original image |
| `q` / Esc | Quit |

## Output

| Path | Content |
|------|---------|
| `data/raw/frames/{name}{frame_id}.jpg` | Canvas snapshots at ~15fps (only on change) |
| `data/raw/sessions/{name}_{timestamp}.json` | All tool operations for the session |

### Log format

```json
{
  "image_name": "pirlo",
  "image_path": "/absolute/path/pirlo.jpg",
  "canvas_size": 256,
  "total_frames": 42,
  "duration_s": 18.73,
  "operations": [
    {
      "timestamp": 1.2034,
      "frame_id": 3,
      "tool": "brush",
      "size": 5,
      "color": [255, 0, 0],
      "start_pos": [120, 80],
      "end_pos": [125, 82]
    }
  ]
}
```

## Utilities

**to_jpg.py** — Convert and resize any image to 256×256 JPEG:

```bash
python data_annotation/to_jpg.py <input> <output.jpg>
```

## Dependencies

```
pygame
numpy
opencv-python
Pillow
```
