# Elysium

AI-driven image manipulation — "Photoshop on images with AI." The goal is to train a model that performs semantic editing guided by user paint strokes and tool actions.

## Overview

This project is in early development. The current focus is collecting training data through an annotation tool where you paint on images. Tool actions and stroke data will be logged and used to supervise the future model.

## Data Annotation Tool

A Pygame-based editor for painting on images and capturing edit operations. Images are resized to 512×512 for annotation.

### Tools

| Tool | Description |
|------|-------------|
| Brush | Draws with configurable size and color |
| Pencil | 1px strokes |
| Eraser | Restores original pixels under the stroke |
| Fill | Flood-fill with current color |

### Controls

- **Sliders**: Adjust R/G/B (0–255) and size (1–50). Click input boxes to type values; Enter to apply, Esc to cancel.
- **c**: Clear canvas to original image
- **q / Esc**: Quit

### Usage

```bash
python data_annotation/annotate.py <image_path>
```

Example:

```bash
python data_annotation/annotate.py data/images/pirlo.jpg
```

Output:

- Annotated frames are auto-saved to `data/drawn/` at ~30 fps when the canvas changes (format: `{image_name}{frame_id}.jpg`)
- Tool operations are saved to `data/logs/{image_name}_log.json` on exit, containing per-operation records with `timestamp`, `frame_id`, `tool`, `size`, `color`, `start_pos`, and `end_pos`

### Dependencies

```
pygame
numpy
Pillow
opencv-python
```

## Utilities

**to_jpg.py** — Convert and resize images to 512×512 JPEG:

```bash
python data_annotation/to_jpg.py <input_image> <output_image.jpg>
```

## Roadmap

- [x] Save tool/stroke logs to disk (JSON)
- [ ] AI model training pipeline
- [ ] Inference API for image manipulation

## Project Structure (Planned)

```
src/
  ai/           # Model loading, inference, pipelines
  image/        # Image processing utilities
  api/          # API routes
  ui/           # Frontend (if applicable)
  utils/        # Shared helpers
  config/       # Config and constants
  types/        # Pydantic models
data_annotation/
  annotate.py   # Painting annotation tool
  to_jpg.py     # Image conversion utility
tests/
  unit/
  integration/
```
