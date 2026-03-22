from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

__all__ = ["logger", "configure_logging"]

_configured = False

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _patcher_caller(record: dict) -> None:
    path_attr = getattr(record["file"], "path", None)
    path_str = path_attr if path_attr is not None else str(record["file"])
    p = Path(path_str).resolve()
    try:
        rel = p.relative_to(_PROJECT_ROOT)
        path_display = rel.as_posix()
    except ValueError:
        path_display = p.as_posix()
    record["extra"]["caller"] = f"{path_display} - {record['function']}"


def configure_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    lvl = (level or os.environ.get("LOGURU_LEVEL", "INFO")).upper()
    logger.remove()
    logger.configure(patcher=_patcher_caller)
    logger.add(
        sys.stderr,
        level=lvl,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>[{extra[caller]}]</magenta> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    _configured = True
