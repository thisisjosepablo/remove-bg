"""Shared image helpers for Cutout Studio."""

from __future__ import annotations

import io
import os

from PIL import Image


def max_image_edge() -> int:
    raw = os.getenv("CUTOUT_MAX_IMAGE_EDGE", "1280")
    try:
        return max(512, min(int(raw), 4096))
    except ValueError:
        return 1280


def fit_image_for_processing(image: Image.Image) -> Image.Image:
    """Downscale very large photos before ML inference."""
    w, h = image.size
    limit = max_image_edge()
    longest = max(w, h)
    if longest <= limit:
        return image
    scale = limit / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def image_to_png_bytes(image: Image.Image, *, fast: bool = False) -> bytes:
    buffer = io.BytesIO()
    if fast:
        image.save(buffer, format="PNG", compress_level=1)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()