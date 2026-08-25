"""Costruzione dei pannelli renali a partire dalle bounding box YOLO."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


def create_kidney_panel(
    image: Image.Image,
    box: Iterable[float],
    size: int = 224,
    margin_ratio: float = 0.05,
) -> Image.Image | None:
    image_width, image_height = image.size
    x1, y1, x2, y2 = map(float, box)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    x1 = max(0, int(np.floor(x1 - box_width * margin_ratio)))
    y1 = max(0, int(np.floor(y1 - box_height * margin_ratio)))
    x2 = min(image_width, int(np.ceil(x2 + box_width * margin_ratio)))
    y2 = min(image_height, int(np.ceil(y2 + box_height * margin_ratio)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return panel


def combine_panels(
    panels: list[Image.Image],
    size: int = 224,
) -> Image.Image:
    if not panels:
        raise ValueError("È necessario almeno un pannello valido")
    selected = list(panels[:2])
    while len(selected) < 2:
        selected.append(Image.new("RGB", (size, size), (0, 0, 0)))

    combined = Image.new("RGB", (size * 2, size), (0, 0, 0))
    combined.paste(selected[0], (0, 0))
    combined.paste(selected[1], (size, 0))
    return combined
