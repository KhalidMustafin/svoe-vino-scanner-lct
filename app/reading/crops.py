"""Кропы кадра для читателей. Только геометрия, никаких решений о цели.

Все рамки — `Box` в долях исходного кадра; на выходе всегда новый массив, исходный
кадр не меняется.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.normalize.decode import resize_long_side
from app.reading.contracts import Box

_EPS = 1e-9  # 0.8 * 200 = 160.00000000000003 не должно превращаться в 161 пиксель


def _check_image(image: np.ndarray) -> tuple[int, int]:
    if image.ndim not in (2, 3) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"ожидается кадр HxW или HxWxC, получено {image.shape}")
    return image.shape[1], image.shape[0]


def box_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """Рамка в пикселях (x0, y0, x1, y1), накрывающая долю целиком; не меньше 1 px."""
    x0 = min(max(math.floor(box.x0 * width + _EPS), 0), width - 1)
    y0 = min(max(math.floor(box.y0 * height + _EPS), 0), height - 1)
    x1 = max(min(math.ceil(box.x1 * width - _EPS), width), x0 + 1)
    y1 = max(min(math.ceil(box.y1 * height - _EPS), height), y0 + 1)
    return x0, y0, x1, y1


def pad_box(box: Box, pad_x: float = 0.0, pad_y: float = 0.0) -> Box:
    """Рамка, расширенная на долю своей ширины и высоты с каждой стороны, в пределах кадра."""
    if pad_x < 0 or pad_y < 0:
        raise ValueError("pad_box: отступы не могут быть отрицательными")
    dx = pad_x * (box.x1 - box.x0)
    dy = pad_y * (box.y1 - box.y0)
    return Box(
        x0=max(0.0, box.x0 - dx),
        y0=max(0.0, box.y0 - dy),
        x1=min(1.0, box.x1 + dx),
        y1=min(1.0, box.y1 + dy),
    )


def label_box(bottle_box: Box, top: float = 0.30, bottom: float = 1.0) -> Box:
    """Полоса этикетки внутри рамки бутылки: доли высоты бутылки от верха рамки."""
    if not 0.0 <= top < bottom <= 1.0:
        raise ValueError(f"label_box: ожидается 0 <= top < bottom <= 1, получено {top}, {bottom}")
    height = bottle_box.y1 - bottle_box.y0
    return Box(
        x0=bottle_box.x0,
        y0=bottle_box.y0 + top * height,
        x1=bottle_box.x1,
        y1=bottle_box.y0 + bottom * height,
    )


def crop_full(image: np.ndarray, long_side: int, *, upscale: bool = False) -> np.ndarray:
    """Кадр целиком с длинной стороной не больше `long_side`."""
    _check_image(image)
    resized = resize_long_side(image, long_side, upscale=upscale)
    return resized.copy() if resized is image else resized


def crop_box(image: np.ndarray, box: Box, pad_x: float = 0.08, pad_y: float = 0.04) -> np.ndarray:
    """Вырез по рамке с отступом: детектор режет бутылку впритык, края этикетки теряются."""
    width, height = _check_image(image)
    x0, y0, x1, y1 = box_pixels(pad_box(box, pad_x, pad_y), width, height)
    return np.ascontiguousarray(image[y0:y1, x0:x1]).copy()


def label_band(
    image: np.ndarray,
    bottle_box: Box,
    top: float = 0.30,
    bottom: float = 1.0,
    *,
    pad_x: float = 0.08,
) -> np.ndarray:
    """Полоса этикетки внутри бутылки: у горлышка букв обычно нет."""
    return crop_box(image, label_box(bottle_box, top, bottom), pad_x=pad_x, pad_y=0.0)


def center_band(image: np.ndarray, width_share: float = 0.6, upscale: float = 1.5) -> np.ndarray:
    """Средняя вертикальная полоса, увеличенная по Ланцошу.

    Середина цилиндра читается лучше краёв: там буквы не сжаты кривизной.
    """
    width, height = _check_image(image)
    if not 0.0 < width_share <= 1.0:
        raise ValueError(f"center_band: width_share в (0, 1], получено {width_share}")
    if upscale <= 0:
        raise ValueError(f"center_band: upscale должен быть > 0, получено {upscale}")
    margin = (1.0 - width_share) / 2
    band = Box(x0=margin, y0=0.0, x1=1.0 - margin, y1=1.0)
    x0, y0, x1, y1 = box_pixels(band, width, height)
    region = np.ascontiguousarray(image[y0:y1, x0:x1])
    if upscale == 1.0:
        return region.copy()
    size = (max(1, round((x1 - x0) * upscale)), max(1, round((y1 - y0) * upscale)))
    interpolation = cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA
    return cv2.resize(region, size, interpolation=interpolation)
