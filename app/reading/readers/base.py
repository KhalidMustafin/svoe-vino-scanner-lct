"""Общие утилиты читателей: размер кадра, JPEG, рамки, порядок строк, статусы, реестр."""

from __future__ import annotations

import base64
import io
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from app.reading.contracts import Box, CropName, Reader, Reading, ReadStatus, TextLine, image_sha1

if TYPE_CHECKING:
    from app.config import Settings

# Сбой среды не кэшируется: повтор может пройти. Итог модели кэшируется, даже плохой.
TRANSIENT_STATUSES: frozenset[ReadStatus] = frozenset({"timeout", "unavailable", "error"})
CACHEABLE_STATUSES: frozenset[ReadStatus] = frozenset({"ok", "empty", "loop", "garbage"})


def is_cacheable(status: ReadStatus) -> bool:
    return status in CACHEABLE_STATUSES


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Проверка входа читателя: RGB uint8, HxWx3."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"ожидается RGB HxWx3, получено {getattr(image, 'shape', type(image))}")
    if image.dtype != np.uint8:
        raise ValueError(f"ожидается uint8, получено {image.dtype}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("пустой кадр")
    return image


def long_side(image: np.ndarray) -> int:
    return int(max(image.shape[0], image.shape[1]))


def resize_long_side(image: np.ndarray, px: int) -> np.ndarray:
    """Кадр с длинной стороной не больше `px`. Только уменьшение: увеличение букв не добавляет."""
    ensure_rgb(image)
    if px <= 0:
        raise ValueError("px должен быть > 0")
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= px:
        return image
    scale = px / side
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    import cv2

    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def encode_jpeg_b64(image: np.ndarray, quality: int = 92) -> str:
    """RGB-массив в base64 JPEG."""
    ensure_rgb(image)
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def crop_px_for(reader: Any, image: np.ndarray) -> int:
    """Размер кропа для ключа. Читатель может задать свой (`crop_px_for`), иначе длинная сторона."""
    custom = getattr(reader, "crop_px_for", None)
    if callable(custom):
        return int(custom(image))
    return long_side(image)


def elapsed_ms_since(t0: float) -> int:
    return max(0, round((time.perf_counter() - t0) * 1000))


def make_reading(
    reader: Any,
    image: np.ndarray,
    *,
    params: str,
    crop: CropName,
    status: ReadStatus,
    elapsed_ms: int,
    lines: list[TextLine] | None = None,
    raw: str | None = None,
    prompt_tokens: int | None = None,
    crop_px: int | None = None,
) -> Reading:
    return Reading(
        reader=reader.id,
        version=reader.version,
        params_hash=params,
        image_sha1=image_sha1(image),
        crop=crop,
        crop_px=crop_px if crop_px is not None else crop_px_for(reader, image),
        lines=lines or [],
        raw=raw,
        status=status,
        elapsed_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
    )


def box_from_quad(points: Any, width: int, height: int) -> Box | None:
    """Четырёхугольник в пикселях → Box в долях кадра. Вырожденная рамка → None."""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0 or width <= 0 or height <= 0:
        return None
    x0 = float(np.clip(arr[:, 0].min() / width, 0.0, 1.0))
    x1 = float(np.clip(arr[:, 0].max() / width, 0.0, 1.0))
    y0 = float(np.clip(arr[:, 1].min() / height, 0.0, 1.0))
    y1 = float(np.clip(arr[:, 1].max() / height, 0.0, 1.0))
    if x1 <= x0 or y1 <= y0:
        return None
    return Box(x0=x0, y0=y0, x1=x1, y1=y1)


def quad_angle(points: Any) -> float | None:
    """Наклон верхней грани рамки в градусах (0 — горизонталь)."""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(arr) < 2:
        return None
    dx, dy = arr[1, 0] - arr[0, 0], arr[1, 1] - arr[0, 1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, dx))


def sort_reading_order(lines: list[TextLine]) -> list[TextLine]:
    """Сверху вниз, в строке — слева направо; id переназначаются по порядку.

    Строки без рамки остаются в конце в исходном порядке. Ряд — рамки, чей центр по вертикали
    ближе половины медианной высоты к центру ряда.
    """
    boxed = [line for line in lines if line.box is not None]
    rest = [line for line in lines if line.box is None]
    if boxed:
        heights = sorted(line.box.y1 - line.box.y0 for line in boxed)  # type: ignore[union-attr]
        tolerance = heights[len(heights) // 2] / 2
        rows: list[list[TextLine]] = []
        row_center = 0.0
        for line in sorted(boxed, key=lambda item: item.box.center[1]):  # type: ignore[union-attr]
            cy = line.box.center[1]  # type: ignore[union-attr]
            if rows and abs(cy - row_center) <= tolerance:
                rows[-1].append(line)
                row_center = sum(item.box.center[1] for item in rows[-1]) / len(rows[-1])  # type: ignore[union-attr]
            else:
                rows.append([line])
                row_center = cy
        boxed = [
            line
            for row in rows
            for line in sorted(row, key=lambda item: item.box.x0)  # type: ignore[union-attr]
        ]
    return [line.model_copy(update={"id": i}) for i, line in enumerate(boxed + rest)]


def build_reader(name: str, settings: Settings | None = None) -> Reader:
    """Реестр: "vlm:<model>", "vlm" (модель из настроек), "easyocr", "rapidocr[:server]"."""
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    kind, _, arg = name.partition(":")
    gpu = settings.device.startswith("cuda")
    if kind == "vlm":
        from app.reading.readers.ollama_vlm import OllamaVlmReader

        return OllamaVlmReader(arg or settings.vlm_model, settings.ollama_url)
    if kind == "easyocr" and not arg:
        from app.reading.readers.easyocr_reader import EasyOcrReader

        return EasyOcrReader(gpu=gpu)
    if kind == "rapidocr" and arg in ("", "mobile", "server"):
        from app.reading.readers.rapidocr_reader import RapidOcrReader

        return RapidOcrReader(rec_model_type=arg or "mobile")  # type: ignore[arg-type]
    raise ValueError(f"неизвестный читатель: {name!r}")
