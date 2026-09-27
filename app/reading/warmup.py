"""Прогрев читателей до замера: холодная загрузка модели не должна съедать бюджет кадра.

Ollama грузит модель в память на первом запросе (у моделей 4B — около 3–4 с), и первый
кадр упирается в сервисный бюджет 2,5 с, а timeout не кэшируется. Прогрев читает
синтетический кадр с большим бюджетом до цикла по кадрам. Кэш чтений он обходит:
`CachedReader` разворачивается до настоящего читателя, чтобы синтетический кадр не лёг в
кэш. Читатель со своим методом `warm` (модель зрения) прогревается им. Ошибки не бросаются:
прогрев возвращает статус чтения или `error`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from app.reading.contracts import Reader, Reading
from app.reading.readers.cache import CachedReader

DEFAULT_WARMUP_BUDGET_MS = 60_000
WARMUP_SIZE = 1024
#: Строки синтетической этикетки: латиница и цифры есть в любом шрифте Pillow.
WARMUP_LINES = ("CHATEAU WARMUP", "Cabernet Sauvignon", "Reserve 2021", "13,5% vol  0,75 L")


def synthetic_label(size: int = WARMUP_SIZE) -> np.ndarray:
    """Белый кадр `size`×`size` с несколькими строками тёмного текста (RGB uint8)."""
    from PIL import Image, ImageDraw, ImageFont

    if size <= 0:
        raise ValueError(f"size должен быть > 0, получено {size}")
    picture = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(picture)
    font_px = max(8, size // 14)
    try:
        font: Any = ImageFont.load_default(size=font_px)
    except (OSError, TypeError, ValueError):  # Pillow без FreeType — растровый шрифт
        font = ImageFont.load_default()
    step = size // (len(WARMUP_LINES) + 1)
    for n, text in enumerate(WARMUP_LINES, start=1):
        draw.text((size // 12, n * step - font_px // 2), text, fill=(20, 20, 20), font=font)
    return np.ascontiguousarray(np.asarray(picture, dtype=np.uint8))


def _unwrap(reader: Reader) -> Reader:
    while isinstance(reader, CachedReader):
        reader = reader.reader
    return reader


def warm_readers(
    readers: Sequence[Reader],
    *,
    image: np.ndarray | None = None,
    budget_ms: int = DEFAULT_WARMUP_BUDGET_MS,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, dict[str, Any]]:
    """Один прогревочный кадр каждому читателю по очереди.

    Возвращает `{id читателя: {"status", "elapsed_ms"}}` (у повтора id — `id#2`); при
    исключении статус `error` и текст ошибки в `error`. Кэш чтений не пишется.
    """
    frame = synthetic_label() if image is None else image
    report: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {}
    for reader in readers:
        name = str(getattr(reader, "id", type(reader).__name__))
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        target = _unwrap(reader)
        started = clock()
        error: str | None = None
        try:
            warm = getattr(target, "warm", None)
            reading: Reading = (
                warm(frame, budget_ms=budget_ms)
                if callable(warm)
                else target.read(frame, crop="full", budget_ms=budget_ms)
            )
            status: str = reading.status
        except Exception as exc:  # noqa: BLE001 — сбой прогрева не роняет прогон
            status, error = "error", f"{type(exc).__name__}: {exc}"
        row: dict[str, Any] = {
            "status": status,
            "elapsed_ms": max(0, round((clock() - started) * 1000)),
        }
        if error is not None:
            row["error"] = error
        report[label] = row
    return report
