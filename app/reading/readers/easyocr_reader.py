"""EasyOCR: строки с рамками и уверенностью.

Не `readtext`, а `detect` + `recognize`: `readtext` на массиве HxWx3 считает его BGR и берёт
серый через BGR2GRAY, а у нас RGB — красные буквы на светлом теряли бы контраст.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from typing import Any

import numpy as np

from app.reading.contracts import CropName, Reading, TextLine, params_hash
from app.reading.readers.base import (
    box_from_quad,
    elapsed_ms_since,
    ensure_rgb,
    make_reading,
    quad_angle,
    sort_reading_order,
)

POSTPROCESS_VERSION = "1"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


class EasyOcrReader:
    id = "easyocr"

    def __init__(
        self,
        langs: tuple[str, ...] = ("ru", "en"),
        gpu: bool = True,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        *,
        download_enabled: bool = False,
    ) -> None:
        self.langs = tuple(langs)
        self.gpu = gpu
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.download_enabled = download_enabled
        self.version = _version("easyocr")
        self.params_hash = params_hash(
            {
                "langs": list(self.langs),
                "canvas_size": canvas_size,
                "mag_ratio": mag_ratio,
                "text_threshold": text_threshold,
                "low_text": low_text,
                "decoder": "greedy",
                "postprocess": POSTPROCESS_VERSION,
            }
        )
        self._engine: Any = None

    def available(self) -> bool:
        return importlib.util.find_spec("easyocr") is not None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            import easyocr

            self._engine = easyocr.Reader(
                list(self.langs),
                gpu=self.gpu,
                verbose=False,
                download_enabled=self.download_enabled,
            )
        return self._engine

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        """Бюджет не прерывает движок; `budget_ms` <= 0 — timeout без запуска."""
        ensure_rgb(image)
        t0 = time.perf_counter()
        params = self.params_hash
        if budget_ms <= 0:
            return make_reading(
                self, image, params=params, crop=crop, status="timeout", elapsed_ms=0
            )
        try:
            engine = self._ensure_engine()
        except Exception as exc:  # noqa: BLE001 — нет пакета или весов
            return make_reading(
                self,
                image,
                params=params,
                crop=crop,
                status="unavailable",
                elapsed_ms=elapsed_ms_since(t0),
                raw=f"{type(exc).__name__}: {exc}",
            )
        try:
            import cv2

            grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            horizontal, free = engine.detect(
                image,
                text_threshold=self.text_threshold,
                low_text=self.low_text,
                canvas_size=self.canvas_size,
                mag_ratio=self.mag_ratio,
                reformat=False,
            )
            results = engine.recognize(
                grey,
                horizontal[0],
                free[0],
                decoder="greedy",
                detail=1,
                paragraph=False,
                reformat=False,
            )
        except Exception as exc:  # noqa: BLE001
            return make_reading(
                self,
                image,
                params=params,
                crop=crop,
                status="error",
                elapsed_ms=elapsed_ms_since(t0),
                raw=f"{type(exc).__name__}: {exc}",
            )
        h, w = image.shape[:2]
        lines = [
            TextLine(
                id=i,
                text=str(text).strip(),
                box=box_from_quad(quad, w, h),
                angle=quad_angle(quad),
                conf=float(conf),
            )
            for i, (quad, text, conf) in enumerate(results)
            if str(text).strip()
        ]
        lines = sort_reading_order(lines)
        return make_reading(
            self,
            image,
            params=params,
            crop=crop,
            status="ok" if lines else "empty",
            elapsed_ms=elapsed_ms_since(t0),
            lines=lines,
        )
