"""Где на кадре бутылка: детектор COCO из torchvision и выбор цели.

Детектор — подсказка, а не условие: не нашёл бутылку или не поднялся — цель берётся
центральной полосой кадра. Рамка считается надёжной, только если уверенность высокая
и цель накрывает центр кадра: заливка соседей по ошибочной рамке стёрла бы саму цель.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.normalize.decode import resize_long_side
from app.reading.contracts import Box

logger = logging.getLogger(__name__)

COCO_BOTTLE = 44
WEIGHTS_FILE = "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"

# Цель без детекций: бутылку у полки ставят в середину кадра.
CENTER_BAND = Box(x0=0.3, y0=0.0, x1=0.7, y1=1.0)
CONFIDENT_SCORE = 0.7
# Рамка, которая на эту долю лежит внутри цели, — часть той же бутылки, а не соседка.
SAME_BOTTLE_OVERLAP = 0.7
# Бутылки стоят в ряд: сдвиг по горизонтали говорит о соседке, по вертикали — о кадрировании.
VERTICAL_WEIGHT = 0.5
_MAX_CENTER_DISTANCE = math.hypot(0.5, 0.5 * VERTICAL_WEIGHT)
# Нижний порог модели; `score_thresh` в `detect` ниже него не действует.
_MODEL_SCORE_FLOOR = 0.05


class Detection(BaseModel):
    """Найденная бутылка: рамка в долях кадра и уверенность детектора."""

    model_config = ConfigDict(frozen=True)

    box: Box
    score: float = Field(ge=0.0, le=1.0)


class TargetSelection(BaseModel):
    """Какая бутылка — цель, какие — соседи, и можно ли рамке доверять маску."""

    target: Box
    neighbors: list[Box] = Field(default_factory=list)
    method: Literal["detector", "center_fallback"]
    score: float = Field(ge=0.0, le=1.0)
    confident: bool


def center_closeness(box: Box) -> float:
    """1 — центр рамки в центре кадра, 0 — в углу. Вертикальный сдвиг весит вдвое меньше."""
    cx, cy = box.center
    distance = math.hypot(cx - 0.5, (cy - 0.5) * VERTICAL_WEIGHT)
    return max(0.0, 1.0 - distance / _MAX_CENTER_DISTANCE)


def covers_center(box: Box) -> bool:
    return box.x0 <= 0.5 <= box.x1 and box.y0 <= 0.5 <= box.y1


def _overlap_share(inner: Box, outer: Box) -> float:
    """Доля площади `inner`, лежащая внутри `outer`."""
    width = min(inner.x1, outer.x1) - max(inner.x0, outer.x0)
    height = min(inner.y1, outer.y1) - max(inner.y0, outer.y0)
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / inner.area


def select_target(
    detections: Sequence[Detection],
    *,
    center_weight: float = 1.0,
    confident_score: float = CONFIDENT_SCORE,
) -> TargetSelection:
    """Цель — максимум «площадь × близость к центру ^ center_weight».

    Без детекций — центральная полоса 40 % ширины, `confident=False`.
    """
    if center_weight < 0:
        raise ValueError(f"select_target: center_weight >= 0, получено {center_weight}")
    if not detections:
        return TargetSelection(
            target=CENTER_BAND, method="center_fallback", score=0.0, confident=False
        )
    best = max(
        detections,
        key=lambda d: (d.box.area * center_closeness(d.box) ** center_weight, d.score),
    )
    neighbors = sorted(
        (
            d.box
            for d in detections
            if d is not best and _overlap_share(d.box, best.box) < SAME_BOTTLE_OVERLAP
        ),
        key=lambda box: box.center,
    )
    return TargetSelection(
        target=best.box,
        neighbors=neighbors,
        method="detector",
        score=best.score,
        confident=best.score >= confident_score and covers_center(best.box),
    )


def detections_from_output(
    boxes: Any,
    labels: Any,
    scores: Any,
    width: int,
    height: int,
    *,
    score_thresh: float = 0.3,
    min_area: float = 0.02,
) -> list[Detection]:
    """Выход детектора в пикселях → бутылки в долях кадра, по убыванию уверенности."""
    pixel_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    label_list = np.asarray(labels).reshape(-1).tolist()
    score_list = np.asarray(scores, dtype=np.float64).reshape(-1).tolist()
    found: list[Detection] = []
    for (x0, y0, x1, y1), label, score in zip(pixel_boxes, label_list, score_list, strict=True):
        if int(label) != COCO_BOTTLE or score < score_thresh:
            continue
        nx0, nx1 = np.clip([x0 / width, x1 / width], 0.0, 1.0)
        ny0, ny1 = np.clip([y0 / height, y1 / height], 0.0, 1.0)
        if nx1 <= nx0 or ny1 <= ny0 or (nx1 - nx0) * (ny1 - ny0) < min_area:
            continue
        box = Box(x0=float(nx0), y0=float(ny0), x1=float(nx1), y1=float(ny1))
        found.append(Detection(box=box, score=min(1.0, max(0.0, float(score)))))
    found.sort(key=lambda d: d.score, reverse=True)
    return found


def weights_path() -> Path:
    """Файл весов в кэше torch — там же, где его ищет `torch.hub`."""
    if "torch" in sys.modules:
        hub = Path(sys.modules["torch"].hub.get_dir())
    elif os.environ.get("TORCH_HOME"):
        hub = Path(os.environ["TORCH_HOME"]).expanduser() / "hub"
    else:
        cache = os.environ.get("XDG_CACHE_HOME") or str(Path("~/.cache").expanduser())
        hub = Path(cache).expanduser() / "torch" / "hub"
    return hub / "checkpoints" / WEIGHTS_FILE


class BottleDetector:
    """Faster R-CNN MobileNetV3-Large FPN (COCO), класс bottle. Модель поднимается лениво."""

    def __init__(self, device: str | None = None, *, allow_download: bool = False) -> None:
        self.requested_device = device or get_settings().device
        self.device: str | None = None
        self.allow_download = allow_download
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self._ensure()

    def _ensure(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if self._failed:
                return False
            if not self.allow_download and not weights_path().is_file():
                logger.warning("Детектор бутылок недоступен: нет весов %s", weights_path())
                self._failed = True
                return False
            try:
                import torch
                from torchvision.models.detection import (
                    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                    fasterrcnn_mobilenet_v3_large_fpn,
                )

                device = self.requested_device
                if device.startswith("cuda") and not torch.cuda.is_available():
                    logger.warning("CUDA недоступна, детектор бутылок пойдёт на CPU")
                    device = "cpu"
                model = fasterrcnn_mobilenet_v3_large_fpn(
                    weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                    box_score_thresh=_MODEL_SCORE_FLOOR,
                )
                self._model = model.to(device).eval()
                self.device = device
                return True
            except Exception as exc:  # noqa: BLE001 — без детектора остаётся центральная полоса
                logger.warning("Детектор бутылок недоступен: %s", exc)
                self._failed = True
                return False

    def detect(
        self,
        image: np.ndarray,
        *,
        proxy_long_side: int = 800,
        score_thresh: float = 0.3,
        min_area: float = 0.02,
    ) -> list[Detection]:
        """Бутылки на кадре RGB uint8. Детектор недоступен — пустой список."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"detect: ожидается RGB uint8 HxWx3, получено {image.shape}")
        if not self._ensure():
            return []
        import torch

        proxy = np.ascontiguousarray(resize_long_side(image, proxy_long_side, upscale=False))
        height, width = proxy.shape[:2]
        tensor = torch.from_numpy(proxy).permute(2, 0, 1).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            out = self._model([tensor.div_(255.0)])[0]
        return detections_from_output(
            out["boxes"].cpu().numpy(),
            out["labels"].cpu().numpy(),
            out["scores"].cpu().numpy(),
            width,
            height,
            score_thresh=score_thresh,
            min_area=min_area,
        )
