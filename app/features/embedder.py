"""Вектор кадра: башня зрения SigLIP.

Признаки ImageNet различают «бутылку на белом», а не этикетку. SigLIP учился сводить
картинку с подписью и поэтому кодирует то, чем одно вино отличается от другого, —
надписи, логотип, рисунок. На 358 честных парах «Лозы» (запрос — фото портала, индекс —
студийные снимки Роскачества) SigLIP-base-224 дал top-1 58 % против 25 % у MobileNetV3.

Из всей модели берётся только башня зрения: текстовая половина — ещё столько же памяти,
а слов мы ей не показываем.

Веса только из локального кэша Hugging Face (`local_files_only=True`). Нет весов или нет
`transformers` — `ModelNotAvailable`, и канал объявляется недоступным; скачивать модель
сканер не станет ни при какой настройке.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from app.config import get_settings
from app.normalize.decode import resize_long_side

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/siglip-base-patch16-224"

#: Сторона квадрата, в который вписывается кадр перед процессором модели. Больше входа
#: модели: процессор уменьшит сам, а центрального кадрирования, срезающего верх и низ
#: высокой бутылки вместе с частью этикетки, не случится.
LETTERBOX_SIDE = 448

#: Цвет полей letterbox. Белый — как фон packshot каталога (проверено в «Лозе»).
LETTERBOX_FILL = 255

DEFAULT_BATCH_SIZE = 16


class ModelNotAvailable(RuntimeError):
    """Нет библиотеки или весов в кэше. Скачивание запрещено — канал работает без CV."""


def letterbox(
    image: np.ndarray, side: int = LETTERBOX_SIDE, fill: int = LETTERBOX_FILL
) -> np.ndarray:
    """Кадр целиком в квадрате `side`×`side`, поля залиты `fill`, пропорции сохранены."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"letterbox: ожидается RGB uint8 HxWx3, получено {image.shape}")
    if side <= 0:
        raise ValueError(f"letterbox: side должен быть > 0, получено {side}")
    resized = resize_long_side(image, side, upscale=True)
    height, width = resized.shape[:2]
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


class SiglipEmbedder:
    """Единичные векторы кадров. Модель поднимается лениво и один раз.

    Прогрев и первый запрос приходят почти одновременно, поэтому подъём под замком:
    иначе второй поток мог бы увидеть модель раньше процессора изображений.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: str = "float32",
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        side: int = LETTERBOX_SIDE,
        fill: int = LETTERBOX_FILL,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"SiglipEmbedder: batch_size должен быть > 0, получено {batch_size}")
        if dtype not in ("float32", "float16"):
            raise ValueError(f"SiglipEmbedder: dtype float32 или float16, получено {dtype!r}")
        self.model_name = model_name
        self.requested_device = device or get_settings().device
        self.requested_dtype = dtype
        self.batch_size = batch_size
        self.side = side
        self.fill = fill
        self.device: str | None = None
        self.dtype: str | None = None
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        """Поднимается ли модель. Ошибку не поднимает: без CV сканер читает буквы."""
        try:
            self._ensure()
        except ModelNotAvailable as exc:
            logger.warning("Модель снимков %s недоступна: %s", self.model_name, exc)
            return False
        return True

    @property
    def dim(self) -> int:
        """Длина вектора модели. Первое обращение поднимает веса."""
        self._ensure()
        assert self._dim is not None
        return self._dim

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoImageProcessor, SiglipVisionModel
            except ImportError as exc:
                raise ModelNotAvailable(
                    f"{self.model_name}: нет пакета для запуска модели ({exc}). "
                    "Установка и скачивание весов — отдельным шагом, сканер их не делает."
                ) from exc
            device = self.requested_device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA недоступна, векторы кадров пойдут на CPU")
                device = "cpu"
            dtype = self.requested_dtype
            if dtype == "float16" and not device.startswith("cuda"):
                # На CPU половинная точность в torch считается медленнее полной, а часть
                # ядер её просто не поддерживает.
                logger.warning("float16 просили на %s: считаем в float32", device)
                dtype = "float32"
            try:
                model = SiglipVisionModel.from_pretrained(self.model_name, local_files_only=True)
                processor = AutoImageProcessor.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception as exc:
                raise ModelNotAvailable(
                    f"{self.model_name}: весов нет в локальном кэше Hugging Face ({exc}). "
                    "Скачивание запрещено: положите веса в кэш заранее."
                ) from exc
            torch_dtype = torch.float16 if dtype == "float16" else torch.float32
            self._torch = torch
            self._processor = processor
            self._model = model.to(device=device, dtype=torch_dtype).eval()
            self._dim = int(model.config.hidden_size)
            self.device, self.dtype = device, dtype
            logger.info(
                "SigLIP %s поднят на %s (%s), dim %d", self.model_name, device, dtype, self._dim
            )

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """Матрица единичных векторов float32, строка на кадр, порядок входа сохранён."""
        if not images:
            raise ValueError("embed: пустой список кадров")
        self._ensure()
        torch = self._torch
        out = np.empty((len(images), self.dim), dtype=np.float32)
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            prepared = [letterbox(image, self.side, self.fill) for image in chunk]
            with torch.inference_mode():
                inputs = self._processor(images=prepared, return_tensors="pt")
                pixels = inputs["pixel_values"].to(device=self.device, dtype=self._model.dtype)
                vectors = self._model(pixel_values=pixels).pooler_output
                vectors = torch.nn.functional.normalize(vectors.float(), dim=1)
            out[start : start + len(chunk)] = vectors.cpu().numpy().astype(np.float32)
        return out


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Строки, нормированные до единичной длины: косинус должен остаться косинусом.

    Страховка индекса: эмбеддер обещает единичные векторы, но индекс обязан считать
    именно косинус, а не скалярное произведение чего попало. Нулевая строка остаётся нулевой.

    NaN и Inf — отказ, а не тихая выдача. Нефинитный вектор превращает все косинусы в NaN,
    а NaN в ранжировании не проигрывает никому: кадр получил бы «ответ» со счётом 0,0 и
    отрывом 0,0 вместо записи в сбойные. Половинная точность на видеокарте переполняется
    молча, поэтому проверка стоит здесь, а не только у модели.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"unit_rows: ожидается матрица (N, dim), получено {matrix.shape}")
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise ValueError(
            f"unit_rows: {bad.size} строк(и) из {len(matrix)} с NaN или Inf, "
            f"первая — {int(bad[0])}; косинус с такой строкой не число"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)
