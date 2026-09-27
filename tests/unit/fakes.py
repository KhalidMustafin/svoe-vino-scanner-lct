"""Подставные эмбеддеры для тестов: векторы детерминированные, весов и сети не нужно.

Юнит-тесты проверяют сборку, формат и поиск, а не саму модель. Настоящая башня зрения
здесь не поднимается: она заняла бы видеокарту и потребовала весов.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from app.features.embedder import unit_rows


class PixelEmbedder:
    """Вектор кадра записан в самом кадре: тест решает, какие выйдут косинусы.

    Координата i — пиксель (0, i) канала 0, делённый на 255. Координаты неотрицательные,
    поэтому косинус лежит в 0..1, как у настоящей выдачи.
    """

    def __init__(self, dim: int = 4, model_name: str = "fake/pixel") -> None:
        self._dim = dim
        self.model_name = model_name
        self.batches: list[int] = []  # размеры порций: видно, что сборка бьёт кадры на батчи

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("embed: пустой список кадров")
        self.batches.append(len(images))
        rows = np.zeros((len(images), self._dim), dtype=np.float32)
        for i, image in enumerate(images):
            head = np.asarray(image)[0, : self._dim, 0].astype(np.float32)
            rows[i, : head.size] = head / 255.0
        return unit_rows(rows)


class HashEmbedder:
    """Вектор — хэш пикселей: тот же кадр даёт тот же вектор, разные почти ортогональны."""

    def __init__(self, dim: int = 8, model_name: str = "fake/hash") -> None:
        self._dim = dim
        self.model_name = model_name
        self.batches: list[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("embed: пустой список кадров")
        self.batches.append(len(images))
        rows = []
        for image in images:
            digest = hashlib.sha1(np.ascontiguousarray(image).tobytes()).digest()
            seed = int.from_bytes(digest[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=self._dim))
        return unit_rows(np.asarray(rows, dtype=np.float32))


class ColorEmbedder:
    """Вектор кадра — его средний цвет: у одноцветного кадра любой кроп даёт тот же вектор.

    Стенд поиска режет запрос на виды, и предсказать вектор по пикселю (0, i), как это делает
    `PixelEmbedder`, уже нельзя. Со средним цветом ответ на одноцветный кадр известен заранее:
    косинус — это угол между цветами.
    """

    def __init__(self, model_name: str = "fake/color") -> None:
        self.model_name = model_name
        self.batches: list[int] = []

    @property
    def dim(self) -> int:
        return 3

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("embed: пустой список кадров")
        self.batches.append(len(images))
        rows = [np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0) for image in images]
        return unit_rows(np.asarray(rows, dtype=np.float32) / 255.0)


def frame(values: Sequence[float], size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Кадр RGB uint8, из которого `PixelEmbedder` достанет именно этот вектор."""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i, value in enumerate(values):
        image[0, i, 0] = round(value * 255)
    return image
