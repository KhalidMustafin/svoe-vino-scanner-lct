"""Линейный адаптер поиска: обученное выбеливание векторов SigLIP (фундаментальный трек 26.09).

Одна матрица `W` (d × D) и центр `mean` (d,) — к 4 окнам запроса после SigLIP и к строкам индекса
один раз при загрузке, затем нормировка: `x' = norm((x − mean) · W)`. Дальше поиск тот же (`zmax`,
top-K, отрыв по каталогу). Карта — Radenović learned whitening по парам «окно кадра — вид эталона
верной карточки» (`research/2026-09-26_fund/adapter/`, `PREREG_final.md`): гасит направления, которыми
кадр отличается от своего эталона (свет, фон, ракурс, телефон).

Включено по умолчанию с 26.09 (`SVS_CANDIDATE=adapter-lw-ranker`, `app/api/config.py`);
`SVS_CANDIDATE=off` — поиск без адаптера, как до 26.09. Карта
зависит от векторов индекса (центр, поворот, пары), поэтому в файле записан sha1 индекса, для
которого она собрана, и с другим индексом сервис не стартует.

Формат файла — `npz`: `mean` (d,) float32, `W` (d, D) float32 и `meta` — строка JSON c полем
`format = "svs-cv-adapter/1"`. Тождество карты — `content_sha1`: sha1 байтов `mean` и `W`
(float32, C-порядок); по нему модель resolve, обученная на выдаче адаптера, узнаёт «свой» адаптер.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

ADAPTER_FORMAT = "svs-cv-adapter/1"


class AdapterError(ValueError):
    """Файл адаптера не читается или не сходится с индексом."""


def content_sha1(mean: np.ndarray, W: np.ndarray) -> str:
    """sha1 карты по содержимому: байты `mean` и `W` в float32, C-порядок."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mean, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(W, dtype=np.float32).tobytes())
    return h.hexdigest()


def file_sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class LinearAdapter:
    """`x -> norm((x − mean) · W)`: те же операции и тот же float32, что у стенда трека.

    Порядок операций повторяет `research/2026-09-26_fund/adapter/linear_maps.LinearMap.apply`
    бит в бит: выдача сервиса с адаптером равна выдаче, на которой оценивались кандидаты.
    """

    mean: np.ndarray
    W: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean)
        W = np.asarray(self.W)
        if W.ndim != 2 or mean.shape != (W.shape[0],):
            raise AdapterError(f"адаптер: mean {mean.shape} и W {W.shape} не сходятся")
        if mean.dtype != np.float32 or W.dtype != np.float32:
            raise AdapterError(f"адаптер: ожидается float32, получено {mean.dtype} / {W.dtype}")
        if not (np.isfinite(mean).all() and np.isfinite(W).all()):
            raise AdapterError("адаптер: в mean или W не числа")

    @property
    def dim_in(self) -> int:
        return int(self.W.shape[0])

    @property
    def dim_out(self) -> int:
        return int(self.W.shape[1])

    @cached_property
    def sha1(self) -> str:
        """Тождество карты по содержимому (`content_sha1`); массивы после загрузки не меняются."""
        return content_sha1(self.mean, self.W)

    @cached_property
    def file_sha1(self) -> str | None:
        return file_sha1(self.path) if self.path is not None else None

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or "")

    @property
    def index_sha1(self) -> str | None:
        value = self.meta.get("index_sha1")
        return str(value) if value else None

    def threshold(self, key: str) -> float | None:
        """Порог на шкале счёта с адаптером (`suggest_not_found_visual_max`, `abstain_visual_floor`)."""
        value = self.meta.get(key)
        return float(value) if value is not None else None

    def apply(self, X: np.ndarray) -> np.ndarray:
        if np.asarray(X).shape[-1] != self.dim_in:
            raise AdapterError(
                f"адаптер ждёт векторы {self.dim_in}, получено {np.asarray(X).shape}"
            )
        Y = (np.asarray(X, dtype=np.float32) - self.mean) @ self.W
        n = np.linalg.norm(Y, axis=-1, keepdims=True)
        return (Y / np.maximum(n, 1e-12)).astype(np.float32)

    def public(self) -> dict[str, Any]:
        """Паспорт для `/v1/health`: без массивов."""
        return {
            "name": self.name,
            "kind": self.meta.get("kind"),
            "sha1": self.sha1,
            "file_sha1": self.file_sha1,
            "index_sha1": self.index_sha1,
            "dim": [self.dim_in, self.dim_out],
            "suggest_not_found_visual_max": self.threshold("suggest_not_found_visual_max"),
            "abstain_visual_floor": self.threshold("abstain_visual_floor"),
        }

    # -------------------------------------------------------------- файл
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {**self.meta, "format": ADAPTER_FORMAT, "content_sha1": self.sha1}
        with path.open("wb") as fh:
            np.savez(
                fh, mean=self.mean, W=self.W, meta=np.array(json.dumps(meta, ensure_ascii=False))
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LinearAdapter:
        path = Path(path)
        if not path.is_file():
            raise AdapterError(f"нет файла адаптера {path}")
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = {"mean", "W", "meta"} - set(data.files)
                if missing:
                    raise AdapterError(f"{path}: в файле нет полей {sorted(missing)}")
                meta = json.loads(str(data["meta"].item()))
                mean = np.asarray(data["mean"])
                W = np.asarray(data["W"])
        except (OSError, ValueError) as exc:
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError(f"{path}: не читается: {exc}") from exc
        if meta.get("format") != ADAPTER_FORMAT:
            raise AdapterError(
                f"{path}: формат {meta.get('format')!r}, ожидается {ADAPTER_FORMAT!r}"
            )
        adapter = cls(mean, W, meta, path)
        recorded = meta.get("content_sha1")
        if recorded and recorded != adapter.sha1:
            raise AdapterError(f"{path}: содержимое не совпадает с записанным sha1 {recorded[:8]}")
        return adapter
