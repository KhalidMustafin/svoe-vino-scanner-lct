"""Контракты визуального канала: виды кадра, метаданные индекса и его выдача.

CV первичен, но вино не выбирает: слой признаков отдаёт слою resolve список кандидатов
с косинусами и отрывом top-1 от top-2, а решение — переранжировать их полями этикетки
или отказать — принимается выше. Поэтому `VisualResult` не знает слова «ответ».

Имена видов совпадают с `CropName` модуля чтения: и читателю, и модели зрения показывают
один и тот же кусок кадра, чтобы «этикетка» в отчёте означала одно и то же.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

#: Вид кропа: вся бутылка, полоса этикетки, её центральная лента, весь кадр.
ViewName = Literal["bottle", "label", "band", "full"]

#: Канонический порядок видов: в нём они кладутся в индекс и в нём считается запрос.
VIEWS: tuple[ViewName, ...] = ("bottle", "label", "band", "full")

#: Как близости нескольких векторов одного slug сводятся к одной. `zmax` — максимум после
#: выравнивания пар «окно запроса × вид эталона» (`app.features.index.align_pairs`).
Aggregation = Literal["max", "mean", "zmax"]


class IndexMeta(BaseModel):
    """Паспорт индекса: чем и из чего он собран.

    `source_sha1` — хэш входного списка эталонов (обычно `slug_photo_map.csv`): по нему
    видно, что индекс собран из того же каталога, что и разметка.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    dim: int = Field(gt=0)
    views: list[ViewName] = Field(default_factory=list)
    n_slugs: int = Field(ge=0)
    n_vectors: int = Field(ge=0)
    built_at: str = ""  # UTC, ISO 8601
    source_sha1: str = ""


class Candidate(BaseModel):
    """Позиция каталога, похожая на запрос."""

    model_config = ConfigDict(frozen=True)

    slug: str
    #: Косинус; отрицательный сведён к нулю. При `per_slug="zmax"` — косинус, приведённый к
    #: общему распределению запроса: та же шкала, но у сильного совпадения бывает выше 1.
    score: float = Field(ge=0.0)
    view: ViewName  # вид эталона, давший лучшую близость, — для разбора промахов
    rank: int = Field(ge=1)


class VisualResult(BaseModel):
    """Выдача визуального канала для одного запроса.

    `margin` — отрыв top-1 от top-2 по slug на сырых косинусах, по всему каталогу, а не по
    срезу `top_k`. Это главный признак уверенности канала: близнецы каталога (одна
    этикетка, разный год) дают отрыв около нуля, и решать между ними должен текст.

    `timings_ms` — миллисекунды с долями: плоский поиск по 6 309 векторам занимает 1,4 мс,
    и в целых числах вся строка была бы нулём.
    """

    candidates: list[Candidate] = Field(default_factory=list)
    margin: float = Field(default=0.0, ge=0.0)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    model: str = ""

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


@runtime_checkable
class Embedder(Protocol):
    """Считает единичные векторы кадров. Реализация — `SiglipEmbedder` или фейк в тестах."""

    model_name: str

    @property
    def dim(self) -> int: ...

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """`images` — RGB uint8 HxWx3; на выходе float32 (N, dim), строки L2-нормированы."""
        ...
