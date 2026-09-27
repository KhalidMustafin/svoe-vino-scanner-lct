"""Контракты чтения этикетки.

OCR не выбирает вино. Он отдаёт признаки этикетки (`LabelFields`) с доказательствами,
а решение принимает слой resolve, переранжируя кандидатов CV. Текстовый top-1 —
только диагностика.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SugarClass(StrEnum):
    """Классы сахара — те же коды, что в разметке каталога (`gt_tokens.jsonl`)."""

    BRUT_NATURE = "brut_nature"
    EXTRA_BRUT = "extra_brut"
    BRUT = "brut"
    DRY = "suhoe"
    SEMI_DRY = "polusuhoe"
    SEMI_SWEET = "polusladkoe"
    SWEET = "sladkoe"


class Color(StrEnum):
    """Цвет вина — значения колонки «Категория» каталога."""

    WHITE = "Белое"
    RED = "Красное"
    ROSE = "Розовое"
    ORANGE = "Оранжевое"


CropName = Literal["full", "bottle", "label", "band"]
ReadStatus = Literal["ok", "empty", "loop", "garbage", "timeout", "unavailable", "error"]
TokenKind = Literal["word", "number", "roman", "year", "abv", "ratio"]
LexField = Literal["producer", "cuvee", "grape", "sugar", "serial", "color"]


class Box(BaseModel):
    """Прямоугольник в долях исходного кадра (0..1)."""

    model_config = ConfigDict(frozen=True)

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Box:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Box: ожидается x0 < x1 и y0 < y1")
        return self

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


class TextLine(BaseModel):
    """Строка, как её вернул читатель."""

    id: int
    text: str
    box: Box | None = None
    angle: float | None = None
    conf: float | None = None  # None — читатель не даёт калиброванной уверенности


class Reading(BaseModel):
    """Одно чтение одного кадра одним читателем в одном режиме кропа."""

    reader: str
    version: str
    params_hash: str
    image_sha1: str
    crop: CropName
    crop_px: int = Field(gt=0)
    lines: list[TextLine] = Field(default_factory=list)
    raw: str | None = None  # сырой ответ модели — для аудита выдумок
    status: ReadStatus = "ok"
    elapsed_ms: int = Field(ge=0)
    prompt_tokens: int | None = None

    @property
    def key(self) -> str:
        """Ключ кэша: меняется при смене модели, параметров, кадра или кропа."""
        return f"{self.reader}@{self.version}|{self.params_hash}|{self.image_sha1}|{self.crop}|{self.crop_px}"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class TokenSpan(BaseModel):
    """Токен строки после нормализации."""

    text: str
    norm: str  # NFKC, нижний регистр, ё→е, гомоглифы сведены
    skeleton: str  # общий «скелет» кириллицы и латиницы для сопоставления транслитераций
    kind: TokenKind
    line_id: int
    reading: str  # Reading.key
    conf: float | None = None


class LexHit(BaseModel):
    """Попадание токена или фразы в закрытый словарь каталога."""

    model_config = ConfigDict(frozen=True)

    canonical: str
    field: LexField
    cost: float = Field(ge=0.0)  # 0 — точное совпадение
    slugs: frozenset[str]


class Evidence[T](BaseModel):
    """Значение поля с тем, откуда оно взялось."""

    value: T
    sources: list[str] = Field(default_factory=list)  # ключи Reading
    support: int = Field(default=1, ge=1)  # число независимых чтений
    conf: float | None = None
    matched: str | None = None  # каноническая форма из словаря, если есть


class LabelFields(BaseModel):
    """Признаки этикетки для resolve. Пустое поле — «не прочитано», а не «нет на этикетке»."""

    producer: list[Evidence[str]] = Field(default_factory=list)
    cuvee: list[Evidence[str]] = Field(default_factory=list)
    grapes: list[Evidence[str]] = Field(default_factory=list)
    sugar: list[Evidence[SugarClass]] = Field(default_factory=list)
    vintage: Evidence[int] | None = None
    serial: list[Evidence[str]] = Field(default_factory=list)  # XXIV, «резерв», «30/70»
    abv: Evidence[float] | None = None
    color: Evidence[Color] | None = None
    unmatched: list[Evidence[str]] = Field(default_factory=list)  # сильные токены вне словаря
    is_wine_label: bool = True


class ReadResult(BaseModel):
    """Итог модуля чтения для одного кадра."""

    readings: list[Reading] = Field(default_factory=list)
    tokens: list[TokenSpan] = Field(default_factory=list)
    fields: LabelFields = Field(default_factory=LabelFields)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)  # например ["vlm_timeout"]


@runtime_checkable
class Reader(Protocol):
    """Читатель этикетки: классический OCR или модель зрения."""

    id: str
    version: str

    def available(self) -> bool: ...

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        """`image` — RGB uint8, HxWx3, уже вырезанный по `crop`."""
        ...


def image_sha1(image: np.ndarray) -> str:
    """Хэш пикселей и формы массива — стабильный ключ кадра для кэша чтений."""
    h = hashlib.sha1()
    h.update(str(image.shape).encode("ascii"))
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def params_hash(params: dict[str, Any]) -> str:
    """Короткий хэш параметров читателя (промпт, размер, num_predict и т. п.)."""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
