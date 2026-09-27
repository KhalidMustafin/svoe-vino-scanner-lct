"""Требования блюда к вину: из описания справочника — в числа. Перенос «Лозы».

Источник — `Code/backend/app/recommend/requirements.py` (f2b4b66) без изменений логики, только
аннотации типов под Python 3.12. У каждого блюда справочника есть строка вида «acidity 3.5-4.5,
tannin 1.5-3, body 2.5-3.5, oak <=1.5, sweetness <=1». Одна ось может быть названа дважды
(«effervescence 0 или >=3» — либо тихое, либо игристое), бывают точные значения без диапазона,
а после чисел идёт обычный текст, в котором цифры попадаться не должны.
"""

from __future__ import annotations

import re
from typing import Any

#: Оси, по которым справочник задаёт требования.
AXES = (
    "acidity",
    "tannin",
    "body",
    "oak",
    "sweetness",
    "alcohol",
    "effervescence",
    "aroma_intensity",
)

#: Полбалла допуска: требования писали люди, и «oak <=1.5» против дуба 1.6 — округление.
SLACK = 0.5

Limits = dict[str, list[tuple[float, float]]]

_NUMBER = r"\d+(?:\.\d+)?"
_CONDITION = r"(<=|>=|=)?\s*(" + _NUMBER + r")(?:\s*-\s*(" + _NUMBER + r"))?"
_REQUIREMENT_RE = re.compile(
    r"\b("
    + "|".join(AXES)
    + r")\s*"
    + _CONDITION
    # Второе условие той же оси идёт без её имени: «0 или >=3».
    + r"(?:\s*или\s*"
    + _CONDITION
    + r")?"
)


def _range_of(operator: str, first: str, second: str) -> tuple[float, float]:
    """Одно условие — в границы. Пустая шкала считается от 0 до 5."""
    if second:
        return float(first), float(second)
    if operator == "<=":
        return 0.0, float(first)
    if operator == ">=":
        return float(first), 5.0
    return float(first), float(first)


def parse(text: str | None) -> Limits:
    """Границы по осям; у оси может быть несколько допустимых диапазонов."""
    limits: Limits = {}
    for match in _REQUIREMENT_RE.finditer(text or ""):
        axis, operator, first, second, alt_op, alt_first, alt_second = match.groups()
        ranges = [_range_of(operator or "", first, second or "")]
        if alt_first:
            ranges.append(_range_of(alt_op or "", alt_first, alt_second or ""))
        limits.setdefault(axis, []).extend(ranges)
    return limits


def satisfies(profile: Any, limits: Limits) -> float:
    """Доля осей, где вино укладывается в требования, от -1 до 1.

    Проверка нарочно бинарная по оси: непрерывный штраф «Лоза» проверила, и он оказался хуже
    (42 блюда из 81 против 54) — мягкий штраф тонет среди прочих весов подбора.
    """
    if not limits:
        return 0.0
    hits = 0
    for axis, ranges in limits.items():
        value = getattr(profile, axis, None)
        if value is None:
            continue
        if any(low - SLACK <= value <= high + SLACK for low, high in ranges):
            hits += 1
    return 2.0 * hits / len(limits) - 1.0
