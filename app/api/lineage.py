"""Объявленные производные разметки каталога: из какого `gt_tokens.jsonl` и чем получен загруженный.

Модель resolve хранит sha1 разметки, на которой училась (`meta.gt_tokens_sha1`). Потом разметку
правили без переобучения модели (веса и `FEATURE_VERSION` те же): таблицей правок карточек Э4
(`data/gt/gt_fixes.tsv`) и дописанными карточками живого портала Э3 (`SVS_LIVE_CARDS`). Такая
разметка не «чужая» — каждый шаг объявлен заранее в `configs/resolve/gt_lineage.json`: sha1 до
шага, после него и того, что шаг добавляет.

`provenance` (`app/api/service.py`) считает разметку согласованной с моделью, если загруженный gt
— тот, на котором обучена модель, или получен из него цепочкой объявленных шагов. Шаг с
`requires_live_cards` действует только при включённом флаге живых карточек. Любой другой gt —
`consistent=false`, как и раньше. Сверка смотрит только на sha1: что шаг воспроизводится из своих
входов, проверяют тесты (`tests/unit/test_gt_lineage.py`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.api.config import GT_LINEAGE_PATH

LINEAGE_FORMAT = "svs-gt-lineage/1"
_SHA1 = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[a-z0-9_]+")


class LineageError(ValueError):
    """Файл объявленных производных не разбирается: сверка разметки не может ему верить."""


@dataclass(frozen=True)
class LineageStep:
    """Шаг «gt `source` + вход `input_sha1` → gt `target`»."""

    name: str
    source: str
    target: str
    input_sha1: str
    requires_live_cards: bool = False

    @property
    def label(self) -> str:
        """Метка шага в `/v1/health`: имя и начало sha1 входа, `gt_fixes@2d7050bf`."""
        return f"{self.name}@{self.input_sha1[:8]}"


@dataclass(frozen=True)
class GtLineage:
    """Объявленные шаги. Пустой — согласован только gt обучения (строгое равенство sha1)."""

    steps: tuple[LineageStep, ...] = ()

    def __post_init__(self) -> None:
        targets = [step.target for step in self.steps]
        repeated = sorted({sha1 for sha1 in targets if targets.count(sha1) > 1})
        if repeated:
            raise LineageError(f"gt {repeated[0][:8]} получается двумя шагами — неоднозначно")
        by_target = {step.target: step for step in self.steps}
        for step in self.steps:
            node, seen = step.source, {step.target}
            while node in by_target:
                if node in seen:
                    raise LineageError(f"цепочка шагов через {node[:8]} замкнута в цикл")
                seen.add(node)
                node = by_target[node].source

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GtLineage:
        if data.get("format") != LINEAGE_FORMAT:
            raise LineageError(f"format {data.get('format')!r}, ожидается {LINEAGE_FORMAT!r}")
        raw = data.get("steps")
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            raise LineageError("steps — не список")
        return cls(tuple(_step(item, number) for number, item in enumerate(raw, start=1)))

    @classmethod
    def load(cls, path: Path) -> GtLineage:
        """Шаги из файла; нет файла — пусто (строгая сверка), кривой файл — `LineageError`."""
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LineageError(f"{path.name} не читается: {exc}") from exc
        if not isinstance(data, Mapping):
            raise LineageError(f"{path.name}: ожидается объект JSON")
        try:
            return cls.from_dict(data)
        except LineageError as exc:
            raise LineageError(f"{path.name}: {exc}") from None

    def derive(
        self, trained: str, loaded: str, *, live_cards: bool
    ) -> tuple[LineageStep, ...] | None:
        """Цепочка шагов от gt обучения к загруженному; `()` — тот же gt, `None` — не выводится.

        Шаги идут от загруженного gt назад, к источнику: у каждого gt не больше одного
        объявленного шага, который его даёт. Шаг с `requires_live_cards` без флага цепочку рвёт.
        """
        if loaded == trained:
            return ()
        by_target = {step.target: step for step in self.steps}
        chain: list[LineageStep] = []
        node = loaded
        while node != trained:
            step = by_target.get(node)
            if step is None or (step.requires_live_cards and not live_cards):
                return None
            chain.append(step)
            node = step.source
        return tuple(reversed(chain))


def _step(item: Any, number: int) -> LineageStep:
    where = f"шаг {number}"
    if not isinstance(item, Mapping):
        raise LineageError(f"{where}: ожидается объект")
    name = item.get("name")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise LineageError(f"{where}: name {name!r} — латиница, цифры и «_»")
    hashes = {}
    for key in ("from", "to", "input_sha1"):
        value = item.get(key)
        if not isinstance(value, str) or not _SHA1.fullmatch(value):
            raise LineageError(f"{where} ({name}): {key} — sha1 из 40 строчных hex, а не {value!r}")
        hashes[key] = value
    if hashes["from"] == hashes["to"]:
        raise LineageError(f"{where} ({name}): from и to совпадают")
    live = item.get("requires_live_cards", False)
    if not isinstance(live, bool):
        raise LineageError(f"{where} ({name}): requires_live_cards — true или false")
    return LineageStep(name, hashes["from"], hashes["to"], hashes["input_sha1"], live)


@lru_cache(maxsize=1)
def default_gt_lineage() -> GtLineage:
    """Шаги репозитория (`configs/resolve/gt_lineage.json`): их видят сервис и `scripts/goal.py`."""
    return GtLineage.load(GT_LINEAGE_PATH)
