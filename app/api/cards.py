"""Карточки вин для ответа сервиса: что показать пользователю по найденному slug.

Источники — те же файлы, что у стендов:

    gt_tokens.jsonl        название, винодельня, регион, категория, сорта и имя файла фото
                           (обязателен);
    strapi_output0709.csv  описание и оттенок цвета из выгрузки организатора (по желанию:
                           без файла карточка остаётся без описания);
    slug_photo_map.csv     путь к packshot каталога на диске (по желанию).

Только выгрузка организатора (решение 24.09): класс сахара карточки — по правилу выгрузки
(название, затем slug, `app.recommend.catalog.organizer_sugar`), а не класс разметки — тот у
2 013 позиций взят из категории живого портала. «Опубликовано» и категория живого портала в
карточку не идут. Описание — «Описание» выгрузки как есть: переносы строк сохраняются.

Карточка ничего не решает: resolve выбирает slug, а здесь только справочные поля к нему.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.reading.contracts import SugarClass
from app.reading.taxonomy import SUGAR_TERMS
from app.recommend.catalog import organizer_sugar

logger = logging.getLogger(__name__)

#: Колонки выгрузки организатора, которые идут в карточку.
CSV_SLUG = "Slug"
CSV_COLUMNS = {
    "name": "Название вина",
    "category": "Категория",
    "color": "Цвет",
    "region": "Регион",
    "grapes": "Сорт винограда",
    "description": "Описание",
    "winery": "Винодельня",
    "photo_name": "Название фото",
}


@dataclass(frozen=True)
class WineCard:
    """Справочные поля позиции каталога. Пустая строка и `None` — «в каталоге не заполнено»."""

    slug: str
    name: str = ""
    winery: str = ""
    region: str = ""
    grapes: list[str] = field(default_factory=list)
    category: str = ""  # «Белое», «Красное» — колонка «Категория»
    color: str = ""  # оттенок из выгрузки: «Светло-соломенный»
    sugar: str = ""  # «сухое», «брют»
    sugar_class: str | None = None  # код SugarClass по правилу выгрузки: suhoe, brut
    description: str = ""  # «Описание» выгрузки как есть
    photo_name: str = ""
    photo_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sugar_label(value: object) -> str:
    """Код класса сахара → русское слово с этикетки («suhoe» → «сухое»)."""
    try:
        return SUGAR_TERMS[SugarClass(str(value))][0]
    except (KeyError, ValueError):
        return ""


def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _split_grapes(value: str) -> list[str]:
    return [part for part in (_text(p) for p in value.split(",")) if part]


def read_catalog_csv(path: Path) -> dict[str, dict[str, str]]:
    """Выгрузка организатора: строка на slug (в файле каждая позиция повторена, берётся первая)."""
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if CSV_SLUG not in (reader.fieldnames or []):
            raise ValueError(f"{path}: нет колонки {CSV_SLUG!r}")
        for row in reader:
            slug = _text(row.get(CSV_SLUG))
            if slug and slug not in rows:
                rows[slug] = {key: _text(row.get(column)) for key, column in CSV_COLUMNS.items()}
                # Описание — как написал организатор: переносы строк и абзацы остаются.
                rows[slug]["description"] = (row.get(CSV_COLUMNS["description"]) or "").strip()
    return rows


def read_photo_map(path: Path) -> dict[str, str]:
    """`slug_photo_map.csv` (колонки `slug`, `path`) → путь к фото по slug."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            slug, photo = _text(row.get("slug")), (row.get("path") or "").strip()
            if slug and photo:
                out.setdefault(slug, photo)
    return out


def read_records(path: Path) -> list[dict[str, Any]]:
    """Записи `gt_tokens.jsonl`: строка — одна позиция каталога."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: не JSON: {exc}") from exc
    return records


def card_of(
    record: Mapping[str, Any], csv_row: Mapping[str, str] | None, photo_path: str | None
) -> WineCard:
    """Карточка из записи разметки, строки выгрузки и пути к фото."""
    csv_row = csv_row or {}
    fields = record.get("fields") or {}
    grapes = [_text(v) for v in (fields.get("grape") or {}).get("values") or [] if _text(v)]
    if not grapes and csv_row.get("grapes"):
        grapes = _split_grapes(csv_row["grapes"])
    slug = str(record["slug"])
    name = _text(record.get("name")) or csv_row.get("name", "")
    sugar_class = organizer_sugar(name, slug)
    return WineCard(
        slug=slug,
        name=name,
        winery=_text(record.get("winery")) or csv_row.get("winery", ""),
        region=_text(record.get("region")) or csv_row.get("region", ""),
        grapes=grapes,
        category=_text(record.get("category")) or csv_row.get("category", ""),
        color=csv_row.get("color", ""),
        sugar=sugar_label(sugar_class) if sugar_class else "",
        sugar_class=sugar_class,
        description=csv_row.get("description", ""),
        photo_name=_text(record.get("photo_name")) or csv_row.get("photo_name", ""),
        photo_path=photo_path,
    )


class CatalogCards:
    """Карточки всех позиций каталога в памяти: 2 103 записи, несколько мегабайт."""

    def __init__(self, cards: Iterable[WineCard], *, sources: Mapping[str, Any] | None = None):
        self.by_slug: dict[str, WineCard] = {card.slug: card for card in cards}
        self.sources: dict[str, Any] = dict(sources or {})

    def __len__(self) -> int:
        return len(self.by_slug)

    def get(self, slug: str) -> WineCard | None:
        return self.by_slug.get(slug)

    @classmethod
    def build(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        csv_path: Path | None = None,
        photo_map: Path | None = None,
    ) -> CatalogCards:
        """Карточки по записям разметки. Нет выгрузки или карты фото — карточки без них."""
        sources: dict[str, Any] = {"csv": None, "photo_map": None}
        csv_rows: dict[str, dict[str, str]] = {}
        if csv_path is not None:
            try:
                csv_rows = read_catalog_csv(csv_path)
                sources["csv"] = str(csv_path)
            except (OSError, ValueError, csv.Error) as exc:
                logger.warning("Карточки без описаний: выгрузка %s не читается (%s)", csv_path, exc)
        photos: dict[str, str] = {}
        if photo_map is not None:
            try:
                photos = read_photo_map(photo_map)
                sources["photo_map"] = str(photo_map)
            except (OSError, ValueError, csv.Error) as exc:
                logger.warning("Карточки без фото: карта %s не читается (%s)", photo_map, exc)
        cards = [
            card_of(record, csv_rows.get(str(record["slug"])), photos.get(str(record["slug"])))
            for record in records
            if record.get("slug")
        ]
        sources["cards"] = len(cards)
        sources["with_description"] = sum(bool(card.description) for card in cards)
        sources["with_photo_path"] = sum(card.photo_path is not None for card in cards)
        return cls(cards, sources=sources)
