"""Признаки карточек каталога: что известно о вине до того, как пришёл кадр.

Таблица собирается из `gt_tokens.jsonl` — того же файла, из которого собран словарь
этикетки. Поэтому канонические формы у карточки и у чтения общие: винодельня — как в
каталоге, сорт — код таксономии, сахар — класс, серия — «резерв» или римский номер.
Сравниваются не сырые строки, а ключи: в каталоге «Reserve», на этикетке «Riserva» —
один ключ «резерв».

Таблица ничего не решает и никуда не ходит: словари в памяти плюс обратные индексы —
винодельня → её позиции, серия двойников → её члены. Решение принимает `rerank`.

Серия — `cluster_B` из разметки: почти одинаковые этикетки, которые CV не различает
(868 позиций из 2 103 лежат в 316 группах по 2–12 штук). Позиция без группы — серия из
себя одной, чтобы у каждого slug была серия и код не расходился на два случая.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.reading.contracts import Color, SugarClass
from app.reading.lexicon.build import GENERIC_WORDS, entry_norm, record_color
from app.reading.taxonomy import LABEL_GENERIC, LABEL_TERMS, canonical_grape, color_of, sugar_class
from app.reading.text.normalize import roman_value

#: Флаги разметки, при которых название позиции — не название, а описание: «Пино Нуар»,
#: «Белое сухое», одно слово. По таким названиям нельзя судить, что этикетка «не та».
CONDITIONAL_NAME_FLAGS = frozenset({"name_grape_only", "name_generic_only", "single_word_name"})

#: Слова, которые не отличают позицию внутри винодельни: тип продукта, предлоги, регионы.
_STOP_WORDS = GENERIC_WORDS | LABEL_GENERIC


# ------------------------------------------------------------------ ключи сравнения
def norm_key(text: object) -> str:
    """Ключ строки — та же норма, что у записей словаря каталога и у токенов этикетки."""
    return entry_norm(str(text)) if text is not None else ""


def grape_key(value: object) -> str:
    """Ключ сорта: код таксономии («Пино Нуар», «pinot noir», `pinot_noir` → `pinot_noir`).

    Сорта вне таксономии (в каталоге их 124) остаются собой в нормализованном виде: код
    каталога и есть каноническая форма словаря, по ней чтение и карточка и сходятся.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    return canonical_grape(text) or norm_key(text)


def sugar_key(value: object) -> SugarClass | None:
    """Класс сахара по коду каталога (`suhoe`) или написанию с этикетки («Brut»)."""
    text = str(value or "").strip()
    return sugar_class(text) if text else None


def color_key(value: object) -> Color | None:
    """Цвет по значению каталога («Красное») или написанию («rosé»)."""
    text = str(value or "").strip()
    return color_of(text) if text else None


#: Пары разных цветов, которые не спорят. Оранжевые вина в РФ маркируются «белое»: категории
#: «оранжевое» в законе нет. На этикетке оранжевого вина — «белое», у части таких карточек в
#: каталоге «Белое», а на этикетке слово «оранж». Такая пара — не спор, но и не совпадение:
#: отличить оранжевое вино от белого той же винодельни она не помогает (Э4).
COMPATIBLE_COLORS: frozenset[frozenset[Color]] = frozenset({frozenset({Color.WHITE, Color.ORANGE})})


def colors_compatible(first: object, second: object) -> bool:
    """Два разных цвета, которые не спорят: «Белое» и «Оранжевое». Одинаковые — не пара."""
    a, b = color_key(first), color_key(second)
    return a is not None and b is not None and a != b and frozenset({a, b}) in COMPATIBLE_COLORS


def serial_key(value: object) -> str:
    """Ключ серии: «Grand Reserve» → «гран резерв», «XXIV» → «XXIV», доля → «30/70»."""
    text = norm_key(value)
    if not text:
        return ""
    term = LABEL_TERMS.lookup(text)
    if term is not None and term.kind == "serial" and term.value is not None:
        return term.value
    if text.isalpha() and roman_value(text) is not None:
        return text.upper()
    return text


def year_key(value: object) -> int | None:
    """Год урожая как целое; мусор и пустое — None."""
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


def abv_key(value: object) -> float | None:
    """Крепость как число; мусор и пустое — None."""
    try:
        return float(str(value).strip().rstrip("%").replace(",", "."))
    except (TypeError, ValueError):
        return None


def name_words(text: str) -> frozenset[str]:
    """Значимые слова названия: без типа продукта, предлогов, регионов и общих слов."""
    words = norm_key(text).split()
    return frozenset(word for word in words if len(word) > 2 and word not in _STOP_WORDS)


# ------------------------------------------------------------------ карточка
@dataclass(frozen=True, slots=True)
class WineAttrs:
    """Признаки одной позиции каталога в ключах сравнения.

    Пустое множество значит «в каталоге не заполнено» — это не «на этикетке нет». Штрафовать
    карточку за то, чего каталог про неё не знает, нельзя: год есть только у 116 позиций.
    """

    slug: str
    winery: str  # как в каталоге, для отчёта
    winery_keys: frozenset[str]  # норма винодельни, её ключевые токены, варианты, бренды
    name: str
    name_keys: frozenset[str]
    cuvee: frozenset[str]
    grapes: frozenset[str]
    sugar: SugarClass | None
    year: int | None
    abv: tuple[float, ...]
    serial: frozenset[str]
    color: Color | None
    cluster: str | None  # cluster_B: группа двойников уровня B
    visual_group: str | None  # более тесная группа: та же этикетка, другой объём или год
    mates: tuple[str, ...]
    flags: frozenset[str]  # noise_flags разметки

    @property
    def series(self) -> str:
        """Серия для агрегации CV: группа двойников, иначе сам slug."""
        return f"cluster:{self.cluster}" if self.cluster else f"slug:{self.slug}"

    @property
    def conditional_name(self) -> bool:
        """Название — описание, а не имя: «Пино Нуар», «Белое сухое», одно слово."""
        return bool(self.flags & CONDITIONAL_NAME_FLAGS)

    @property
    def text_keys(self) -> frozenset[str]:
        """Всё, чем позиция отличается словами: название, кюве, сорта, серия."""
        return self.name_keys | self.cuvee | self.grapes | frozenset(self.serial)


def _field(record: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (record.get("fields") or {}).get(name) or {}


def _group_key(value: object) -> str | None:
    """`visual_group` в разметке — пара [B, подгруппа]; ключом делаем строку «111/0»."""
    if value is None:
        return None
    if isinstance(value, list | tuple):
        return "/".join(str(part) for part in value)
    return str(value)


def wine_attrs(record: Mapping[str, Any]) -> WineAttrs:
    """Одна запись `gt_tokens.jsonl` → признаки карточки."""
    slug = str(record["slug"])
    winery = str(record.get("winery") or "").strip()
    wf = _field(record, "winery")
    keys = {norm_key(winery), norm_key(" ".join(wf.get("key_tokens") or []))}
    for value in (*(wf.get("key_tokens") or []), *(wf.get("variants") or [])):
        keys.add(norm_key(value))
    for brand in wf.get("brands") or []:
        keys.add(norm_key(brand))
    grapes = _field(record, "grape")
    codes = {grape_key(code.removeprefix("csv:")) for code in grapes.get("codes") or []}
    codes |= {grape_key(value) for value in grapes.get("values") or []}
    serial = _field(record, "serial")
    name = str(record.get("name") or "")
    return WineAttrs(
        slug=slug,
        winery=winery,
        winery_keys=frozenset(key for key in keys if key),
        name=name,
        name_keys=name_words(name),
        cuvee=frozenset(
            key for value in _field(record, "cuvee").get("tokens") or [] if (key := norm_key(value))
        ),
        grapes=frozenset(code for code in codes if code),
        sugar=sugar_key(_field(record, "sugar").get("class")),
        year=year_key(_field(record, "year").get("value")),
        abv=tuple(
            value for raw in _field(record, "abv").get("value") or [] if (value := abv_key(raw))
        ),
        serial=frozenset(
            key
            for value in (*(serial.get("tokens") or []), *(serial.get("keywords") or []))
            if (key := serial_key(value))
        ),
        color=color_key(record_color(record)),
        cluster=None if record.get("cluster_B") is None else str(record["cluster_B"]),
        visual_group=_group_key(record.get("visual_group")),
        mates=tuple(str(mate) for mate in record.get("visual_mates") or []),
        flags=frozenset(str(flag) for flag in record.get("noise_flags") or []),
    )


# ------------------------------------------------------------------ таблица каталога
class CatalogAttrs:
    """Признаки всех позиций каталога и обратные индексы к ним."""

    def __init__(
        self, wines: Iterable[WineAttrs], *, meta: Mapping[str, Any] | None = None
    ) -> None:
        self.by_slug: dict[str, WineAttrs] = {}
        for wine in wines:
            if wine.slug in self.by_slug:
                raise ValueError(f"CatalogAttrs: повтор slug {wine.slug}")
            self.by_slug[wine.slug] = wine
        self.meta: dict[str, Any] = dict(meta or {})
        by_winery: dict[str, list[str]] = {}
        by_series: dict[str, list[str]] = {}
        for wine in self.by_slug.values():
            for key in wine.winery_keys:
                by_winery.setdefault(key, []).append(wine.slug)
            by_series.setdefault(wine.series, []).append(wine.slug)
        self._by_winery = {key: tuple(slugs) for key, slugs in by_winery.items()}
        self._by_series = {key: tuple(slugs) for key, slugs in by_series.items()}

    def __len__(self) -> int:
        return len(self.by_slug)

    def __contains__(self, slug: object) -> bool:
        return isinstance(slug, str) and slug in self.by_slug

    def __iter__(self) -> Iterator[WineAttrs]:
        return iter(self.by_slug.values())

    def get(self, slug: str) -> WineAttrs | None:
        return self.by_slug.get(slug)

    def series_of(self, slug: str) -> str:
        """Серия slug; неизвестный каталогу slug — серия из себя одной."""
        wine = self.by_slug.get(slug)
        return wine.series if wine else f"slug:{slug}"

    def members(self, series: str) -> tuple[str, ...]:
        """Члены серии в порядке каталога; серия неизвестна — пусто."""
        return self._by_series.get(series, ())

    def slugs_of_winery(self, value: object) -> tuple[str, ...]:
        """Позиции винодельни по любому её написанию; незнакомая винодельня — пусто.

        Одна норма у нескольких виноделен («шато») даёт позиции их всех — ровно так же,
        как словарь этикетки: спорное слово не указывает на одну винодельню.
        """
        return self._by_winery.get(norm_key(value), ())

    # ------------------------------------------------------------------ загрузка
    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]], *, meta: Mapping[str, Any] | None = None
    ) -> CatalogAttrs:
        return cls((wine_attrs(r) for r in records if r.get("slug")), meta=meta)

    @classmethod
    def load(cls, path: Path | str) -> CatalogAttrs:
        """Таблица из `gt_tokens.jsonl`. Сети и моделей не нужно, только файл разметки."""
        path = Path(path)
        records = []
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: не JSON: {exc}") from exc
        return cls.from_records(records, meta={"source": path.name, "records": len(records)})

    def stats(self) -> dict[str, Any]:
        """Сводка для отчёта прогона: сколько позиций, серий и заполненных признаков."""
        clustered = [wine for wine in self if wine.cluster]
        sizes = [len(members) for members in self._by_series.values()]
        return {
            "slugs": len(self),
            "wineries": len({wine.winery for wine in self if wine.winery}),
            "series": len(self._by_series),
            "clustered_slugs": len(clustered),
            "largest_series": max(sizes, default=0),
            "with_year": sum(wine.year is not None for wine in self),
            "with_serial": sum(bool(wine.serial) for wine in self),
            "with_cuvee": sum(bool(wine.cuvee) for wine in self),
            "with_grape": sum(bool(wine.grapes) for wine in self),
            "conditional_names": sum(wine.conditional_name for wine in self),
        }


def default_attrs_path(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).data_dir / "gt" / "gt_tokens.jsonl"


def load_attrs(path: Path | str | None = None) -> CatalogAttrs:
    """Таблица признаков каталога; путь по умолчанию — `data/gt/gt_tokens.jsonl`."""
    return CatalogAttrs.load(path if path is not None else default_attrs_path())


def read_keys(values: Iterable[object], key: Callable[[object], str]) -> frozenset[str]:
    """Прочитанные значения одного поля в ключах сравнения; пустые отбрасываются."""
    return frozenset(k for value in values if (k := key(value)))
