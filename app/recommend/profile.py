"""Профиль стиля вина — перенос `SensoryProfile` «Лозы» (`Code/backend/app/domain/profile.py`).

Восемь осей по шкале 0–5, как у WSET: сладость, кислотность, танины, тело, крепость, дуб,
интенсивность аромата, пузырьки. Профиль нужен «Сомелье у полки» (`shelf.py`): по нему
чип «Какое хочется?» выбирает направление, а подписи «по сорту» отличают догадку от факта.
«Роза ветров» на карточке обязательна (решение 24.09): рисуется у каждого вина выгрузки.

**Источник каждой оси.**

* Сахар, игристость и крепость — «каталог» (`catalog`): это факты карточки. Сладость
  выводится из сахара, пузырьки — из игристости, крепость — из slug выгрузки (правило
  договора, `catalog.alcohol_of`).
* Кислотность, танины, тело, дуб и аромат — «по сорту» (`grape`): приоры сорта «Лозы»
  (`grape_priors.json`) с её поправками на регион, год урожая и слова названия («резерв»,
  «молодое»). Это оценка по сорту, а не описание именно этого вина.
* `None` — данных нет. Сахар или крепость неизвестны, либо ни один сорт вина не найден в
  приорах: тогда оси «по сорту» — типичные для цвета (запасной профиль «Лозы», `basis="low"`),
  и выдавать их за свойство вина нельзя. Значение оси при этом то же, что дала бы «Лоза».
* «По стилю» (`style`) — только для показа («роза ветров» и шкалы сомелье, `StylePriors`):
  пять осей «по сорту», которых у вина нет, берутся медианой тех же осей у вин того же стиля
  выгрузки (цвет × игристость × сахар), чей профиль опирается на сорт. Сорт не в приорах у 31
  позиции из 2 103 с неизвестными сахаром и крепостью — без этой оценки у них оставалось меньше
  трёх осей, и «розы ветров» на карточке не было, а она обязательна (решение 24.09). Профиль
  вина (`build.profile_of`) она не меняет: подбор, пары и подача по-прежнему видят `None`.
  Сахар и крепость по стилю не оцениваются: это факты карточки, и «сахар не указан» остаётся
  правдой.

**Что поменялось при переносе.**

* pydantic-модель стала замороженным dataclass: профили строятся на весь справочник при
  старте, а пакет остаётся чистым Python.
* Добавлены источники осей и `basis` (у «Лозы» это второй результат `build_profile`).
* Не перенесены `distance`, `similarity`, `differences` и `average`. Похожие вина считает
  `facts.py` по фактам, а не по осям (план, §0 п. 2): у «Лозы» объяснение по осям у 96 %
  вин вырождалось в «расхождений почти нет».
* Не перенесён словарь подписей ароматов: дескрипторы остаются кодами приоров, на странице
  их не показывают.
* Из `enums.py` «Лозы» перенесены `WineKind` и `RussianPGI`. Цвет и сахар — коды сканера
  (`Color`, `SugarClass`): у «Лозы» те же значения, только по-английски (`dry` = `suhoe`).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

#: Оси профиля в фиксированном порядке — порядок задаёт раскладку вектора.
AXES: tuple[str, ...] = (
    "sweetness",
    "acidity",
    "tannin",
    "body",
    "alcohol",
    "oak",
    "aroma_intensity",
    "effervescence",
)
SCALE_MAX = 5.0

#: Источники осей: факт карточки каталога, оценка по сорту и — только для показа — по стилю.
CATALOG = "catalog"
GRAPE = "grape"
STYLE = "style"
#: Оси, которые подтверждает карточка; остальные — «по сорту».
CATALOG_AXES = frozenset({"sweetness", "alcohol", "effervescence"})
#: Оси, которые оценка по стилю закрывает, — те же пять, что даёт приор сорта.
STYLE_AXES: tuple[str, ...] = tuple(axis for axis in AXES if axis not in CATALOG_AXES)
#: Сколько вин с осями «по сорту» нужно группе стиля, чтобы её медиане верить; меньше — группа
#: шире: цвет × игристость, затем игристость, затем весь справочник.
STYLE_MIN_WINES = 5

#: Подписи осей для объяснений: «кислотность выше».
AXIS_LABELS: dict[str, str] = {
    "sweetness": "сладость",
    "acidity": "кислотность",
    "tannin": "танины",
    "body": "тело",
    "alcohol": "крепость",
    "oak": "дуб",
    "aroma_intensity": "аромат",
    "effervescence": "пузырьки",
}


class WineKind(StrEnum):
    """Тип вина по технологии, как его различает потребитель (`enums.py` «Лозы»)."""

    STILL = "still"
    SPARKLING = "sparkling"
    FORTIFIED = "fortified"
    DESSERT = "dessert"


class RussianPGI(StrEnum):
    """Защищённые географические указания РФ — верхний уровень (`enums.py` «Лозы»)."""

    KUBAN = "kuban"
    CRIMEA = "crimea"
    DON_VALLEY = "don_valley"
    DAGESTAN = "dagestan"
    STAVROPOL = "stavropol"
    TEREK_VALLEY = "terek_valley"
    LOWER_VOLGA = "lower_volga"
    SEVASTOPOL = "sevastopol"
    OTHER = "other"


def clamp(value: float) -> float:
    """Ось в пределах шкалы 0–5."""
    return max(0.0, min(SCALE_MAX, value))


def abv_to_alcohol_axis(abv: float | None) -> float:
    """Крепость в % об. → шкала 0–5 («Лоза»): 9 % — лёгкое (1), 12,5 % — среднее (3), 15 %+ — 5."""
    if abv is None:
        return 3.0
    return clamp((abv - 8.0) / 1.4)


def _frozen(mapping: Mapping[str, str | None]) -> Mapping[str, str | None]:
    return MappingProxyType(dict(mapping))


@dataclass(frozen=True, slots=True)
class StyleProfile:
    """Профиль стиля: оси 0–5, их источники, дескрипторы приоров и основа профиля.

    Значения по умолчанию — как у `SensoryProfile` «Лозы». `basis` — насколько профиль
    опирается на сорт: `high` — все сорта вина есть в приорах, `medium` — часть, `low` —
    ни одного (оси «по сорту» взяты по цвету).
    """

    sweetness: float = 0.0
    acidity: float = 3.0
    tannin: float = 0.0
    body: float = 3.0
    alcohol: float = 3.0
    oak: float = 0.0
    aroma_intensity: float = 3.0
    effervescence: float = 0.0
    descriptors: tuple[str, ...] = ()
    basis: str = "low"
    sources: Mapping[str, str | None] = field(default_factory=lambda: _frozen({}))

    def __post_init__(self) -> None:
        for axis in AXES:
            value = getattr(self, axis)
            if not 0.0 <= value <= SCALE_MAX:
                raise ValueError(f"StyleProfile: ось {axis} = {value} вне шкалы 0–{SCALE_MAX}")
        object.__setattr__(self, "sources", _frozen(self.sources))

    def axis(self, name: str) -> float:
        return float(getattr(self, name))

    def source(self, name: str) -> str | None:
        """Источник оси: `catalog`, `grape`, `style` (только у профиля для показа) или `None`."""
        return self.sources.get(name)

    def known(self, name: str) -> bool:
        return self.source(name) is not None

    def to_vector(self) -> list[float]:
        """Оси в порядке `AXES`."""
        return [self.axis(axis) for axis in AXES]

    def as_dict(self) -> dict[str, object]:
        """Для ответа или отладки: оси с двумя знаками, источники и основа."""
        return {
            "axes": {axis: round(self.axis(axis), 2) for axis in AXES},
            "sources": {axis: self.source(axis) for axis in AXES},
            "basis": self.basis,
        }


# ------------------------------------------------------------------ оценка по стилю
#: Стиль вина: цвет выгрузки, игристость и код сахара (`None` — не указан).
StyleKey = tuple[str | None, bool, str | None]
#: Белое и розовое не бывают танинными (`build.build_profile`): предел оси и для оценки по стилю.
WHITE_TANNIN_MAX: dict[str, float] = {"Белое": 0.3, "Розовое": 0.6}


@dataclass(frozen=True)
class StylePriors:
    """Медианы осей «по сорту» у вин одного стиля — оценка для вин, чьего сорта нет в приорах.

    Считаются по самой выгрузке организатора: у каждой группы стиля — медиана каждой из пяти
    осей по винам, у которых эта ось «по сорту» (`grape`). Группы — от узкой к широкой: цвет ×
    игристость × сахар (если сахар известен), цвет × игристость, игристость, весь справочник.
    Берётся первая, где набралось `STYLE_MIN_WINES` вин, но цвет сильнее счёта: пока есть группа
    того же цвета, красные в оценку белого не попадают (в выгрузке у каждого цвета групп хватает,
    малы они только в тестах). Белое и розовое к тому же не бывают танинными — тот же предел, что
    у `build.build_profile`. Значения округлены до сотых: оценка детерминирована и не зависит от
    порядка вин.
    """

    groups: Mapping[tuple[object, ...], Mapping[str, float]]
    counts: Mapping[tuple[object, ...], int]
    min_wines: int = STYLE_MIN_WINES

    @staticmethod
    def _levels(key: StyleKey) -> tuple[tuple[object, ...], ...]:
        color, sparkling, sugar = key
        levels: list[tuple[object, ...]] = []
        if sugar is not None:
            levels.append(("style", color, sparkling, sugar))
        levels.extend((("color", color, sparkling), ("sparkling", sparkling), ("all",)))
        return tuple(levels)

    @classmethod
    def build(
        cls, wines: Iterable[tuple[StyleKey, StyleProfile]], *, min_wines: int = STYLE_MIN_WINES
    ) -> StylePriors:
        """Медианы по винам справочника: `(стиль, профиль)` на каждое вино."""
        values: dict[tuple[object, ...], dict[str, list[float]]] = {}
        counts: dict[tuple[object, ...], int] = {}
        for key, profile in wines:
            grape_axes = [axis for axis in STYLE_AXES if profile.source(axis) == GRAPE]
            if not grape_axes:
                continue
            for level in cls._levels(key):
                counts[level] = counts.get(level, 0) + 1
                bucket = values.setdefault(level, {})
                for axis in grape_axes:
                    bucket.setdefault(axis, []).append(profile.axis(axis))
        groups = {
            level: MappingProxyType(
                {axis: round(statistics.median(items), 2) for axis, items in sorted(axes.items())}
            )
            for level, axes in values.items()
        }
        return cls(MappingProxyType(groups), MappingProxyType(counts), min_wines)

    def prior(self, key: StyleKey) -> Mapping[str, float]:
        """Оси стиля: первая группа, где хватает вин; пустой справочник — пусто."""
        levels = self._levels(key)
        colored = [level for level in levels if level[0] in ("style", "color")] if key[0] else []
        wide = [level for level in levels if level not in colored]
        for tier in (colored, wide):
            for level in tier:
                if self.counts.get(level, 0) >= self.min_wines:
                    return self.groups[level]
            for level in tier:  # малый справочник (тесты): хоть какая-то группа этого яруса
                if level in self.groups:
                    return self.groups[level]
        return MappingProxyType({})

    def fill(self, profile: StyleProfile, key: StyleKey) -> StyleProfile:
        """Профиль для показа: оси «по сорту», которых нет, — по стилю (`style`).

        Оси с источником (`catalog`, `grape`) не меняются ни значением, ни источником; сахар и
        крепость без источника так и остаются без него.
        """
        missing = [axis for axis in STYLE_AXES if profile.source(axis) is None]
        if not missing:
            return profile
        prior = self.prior(key)
        filled = {axis: clamp(prior[axis]) for axis in missing if axis in prior}
        if not filled:
            return profile
        limit = WHITE_TANNIN_MAX.get(key[0] or "")
        if limit is not None and "tannin" in filled:
            filled["tannin"] = min(filled["tannin"], limit)
        sources = dict(profile.sources)
        sources.update(dict.fromkeys(filled, STYLE))
        return replace(profile, **filled, sources=sources)
