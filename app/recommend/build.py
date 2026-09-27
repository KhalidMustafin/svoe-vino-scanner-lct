"""Сборка профиля стиля — перенос `build_profile` «Лозы» и приоров сортов.

Источник — `Code/scripts/build_catalog.py:318-386` (ccae5dc) с константами `COLOR_FALLBACK`
(:93), `REGION_SHIFTS` (:108), `OAK_MARKERS`, `FRESH_MARKERS`, `EFFERVESCENCE` (:133) и
функциями `_oak_from_name`, `_age_shift`. Приоры — `Code/data/reference/grape_priors.json`
(20e40f4) побайтно, рядом с модулем: 53 сорта, у каждого пять осей и дескрипторы.

Профиль строится так же, как у «Лозы»:

1. Основа — среднее приоров по сортам вина, которые в приорах есть. Нет ни одного — грубый
   профиль по цвету (`basis="low"`).
2. Поправки на регион, возраст урожая и слова названия («резерв», «барик» — дуб; «молодое»,
   «нуво» — без бочки).
3. Белое и розовое не бывают танинными, что бы ни говорил приор купажа.
4. Сладость — из сахара, пузырьки — из игристости, крепость — из `abv`.

**Одно отличие — крепость.** У Роскачества крепости не было, и «Лоза» ставила 3,0 тихому
вину и 4,6 креплёному. У справочника сканера крепость из slug выгрузки есть у 1 567 из 2 103
позиций, поэтому при известной `abv` ось считается по `abv_to_alcohol_axis` «Лозы» (у неё эта
функция написана, но не вызывалась). Без `abv` — прежнее правило. Золотой тест
(`research/2026-09-24_after/golden_profile.py`) сверяет перенос с осями `derived_*` join CSV
именно без `abv`: так их и считал `strapi_join.py:345` (вне репозитория).

**Не перенесена поправка на экспертный балл** (`_quality_shift`). Балл — данные Роскачества,
в сканере их нет; `derived_*` тоже посчитаны без него (`score_100=None`).

Входы сервиса — поля справочника (`RecoWine`): сорта кодами таксономии сканера (47 из 53 кодов
приоров совпадают; нет `dornfelder`, `gamay`, `gurzufsky_rozovy`, `livadiysky_cherny`,
`regent`, `sauvignon_gris`), цвет и сахар кодами сканера, игристость, регион словами
выгрузки, название, год.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.reading.contracts import Color, SugarClass
from app.recommend.profile import (
    AXES,
    CATALOG,
    CATALOG_AXES,
    GRAPE,
    RussianPGI,
    StyleProfile,
    WineKind,
    abv_to_alcohol_axis,
    clamp,
)

if TYPE_CHECKING:
    from app.recommend.catalog import Alcohol, RecoCatalog, RecoWine

PRIORS_PATH = Path(__file__).with_name("grape_priors.json")

#: Год снимка для возраста урожая — тот же, что у «Лозы» (`_age_shift`, snapshot_year).
SNAPSHOT_YEAR = 2026

#: Оси, которые даёт приор сорта.
PRIOR_AXES: tuple[str, ...] = ("tannin", "body", "acidity", "aroma_intensity", "oak")

#: Сладость по сахару (`SUGAR_TO_SWEETNESS` таксономии «Лозы»).
SUGAR_TO_SWEETNESS: dict[SugarClass, float] = {
    SugarClass.BRUT_NATURE: 0.0,
    SugarClass.EXTRA_BRUT: 0.3,
    SugarClass.BRUT: 0.7,
    SugarClass.DRY: 0.4,
    SugarClass.SEMI_DRY: 1.8,
    SugarClass.SEMI_SWEET: 3.2,
    SugarClass.SWEET: 4.5,
}

#: Профиль по цвету — запасной вариант, когда сорт распознать не удалось.
COLOR_FALLBACK: dict[Color, dict[str, float]] = {
    Color.RED: {"tannin": 3.2, "body": 3.4, "acidity": 3.3, "aroma_intensity": 3.2, "oak": 1.8},
    Color.WHITE: {"tannin": 0.2, "body": 2.7, "acidity": 3.6, "aroma_intensity": 3.0, "oak": 1.0},
    Color.ROSE: {"tannin": 0.6, "body": 2.5, "acidity": 3.7, "aroma_intensity": 3.2, "oak": 0.5},
    Color.ORANGE: {"tannin": 2.6, "body": 3.2, "acidity": 3.5, "aroma_intensity": 3.4, "oak": 1.2},
}

#: Терруарные сдвиги профиля по регионам. Без них профиль определяется одним сортом, и сотни
#: позиций получают одинаковые оси. В жарком Дагестане и на Тамани виноград набирает сахар и
#: теряет кислоту, на Дону и в предгорьях Крыма сезон прохладнее и кислотность сохраняется.
REGION_SHIFTS: dict[RussianPGI, dict[str, float]] = {
    RussianPGI.KUBAN: {"acidity": -0.15, "body": +0.15, "alcohol": +0.10},
    RussianPGI.CRIMEA: {"acidity": -0.05, "body": +0.05},
    RussianPGI.SEVASTOPOL: {"acidity": +0.20, "body": -0.10},
    RussianPGI.DON_VALLEY: {"acidity": +0.35, "body": -0.20, "alcohol": -0.15},
    RussianPGI.DAGESTAN: {"acidity": -0.30, "body": +0.25, "alcohol": +0.20},
    RussianPGI.STAVROPOL: {"acidity": -0.10, "body": +0.10},
    RussianPGI.LOWER_VOLGA: {"acidity": +0.25, "body": -0.15},
    RussianPGI.TEREK_VALLEY: {"acidity": -0.10, "body": +0.15},
}

#: Регион выгрузки → ЗГУ «Лозы» (как при сборке join CSV «Лозы», вне репозитория: Самара и Дальний Восток — «другое»).
REGION_CODES: dict[str, RussianPGI] = {
    "Кубань": RussianPGI.KUBAN,
    "Крым": RussianPGI.CRIMEA,
    "Дагестан": RussianPGI.DAGESTAN,
    "Долина Дона": RussianPGI.DON_VALLEY,
    "Ставрополье": RussianPGI.STAVROPOL,
    "Нижняя Волга": RussianPGI.LOWER_VOLGA,
    "Самара": RussianPGI.OTHER,
    "Северная Осетия — Алания": RussianPGI.TEREK_VALLEY,
    "Дальневосточная зона": RussianPGI.OTHER,
}

#: Слова в названии, прямо указывающие на дуб и выдержку.
OAK_MARKERS: dict[str, float] = {
    "резерв": 1.2, "reserve": 1.2, "reserva": 1.2,
    "выдержанное": 1.0, "выдержанный": 1.0, "выдержка": 1.0,
    "барик": 1.4, "баррик": 1.4, "barrique": 1.4, "дуб": 1.2,
    "коллекционное": 0.8, "фамильное": 0.6, "гран": 0.6, "grand": 0.6,
}  # fmt: skip

#: Слова, означающие обратное: вино сделано на свежесть, без бочки.
FRESH_MARKERS = frozenset({"молодое", "нуво", "primeur", "фреш", "легкое", "лёгкое"})

#: Насыщенность пузырьками по типу производства (у справочника типа нет — только игристость).
EFFERVESCENCE: dict[str, float] = {"Классический": 4.4, "Акратофорный": 3.6}
#: Пузырьки игристого без известного метода и тихого вина.
SPARKLING_EFFERVESCENCE = 4.0

#: Крепость без `abv`: креплёное заведомо тёплое, остальное — середина шкалы.
FORTIFIED_ALCOHOL = 4.6

COLOR_VALUES = frozenset(color.value for color in Color)
SUGAR_VALUES = frozenset(sugar.value for sugar in SugarClass)


# ------------------------------------------------------------------ приоры
@dataclass(frozen=True, slots=True)
class GrapePrior:
    """Приор сорта: пять осей и ароматические дескрипторы."""

    label: str
    axes: Mapping[str, float]
    descriptors: tuple[str, ...]


def load_priors(path: Path = PRIORS_PATH) -> dict[str, GrapePrior]:
    """Приоры сортов: код → оси и дескрипторы. Файл лежит рядом с модулем и едет с кодом."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    priors = raw.get("priors") if isinstance(raw, Mapping) else None
    if not isinstance(priors, Mapping):
        raise TypeError(f"{path}: нет раздела priors (объект код сорта → приор)")
    return {
        str(code): GrapePrior(
            label=str(item.get("label") or code),
            axes={axis: float(value) for axis, value in (item.get("axes") or {}).items()},
            descriptors=tuple(str(d) for d in item.get("descriptors") or []),
        )
        for code, item in priors.items()
    }


# ------------------------------------------------------------------ поправки
def oak_from_name(name: str) -> float | None:
    """Выраженность дуба, объявленная в названии вина; «молодое» и «нуво» — 0,2."""
    lowered = name.lower()
    if any(marker in lowered for marker in FRESH_MARKERS):
        return 0.2
    best = None
    for marker, value in OAK_MARKERS.items():
        if marker in lowered:
            best = value if best is None else max(best, value)
    return best


def age_shift(vintage: int | None, snapshot_year: int = SNAPSHOT_YEAR) -> dict[str, float]:
    """Что делает с вином время: минус кислотность, плюс тело и аромат; растёт до 12 лет."""
    if not vintage:
        return {}
    age = max(0, min(snapshot_year - vintage, 12))
    if age < 3:
        return {}
    factor = (age - 2) / 10.0
    return {
        "acidity": -0.25 * factor,
        "body": +0.20 * factor,
        "aroma_intensity": +0.15 * factor,
    }


# ------------------------------------------------------------------ профиль
def build_profile(
    grapes: Sequence[str],
    color: Color | None,
    sugar: SugarClass | None,
    kind: WineKind,
    priors: Mapping[str, GrapePrior],
    *,
    method: str = "",
    region: RussianPGI = RussianPGI.OTHER,
    name: str = "",
    vintage: int | None = None,
    abv: float | None = None,
) -> StyleProfile:
    """Профиль по фактам вина — `build_profile` «Лозы» с источниками осей.

    `color=None` — как белое, `sugar=None` — как сухое: так «Лоза» собирала `derived_*` для
    карточек без этих полей. У таких осей источник `None`. `abv=None` — крепость по правилу
    «Лозы» (3,0 или 4,6 у креплёного), источник тоже `None`.
    """
    matched = [priors[code] for code in grapes if code in priors]
    if matched:
        axes: dict[str, float] = {}
        for axis in PRIOR_AXES:
            values = [prior.axes[axis] for prior in matched if axis in prior.axes]
            if values:
                axes[axis] = sum(values) / len(values)
        descriptors = tuple(sorted({d for prior in matched for d in prior.descriptors}))
        basis = "high" if len(matched) == len(grapes) else "medium"
    else:
        axes = dict(COLOR_FALLBACK[color or Color.WHITE])
        descriptors = ()
        basis = "low"

    # Терруар и возраст уточняют профиль там, где сорт молчит.
    for shifts in (REGION_SHIFTS.get(region, {}), age_shift(vintage)):
        for axis, delta in shifts.items():
            axes[axis] = axes.get(axis, 3.0) + delta

    declared_oak = oak_from_name(name)
    if declared_oak is not None:
        axes["oak"] = (
            max(axes.get("oak", 0.0), declared_oak) if declared_oak > 0.5 else declared_oak
        )

    # Белое и розовое не бывают танинными независимо от приора купажа.
    if (color or Color.WHITE) in (Color.WHITE, Color.ROSE):
        limit = 0.6 if color == Color.ROSE else 0.3
        axes["tannin"] = min(axes.get("tannin", 0.0), limit)

    # Сдвиги складываются и без ограничения могут вывести ось за границу 0–5.
    for axis in list(axes):
        axes[axis] = clamp(axes[axis])

    if abv is not None:
        alcohol = abv_to_alcohol_axis(abv)
    else:
        alcohol = FORTIFIED_ALCOHOL if kind == WineKind.FORTIFIED else 3.0
    sparkling = kind == WineKind.SPARKLING
    sources: dict[str, str | None] = {
        axis: (GRAPE if basis != "low" else None) for axis in AXES if axis not in CATALOG_AXES
    }
    sources.update(
        sweetness=CATALOG if sugar is not None else None,
        alcohol=CATALOG if abv is not None else None,
        effervescence=CATALOG,
    )
    return StyleProfile(
        sweetness=SUGAR_TO_SWEETNESS.get(sugar or SugarClass.DRY, 0.5),
        acidity=axes.get("acidity", 3.3),
        tannin=axes.get("tannin", 0.0),
        body=axes.get("body", 3.0),
        alcohol=alcohol,
        oak=axes.get("oak", 1.0),
        aroma_intensity=axes.get("aroma_intensity", 3.0),
        effervescence=EFFERVESCENCE.get(
            method.strip(), SPARKLING_EFFERVESCENCE if sparkling else 0.0
        ),
        descriptors=descriptors,
        basis=basis,
        sources=sources,
    )


def profile_of(
    wine: RecoWine, priors: Mapping[str, GrapePrior], *, alcohol: Alcohol | None = None
) -> StyleProfile:
    """Профиль позиции справочника. Крепость — по правилу договора (`alcohol`), если известна.

    Креплёного типа у справочника нет, поэтому без крепости ось всегда 3,0 — и источника нет.
    """
    color = wine.color if wine.color in COLOR_VALUES else None
    sugar = wine.sugar if wine.sugar in SUGAR_VALUES else None
    return build_profile(
        wine.grapes,
        Color(color) if color else None,
        SugarClass(sugar) if sugar else None,
        WineKind.SPARKLING if wine.sparkling else WineKind.STILL,
        priors,
        region=REGION_CODES.get(wine.region, RussianPGI.OTHER),
        name=wine.title,
        vintage=wine.year,
        abv=alcohol.value if alcohol is not None else None,
    )


def profiles_of(
    catalog: RecoCatalog, priors: Mapping[str, GrapePrior] | None = None
) -> dict[str, StyleProfile]:
    """Профили всех позиций справочника: 2 176 вин за десятки миллисекунд."""
    priors = load_priors() if priors is None else priors
    return {
        wine.slug: profile_of(wine, priors, alcohol=catalog.alcohol(wine.slug)) for wine in catalog
    }
