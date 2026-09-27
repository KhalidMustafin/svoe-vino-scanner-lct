"""Похожие вина из других виноделен, объяснённые фактами каталога.

Почему не «Лоза». Её `find_similar` ищет по осям профиля, выведенным из сорта, и у 96 %
вин объяснение вырождается в «расхождений почти нет» (2 019 из 2 103, зонд инвентаризации).
Здесь сравниваются только факты карточки: цвет, игристость, сахар, сорта, крепость, регион.
Объяснение тоже собирается из них, поэтому оно проверяемо на самой карточке.

Подбор (`similar`):

* отбрасываются вина той же винодельни, той же группы `wine_id` (другой объём или год того
  же вина), само вино и всё, что не входит в пул (неканонические и без фото выгрузки);
* остаются вина того же цвета и той же игристости, сахар — в пределах одной ступени;
* ключ сортировки: мера Жаккара по сортам → тот же сахар → разница крепости → регион → slug;
* в выдаче одна винодельня — одно место: «похожие из других виноделен» не должны оказаться
  тремя винами одного хозяйства (в заглушке договора так и вышло — два вина Гай-Кодзора).

Обычная сортировка (`plain`) — ст. 10.2-2 149-ФЗ, консервативно: та же категория (цвет,
игристость, сахар) по названию, без своей винодельни и без объяснений.

Процентов похожести нет нигде, крепость пишется градусами: «Крепость 13,5° против 14°».
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from app.reading.taxonomy import grape_label
from app.recommend.catalog import (
    SPARKLING_SUGARS,
    SUGAR_STEPS,
    SUGAR_WORDS,
    Alcohol,
    RecoCatalog,
    RecoWine,
    degrees_label,
    name_key,
)

#: Плашка блоков рекомендаций (ст. 10.2-2 149-ФЗ) — при `order=reco`.
NOTICE = "Применяются рекомендательные технологии"
ORDERS = ("reco", "plain")
DEFAULT_LIMIT = 3
MAX_LIMIT = 12
#: Сколько вин винодельни показывает «Не тупик».
SAME_WINERY_LIMIT = 6
#: Разница крепости, если у одного из вин она неизвестна: хуже любой известной близкой.
UNKNOWN_ABV_GAP = 3.0
#: Крепость «та же», если разница меньше этого.
SAME_ABV = 0.05
#: Отличие фактом по крепости начинается с полградуса — как в зонде плана.
ABV_DIFFERENCE = 0.5

#: Цвет во множественном числе родительного падежа: «Оранжевых игристых в каталоге нет».
_COLOR_GEN_PL = {
    "Белое": "белых",
    "Красное": "красных",
    "Розовое": "розовых",
    "Оранжевое": "оранжевых",
}
#: Числительные в родительном падеже: «Нашлось меньше трёх».
_GEN = {
    1: "одного",
    2: "двух",
    3: "трёх",
    4: "четырёх",
    5: "пяти",
    6: "шести",
    7: "семи",
    8: "восьми",
    9: "девяти",
    10: "десяти",
    11: "одиннадцати",
    12: "двенадцати",
}


@dataclass(frozen=True, slots=True)
class Note:
    """Честная фраза ответа: `code` — для тестов и страницы, `text` — показать как есть."""

    code: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "text": self.text}


def fewer_note(found: int, limit: int) -> Note:
    if found == 0:
        return Note("fewer_than_three", "Похожих в каталоге не нашлось")
    return Note("fewer_than_three", f"Нашлось меньше {_GEN.get(limit, str(limit))}")


# ------------------------------------------------------------------ что известно о вине
@dataclass(frozen=True, slots=True)
class Facts:
    """Факты вина, к которому ищутся похожие: из карточки каталога или с этикетки.

    `None` и пустой кортеж — «не известно», а не «нет»: неизвестный признак не фильтрует
    и не объясняет. `slug`, `wine_id` и `wineries` — что исключить из выдачи.
    """

    color: str | None = None
    sparkling: bool | None = None
    sugar: str | None = None
    grapes: tuple[str, ...] = ()
    alcohol: Alcohol = field(default_factory=Alcohol)
    region: str | None = None
    slug: str | None = None
    wine_id: str | None = None
    wineries: frozenset[str] = frozenset()

    @classmethod
    def of_wine(cls, wine: RecoWine, alcohol: Alcohol | None = None) -> Facts:
        return cls(
            color=wine.color,
            sparkling=wine.sparkling,
            sugar=wine.sugar,
            grapes=wine.grapes,
            alcohol=alcohol if alcohol is not None else wine.alcohol,
            region=wine.region or None,
            slug=wine.slug,
            wine_id=wine.wine_id,
            wineries=frozenset({wine.winery_norm}),
        )

    @property
    def has_style(self) -> bool:
        """Есть ли по чему подбирать: цвет, сахар, сорт или признак игристого."""
        return bool(self.color or self.sugar or self.grapes or self.sparkling)

    @property
    def style_label(self) -> str:
        """Категория фактов словами: «Красное сухое», «Оранжевое брют»."""
        return " ".join(p for p in (self.color or "", SUGAR_WORDS.get(self.sugar or "", "")) if p)


# ------------------------------------------------------------------ сравнение
def sugar_close(a: str | None, b: str | None) -> bool:
    """Сахар в пределах одной ступени; неизвестный с любым — близок."""
    if a is None or b is None or a not in SUGAR_STEPS or b not in SUGAR_STEPS:
        return True
    return abs(SUGAR_STEPS[a] - SUGAR_STEPS[b]) <= 1


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def abv_gap(a: Alcohol, b: Alcohol) -> float:
    if a.value is None or b.value is None:
        return UNKNOWN_ABV_GAP
    return abs(a.value - b.value)


def excluded(wine: RecoWine, facts: Facts) -> bool:
    """Само вино, его группа `wine_id` и его винодельня в похожие не идут."""
    return (
        wine.slug == facts.slug
        or (facts.wine_id is not None and wine.wine_id == facts.wine_id)
        or wine.winery_norm in facts.wineries
    )


def differs(facts: Facts, wine: RecoWine, alcohol: Alcohol) -> bool:
    """Отличается ли вино хоть одним фактом — те же признаки, что в зонде плана, и цвет."""
    if facts.region and wine.region and wine.region != facts.region:
        return True
    if (
        facts.alcohol.value is not None
        and alcohol.value is not None
        and abs(facts.alcohol.value - alcohol.value) >= ABV_DIFFERENCE
    ):
        return True
    if facts.grapes and set(wine.grapes) != set(facts.grapes):
        return True
    if facts.sugar and wine.sugar and wine.sugar != facts.sugar:
        return True
    return bool(facts.color and wine.color and wine.color != facts.color)


def _labels(codes: Sequence[str]) -> str:
    """«Сира», «Сира и Мурведр», «Каберне Совиньон, Мерло и Саперави»."""
    names = [grape_label(code) for code in codes]
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} и {names[-1]}"


def explain(
    facts: Facts,
    wine: RecoWine,
    alcohol: Alcohol,
    *,
    region: bool = True,
    strength: bool = True,
) -> list[str]:
    """Объяснение фактами: сорта, сахар и игристость, цвет, регион, крепость — по порядку.

    Только то, что можно проверить по двум карточкам: «Тот же сорт — Сира», «Полусухое, а не
    сухое», «Крым, а не Кубань», «Крепость 13,5° против 14°». Без процентов и эпитетов.
    """
    out: list[str] = []
    if facts.grapes and wine.grapes:
        mine = set(facts.grapes)
        shared = [code for code in wine.grapes if code in mine][:3]
        if shared and set(wine.grapes) == mine:
            out.append(
                ("Тот же сорт — " if len(shared) == 1 else "Те же сорта — ") + _labels(shared)
            )
        elif shared:
            out.append(
                ("Общий сорт — " if len(shared) == 1 else "Общие сорта — ") + _labels(shared)
            )
        else:
            head = "Сорт" if len(wine.grapes) == 1 else "Сорта"
            out.append(f"{head} — {_labels(wine.grapes[:2])}, а не {_labels(facts.grapes[:2])}")
    if facts.sugar and wine.sugar:
        theirs, ours = SUGAR_WORDS[wine.sugar], SUGAR_WORDS[facts.sugar]
        if wine.sugar == facts.sugar:
            out.append(f"Тоже игристое {theirs}" if wine.sparkling else f"Тоже {theirs}")
        elif wine.sparkling:
            out.append(f"Игристое {theirs}, а не {ours}")
        else:
            out.append(f"{theirs.capitalize()}, а не {ours}")
    elif facts.sparkling and wine.sparkling:
        out.append("Тоже игристое")
    if facts.color and wine.color and wine.color != facts.color:
        out.append(f"{wine.color}, а не {facts.color.lower()}")
    if region and facts.region and wine.region:
        same = wine.region == facts.region
        out.append(f"Тоже {wine.region}" if same else f"{wine.region}, а не {facts.region}")
    if strength and facts.alcohol.value is not None and alcohol.value is not None:
        theirs_deg, ours_deg = degrees_label(alcohol), degrees_label(facts.alcohol)
        if abv_gap(facts.alcohol, alcohol) < SAME_ABV and alcohol.max == facts.alcohol.max:
            out.append(f"Крепость та же — {theirs_deg}")
        else:
            out.append(f"Крепость {theirs_deg} против {ours_deg}")
    return out


# ------------------------------------------------------------------ подбор
@dataclass(frozen=True, slots=True)
class Pick:
    """Вино выдачи и почему оно здесь."""

    wine: RecoWine
    reasons: tuple[str, ...] = ()
    differs: bool = False


@dataclass(frozen=True, slots=True)
class Selection:
    """Итог подбора: вина по порядку и честные фразы."""

    picks: tuple[Pick, ...]
    notes: tuple[Note, ...] = ()

    @property
    def wines(self) -> list[RecoWine]:
        return [pick.wine for pick in self.picks]


def _candidates(catalog: RecoCatalog, facts: Facts, *, color: bool) -> list[RecoWine]:
    """Пул без исключённых: цвет (если `color`), игристость и сахар ±1 ступень."""
    return [
        wine
        for wine in catalog.style_pool(facts.color if color else None, facts.sparkling)
        if not excluded(wine, facts) and sugar_close(facts.sugar, wine.sugar)
    ]


def _rank(
    catalog: RecoCatalog, facts: Facts, wines: Iterable[RecoWine], *, relaxed_color: bool
) -> list[RecoWine]:
    """Ключ плана: Жаккар по сортам → тот же сахар → разница крепости → регион → slug.

    Ослабленный цвет ставит вина нужного цвета первыми. Игристость не прочитана — первыми
    идут тихие: этикетка игристого почти всегда говорит «брют» или «игристое».
    """
    grapes = set(facts.grapes)

    def key(wine: RecoWine) -> tuple:
        return (
            relaxed_color and wine.color != facts.color,
            facts.sparkling is None and wine.sparkling,
            -jaccard(grapes, wine.grapes) if grapes else 0.0,
            facts.sugar is not None and wine.sugar != facts.sugar,
            abv_gap(facts.alcohol, catalog.alcohol(wine.slug)),
            facts.region is not None and wine.region != facts.region,
            wine.slug,
        )

    return sorted(wines, key=key)


def one_per_winery(ranked: Sequence[RecoWine], limit: int) -> list[RecoWine]:
    """Первые `limit` по порядку, одна винодельня — одно место; не хватило — добор по порядку."""
    chosen: list[RecoWine] = []
    spare: list[RecoWine] = []
    seen: set[str] = set()
    for wine in ranked:
        if len(chosen) >= limit:
            break
        if wine.winery_norm in seen:
            spare.append(wine)
            continue
        chosen.append(wine)
        seen.add(wine.winery_norm)
    if len(chosen) < limit:
        chosen.extend(spare[: limit - len(chosen)])
    return chosen


def with_contrast(
    catalog: RecoCatalog, facts: Facts, ranked: Sequence[RecoWine], chosen: Sequence[RecoWine]
) -> list[RecoWine]:
    """Приоритет отличия фактом: тройка «всё то же самое» меняет последнее место.

    Если ни одно из выбранных вин не отличается от исходного ни регионом, ни крепостью, ни
    сортами, ни сахаром, последнее место отдаётся первому по порядку вину другой винодельни,
    которое отличается. Так в тройке есть и «самое близкое», и «близкое, но из Крыма, а не с
    Кубани» — объяснение перестаёт быть тремя одинаковыми строками. Нет такого вина — выдача
    прежняя.
    """
    chosen = list(chosen)
    if len(chosen) < 2 or any(differs(facts, wine, catalog.alcohol(wine.slug)) for wine in chosen):
        return chosen
    taken = {wine.slug for wine in chosen}
    wineries = {wine.winery_norm for wine in chosen[:-1]}
    for wine in ranked:
        if wine.slug in taken or wine.winery_norm in wineries:
            continue
        if differs(facts, wine, catalog.alcohol(wine.slug)):
            chosen[-1] = wine
            break
    return chosen


def _relaxed_color_note(catalog: RecoCatalog, facts: Facts) -> Note:
    """«Оранжевых игристых в каталоге нет — показываем игристые брют других цветов»."""
    color = _COLOR_GEN_PL.get(facts.color or "", "такого цвета")
    what = f"{color} игристых" if facts.sparkling else f"{color} вин такого стиля"
    what = what[:1].upper() + what[1:]
    if facts.sparkling:
        sugar = SUGAR_WORDS.get(facts.sugar or "", "")
        rest = f"игристые {sugar}".strip()
    else:
        rest = "близкие по стилю"
    if catalog.style_pool(facts.color, facts.sparkling):
        return Note("relaxed_color", f"{what} в каталоге мало — добавили {rest} других цветов")
    return Note("relaxed_color", f"{what} в каталоге нет — показываем {rest} других цветов")


def similar(
    catalog: RecoCatalog,
    facts: Facts,
    limit: int = DEFAULT_LIMIT,
    *,
    relax_color: bool = False,
) -> Selection:
    """Похожие из других виноделен (`order=reco`).

    `relax_color` — для «Не тупика»: если вин того же цвета и игристости меньше `limit`,
    цвет ослабляется, а вина нужного цвета остаются первыми (нота `relaxed_color`).
    """
    notes: list[Note] = []
    pool = _candidates(catalog, facts, color=True)
    relaxed = False
    if relax_color and facts.color is not None and len(pool) < limit:
        pool = _candidates(catalog, facts, color=False)
        relaxed = True
    ranked = _rank(catalog, facts, pool, relaxed_color=relaxed)
    wines = with_contrast(catalog, facts, ranked, one_per_winery(ranked, limit))
    if relaxed and any(wine.color != facts.color for wine in wines):
        notes.append(_relaxed_color_note(catalog, facts))
    picks = []
    for wine in wines:
        alcohol = catalog.alcohol(wine.slug)
        picks.append(
            Pick(
                wine,
                tuple(explain(facts, wine, alcohol)),
                differs(facts, wine, alcohol),
            )
        )
    if len(picks) < limit:
        notes.append(fewer_note(len(picks), limit))
    return Selection(tuple(picks), tuple(notes))


def plain(catalog: RecoCatalog, facts: Facts, limit: int = DEFAULT_LIMIT) -> Selection:
    """Обычная сортировка: та же категория (цвет, игристость, сахар) по названию, без объяснений.

    Неизвестный признак категорию не сужает. Своя винодельня и своя группа исключены, как и
    в рекомендациях, — иначе «обычная сортировка» подсовывала бы то же вино в другом объёме.
    """
    wines = sorted(
        (
            wine
            for wine in catalog.style_pool(facts.color, facts.sparkling)
            if not excluded(wine, facts) and (facts.sugar is None or wine.sugar == facts.sugar)
        ),
        key=name_key,
    )[:limit]
    notes = (fewer_note(len(wines), limit),) if len(wines) < limit else ()
    return Selection(tuple(Pick(wine) for wine in wines), notes)


def same_winery(
    catalog: RecoCatalog,
    wineries: Iterable[str],
    facts: Facts,
    *,
    order: str = "reco",
    limit: int = SAME_WINERY_LIMIT,
) -> Selection:
    """Вина прочитанной винодельни для «Не тупика».

    `reco`: сначала совпавшие по игристости, цвету и сахару, затем по общим сортам и названию;
    объяснение — те же факты без региона и крепости (у винодельни они и так общие).
    `plain`: по названию, без объяснений.
    """
    wines = [wine for norm in sorted(set(wineries)) for wine in catalog.winery_pool(norm)]
    if order == "plain":
        return Selection(tuple(Pick(wine) for wine in sorted(wines, key=name_key)[:limit]))
    grapes = set(facts.grapes)

    def key(wine: RecoWine) -> tuple:
        return (
            facts.sparkling is not None and wine.sparkling != facts.sparkling,
            facts.color is not None and wine.color != facts.color,
            facts.sugar is not None and wine.sugar != facts.sugar,
            -jaccard(grapes, wine.grapes) if grapes else 0.0,
            *name_key(wine),
        )

    picks = []
    for wine in sorted(wines, key=key)[:limit]:
        reasons = explain(facts, wine, catalog.alcohol(wine.slug), region=False, strength=False)
        picks.append(Pick(wine, tuple(reasons or ["Та же винодельня"])))
    return Selection(tuple(picks))


def winery_region(catalog: RecoCatalog, wineries: Iterable[str]) -> str | None:
    """Самый частый регион вин винодельни: у этикетки региона нет, а у хозяйства он есть."""
    regions = Counter(
        wine.region for norm in wineries for wine in catalog.winery_pool(norm) if wine.region
    )
    return regions.most_common(1)[0][0] if regions else None


def sparkling_by_sugar(sugar: str | None) -> bool | None:
    """Брют бывает только у игристого; остальной сахар об игристости ничего не говорит."""
    return True if sugar in SPARKLING_SUGARS else None


def tile(wine: RecoWine, reasons: Sequence[str] = (), *, photo_url: str | None) -> dict:
    """Плитка вина для блоков рекомендаций (договор, «Общие объекты»).

    `photo_url` решает сервис: он знает, есть ли фото на этой машине.
    """
    return {
        "slug": wine.slug,
        "name": wine.title,
        "winery": wine.winery,
        "region": wine.region,
        "style_label": wine.style_label,
        "sparkling": wine.sparkling,
        "grapes": wine.grape_labels,
        "photo_url": photo_url,
        "portal_url": wine.portal_url,
        "reasons": list(reasons),
    }
