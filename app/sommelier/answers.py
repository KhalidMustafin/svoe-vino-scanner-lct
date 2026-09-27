"""Ответы сомелье: заметка карточки и пакет фактов на каждый вопрос листа.

Всё считается на CPU из готовых данных, без модели (договор `docs/api-sommelier.md`, §2–§4):

* факты вина — карточка без портала (`AfterSearch.card`) и позиция справочника
  рекомендаций; профиль по 8 осям — `app/recommend/profile.py` через «Сомелье у полки»;
* пары с блюдами — `pairs.json` и правила `dishes.json` (§7.1–7.2), подача — наша таблица
  `serve.json` (§7.3), справочник и портреты сортов — `knowledge.json` (§7.5);
* вина других виноделен — правила «Сомелье у полки» (направление «помягче / посвежее»,
  `Shelf.move`) и похожих (`facts.similar`), с блюдом — только вина с вердиктом `yes` к нему;
  подборка после оговорки и «скорее нет» идёт в сторону причины (`REASONS`): «полегче»,
  «помощнее», «суше»…, и не держится цвета вина карточки, если причина зовёт в другой стиль.

`Sommelier.card` отдаёт тело `GET /v1/wines/{slug}/sommelier`, `Sommelier.answer` — генератор:
он отдаёт этапы по мере работы («card», «rules», «catalog») и возвращает `FactsPackage`.
Пакет — это тело события `facts` (`public`) и то, что увидит модель: строки фактов без
вопроса гостя, описания, дескрипторов и чисел профиля, плюс списки разрешённых имён и
чисел для проверок текста (§6.4). Текст модели сюда не возвращается никогда.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import canonical_grape, grape_label
from app.recommend.build import (
    COLOR_VALUES,
    REGION_CODES,
    SUGAR_VALUES,
    build_profile,
    load_priors,
    profile_of,
)
from app.recommend.catalog import (
    SPARKLING_SUGARS,
    SUGAR_STEPS,
    SUGAR_WORDS,
    RecoCatalog,
    RecoWine,
    card_sugar,
    name_key,
    sweet_name,
)
from app.recommend.catalog import style_label as catalog_style_label
from app.recommend.content_filter import check
from app.recommend.facts import (
    Facts,
    abv_gap,
    excluded,
    explain,
    jaccard,
    one_per_winery,
    similar,
    sugar_close,
)
from app.recommend.profile import (
    AXES,
    GRAPE,
    RussianPGI,
    StyleKey,
    StylePriors,
    StyleProfile,
    WineKind,
)
from app.recommend.shelf import (
    LOW_TANNIN_COLOURS,
    MAX_REASONS,
    NEAR_POOL,
    NO_MOVE,
    TANNIC_COLOURS,
    UNKNOWN_SUGAR_GAP,
    Move,
    Shelf,
)
from app.recommend.somm_data import rule_order
from app.sommelier import templates as t
from app.sommelier.barrier import Refusal
from app.sommelier.router import Context, Route
from app.sommelier.smalltalk import reply
from app.sommelier.text import upper_first
from app.sommelier.voice import USER_BUDGET_CHARS

#: Подписи осей для шкал и «розы ветров» (договор, §1).
AXIS_UI: dict[str, tuple[str, str, str]] = {
    "sweetness": ("Сладость", "сухое", "сладкое"),
    "acidity": ("Кислотность", "мягкая", "живая"),
    "tannin": ("Танины", "мягкие", "терпкие"),
    "body": ("Тело", "лёгкое", "плотное"),
    "alcohol": ("Крепость", "лёгкое", "крепкое"),
    "oak": ("Дуб", "без дуба", "заметный"),
    "aroma_intensity": ("Аромат", "сдержанный", "яркий"),
    "effervescence": ("Пузырьки", "тихое", "игристое"),
}
#: Сорта-заглушки выгрузки: не сорт, а группа.
GRAPE_PLACEHOLDERS = frozenset({"белые сорта винограда", "красные сорта винограда"})
#: Поля вина в правилах сочетаний, которые всегда из карточки выгрузки.
CATALOG_FIELDS = frozenset({"region", "color", "grapes", "sparkling"})
SOURCE_RANK = {"catalog": 0, "grape": 1, "type": 2}
SOURCE_NAMES = ("catalog", "grape", "type")
#: Источники осей, по которым вина сравниваются для наложения на «розу»: оценка по стилю — нет.
COMPARED = frozenset({"catalog", "grape"})
MAX_PLUS = 3
MAX_MINUS = 2
LIMIT = 3
MAX_SHOWN = 12
#: Строки фактов для модели — не длиннее того, что остаётся от ≈ 700 токенов входа после
#: системного сообщения (договор, §6.4; `voice.MAX_INPUT_CHARS`).
MAX_VOICE_CHARS = USER_BUDGET_CHARS
#: Блюда, о которых сомелье предлагает спросить чипом «А к …?» — частые на столе.
POPULAR_DISHES = (
    "shashlyk_svinina", "shashlyk_baranina", "steik_ribay", "cheese_plate", "plov",
    "solenaya_seld", "oysters", "borsch", "pizza", "sushi", "napoleon", "shokolad", "frukty",
)  # fmt: skip
#: Намерения, которые модель пересказывает (договор, §3.3 `not_voiced`): только блюда. Подборки
#: («Помягче», «Посвежее», «Чем заменить», финал «Помогите выбрать») — всегда шаблон: в окне
#: видеокарты 25.09 модель выдала вино карточки за вино подборки в 10 из 40 прошедших проверки
#: текстов (договор, §6.5). Какие ответы о блюдах пересказываются — `voice_fits`.
VOICED = frozenset({"what_to_eat", "dish_check"})
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
#: Подборка после оговорки и «скорее нет» — от причины сильнейшего правила «−» пары. Проверка 25.09,
#: вечер: у оговорки заголовок и вина отвечают на причину, а не всегда «Если хочется мягче».
#: Третий круг проверки 25.09: так же и после «скорее нет» (раньше там были просто вина к блюду, того
#: же цвета, что вино карточки: к шашлыку из баранины у Шардоне — белые), сдвиг засчитывается
#: только по тому, что называет причина, и только заметный (`shelf.reason_move`): у «Крепость
#: перекрывает блюдо» «полегче» — правда слабее, у «Мягкому вину тяжело с жирным» — кислотность
#: выше, а не «крепость ниже». Цвет вина карточки подборку не держит, если причина зовёт в другой
#: стиль (`Sommelier._colours`): к рыбе от терпкого красного — белое и розовое, помощнее к мясу у
#: белого — красное.
#:
#: Правило «−» → варианты: (ось профиля и порог, от которого вариант — причина у вина карточки,
#: или `None`; направление; доказательства сдвига по порядку). У правил с несколькими условиями
#: («Танины разжигают острое»: танины от 3, или крепость от 4, или дуб от 3) причина — первый
#: вариант, порог которого вино карточки переходит; ни один — первый. Пороги — из условий правил
#: (`tools/loza_engines`, накладка): в `dishes.json` условий нет.
REASONS: dict[str, tuple[tuple[tuple[str, float] | None, str, tuple[str, ...]], ...]] = {
    "wine_overpowers_dish": ((None, "lighter", ("strength", "body")),),
    "heavy_wine_on_delicate_dish": (
        (("body", 3.8), "lighter", ("strength", "body")),
        (("tannin", 3.0), "softer", ("colour", "tannin")),
        (("oak", 3.0), "lighter", ("oak",)),
    ),
    "high_alcohol_on_delicate": ((None, "lighter", ("strength",)),),
    "oak_crushes_delicate": ((None, "lighter", ("oak",)),),
    "intensity_mismatch": ((None, "fuller", ("strength", "body")),),
    "flat_wine_on_fat_dish": ((None, "fresher", ("acidity",)),),
    "flat_wine_on_sour_dish": ((None, "fresher", ("acidity",)),),
    "bare_tannin_on_lean_dish": ((None, "softer", ("colour", "tannin")),),
    "umami_vs_tannin_clash": (
        (("tannin", 3.5), "softer", ("colour", "tannin")),
        (("oak", 3.5), "lighter", ("oak",)),
    ),
    "no_tannin_with_oily_fish": ((None, "softer", ("colour", "tannin")),),
    "tannin_vs_spice_clash": (
        (("tannin", 3.0), "softer", ("colour", "tannin")),
        (("alcohol", 4.0), "lighter", ("strength",)),
        (("oak", 3.0), "lighter", ("oak",)),
    ),
    "dessert_wine_on_savoury_dish": ((None, "drier", ("sugar",)),),
    "semisweet_wine_on_savoury_dish": ((None, "drier", ("sugar",)),),
    "sweet_wine_on_savoury_main": ((None, "drier", ("sugar",)),),
    "sweet_wine_on_fish": ((None, "drier", ("sugar",)),),
    # Оговорка о рыбе у вина, возможно сладкого по названию: «суше» — вина, сухие по карточке.
    "maybe_sweet_wine_on_fish": ((None, "drier", ("sugar",)),),
    "semisweet_wine_on_dessert": ((None, "sweeter", ("sugar",)),),
    "dry_wine_on_dessert": ((None, "sweeter", ("sugar",)),),
}
#: Ось мини-шкалы под плитками (§1) — по последнему доказательству причины: у «полегче» по
#: крепости и телу — тело, по одной крепости — крепость, у «помягче» — танины.
EVIDENCE_AXES: dict[str, str] = {
    "strength": "alcohol",
    "body": "body",
    "oak": "oak",
    "colour": "tannin",
    "tannin": "tannin",
    "acidity": "acidity",
    "sugar": "sweetness",
}
#: Подпись плитки, когда сдвиг подтверждён осью «по сорту» (`Move.axis`).
REASON_PILLS: dict[tuple[str, str], str] = {
    ("lighter", "body"): "тело легче — по сорту",
    ("fuller", "body"): "тело плотнее — по сорту",
    ("lighter", "oak"): "дуба меньше — по сорту",
    ("softer", "tannin"): "танины мягче — по сорту",
    ("fresher", "acidity"): "кислотность выше — по сорту",
}
#: Правило, которое срабатывает у блюда, если блюдо зовёт свой стиль (`Sommelier._colours`): рыба
#: и морепродукты — вино почти без танинов, белок и жир мяса — терпкое красное.
FISH_RULE = "no_tannin_with_oily_fish"
PROTEIN_RULE = "tannin_meets_protein_fat"
RED = "Красное"


@dataclass(frozen=True, slots=True)
class Reason:
    """Причина «−» пары и куда из-за неё идти подборке: направление, доказательства сдвига
    (`shelf.reason_move`) и цвета, куда зовёт причина (`None` — цвет вина карточки первым)."""

    rule: str
    want: str
    by: tuple[str, ...]
    colours: tuple[str, ...] | None = None

    @property
    def axis(self) -> str:
        return EVIDENCE_AXES[self.by[-1]]


VERDICT_WORDS = {
    "yes": "подходит",
    "caveat": "подходит с оговоркой",
    "no": "скорее нет",
    "neutral": "правила молчат",
}


def voice_fits(intent: str, dishes: Sequence[Mapping[str, Any]], wines: Sequence[Any]) -> bool:
    """Пересказывает ли голос ответ такого вида (договор, §6.5): блюда (`VOICED`) без вин
    подборки, а у `dish_check` — только вердикт «да».

    Повторный замер окна видеокарты 25.09: из 12 прошедших проверки текстов
    `dish_check` с вердиктом не «да» 6 из 8 при «скорее нет» грубо неверны — вердикт перевёрнут
    («К утке с яблоками подходит розовое брют…»), вино карточки выдано за вино подборки, сорт
    вина подборки приписан вину карточки. Все такие пакеты — с подборкой: модель путает якорь с
    подборкой, как и у подборок, с которых голос сняли раньше. Проверки текста этого не видят —
    имена и сорта в пакете есть, — поэтому здесь не проверка, а ворота: шаблон с
    `reason: "not_voiced"`. Ответы «да» (120 из 120 чипов карточки) и «К чему подать» голос
    по-прежнему пересказывает.
    """
    if intent not in VOICED or wines:
        return False
    if intent == "dish_check":
        return bool(dishes) and all(dish.get("verdict") == "yes" for dish in dishes)
    return True


# ------------------------------------------------------------------ пакет фактов (§6.4)
@dataclass(frozen=True)
class FactsPackage:
    """Готовый ответ: `public` — тело события `facts`, остальное — для голоса и проверок."""

    public: Mapping[str, Any]
    verdict_template: str
    voiced: bool
    intent_label: str
    voice_input: tuple[str, ...]
    allowed_entities: frozenset[str]
    allowed_numbers: frozenset[str]
    allowed_text: str


# ------------------------------------------------------------------ источник вин
@dataclass(frozen=True)
class WineSource:
    """Что сомелье берёт у слоя «после поиска»: справочник, «Сомелье у полки», карточку, плитку."""

    catalog: RecoCatalog
    shelf: Shelf
    card: Callable[[str], Mapping[str, Any] | None]
    tile: Callable[[RecoWine, Sequence[str]], dict[str, Any]]

    @classmethod
    def of_after(cls, after: Any) -> WineSource:
        """Из `AfterSearch` сервиса: `catalog`, `sommelier` (Shelf), `card`, `tile`."""
        return cls(after.catalog, after.sommelier, after.card, after.tile)


# ------------------------------------------------------------------ профиль и подача
def profile_items(profile: StyleProfile | None) -> list[dict[str, Any]]:
    """8 осей в порядке `AXES`: `value = round(ось / 5, 2)`, у оси без источника — `null`.

    Профиль — для показа (`Sommelier.shown`): оси «по сорту», которых у вина нет, уже закрыты
    оценкой по стилю с источником `style`.
    """
    items = []
    for axis in AXES:
        label, left, right = AXIS_UI[axis]
        source = profile.source(axis) if profile is not None else None
        value = round(profile.axis(axis) / 5, 2) if profile is not None and source else None
        items.append(
            {
                "axis": axis,
                "label": label,
                "left": left,
                "right": right,
                "value": value,
                "source": source,
            }
        )
    return items


@dataclass(frozen=True, slots=True)
class Serve:
    """Правило подачи вина и сверялось ли оно с телом, известным по сорту.

    `body_unknown` — правило с телом (`red_full`, `white_full`) не подошло только потому, что тела
    нет: взято общее правило цвета, и его текст про лёгкие вина к этому вину не относится.
    """

    rule: Any  # ServeRule
    by_grape: bool
    body_unknown: bool = False

    @property
    def temperature(self) -> tuple[int, int]:
        return tuple(self.rule.temperature_c)  # type: ignore[return-value]

    def text(self, color: str | None) -> str:
        """Текст правила; тело неизвестно — нейтральная фраза без «лёгкие красные»."""
        return t.serve_neutral(color) if self.body_unknown else self.rule.text

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature_c": list(self.rule.temperature_c),
            "source": "rule",
            "rule": self.rule.id,
            "by_grape": self.by_grape,
        }


def serve_of(
    rules: Iterable[Any],
    *,
    color: str | None,
    sugar: str | None,
    sparkling: bool,
    profile: StyleProfile,
    sweet_name: bool = False,
) -> Serve | None:
    """Первое подошедшее правило `serve.json` (§7.3); неизвестное тело `body_min` не отвечает.

    `sweet_name` — сахар неизвестен, а название креплёного, десертного или мускатного вина
    (`catalog.sweet_name`): правило `sweet_name` подаёт его как сладкое, а не как сухое белое.
    """
    body_known = profile.source("body") is not None
    body_checked = False
    for rule in rules:
        if rule.sparkling is not None and rule.sparkling != sparkling:
            continue
        rule_sweet = getattr(rule, "sweet_name", None)
        if rule_sweet is not None and rule_sweet != sweet_name:
            continue
        if rule.sugar is not None and sugar not in rule.sugar:
            continue
        if rule.color is not None and color not in rule.color:
            continue
        if rule.body_min is not None:
            body_checked = True
            if not body_known or profile.body < rule.body_min:
                continue
        return Serve(
            rule,
            body_checked and profile.source("body") == GRAPE,
            body_unknown=body_checked and not body_known,
        )
    return None


@lru_cache(maxsize=1)
def _priors() -> Mapping[str, Any]:
    return load_priors()


def style_key(color: Any, sparkling: Any, sugar: Any) -> StyleKey:
    """Стиль вина для оценки по стилю: цвет и сахар — коды выгрузки или `None`."""
    return (
        color if color in COLOR_VALUES else None,
        bool(sparkling),
        sugar if sugar in SUGAR_VALUES else None,
    )


def profile_from_card(card: Mapping[str, Any], sugar: str | None) -> StyleProfile:
    """Профиль по фактам карточки — когда позиции нет в справочнике рекомендаций; `sugar` — сахар
    по карточке (`catalog.card_sugar`: поле карточки, а без него — «Описание»)."""
    grapes = [code for code in (canonical_grape(str(g)) for g in card.get("grapes") or []) if code]
    color = card.get("color_label")
    return build_profile(
        grapes,
        Color(color) if color in Color._value2member_map_ else None,
        SugarClass(sugar) if sugar in SugarClass._value2member_map_ else None,
        WineKind.SPARKLING if card.get("sparkling") else WineKind.STILL,
        _priors(),
        region=REGION_CODES.get(str(card.get("region") or ""), RussianPGI.OTHER),
        name=str(card.get("name") or ""),
        abv=card.get("alcohol"),
    )


# ------------------------------------------------------------------ вино вопроса
@dataclass(frozen=True)
class Anchor:
    """Вино карточки со всеми фактами, которые нужны ответам."""

    slug: str
    name: str
    winery: str
    region: str
    style_label: str
    color: str | None
    #: Сахар по карточке (`catalog.card_sugar`): название, slug, а без них «сладкое вино» в
    #: «Описании»; у такого вина с ним и `style_label` («Розовое сладкое»), и `wine`, и `profile`.
    sugar: str | None
    sparkling: bool
    grapes: tuple[str, ...]  # коды сортов
    grape_names: tuple[str, ...]  # «Сорт винограда» выгрузки без заглушек
    wine: RecoWine | None
    profile: StyleProfile  # профиль вина: подбор, пары и подача видят только его
    shown: StyleProfile  # профиль для «розы ветров» и шкал: пустые оси «по сорту» — по стилю
    serve: Serve | None
    pairs: Any | None  # WinePairs
    #: «Возможно сладкое по названию» (`catalog.sweet_name`): сахара по карточке нет, название —
    #: креплёного, десертного или мускатного вина, а ни название, ни «Описание» не говорят «сухое».
    sweet_name: bool = False

    @property
    def dishes(self) -> tuple[str, ...]:
        return tuple(self.pairs.top) if self.pairs is not None else ()


@dataclass
class Draft:
    """Ответ в сборке: поля события `facts` и строки для модели."""

    intent: str
    context: Context
    verdict: str = ""
    detail: str | None = None
    basis: list[str] = field(default_factory=list)
    dishes: list[dict[str, Any]] = field(default_factory=list)
    wines_title: str | None = None
    wines: list[dict[str, Any]] = field(default_factory=list)
    compare: dict[str, Any] | None = None
    serve: dict[str, Any] | None = None
    question: dict[str, Any] | None = None
    refusal: dict[str, Any] | None = None
    chips: list[dict[str, Any]] = field(default_factory=list)
    voiced: bool = False
    intent_label: str = ""
    rule_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Pick:
    wine: RecoWine
    move: Move
    reasons: tuple[str, ...]


# ------------------------------------------------------------------ сомелье
class Sommelier:
    """Сомелье на данных `SommData` поверх слоя «после поиска»."""

    def __init__(self, data: Any, source: WineSource, *, live: bool, input_enabled: bool) -> None:
        self.data = data
        self.source = source
        self.live = live
        self.input_enabled = input_enabled
        self.food_dishes: dict[str, tuple[str, ...]] = {}
        for dish in data.dishes.values():
            if dish.food:
                self.food_dishes[dish.food] = (*self.food_dishes.get(dish.food, ()), dish.id)
        self._style: StylePriors | None = None
        self._fires: dict[tuple[str, str], bool] = {}

    # -------------------------------------------------------------- профиль для показа
    def style_priors(self) -> StylePriors:
        """Медианы осей «по сорту» по стилям справочника — раз на сомелье, при первом показе."""
        if self._style is None:
            profiles = self.source.shelf.profiles
            self._style = StylePriors.build(
                (style_key(wine.color, wine.sparkling, wine.sugar), profiles[wine.slug])
                for wine in self.source.catalog
                if wine.slug in profiles
            )
        return self._style

    def shown(self, profile: StyleProfile | None, key: StyleKey) -> StyleProfile | None:
        """Профиль для «розы ветров» и шкал: пустые оси «по сорту» — оценка по стилю."""
        if profile is None:
            return None
        return self.style_priors().fill(profile, key)

    def shown_of(self, slug: str) -> StyleProfile | None:
        """Профиль для показа вина справочника (плитки сравнения)."""
        wine = self.source.catalog.get(slug)
        profile = self.source.shelf.profiles.get(slug)
        if wine is None or profile is None:
            return profile
        return self.shown(profile, style_key(wine.color, wine.sparkling, wine.sugar))

    # -------------------------------------------------------------- вино
    def anchor(self, slug: str) -> Anchor | None:
        """Вино карточки или `None` — такой карточки нет (404).

        Сахар вина — сахар по карточке (`catalog.card_sugar`), как у сборки данных: сахар выгрузки,
        а без него «сладкое», если так прямо написано в «Описании» (проверка 25.09:
        «Мускат позднего сбора розовый» — «Сладкое розовое вино»). У такого вина стиль, профиль,
        подача, хранение и позиция справочника для подбора (`Anchor.wine` — копия с этим сахаром)
        — сладкого по карточке, а не догадки по названию.
        """
        card = self.source.card(slug)
        if card is None:
            return None
        wine = self.source.catalog.get(slug)
        if wine is not None:
            color, sugar, sparkling = wine.color, wine.sugar, bool(wine.sparkling)
            grapes = tuple(wine.grapes)
        else:
            color = card.get("color_label") or None
            sugar = card.get("sugar_class") or None
            sparkling = bool(card.get("sparkling"))
            grapes = tuple(
                code for code in (canonical_grape(str(g)) for g in card.get("grapes") or []) if code
            )
        style = str(card.get("style_label") or (wine.style_label if wine else ""))
        profile = self.source.shelf.profiles.get(slug) if wine is not None else None
        described = card_sugar(sugar, card.get("description"))
        if described != sugar:
            # Сахара нет ни в названии, ни в slug, а «Описание» называет вино сладким: стиль,
            # позиция справочника для подбора и профиль — сладкого по карточке (сладость «из
            # карточки»), а не сухого по умолчанию. Справочник и карточка «после поиска» — как были.
            sugar = described
            style = catalog_style_label(color, sugar, sparkling)
            profile = None
            if wine is not None:
                wine = replace(wine, sugar=sugar)
                profile = profile_of(wine, _priors(), alcohol=self.source.catalog.alcohol(slug))
        if profile is None:
            profile = profile_from_card(card, sugar)
        names = tuple(
            str(g).strip()
            for g in card.get("grapes") or []
            if str(g).strip() and str(g).strip().casefold() not in GRAPE_PLACEHOLDERS
        )
        shown = self.shown(profile, style_key(color, sparkling, sugar)) or profile
        # Догадка «возможно сладкое» — по названию и «Описанию» карточки, как у сборки данных.
        by_name = sweet_name(
            card.get("name") or (wine.title if wine else ""), sugar, card.get("description")
        )
        return Anchor(
            slug=slug,
            name=str(card.get("name") or (wine.title if wine else slug)),
            winery=str(card.get("winery") or (wine.winery if wine else "")),
            region=str(card.get("region") or (wine.region if wine else "")),
            style_label=style,
            color=color,
            sugar=sugar,
            sparkling=sparkling,
            grapes=grapes,
            grape_names=names,
            wine=wine,
            profile=profile,
            shown=shown,
            serve=serve_of(
                self.data.serve,
                color=color,
                sugar=sugar,
                sparkling=sparkling,
                profile=profile,
                sweet_name=by_name,
            ),
            pairs=self.data.pairs.get(slug),
            sweet_name=by_name,
        )

    # -------------------------------------------------------------- блюда
    def chip_source(self, rule: Any, anchor: Anchor) -> str:
        """Самый слабый источник среди полей вина, которые читает правило (§1)."""
        rank = 0
        for name in rule.wine_fields:
            if name in AXES:
                source = anchor.profile.source(name) or "type"
            elif name in CATALOG_FIELDS:
                source = "catalog"
            elif name == "descriptors":
                source = "grape" if anchor.profile.basis != "low" else "type"
            elif name == "serve":
                source = "grape" if anchor.serve and anchor.serve.by_grape else "catalog"
            else:
                source = "type"
            rank = max(rank, SOURCE_RANK[source])
        return SOURCE_NAMES[rank]

    def rule_chips(self, rule_ids: Sequence[str], anchor: Anchor, limit: int) -> list[dict]:
        rules = [self.data.rules[rid] for rid in rule_ids if rid in self.data.rules]
        rules.sort(key=lambda rule: rule_order(rule.id, rule.weight))
        return [
            {"id": rule.id, "text": rule.chip, "source": self.chip_source(rule, anchor)}
            for rule in rules[:limit]
        ]

    def pair(self, anchor: Anchor, dish_id: str) -> Any:
        if anchor.pairs is None:
            return _NEUTRAL
        return anchor.pairs.pair(dish_id)

    def dish_item(self, anchor: Anchor, dish_id: str) -> dict[str, Any]:
        """Блюдо с вердиктом и чипами правил; `source` — общий источник всех чипов блюда.

        Если у всех чипов блюда один источник, страница пишет его раз в строке блюда («подходит ·
        правила по сорту»), а не «· по сорту» у каждого чипа; разные источники или чипов нет —
        `null`, и источник остаётся у чипа.
        """
        dish = self.data.dishes[dish_id]
        pair = self.pair(anchor, dish_id)
        plus = self.rule_chips(pair.plus, anchor, MAX_PLUS)
        minus = self.rule_chips(pair.minus, anchor, MAX_MINUS)
        sources = {chip["source"] for chip in (*plus, *minus)}
        return {
            "id": dish.id,
            "name": dish.name,
            "category": dish.category,
            "verdict": pair.verdict,
            "plus": plus,
            "minus": minus,
            "source": sources.pop() if len(sources) == 1 else None,
        }

    def top_dishes(self, anchor: Anchor, order: str, limit: int = LIMIT) -> list[str]:
        """Первые блюда `top` (§7.1); при `plain` — те же, по названию."""
        ids = [dish for dish in anchor.dishes if dish in self.data.dishes][:limit]
        if order == "plain":
            ids.sort(key=lambda dish: self.data.dishes[dish].name.casefold())
        return ids

    # -------------------------------------------------------------- чипы
    def entry_chips(self, anchor: Anchor) -> list[dict[str, Any]]:
        """Входы в лист (§2): пять вопросов, сорт, «брют», «Помогите выбрать»."""
        chips = [t.chip(chip_id) for chip_id in ("what_to_eat", "serve", "softer", "fresher")]
        chips.append(t.chip("replace"))
        portrait = next((code for code in anchor.grapes if code in self.data.grapes), None)
        if portrait is not None:
            chips.append(t.grape_chip(portrait, self.data.grapes[portrait].label))
        if anchor.sugar in SPARKLING_SUGARS and "brut_scale" in self.data.topics:
            chips.append(t.term_chip("brut_scale", self.data.topics["brut_scale"].name))
        chips.append(t.chip("guided"))
        return chips

    def suggest_dish(self, exclude: Iterable[str]) -> dict[str, Any] | None:
        """Чип «А к …?» — частое блюдо, о котором ещё не говорили."""
        families = {self.data.dishes[d].family for d in exclude if d in self.data.dishes}
        for dish_id in POPULAR_DISHES:
            dish = self.data.dishes.get(dish_id)
            if dish is not None and dish.family not in families:
                return t.dish_chip(dish.id, dish.dative)
        return None

    def topic_chips(self, topic: Any) -> list[dict[str, Any]]:
        """Чипы темы справочника, у которых все ссылки есть в данных."""
        out = []
        for item in topic.chips:
            chip_id = item.get("id")
            args = dict(item.get("args") or {})
            if chip_id == "dish_check" and args.get("dish") not in self.data.dishes:
                continue
            if chip_id == "term" and args.get("topic") not in self.data.topics:
                continue
            if chip_id == "grape" and args.get("grape") not in self.data.grapes:
                continue
            out.append(t.chip(chip_id, item.get("text"), **args))
        return out

    # -------------------------------------------------------------- GET …/sommelier (§2)
    def card(self, slug: str, order: str) -> dict[str, Any] | None:
        anchor = self.anchor(slug)
        if anchor is None:
            return None
        dish_ids = self.top_dishes(anchor, "reco")
        names = [self.data.dishes[d].name for d in dish_ids]
        note = t.note_text(
            name=anchor.name,
            style=t.style_words(anchor.style_label, anchor.sparkling),
            # Сорт, который уже в названии, не повторяется: «Саперави — красное сухое.»
            grapes=t.grapes_not_in_name(anchor.grape_names, anchor.name),
            winery=anchor.winery,
            region=anchor.region,
            temperature=anchor.serve.temperature if anchor.serve else None,
            dishes=names,
        )
        if order == "plain":
            dish_ids = sorted(dish_ids, key=lambda d: self.data.dishes[d].name.casefold())
        return {
            "slug": anchor.slug,
            "name": anchor.name,
            "winery": anchor.winery,
            "order": order,
            "note": {
                "text": note,
                "basis": t.NOTE_BASIS,
                "generated": False,
                "label": t.LABEL_ALGO,
            },
            "profile": profile_items(anchor.shown),
            "serve": anchor.serve.as_dict() if anchor.serve else None,
            "dishes": [self.dish_item(anchor, d) for d in dish_ids],
            "chips": self.entry_chips(anchor),
            "live": self.live,
            "input": self.input_enabled,
            "notice_149": t.NOTICE_149 if order == "reco" else None,
        }

    # -------------------------------------------------------------- POST ask (§3)
    def answer(
        self, anchor: Anchor, route: Route, order: str, context: Context
    ) -> Generator[str, None, FactsPackage]:
        """Этапы по ходу работы (`yield`), затем пакет фактов (`return`)."""
        draft = Draft(route.intent, context=_carry(context, route.intent))
        intent = route.intent
        if intent == "refuse" and route.refusal is not None:
            self._refuse(draft, route.refusal)
        elif intent == "smalltalk":
            draft.verdict = reply(route.smalltalk or "help")
            draft.chips = self.entry_chips(anchor)
        elif intent == "unknown":
            self._unknown(draft, anchor, route)
        elif intent == "term":
            self._term(draft, route)
        elif intent == "guided" and not (route.food and route.want):
            self._guided_step(draft, anchor, route)
        else:
            yield "card"
            if intent == "what_to_eat":
                yield "rules"
                self._what_to_eat(draft, anchor, order)
            elif intent == "dish_check":
                yield from self._dish_check(draft, anchor, route, order)
            elif intent in ("softer", "fresher"):
                yield from self._direction(draft, anchor, route, order)
            elif intent == "replace":
                yield "catalog"
                self._replace(draft, anchor, order)
            elif intent == "serve":
                self._serve(draft, anchor)
            elif intent == "grape":
                self._grape(draft, route)
            elif intent == "fact":
                self._fact(draft, anchor, route)
            elif intent == "guided":
                yield from self._guided_final(draft, anchor, route, order)
            else:
                self._unknown(draft, anchor, Route("unknown"))
        return self._package(draft, anchor, order)

    # -------------------------------------------------------------- намерения
    def _refuse(self, draft: Draft, refusal: Refusal) -> None:
        draft.verdict = refusal.text
        draft.refusal = {"topic": refusal.topic, "text": refusal.text}
        draft.chips = [t.chip(chip_id) for chip_id in refusal.chips]
        draft.context = replace(
            draft.context, refusal_topic=refusal.topic if refusal.sticky else None
        )

    def _unknown(self, draft: Draft, anchor: Anchor, route: Route) -> None:
        draft.intent = "unknown"
        draft.context = replace(draft.context, intent="unknown")
        top = [d for d in anchor.dishes if d in self.data.dishes][:LIMIT]
        if route.unknown_dish and top:
            draft.verdict = t.UNKNOWN_DISH
            draft.chips = [t.dish_chip(d, self.data.dishes[d].dative) for d in top]
            return
        draft.verdict = t.UNKNOWN
        draft.chips = self.entry_chips(anchor)

    def _term(self, draft: Draft, route: Route) -> None:
        topic = self.data.topics.get(route.topic or "")
        if topic is None:
            draft.verdict = t.UNKNOWN
            return
        draft.verdict = topic.answer
        draft.detail = topic.detail or None
        draft.basis = [t.BASIS_KNOWLEDGE]
        draft.chips = self.topic_chips(topic)

    def _what_to_eat(self, draft: Draft, anchor: Anchor, order: str) -> None:
        ids = self.top_dishes(anchor, order)
        draft.dishes = [self.dish_item(anchor, d) for d in ids]
        names = [self.data.dishes[d].name for d in ids]
        first = self.pair(anchor, ids[0]) if ids else None
        rule = self._rule_text(first.plus if first else ())
        draft.verdict = t.what_to_eat(names, rule)
        draft.rule_texts = [rule] if rule else []
        draft.basis = self._pair_basis(draft.dishes)
        draft.voiced = bool(ids)
        draft.intent_label = "к чему подать это вино"
        chips = [self.suggest_dish(ids), t.chip("serve"), t.chip("replace")]
        draft.chips = [c for c in chips if c]

    def _dish_check(
        self, draft: Draft, anchor: Anchor, route: Route, order: str
    ) -> Generator[str, None, None]:
        dish = self.data.dishes[route.dish]  # type: ignore[index]
        yield "rules"
        pair = self.pair(anchor, dish.id)
        item = self.dish_item(anchor, dish.id)
        draft.dishes = [item]
        draft.basis = self._pair_basis([item])
        draft.intent_label = f"подходит ли вино к блюду «{dish.name}»"
        # «Правила молчат» пересказывать нечего; оговорку и «скорее нет» голос не пересказывает
        # вовсе (`voice_fits`, решение 25.09): пакет с ними несёт и вина подборки.
        draft.voiced = pair.verdict != "neutral"
        rule_ids = pair.plus if pair.verdict == "yes" else pair.minus
        rule = self._rule_text(rule_ids)
        draft.rule_texts = [rule] if rule else []
        tail: str | None = None
        want: str | None = None
        if pair.verdict in ("caveat", "no"):
            yield "catalog"
            accept = self._goes_with_dish(dish.id)
            reason: Reason | None = None
            if order == "plain":
                picks = self._plain(anchor, accept, draft.context.shown)
                if picks:
                    tail = t.plain_text(anchor.style_label)
                    draft.wines_title = t.plain_title(anchor.style_label)
            else:
                picks = []
                # Подборка — в сторону причины и после оговорки, и после «скорее нет»; цвета —
                # куда зовёт причина (третий круг проверки 25.09).
                reason = self._reason(pair, anchor, dish.id)
                if reason is not None:
                    picks = self._select(
                        anchor, reason.want, accept, draft.context.shown, reason=reason
                    )
                    if picks:
                        want = reason.want
                        tail = (
                            t.caveat_tail(want, dish.dative)
                            if pair.verdict == "caveat"
                            else t.no_tail(want, dish.dative)
                        )
                        draft.wines_title = t.caveat_title(want, dish.dative)
                if not picks:
                    # Сдвига в сторону причины в каталоге нет или его не оценить: вина к блюду
                    # без направления, но того стиля, куда зовёт причина.
                    colours = reason.colours if reason is not None else None
                    reason = None
                    picks = self._select(
                        anchor, "none", accept, draft.context.shown, near=True, colours=colours
                    )
                    if picks:
                        tail = t.matching_tail(dish.dative)
                        draft.wines_title = upper_first(dish.dative)
            dish_reason = f"{upper_first(dish.dative)} — по правилам сочетаний"
            self._tiles(draft, anchor, picks, order, want or "none", dish_reason, reason=reason)
        if pair.verdict == "neutral" and anchor.sugar is None and dish.food == "dessert":
            # Сахара вина в выгрузке нет, а к сладкому решает он: сборка не ставит таким парам
            # правил (`build_somm.sugar_decides`), и шаблон не называет вино сухим.
            draft.verdict = t.sweet_dish_sugar_unknown(dish.dative)
        else:
            draft.verdict = t.dish_verdict(pair.verdict, dish.dative, rule, tail)
        # В контексте разговора направления — только чипов листа (§4.3): «полегче» и «суше»
        # подборки оговорки следующий ход не продолжает.
        chip_want = want if want in ("softer", "fresher") else None
        draft.context = replace(draft.context, dish=dish.id, want=chip_want)
        other = "fresher" if want == "softer" else "softer"
        chips = [self.suggest_dish([dish.id]), t.chip("serve"), t.chip(other, dish=dish.id)]
        draft.chips = [c for c in chips if c]

    def _direction(
        self, draft: Draft, anchor: Anchor, route: Route, order: str
    ) -> Generator[str, None, None]:
        want = route.intent
        dish = self.data.dishes.get(route.dish or "")
        accept: Callable[[RecoWine], bool] | None = None
        if dish is not None:
            yield "rules"
            accept = self._goes_with_dish(dish.id)
        yield "catalog"
        dative = dish.dative if dish is not None else None
        draft.intent_label = f"вина {t.WANT_WORDS[want]}"
        draft.context = replace(draft.context, dish=dish.id if dish else None, want=want)
        if order == "plain":
            picks = self._plain(anchor, accept, draft.context.shown)
            draft.verdict = t.plain_text(anchor.style_label) if picks else t.NO_SIMILAR
            if picks:
                draft.wines_title = t.plain_title(anchor.style_label)
            self._tiles(draft, anchor, picks, order, "none", None)
            draft.basis = [t.BASIS_CARD]
            draft.chips = [t.chip("what_to_eat"), t.chip("serve")]
            return
        picks = self._select(anchor, want, accept, draft.context.shown)
        dish_reason = f"{upper_first(dative)} — по правилам сочетаний" if dative else None
        if picks:
            expanded = self._expanded(anchor, picks)
            draft.verdict = t.direction_text(want, dative, len(picks), expanded=expanded)
            draft.wines_title = t.direction_title(want, dative)
        elif anchor.wine is not None and not self.source.shelf.evaluable(want, anchor.wine):
            draft.verdict = t.direction_unknown(want, anchor.wine)
        else:
            none_at_all = not self._select(anchor, want, None, (), limit=1)
            draft.verdict = t.direction_empty(want, dative, no_difference=none_at_all)
        self._tiles(draft, anchor, picks, order, want, dish_reason)
        draft.voiced = bool(picks)
        other = "fresher" if want == "softer" else "softer"
        draft.chips = [
            t.chip(other, dish=dish.id if dish else None),
            t.chip("replace"),
            t.chip("what_to_eat"),
        ]

    def _replace(self, draft: Draft, anchor: Anchor, order: str) -> None:
        draft.intent_label = "чем заменить вино"
        draft.chips = [t.chip("softer"), t.chip("fresher"), t.chip("what_to_eat")]
        if anchor.wine is None:
            draft.verdict = t.NO_SIMILAR
            return
        shown = set(draft.context.shown)
        if order == "plain":
            picks = self._plain(anchor, None, draft.context.shown)
            draft.verdict = t.plain_text(anchor.style_label) if picks else t.NO_SIMILAR
            if picks:
                draft.wines_title = t.plain_title(anchor.style_label)
            self._tiles(draft, anchor, picks, order, "none", None)
            draft.basis = [t.BASIS_CARD]
            return
        facts = Facts.of_wine(anchor.wine, self.source.catalog.alcohol(anchor.slug))
        selection = similar(self.source.catalog, facts, LIMIT + len(shown))
        picks = [
            _Pick(p.wine, NO_MOVE, tuple(p.reasons))
            for p in selection.picks
            if p.wine.slug not in shown
        ][:LIMIT]
        self._tiles(draft, anchor, picks, order, "replace", None)
        if picks:
            first = picks[0].wine
            shared = next((code for code in anchor.grapes if code in first.grapes), None)
            draft.verdict = t.replace_text(
                t.style_words(anchor.style_label, anchor.sparkling),
                grape_label(shared) if shared else None,
                same_profile=self._same_profile(anchor, [p.wine.slug for p in picks]),
            )
            draft.wines_title = t.CHIP_TEXTS["replace"]
        else:
            draft.verdict = t.NO_SIMILAR
        draft.voiced = bool(picks)

    def _serve(self, draft: Draft, anchor: Anchor) -> None:
        draft.chips = [t.chip("what_to_eat")]
        if "decanting" in self.data.topics:
            draft.chips.append(t.term_chip("decanting", self.data.topics["decanting"].name))
        draft.chips.append(t.chip("replace"))
        if anchor.serve is None:
            draft.verdict = t.NO_SERVE
            return
        draft.serve = anchor.serve.as_dict()
        draft.verdict = t.serve_text(
            anchor.name, anchor.serve.temperature, anchor.serve.text(anchor.color)
        )
        draft.basis = [t.BASIS_SERVE, t.BASIS_CARD]
        if anchor.serve.by_grape:
            draft.basis.append(t.BASIS_GRAPE)
        decanting = self.data.topics.get("decanting")
        tannic = anchor.profile.source("tannin") == GRAPE and anchor.profile.tannin >= 3.5
        if decanting is not None and tannic:
            draft.detail = decanting.answer

    def _fact(self, draft: Draft, anchor: Anchor, route: Route) -> None:
        """Факт карточки словами: сахар, крепость, срок хранения по нашему правилу стиля."""
        topics = self.data.topics
        draft.basis = [t.BASIS_CARD]
        chips: list[dict[str, Any] | None]
        if route.fact == "alcohol":
            card = self.source.card(anchor.slug) or {}
            draft.verdict = t.alcohol_fact(card.get("alcohol"), card.get("alcohol_max"))
            chips = [_term_chip(topics, "alcohol_strength"), t.chip("serve")]
        elif route.fact == "storage":
            body_known = anchor.profile.source("body") is not None
            rule = t.storage_rule(
                color=anchor.color,
                sugar=anchor.sugar,
                sparkling=anchor.sparkling,
                sweet_name=anchor.sweet_name,
                body=anchor.profile.body if body_known else None,
            )
            draft.verdict = t.storage_fact(rule)
            draft.basis.append(t.BASIS_KNOWLEDGE)
            if rule in ("red_full", "red") and anchor.profile.source("body") == GRAPE:
                draft.basis.append(t.BASIS_GRAPE)
            chips = [_term_chip(topics, "storage"), _term_chip(topics, "open_bottle")]
        else:
            sweeter = route.want == "sweeter"
            style = t.style_words(anchor.style_label, anchor.sparkling)
            draft.verdict = t.sugar_fact(style, anchor.sugar, sweeter=sweeter)
            scale = "brut_scale" if anchor.sugar in SPARKLING_SUGARS else "dry_vs_semi"
            chips = [_term_chip(topics, scale)]
            if sweeter:
                chips += [t.chip("softer"), t.chip("fresher")]
        draft.chips = [chip for chip in chips if chip] + [t.chip("what_to_eat")]

    def _grape(self, draft: Draft, route: Route) -> None:
        note = self.data.grapes.get(route.grape or "")
        draft.chips = [t.chip("what_to_eat"), t.chip("serve"), t.chip("replace")]
        if note is None:
            draft.verdict = t.UNKNOWN_GRAPE
            return
        draft.verdict = note.text
        draft.basis = [t.BASIS_KNOWLEDGE]

    def _guided_step(self, draft: Draft, anchor: Anchor, route: Route) -> None:
        if not route.food:
            draft.question = {"step": "food", "text": t.GUIDED_FOOD_QUESTION}
            draft.verdict = t.GUIDED_FOOD_QUESTION
            draft.chips = [t.chip("guided", text, food=food) for food, text in t.GUIDED_FOODS]
            draft.context = replace(draft.context, step="food", food=None, want=None)
            return
        draft.question = {"step": "want", "text": t.GUIDED_WANT_QUESTION}
        head = ""
        if route.food != "none":
            head = t.guided_food_verdict(route.food, self._group_verdict(anchor, route.food))
            draft.basis = [t.BASIS_PAIRS]
        draft.verdict = f"{head} {t.GUIDED_WANT_QUESTION}".strip()
        draft.chips = [
            t.chip("guided", text, food=route.food, want=want) for want, text in t.GUIDED_WANTS
        ]
        draft.context = replace(draft.context, step="want", food=route.food, want=None)

    def _guided_final(
        self, draft: Draft, anchor: Anchor, route: Route, order: str
    ) -> Generator[str, None, None]:
        food, want = route.food or "none", route.want or "none"
        accept: Callable[[RecoWine], bool] | None = None
        if food != "none":
            yield "rules"
            accept = self._goes_with_food(food)
        yield "catalog"
        draft.intent_label = "подбор к блюду"
        draft.context = replace(draft.context, step=None, food=food, want=want)
        draft.chips = [t.chip("guided"), t.chip("what_to_eat"), t.chip("serve")]
        if order == "plain":
            picks = self._plain(anchor, accept, draft.context.shown)
            draft.verdict = (
                t.plain_text(anchor.style_label) if picks else t.guided_empty(food, None)
            )
            if picks:
                draft.wines_title = t.plain_title(anchor.style_label)
            self._tiles(draft, anchor, picks, order, "none", None)
            draft.basis = [t.BASIS_CARD]
            return
        picks = self._select(anchor, want, accept, draft.context.shown, near=food != "none")
        food_word = food if food in t.FOOD_DATIVE else None
        want_word = want if want in t.WANT_WORDS else None
        if picks:
            draft.verdict = t.guided_text(food_word, want_word, len(picks))
            draft.wines_title = t.guided_title(food_word, want_word)
        elif (
            want_word
            and anchor.wine is not None
            and not self.source.shelf.evaluable(want, anchor.wine)
        ):
            draft.verdict = t.direction_unknown(want, anchor.wine)
        else:
            draft.verdict = t.guided_empty(food_word, want_word)
        reason = (
            f"{upper_first(t.FOOD_DATIVE[food])} — по правилам сочетаний" if food_word else None
        )
        self._tiles(draft, anchor, picks, order, want, reason)
        draft.voiced = bool(picks)

    # -------------------------------------------------------------- подбор вин
    def _reason(self, pair: Any, anchor: Anchor, dish_id: str) -> Reason | None:
        """Причина подборки — сильнейшее правило «−» пары с направлением (`REASONS`).

        У правил с несколькими условиями причина — то, что у вина карточки правда сработало:
        белому «Бульон подчёркивает терпкость» ставит дуб, а не танины, и подборка ему —
        «полегче» по дубу. Правил «−» нет или направления у них нет — `None`: подборка тогда без
        заголовка о причине.
        """
        rules = [self.data.rules[rid] for rid in pair.minus if rid in self.data.rules]
        rules.sort(key=lambda rule: rule_order(rule.id, rule.weight))
        for rule in rules:
            ways = REASONS.get(rule.id)
            if not ways:
                continue
            profile = anchor.profile
            _, want, by = next(
                (
                    way
                    for way in ways
                    if way[0] is None
                    or (profile.source(way[0][0]) and profile.axis(way[0][0]) >= way[0][1])
                ),
                ways[0],
            )
            return Reason(rule.id, want, by, self._colours(want, anchor, dish_id))
        return None

    def _colours(self, want: str, anchor: Anchor, dish_id: str) -> tuple[str, ...] | None:
        """Цвета, куда зовёт причина, если это не цвет вина карточки (третий круг проверки 25.09).

        * Блюдо, у которого терпкое вино спорит с рыбой (`FISH_RULE` срабатывает на нём: рыба и
          морепродукты), а вино карточки красное или оранжевое, — белое и розовое, тихие и
          игристые: и после «Танины спорят с рыбой», и после «Сладость спорит с рыбой» у Кагора.
        * «Помощнее» к блюду, где белок и жир связывают танины (`PROTEIN_RULE`: шашлык, стейк,
          гусь), у вина карточки не красного цвета — красное: Шардоне к шашлыку из баранины —
          помощнее и красное, а не белое поплотнее.

        Иначе `None`: первыми — вина цвета и игристости вина карточки, как у «Сомелье у полки».
        """
        color = anchor.color
        if color in TANNIC_COLOURS and self._dish_fires(dish_id, FISH_RULE):
            return LOW_TANNIN_COLOURS
        if want == "fuller" and color != RED and self._dish_fires(dish_id, PROTEIN_RULE):
            return (RED,)
        return None

    def _dish_fires(self, dish_id: str, rule_id: str) -> bool:
        """Срабатывает ли правило у блюда хоть с одним вином — значит, условие блюда верно."""
        key = (dish_id, rule_id)
        if key not in self._fires:
            self._fires[key] = any(
                rule_id in pair.plus or rule_id in pair.minus
                for entry in self.data.pairs.values()
                if (pair := entry.dishes.get(dish_id)) is not None
            )
        return self._fires[key]

    def _anchor_sugar(self, anchor: Anchor) -> str | None:
        """Сахар вина карточки для подборки от причины: по карточке (`Anchor.sugar`: выгрузка или
        «сладкое вино» в «Описании»), а без него у вина, возможно сладкого по названию
        (`Anchor.sweet_name`), — сладкое, как судят правила (`rule_sugar` сборки): подборка «суше»
        после оговорки о рыбе — вина, сухие по карточке."""
        if anchor.sugar is not None:
            return anchor.sugar
        return "sladkoe" if anchor.sweet_name else None

    def _goes_with_dish(self, dish_id: str) -> Callable[[RecoWine], bool]:
        pairs = self.data.pairs

        def accept(wine: RecoWine) -> bool:
            entry = pairs.get(wine.slug)
            return entry is not None and entry.pair(dish_id).verdict == "yes"

        return accept

    def _goes_with_food(self, food: str) -> Callable[[RecoWine], bool]:
        pairs = self.data.pairs
        dishes = self.food_dishes.get(food, ())

        def accept(wine: RecoWine) -> bool:
            entry = pairs.get(wine.slug)
            return entry is not None and any(entry.pair(d).verdict == "yes" for d in dishes)

        return accept

    def _group_verdict(self, anchor: Anchor, food: str) -> str | None:
        """Вердикт вина к группе блюд: лучший вердикт среди её блюд."""
        verdicts = {self.pair(anchor, d).verdict for d in self.food_dishes.get(food, ())}
        for verdict in ("yes", "caveat", "no"):
            if verdict in verdicts:
                return verdict
        return None

    def _nearness(self, facts: Facts, wine: RecoWine) -> tuple:
        """Ключ близости «Сомелье у полки»: сорта → сахар в ступенях → крепость → регион → slug."""
        ours = SUGAR_STEPS.get(facts.sugar or "")
        theirs = SUGAR_STEPS.get(wine.sugar or "")
        gap = abs(ours - theirs) if ours is not None and theirs is not None else UNKNOWN_SUGAR_GAP
        return (
            -jaccard(facts.grapes, wine.grapes) if facts.grapes else 0.0,
            gap,
            abv_gap(facts.alcohol, self.source.catalog.alcohol(wine.slug)),
            facts.region is not None and wine.region != facts.region,
            wine.slug,
        )

    def _select(
        self,
        anchor: Anchor,
        want: str,
        accept: Callable[[RecoWine], bool] | None,
        shown: Sequence[str],
        *,
        near: bool = False,
        limit: int = LIMIT,
        reason: Reason | None = None,
        colours: Sequence[str] | None = None,
    ) -> list[_Pick]:
        """Вина других виноделен по правилу «Сомелье у полки» (`Shelf.answer`) с фильтром блюда.

        `want` — направление чипа (`softer`, `fresher`) или `none`; с `reason` — направление
        причины «−» пары, и сдвиг считает `Shelf.reason_move` только по доказательствам причины.
        `none` без `near` — сахар в пределах ступени, как у похожих; с `near` (альтернативы к
        блюду) сахар не ограничен: сухому к десерту подходят сладкие. Сначала 60 ближайших того
        же стиля, меньше `limit` — весь пул, вина того же стиля первыми. Стиль — цвет и
        игристость вина карточки, а если причина зовёт в другой стиль (`colours`, у `reason` —
        `reason.colours`), — эти цвета, тихие и игристые. Одна винодельня — одно место.
        """
        wine = anchor.wine
        if wine is None:
            return []
        catalog = self.source.catalog
        shelf = self.source.shelf
        facts = Facts.of_wine(wine, catalog.alcohol(anchor.slug))
        skip = set(shown)
        if reason is not None:
            colours = reason.colours
        anchor_sugar = self._anchor_sugar(anchor) if reason is not None else None

        def candidates(pool: Iterable[RecoWine]) -> list[tuple[RecoWine, Move]]:
            out = []
            for other in pool:
                if other.slug in skip or excluded(other, facts):
                    continue
                if accept is not None and not accept(other):
                    continue
                move: Move | None
                if reason is not None:
                    move = shelf.reason_move(reason.want, reason.by, wine, other, anchor_sugar)
                elif want != "none":
                    move = shelf.move(want, wine, other)
                elif near or sugar_close(facts.sugar, other.sugar):
                    move = NO_MOVE
                else:
                    move = None
                if move is not None:
                    out.append((other, move))
            return out

        if colours:
            styles = tuple(colours)
            near_wines: Iterable[RecoWine] = (
                other for color in styles for other in catalog.style_pool(color, None)
            )

            def style_key(other: RecoWine) -> tuple:
                return (other.color not in styles,)

        else:
            near_wines = catalog.style_pool(wine.color, wine.sparkling)

            def style_key(other: RecoWine) -> tuple:
                return (other.color != wine.color, other.sparkling != wine.sparkling)

        near_pool = candidates(near_wines)
        near_pool = sorted(near_pool, key=lambda item: self._nearness(facts, item[0]))[:NEAR_POOL]
        if len(near_pool) >= limit:
            ranked = sorted(
                near_pool, key=lambda item: (item[1].tier, self._nearness(facts, item[0]))
            )
        else:
            ranked = sorted(
                candidates(catalog.pool),
                key=lambda item: (
                    *style_key(item[0]),
                    item[1].tier,
                    self._nearness(facts, item[0]),
                ),
            )
        moves = {other.slug: move for other, move in ranked}
        chosen = one_per_winery([other for other, _ in ranked], limit)
        return [_Pick(other, moves[other.slug], ()) for other in chosen]

    def _plain(
        self,
        anchor: Anchor,
        accept: Callable[[RecoWine], bool] | None,
        shown: Sequence[str],
        limit: int = LIMIT,
    ) -> list[_Pick]:
        """Обычная сортировка: та же категория (цвет, игристость, сахар) по названию."""
        wine = anchor.wine
        if wine is None:
            return []
        facts = Facts.of_wine(wine, self.source.catalog.alcohol(anchor.slug))
        skip = set(shown)
        wines = sorted(
            (
                other
                for other in self.source.catalog.style_pool(wine.color, wine.sparkling)
                if other.slug not in skip
                and not excluded(other, facts)
                and (wine.sugar is None or other.sugar == wine.sugar)
                and (accept is None or accept(other))
            ),
            key=name_key,
        )[:limit]
        return [_Pick(other, NO_MOVE, ()) for other in wines]

    def _expanded(self, anchor: Anchor, picks: Sequence[_Pick]) -> bool:
        wine = anchor.wine
        return wine is not None and any(
            p.wine.color != wine.color or p.wine.sparkling != wine.sparkling for p in picks
        )

    def _reasons(self, anchor: Anchor, pick: _Pick, dish_reason: str | None) -> tuple[str, ...]:
        """Объяснение плитки — порядок `Shelf._pick`: блюдо → направление → цвет → факты."""
        if pick.reasons:
            return pick.reasons[:MAX_REASONS]
        wine = anchor.wine
        assert wine is not None
        facts = Facts.of_wine(wine, self.source.catalog.alcohol(anchor.slug))
        reasons: list[str] = [dish_reason] if dish_reason else []
        reasons.extend(pick.move.reasons)
        shown = facts
        other = pick.wine
        if facts.color and other.color and other.color != facts.color:
            # Цвет, который уже назван сдвигом («Помягче: белое, а не красное»), не повторяется.
            if not pick.move.colour:
                reasons.append(f"{other.color}, а не {facts.color.lower()}")
            shown = replace(shown, color=None)
        if pick.move.sugar:
            shown = replace(shown, sugar=None, sparkling=None)
        if pick.move.grape:
            shown = replace(shown, grapes=())
        alcohol = self.source.catalog.alcohol(other.slug)
        reasons.extend(explain(shown, other, alcohol, strength=not pick.move.strength))
        return tuple(reasons[:MAX_REASONS])

    def _pill(
        self, anchor: Anchor, pick: _Pick, want: str, reason: Reason | None = None
    ) -> str | None:
        """Главное отличие короткой строкой: «танины мягче — по сорту», «полусухое, а не сухое».

        У подборки от причины — факт, который отвечает причине, в порядке её доказательств
        (`Reason.by`): у «Крепость перекрывает блюдо» — «крепость ниже», у «Танины спорят с
        рыбой» — «белое, а не красное» или «танины мягче — по сорту».
        """
        wine, other, move = anchor.wine, pick.wine, pick.move
        if wine is None:
            return None
        if reason is not None:
            return self._reason_pill(anchor, pick, reason)
        if move.catalog:
            if move.sugar and wine.sugar and other.sugar:
                return f"{SUGAR_WORDS[other.sugar]}, а не {SUGAR_WORDS[wine.sugar]}"
            if move.strength:
                return "крепость выше" if want == "fuller" else "крепость ниже"
        if move.grape:
            if want == "fresher":
                return "кислотность выше — по сорту"
            if wine.color in ("Белое", "Розовое"):
                return "кислотность ниже — по сорту"
            return "танины мягче — по сорту"
        if want == "replace":
            return next((r for r in pick.reasons if ", а не " in r), None)
        if wine.color and other.color and other.color != wine.color:
            return f"{other.color.lower()}, а не {wine.color.lower()}"
        return None

    def _reason_pill(self, anchor: Anchor, pick: _Pick, reason: Reason) -> str | None:
        """Подпись плитки подборки от причины: первое доказательство причины, которое есть у
        сдвига вина (`shelf.reason_move`)."""
        wine, other, move = anchor.wine, pick.wine, pick.move
        assert wine is not None
        for kind in reason.by:
            if kind == "sugar" and move.sugar and other.sugar:
                if anchor.sugar:
                    return f"{SUGAR_WORDS[other.sugar]}, а не {SUGAR_WORDS[anchor.sugar]}"
                return f"{SUGAR_WORDS[other.sugar]} по карточке"
            if kind == "strength" and move.strength:
                return "крепость выше" if reason.want == "fuller" else "крепость ниже"
            if kind == "colour" and move.colour and wine.color and other.color:
                return f"{other.color.lower()}, а не {wine.color.lower()}"
            if kind == move.axis:
                return REASON_PILLS.get((reason.want, kind))
        return None

    def _tiles(
        self,
        draft: Draft,
        anchor: Anchor,
        picks: Sequence[_Pick],
        order: str,
        want: str,
        dish_reason: str | None,
        *,
        reason: Reason | None = None,
    ) -> None:
        """Плитки подборки, сравнение для шкал и «розы ветров», показанные вина в контекст."""
        plain = order == "plain"
        tiles = []
        for pick in picks:
            reasons = () if plain else self._reasons(anchor, pick, dish_reason)
            tile = dict(self.source.tile(pick.wine, reasons))
            tile["reasons"] = [] if plain else list(tile.get("reasons") or [])
            pill = None if plain else self._pill(anchor, pick, want, reason)
            tile["pill"] = pill if pill and check(pill).clean else None
            tile["want_source"] = (
                pick.move.source if not plain and want not in ("none", "replace") else None
            )
            tiles.append(tile)
        draft.wines = tiles
        if not tiles:
            draft.wines_title = None
        shown = [*draft.context.shown, *(tile["slug"] for tile in tiles)]
        draft.context = replace(draft.context, shown=tuple(dict.fromkeys(shown))[-MAX_SHOWN:])
        if plain or not tiles:
            draft.compare = None
            if not draft.basis:
                draft.basis = [t.BASIS_CARD] if tiles else []
            return
        axes = self._compare_axes(anchor, want, reason)
        draft.compare = {
            "anchor": {
                "slug": anchor.slug,
                "name": anchor.name,
                "profile": profile_items(anchor.shown),
            },
            "axes": axes,
            "wines": [
                {"slug": tile["slug"], "profile": profile_items(self.shown_of(tile["slug"]))}
                for tile in tiles
            ],
            "overlay": self._overlay(anchor, [tile["slug"] for tile in tiles]),
        }
        if not draft.basis:
            basis = [t.BASIS_PICK, t.BASIS_CARD]
            if any(tile["want_source"] == GRAPE for tile in tiles):
                basis.append(t.BASIS_GRAPE)
            draft.basis = basis

    # -------------------------------------------------------------- наложение на «розу»
    def _distances(self, anchor: Anchor, slugs: Sequence[str]) -> list[tuple[float, int]]:
        """Отличие каждого вина от вина карточки: сумма разниц по общим осям и сколько из них —
        по сорту у обоих.

        Общие оси — те, что у обоих вин из карточки или по сорту; оценка по стилю не в счёт: у
        двух вин одного стиля без сорта в приорах она одна и та же и отличия не покажет. Разница —
        в значениях, как их рисует «роза» (`round(ось / 5, 2)`).
        """
        own = anchor.profile
        out = []
        for slug in slugs:
            other = self.source.shelf.profiles.get(slug)
            total, by_grape = 0.0, 0
            for axis in AXES if other is not None else ():
                mine, theirs = own.source(axis), other.source(axis)
                if mine in COMPARED and theirs in COMPARED:
                    by_grape += mine == theirs == GRAPE
                    total += abs(round(own.axis(axis) / 5, 2) - round(other.axis(axis) / 5, 2))
            out.append((round(total, 2), by_grape))
        return out

    def _overlay(self, anchor: Anchor, slugs: Sequence[str]) -> str | None:
        """Вино для наложения на «розу ветров»: сильнее всех отличается от вина карточки по общим
        осям (при равенстве — первое в подборке). Все совпали или сравнить не по чему — `null`:
        одна обводка с серой полосой под ней ничего бы не показала (проверка страницы 24.09)."""
        distances = self._distances(anchor, slugs)
        best = max(range(len(slugs)), key=lambda i: (distances[i][0], -i), default=None)
        if best is None or distances[best][0] <= 0:
            return None
        return slugs[best]

    def _same_profile(self, anchor: Anchor, slugs: Sequence[str]) -> bool:
        """Все похожие совпали с вином карточки по общим осям, и у каждого есть общая ось по сорту:
        тогда «профиль по сорту и карточке у них такой же» — правда, а не молчание данных."""
        distances = self._distances(anchor, slugs)
        return bool(slugs) and all(by_grape and total <= 0 for total, by_grape in distances)

    def _compare_axes(self, anchor: Anchor, want: str, reason: Reason | None = None) -> list[str]:
        """Оси мини-шкалы под плитками (§1): у подборки от причины — ось её доказательства
        (`Reason.axis`: тело, крепость, дуб, танины, кислотность, сладость), у чипа «помягче» —
        танины или кислотность, у «посвежее» — кислотность, у «послаще» — сладость."""
        if reason is not None:
            axis = reason.axis
        elif want == "softer":
            axis = "acidity" if anchor.color in ("Белое", "Розовое") else "tannin"
        elif want == "fresher":
            axis = "acidity"
        elif want == "sweeter":
            axis = "sweetness"
        else:
            return []
        return [axis] if anchor.profile.source(axis) is not None else []

    # -------------------------------------------------------------- мелочи
    def _rule_text(self, rule_ids: Sequence[str]) -> str | None:
        """Текст сильнейшего правила: по модулю веса, при равенстве — по id."""
        rules = [self.data.rules[rid] for rid in rule_ids if rid in self.data.rules]
        if not rules:
            return None
        rules.sort(key=lambda rule: rule_order(rule.id, rule.weight))
        return rules[0].text

    def _pair_basis(self, dishes: Sequence[Mapping[str, Any]]) -> list[str]:
        """«правила сочетаний» и чем подтверждены чипы: карточкой или сортом."""
        sources = {chip["source"] for dish in dishes for chip in (*dish["plus"], *dish["minus"])}
        basis = [t.BASIS_PAIRS]
        if "catalog" in sources:
            basis.append(t.BASIS_CARD)
        if "grape" in sources:
            basis.append(t.BASIS_GRAPE)
        return basis

    # -------------------------------------------------------------- пакет
    def _package(self, draft: Draft, anchor: Anchor, order: str) -> FactsPackage:
        notice = t.NOTICE_149 if draft.wines and order == "reco" else None
        voiced = (
            draft.voiced and order == "reco" and voice_fits(draft.intent, draft.dishes, draft.wines)
        )
        public = {
            "slug": anchor.slug,
            "intent": draft.intent,
            "order": order,
            "verdict_template": draft.verdict,
            "detail": draft.detail,
            "basis": draft.basis,
            "dishes": draft.dishes,
            "wines_title": draft.wines_title if draft.wines else None,
            "wines": draft.wines,
            "compare": draft.compare,
            "serve": draft.serve,
            "question": draft.question,
            "refusal": draft.refusal,
            "chips": draft.chips,
            "notice_149": notice,
            "voice": voiced and self.live,
            "context": replace(draft.context, intent=draft.intent).as_dict(),
        }
        lines = self._voice_lines(draft, anchor)
        texts = [draft.verdict, *lines, *draft.rule_texts]
        return FactsPackage(
            public=public,
            verdict_template=draft.verdict,
            voiced=voiced,
            intent_label=draft.intent_label or draft.intent,
            voice_input=tuple(lines),
            allowed_entities=self._entities(draft, anchor),
            allowed_numbers=frozenset(
                match.replace(".", ",") for text in texts for match in _NUMBER_RE.findall(text)
            ),
            allowed_text="\n".join(texts),
        )

    def _voice_lines(self, draft: Draft, anchor: Anchor) -> list[str]:
        """Строки фактов для модели (§6.4): никаких вопроса, описания и чисел профиля."""
        grapes = ", ".join(anchor.grape_names) if anchor.grape_names else "сорт не указан"
        lines = [
            f"Вино: {anchor.name}",
            f"Винодельня: {anchor.winery}",
            f"Регион: {anchor.region}" if anchor.region else "",
            f"Стиль: {t.style_words(anchor.style_label, anchor.sparkling)}",
            f"Сорта: {grapes}",
        ]
        if anchor.serve is not None:
            lines.append(f"Подача: {t.serve_range(anchor.serve.temperature)}")
        for dish in draft.dishes:
            plus = "; ".join(chip["text"] for chip in dish["plus"])
            minus = "; ".join(chip["text"] for chip in dish["minus"])
            line = f"Блюдо: {dish['name']} — {VERDICT_WORDS[dish['verdict']]}"
            line += f"; за: {plus}" if plus else ""
            line += f"; против: {minus}" if minus else ""
            lines.append(line)
        if draft.wines_title and draft.wines:
            lines.append(f"Подборка: {draft.wines_title}")
        wine_lines = []
        for tile in draft.wines:
            parts = [
                tile["name"],
                tile["winery"],
                tile.get("region") or "",
                (tile.get("style_label") or "").lower(),
                ", ".join(tile.get("grapes") or []),
                tile.get("pill") or "",
            ]
            wine_lines.append("Вино подборки: " + " · ".join(p for p in parts if p))
        lines = [line for line in lines if line]
        total = sum(len(line) + 1 for line in lines)
        while wine_lines and total + sum(len(line) + 1 for line in wine_lines) > MAX_VOICE_CHARS:
            wine_lines.pop()
        return [*lines, *wine_lines]

    def _entities(self, draft: Draft, anchor: Anchor) -> frozenset[str]:
        """Имена пакета: вино, винодельни, регионы, сорта, блюда — модели можно только их."""
        names: set[str] = {anchor.name, anchor.winery, anchor.region, *anchor.grape_names}
        names.update(grape_label(code) for code in anchor.grapes)
        for dish in draft.dishes:
            info = self.data.dishes.get(dish["id"])
            names.update((dish["name"], info.dative) if info else (dish["name"],))
        for tile in draft.wines:
            names.update((tile["name"], tile["winery"], tile.get("region") or ""))
            names.update(tile.get("grapes") or [])
        names.discard("")
        return frozenset({*names, *(name.lower() for name in names)})


def _carry(context: Context, intent: str) -> Context:
    """Контекст следующего хода: намерение этого ответа, показанные вина — прежние."""
    return Context(intent=intent, shown=context.shown)


def _term_chip(topics: Mapping[str, Any], topic_id: str) -> dict[str, Any] | None:
    """Чип темы справочника, если тема есть в данных (в заглушках их четыре из 32)."""
    topic = topics.get(topic_id)
    return t.term_chip(topic_id, topic.name) if topic is not None else None


class _NeutralPair:
    verdict = "neutral"
    plus: tuple[str, ...] = ()
    minus: tuple[str, ...] = ()


_NEUTRAL = _NeutralPair()
