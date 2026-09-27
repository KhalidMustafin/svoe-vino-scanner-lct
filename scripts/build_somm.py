"""Сборка данных сомелье `data/somm/*.json` — на CPU, из выгрузки организатора и правил «Лозы».

Договор — `docs/api-sommelier.md`, §7; загрузчик сервиса — `app/recommend/somm_data.py`.

Входы (только чтение, решение 24.09 — ничего с портала):

    --csv       strapi_output0709.csv организатора: название, винодельня, «Категория» (цвет),
                регион, «Сорт винограда»; 4 147 строк, 2 103 slug (повторы строк одинаковы)
    --gt        gt_tokens.jsonl — производная выгрузки: коды сортов таксономии сканера,
                крепость и год из slug и названия. Поля живого портала (`published`,
                `live_category`, `fields.sugar` с `src: live_api`) сборка не читает
    --groups    wine_groups.json — наш словарь групп: каноническая позиция группы (для пула и
                счёта вин в портретах сортов)
    app/recommend/grape_priors.json — приоры сортов, профиль по 8 осям (`app/recommend/build.py`)
    tools/loza_engines/ — движок сочетаний, справочники «Лозы» и наша накладка `overlay.json`

Снимка портала `data/portal/`, данных Роскачества, цен и `expert_score` среди входов нет;
«Описание» выгрузки читается только ради сахара («сладкое вино» и «сухое», `rule_sugar`) и в
файлы не идёт. Факты вина — по правилам договора «после поиска», §3, теми же функциями, что у
карточки сервиса (`app/recommend/catalog.py`): сахар — из названия, затем из slug; игристость —
брют-семейство или слово игристого в названии или slug; крепость — из slug, без года урожая.
Правилам сочетаний и подаче сахар без названия и slug даёт ещё «Описание»: «Сладкое розовое
вино» — сладкое по карточке, а не догадка (`card_sugar`, проверка `acc-somm3` 25.09).

Что делает:

1. Профиль каждого из 2 103 вин — `build_profile` по фактам организатора; подача — первое
   подошедшее правило нашей таблицы (`SERVE_RULES`, та же функция `match_serve`, что у сервиса).
2. Все пары «вино × блюдо» (2 103 × 82) считает движок «Лозы» в подпроцессе
   (`tools/loza_engines/run.py`, `python -I -S`): пакет «Лозы» зовётся `app`, как у сканера.
3. Вердикт пары и `top` — по §7.1 и фильтрам качества после выборочной проверки пар (см. `top_dishes`); сахар
   вина неизвестен, а блюдо сладкое — `neutral` без правил (`sugar_decides`); название
   креплёного, десертного или мускатного вина без сахара — для правил возможно сладкое
   (`rule_sugar`, `sweet_by_name`): к рыбе — оговорка с условной фразой, а не «скорее нет»
   (`HEDGE_RULES`); название оранжевого вина или «сухое» в «Описании» догадку снимают, а
   «сладкое вино» в «Описании» делает вино сладким по карточке (`sugar_by_card`).
4. Справочник «Лозы» чистится по 38-ФЗ (`TOPIC_REWRITES`), подсказки-продолжения становятся
   чипами сканера (`FOLLOW_UP_CHIPS`), портреты сортов считаются по приорам и выгрузке.
5. Каждая строка каждого файла проходит `content_filter.check`, стоп-лист и запрет превосходных
   степеней (`legal_violations`); нарушение — ошибка сборки, а не тихий пропуск. Так же подписи и
   фразы правил: слово о блюде («жирное», «лёгкое», «бульон», «рыба») — правда о каждом блюде,
   к которому правило сработало (`DISH_CLAIMS`, `false_claims`).

Детерминированно: без времени и случайности в файлах, `--check` пересобирает в памяти и сверяет
с диском байт в байт (код выхода 1 при расхождении).

    python scripts/build_somm.py --csv "<Датасет>/strapi_output0709.csv"
    python scripts/build_somm.py --csv "<Датасет>/strapi_output0709.csv" --check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import GRAPE_SYNONYMS, grape_label
from app.recommend.build import PRIORS_PATH, REGION_CODES, GrapePrior, build_profile, load_priors
from app.recommend.catalog import (
    alcohol_of,
    card_sugar,
    organizer_sparkling,
    organizer_sugar,
    slug_abv,
    sugar_of_slug,
    sweet_name,
)
from app.recommend.content_filter import check
from app.recommend.profile import AXES, RussianPGI, StyleProfile, WineKind
from app.recommend.somm_data import (
    SOMM_FILES,
    ServeMatch,
    ServeRule,
    load_somm_data,
    match_serve,
    rule_order,
)

ROOT = Path(__file__).resolve().parents[1]
LOZA_DIR = ROOT / "tools" / "loza_engines"
REFERENCE = LOZA_DIR / "reference"

VERSION = 1

# ------------------------------------------------------------------ факты выгрузки
#: Цвет выгрузки → код цвета «Лозы» (`WineColor`).
COLOR_CODES: dict[str, str] = {
    "Белое": "white",
    "Красное": "red",
    "Розовое": "rose",
    "Оранжевое": "orange",
}

#: Сорта-заглушки выгрузки: не сорт, а группа.
PLACEHOLDER_GRAPES = frozenset({"белые сорта винограда", "красные сорта винограда"})

#: Сахар, которому по фильтру качества положены только десерты, фрукты, сыр и острое.
SWEET_SUGARS = frozenset({"sladkoe", "polusladkoe"})
#: Сахар правил у вина без сахара в выгрузке с названием креплёного или десертного (`rule_sugar`).
SWEET_NAME_SUGAR = "sladkoe"

SUGAR_WORDS: dict[str, str] = {
    "brut_nature": "брют натюр",
    "extra_brut": "экстра брют",
    "brut": "брют",
    "suhoe": "сухое",
    "polusuhoe": "полусухое",
    "polusladkoe": "полусладкое",
    "sladkoe": "сладкое",
}


@dataclass(frozen=True, slots=True)
class WineFacts:
    """Факты одной позиции выгрузки — только то, что написал организатор, и наши правила."""

    slug: str
    name: str
    winery: str
    region: str
    color: str
    grapes: tuple[str, ...]  # подписи выгрузки без заглушек
    codes: tuple[str, ...]  # коды сортов таксономии сканера
    sugar: str | None  # сахар карточки «после поиска» (§3): название, затем slug
    sparkling: bool
    abv: float | None
    year: int | None
    canonical: bool
    #: «Описание» выгрузки: только ради сахара — «сладкое вино» в нём делает вино сладким по
    #: карточке (`sugar_by_card`), «сухое» или «сухость» снимает догадку «возможно сладкое по
    #: названию» (`sweet_by_name`). В файлы данных не идёт.
    description: str = ""


def clean(value: object) -> str:
    """Строка выгрузки без переводов строк и двойных пробелов."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def slug_sugar(slug: str) -> str | None:
    """Последнее слово сахара в slug: «…-igristoe-sladkoe-krasnoe-bryut-125» — брют."""
    return sugar_of_slug(slug)


def sugar_of(name: str, slug: str) -> str | None:
    """Правило сахара договора: класс, названный в «Название вина», затем slug, иначе `None`."""
    return organizer_sugar(name, slug)


def abv_of(values: Any, year: Any = None) -> float | None:
    """Крепость из `fields.abv.value`: одно число — оно, диапазон — меньшее, вне 3–25 — нет.

    Число, равное двум последним цифрам года урожая, — год, а не крепость (`slug_abv`).
    """
    return alcohol_of(slug_abv(values, year)).value


def load_facts(csv_path: Path, gt_path: Path, groups_path: Path) -> list[WineFacts]:
    """Факты всех позиций выгрузки по slug, по порядку slug."""
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(clean(row["Slug"]), row)
    gt = {}
    with gt_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                gt[item["slug"]] = item
    missing = sorted(set(rows) - set(gt))
    if missing:
        raise ValueError(f"{gt_path}: нет {len(missing)} slug выгрузки, например {missing[:3]}")
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    canonical = {str(group["canonical"]) for group in groups.values()}

    facts = []
    for slug in sorted(rows):
        row, fields = rows[slug], gt[slug]["fields"]
        name = clean(row["Название вина"])
        grapes = tuple(
            label
            for label in (clean(part) for part in row["Сорт винограда"].split(","))
            if label and label.lower() not in PLACEHOLDER_GRAPES
        )
        sugar = sugar_of(name, slug)
        year = fields.get("year", {}).get("value")
        facts.append(
            WineFacts(
                slug=slug,
                name=name,
                winery=clean(row["Винодельня"]),
                region=clean(row["Регион"]),
                color=clean(row["Категория"]),
                grapes=grapes,
                codes=tuple(fields.get("grape", {}).get("codes") or ()),
                sugar=sugar,
                sparkling=organizer_sparkling(sugar, name, slug),
                abv=abv_of(fields.get("abv", {}).get("value") or (), year),
                year=int(year) if isinstance(year, int) else None,
                canonical=slug in canonical,
                description=clean(row.get("Описание")),
            )
        )
    return facts


def sugar_by_card(wine: WineFacts) -> str | None:
    """Сахар по карточке (`catalog.card_sugar`): сахар выгрузки, а без него — «сладкое», если
    «Описание» организатора прямо называет вино сладким и нигде — сухим. На выгрузке 25.09 — одно
    вино: «Мускат позднего сбора розовый» («Сладкое розовое вино»; проверка `acc-somm3` 25.09).
    """
    return card_sugar(wine.sugar, wine.description)


def sweet_by_name(wine: WineFacts) -> bool:
    """«Возможно сладкое по названию» (`catalog.sweet_name`): сахара нет ни в выгрузке, ни в
    «Описании» (`sugar_by_card`), название — креплёного, десертного или мускатного вина, а ни
    название («Мускат Оранж»), ни «Описание» («абсолютная сухость») не говорят, что вино сухое.
    24 позиции выгрузки из 27 с таким названием.
    """
    return sweet_name(wine.name, wine.sugar, wine.description)


def rule_sugar(wine: WineFacts) -> str | None:
    """Сахар, по которому судят правила сочетаний: сахар по карточке (`sugar_by_card`: выгрузка,
    затем «сладкое вино» в «Описании»), а без него у вина, возможно сладкого по названию
    (`sweet_by_name`), — сладкое.

    Проверка 25.09, вечер: 25 таких вин («Портвейн Крымский», «Массандра Херес», «Grand Dessert
    Nectar», мускаты) профиль считал сухими, и к ним «подходили» устрицы, сельдь и сёмга — 349 пар
    с рыбой, хотя подача у них уже по правилу `sweet_name` (10–14 °C, как у сладких), а «к чему
    подать» — только сыр и орехи. Теперь для правил о сахаре, фильтров «к чему подать» и приёмки
    «сладкое без рыбы» они — сладкие; чип такого правила помечен «типично для стиля»: сахар взят
    по названию, а не из карточки (`answers.Sommelier.chip_source`).

    Проверка третьего круга 25.09: это догадка, а не факт. К рыбе она даёт не «скорее нет» с фразой
    о сахаре, а оговорку «если вино сладкое…» (`maybe_sweet_wine_on_fish`, `HEDGE_RULES`); движок
    видит её полем `sugar_by_name` (`engine_request`). Проверка `acc-somm3`: «Сладкое розовое вино»
    в «Описании» — не догадка, а сахар по карточке: к рыбе — «скорее нет», как у сладкого.
    """
    sugar = sugar_by_card(wine)
    if sugar is not None:
        return sugar
    return SWEET_NAME_SUGAR if sweet_by_name(wine) else None


def profile_of(wine: WineFacts, priors: Mapping[str, GrapePrior]) -> StyleProfile:
    """Профиль по 8 осям — `build_profile` на фактах организатора; сахар — `rule_sugar`."""
    sugar = rule_sugar(wine)
    return build_profile(
        wine.codes,
        Color(wine.color) if wine.color in COLOR_CODES else None,
        SugarClass(sugar) if sugar else None,
        WineKind.SPARKLING if wine.sparkling else WineKind.STILL,
        priors,
        region=REGION_CODES.get(wine.region, RussianPGI.OTHER),
        name=wine.name,
        vintage=wine.year,
        abv=wine.abv,
    )


# ------------------------------------------------------------------ подача
#: Наша таблица подачи (§7.3): первое подошедшее правило по порядку. Основание — тема
#: `serve_temp` справочника «Лозы» (`reference/knowledge.json`): «Игристые и лёгкие белые подают
#: при 6–8 °C, плотные белые и розовые — при 8–12 °C, лёгкие красные — при 12–14 °C, плотные
#: красные — при 16–18 °C… Сладкие и креплёные вина подают при 10–14 °C»; оранжевые — тема
#: `orange_wine`: «не ледяными, 10–12 °C». Это обычная практика подачи (холод подчёркивает
#: кислотность и прячет спирт, тепло раскрывает аромат); «плотное» — ось `body` профиля от 3,5,
#: а она известна только по сорту, поэтому такие правила помечают подачу `by_grape`. Вино без
#: сахара в выгрузке, но с названием креплёного, десертного или мускатного (`sweet_name`: 24
#: позиции — «Портвейн Крымский», «Массандра Херес», «Grand Dessert Nectar», «Поздний сбор»,
#: мускаты) подаётся как сладкое и креплёное, а не как сухое белое при 6–8 °C или плотное
#: красное при 16–18 °C; правила сочетаний судят его как сладкое (`rule_sugar`). Вино, сладкое
#: по «Описанию» (`sugar_by_card`), подаётся по правилу `sweet`, как сладкое по выгрузке.
SERVE_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "sparkling",
        "when": {"sparkling": True},
        "temperature_c": [6, 8],
        "text": "Игристые подают хорошо охлаждёнными.",
    },
    {
        "id": "sweet",
        "when": {"sugar": ["sladkoe"]},
        "temperature_c": [10, 14],
        "text": "Сладкие вина подают прохладными, но не ледяными.",
    },
    {
        "id": "sweet_name",
        "when": {"sweet_name": True},
        "temperature_c": [10, 14],
        "text": "Креплёные, десертные и мускатные вина подают прохладными, но не ледяными.",
    },
    {
        "id": "orange",
        "when": {"color": ["Оранжевое"]},
        "temperature_c": [10, 12],
        "text": "Оранжевые вина подают прохладными, но не ледяными.",
    },
    {
        "id": "white_full",
        "when": {"color": ["Белое"], "body_min": 3.5},
        "temperature_c": [8, 12],
        "text": "Плотные белые подают чуть теплее лёгких: так раскрывается аромат.",
    },
    {
        "id": "white",
        "when": {"color": ["Белое"]},
        "temperature_c": [6, 8],
        "text": "Лёгкие белые подают охлаждёнными: холод подчёркивает свежесть.",
    },
    {
        "id": "rose",
        "when": {"color": ["Розовое"]},
        "temperature_c": [8, 12],
        "text": "Розовые подают прохладными.",
    },
    {
        "id": "red_full",
        "when": {"color": ["Красное"], "body_min": 3.5},
        "temperature_c": [16, 18],
        "text": "Плотные красные подают при 16–18 °C: тепло раскрывает аромат, а холод сделал бы "
        "танины жёстче.",
    },
    {
        "id": "red",
        "when": {"color": ["Красное"]},
        "temperature_c": [12, 14],
        "text": "Лёгкие красные подают слегка охлаждёнными.",
    },
)


def serve_rules() -> tuple[ServeRule, ...]:
    """`SERVE_RULES` в записях загрузчика — чтобы сборка и сервис звали одну `match_serve`."""
    out = []
    for rule in SERVE_RULES:
        when = rule["when"]
        out.append(
            ServeRule(
                id=rule["id"],
                temperature_c=(rule["temperature_c"][0], rule["temperature_c"][1]),
                text=rule["text"],
                sparkling=when.get("sparkling"),
                sugar=frozenset(when["sugar"]) if "sugar" in when else None,
                color=frozenset(when["color"]) if "color" in when else None,
                body_min=when.get("body_min"),
                sweet_name=when.get("sweet_name"),
            )
        )
    return tuple(out)


def serve_of(wine: WineFacts, profile: StyleProfile, rules: Sequence[ServeRule]) -> ServeMatch:
    found = match_serve(
        rules,
        color=wine.color,
        sugar=sugar_by_card(wine),
        sparkling=wine.sparkling,
        body=profile.body,
        body_source=profile.source("body"),
        sweet_name=sweet_by_name(wine),
    )
    if found is None:
        raise ValueError(f"{wine.slug}: ни одно правило подачи не подошло ({wine.color})")
    return found


# ------------------------------------------------------------------ правила и блюда
#: Подпись чипа, фраза шаблона (если `explanation_ru` «Лозы» не годится) и поля вина, которые
#: читает условие правила (§7.2). Фразы переписаны там, где у «Лозы» оценочные слова: «лучший
#: спутник», «отлично идёт», «классическое сочетание», «всегда звучит честнее», — и где фраза
#: говорит о блюде больше, чем гарантирует условие блюда (`DISH_CLAIMS`): «жареное и майонезное»
#: у правила только для жареного, «бульонно-грибной вкус» у правила только для супов на бульоне
#: (25.09). Проверка 25.09, вечер: «деликатное» у солёных огурцов, квашеной капусты и щей — условие
#: знает только, что блюдо нежирное и негромкое, поэтому «лёгкое»; «сахар спорит с солью и
#: мясом» у грибов в сметане и «горячее» у окрошки — «несладкое блюдо» (не «сытное»: то же правило
#: срабатывает у овощей на гриле и окрошки, которые шаблон «Вино мощнее блюда» зовёт лёгкими);
#: «дымок с углей» у блинов и хачапури (тег `toast`) — «дымок или поджаристая корочка»; «уксус и
#: соленья» у борща и пасты — «кислинка блюда»; «морская нота» у ухи и строганины — «рыба и
#: морепродукты»; «солёная закуска» у борща и котлет — «солёное»; «без жира и мяса» у куриной
#: грудки — «мало жира»; «мясо сочнее» у красной рыбы — «блюдо сочнее»; «нужно белое, розовое или
#: игристое» у рыбы — «почти без танинов» (подборка к рыбе даёт и лёгкие красные: танины Пино Нуар
#: ниже порога правила). Разная сила вкуса — два правила по направлению (накладка 25.09, вечер):
#: подборка идёт «полегче» или «помощнее» (`app/sommelier/answers.py`, `REASONS`). Третий круг
#: проверки 25.09 — два своих правила сканера: «Сладость спорит с рыбой» (сладкое к рыбе — «скорее
#: нет») и «Сладкое к сыру и орехам» (давняя пара — «да»); их фразы — `explanation_ru` накладки.
RULE_META: dict[str, tuple[str, str | None, tuple[str, ...]]] = {
    "wine_sweeter_than_dish": ("Вино слаще десерта", None, ("sweetness", "acidity")),
    "delicate_dish_needs_light_wine": (
        "Лёгкое к лёгкому",
        (
            "Блюдо лёгкое, и вино ему под стать: свежее, неплотное, без дуба — оно поддерживает "
            "вкус, а не перебивает его."
        ),
        ("body", "acidity", "oak", "tannin"),
    ),
    "acid_cuts_fat": ("Кислотность освежает жирное", None, ("acidity",)),
    "tannin_meets_protein_fat": (
        "Танинам — белок и жир",
        "Белок и жир блюда связывают танины: вино кажется мягче, а блюдо — сочнее.",
        ("tannin", "body"),
    ),
    "spice_needs_sweet_and_calm": (
        "Сладость гасит остроту",
        None,
        ("sweetness", "alcohol", "tannin"),
    ),
    "intensity_match": ("Равны по силе вкуса", None, ("body", "aroma_intensity", "alcohol")),
    "bubbles_cut_fat_and_fry": (
        "Пузырьки освежают жареное",
        (
            "Пузырьки вместе с кислотностью смывают жир и обновляют вкус — жареное сразу "
            "становится легче."
        ),
        ("effervescence", "sweetness", "acidity"),
    ),
    "seafood_loves_mineral_white": (
        "Минеральность к рыбе и морю",
        "Минеральная свежесть вина подчёркивает вкус рыбы и морепродуктов.",
        ("color", "acidity", "tannin", "oak"),
    ),
    "umami_needs_acid_and_fruit": ("Насыщенному — свежесть", None, ("acidity", "tannin")),
    "salt_loves_acidity": ("Кислотность к солёному", None, ("acidity",)),
    "acidity_mirror": ("Свежесть вровень с блюдом", None, ("acidity",)),
    "bubbles_for_salty_snacks": (
        "Игристое к солёному",
        "Игристое освежает после солёного и не спорит с ним.",
        ("effervescence", "sweetness"),
    ),
    "regional_affinity": ("Вино и кухня одних мест", None, ("region",)),
    "oak_smoke_bridge": (
        "Дуб к дымку и корочке",
        (
            "Дымок или поджаристая корочка блюда перекликаются с бочковыми тонами вина — вкус "
            "получается общий."
        ),
        ("oak", "descriptors"),
    ),
    "aroma_bridge": ("Общий ароматический тон", None, ("descriptors",)),
    "sweet_wine_bridges_salt_fat": (
        "Сладость оттеняет солёное",
        "Солёное и жирное рядом с чуть сладким вином дают контраст, и вкус становится объёмнее.",
        ("sweetness", "acidity"),
    ),
    "salt_rescues_umami": ("Соль выручает пару", None, ("tannin", "acidity")),
    "salt_softens_tannin": ("Соль смягчает танины", None, ("tannin",)),
    "serving_temperature_match": ("Температура под стать блюду", None, ("serve",)),
    "autochthon_bonus": (
        "Автохтон к своей кухне",
        "Автохтонный сорт и местная кухня: вино и блюдо из одной земли.",
        ("grapes",),
    ),
    "high_alcohol_on_delicate": (
        "Крепость перекрывает блюдо",
        "Крепкое вино перекрывает негромкий вкус блюда: остаётся в основном спиртовая теплота.",
        ("alcohol",),
    ),
    "intensity_mismatch": (
        "Блюдо насыщеннее вина",
        "Блюдо заметно насыщеннее вина: рядом с ним вино теряется и кажется пустым.",
        ("body", "aroma_intensity", "alcohol"),
    ),
    "wine_overpowers_dish": (
        "Вино насыщеннее блюда",
        None,
        ("body", "aroma_intensity", "alcohol"),
    ),
    "oak_crushes_delicate": (
        "Дуб спорит с лёгким",
        "Дуб в вине спорит с лёгким блюдом: во вкусе остаются в основном ваниль и древесина.",
        ("oak",),
    ),
    "flat_wine_on_fat_dish": ("Мягкому вину тяжело с жирным", None, ("acidity",)),
    "bare_tannin_on_lean_dish": (
        "Танинам не за что зацепиться",
        (
            "Танинам вина здесь не за что зацепиться: в блюде мало жира, и они ощущаются сухо и "
            "жёстко."
        ),
        ("tannin",),
    ),
    "flat_wine_on_sour_dish": (
        "Кислое блюдо гасит вино",
        "Кислинка блюда «съест» мягкое вино: рядом с ней оно станет вялым и плоским.",
        ("acidity",),
    ),
    "semisweet_wine_on_dessert": ("Десерт слаще вина", None, ("sweetness",)),
    "dessert_wine_on_savoury_dish": (
        "Сладость перебивает блюдо",
        (
            "Сладость вина перебивает несладкое блюдо: сахар остаётся сам по себе, а вкус еды "
            "теряется."
        ),
        ("sweetness",),
    ),
    "semisweet_wine_on_savoury_dish": ("Сладость спорит с солёным", None, ("sweetness",)),
    "umami_vs_tannin_clash": (
        "Бульон подчёркивает терпкость",
        (
            "Насыщенный бульон вытягивает из вина горечь: терпкое вино рядом кажется жёстче, чем "
            "оно есть."
        ),
        ("tannin", "oak"),
    ),
    "no_tannin_with_oily_fish": (
        "Танины спорят с рыбой",
        (
            "Терпкое вино с рыбой и морепродуктами даёт неприятный металлический привкус — здесь "
            "нужно вино почти без танинов."
        ),
        ("tannin",),
    ),
    "heavy_wine_on_delicate_dish": (
        "Вино мощнее блюда",
        "Вино слишком мощное для такого лёгкого блюда: рядом с ним оно кажется тяжёлым.",
        ("body", "tannin", "oak"),
    ),
    "sweet_wine_on_savoury_main": (
        "Сладость спорит с блюдом",
        "Сладкое вино рядом с несладким блюдом звучит чужеродно: сахар спорит со вкусом еды.",
        ("sweetness",),
    ),
    "tannin_vs_spice_clash": (
        "Танины разжигают острое",
        (
            "Острое блюдо и терпкое или крепкое вино разжигают друг друга: жжёт сильнее, а вино "
            "кажется грубым и горьким."
        ),
        ("tannin", "alcohol", "oak"),
    ),
    "dry_wine_on_dessert": ("Сухое на фоне десерта резче", None, ("sweetness",)),
    "sweet_wine_on_fish": ("Сладость спорит с рыбой", None, ("sweetness",)),
    "maybe_sweet_wine_on_fish": ("Если сладкое — спорит с рыбой", None, ("sweetness",)),
    "sweet_wine_with_cheese_and_nuts": ("Сладкое к сыру и орехам", None, ("sweetness",)),
}

#: Вес правила «−», от которого пара — жёсткий конфликт и вердикт `no` (§7.1).
HARD_CONFLICT = -3.0
#: Порог оценки «Лозы» (`min_score` в `pair_dishes`).
MIN_SCORE = 0.55
#: Сигмоида оценки движка (`tools/loza_engines/pairing.py`: `_SCORE_SCALE`, `MAX_DISPLAY_SCORE`) —
#: чтобы судить пару без правила-оговорки (`verdict_of`); тест сверяет её с оценками движка.
SCORE_SCALE = 5.0
MAX_DISPLAY_SCORE = 0.96
#: Правила-оговорки: «−» о догадке, а не о факте карточки (проверка третьего круга 25.09). Сахара в
#: карточке нет, сладкое вино только по названию (`sweet_by_name`) — к рыбе условная фраза «если
#: так, сладость будет спорить с рыбой». Вердикт такое правило не решает: пара судится без него, и
#: «да» или `neutral` становятся оговоркой, а «скорее нет» остаётся только по другим правилам «−».
#: Вес −1,0 — меньше по модулю, чем у любого другого «−», поэтому в объяснении правило последнее.
HEDGE_RULES = frozenset({"maybe_sweet_wine_on_fish"})
TOP_LIMIT = 5
#: Меньше блюд в «к чему подать» у вина быть не должно (приёмка сборки, `quality`).
MIN_TOP = 3

#: Категории блюд «Лозы» → подписи (§7.2).
CATEGORIES: dict[str, str] = {
    "soup": "Супы",
    "dumplings": "Пельмени и вареники",
    "savory_pastry": "Пироги и выпечка",
    "cold_appetizer": "Холодные закуски",
    "salad": "Салаты",
    "pickles": "Соленья",
    "seafood": "Рыба и морепродукты",
    "main_meat": "Мясо",
    "main_poultry": "Птица",
    "main_vegetable": "Овощи и гарниры",
    "hot_appetizer": "Горячие закуски",
    "seafood_raw": "Сырая рыба",
    "dessert": "Десерты",
    "sweet_preserve": "Варенье и мёд",
    "fast_food": "Фастфуд",
    "foreign_dish": "Паста, пицца, суши",
    "appetizer": "Сыры",
}

#: Группы `food` для чипов `/shelf` и шага `guided` (§7.2).
FOOD_MEAT_EXTRA = frozenset({"buzhenina", "pastroma", "kholodets"})
FOOD_FISH_EXTRA = frozenset(
    {
        "malosolnaya_semga",
        "solenaya_seld",
        "kopchenaya_skumbriya",
        "ikra_krasnaya",
        "ikra_chernaya",
        "bliny_ikra",
        "seledka_pod_shuboy",
        "ukha",
        "sushi",
        "rolly",
    }
)
DESSERT_CATEGORIES = frozenset({"dessert", "sweet_preserve"})
#: Острое блюдо сладкому вину — только когда правило «Сладость гасит остроту» сработало на этой
#: паре: оно про лёгкую сладость и невысокий градус (сладость 1,5–3,5). Одной остроты блюда
#: (`spice ≥ 3`) мало: ТБА, ледяному вину и сладкому Ркацители доставалась шаверма, причём первой
#: — за «кислотность к солёному» и «сладость оттеняет солёное», а не за остроту.
SPICE_RULE = "spice_needs_sweet_and_calm"


def cheese_or_nuts(dish: Mapping[str, Any]) -> bool:
    """Сырная тарелка (категория `appetizer`) и орехи (холодная закуска с тегом `walnut`) — давние
    пары сладкого и креплёного: те же блюда, что в условии правила накладки
    `sweet_wine_with_cheese_and_nuts` (третий круг проверки 25.09). Сациви и харчо с грецким
    орехом — не закуски, и сюда не входят."""
    category = dish["category"]
    return category == "appetizer" or (
        category == "cold_appetizer" and "walnut" in (dish.get("flavor_tags") or ())
    )


def food_of(dish: Mapping[str, Any]) -> str | None:
    category, dish_id = dish["category"], dish["id"]
    if category == "main_meat" or dish_id in FOOD_MEAT_EXTRA:
        return "meat"
    if category in ("seafood", "seafood_raw") or dish_id in FOOD_FISH_EXTRA:
        return "fish"
    if dish_id == "cheese_plate":
        return "cheese"
    if category in DESSERT_CATEGORIES:
        return "dessert"
    return None


def _name_words(name: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", name.lower().replace("ё", "е")).split()


def families(dishes: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Семья блюда — блюдо с самым коротким названием-началом: «Борщ» у «Борща с пампушками».

    Как `unique_by_dish` «Лозы»: сравниваются слова, а не буквы, поэтому «Икра красная» и
    «Икра чёрная» — разные семьи, а «Шашлык из свинины» и «Шашлык из баранины» — тоже.
    """
    words = {dish["id"]: _name_words(dish["name"]) for dish in dishes}
    out = {}
    for dish_id, mine in words.items():
        best, best_len = dish_id, len(mine)
        for other, theirs in words.items():
            if other != dish_id and len(theirs) < best_len and mine[: len(theirs)] == theirs:
                best, best_len = other, len(theirs)
        out[dish_id] = best
    return out


def pair_score(total: float) -> float:
    """Оценка пары по сумме весов — та же сигмоида с потолком, что у движка «Лозы»."""
    return min(1.0 / (1.0 + math.exp(-total / SCORE_SCALE)), MAX_DISPLAY_SCORE)


def verdict_of(
    score: float,
    plus: Sequence[str],
    minus: Sequence[str],
    weights: Mapping[str, float],
    total: float | None = None,
) -> str:
    """Вердикт пары по §7.1: `neutral` → `no` (жёсткий «−» или оценка ниже 0,55) → `caveat` → `yes`.

    Правило-оговорка (`HEDGE_RULES`) вердикт не решает: пара судится без него — без его веса в
    сумме `total` и без него среди «−». Вышло «скорее нет» по другим правилам «−» — остаётся
    «скорее нет»; иначе — «да, с оговоркой».
    """
    hedges = [rule for rule in minus if rule in HEDGE_RULES]
    if hedges:
        if total is None:
            raise ValueError(f"правило-оговорка {hedges} без суммы весов пары")
        firm = [rule for rule in minus if rule not in HEDGE_RULES]
        base = pair_score(total - sum(weights[rule] for rule in hedges))
        return "no" if firm and verdict_of(base, plus, firm, weights) == "no" else "caveat"
    if not plus and not minus:
        return "neutral"
    if any(weights[rule] <= HARD_CONFLICT for rule in minus) or score < MIN_SCORE:
        return "no"
    return "caveat" if minus else "yes"


def sugar_decides(wine: WineFacts, dish: Mapping[str, Any]) -> bool:
    """Сахар вина неизвестен, а блюдо — десерт, варенье или мёд: пару решает сахар (§7.1).

    Профиль вина без сахара в выгрузке считает сладость как у сухого (`build_profile`), и правила
    о сахаре судили бы по допущению: «Сухое на фоне десерта резче» и «скорее нет» к шоколаду у
    вина, сахара которого в выгрузке нет (проверка 25.09). Такая пара — `neutral`
    без правил: шаблон говорит, что сахара в карточке нет, а не что вино сухое. Название
    креплёного, десертного или мускатного вина — не «сахар неизвестен»: правила судят его как
    сладкое (`rule_sugar`, проверка 25.09, вечер).
    """
    return rule_sugar(wine) is None and dish["category"] in DESSERT_CATEGORIES


def ordered(rules: Iterable[str], weights: Mapping[str, float]) -> list[str]:
    """Правила по убыванию модуля веса, при равенстве — правила о рыбе, затем по `id` (§7.1,
    `rule_order`)."""
    return sorted(rules, key=lambda rule: rule_order(rule, weights[rule]))


@dataclass(frozen=True, slots=True)
class DishPair:
    """Пара, как её видит сборка: сумма весов для порядка, вердикт и правила."""

    total: float
    verdict: str
    plus: tuple[str, ...]
    minus: tuple[str, ...]


def top_dishes(
    wine: WineFacts,
    row: Mapping[str, DishPair],
    dishes: Mapping[str, Mapping[str, Any]],
    family: Mapping[str, str],
) -> list[str]:
    """До пяти блюд «к чему подать» — §7.1 и фильтры качества после выборочной проверки пар 24.09.

    1. Только `yes`: блюдо, против которого сработало хоть одно правило «−», в «к чему подать»
       не попадает (зонд: «гусь» и «шашлык» с чипом «бульон подчёркивает терпкость»).
    2. Сладкое и полусладкое — только десерты и фрукты, варенье и мёд, сырная тарелка, у
       сладкого ещё орехи (`cheese_or_nuts`), и острое, когда сработало правило `SPICE_RULE`
       (зонд: сладкому Мускателю белому доставались сельдь и сёмга «за живую кислотность»; проверка:
       ледяному вину — шаверма). Сахар — `rule_sugar`: креплёное, десертное или мускатное название
       без сахара в выгрузке — сладкое. Сыр и орехи сладкому и креплёному — «да» с третьего круга
       проверки 25.09 (правило накладки `sweet_wine_with_cheese_and_nuts`); до него орехов в списке
       не было, а сыр у сладких был оговоркой «Сладость перебивает блюдо» и в список не попадал.
       Полусладкому орехи в список не добавлены: давняя пара — у сладкого и креплёного.
    3. Сахар неизвестен — без десертов, варенья и мёда: сладость вина не подтверждена.
    4. Порядок — по сумме весов движка, затем по `id`; одна семья блюда — одно место.
    """
    sugar = rule_sugar(wine)
    candidates = []
    for dish_id, pair in row.items():
        if pair.verdict != "yes":
            continue
        dish = dishes[dish_id]
        dessert = dish["category"] in DESSERT_CATEGORIES
        classic = cheese_or_nuts(dish) if sugar == "sladkoe" else dish_id == "cheese_plate"
        if sugar in SWEET_SUGARS and not (dessert or classic or SPICE_RULE in pair.plus):
            continue
        if sugar is None and dessert:
            continue
        candidates.append((-pair.total, dish_id))
    out: list[str] = []
    taken: set[str] = set()
    for _, dish_id in sorted(candidates):
        if family[dish_id] in taken:
            continue
        taken.add(family[dish_id])
        out.append(dish_id)
        if len(out) == TOP_LIMIT:
            break
    return out


# ------------------------------------------------------------------ движок «Лозы»
def engine_request(
    facts: Sequence[WineFacts],
    profiles: Mapping[str, StyleProfile],
    serves: Mapping[str, ServeMatch],
) -> dict[str, Any]:
    """Запрос подпроцессу: вино словарём `WineView` «Лозы». Портальных блюд нет."""
    wines = []
    for wine in facts:
        profile = profiles[wine.slug]
        wines.append(
            {
                "id": wine.slug,
                "profile": {axis: round(profile.axis(axis), 6) for axis in AXES},
                "descriptors": list(profile.descriptors),
                "color": COLOR_CODES.get(wine.color, "white"),
                "kind": "sparkling" if wine.sparkling else "still",
                "region": REGION_CODES.get(wine.region, RussianPGI.OTHER).value,
                "grapes": list(wine.codes),
                "serve_temp_c": list(serves[wine.slug].rule.temperature_c),
                # Сладость профиля — догадка по названию, а не сахар карточки (`sweet_by_name`).
                "sugar_by_name": sweet_by_name(wine),
            }
        )
    return {"wines": wines, "overlay": True}


def run_engine(request: Mapping[str, Any], loza_dir: Path = LOZA_DIR) -> dict[str, Any]:
    """Движок «Лозы» в отдельном процессе: `python -I -S`, свой `sys.path`, без `app` сканера."""
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(loza_dir / "run.py")],
        input=json.dumps(request, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "движок «Лозы» упал: " + completed.stderr.decode("utf-8", "replace")[-2000:]
        )
    return json.loads(completed.stdout.decode("utf-8"))


# ------------------------------------------------------------------ справочник «Лозы»
#: Подсказки-продолжения «Лозы» → чипы сканера (§4.2). `None` — выбросить: рейтинги, «лучшие»,
#: «что купить», аналоги зарубежных стилей и сравнения регионов сомелье не делает.
FOLLOW_UP_CHIPS: dict[str, dict[str, Any] | None] = {
    "Аналог просекко": None,
    "Лучшие игристые": None,
    "Что к устрицам": {"id": "dish_check", "text": "А к устрицам?", "args": {"dish": "oysters"}},
    "Какой бокал для белого": {
        "id": "term",
        "text": "Какой нужен бокал",
        "args": {"topic": "glassware"},
    },
    "Сколько хранится открытая бутылка": {
        "id": "term",
        "text": "Сколько живёт открытая бутылка",
        "args": {"topic": "open_bottle"},
    },
    "Что к борщу": {"id": "dish_check", "text": "А к борщу?", "args": {"dish": "borsch"}},
    "При какой температуре подавать красное": {"id": "serve", "text": "Как подать"},
    "Что значит выдержка в дубе": {
        "id": "term",
        "text": "Что даёт дуб",
        "args": {"topic": "oak_aging"},
    },
    "Лучшие вина каталога": None,
    "Откуда вина в каталоге": None,
    "Сравнить Крым и Кубань": None,
    "Самый популярный сорт": None,
    "Как хранить закрытое вино": {
        "id": "term",
        "text": "Как хранить вино",
        "args": {"topic": "storage"},
    },
    "Что такое танины": {"id": "term", "text": "Что такое танины", "args": {"topic": "tannins"}},
    "Аналог бордо": None,
    "Вина с высокой оценкой": None,
    "Что купить новичку": None,
    "Вино к острому": None,
    "Зачем декантировать вино": {
        "id": "term",
        "text": "Зачем декантировать",
        "args": {"topic": "decanting"},
    },
    "Вино к стейку": {"id": "dish_check", "text": "А к стейку?", "args": {"dish": "steik_ribay"}},
    "Как читать этикетку": {
        "id": "term",
        "text": "Как читать этикетку",
        "args": {"topic": "read_label"},
    },
    "Что такое ЗГУ": {"id": "term", "text": "Что такое ЗГУ", "args": {"topic": "zgu_znmp"}},
    "Вина из Красностопа": None,
    "Сравнить Крым и Долину Дона": None,
    "Отсканировать этикетку": None,
    "Аналог шампанского": None,
    "Чем отличается брют от экстра брюта": {
        "id": "term",
        "text": "Что значит «брют»?",
        "args": {"topic": "brut_scale"},
    },
    "Вино к плову": {"id": "dish_check", "text": "А к плову?", "args": {"dish": "plov"}},
    "Аналог грузинского квеври": None,
    "Сколько живёт открытая бутылка": {
        "id": "term",
        "text": "Сколько живёт открытая бутылка",
        "args": {"topic": "open_bottle"},
    },
    "Что значит год урожая": {
        "id": "term",
        "text": "Что значит год урожая",
        "args": {"topic": "vintage"},
    },
    "Самое плотное вино": None,
    "Что к стейку": {"id": "dish_check", "text": "А к стейку?", "args": {"dish": "steik_ribay"}},
    "Что такое резерв": {
        "id": "term",
        "text": "Что значит «резерв»",
        "args": {"topic": "reserve_label"},
    },
    "Что такое дуб в вине": {
        "id": "term",
        "text": "Что даёт дуб",
        "args": {"topic": "oak_aging"},
    },
    "Что такое выдержка в дубе": {
        "id": "term",
        "text": "Что даёт дуб",
        "args": {"topic": "oak_aging"},
    },
    "Самое дубовое вино": None,
    "Что к красной рыбе": {
        "id": "dish_check",
        "text": "А к красной рыбе?",
        "args": {"dish": "krasnaya_ryba"},
    },
    "Что такое оранжевое вино": {
        "id": "term",
        "text": "Что такое оранжевое вино",
        "args": {"topic": "orange_wine"},
    },
    "Как делают игристое": {
        "id": "term",
        "text": "Как делают игристое",
        "args": {"topic": "sparkling_methods"},
    },
    "Что такое брют": {
        "id": "term",
        "text": "Что значит «брют»?",
        "args": {"topic": "brut_scale"},
    },
    "Аналог шабли": None,
    "Что такое выдержка": {"id": "term", "text": "Что такое выдержка", "args": {"topic": "aging"}},
    "Чем заменить Сотерн": None,
    "Что к десерту": {"id": "guided", "text": "К десерту", "args": {"food": "dessert"}},
    "Что такое тело вина": {
        "id": "term",
        "text": "Что такое тело вина",
        "args": {"topic": "wine_body"},
    },
    "Какие креплёные есть": None,
    "Что к селёдке под шубой": {
        "id": "dish_check",
        "text": "А к селёдке под шубой?",
        "args": {"dish": "seledka_pod_shuboy"},
    },
    "Аналог Шабли": None,
    "Что такое терруар": {"id": "term", "text": "Что такое терруар", "args": {"topic": "terroir"}},
    "Какой сорт самый распространённый": None,
    "Что означает ЗГУ на этикетке": {
        "id": "term",
        "text": "Что такое ЗГУ",
        "args": {"topic": "zgu_znmp"},
    },
    "Что такое декантация": {
        "id": "term",
        "text": "Зачем декантировать",
        "args": {"topic": "decanting"},
    },
    "Сколько лет может храниться вино": {
        "id": "term",
        "text": "Как хранить вино",
        "args": {"topic": "storage"},
    },
    "Как хранить открытую бутылку": {
        "id": "term",
        "text": "Сколько живёт открытая бутылка",
        "args": {"topic": "open_bottle"},
    },
    "Что такое сульфиты": {
        "id": "term",
        "text": "Зачем в вине сера",
        "args": {"topic": "sulfites"},
    },
    "Что такое пет-нат": {"id": "term", "text": "Что такое пет-нат", "args": {"topic": "pet_nat"}},
    "Расскажи про Долину Дона": None,
}

#: Чистка тем «Лозы» по 38-ФЗ и честности данных: (тема, поле, было, стало). «Было» обязано
#: найтись в тексте — иначе сборка падает: справочник поменялся, и правку нужно пересмотреть.
TOPIC_REWRITES: tuple[tuple[str, str, str, str], ...] = (
    # «Для покупателя» — покупка; суть фразы — про того, кто читает этикетку.
    ("zgu_znmp", "detail", "Для покупателя надпись", "Надпись"),
    # «Дешёвая имитация» — про цену; профиль сканера по дубу — оценка по сорту, а не замер.
    (
        "oak_aging",
        "detail",
        "Есть и дешёвая имитация — дубовая щепа",
        "Есть и замена бочке — дубовая щепа",
    ),
    (
        "oak_aging",
        "detail",
        "В профилях вкуса нашего каталога дуб — отдельная ось: видно, сколько его в конкретном вине.",
        (
            "В профиле вкуса на карточке дуб — отдельная ось; она оценена по сорту и словам названия, а "
            "не замерена в конкретной бутылке."
        ),
    ),
    # «Лучше», «великие», «попробуйте», «дешёвого» — оценка, превосходная степень и призыв.
    (
        "dry_vs_semi",
        "answer",
        "«Лучше» здесь нет — есть разный остаточный сахар:",
        "Разница здесь не в качестве, а в остаточном сахаре:",
    ),
    ("dry_vs_semi", "detail", "родилась из эпохи", "родилась в эпоху"),
    ("dry_vs_semi", "detail", "недостатки дешёвого виноматериала", "недостатки виноматериала"),
    ("dry_vs_semi", "detail", "великие рислинги Германии", "многие рислинги Германии"),
    ("dry_vs_semi", "detail", "Практика: к еде", "К еде"),
    ("dry_vs_semi", "detail", "к острой азиатской кухне", "к острым блюдам"),
    (
        "dry_vs_semi",
        "detail",
        (
            "Если вы только начинаете, попробуйте оба стиля одного сорта — разница скажет больше любых "
            "описаний."
        ),
        "Разница стилей нагляднее на одном сорте — она скажет больше любых описаний.",
    ),
    (
        "tannins",
        "detail",
        (
            "В профилях нашего каталога танинность — отдельная ось: по ней видно, брать ли вино к "
            "стейку или к лёгкой закуске."
        ),
        (
            "В профиле вкуса на карточке танины — отдельная ось, оценённая по сорту: по ней видно, "
            "подавать ли вино к стейку или к лёгкой закуске."
        ),
    ),
    ("vintage", "detail", "Правило покупателя простое", "Правило простое"),
    # Сравнение доз с сухофруктами читается как «безвредно» — не наша тема.
    (
        "sulfites",
        "detail",
        " Дозы в вине жёстко нормированы и в разы ниже, чем в сухофруктах.",
        "",
    ),
    (
        "autochthon",
        "detail",
        "Хотите понять русское вино — начните с автохтонов: этот вкус не привезёшь из-за границы.",
        "Автохтоны и отличают российское вино: этот вкус не привезёшь из-за границы.",
    ),
    (
        "read_label",
        "detail",
        (
            "А самый простой способ — сфотографировать этикетку в этом приложении: расскажу, что это "
            "за вино и с чем его пить."
        ),
        (
            "А ещё можно сфотографировать этикетку в этом приложении: расскажу, что это за вино и к "
            "чему его подать."
        ),
    ),
    ("sparkling_methods", "detail", "дороже и дольше", "дольше и сложнее"),
    (
        "orange_wine",
        "detail",
        "Это самый древний способ виноделия — в Грузии",
        "Это древний способ виноделия: в Грузии",
    ),
    (
        "orange_wine",
        "detail",
        "Пить их стоит не ледяными, 10–12 °C.",
        "Подают их не ледяными, при 10–12 °C.",
    ),
    (
        "storage",
        "answer",
        "Кухня у плиты и дверца холодильника — худшие места в доме.",
        "Кухня у плиты и дверца холодильника для этого не годятся.",
    ),
    (
        "storage",
        "detail",
        (
            "большинство вин на полке сделаны, чтобы пить их молодыми, — «полежит и станет лучше» "
            "относится к плотным выдержанным красным, а не к любой бутылке."
        ),
        (
            "большинство вин на полке сделаны, чтобы открыть их молодыми; годами в бутылке "
            "развиваются плотные выдержанные красные, а не любое вино."
        ),
    ),
    (
        "wine_body",
        "detail",
        (
            "В карточке каждого вина каталога тело показано отдельной осью профиля, поэтому «самое "
            "плотное вино» — вопрос со счётным ответом, а не с делом вкуса."
        ),
        (
            "В карточке вина тело — отдельная ось профиля, оценённая по сорту, а не замеренная в "
            "конкретной бутылке."
        ),
    ),
    (
        "reserve_label",
        "answer",
        (
            "которое хозяйство считает лучшим в линейке и держит дольше обычного, но никакого "
            "обязательного срока за этим словом не стоит."
        ),
        (
            "которое хозяйство выделяет в линейке и выдерживает дольше обычного, но никакого "
            "обязательного срока за этим словом нет."
        ),
    ),
    (
        "reserve_label",
        "detail",
        (
            "В России проверить это обещание можно только по самому вину: по баллу слепой дегустации "
            "и по тому, что написано в составе."
        ),
        "В России это обещание проверяется только самим вином и тем, что написано на этикетке.",
    ),
    ("wine_colors", "detail", "уживается с самым разным столом", "уживается с разным столом"),
    (
        "pet_nat",
        "detail",
        " В российском каталоге такие вина встречаются у донских и кубанских хозяйств.",
        "",
    ),
    (
        "sommelier",
        "answer",
        (
            "составляет карту, закупает, хранит, советует гостю и подаёт. Не то же самое, что кавист "
            "(продавец винного магазина) и не дегустатор-эксперт, который оценивает вино вслепую."
        ),
        (
            "составляет винную карту, следит за хранением, советует гостю и подаёт. "
            "Дегустатор-эксперт, который оценивает вино вслепую, — другая профессия."
        ),
    ),
    ("tannins", "answer", "они дают то самое вяжущее", "они дают вяжущее"),
    (
        "botrytis",
        "answer",
        "и получаются великие сладкие вина — Сотерн",
        "и так получаются сладкие вина — Сотерн",
    ),
    (
        "alcohol_strength",
        "answer",
        (
            "Крепость показывает его долю: у сухих вин обычно 11–14 % об., у игристых 11–12,5 %, у "
            "креплёных 15–20 %."
        ),
        (
            "Крепость показывает долю спирта в объёме вина: у сухих обычно от 11 до 14 градусов, у "
            "игристых — от 11 до 12,5, у креплёных — от 15 до 20."
        ),
    ),
    (
        "acidity",
        "answer",
        "Чувствуется как сок под языком и желание сделать следующий глоток.",
        "Чувствуется как сок под языком.",
    ),
    (
        "acidity",
        "detail",
        "В карточках каталога кислотность показана отдельной осью профиля.",
        "В карточке вина кислотность — отдельная ось профиля, оценённая по сорту.",
    ),
    (
        "blend",
        "detail",
        (
            "Купаж не значит «хуже сортового»: по этому принципу сделаны бордо, шампанское и "
            "большинство великих вин мира."
        ),
        (
            "Купаж не значит «проще сортового»: по этому принципу сделаны бордо, шампанское и многие "
            "известные вина мира."
        ),
    ),
    (
        "wine_drink",
        "detail",
        (
            "Различать их проще по этой надписи, а не по цене или бутылке. В каталоге «Лозы» винных "
            "напитков нет: здесь только вина с защищённым географическим указанием и наименованием "
            "места происхождения."
        ),
        "Различать их проще по этой надписи, а не по бутылке.",
    ),
    (
        "closure",
        "answer",
        "Винтовая крышка не хуже пробки.",
        "Винтовая крышка — не уступка качеству.",
    ),
    (
        "closure",
        "answer",
        "винт герметичен и лучше держит свежесть",
        "винт герметичен и держит свежесть",
    ),
    ("closure", "detail", "закрывают и дорогие вина", "закрывают и выдержанные вина"),
)

#: Слова контекста тем, которые сами — оценка («лучше», «хуже»): в данных их нет, вопрос «что
#: лучше» ловят соседние основы (`выбра`, `отлича`, `разниц`) и добавленные `предпоч`, `выбор`.
CONTEXT_REPLACE: dict[str, tuple[str, ...]] = {"лучше": ("предпоч", "выбор"), "хуже": ()}


def clean_topics(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """32 темы «Лозы» после чистки: правки `TOPIC_REWRITES`, чипы вместо подсказок."""
    rewrites: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for topic_id, field, old, new in TOPIC_REWRITES:
        rewrites[(topic_id, field)].append((old, new))
    ids = {topic["id"] for topic in source["topics"]}
    unknown = {topic_id for topic_id, _ in rewrites} - ids
    if unknown:
        raise ValueError(f"правки для несуществующих тем: {sorted(unknown)}")

    out = []
    for topic in source["topics"]:
        entry: dict[str, Any] = {"id": topic["id"], "name": topic["name"]}
        entry["triggers"] = list(topic["triggers"])
        context: list[str] = []
        for word in topic.get("context") or ():
            for replacement in CONTEXT_REPLACE.get(word, (word,)):
                if replacement not in context:
                    context.append(replacement)
        entry["context"] = context
        for field in ("answer", "detail"):
            text = topic.get(field) or ""
            for old, new in rewrites.get((topic["id"], field), ()):
                if old not in text:
                    raise ValueError(f"тема {topic['id']}.{field}: нет фрагмента {old!r}")
                text = text.replace(old, new)
            entry[field] = text
        chips: list[dict[str, Any]] = []
        for follow in topic.get("follow_up") or ():
            if follow not in FOLLOW_UP_CHIPS:
                raise ValueError(f"тема {topic['id']}: подсказка {follow!r} не разобрана")
            chip = FOLLOW_UP_CHIPS[follow]
            if chip is None or chip.get("args", {}).get("topic") == topic["id"] or chip in chips:
                continue
            chips.append(chip)
        entry["chips"] = chips
        out.append(entry)
    return out


# ------------------------------------------------------------------ портреты сортов
#: Подписи осей приоров — `recommend/grapes.py` «Лозы» (`_AXIS_WORDS`, пороги 2,2 и 3,6).
AXIS_WORDS: dict[str, tuple[str, str, str]] = {
    "tannin": ("мягкие танины", "заметные танины", "плотные танины"),
    "body": ("лёгкое тело", "среднее тело", "плотное тело"),
    "acidity": ("невысокая кислотность", "живая кислотность", "высокая кислотность"),
    "aroma_intensity": ("сдержанный аромат", "выразительный аромат", "яркий аромат"),
}
PORTRAIT_AXES = ("tannin", "acidity", "body", "aroma_intensity")
#: Белые сорта танинов не дают — подпись «мягкие танины» у них вводила бы в заблуждение.
NO_TANNIN_BELOW = 1.0


def axis_word(axis: str, value: float) -> str:
    words = AXIS_WORDS[axis]
    return words[0] if value < 2.2 else words[1] if value < 3.6 else words[2]


def plural(count: int, one: str, few: str, many: str) -> str:
    """«1 вино», «3 вина», «120 вин»."""
    if count % 10 == 1 and count % 100 != 11:
        return one
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return few
    return many


def join_ru(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} и {items[-1]}"


def grape_notes(
    facts: Sequence[WineFacts], priors: Mapping[str, GrapePrior]
) -> dict[str, dict[str, Any]]:
    """Портреты сортов: слова осей приоров и счёт канонических вин выгрузки по регионам.

    Считаются канонические позиции (одно вино в разных объёмах — одно вино). Сорт без приоров
    или без вин выгрузки портрета не получает. Лучших и рейтингов нет.
    """
    regions: dict[str, Counter[str]] = defaultdict(Counter)
    for wine in facts:
        if not wine.canonical:
            continue
        for code in dict.fromkeys(wine.codes):
            regions[code][wine.region] += 1
    out: dict[str, dict[str, Any]] = {}
    for code in sorted(regions):
        prior = priors.get(code)
        if prior is None:
            continue
        label = grape_label(code)
        traits = [
            axis_word(axis, prior.axes[axis])
            for axis in PORTRAIT_AXES
            if axis in prior.axes and not (axis == "tannin" and prior.axes[axis] < NO_TANNIN_BELOW)
        ]
        count = sum(regions[code].values())
        top = [
            region for region, _ in sorted(regions[code].items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        text = f"{label} по сорту: {join_ru(traits)}." if traits else f"{label}."
        if count == 1:
            text += f" В каталоге одно вино с этим сортом, регион — {top[0]}."
        elif len(top) == 1:
            wines = plural(count, "вино", "вина", "вин")
            text += f" В каталоге {count} {wines} с этим сортом, все — {top[0]}."
        else:
            text += (
                f" В каталоге {count} {plural(count, 'вино', 'вина', 'вин')} с этим сортом, "
                f"чаще всего — {join_ru(top[:2])}."
            )
        out[code] = {"label": label, "text": text, "count": count, "regions": top[:3]}
    return out


# ------------------------------------------------------------------ словарь замка
def words(text: str) -> tuple[str, ...]:
    """Слова списка, записанного абзацем: так словари читаются глазами."""
    return tuple(text.split())


#: Родовые слова, способы готовки и вкусовые основы — `llm/entity_lock.py` «Лозы» (`_GENERIC`,
#: `_COOKING`, `_TASTE_STEMS`); к основам добавлены основы из заглушки договора.
GENERIC_WORDS = words("""
    блюдо блюда еда закуска закуски перекус ужин обед стол трапеза
    вино вина винишко напиток бокал бутылка
    мясо мясу мяса рыба рыбе рыбу рыбы птица птице дичь морепродукты
    овощи овощ овощам фрукты фруктам ягоды ягодам грибы грибам зелень
    салат салаты салату суп супы супу гарнир соус соусу специи пряности
    сыр сыру сыра сыры десерт десерту сладкое выпечка хлеб хлебу каша каше
    пара пару часа часов бокала бокалов
    красное белое розовое игристое сухое полусухое полусладкое
""")
COOKING_WORDS = (
    *words("""
    гриль гриле гриля жареный жареная жареное жаренный копчёный копченый
    копчёная копченая запечённый запеченный запечённая запеченная варёный
    вареный варёная вареная тушёный тушеный тушёная тушеная маринованный
    маринованная солёный соленый солёная соленая вяленый вяленая
    панировке кляре фритюре мангале углях вертеле
"""),
    "на пару",
)
TASTE_STEMS = (
    "свеж", "холодн", "дубов", "лёгк", "легк", "плотн", "мягк", "ярк",
    "спел", "зрел", "молод", "тёпл", "тепл", "сочн", "терпк", "кисл",
    "танин", "сладк", "сух", "тел", "крепост", "пузыр", "аромат", "вкус", "фрукт",
)  # fmt: skip

#: Ароматические слова для проверки `descriptors` (§6.5): модель не вправе приписать вину
#: аромат, которого нет в пакете фактов. Коды приоров «Лозы» (`AROMA_DESCRIPTORS`) словами и
#: частые слова описаний вин — фрукты, цветы, травы, пряности, тона выдержки.
DESCRIPTOR_WORDS = words("""
    вишня черешня малина клубника земляника ежевика черника смородина клюква брусника
    крыжовник шелковица слива чернослив инжир финик изюм курага персик абрикос нектарин
    груша яблоко айва лимон лайм грейпфрут апельсин мандарин цитрус цитрусовые ананас манго
    маракуйя личи дыня банан гранат ягоды фиалка роза акация жасмин бузина липа лаванда
    пион мята эвкалипт хвоя шалфей розмарин тимьян чабрец фенхель анис лакрица ваниль корица
    гвоздика перец мускат табак кожа шоколад кофе какао карамель тоффи дым бриошь тост
    дрожжи выпечка мёд орех миндаль фундук кедр сандал подлесок трюфель кремень графит сено
    чай оливка джем конфитюр сухофрукты сливки минеральность
""")

#: Слова названий виноделен, которые сами по себе — обычная речь: замок не должен браковать
#: «ароматное вино» из-за винодельни «Ароматное» или «долину» из-за «Долины Лефкадии».
WINERY_COMMON = frozenset(
    words("""
    винодельня винодельни винодельческое винодельческий хозяйство усадьба шато имение
    поместье долина дом вино вина винный вины кооператив форт виноградники семейная семья
    сыновья братьев два дача поле берег новый свет шампанских союз родное гнездо ароматное
    вилла солнечная скалистый золотая золотое
    winery wine wines vineyard vineyards vines estate cellar master organic valley villa
    chateau château domaine grand radio vibes vino
    """)
)
#: Окончания прилагательных: в многословном названии такое слово — чаще обиходное.
ADJECTIVE_ENDINGS = ("ая", "яя", "ое", "ее", "ый", "ий", "ой", "ые", "ие", "ого", "его", "ских")


def winery_words(wineries: Iterable[str], taken: Iterable[str] = ()) -> list[str]:
    """Слова виноделен для замка: однословные имена целиком и отличительные слова остальных.

    Правило «Лозы» — только однословные имена: «Хорошая компания» рассыпается на обиходные
    слова. Здесь из многословных берутся и отличительные слова (фамилии, топонимы): без
    обиходных (`WINERY_COMMON`), прилагательных и слов короче четырёх букв. Слово, которое уже
    значит другое — сорт, блюдо, аромат, родовое слово (`taken`: «Шато Пино» и «Пино Нуар»), —
    винодельней не считается: иначе замок браковал бы честное «Пино Нуар» у чужой винодельни.
    """
    busy = {word for phrase in taken for word in phrase.lower().split()}
    out: set[str] = set()
    for winery in wineries:
        words = [w for w in re.split(r"[\s&.,\"()«»]+", winery.lower()) if w]
        if len(words) == 1:
            candidates = words
        else:
            candidates = [w for w in words if not w.endswith(ADJECTIVE_ENDINGS)]
        for word in candidates:
            word = word.strip("-'")
            if (
                len(word) < 4
                or word in WINERY_COMMON
                or word in busy
                or any(ch.isdigit() for ch in word)
            ):
                continue
            out.add(word)
            if word.endswith("ъ"):
                out.add(word[:-1])  # «Гусевъ» пишут и «Гусев»
    return sorted(out)


def build_vocab(facts: Sequence[WineFacts], dishes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wineries = sorted({wine.winery for wine in facts})
    grapes: set[str] = set()
    for spellings in GRAPE_SYNONYMS.values():
        grapes.update(spelling.lower() for spelling in spellings)
    for wine in facts:
        grapes.update(label.lower() for label in wine.grapes)
    dish_words: set[str] = set()
    for dish in dishes:
        dish_words.add(dish["name"].lower())
        dish_words.update(alias.lower() for alias in dish["aliases"])
    generic = set(GENERIC_WORDS)
    # Родовое слово («ягоды», «выпечка») не бракуется — значит, и ароматом его не считать.
    descriptors = set(DESCRIPTOR_WORDS) - generic
    taken = grapes | dish_words | descriptors | generic
    return {
        "version": VERSION,
        "wineries": wineries,
        "winery_words": winery_words(wineries, taken),
        "grapes": sorted(grapes),
        "dishes": sorted(dish_words),
        "cooking": sorted(set(COOKING_WORDS)),
        "regions": sorted({wine.region for wine in facts if wine.region}),
        "descriptors": sorted(descriptors),
        "generic": sorted(generic),
        "taste_stems": sorted(set(TASTE_STEMS)),
    }


# ------------------------------------------------------------------ право
#: Рубли как деньги: «рубль», «рубля», «рублей», «руб.», «руб». Не деньги — «рубленые котлеты»
#: (алиас блюда «Лозы») и сорт «Рубиновый Магарача»: их стоп-лист не трогает.
CURRENCY = re.compile(r"\bрубл(?!ен)|\bруб(?:\.|\b)|₽", re.IGNORECASE)

#: Стоп-лист договора (§6.5) и запрет превосходных степеней: по всем строкам для человека.
STOPLIST = re.compile(
    r"лучш|худш|идеальн|превосходн|шедевр|безупречн|непревзойд|уникальн|великолепн|вино недели"
    r"|номер один|рейтинг|балл|звёзд|скидк|\bакци[яиюей]|\bкупи|\bпокуп|закаж|\bцен[аыуе]\b"
    r"|стоимост|" + CURRENCY.pattern + r"|пейте|выпейте|попробуйте|наслаждайтесь|\bобязательно\b"
    r"|\bсам(?:ый|ая|ое|ые|ого|ой|ому|ом|ых|ым|ыми|ую)\b|\bнаи\w+ш|\bвелик(?:ий|ие|их|ая|ое)\b"
    r"|дешёв|дешев|дорог(?:ой|ие|ая|ое|их|о\b)|expert_score|typical_price|price|%",
    re.IGNORECASE,
)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def legal_violations(texts: Iterable[str]) -> list[str]:
    """Нарушения по строкам: `content_filter.check`, стоп-лист, превосходные степени, `%`."""
    bad = []
    for text in texts:
        verdict = check(text)
        if not verdict.clean:
            bad.append(f"{verdict.violations}: {text}")
        found = STOPLIST.search(_nfkc(text))
        if found:
            bad.append(f"«{found.group(0)}»: {text}")
    return bad


def strings(value: Any) -> list[str]:
    """Все строки JSON-объекта: ключи словарей и значения."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for key, item in value.items() for s in (str(key), *strings(item))]
    if isinstance(value, list | tuple):
        return [s for item in value for s in strings(item)]
    return []


# ------------------------------------------------------------------ сборка
@dataclass
class Build:
    """Результат сборки: байты файлов и то, что нужно отчёту."""

    files: dict[str, bytes]
    facts: list[WineFacts]
    pairs: dict[str, dict[str, DishPair]]
    tops: dict[str, list[str]]
    dishes: dict[str, dict[str, Any]]
    rules: dict[str, dict[str, Any]]
    serves: dict[str, ServeMatch]
    seconds: dict[str, float]


def digest(paths: Iterable[Path]) -> str:
    """Короткие sha256 входов для поля `built`: пересборка на тех же входах — те же байты.

    Концы строк приводятся к LF: git на Windows (`core.autocrlf`) выдаёт те же справочники с
    CRLF, и без этого `--check` расходился бы между машинами при одинаковых данных.
    """
    parts = []
    for path in paths:
        body = path.read_bytes().replace(b"\r\n", b"\n")
        parts.append(f"{path.name} {hashlib.sha256(body).hexdigest()[:12]}")
    return ", ".join(parts)


def dump(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def dump_pairs(obj: Mapping[str, Any]) -> bytes:
    """`pairs.json` — одна строка на вино: 2 103 × 82 пары без отступов внутри вина."""
    lines = [
        "{",
        f'  "version": {obj["version"]},',
        f'  "built": {json.dumps(obj["built"], ensure_ascii=False)},',
        '  "wines": {',
    ]
    items = list(obj["wines"].items())
    for index, (slug, entry) in enumerate(items):
        comma = "," if index < len(items) - 1 else ""
        body = json.dumps(entry, ensure_ascii=False, separators=(", ", ": "))
        lines.append(f"    {json.dumps(slug, ensure_ascii=False)}: {body}{comma}")
    lines += ["  }", "}"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(csv_path: Path, gt_path: Path, groups_path: Path, loza_dir: Path = LOZA_DIR) -> Build:
    """Полная сборка по выгрузке организатора: факты → движок → пять файлов."""
    started = time.perf_counter()
    facts = load_facts(csv_path, gt_path, groups_path)
    loaded = time.perf_counter() - started
    inputs = [csv_path, gt_path, groups_path, PRIORS_PATH, loza_dir / "overlay.json"]
    inputs += sorted((loza_dir / "reference").glob("*.json"))
    built = f"scripts/build_somm.py v{VERSION}; входы: {digest(inputs)}"
    result = build_from_facts(facts, built, loza_dir)
    result.seconds["facts"] += loaded
    return result


def build_from_facts(facts: Sequence[WineFacts], built: str, loza_dir: Path = LOZA_DIR) -> Build:
    """Сборка по готовым фактам — её же тесты зовут на синтетических винах."""
    facts = list(facts)
    seconds: dict[str, float] = {}
    started = time.perf_counter()
    priors = load_priors()
    profiles = {wine.slug: profile_of(wine, priors) for wine in facts}
    rules_serve = serve_rules()
    serves = {wine.slug: serve_of(wine, profiles[wine.slug], rules_serve) for wine in facts}
    seconds["facts"] = time.perf_counter() - started

    started = time.perf_counter()
    engine = run_engine(engine_request(facts, profiles, serves), loza_dir)
    seconds["engine"] = time.perf_counter() - started

    started = time.perf_counter()
    reference = json.loads((loza_dir / "reference" / "pairing.json").read_text(encoding="utf-8"))
    dative = json.loads((loza_dir / "reference" / "dish_dative.json").read_text(encoding="utf-8"))
    knowledge_src = json.loads(
        (loza_dir / "reference" / "knowledge.json").read_text(encoding="utf-8")
    )
    dish_list = reference["dishes"]
    dishes = {dish["id"]: dish for dish in dish_list}
    family = families(dish_list)
    rules = {rule["id"]: rule for rule in engine["rules"]}
    if set(rules) != set(RULE_META):
        raise ValueError(f"правила движка и RULE_META разошлись: {set(rules) ^ set(RULE_META)}")
    weights = {rule_id: float(rule["weight"]) for rule_id, rule in rules.items()}

    pairs: dict[str, dict[str, DishPair]] = {}
    tops: dict[str, list[str]] = {}
    for wine in facts:
        row = {}
        for dish_id, (total, score, plus, minus) in engine["pairs"][wine.slug].items():
            if sugar_decides(wine, dishes[dish_id]):
                row[dish_id] = DishPair(total=total, verdict="neutral", plus=(), minus=())
                continue
            row[dish_id] = DishPair(
                total=total,
                verdict=verdict_of(score, plus, minus, weights, total),
                plus=tuple(ordered(plus, weights)),
                minus=tuple(ordered(minus, weights)),
            )
        pairs[wine.slug] = row
        tops[wine.slug] = top_dishes(wine, row, dishes, family)

    rule_order = sorted(rules, key=lambda rule_id: (-weights[rule_id], rule_id))
    pairs_json = {
        "version": VERSION,
        "built": built,
        "wines": {
            wine.slug: {
                "top": tops[wine.slug],
                "dishes": {
                    dish_id: [pair.verdict, list(pair.plus), list(pair.minus)]
                    for dish_id, pair in sorted(pairs[wine.slug].items())
                },
            }
            for wine in facts
        },
    }
    dishes_json = {
        "version": VERSION,
        "source": "tools/loza_engines/reference: pairing.json, pairing_rules_fixes.json, "
        "dish_dative.json (команда «Лозы»), накладка overlay.json",
        "categories": CATEGORIES,
        "dishes": [
            {
                "id": dish["id"],
                "name": dish["name"],
                "dative": dative["forms"][dish["id"]],
                "category": dish["category"],
                "food": food_of(dish),
                "family": family[dish["id"]],
                "aliases": list(dish["aliases"]),
            }
            for dish in dish_list
        ],
        "rules": [
            {
                "id": rule_id,
                "sign": "+" if weights[rule_id] > 0 else "−",
                "weight": weights[rule_id],
                "name": rules[rule_id]["name"].replace(" (конфликт)", ""),
                "chip": RULE_META[rule_id][0],
                "text": RULE_META[rule_id][1] or rules[rule_id]["explanation_ru"],
                "wine_fields": list(RULE_META[rule_id][2]),
            }
            for rule_id in rule_order
        ],
    }
    serve_json = {
        "version": VERSION,
        "source": "наша таблица подачи по темам serve_temp и orange_wine справочника «Лозы»",
        "rules": [dict(rule) for rule in SERVE_RULES],
    }
    knowledge_json = {
        "version": VERSION,
        "source": "tools/loza_engines/reference/knowledge.json (команда «Лозы»), чистка по "
        "38-ФЗ; портреты сортов — приоры сортов и выгрузка организатора",
        "topics": clean_topics(knowledge_src),
        "grapes": grape_notes(facts, priors),
    }
    vocab_json = build_vocab(facts, dish_list)

    files = {
        "pairs.json": dump_pairs(pairs_json),
        "dishes.json": dump(dishes_json),
        "serve.json": dump(serve_json),
        "vocab.json": dump(vocab_json),
        "knowledge.json": dump(knowledge_json),
    }
    seconds["files"] = time.perf_counter() - started
    return Build(
        files=files,
        facts=facts,
        pairs=pairs,
        tops=tops,
        dishes=dishes,
        rules=rules,
        serves=serves,
        seconds=seconds,
    )


#: Слова, которыми правило называет вино сухим: «Сухое на фоне десерта резче», «сухое вино теряет
#: фруктовость». Вину, сахара которого в выгрузке нет, такое правило не ставится (`quality`).
DRY_CLAIM = re.compile(r"\bсух(?:ое|ого|ому|им|ом|ие|их)\b", re.IGNORECASE)

#: Рыбные теги блюд «Лозы» — те же, что в условиях накладки (`overlay.json`).
FISH_TAGS = frozenset(
    {"oily_fish", "roe", "marine", "raw_fish", "white_fish", "shellfish", "seafood"}
)


def _axis(dish: Mapping[str, Any], name: str, default: float) -> float:
    """Ось блюда «Лозы» с умолчанием модели `Dish` (`DishView`)."""
    return float(dish.get(name, default))


def _tags(dish: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(dish.get("flavor_tags") or ())


#: Что подпись или фраза правила говорит о блюде — и чем это подтверждается у блюда (проверка 25.09,
#: вечер: «деликатное» у солёных огурцов, «сахар спорит с солью и мясом» у грибов в сметане,
#: «дымок с углей» у блинов). Каждое блюдо, к которому правило сработало хоть у одного вина,
#: отвечает признаку каждого слова, найденного в подписи и фразе; иначе сборка падает (`quality`).
DISH_CLAIMS: tuple[tuple[str, re.Pattern[str], Callable[[Mapping[str, Any]], bool]], ...] = (
    ("жирное", re.compile(r"жирн"), lambda d: _axis(d, "fat", 2.0) >= 3.5),
    (
        "лёгкое",
        re.compile(r"лёгк\w*\s+блюд|блюдо лёгкое|к лёгкому|с лёгким"),
        lambda d: _axis(d, "intensity", 3.0) <= 3.0 and _axis(d, "fat", 2.0) <= 3.0,
    ),
    ("негромкое", re.compile(r"негромк"), lambda d: _axis(d, "intensity", 3.0) <= 3.0),
    ("солёное", re.compile(r"солён"), lambda d: _axis(d, "salt", 2.0) >= 2.5),
    ("кислое", re.compile(r"кислинк|кислое блюдо"), lambda d: _axis(d, "acidity", 1.0) >= 3.5),
    ("острое", re.compile(r"остр"), lambda d: _axis(d, "spice", 0.0) >= 3.0),
    ("десерт", re.compile(r"десерт"), lambda d: d["category"] in DESSERT_CATEGORIES),
    (
        "бульон",
        re.compile(r"бульон"),
        lambda d: d["category"] == "soup" or "meat_broth" in _tags(d),
    ),
    ("жареное", re.compile(r"жарен"), lambda d: "fried" in _tags(d)),
    (
        "рыба и море",
        re.compile(r"рыб|морепродукт|морск|\bмор[юяе]\b"),
        lambda d: bool(_tags(d) & FISH_TAGS),
    ),
    (
        "дымок и корочка",
        re.compile(r"дым|корочк"),
        lambda d: bool(_tags(d) & {"smoked", "grilled", "char", "toast"}),
    ),
    (
        "насыщенное",
        re.compile(r"насыщенн(?:ый|ое|ому|ого)\b"),
        lambda d: _axis(d, "umami", 2.0) >= 3.5,
    ),
    (
        "белок и жир",
        re.compile(r"белок"),
        lambda d: _axis(d, "fat", 2.0) >= 3.0 and _axis(d, "umami", 2.0) >= 3.5,
    ),
    ("мало жира", re.compile(r"мало жира"), lambda d: _axis(d, "fat", 2.0) <= 2.0),
    ("несладкое", re.compile(r"несладк"), lambda d: _axis(d, "sweetness", 0.0) <= 3.0),
)
#: Слова о блюде, которых условия правил не гарантируют ни одному блюду: «деликатное» и «тонкое»
#: (условие знает только, что блюдо нежирное и негромкое), «мясо», «горячее» и «сытное» (сладкое
#: спорит и с грибами, и с окрошкой, и с овощами на гриле), «уксус и соленья», «закуска», «угли»,
#: «морская нота».
DISH_NEVER = re.compile(r"деликатн|тонк|мяс|горяч|сытн|уксус|соленья|закуск|углей|нота блюда")


def false_claims(result: Build) -> list[tuple[str, str, str]]:
    """Подписи и фразы правил, неправдивые о блюде, к которому правило сработало: `DISH_CLAIMS`
    и `DISH_NEVER` по всем парам сборки. Пусто — каждая подпись правдива о своих блюдах."""
    fired: dict[str, set[str]] = defaultdict(set)
    for row in result.pairs.values():
        for dish_id, pair in row.items():
            for rule_id in (*pair.plus, *pair.minus):
                fired[rule_id].add(dish_id)
    out = []
    for rule in json.loads(result.files["dishes.json"])["rules"]:
        said = f"{rule['chip']} {rule['text']}".lower()
        never = DISH_NEVER.search(said)
        if never:
            out.append((rule["id"], "*", never.group(0)))
        for claim, pattern, holds in DISH_CLAIMS:
            if not pattern.search(said):
                continue
            for dish_id in sorted(fired[rule["id"]]):
                if not holds(result.dishes[dish_id]):
                    out.append((rule["id"], dish_id, claim))
    return out


def file_strings(files: Mapping[str, bytes]) -> list[str]:
    """Все различные строки всех файлов: и тексты для человека, и коды, и словарь замка."""
    out: set[str] = set()
    for name in SOMM_FILES:
        out.update(strings(json.loads(files[name])))
    return sorted(out)


def quality(result: Build) -> dict[str, Any]:
    """Приёмка сборки: блюда у каждого вина (хотя бы одно и не меньше трёх в «к чему подать»),
    подача, конфликты в тройке, сладкое и рыба, сладкое и сыр с орехами, «сухое» у вина без
    сахара, правдивость подписей правил о блюде (`false_claims`), право."""
    fish = {dish_id for dish_id, dish in result.dishes.items() if food_of(dish) == "fish"}
    no_dish = [slug for slug, top in result.tops.items() if not top]
    top_short = [slug for slug, top in result.tops.items() if len(top) < MIN_TOP]
    # Третий круг проверки 25.09: сладкое к рыбе и морепродуктам — «скорее нет», а не оговорка; к
    # сырной тарелке и орехам — «да» (и креплёному названию без сахара). Сладкое по «Описанию»
    # (`sugar_by_card`, проверка `acc-somm3`) — сладкое по карточке: к рыбе тоже «скорее нет».
    sweet = [wine for wine in result.facts if rule_sugar(wine) == "sladkoe"]
    seafood = sorted(d for d, dish in result.dishes.items() if _tags(dish) & FISH_TAGS)
    sweet_fish_soft = [
        (wine.slug, dish_id)
        for wine in sweet
        if not sweet_by_name(wine)
        for dish_id in seafood
        if result.pairs[wine.slug][dish_id].verdict != "no"
    ]
    # Проверка третьего круга: сладкое только по названию к рыбе — оговорка с условной фразой
    # (`maybe_sweet_wine_on_fish`), а не «скорее нет» с фразой о сахаре. «Скорее нет» — только по
    # другому правилу «−», и первым в объяснении тогда идёт оно. Правило-оговорка — только у таких
    # вин, жёсткое правило о сладком к рыбе у них — никогда.
    maybe_sweet_fish = []
    for wine in result.facts:
        guessed = sweet_by_name(wine)
        for dish_id in seafood:
            pair = result.pairs[wine.slug][dish_id]
            hedged = bool(HEDGE_RULES & set(pair.minus))
            if not guessed:
                if hedged:
                    maybe_sweet_fish.append((wine.slug, dish_id, "оговорка не у догадки"))
                continue
            if not hedged or "sweet_wine_on_fish" in pair.minus:
                maybe_sweet_fish.append((wine.slug, dish_id, "нет оговорки"))
            elif pair.verdict not in ("caveat", "no"):
                maybe_sweet_fish.append((wine.slug, dish_id, pair.verdict))
            elif pair.verdict == "no" and pair.minus[0] in HEDGE_RULES:
                maybe_sweet_fish.append((wine.slug, dish_id, "«скорее нет» из-за оговорки"))
    classic = sorted(d for d, dish in result.dishes.items() if cheese_or_nuts(dish))
    sweet_cheese = [
        (wine.slug, dish_id)
        for wine in sweet
        for dish_id in classic
        if result.pairs[wine.slug][dish_id].verdict != "yes"
    ]
    no_dish_pool = [
        wine.slug for wine in result.facts if wine.canonical and not result.tops[wine.slug]
    ]
    conflicts = [
        (slug, dish_id)
        for slug, top in result.tops.items()
        for dish_id in top[:3]
        if result.pairs[slug][dish_id].minus
    ]
    # Сладкое и полусладкое не получают рыбу ни в «к чему подать», ни вердиктом «подходит»; так
    # же креплёные, десертные и мускатные названия без сахара в выгрузке (`rule_sugar`).
    sweet_fish = [
        (wine.slug, dish_id)
        for wine in result.facts
        if rule_sugar(wine) in SWEET_SUGARS
        for dish_id in sorted(fish)
        if dish_id in result.tops[wine.slug] or result.pairs[wine.slug][dish_id].verdict == "yes"
    ]
    # Вино без сахара в выгрузке не называется сухим ни чипом, ни фразой правила (проверка 25.09).
    rules = json.loads(result.files["dishes.json"])["rules"]
    dry = {rule["id"] for rule in rules if DRY_CLAIM.search(f"{rule['chip']} {rule['text']}")}
    dry_guess = [
        (wine.slug, dish_id, rule_id)
        for wine in result.facts
        if wine.sugar is None
        for dish_id, pair in sorted(result.pairs[wine.slug].items())
        for rule_id in (*pair.plus, *pair.minus)
        if rule_id in dry
    ]
    raw = b"".join(result.files.values()).decode("utf-8")
    forbidden = {
        word: len(re.findall(pattern, raw, re.IGNORECASE))
        for word, pattern in (
            ("expert_score", r"expert_score"),
            ("price", r"price"),
            ("руб", CURRENCY.pattern),
            ("%", r"%"),
            ("typical_price", r"typical_price"),
        )
    }
    texts = file_strings(result.files)
    return {
        "no_dish": no_dish,
        "no_dish_pool": no_dish_pool,
        "top_short": top_short,
        "conflicts_top3": conflicts,
        "sweet_fish": sweet_fish,
        "sweet_fish_soft": sweet_fish_soft,
        "maybe_sweet_fish": maybe_sweet_fish,
        "sweet_cheese": sweet_cheese,
        "dry_guess": dry_guess,
        "false_claims": false_claims(result),
        "forbidden": forbidden,
        "legal": legal_violations(texts),
    }


def sample_lines(result: Build, count: int, seed: int = 2409) -> list[str]:
    """Случайные вина пула и их «к чему подать» — для проверки пар глазами."""
    pool = sorted(wine.slug for wine in result.facts if wine.canonical)
    facts = {wine.slug: wine for wine in result.facts}
    lines = []
    for slug in random.Random(seed).sample(pool, min(count, len(pool))):
        wine = facts[slug]
        style = " ".join(
            part
            for part in (wine.color.lower(), SUGAR_WORDS.get(sugar_by_card(wine) or "", "сахар ?"))
            if part
        )
        grapes = ", ".join(wine.grapes) or "сорт не указан"
        dishes = []
        for dish_id in result.tops[slug][:3]:
            pair = result.pairs[slug][dish_id]
            chip = RULE_META[pair.plus[0]][0] if pair.plus else "—"
            dishes.append(f"{result.dishes[dish_id]['name']} (+ {chip})")
        serve = result.serves[slug].rule
        lines.append(
            f"{wine.name} · {wine.winery} · {style} · {grapes} · "
            f"{serve.temperature_c[0]}–{serve.temperature_c[1]} °C → " + "; ".join(dishes)
        )
    return lines


def write_or_check(files: Mapping[str, bytes], out_dir: Path, check_only: bool) -> list[str]:
    """Пишет файлы или сверяет их с диском; вернёт список расхождений."""
    diffs = []
    for name in SOMM_FILES:
        path = out_dir / name
        if check_only:
            if not path.is_file():
                diffs.append(f"{name}: файла нет")
            elif path.read_bytes() != files[name]:
                diffs.append(f"{name}: байты расходятся с пересборкой")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".json.tmp")
        temp.write_bytes(files[name])
        temp.replace(path)
    return diffs


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, default=settings.dataset_dir / "strapi_output0709.csv")
    parser.add_argument("--gt", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl")
    parser.add_argument(
        "--groups", type=Path, default=settings.data_dir / "catalog" / "wine_groups.json"
    )
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "somm")
    parser.add_argument(
        "--check", action="store_true", help="пересобрать в памяти и сверить с --out-dir"
    )
    parser.add_argument("--sample", type=int, default=30, help="сколько вин показать глазами")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    result = build(args.csv, args.gt, args.groups)
    report = quality(result)

    problems = []
    if report["no_dish"]:
        problems.append(f"без блюд: {len(report['no_dish'])}, например {report['no_dish'][:3]}")
    if report["top_short"]:
        problems.append(
            f"меньше {MIN_TOP} блюд в «к чему подать»: {len(report['top_short'])}, например "
            f"{report['top_short'][:3]}"
        )
    if report["conflicts_top3"]:
        problems.append(f"конфликтные блюда в тройке: {report['conflicts_top3'][:3]}")
    if report["sweet_fish"]:
        problems.append(f"сладкое с рыбой: {report['sweet_fish'][:3]}")
    if report["sweet_fish_soft"]:
        problems.append(f"сладкое с рыбой не «скорее нет»: {report['sweet_fish_soft'][:3]}")
    if report["maybe_sweet_fish"]:
        problems.append(
            f"сладкое по названию с рыбой не оговорка: {report['maybe_sweet_fish'][:3]}"
        )
    if report["sweet_cheese"]:
        problems.append(f"сладкое с сыром или орехами не «да»: {report['sweet_cheese'][:3]}")
    if report["dry_guess"]:
        problems.append(f"«сухое» у вина без сахара: {report['dry_guess'][:3]}")
    if report["false_claims"]:
        problems.append(f"подпись правила неправда о блюде: {report['false_claims'][:5]}")
    if any(report["forbidden"].values()):
        problems.append(f"запретные вхождения: {report['forbidden']}")
    if report["legal"]:
        problems.append("право: " + " | ".join(report["legal"][:5]))
    if problems:
        print("Сборка не прошла приёмку:\n  " + "\n  ".join(problems))
        return 2

    diffs = write_or_check(result.files, args.out_dir, args.check)
    if not args.check or not diffs:
        load_somm_data(args.out_dir, fallback=None, strict=True)

    facts = result.facts
    verdicts = Counter(pair.verdict for row in result.pairs.values() for pair in row.values())
    serve_rules_used = Counter(serve.rule.id for serve in result.serves.values())
    tops = [len(top) for top in result.tops.values()]
    print(f"Вин выгрузки: {len(facts)}, канонических: {sum(w.canonical for w in facts)}")
    described = sum(w.sugar is None and sugar_by_card(w) is not None for w in facts)
    print(
        f"Сахар известен: {sum(w.sugar is not None for w in facts)} (и по «Описанию» {described}), "
        f"игристых: {sum(w.sparkling for w in facts)}, "
        f"крепость известна: {sum(w.abv is not None for w in facts)}"
    )
    print(f"Пар: {sum(len(row) for row in result.pairs.values())}; вердикты: {dict(verdicts)}")
    print(f"Блюд в top: мин {min(tops)}, распределение {dict(sorted(Counter(tops).items()))}")
    print(f"Подача по правилам: {dict(serve_rules_used)}")
    print(
        "Время, с: "
        + ", ".join(f"{key} {value:.1f}" for key, value in result.seconds.items())
        + f", всего {time.perf_counter() - started:.1f}"
    )
    print("Размеры: " + ", ".join(f"{n} {len(b) / 1e6:.2f} МБ" for n, b in result.files.items()))
    if args.sample:
        print(f"\n{args.sample} случайных вин пула → «к чему подать» (первые три):")
        for line in sample_lines(result, args.sample):
            print("  " + line)
    if args.check:
        if diffs:
            print("\n--check: расхождения:\n  " + "\n  ".join(diffs))
            return 1
        print(f"\n--check: {args.out_dir} совпадает с пересборкой байт в байт")
    else:
        print(f"\nЗаписано в {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
