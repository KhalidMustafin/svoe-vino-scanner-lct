"""Справочник для рекомендаций: позиции выгрузки организатора с фактами карточки.

Решение 24.09: в продукте только выгрузка организатора и наши алгоритмы. Снимок портала
(`data/portal/`), 73 карточки живого портала вне выгрузки, статус «опубликовано» и сахар из
категории живого портала справочник не читает. Источники (все в `data/`, лежат в git — `data/README.md`):

    gt/gt_tokens.jsonl        2 103 позиции выгрузки `strapi_output0709.csv`: название,
                              винодельня, регион, «Категория» (цвет), крепость из slug, год,
                              имя фото. Разметка сканера собрана из выгрузки, но её класс
                              сахара у 2 013 позиций взят из категории живого портала — он здесь
                              не читается, сахар считается заново по правилу ниже.
    catalog/wines.jsonl       наш словарь групп: группа `wine_id` (одно вино в разных объёмах и
                              годах), каноническая позиция, ключ винодельни и коды сортов из
                              колонки «Сорт винограда». Остальные поля строки (цвет, сахар,
                              игристость, крепость, «опубликовано», фото живого портала) и 73
                              строки вне выгрузки не читаются.
    catalog/wine_groups.json  состав групп `wine_id` — только позиции выгрузки.

Факты карточки (договор «после поиска», §3) — только то, что написал организатор:

* **сахар** — класс, названный в «Название вина» (`sugar_of_name`), иначе последнее слово
  сахара в slug (`sugar_of_slug`), иначе неизвестен. Замер 24.09: известен у 1 727 из 2 103;
  у всех, кроме опечатки «полсусладкое» (теперь из slug), совпадает с прежним классом разметки;
* **игристость** — сахар брют-семейства или слово игристого в названии либо slug, пет-нат в
  любом написании (`denisov_pet_nat_rubin`): 297;
* **крепость** — `fields.abv.value` разметки (разобрано из slug): одно число или диапазон,
  вне 3–25 — неизвестна; число, равное году урожая (`daniel-22` у «Daniel, 2022»), — год, а не
  крепость (`slug_abv`). Известна у 1 567, диапазонов 32.

Пул похожих — канонические позиции групп с фото выгрузки. Если канонической позицией группы в
словаре была карточка живого портала, её место занимает первая позиция выгрузки той же группы
(так в пул возвращается `merlo-litavshhuk`). Фото, на котором другое вино (`WRONG_PHOTOS`, 23
позиции), не показывается, но вино остаётся в пуле: его плитка выходит с силуэтом бутылки.

Файл фото (`RecoWine.photo` — имя фото выгрузки) сервис ищет сам (`AfterSearch.photo_file`):
путь машины наружу не отдаётся.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from app.reading.taxonomy import GRAPE_SYNONYMS, grape_label

logger = logging.getLogger(__name__)

TOKENS_NAME = "gt_tokens.jsonl"
WINES_NAME = "wines.jsonl"
GROUPS_NAME = "wine_groups.json"

#: Единственная внешняя ссылка ответа: адрес карточки вина на портале. Это адрес, а не данные
#: портала. Ссылка есть только у позиций, которые есть в карте сайта вин из дампа Strapi
#: организатора (`portal_links.json`, `scripts/build_portal_links.py`): у 66 из 2 103 slug
#: выгрузки страницы на портале нет, и `portal_url` у них `null`.
PORTAL_WINE_URL = "https://vino-svoe.ru/wines/"
PORTAL_LINKS_PATH = Path(__file__).with_name("portal_links.json")

#: Крепость правдоподобна в этих пределах.
ABV_MIN = 3.0
ABV_MAX = 25.0

#: Цвета каталога (значения `Color` разметки) и их форма после «а не»: «Белое, а не оранжевое».
COLORS: tuple[str, ...] = ("Белое", "Красное", "Розовое", "Оранжевое")

#: Сахар словом — как на этикетке («Белое экстра брют», «Красное полусухое»).
SUGAR_WORDS: dict[str, str] = {
    "brut_nature": "брют натюр",
    "extra_brut": "экстра брют",
    "brut": "брют",
    "suhoe": "сухое",
    "polusuhoe": "полусухое",
    "polusladkoe": "полусладкое",
    "sladkoe": "сладкое",
}

#: Ступени сладости для «сахар в пределах одной ступени». Брют и сухое — одна ступень:
#: тихое сухое и игристое брют по сахару рядом, а разводит их игристость, которая сравнивается
#: отдельно.
SUGAR_STEPS: dict[str, int] = {
    "brut_nature": -1,
    "extra_brut": 0,
    "brut": 1,
    "suhoe": 1,
    "polusuhoe": 2,
    "polusladkoe": 3,
    "sladkoe": 4,
}

#: Сахар, который бывает только у игристых.
SPARKLING_SUGARS = frozenset({"brut_nature", "extra_brut", "brut"})

#: Класс сахара в названии — таблица `RU_SUGAR` сборки разметки (`scripts/build_gt_tokens.py`,
#: `ru_sugar`), по порядку: первый найденный класс и есть класс названия. «Экстра брют»
#: поглощает «брют», «Brut Zero Dosage» — брют натюр. Одно отличие от `RU_SUGAR`: «сухое» и
#: «сладкое» — только целым словом. Опечатку выгрузки «Абрау Купаж красный полсусладкое» таблица
#: разметки читала как «сладкое», хотя slug позиции говорит `polusladkoe`; теперь в названии сахара
#: нет, и он берётся из slug. Разметку (`gt_tokens.jsonl`) это не трогает: её читает распознавание.
_NAME_SUGAR: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), sugar)
    for pattern, sugar in (
        (r"экстра[\s\-]*брют|extra[\s\-]*brut", "extra_brut"),
        (r"брют[\s\-]*натюр|brut[\s\-]*nature|zero[\s\-]*dosage|pas[\s\-]*dos", "brut_nature"),
        (r"полусух|semi[\s\-]*dry", "polusuhoe"),
        (r"полуслад|semi[\s\-]*sweet|demi[\s\-]*sec", "polusladkoe"),
        (r"(?<![а-яё])сухое|(?<!semi[\s\-])\bdry\b", "suhoe"),
        (r"(?<![а-яё])сладкое|десертн", "sladkoe"),
        (r"(?<!экстра )(?<!экстра-)брют|\bbrut\b", "brut"),
    )
)
#: Слова сахара в slug выгрузки. Хвост slug «цвет-сахар-крепость» пишет категорию организатора,
#: поэтому берётся последнее слово: «…-igristoe-sladkoe-krasnoe-bryut-125» — брют.
_SLUG_SUGAR = re.compile(
    r"(?<![a-z0-9])(ekstra-bryut|extra-brut|bryut-natyur|brut-nature"
    r"|polusladkoe|polusuhoe|sladkoe|suhoe|bryut|brut)(?![a-z0-9])"
)
_SLUG_SUGAR_CLASS = {
    "ekstra-bryut": "extra_brut",
    "extra-brut": "extra_brut",
    "bryut-natyur": "brut_nature",
    "brut-nature": "brut_nature",
    "polusladkoe": "polusladkoe",
    "polusuhoe": "polusuhoe",
    "sladkoe": "sladkoe",
    "suhoe": "suhoe",
    "bryut": "brut",
    "brut": "brut",
}
#: Слова игристого — те же, что на этикетке (`after_layer` читает ими строки VLM). Пет-нат пишут
#: слитно, через дефис или пробел («Петнат», «Пет-Нат», «Пет Нат»), а в slug — и через
#: подчёркивание (`denisov_pet_nat_rubin`).
SPARKLING_WORDS = (
    r"игрист|шампанск|шипуч|(?<![а-яё])пет[\s-]?нат|просекко|креман"
    r"|sparkling|spumante|frizzante|prosecco|cremant|crémant|pet[\s_-]?nat|p[eé]tillant"
)
#: В названии и slug выгрузки — ещё и транслитом: «…-igristoe-…», «shampanskoe».
_SPARKLING_FACT = re.compile(SPARKLING_WORDS + r"|igrist|shampansk")
#: Слова креплёного, десертного и мускатного вина в названии: «Портвейн Крымский», «Массандра
#: Херес», «Grand Dessert Nectar», «Массандра Мускат», «Поздний сбор, Белое». У 27 позиций
#: выгрузки с таким названием сахара нет ни в названии, ни в slug; у 24 из них ни название
#: («Мускат Оранж»), ни «Описание» не говорят, сухое вино («абсолютная сухость») или сладкое
#: («Сладкое розовое вино» у «Мускат позднего сбора розовый» — сахар по карточке, `card_sugar`).
#: Сухим тихим вином их считать нельзя — ни в подаче (правило `sweet_name` таблицы подачи), ни
#: в блюдах: правила сочетаний судят их как возможно сладкие (`scripts/build_somm.py`,
#: `rule_sugar`). Мускат в выгрузке чаще сладкий или креплёный, чем сухой, поэтому он здесь же.
#: «Поздний сбор» в именительном падеже (две позиции без сахара) и ледяное вино добавлены при
#: проверке 25.09, вечер.
SWEET_STYLE_HINT = re.compile(
    r"портвейн|мадер[аы]|херес|кагор|марсал|токай|ликёрн|ликерн|десерт|поздн\w*\s+сбор|айсвайн"
    r"|ледян\w*\s+вин|мускат|\bport\b|sherry|madeira|dessert|late harvest|ice\s*wine|muscat"
    r"|moscato",
    re.IGNORECASE,
)
#: Название оранжевого вина: «Мускат Оранж» — вино на мезге, а не десертный мускат (проверка
#: 25.09). Такое название догадку «сладкое по названию» снимает.
ORANGE_NAME = re.compile(r"оранж|orange", re.IGNORECASE)
#: Слова сухости в «Описании» организатора: «сухое», «сухость» («несмотря на абсолютную сухость
#: вина» у «Жемчужная 9 Мускат Розовый, Пино гри»), «полусухое», «брют». Только целые формы слова:
#: «сухофрукты» — не сухость. Описание прямо говорит, что вино не сладкое, — догадка по названию
#: снимается (проверка 25.09).
DRY_DESCRIPTION = re.compile(
    r"(?<![а-яё])(?:полу)?сух(?:ое|ой|ая|ую|ие|ого|ому|ость|ости|остью)(?![а-яё])"
    r"|(?<![а-яё])брют",
    re.IGNORECASE,
)
#: «Описание» организатора прямо называет вино сладким: «сладкое», «сладкий», «десертное»,
#: «десертный» или «ликёрное» целым словом, а за ним, не дальше чем через два слова, — «вино»
#: («Сладкое розовое вино ЗГУ Крым», «Сладкое вино из сорта Мускат розовый» у «Мускат позднего
#: сбора розовый»; проверка 25.09). Слово «вино» обязательно: без него «сладкий» — о вкусе
#: и аромате («Сладкий, лакрично-черничный аромат» у красного VIBES без сахара в выгрузке,
#: «сладкий персик» у брюта, «вкус сладкий» у полусладких), а «сладких специй», «десертной
#: сладостью», «полусладкое вино» — другие слова. «Не сладкое вино» — не сладкое. На выгрузке
#: 25.09 шаблон находит одно вино из 2 103.
SWEET_DESCRIPTION = re.compile(
    r"(?<![а-яё])(?<!не )(?:сладк(?:ое|ий)|десертн(?:ое|ый)|лик[её]рное)"
    r"(?:\s+[а-яё-]+){0,2}\s+вино(?![а-яё])",
    re.IGNORECASE,
)
#: Сахар, который даёт «Описание» организатора (`card_sugar`).
DESCRIPTION_SUGAR = "sladkoe"

#: Фото выгрузки, на котором другое вино: все 23 строки `wrong_photo` правки эталонов 23.09
#: (`research/2026-09-23_packshot-fix/replacements.tsv`; тест сверяет список с файлом). Одно
#: правило на всех: чужую бутылку не показываем — `photo_url: null`, `/photo` отвечает 404,
#: страница рисует силуэт (договор «после поиска», §4). Замену правка брала с живого портала
#: или у Роскачества — после решения 24.09 её тоже нет. В пуле подбора вино остаётся: фото
#: выгрузки у него есть, просто на нём не оно, и плитка выходит с силуэтом. Векторы индекса
#: распознавания остаются как есть: их не показывают.
WRONG_PHOTOS: frozenset[str] = frozenset(
    {
        "abrau-dyurso-az-abrau-bayanshira-beloe-suhoe-12",
        "agora-yachting-cabernet-sauvignon",
        "agrolayn-heritage-dg-skin-contact-rkatsiteli-rkatsiteli-krasnoe-suhoe-12",
        "agrolayn-mountain-eagle-traminer-traminer-beloe-suhoe-12",
        "belmas-winery-syrah-katya-sira-krasnoe-suhoe-125",
        "belmas-winery-vi-vione-beloe-suhoe-122",
        "chateau-le-grand-vostock-krasnostop-rezerv-krasnoe-suhoe-145",
        "derbent-vino-endemy-saperavi-krasnoe-suhoe-13",
        "derbent-vino-endemy-shardone-beloe-suhoe-13",
        "esse-demi-sec-muscat-nectar-muskat-belyy-beloe-ekstra-bryut-115",
        "fanagoriya-velvet-season-muskat-ottonel-beloe-sladkoe-13",
        "fanagoriya-velvet-season-risling-beloe-sladkoe-12",
        "legato-legato-sovinon-blan-beloe-suhoe-125",
        "method-classic-kokur",
        "novyj-svet-vyderzhannoe-bryut",
        "one-barrel-uan-barrel",
        "pinot-noir-2024-one-barrel-by-dmitry-maslov-pino-nuar-2024-uan-barrel-dmitrij-maslov",
        "skalistyy-bereg-shyopot-tsvetov-risling-beloe-suhoe-109",
        "sober-bash-kaberne-fran-krasnoe-suhoe-127",
        "sober-bash-risling-risling-reynskiy-beloe-suhoe-11",
        "valeriy-zaharin-bastardo-kefesiya-avtohtonnoe-vino-kryma-bastardo-magarachskiy-krasnoe-suhoe-115",
        "vibes-vermentino-viognier-barrel-fermented-2022",
        "vinodelnya-vedernikov-tsimlyanskiy-chernyy-rezerv-krasnoe-suhoe-145",
    }
)

_SPACES = re.compile(r"\s+")


# ------------------------------------------------------------------ ссылка на портал
@cache
def unlisted_slugs(path: Path = PORTAL_LINKS_PATH) -> frozenset[str]:
    """Slug выгрузки, которых нет в карте сайта вин организатора; нет файла — пусто."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Список публикации %s не читается (%s): ссылка есть у всех", path, exc)
        return frozenset()
    return frozenset(str(slug) for slug in data.get("unlisted") or [])


def portal_url_of(slug: str) -> str | None:
    """`https://vino-svoe.ru/wines/<slug>` или `None`, если страницы вина на портале нет."""
    return None if slug in unlisted_slugs() else f"{PORTAL_WINE_URL}{slug}"


# ------------------------------------------------------------------ мелочи
def clean_text(value: object) -> str:
    """Строка для показа: без переводов строк и двойных пробелов («ЛОРИО -  семейная»)."""
    return _SPACES.sub(" ", str(value)).strip() if value is not None else ""


def plausible_abv(value: object) -> float | None:
    """Крепость числом, если она правдоподобна (3–25); иначе `None`."""
    try:
        number = float(str(value).strip().rstrip("%").replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if ABV_MIN <= number <= ABV_MAX else None


def number_label(value: float) -> str:
    """13.5 → «13,5», 14.0 → «14»: русская запятая, без хвостовых нулей."""
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _abv_values(value: object) -> tuple[float, ...]:
    """`abv`: число, список (диапазон «10,5–12,5») или пусто."""
    items = value if isinstance(value, list | tuple) else [value]
    out = []
    for item in items:
        if item is None:
            continue
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


# ------------------------------------------------------------------ факты выгрузки
def sugar_of_name(name: object) -> str | None:
    """Класс сахара, названный в «Название вина»; первый по таблице `RU_SUGAR` разметки."""
    text = str(name or "").lower()
    found = [sugar for pattern, sugar in _NAME_SUGAR if pattern.search(text)]
    if "extra_brut" in found and "brut" in found:
        found.remove("brut")
    return found[0] if found else None


def sugar_of_slug(slug: object) -> str | None:
    """Последнее слово сахара в slug выгрузки: `…-krasnoe-suhoe-135` — сухое."""
    words = _SLUG_SUGAR.findall(str(slug or "").lower().replace("_", "-"))
    return _SLUG_SUGAR_CLASS[words[-1]] if words else None


def organizer_sugar(name: object, slug: object) -> str | None:
    """Правило сахара договора (§3): название, затем slug, иначе неизвестен."""
    return sugar_of_name(name) or sugar_of_slug(slug)


def organizer_sparkling(sugar: str | None, name: object, slug: object) -> bool:
    """Игристое: сахар брют-семейства или слово игристого в названии либо slug."""
    if sugar in SPARKLING_SUGARS:
        return True
    text = f"{name or ''} {slug or ''}".casefold().replace("ё", "е")
    return bool(_SPARKLING_FACT.search(text))


def sweet_description(description: object) -> bool:
    """«Описание» организатора прямо называет вино сладким (`SWEET_DESCRIPTION`) и нигде не
    называет его сухим (`DRY_DESCRIPTION`): противоречивое описание фактом не считается."""
    text = str(description or "")
    return bool(SWEET_DESCRIPTION.search(text)) and not DRY_DESCRIPTION.search(text)


def card_sugar(sugar: str | None, description: object) -> str | None:
    """Сахар по карточке для сомелье: сахар выгрузки (`organizer_sugar`: название, затем slug), а
    без него — «сладкое», если так прямо написано в «Описании» организатора (`sweet_description`).

    Это факт карточки, а не догадка: «Мускат позднего сбора розовый» — «Сладкое розовое вино» в
    описании (проверка 25.09), и к рыбе он сладкий по карточке — «скорее нет», а не
    оговорка «если вино сладкое…». Карточка «после поиска» (§3 договора) сахар по-прежнему берёт
    только из названия и slug; описание читает сомелье (`scripts/build_somm.py`, `rule_sugar`;
    `app/sommelier/answers.py`, `Sommelier.anchor`).
    """
    if sugar is not None:
        return sugar
    return DESCRIPTION_SUGAR if sweet_description(description) else None


def sweet_name(name: object, sugar: str | None, description: object = "") -> bool:
    """«Возможно сладкое по названию»: сахар неизвестен и по карточке (`card_sugar`: ни название,
    ни slug, ни «Описание» его не называют), а название — креплёного, десертного или мускатного
    вина.

    Это догадка, а не факт карточки: мускат и херес бывают и сухими. Поэтому она снимается, если
    название — оранжевого вина (`ORANGE_NAME`: «Мускат Оранж») или «Описание» организатора
    называет вино сухим (`DRY_DESCRIPTION`), и к рыбе даёт оговорку, а не «скорее нет»
    (`scripts/build_somm.py`, правило `maybe_sweet_wine_on_fish`). «Сладкое вино» в «Описании» —
    уже не догадка, а сахар по карточке (`card_sugar`).
    """
    text = str(name or "")
    return (
        card_sugar(sugar, description) is None
        and bool(SWEET_STYLE_HINT.search(text))
        and not ORANGE_NAME.search(text)
        and not DRY_DESCRIPTION.search(str(description or ""))
    )


def slug_abv(values: object, year: object) -> tuple[float, ...]:
    """Крепость из slug без года урожая: у `daniel-22` («Daniel, 2022») хвост `22` — это год.

    Разбор slug в разметке (`fields.abv.value`) читает последнее число slug как крепость. Целое
    число, равное двум последним цифрам года урожая позиции, — год, а не крепость; такое число
    отбрасывается. Год — `fields.year.value` разметки (из названия или slug).
    """
    numbers = _abv_values(values)
    vintage = _year(year)
    if vintage is None:
        return numbers
    return tuple(
        number for number in numbers if not (number.is_integer() and int(number) == vintage % 100)
    )


def style_label(color: str | None, sugar: str | None, sparkling: bool) -> str:
    """Цвет и сахар: «Красное сухое», «Белое брют»; сахар неизвестен — «Белое игристое» у
    игристого, иначе один цвет (договор «после поиска», §3)."""
    tail = SUGAR_WORDS.get(sugar or "", "") or ("игристое" if sparkling else "")
    return " ".join(part for part in (color or "", tail) if part)


# ------------------------------------------------------------------ крепость
@dataclass(frozen=True, slots=True)
class Alcohol:
    """Крепость карточки: значение, верх диапазона и откуда взята (`catalog` — из slug)."""

    value: float | None = None
    max: float | None = None
    src: str | None = None


def alcohol_of(abv: Iterable[float]) -> Alcohol:
    """Правило договора (§3): одно число — крепость, два — диапазон; вне 3–25 — неизвестна."""
    values = [value for value in abv if plausible_abv(value) is not None]
    if not values:
        return Alcohol()
    low, high = min(values), max(values)
    return Alcohol(low, high if high > low else None, "catalog")


def degrees_label(alcohol: Alcohol) -> str:
    """Крепость для объяснений похожих: «13,5°», «10–12,5°». Знака процента нет."""
    if alcohol.value is None:
        return ""
    if alcohol.max is not None:
        return f"{number_label(alcohol.value)}–{number_label(alcohol.max)}°"
    return f"{number_label(alcohol.value)}°"


# ------------------------------------------------------------------ вино
@dataclass(frozen=True, slots=True)
class RecoWine:
    """Одна позиция выгрузки: наша группировка и факты карточки для сравнения."""

    slug: str
    wine_id: str
    canonical: str
    is_canonical: bool
    winery: str
    winery_norm: str
    title: str
    grapes: tuple[str, ...]  # коды таксономии: syrah, chardonnay
    grapes_src: str | None
    color: str | None
    sugar: str | None  # код SugarClass
    sparkling: bool
    abv: tuple[float, ...]  # одно число или диапазон
    region: str
    photo: str | None  # имя фото выгрузки; None — фото нет
    year: int | None = None  # год урожая из slug или названия — для профиля стиля
    photo_wrong: bool = False  # на фото выгрузки другое вино (`WRONG_PHOTOS`): не показывается

    @property
    def in_pool(self) -> bool:
        """Входит ли в пул похожих: каноническое и с фото выгрузки (пусть и чужой бутылки)."""
        return self.is_canonical and bool(self.photo)

    @property
    def photo_shown(self) -> bool:
        """Показывается ли фото: оно есть, и на нём это вино."""
        return bool(self.photo) and not self.photo_wrong

    @property
    def alcohol(self) -> Alcohol:
        return alcohol_of(self.abv)

    @property
    def sugar_word(self) -> str:
        return SUGAR_WORDS.get(self.sugar or "", "")

    @property
    def style_label(self) -> str:
        return style_label(self.color, self.sugar, self.sparkling)

    @property
    def grape_labels(self) -> list[str]:
        """Сорта для показа: подписи таксономии («Сира»)."""
        return [grape_label(code) for code in self.grapes]

    @property
    def portal_url(self) -> str | None:
        return portal_url_of(self.slug)


def wine_of(row: Mapping[str, Any]) -> RecoWine:
    """Строка фактов (`organizer_row` или тестовая) → позиция справочника."""
    slug = str(row["slug"])
    color = clean_text(row.get("color")) or None
    sugar = clean_text(row.get("sugar")) or None
    grapes_src = row.get("grapes_src")
    return RecoWine(
        slug=slug,
        wine_id=str(row.get("wine_id") or slug),
        canonical=str(row.get("canonical") or slug),
        is_canonical=bool(row.get("is_canonical")),
        winery=clean_text(row.get("winery")),
        winery_norm=clean_text(row.get("winery_norm")) or clean_text(row.get("winery")).casefold(),
        title=clean_text(row.get("title")) or slug,
        grapes=tuple(str(code) for code in row.get("grapes") or [] if code),
        grapes_src=str(grapes_src) if grapes_src else None,
        color=color if color in COLORS else None,
        sugar=sugar if sugar in SUGAR_STEPS else None,
        sparkling=bool(row.get("sparkling")),
        abv=_abv_values(row.get("abv")),
        region=clean_text(row.get("region")),
        photo=str(row.get("photo") or "") or None,
        year=_year(row.get("year")),
        photo_wrong=bool(row.get("photo_wrong")),
    )


def _year(value: object) -> int | None:
    """Год урожая: целое 1950–2049, иначе ничего."""
    try:
        year = int(str(value).strip()) if value is not None else None
    except ValueError:
        return None
    return year if year is not None and 1950 <= year <= 2049 else None


def _field(record: Mapping[str, Any], name: str, key: str) -> Any:
    block = (record.get("fields") or {}).get(name)
    return block.get(key) if isinstance(block, Mapping) else None


def organizer_row(
    record: Mapping[str, Any],
    grouping: Mapping[str, Any] | None = None,
    *,
    wrong_photos: frozenset[str] = WRONG_PHOTOS,
) -> dict[str, Any]:
    """Запись разметки (`gt_tokens.jsonl`) и строка словаря групп → факты позиции выгрузки.

    Из разметки — поля выгрузки как есть и крепость из slug; сахар и игристость — по правилам
    договора, а не классом разметки. Из словаря групп — только группа, каноническая позиция,
    ключ винодельни и коды сортов колонки «Сорт винограда» (`grapes_src = "csv"`). Нет строки
    словаря — вино само себе группа, сорта — коды разметки из той же колонки.
    """
    slug = str(record["slug"])
    name = clean_text(record.get("name"))
    grouping = grouping or {}
    if grouping:
        from_csv = grouping.get("grapes_src") == "csv"
        grapes = [str(code) for code in grouping.get("grapes") or []] if from_csv else []
        grapes_src = "csv" if grapes else None
    else:
        codes = _field(record, "grape", "codes") or []
        grapes = [code for code in codes if isinstance(code, str) and code in GRAPE_SYNONYMS]
        grapes_src = "csv" if grapes else None
    sugar = organizer_sugar(name, slug)
    photo = clean_text(record.get("photo_name"))
    return {
        "slug": slug,
        "wine_id": grouping.get("wine_id") or slug,
        "canonical": grouping.get("canonical") or slug,
        "is_canonical": bool(grouping.get("is_canonical", True)),
        "winery": clean_text(record.get("winery")),
        "winery_norm": grouping.get("winery_norm"),
        "title": name,
        "grapes": list(dict.fromkeys(grapes)),
        "grapes_src": grapes_src,
        "color": clean_text(record.get("category")),
        "sugar": sugar,
        "sparkling": organizer_sparkling(sugar, name, slug),
        "abv": list(slug_abv(_field(record, "abv", "value"), _field(record, "year", "value"))),
        "region": clean_text(record.get("region")),
        "photo": photo or None,
        "photo_wrong": bool(photo) and slug in wrong_photos,
        "year": _field(record, "year", "value"),
    }


# ------------------------------------------------------------------ справочник
class RecoCatalog:
    """Все позиции выгрузки в памяти и индексы для подбора: 2 103 записи, пара мегабайт."""

    def __init__(
        self,
        wines: Iterable[RecoWine],
        *,
        groups: Mapping[str, tuple[str, ...]] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.by_slug: dict[str, RecoWine] = {}
        for wine in wines:
            if wine.slug in self.by_slug:
                raise ValueError(f"RecoCatalog: повтор slug {wine.slug}")
            self.by_slug[wine.slug] = wine
        self.groups: dict[str, tuple[str, ...]] = dict(groups or {})
        self.meta: dict[str, Any] = dict(meta or {})
        self.pool: tuple[RecoWine, ...] = tuple(w for w in self.by_slug.values() if w.in_pool)
        by_style: dict[tuple[str | None, bool], list[RecoWine]] = {}
        by_winery: dict[str, list[RecoWine]] = {}
        for wine in self.pool:
            by_style.setdefault((wine.color, wine.sparkling), []).append(wine)
            by_winery.setdefault(wine.winery_norm, []).append(wine)
        self._by_style = {key: tuple(items) for key, items in by_style.items()}
        self._by_winery = {
            key: tuple(sorted(items, key=name_key)) for key, items in by_winery.items()
        }
        self.winery_names: dict[str, str] = {}
        for wine in self.by_slug.values():
            self.winery_names.setdefault(wine.winery_norm, wine.winery)
        # Крепость по правилу договора считается один раз: подбор сравнивает её сотни раз.
        self._alcohol: dict[str, Alcohol] = {slug: w.alcohol for slug, w in self.by_slug.items()}

    def __len__(self) -> int:
        return len(self.by_slug)

    def __contains__(self, slug: object) -> bool:
        return isinstance(slug, str) and slug in self.by_slug

    def __iter__(self) -> Iterator[RecoWine]:
        return iter(self.by_slug.values())

    def get(self, slug: str) -> RecoWine | None:
        return self.by_slug.get(slug)

    def alcohol(self, slug: str) -> Alcohol:
        """Крепость карточки по правилу договора; неизвестный slug — пустая."""
        return self._alcohol.get(slug, Alcohol())

    def style_pool(self, color: str | None, sparkling: bool | None) -> tuple[RecoWine, ...]:
        """Вина пула того же цвета и игристости; `None` — признак не важен."""
        if color is not None and sparkling is not None:
            return self._by_style.get((color, sparkling), ())
        return tuple(
            wine
            for (wine_color, wine_sparkling), items in self._by_style.items()
            if (color is None or wine_color == color)
            and (sparkling is None or wine_sparkling == sparkling)
            for wine in items
        )

    def winery_pool(self, winery_norm: str) -> tuple[RecoWine, ...]:
        """Вина пула этой винодельни по названию."""
        return self._by_winery.get(winery_norm, ())

    def group_members(self, wine_id: str) -> tuple[str, ...]:
        return self.groups.get(wine_id, ())

    def stats(self) -> dict[str, Any]:
        """Сколько позиций и какие факты известны — для `/v1/health.after` и зондов."""
        wines = list(self)
        return {
            "wines": len(wines),
            "pool": len(self.pool),
            "groups": len(self.groups),
            "sugar_known": sum(wine.sugar is not None for wine in wines),
            "sparkling": sum(wine.sparkling for wine in wines),
            "abv_known": sum(wine.alcohol.value is not None for wine in wines),
            "abv_ranges": sum(wine.alcohol.max is not None for wine in wines),
            "with_photo": sum(wine.photo_shown for wine in wines),
            "photo_wrong": sum(wine.photo_wrong for wine in wines),
            **self.meta,
        }

    # -------------------------------------------------------------- сборка
    @classmethod
    def build(
        cls,
        records: Iterable[Mapping[str, Any]],
        grouping: Iterable[Mapping[str, Any]] = (),
        *,
        groups: Mapping[str, Iterable[str]] | None = None,
        wrong_photos: frozenset[str] = WRONG_PHOTOS,
        meta: Mapping[str, Any] | None = None,
    ) -> RecoCatalog:
        """Справочник из записей разметки и нашего словаря групп.

        Позиции — только записи разметки (выгрузка организатора). Строки словаря и члены групп
        вне выгрузки (73 карточки живого портала) пропускаются. Группа без канонической позиции
        выгрузки получает её заново: первая позиция группы с фото, иначе первая.
        """
        records = [record for record in records if record.get("slug")]
        ours = {str(record["slug"]) for record in records}
        by_slug = {str(row["slug"]): row for row in grouping if row.get("slug") in ours}
        members: dict[str, list[str]] = {}
        for wine_id, slugs in (groups or {}).items():
            kept = [str(slug) for slug in slugs if str(slug) in ours]
            if kept:
                members[str(wine_id)] = kept
        rows = [organizer_row(r, by_slug.get(str(r["slug"])), wrong_photos=wrong_photos)
                for r in records]  # fmt: skip
        for row in rows:  # член группы, которого нет в составе групп, — в конец группы
            group = members.setdefault(row["wine_id"], [])
            if row["slug"] not in group:
                group.append(row["slug"])
        with_photo = {row["slug"] for row in rows if row["photo"]}
        shown = {row["slug"] for row in rows if row["photo"] and not row["photo_wrong"]}
        named: dict[str, set[str]] = {}
        for row in rows:
            named.setdefault(row["wine_id"], set()).add(row["canonical"])
        canonical: dict[str, str] = {}
        for wine_id, slugs in members.items():
            chosen = next((slug for slug in slugs if slug in named.get(wine_id, ())), None)
            if chosen is None:  # канонической в словаре была карточка живого портала
                chosen = next(
                    (slug for slug in slugs if slug in shown),
                    next((slug for slug in slugs if slug in with_photo), slugs[0]),
                )
            canonical[wine_id] = chosen
        wines = []
        for row in rows:
            row["canonical"] = canonical[row["wine_id"]]
            row["is_canonical"] = row["slug"] == row["canonical"]
            wines.append(wine_of(row))
        return cls(wines, groups={key: tuple(value) for key, value in members.items()}, meta=meta)

    @classmethod
    def load(
        cls,
        tokens_path: Path,
        *,
        wines_path: Path | None = None,
        groups_path: Path | None = None,
        wrong_photos: frozenset[str] = WRONG_PHOTOS,
    ) -> RecoCatalog:
        """Справочник с диска: разметка обязательна, словарь групп — по возможности.

        Нет словаря групп — каждое вино само себе группа (с предупреждением): похожие остаются,
        но «одно вино в другом объёме» из них уже не убирается.
        """
        records = read_jsonl(tokens_path)
        grouping: list[dict[str, Any]] = []
        if wines_path is not None and wines_path.is_file():
            grouping = read_jsonl(wines_path)
        elif wines_path is not None:
            logger.warning("Нет словаря групп %s: каждое вино — своя группа", wines_path)
        groups: dict[str, list[str]] = {}
        if groups_path is not None and groups_path.is_file():
            raw = json.loads(groups_path.read_text(encoding="utf-8"))
            groups = {
                str(wine_id): [str(slug) for slug in (group or {}).get("members") or []]
                for wine_id, group in raw.items()
            }
        meta = {
            "tokens_source": tokens_path.name,
            "groups_source": wines_path.name if grouping else None,
        }
        return cls.build(records, grouping, groups=groups, wrong_photos=wrong_photos, meta=meta)


def name_key(wine: RecoWine) -> tuple[str, str]:
    """Порядок «по названию»: без регистра, при равных названиях — slug."""
    return (wine.title.casefold(), wine.slug)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: не JSON: {exc}") from exc
    return rows


def known_grape(code: object) -> str | None:
    """Код сорта таксономии или `None`: шум чтения («каберне», «совиньон») отбрасывается."""
    text = str(code or "").strip()
    return text if text in GRAPE_SYNONYMS else None
