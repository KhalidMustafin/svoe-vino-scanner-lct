"""Слой «после поиска»: карточка, фото, подсказка экрана, похожие, «Не тупик», «Сомелье у полки».

Договор ответов — `docs/api-after-search.md`, маршруты — `app.api.after`. Здесь логика без
FastAPI: её собирает `ScannerService`, а сервис импортируют и стенды без зависимостей `api`.

Два поля ответа `/v1/scan` (`ScannerService.scan_body`): `candidates` — top-5 с именем,
винодельней и фото, без счётов; `after` — какой экран открыть и что прочитано на этикетке.
В `/v1/eval/predict` ни то, ни другое не попадает: его тело — `ScanResult.predict_body()`.

Правило экрана на Д1 (договор, §2): ошибка → `error`; отказ, нет вероятности, спорный кадр
или p(top-1) < 0,5 → `check`; иначе `found`. `not_found` сервер не возвращает: калибровка Д2
(`research/2026-09-24_after/hint.md`) дала полноту 19 % при ≤ 2 % ложных, и по плану этот
экран открывает кнопка «Моего вина здесь нет». Правило подсказки в коде есть и выключено
(`NOT_FOUND_VISUAL_MAX`).

Подсказка Д3 `suggest_not_found`: счёт CV лучшей серии ниже `SUGGEST_NOT_FOUND_VISUAL_MAX` —
флаг поднят, `found` становится `check`, в причинах `visual_low`. Экрана `not_found` она не
открывает и slug не меняет: скрипт организатора читает только top-1 `/v1/eval/predict`.

Винодельня этикетки (`AfterSearch.resolve_winery`) считается только по однозначному
написанию: родовое слово («шато», «villa») и слово нескольких виноделен её не называют. Чужую
винодельню словарь каталога не знает — её видно в строках VLM по слову хозяйства
(`winery_of_lines`), и тогда `winery_in_catalog: false`.

Без портала (решение 24.09): карточка, плитки и подбор — только выгрузка организатора
(`app.recommend.catalog`) и наши правила; блюда чипа «К чему?» — правила сочетаний
(`data/somm/`, `app.recommend.somm_data`). Снимок портала, иконки блюд, 73 живые карточки вне
выгрузки и их фото, «опубликовано» и сахар живого портала слой не читает; фото — только
выгрузки (`photo_file`).

Право (план, §8): в блоках рекомендаций нет знака процента, цен и эпитетов; при
`order=reco` — плашка «Применяются рекомендательные технологии», `order=plain` — обычная
сортировка. Каждая строка объяснения проходит `content_filter` ещё и во время ответа: не
прошедшая строка выбрасывается (и пишется в журнал), а не уходит пользователю.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.api.cards import CatalogCards
from app.reading.lexicon.build import GENERIC_WORDS
from app.reading.lexicon.correct import bounded_levenshtein
from app.reading.taxonomy import (
    LABEL_GENERIC,
    LABEL_TERMS,
    WINE_WORDS,
    canonical_grape,
    color_of,
    grape_label,
    sugar_class,
)
from app.reading.text.translit import romanize, skeleton
from app.recommend import facts as reco
from app.recommend.catalog import (
    COLORS,
    GROUPS_NAME,
    SPARKLING_SUGARS,
    SPARKLING_WORDS,
    SUGAR_WORDS,
    WRONG_PHOTOS,
    Alcohol,
    RecoCatalog,
    RecoWine,
    alcohol_of,
    clean_text,
    known_grape,
    name_key,
    number_label,
    organizer_sparkling,
    organizer_sugar,
    plausible_abv,
    portal_url_of,
    style_label,
)
from app.recommend.content_filter import check
from app.recommend.shelf import Food, Shelf, ShelfAnswer, Want, shelf_tile
from app.recommend.somm_data import SommData, load_somm_data
from app.resolve.attrs import CatalogAttrs, norm_key
from app.sommelier.templates import check_question

if TYPE_CHECKING:
    from app.api.config import ServiceSettings
    from app.api.service import ScanResult

logger = logging.getLogger(__name__)

#: Фото меняются только с пачкой данных: сутки кэша на телефоне.
CACHE_CONTROL = "public, max-age=86400"
#: Путь карты фото не из выгрузки: правка эталонов 23.09 (`method = packshot_fix`) положила туда
#: фото живого портала и Роскачества (`data/catalog/packshots_fixed/`). Их не показываем.
FOREIGN_PHOTO_METHODS = frozenset({"packshot_fix"})
FOREIGN_PHOTO_DIR = "packshots_fixed"
#: Коды ошибок `/v1/scan` — префикс поля `error` до двоеточия (договор, §2).
ERROR_CODES = frozenset(
    {"no_image", "bad_request", "too_large", "decode", "cv", "not_ready", "internal"}
)
#: Порог «проверьте» — тот же, что у `outcome="ambiguous"` сервиса (`service.AMBIGUOUS_P_TOP1`).
CHECK_P_TOP1 = 0.5
#: Экран `not_found` (план, Д2, п. 6): порог счёта CV лучшей серии (zmax), ниже которого
#: срабатывает правило «винодельня прочитана, ни одна её позиция не сходится». `None` — сервер
#: этот экран не возвращает. Калибровка на записанных ответах поля
#: (`research/2026-09-24_after/hint.md`) при ≤ 7 ложных на 353 кадрах каталога дала полноту на
#: 409 кадрах вне каталога ниже 50 %, и по плану автоэкрана нет: остаются `check` и кнопка
#: «Моего вина здесь нет». Замер повторяется `calibrate_hint.py` на новом прогоне.
NOT_FOUND_VISUAL_MAX: float | None = None
#: Подсказка «похоже, этой бутылки может не быть в каталоге» (`after.suggest_not_found`, Д3):
#: порог счёта CV лучшей серии (zmax, `best_visual` — то же значение, что у `NOT_FOUND_VISUAL_MAX`
#: и `calibrate_hint.py`). Ниже порога флаг поднят: экран `found` становится `check`, в `reasons`
#: добавляется `visual_low`. Экрана `not_found` флаг не открывает, slug и predict не трогает.
#: Правило калибровки (`research/2026-09-24_after/hint.md`, «Подсказка suggest_not_found»):
#: наибольший порог, при котором флаг стоит не больше чем на 7 из 353 кадров каталога (≤ 2 %).
#: Порог подобран на тех же кадрах: флаг на 6 из 353, на 409 кадрах вне каталога — на 263.
#: Вне выборки (5 фолдов по винам) — медиана 8 из 353 и 58 % вне каталога. `None` — флага нет.
SUGGEST_NOT_FOUND_VISUAL_MAX: float | None = 0.8024
#: Код причины подсказки в `after.reasons` — не текст для человека (договор, §2).
VISUAL_LOW = "visual_low"
#: Тип файла по расширению. `mimetypes` на Windows не знает `.webp` и отдаёт
#: `application/octet-stream` — договор обещает `image/webp`.
IMAGE_TYPES = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".avif": "image/avif",
    ".gif": "image/gif",
}


def image_type(path: Path) -> str:
    return IMAGE_TYPES.get(path.suffix.lower(), "application/octet-stream")


#: Слова игристого на этикетке. Сахар «брют» тоже говорит об игристости, но он приходит полем.
_SPARKLING_WORDS = re.compile(SPARKLING_WORDS)

State = Literal["found", "check", "not_found", "error"]
#: Поля `after.read`, кроме винодельни: прочитано ли с этикетки хоть что-то.
LABEL_KEYS = ("color", "sugar", "grapes", "abv", "sparkling")
Order = Literal["reco", "plain"]


# ------------------------------------------------------------------ экран
def scan_state(result: ScanResult) -> tuple[State, list[str]]:
    """Какой экран открыть по ответу сканера — правило Д1 договора, сверху вниз."""
    if result.outcome == "error":
        code = str(result.error or "internal").split(":", 1)[0].strip()
        return "error", [code if code in ERROR_CODES else "internal"]
    if result.outcome == "out_of_catalog":
        return "check", ["abstain"]
    top1 = result.confidence.top1
    if top1 is None:
        return "check", ["no_confidence"]
    if result.outcome == "ambiguous":
        return "check", ["ambiguous"]
    if top1 < CHECK_P_TOP1:
        return "check", ["low_confidence"]
    return "found", []


# ------------------------------------------------------------------ прочитанное с этикетки
@dataclass(frozen=True, slots=True)
class LabelRead:
    """Поля этикетки в кодах каталога. Пустое — «не прочитано», а не «нет на этикетке»."""

    wineries: tuple[str, ...] = ()  # как прочитано: сначала полная форма словаря, потом токены
    color: str | None = None
    sugar: str | None = None
    grapes: tuple[str, ...] = ()
    abv: float | None = None
    sparkling: bool | None = None
    house: str | None = None  # винодельня в строках со словом хозяйства (`winery_of_lines`)


def _color(value: object) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    color = color_of(text)
    return color.value if color is not None and color.value in COLORS else None


def _sugar(value: object) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    sugar = sugar_class(text)
    return sugar.value if sugar is not None and sugar.value in SUGAR_WORDS else None


def _grapes(values: Iterable[object]) -> tuple[str, ...]:
    """Сорта в кодах таксономии; шум чтения («каберне», «совиньон» по отдельности) отброшен."""
    codes: list[str] = []
    for value in values:
        text = clean_text(value)
        code = known_grape(text) or (canonical_grape(text) if text else None)
        if code and known_grape(code) and code not in codes:
            codes.append(code)
    return tuple(codes)


def read_of_evidence(evidence: Mapping[str, Any]) -> LabelRead:
    """Прочитанное с этикетки из доказательств скана (`evidence.vlm.fields` и строки VLM).

    Сорта — только коды таксономии: у чтения рядом с `cabernet_sauvignon` лежат токены
    «каберне» и «совиньон», это не сорта. Два разных сахара сразу — сахар не прочитан.
    Игристое — сахар брют-семейства или слово «игристое», «шампанское», spumante… в строках.
    Винодельня вне словаря каталога видна только в строках (`winery_of_lines`).
    """
    vlm = evidence.get("vlm") or {}
    fields = vlm.get("fields") or {}
    if not isinstance(fields, Mapping):
        return LabelRead()
    wineries = tuple(w for w in (clean_text(v) for v in fields.get("winery") or []) if w)
    sugars = {s for s in (_sugar(v) for v in fields.get("sugar") or []) if s}
    sugar = next(iter(sugars)) if len(sugars) == 1 else None
    grapes = tuple(code for v in fields.get("grapes") or [] if (code := known_grape(v)))
    lines = [str(line) for line in vlm.get("lines") or []]
    text = " ".join(lines).casefold().replace("ё", "е")
    sparkling = True if sugar in SPARKLING_SUGARS or _SPARKLING_WORDS.search(text) else None
    return LabelRead(
        wineries=wineries,
        color=_color(fields.get("color")),
        sugar=sugar,
        grapes=tuple(dict.fromkeys(grapes)),
        abv=plausible_abv(fields.get("abv")),
        sparkling=sparkling,
        house=winery_of_lines(lines),
    )


def read_label(read: Mapping[str, Any]) -> str | None:
    """«Фанагория · игристое · оранжевое · брют · Шардоне · 12 % об.»; ничего нет — `None`.

    Это строка карточки «Прочитали на этикетке: …», не блок рекомендаций: крепость здесь
    пишется так, как на этикетке, со знаком процента.
    """
    parts: list[str] = []
    if read.get("winery"):
        parts.append(str(read["winery"]))
    if read.get("sparkling"):
        parts.append("игристое")
    if read.get("color"):
        parts.append(str(read["color"]).lower())
    if read.get("sugar"):
        parts.append(SUGAR_WORDS.get(str(read["sugar"]), str(read["sugar"])))
    grapes = [grape_label(code) for code in read.get("grapes") or []]
    if grapes:
        parts.append(", ".join(grapes))
    if read.get("abv") is not None:
        parts.append(f"{number_label(float(read['abv']))} % об.")
    return " · ".join(parts) or None


# ------------------------------------------------------------------ винодельня в строках
#: Слова хозяйства перед именем: «Шато …», «Domaine …», «Усадьба …». «Винодельня» сюда не
#: входит: на этикетках каталога за ней чаще идёт кюве («винодельня AZUR» у Криницы).
_HOUSE_BEFORE = (
    "шато chateau domaine домен villa вилла усадьба имение поместье хутор "
    "cantina tenuta bodega bodegas weingut casa maison clos podere fattoria cascina"
)
#: Слова хозяйства после имени: «Muzaradi Winery», «Dugladze Wine Company».
_HOUSE_AFTER = "winery estate estates vineyard vineyards company"
#: Прочие родовые слова: винодельню они не называют («долина», «Винная усадьба», «Single
#: Vineyard»). «Долина» и «винный» в словаре каталога — у двух виноделен каждое, а
#: «хозяйство» и «кооператив» — у одной («Донское винодельческое хозяйство», «Винный
#: кооператив»), но слово и тогда родовое.
_HOUSE_OTHER = (
    "долина valley винный винная винное домашнее family семейная семейный завод змв дом "
    "house cellar cellars коллекция collection single хозяйство кооператив"
)
HOUSE_BEFORE = frozenset(norm_key(word) for word in _HOUSE_BEFORE.split())
HOUSE_AFTER = frozenset(norm_key(word) for word in _HOUSE_AFTER.split())
#: Слова, из которых одних винодельня не складывается: хозяйство, тип продукта, предлоги,
#: регионы (`GENERIC_WORDS` словаря) и общие слова этикетки.
GENERIC_WINERY_WORDS: frozenset[str] = (
    HOUSE_BEFORE
    | HOUSE_AFTER
    | frozenset(norm_key(word) for word in _HOUSE_OTHER.split())
    | GENERIC_WORDS
    | WINE_WORDS
    | LABEL_GENERIC
)
# Скелет ловит транслитерацию: «шато», «chateau» и «чатеау» словаря — один скелет «xato».
_GENERIC_SKELETONS = frozenset(skeleton(word) for word in GENERIC_WINERY_WORDS)
_HOUSE_BEFORE_SKELETONS = frozenset(skeleton(word) for word in HOUSE_BEFORE)
_HOUSE_AFTER_SKELETONS = frozenset(skeleton(word) for word in HOUSE_AFTER)
#: Больше слов в строке — это фраза («История крымского виноделия»), а не имя винодельни.
_NAME_MAX_WORDS = 5


def _generic_word(word: str) -> bool:
    return len(word) < 3 or word in GENERIC_WINERY_WORDS or skeleton(word) in _GENERIC_SKELETONS


def generic_winery(text: object) -> bool:
    """Написание из одних родовых слов («шато», «Domaine», «Винная усадьба») — не винодельня."""
    return all(_generic_word(word) for word in norm_key(text).split())


def _style_word(word: str) -> bool:
    """Цвет, сахар, сорт или серия: «КРАСНОЕ» под «УСАДЬБОЙ», «RESERVE Single Vineyard»."""
    return LABEL_TERMS.lookup(word) is not None or bool(color_of(word) or sugar_class(word))


def own_words(text: object) -> list[str]:
    """Свои слова написания: буквы, не короче трёх, не родовые, не цвет, сахар, сорт или серия."""
    return [
        word
        for word in norm_key(text).split()
        if word.isalpha() and not _generic_word(word) and not _style_word(word)
    ]


def _names_winery(text: str) -> bool:
    words = norm_key(text).split()
    return (
        0 < len(words) <= _NAME_MAX_WORDS
        and not any(_style_word(word) for word in words)
        and bool(own_words(text))
    )


def _house_position(text: str) -> tuple[bool, bool]:
    """Есть ли в строке слово хозяйства, которое стоит перед именем и после него."""
    words = norm_key(text).split()
    before = any(w in HOUSE_BEFORE or skeleton(w) in _HOUSE_BEFORE_SKELETONS for w in words)
    after = any(w in HOUSE_AFTER or skeleton(w) in _HOUSE_AFTER_SKELETONS for w in words)
    return before, after


def _house_name(texts: Sequence[str], i: int) -> str | None:
    """Имя хозяйства из строки `i`: сама строка или она со строкой-соседом."""
    text = texts[i]
    before, after = _house_position(text)
    if not (before or after):
        return None
    if own_words(text):
        return text if _names_winery(text) else None
    if before and i + 1 < len(texts) and _names_winery(texts[i + 1]):
        return f"{text} {texts[i + 1]}"
    if after and i > 0 and _names_winery(texts[i - 1]):
        return f"{texts[i - 1]} {text}"
    return None


def winery_of_lines(lines: Iterable[object]) -> str | None:
    """Винодельня, которую строки этикетки называют словом хозяйства; нет такой — `None`.

    Словарь каталога знает только винодельни каталога, поэтому чужая винодельня в поле
    `winery` чтения не попадает: её видно лишь в строках VLM («Château de Châtaignier»,
    «DOMAINE RENAUD», «Muzaradi Winery»). Имя считается прочитанным, когда в строке есть
    слово хозяйства и хотя бы одно своё слово (`own_words`), а цвета, сахара, сорта и серии
    нет. Слово хозяйства отдельной строкой берёт имя с соседней: «CHÂTEAU» / «de
    CHÂTAIGNIER», «Muzaradi» / «Winery». Имя без слова хозяйства («MASSIMO VISCONTI») не
    отличить от кюве или бренда — такого чтения нет. Строка уходит в ответ как прочитана,
    поэтому не прошедшая `content_filter` винодельней не считается.
    """
    texts = [text for text in (clean_text(line) for line in lines) if text]
    for i in range(len(texts)):
        name = _house_name(texts, i)
        if name is not None and check(name).clean:
            return name
    return None


@dataclass(frozen=True, slots=True)
class WineryMatch:
    """Чем оказалась прочитанная винодельня в каталоге."""

    name: str | None  # написание каталога, если нашлась; иначе как прочитано; None — не прочитана
    in_catalog: bool | None
    keys: tuple[str, ...] = ()  # винодельни справочника (winery_norm)


#: Сколько slug принимает `exclude` у «Не тупика»: пять кандидатов скана с запасом.
EXCLUDE_MAX = 20


class ByLabelBody(BaseModel):
    """Тело `POST /v1/similar/by-label` — объект `after.read` ответа `/v1/scan`.

    `exclude` — slug, которых в выдаче быть не должно (договор, §6): кандидаты скана, которые
    человек отверг кнопкой «Моего вина здесь нет». Не больше `EXCLUDE_MAX`, иначе — 422.
    """

    model_config = ConfigDict(extra="ignore")

    winery: str | None = Field(default=None, max_length=200)
    color: str | None = Field(default=None, max_length=40)
    sugar: str | None = Field(default=None, max_length=40)
    grapes: list[str] = Field(default_factory=list, max_length=20)
    abv: float | None = None
    sparkling: bool | None = None
    exclude: list[Annotated[str, Field(max_length=200)]] = Field(
        default_factory=list, max_length=EXCLUDE_MAX
    )


class CatalogWithout:
    """Справочник рекомендаций без исключённых позиций — пул подбора «Не тупика».

    Подбор (`reco.same_winery`, `reco.similar`, `reco.plain`) берёт вина только из
    `style_pool` и `winery_pool`: здесь они без исключённых, остальное — как у справочника.
    Исключение до подбора, а не после: «одна винодельня — одно место» и ослабление цвета
    считаются уже без отвергнутых вин, и мест в выдаче не становится меньше.
    """

    def __init__(self, catalog: RecoCatalog, slugs: frozenset[str]) -> None:
        self.catalog = catalog
        self.slugs = slugs

    def __getattr__(self, name: str) -> Any:
        return getattr(self.catalog, name)

    def style_pool(self, color: str | None, sparkling: bool | None) -> tuple[RecoWine, ...]:
        pool = self.catalog.style_pool(color, sparkling)
        return tuple(wine for wine in pool if wine.slug not in self.slugs)

    def winery_pool(self, winery_norm: str) -> tuple[RecoWine, ...]:
        pool = self.catalog.winery_pool(winery_norm)
        return tuple(wine for wine in pool if wine.slug not in self.slugs)


# ------------------------------------------------------------------ слой
def foreign_photo_slugs(photo_map: Path | None) -> frozenset[str]:
    """Slug, у которых путь карты фото ведёт не к фото выгрузки (правка эталонов 23.09).

    Нет карты или колонки `method` — пусто: тогда держит только проверка каталога
    `packshots_fixed` в самом пути (`photo_file`).
    """
    if photo_map is None or not photo_map.is_file():
        return frozenset()
    try:
        with photo_map.open(encoding="utf-8-sig", newline="") as handle:
            return frozenset(
                (row.get("slug") or "").strip()
                for row in csv.DictReader(handle)
                if (row.get("method") or "").strip() in FOREIGN_PHOTO_METHODS
            )
    except (OSError, csv.Error) as exc:
        logger.warning("Карта фото %s не читается: %s", photo_map, exc)
        return frozenset()


class AfterSearch:
    """Данные и логика «после поиска» поверх карточек и признаков каталога сервиса.

    `catalog` — справочник рекомендаций (может быть пустым: тогда похожих нет). `photo_dir` —
    лёгкие копии фото выгрузки (`<slug>.webp`). `somm` — данные правил сочетаний для чипа
    «К чему?» (нет — подбор к блюдам недоступен, остальное работает). `foreign_photos` — slug,
    чей путь карты фото ведёт не к фото выгрузки; `wrong_photos` — на фото выгрузки другое вино.
    """

    def __init__(
        self,
        catalog: RecoCatalog,
        cards: CatalogCards,
        attrs: CatalogAttrs,
        *,
        photo_dir: Path | None = None,
        somm: SommData | None = None,
        foreign_photos: frozenset[str] = frozenset(),
        wrong_photos: frozenset[str] = WRONG_PHOTOS,
    ) -> None:
        self.catalog = catalog
        self.cards = cards
        self.attrs = attrs
        self.photo_dir = photo_dir
        self.somm = somm if somm is not None else SommData()
        self.foreign_photos = foreign_photos
        self.wrong_photos = wrong_photos
        # Винодельни справочника по ключу нормы — и в написании выгрузки, и ключом словаря групп.
        self._winery_by_key: dict[str, str] = {}
        for wine in catalog:
            for text in (wine.winery, wine.winery_norm):
                key = norm_key(text)
                if key:
                    self._winery_by_key.setdefault(key, wine.winery_norm)
        # Все позиции винодельни — и вне пула похожих: «этой позиции нет» сверяется со всеми.
        positions: dict[str, list[str]] = {}
        for slug in dict.fromkeys([*catalog.by_slug, *attrs.by_slug]):
            key = self.winery_key(slug)
            if key:
                positions.setdefault(key, []).append(slug)
        self._positions = {key: tuple(slugs) for key, slugs in positions.items()}
        # Свои слова виноделен каталога — скелетом и латиницей, для сверки прочитанного имени.
        words: set[str] = set()
        for wine in catalog:
            words.update(own_words(wine.winery))
        for item in attrs:
            words.update(own_words(item.winery))
            for key in item.winery_keys:
                words.update(own_words(key))
        self._winery_skeletons = frozenset(skeleton(word) for word in words)
        self._winery_latin = frozenset(romanize(word) for word in words)
        self.not_found_max = NOT_FOUND_VISUAL_MAX
        self.suggest_max = SUGGEST_NOT_FOUND_VISUAL_MAX
        # Состав групп `wine_id` по справочнику: «исключить» у «Не тупика» снимает вино целиком.
        groups: dict[str, set[str]] = {}
        for wine in catalog:
            groups.setdefault(wine.wine_id, {wine.slug}).add(wine.slug)
        for wine_id, members in catalog.groups.items():
            groups.setdefault(wine_id, set()).update(members)
        self._group_of: dict[str, frozenset[str]] = {
            wine.slug: frozenset(groups[wine.wine_id]) for wine in catalog
        }
        self._photos: dict[str, Path | None] = {}
        #: «Сомелье у полки»: профили стиля всех вин справочника — один раз, ~40 мс.
        self.sommelier = Shelf(catalog, somm=self.somm)

    @classmethod
    def load(
        cls, settings: ServiceSettings, cards: CatalogCards, attrs: CatalogAttrs
    ) -> AfterSearch:
        """Справочник выгрузки, словарь групп и данные сомелье с диска. Нет файлов — без них.

        Без справочника сервис для скрипта организатора всё равно поднимается: `/v1/eval/predict`
        от этого слоя не зависит, а в журнале и `/v1/health` видно, чего не хватает.
        """
        catalog = RecoCatalog([])
        tokens = settings.attrs_path
        if tokens.is_file():
            try:
                catalog = RecoCatalog.load(
                    tokens,
                    wines_path=settings.wines_path,
                    groups_path=settings.wines_path.with_name(GROUPS_NAME),
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Справочник рекомендаций по %s не собирается: %s", tokens, exc)
        else:
            logger.warning("Нет разметки каталога %s: похожих нет", tokens)
        return cls(
            catalog,
            cards,
            attrs,
            photo_dir=settings.photo_dir,
            somm=load_somm_data(settings.somm_dir),
            foreign_photos=foreign_photo_slugs(settings.photo_map),
        )

    def stats(self) -> dict[str, Any]:
        return {
            **self.catalog.stats(),
            "photos_foreign": len(self.foreign_photos),
            "photos_wrong": len(self.wrong_photos),
            "somm_pairs": self.somm.stats()["pairs.json"],
        }

    # -------------------------------------------------------------- файлы
    def photo_file(self, slug: str, *, extra_dirs: Sequence[Path] = ()) -> Path | None:
        """Файл фото выгрузки организатора на этой машине или `None`. Путь наружу не отдаётся.

        Только позиции выгрузки (карточек 73 живых вин портала больше нет) и только фото
        выгрузки: лёгкая копия `<каталог фото>/<slug>.webp` (или по имени файла выгрузки), затем
        путь из `slug_photo_map.csv`. Путь карты, который правка эталонов 23.09 направила на
        фото живого портала или Роскачества (`foreign_photos`, каталог `packshots_fixed`), не
        берётся: у таких вин остаётся лёгкая копия исходного фото выгрузки. Фото, на котором
        другое вино (`wrong_photos`), не отдаётся вовсе.
        """
        card = self.cards.get(slug)
        if card is None or slug in self.wrong_photos:
            return None
        if not extra_dirs and slug in self._photos:
            return self._photos[slug]
        found: Path | None = None
        name = card.photo_name.strip()
        for directory in (*extra_dirs, self.photo_dir):
            if directory is None:
                continue
            for candidate in (directory / f"{slug}.webp", directory / name if name else None):
                if candidate is not None and candidate.is_file():
                    found = candidate
                    break
            if found is not None:
                break
        if found is None and card.photo_path and slug not in self.foreign_photos:
            mapped = Path(card.photo_path)
            if FOREIGN_PHOTO_DIR not in mapped.parts and mapped.is_file():
                found = mapped
        if not extra_dirs:
            self._photos[slug] = found
        return found

    def photo_url(self, slug: str) -> str | None:
        return f"/v1/wines/{slug}/photo" if self.photo_file(slug) is not None else None

    # -------------------------------------------------------------- карточка
    def _alcohol(self, slug: str, wine: RecoWine | None) -> Alcohol:
        if wine is not None:
            return self.catalog.alcohol(slug)
        attrs = self.attrs.get(slug)
        return alcohol_of(attrs.abv) if attrs else Alcohol()

    def card(self, slug: str | None) -> dict[str, Any] | None:
        """Карточка по договору (§3): только выгрузка организатора; неизвестный slug — `None`.

        Сахар и игристость — по правилам выгрузки (`organizer_sugar`, `organizer_sparkling`),
        крепость — из slug, описание — «Описание» выгрузки как есть. Блюд, подачи, градиента,
        «опубликовано» и категории живого портала нет: блюда и подача — `GET …/sommelier`.
        """
        base = self.cards.get(slug) if slug else None
        if base is None:
            return None
        wine = self.catalog.get(base.slug)
        sugar = wine.sugar if wine is not None else organizer_sugar(base.name, base.slug)
        if wine is not None:
            sparkling = wine.sparkling
        else:
            sparkling = organizer_sparkling(sugar, base.name, base.slug)
        color = base.category if base.category in COLORS else None
        sugar_word = SUGAR_WORDS.get(sugar or "", "")
        alcohol = self._alcohol(base.slug, wine)
        description = base.description.strip()
        return {
            "slug": base.slug,
            "name": base.name,
            "winery": base.winery,
            "region": base.region,
            "grapes": list(base.grapes),
            "category": base.category,
            "color": base.color,
            "sugar": sugar_word,
            "sugar_class": sugar,
            "photo_name": base.photo_name,
            "color_label": base.category,
            "sugar_label": sugar_word,
            "sparkling": sparkling,
            "style_label": style_label(color, sugar, sparkling),
            "description": description,
            "description_src": "catalog" if description else None,
            "photo_url": self.photo_url(base.slug),
            "portal_url": portal_url_of(base.slug),
            "alcohol": alcohol.value,
            "alcohol_max": alcohol.max,
            "alcohol_src": alcohol.src,
        }

    # -------------------------------------------------------------- скан
    def names(self, slug: str) -> tuple[str, str]:
        wine = self.catalog.get(slug)
        if wine is not None:
            return wine.title, wine.winery
        card = self.cards.get(slug)
        return (card.name, card.winery) if card else ("", "")

    def candidates(self, result: ScanResult) -> list[dict[str, Any]]:
        """top-5 ответа для «Не то вино?»: имя, винодельня, фото. Счетов и вероятностей нет."""
        out = []
        for item in result.top5:
            name, winery = self.names(item.slug)
            out.append(
                {
                    "slug": item.slug,
                    "name": name,
                    "winery": winery,
                    "photo_url": self.photo_url(item.slug),
                }
            )
        return out

    def _winery_keys(self, value: str) -> dict[str, str]:
        """Винодельни каталога, которым подходит написание: ключ → написание каталога."""
        keys: dict[str, str] = {}
        for slug in self.attrs.slugs_of_winery(value):
            wine = self.catalog.get(slug)
            if wine is not None:
                keys.setdefault(wine.winery_norm, wine.winery)
            else:
                attrs = self.attrs.get(slug)
                if attrs is not None and attrs.winery:
                    keys.setdefault(f"~{norm_key(attrs.winery)}", attrs.winery)
        norm = self._winery_by_key.get(norm_key(value))
        if norm is not None:
            keys.setdefault(norm, self.catalog.winery_names.get(norm, value))
        return keys

    def winery_key(self, slug: str) -> str | None:
        """Ключ винодельни позиции — тот же, что у `_winery_keys`."""
        wine = self.catalog.get(slug)
        if wine is not None:
            return wine.winery_norm
        attrs = self.attrs.get(slug)
        return f"~{norm_key(attrs.winery)}" if attrs is not None and attrs.winery else None

    def specific_key(self, text: str, keys: Mapping[str, str] | None = None) -> str | None:
        """Винодельня, на которую написание указывает однозначно; нет такой — `None`.

        Однозначно — имя винодельни каталога целиком («Château Le Grand Vostock» есть в
        написаниях двух виноделен, но именем — у одной; «Кубань-Вино» — имя, хотя оба слова
        родовые) или написание ровно одной винодельни. Родовое слово не указывает ни на
        какую, даже если в каталоге оно есть у одной: «villa» на этикетке — не «Villa di
        Alma», «chateau» — не «Chateau de Talu», «Кубань» — не «Кубань-Вино». Слово сорта или
        сахара тоже: «блан» в написаниях Усадьбы Маркотх — это и «Совиньон блан».
        """
        keys = self._winery_keys(text) if keys is None else keys
        exact = [key for key, name in keys.items() if norm_key(name) == norm_key(text)]
        if len(exact) == 1:
            return exact[0]
        words = norm_key(text).split()
        if len(keys) != 1 or all(_generic_word(w) or _style_word(w) for w in words):
            return None
        return next(iter(keys))

    def near_catalog_winery(self, text: str) -> bool:
        """Похоже ли прочитанное имя на винодельню каталога с ошибкой чтения.

        «ВИНОДЕЛЬНЯ МЕЙСХАКО» — это Мысхако, «ШАТО ДЕ ТАЛИО» — Chateau de Talu, «BRAU
        ESTATES» — обрезанное Абрау: сказать про них «этой винодельни в каталоге нет» было бы
        неправдой. Свои слова имени сверяются скелетом (одна правка до шести знаков, две —
        дальше) и латиницей как часть слова каталога («brau» в «abrau»).
        """
        for word in own_words(text):
            sk = skeleton(word)
            if len(sk) >= 4:
                budget = 1 if len(sk) <= 6 else 2
                for known in self._winery_skeletons:
                    if abs(len(known) - len(sk)) <= budget and (
                        bounded_levenshtein(sk, known, budget) <= budget
                    ):
                        return True
            elif sk in self._winery_skeletons:
                return True
            latin = romanize(word)
            if len(latin) >= 4 and any(
                latin in known or (len(known) >= 5 and known in latin)
                for known in self._winery_latin
            ):
                return True
        return False

    def resolve_winery(
        self, values: Iterable[str], *, prefer: Iterable[str] = (), house: str | None = None
    ) -> WineryMatch:
        """Прочитанные написания винодельни → одна винодельня каталога или ничего.

        Чтение отдаёт несколько написаний: полную форму словаря и отдельные токены («Château Le
        Grand Vostock», «chateau», «Усадьба Маркотх» — Маркотх ещё и хребет над Новороссийском).
        В счёт идут только однозначные написания (`specific_key`), по порядку:

        1. винодельня кандидата сканера (`prefer` — ключи виноделен top-5 по рангу), если её
           однозначное написание есть среди прочитанного: так «Château Le Grand Vostock»
           побеждает случайную «Усадьбу Маркотх» на той же этикетке. Родовой токен («château»)
           винодельню кандидата не подтверждает: на кадрах вне каталога он «находил» Le Grand
           Vostock у Château de Châtaignier;
        2. `house` — винодельня, названная в строках словом хозяйства (`winery_of_lines`), если
           она не похожа ни на одну винодельню каталога (`near_catalog_winery`): она прочитана,
           и её в каталоге нет. «CHÂTEAU / de CHÂTAIGNIER» на этикетке весомее «Усадьбы
           Маркотх», которую словарь нашёл нечётко и которую сканер не предлагал;
        3. написание, которое совпадает с именем винодельни каталога целиком;
        4. написание, которое указывает ровно на одну винодельню.

        Спорное или родовое написание («долина», «шато») винодельней не считается: «прочитали:
        долина» ничего не говорит человеку. Написание вне словаря (так приходит тело «Не
        тупика») — винодельня не из каталога, если оно не похоже на винодельню каталога.
        """
        options: list[tuple[str, dict[str, str], str | None]] = []
        for text in (clean_text(value) for value in values):
            if text:
                keys = self._winery_keys(text)
                options.append((text, keys, self.specific_key(text, keys)))
        for key in prefer:
            for _, keys, specific in options:
                if specific == key:
                    return WineryMatch(keys[key], True, (key,))
        house = clean_text(house) if house else ""
        if house and not self.near_catalog_winery(house):
            return WineryMatch(house, False)
        for text, keys, specific in options:
            if specific is not None and norm_key(keys[specific]) == norm_key(text):
                return WineryMatch(keys[specific], True, (specific,))
        for _, keys, specific in options:
            if specific is not None:
                return WineryMatch(keys[specific], True, (specific,))
        unknown = [text for text, keys, _ in options if not keys and not generic_winery(text)]
        if (
            unknown
            and not any(keys for _, keys, _ in options)
            and not self.near_catalog_winery(unknown[0])
        ):
            return WineryMatch(unknown[0], False)
        return WineryMatch(None, None)

    def positions(self, keys: Iterable[str]) -> tuple[str, ...]:
        """Все позиции виноделен каталога — с неканоническими и без фото."""
        return tuple(slug for key in keys for slug in self._positions.get(key, ()))

    def contradicts(self, slug: str, label: LabelRead) -> bool:
        """Спорит ли позиция с прочитанным: сахар, цвет или сорта, когда обе стороны известны.

        Сахар и цвет — как `app.resolve.ambiguous.contradicts`; сорта спорят, когда у позиции
        они есть, на этикетке прочитаны, и общих нет. Факты — выгрузки организатора: из
        справочника рекомендаций, а без него — из признаков каталога, но сахар и тогда по
        правилу выгрузки (класс сахара разметки взят из категории живого портала).
        """
        wine = self.catalog.get(slug)
        if wine is not None:
            color, sugar, grapes = wine.color, wine.sugar, frozenset(wine.grapes)
        else:
            attrs = self.attrs.get(slug)
            if attrs is None:
                return False
            color = attrs.color.value if attrs.color else None
            sugar = organizer_sugar(attrs.name, slug)
            grapes = attrs.grapes
        if label.sugar and sugar and label.sugar != sugar:
            return True
        if label.color and color and label.color != color:
            return True
        return bool(label.grapes and grapes and not set(label.grapes) & grapes)

    def conditional_names(self, slugs: Iterable[str]) -> bool:
        """Все названия позиций условные («Пино Нуар», «Белое сухое») — как у правила S10."""
        wines = [wine for slug in slugs if (wine := self.attrs.get(slug)) is not None]
        return bool(wines) and all(wine.conditional_name for wine in wines)

    def not_found_conditions(
        self, result: ScanResult, label: LabelRead, match: WineryMatch
    ) -> dict[str, Any]:
        """Условия подсказки «похоже, этой позиции нет в каталоге» по отдельности (план, Д2).

        Как `evidence.abstain`: каждое условие видно само, решает `not_found_reasons`.

        - `winery`: `catalog` — прочитана однозначно и есть в каталоге; `unknown` — названа
          словом хозяйства, и в каталоге её нет; `None` — не прочитана или спорная;
        - `positions`, `no_position_agrees`: сколько позиций у винодельни в каталоге и спорит
          ли с прочитанным каждая из них (`contradicts`);
        - `conditional_names`: все названия винодельни условные — по ним судить нельзя;
        - `visual`: счёт CV лучшей серии (zmax, `best_visual`).
        """
        slugs = self.positions(match.keys) if match.in_catalog else ()
        winery = None
        if match.in_catalog is not None:
            winery = "catalog" if match.in_catalog else "unknown"
        return {
            "winery": winery,
            "positions": len(slugs),
            "no_position_agrees": bool(slugs) and all(self.contradicts(s, label) for s in slugs),
            "conditional_names": self.conditional_names(slugs),
            "visual": best_visual(result.evidence),
        }

    def winery_slugs(self, match: WineryMatch) -> list[str]:
        """Канонические вина винодельни с фото выгрузки, по названию."""
        wines = [wine for key in match.keys for wine in self.catalog.winery_pool(key)]
        return [wine.slug for wine in sorted(wines, key=name_key)]

    def after(self, result: ScanResult) -> dict[str, Any]:
        return after_block(result, self)

    # -------------------------------------------------------------- рекомендации
    def tile(self, wine: RecoWine, reasons: Sequence[str] = ()) -> dict[str, Any]:
        return reco.tile(wine, safe_texts(reasons), photo_url=self.photo_url(wine.slug))

    def similar(self, slug: str, *, limit: int, order: Order) -> dict[str, Any] | None:
        """`GET /v1/wines/{slug}/similar`; неизвестный slug — `None` (404)."""
        wine = self.catalog.get(slug)
        if wine is None and self.cards.get(slug) is None:
            return None
        if wine is None:
            # Карточка есть, а в справочнике рекомендаций позиции нет: подбирать не по чему.
            selection = reco.Selection((), (reco.fewer_note(0, limit),))
            category = ""
        else:
            facts = reco.Facts.of_wine(wine, self.catalog.alcohol(slug))
            if order == "plain":
                selection = reco.plain(self.catalog, facts, limit)
            else:
                selection = reco.similar(self.catalog, facts, limit)
            category = wine.style_label
        return {
            "slug": slug,
            "order": order,
            "limit": limit,
            "notice": reco.NOTICE if order == "reco" else None,
            "category_label": category,
            "items": [
                self.tile(p.wine, p.reasons if order == "reco" else ()) for p in selection.picks
            ],
            "notes": notes_of(selection.notes),
        }

    def shelf(self, slug: str, *, food: Food, want: Want, order: Order) -> dict[str, Any] | None:
        """`GET /v1/wines/{slug}/shelf` — «Сомелье у полки»; неизвестный slug — `None` (404).

        Три вина или честная фраза (`app.recommend.shelf`). При `plain` направление не
        применяется: та же категория с блюдами чипа по названию, без объяснений.
        """
        wine = self.catalog.get(slug)
        if wine is None and self.cards.get(slug) is None:
            return None
        if wine is None:
            # Карточка есть, а в справочнике рекомендаций позиции нет: подбирать не по чему.
            answer = ShelfAnswer((), "near", (reco.fewer_note(0, reco.DEFAULT_LIMIT),))
        elif order == "plain":
            answer = self.sommelier.plain(wine, food)
        else:
            answer = self.sommelier.answer(wine, food, want)
        plain = order == "plain"
        return {
            "slug": slug,
            "food": food,
            "want": want,
            "order": order,
            "notice": None if plain else reco.NOTICE,
            "pool": answer.pool,
            "items": [
                shelf_tile(self.tile(pick.wine, () if plain else pick.reasons), pick)
                for pick in answer.picks
            ],
            "notes": notes_of(answer.notes),
        }

    def excluded_slugs(self, slugs: Iterable[str]) -> frozenset[str]:
        """Присланные slug справочника вместе со всей их группой `wine_id`; чужие — мимо.

        Отвергнутое вино не должно вернуться в выдачу тем же вином в другом объёме или году.
        """
        out: set[str] = set()
        for slug in slugs:
            out |= self._group_of.get(str(slug).strip(), frozenset())
        return frozenset(out)

    def by_label(self, body: ByLabelBody, *, limit: int, order: Order) -> dict[str, Any]:
        """`POST /v1/similar/by-label`: «этой позиции нет, но винодельня в каталоге есть».

        `exclude` снимает присланные вина и их группы и из `same_winery`, и из `similar`
        (`CatalogWithout`). Число вин винодельни и её регион считаются по всему справочнику.
        """
        exclude = self.excluded_slugs(body.exclude)
        pool = CatalogWithout(self.catalog, exclude) if exclude else self.catalog
        read = normalize_read(body)
        given = read["winery"]
        match = self.resolve_winery([given] if given else [])
        # Эхо — написание каталога, если винодельня нашлась; спорное слово остаётся как прислано.
        read["winery"] = match.name or given
        sparkling = read["sparkling"]
        if sparkling is None:
            sparkling = reco.sparkling_by_sugar(read["sugar"])
        facts = reco.Facts(
            color=read["color"],
            sparkling=sparkling,
            sugar=read["sugar"],
            grapes=tuple(read["grapes"]),
            alcohol=Alcohol(read["abv"], None, "label") if read["abv"] is not None else Alcohol(),
            region=reco.winery_region(self.catalog, match.keys),
            wineries=frozenset(match.keys),
        )
        notes: list[reco.Note] = []
        if not match.in_catalog:
            notes.append(reco.Note("winery_unknown", winery_note(read, match)))
        same = (
            reco.same_winery(pool, match.keys, facts, order=order)  # type: ignore[arg-type]
            if match.in_catalog
            else reco.Selection(())
        )
        if not facts.has_style:
            notes.append(
                reco.Note(
                    "no_facts",
                    "Цвет, сахар и сорт на этикетке не прочитаны — похожие подобрать не по чему",
                )
            )
            similar = reco.Selection(())
        elif order == "plain":
            similar = reco.plain(pool, facts, limit)  # type: ignore[arg-type]
        else:
            similar = reco.similar(pool, facts, limit, relax_color=True)  # type: ignore[arg-type]
        for note in similar.notes:
            if exclude and note.code == "relaxed_color":
                # Фраза про цвет — о каталоге: исключённые вина в нём есть, и «…в каталоге
                # нет» было бы неправдой, когда исключены все вина этого цвета.
                note = reco._relaxed_color_note(self.catalog, facts)
            notes.append(note)
        plain = order == "plain"
        return {
            "read": read,
            "winery": {
                "name": read["winery"],
                "in_catalog": match.in_catalog,
                "count": sum(len(self.catalog.winery_pool(key)) for key in match.keys),
            },
            "order": order,
            "notice": None if plain else reco.NOTICE,
            "same_winery": [self.tile(p.wine, () if plain else p.reasons) for p in same.picks],
            "similar": [self.tile(p.wine, () if plain else p.reasons) for p in similar.picks],
            "notes": notes_of(notes),
        }


def winery_note(read: Mapping[str, Any], match: WineryMatch) -> str:
    """Честная фраза «Не тупика», когда винодельни каталога нет.

    Прочитанного написания нет, а остальное прочитано — винодельню не узнали, а не «не
    прочитали»: словарь знает только винодельни каталога, и чужую («MASSIMO VISCONTI»)
    чтение видит строкой, но винодельней не называет. «Не прочитали» — только когда с
    этикетки не пришло ничего.
    """
    if read.get("winery"):
        if match.in_catalog is None:
            return "По прочитанному названию винодельню однозначно не определить"
        return "Этой винодельни в каталоге нет"
    if any(read.get(key) not in (None, "", []) for key in LABEL_KEYS):
        return "Винодельню по этикетке определить не удалось"
    return "Винодельню на этикетке не прочитали"


def safe_texts(texts: Iterable[str]) -> list[str]:
    """Строки, прошедшие `content_filter`. Не прошедшая выбрасывается и пишется в журнал."""
    out = []
    for text in texts:
        verdict = check(text)
        if verdict.clean:
            out.append(text)
        else:
            logger.error("Строка объяснения не прошла фильтр (%s): %r", verdict.violations, text)
    return out


def notes_of(notes: Iterable[reco.Note]) -> list[dict[str, str]]:
    return [note.as_dict() for note in notes if check(note.text).clean]


def normalize_read(body: ByLabelBody) -> dict[str, Any]:
    """Тело «Не тупика» в кодах каталога: цвет «Оранжевое», сахар `brut`, сорта `syrah`."""
    return {
        "winery": clean_text(body.winery) or None,
        "color": _color(body.color),
        "sugar": _sugar(body.sugar),
        "grapes": list(_grapes(body.grapes)),
        "abv": plausible_abv(body.abv) if body.abv is not None else None,
        "sparkling": body.sparkling,
    }


def best_visual(evidence: Mapping[str, Any]) -> float | None:
    """Счёт CV лучшей серии (zmax): `evidence.abstain.best_visual`, иначе счёт CV top-1."""
    abstain = evidence.get("abstain")
    value = abstain.get("best_visual") if isinstance(abstain, Mapping) else None
    if value is None:
        top5 = (evidence.get("cv") or {}).get("top5") or []
        value = top5[0].get("score") if top5 and isinstance(top5[0], Mapping) else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def not_found_reasons(conditions: Mapping[str, Any], visual_max: float | None) -> list[str]:
    """Правило подсказки `not_found` по условиям `not_found_conditions`; `None` — экрана нет.

    Счёт CV ниже `visual_max` и одно из двух: винодельня каталога прочитана, её названия не
    условные, и каждая её позиция спорит с этикеткой (`winery_no_match`), или винодельня
    названа словом хозяйства и её в каталоге нет (`winery_not_in_catalog`).
    """
    visual = conditions.get("visual")
    if visual_max is None or visual is None or visual >= visual_max:
        return []
    if conditions.get("winery") == "catalog":
        if conditions.get("no_position_agrees") and not conditions.get("conditional_names"):
            return ["winery_no_match"]
        return []
    return ["winery_not_in_catalog"] if conditions.get("winery") == "unknown" else []


def visual_low(visual: float | None, visual_max: float | None) -> bool:
    """Счёт CV лучшей серии ниже порога подсказки; нет счёта или порога — подсказки нет."""
    return visual_max is not None and visual is not None and visual < visual_max


def after_block(result: ScanResult, search: AfterSearch | None = None) -> dict[str, Any]:
    """Поле `after` ответа `/v1/scan`. Без слоя (`search=None`) винодельня не сверяется.

    `suggest_not_found` (договор, §2) есть всегда: счёт CV лучшей серии ниже
    `SUGGEST_NOT_FOUND_VISUAL_MAX`. Тогда `found` становится `check`, а к причинам любого экрана,
    кроме `error`, добавляется `visual_low`. Подсказке нужен только ответ сканера, поэтому без
    слоя она считается так же.

    `question` (договор сомелье `docs/api-sommelier.md`, §5) — один вопрос экрана `check` по
    полю, которым различаются варианты (`templates.check_question`); на других экранах, без
    слоя и когда различать нечем — `None`. В predict, как и всё `after`, не попадает.
    """
    state, reasons = scan_state(result)
    label = read_of_evidence(result.evidence)
    if search is not None:
        prefer = [key for item in result.top5 if (key := search.winery_key(item.slug))]
        match = search.resolve_winery(
            label.wineries, prefer=dict.fromkeys(prefer), house=label.house
        )
        slugs = search.winery_slugs(match) if match.in_catalog else []
        if state != "error" and search.not_found_max is not None:
            conditions = search.not_found_conditions(result, label, match)
            hint = not_found_reasons(conditions, search.not_found_max)
            if hint:
                state, reasons = "not_found", hint
    else:
        match = WineryMatch(label.wineries[0] if label.wineries else None, None)
        slugs = []
    suggest_max = search.suggest_max if search is not None else SUGGEST_NOT_FOUND_VISUAL_MAX
    suggest = state != "error" and visual_low(best_visual(result.evidence), suggest_max)
    if suggest:
        state = "check" if state == "found" else state
        reasons = [*reasons, VISUAL_LOW]
    read = {
        "winery": match.name,
        "color": label.color,
        "sugar": label.sugar,
        "grapes": list(label.grapes),
        "abv": label.abv,
        "sparkling": label.sparkling,
    }
    question = None
    if state == "check" and search is not None:
        question = check_question(read, question_candidates(result, search))
    return {
        "state": state,
        "reasons": reasons,
        "read": read,
        "read_label": read_label(read),
        "winery_in_catalog": match.in_catalog,
        "winery_slugs": slugs,
        "suggest_not_found": suggest,
        "question": question if question and check(question["text"]).clean else None,
    }


#: Сорта-заглушки выгрузки: не сорт, а группа — вариантом вопроса не бывают.
_GRAPE_PLACEHOLDERS = frozenset({"белые сорта винограда", "красные сорта винограда"})


def question_candidates(result: ScanResult, search: AfterSearch) -> list[dict[str, Any]]:
    """Кандидаты `top5` с фактами карточки для вопроса экрана `check` (договор сомелье, §5).

    Факты — поля карточки без портала: сахар, цвет, игристость, сорта (без заглушек вроде
    «Белые сорта винограда»), крепость и название. Нет карточки — кандидата нет.
    """
    out = []
    for item in result.top5:
        card = search.card(item.slug)
        if card is None:
            continue
        grapes = [
            str(g).strip()
            for g in card.get("grapes") or []
            if str(g).strip() and str(g).strip().casefold() not in _GRAPE_PLACEHOLDERS
        ]
        color = card.get("color_label")
        out.append(
            {
                "slug": item.slug,
                "name": card.get("name") or "",
                "facts": {
                    "sugar": card.get("sugar_class") or None,
                    "color": color if color in COLORS else None,
                    "sparkling": bool(card.get("sparkling")),
                    "grapes": grapes,
                    "abv": card.get("alcohol"),
                },
            }
        )
    return out
