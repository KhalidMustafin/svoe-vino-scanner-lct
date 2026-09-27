"""Признаки пары «запрос — кандидат» для обучаемого resolve.

Пара — это кадр и одна позиция из top-K кандидатов CV. Признаки считаются ТОЛЬКО из того,
что есть у сервиса в момент ответа: выдача CV, поля этикетки (и сырой текст читателя) и
таблица каталога. Верного slug функции не получают вовсе: даже если на вход пришла запись
прогона с полем `slug`, оно не читается. Подмена эталона в записи не меняет ни одного
числа — это проверяет тест.

Нормализация без утечки: всё, что сравнивает кандидата с другими, сравнивает его с
кандидатами ЭТОГО ЖЕ запроса — отрыв от CV-top1, z-оценка и min-max в выдаче, «слов
названия найдено меньше, чем у лучшего соседа». Глобальные средние и σ для стандартизации
считает `LogisticRanker.fit` по обучающим запросам, а не этот модуль.

Счёт модели линейный, и ответ — argmax внутри запроса. Поэтому признак, одинаковый у всех
кандидатов запроса («текст пустой», «доля прочитанных полей», отрыв top-1 сам по себе), на
ответ не влияет вовсе: такие признаки не заводятся, отрыв top-1 входит только
взаимодействием «отрыв × это top-1». По той же причине у согласия нет столбца «не
прочитано»: он был бы суммой двух других и константы, и L2 делил бы вес между ними. Базовый
уровень — «не прочитано», вес `*_match` и `*_conflict` — сразу сдвиг счёта против него.

Группы признаков (`FEATURE_GROUPS`) — единица выбора при обучении:

    cv           счёт, ранг, 1/ранг, отрыв от top-1, z и min-max в выдаче, «это CV-top1»,
                 отрыв top-1 запроса × «это CV-top1»
    cv_group     размер визуальной группы и cluster_B кандидата, «двойник CV-top1»
    winery       согласие / противоречие винодельни и они же × уверенность чтения
    winery_text  доля слов названия винодельни кандидата в прочитанном тексте и самое
                 редкое (по каталогу) её слово, которое прочитано
    grape sugar year serial color abv cuvee
                 согласие / противоречие; год — только у карточек с годом
    name         сколько значимых слов названия кандидата (без сорта, сахара, цвета, серии и
                 винодельни — их сравнивают свои группы) найдено в прочитанном (потолок 3)
                 и то же минус лучший кандидат запроса. Только положительная сторона: за
                 слово, которого читатель не прочёл, кандидат не штрафуется — читатель
                 недочитывает этикетку чаще, чем на ней нет слова
    published    карточка не опубликована на портале — слабый признак, по умолчанию выключен

Знак веса (`feature_sign`): согласие не может снижать счёт, противоречие — повышать,
найденное слово — снижать. Обучение держит эти знаки (`LogisticRanker(signs=...)`).

Текстовые группы считаются для каждого читателя отдельно и получают префикс `читатель.`:
модель, обученная на `vlm35.winery_match`, ждёт на входе чтение под тем же ключом.
"""

from __future__ import annotations

import math
import weakref
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.features.contracts import Candidate, VisualResult
from app.reading.contracts import LabelFields
from app.reading.lexicon.build import GENERIC_WORDS
from app.reading.taxonomy import (
    COLOR_ADJECTIVES,
    COLOR_TERMS,
    GRAPE_COLOR_ADJECTIVES,
    LABEL_GENERIC,
    LABEL_TERMS,
    STYLE_WORDS,
    SUGAR_TERMS,
    PhraseMatch,
    Term,
    adjective_spellings,
    phrase_words,
    words_of,
)
from app.reading.text.normalize import roman_value
from app.reading.text.translit import skeleton
from app.resolve.attrs import CatalogAttrs, WineAttrs, colors_compatible
from app.resolve.rerank import Agreement, ReadKeys, agreement, read_fields

#: Версия набора признаков: меняется при любой правке смысла или имени признака, в том числе
#: при правке разбора полей этикетки (`app/reading/fields.py`, `taxonomy.py`), из которых
#: признаки считаются. Модель хранит её у себя и не загружается с другой версией.
#: 3 — цвет и сахар в любом роде в полях этикетки, точное кюве рядом с ними, цвет в
#: `name_found` — только не средний род и только у сортов, различающихся цветом в каталоге.
#: Исключение (Э4, 25.09): пара «Белое» / «Оранжевое» в `color_*` — «не известно» вместо
#: «спор». Версия не менялась намеренно: у замороженной модели те же веса, с новой версией она
#: бы не загрузилась, а смысл сменился только у этой пары цветов.
FEATURE_VERSION = "resolve-features/3"
DEFAULT_TOP_K = 20

#: Слова одного корня короче этого сравниваются только точно или по скелету.
FUZZY_MIN_LEN = 5
#: Потолок счётчиков слов: «три найденных слова» и «семь» для решения одно и то же.
WORDS_CAP = 3
#: Опечатка в слове этикетки («полсусладкое») узнаётся с одной правкой от этой длины скелета.
LABEL_TYPO_MIN_LEN = 7
_EPS = 1e-9


# ------------------------------------------------------------------ имена и группы
def _pair(name: str) -> tuple[str, str]:
    return (f"{name}_match", f"{name}_conflict")


CV_FEATURES: tuple[str, ...] = (
    "cv_score",
    "cv_rank",
    "cv_rank_inv",
    "cv_gap_top1",
    "cv_z",
    "cv_minmax",
    "cv_is_top1",
    "cv_top1_x_margin",
)
CV_GROUP_FEATURES: tuple[str, ...] = (
    "vgroup_size_log",
    "vgroup_mate_of_top1",
    "cluster_size_log",
    "cluster_mate_of_top1",
)
AGREEMENT_FEATURES: tuple[str, ...] = (
    "winery",
    "grape",
    "sugar",
    "year",
    "serial",
    "color",
    "abv",
    "cuvee",
)
TEXT_GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "winery": (*_pair("winery"), "winery_match_conf", "winery_conflict_conf"),
    "winery_text": ("winery_text_share", "winery_text_idf"),
    "grape": _pair("grape"),
    "sugar": _pair("sugar"),
    "year": _pair("year"),
    "serial": _pair("serial"),
    "color": _pair("color"),
    "abv": _pair("abv"),
    "cuvee": _pair("cuvee"),
    "name": ("name_found", "name_found_rel"),
}
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "cv": CV_FEATURES,
    "cv_group": CV_GROUP_FEATURES,
    **TEXT_GROUP_FEATURES,
    "published": ("unpublished",),
}
TEXT_GROUPS: frozenset[str] = frozenset(TEXT_GROUP_FEATURES)
#: Без `cv` модель не видит картинку: эту группу выбор признаков не выключает.
REQUIRED_GROUPS: frozenset[str] = frozenset({"cv"})
#: `published` выключен по умолчанию: это не свойство кадра, а свойство портала.
DEFAULT_GROUPS: tuple[str, ...] = tuple(group for group in FEATURE_GROUPS if group != "published")

_GROUP_OF_BASE: dict[str, str] = {
    base: group for group, names in FEATURE_GROUPS.items() for base in names
}

#: Знак веса признака с очевидным направлением: +1 — вес ≥ 0, −1 — вес ≤ 0. Остальные
#: (CV, группы, `unpublished`) свободны: у признаков CV направление задаёт их сочетание.
_SIGN_OF_BASE: dict[str, int] = {
    **{f"{name}_match": 1 for name in AGREEMENT_FEATURES},
    **{f"{name}_conflict": -1 for name in AGREEMENT_FEATURES},
    "winery_match_conf": 1,
    "winery_conflict_conf": -1,
    "winery_text_share": 1,
    "winery_text_idf": 1,
    "name_found": 1,
    "name_found_rel": 1,  # значение ≤ 0: «найдено меньше, чем у лучшего» не может помогать
}


# ------------------------------------------------------------------ слова этикетки
@lru_cache(maxsize=65536)
def _skeleton(word: str) -> str:
    return skeleton(word)


_ADJECTIVE_ENDINGS = ("ой", "ый", "ий", "ая", "яя", "ое", "ее", "ые", "ие")
#: Тип продукта в названиях позиций: «Мускат игристый», «Каберне столовое».
_TYPE_WORDS = ("игристое", "тихое", "столовое", "выдержанное", "ординарное", "марочное")


def _adjective_forms(word: str) -> set[str]:
    """«игристое» → игристый, игристая, игристые…: с запасом, для слов-исключений названия."""
    if len(word) > 4 and word.endswith(("ое", "ее")):
        stem = word[:-2]
        return {stem + ending for ending in _ADJECTIVE_ENDINGS}
    return {word}


def _single_words(table: Mapping[Any, Sequence[str]]) -> set[str]:
    words: set[str] = set()
    for variants in table.values():
        for text in variants:
            parts = phrase_words(text)
            if len(parts) == 1:
                words |= _adjective_forms(parts[0])
    return words


def _color_families() -> dict[str, frozenset[str]]:
    """Форма прилагательного цвета → все его формы: «белая» → белый, белая, белое, белые,
    belyy… Цвета вина и цвета ягоды («черный», «зеленый», «серый»)."""
    adjectives = (
        *(adjective for group in COLOR_ADJECTIVES.values() for adjective in group),
        *GRAPE_COLOR_ADJECTIVES,
    )
    families: dict[str, frozenset[str]] = {}
    for adjective in adjectives:
        forms = frozenset(" ".join(phrase_words(text)) for text in adjective_spellings(adjective))
        for form in forms:
            families[form] = forms
    return families


_COLOR_FAMILIES = _color_families()
#: Скелет формы → её прилагательное: «chernyj» и «belyj» в названии каталога — тоже цвет.
_COLOR_FAMILY_SKELETONS: dict[str, frozenset[str]] = {
    sk: forms for form, forms in _COLOR_FAMILIES.items() if len(sk := _skeleton(form)) >= 4
}


@dataclass(frozen=True, slots=True)
class _WordClass:
    """Слова одного класса этикетки: нормы и их скелеты (скелет роднит «сухой» и «suhoy»)."""

    norms: frozenset[str]
    skeletons: frozenset[str]

    @classmethod
    def of(cls, words: Iterable[str]) -> _WordClass:
        norms = frozenset(words)
        return cls(norms, frozenset(sk for word in norms if len(sk := _skeleton(word)) >= 4))

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        if word in self.norms:
            return True
        sk = _skeleton(word)
        if len(sk) < 4:
            return False
        if sk in self.skeletons:
            return True
        # «полсусладкое»: опечатка каталога в длинном слове сахара — всё ещё сахар.
        return len(sk) >= LABEL_TYPO_MIN_LEN and any(
            len(other) >= LABEL_TYPO_MIN_LEN
            and abs(len(other) - len(sk)) <= 1
            and Levenshtein.distance(sk, other, score_cutoff=1) <= 1
            for other in self.skeletons
        )


#: Сахар и тип продукта во всех родах: в названии позиции они ничего не различают.
SUGAR_TYPE_WORDS = _WordClass.of(
    _single_words(SUGAR_TERMS) | {form for word in _TYPE_WORDS for form in _adjective_forms(word)}
)
#: Цвет во всех родах. Словом названия он остаётся, только когда отличает позицию от
#: соседей по линейке (`CatalogStats.name_words`): сразу после сорта или стиля («Мускат
#: Чёрный», «Мускатель белый», «Портвейн красный») или последним словом названия сорта
#: («Мускат белый» — «Мускат розовый»). В остальных местах это цвет вина («Купаж
#: красный»), его сравнивает группа `color`.
COLOR_WORDS = _WordClass.of(
    _single_words(COLOR_TERMS) | _single_words({"adjectives": tuple(_COLOR_FAMILIES)})
)


def color_forms(word: str) -> frozenset[str]:
    """Формы того же цвета в любом роде и числе: «белый» → белый, белая, белое, белые, belyy…

    Слово не из прилагательных цвета («blanc», «rosso») — только оно само.
    """
    forms = _COLOR_FAMILIES.get(word)
    if forms is None:
        forms = _COLOR_FAMILY_SKELETONS.get(_skeleton(word))
    return forms if forms is not None else frozenset({word})


#: Слова, которые не отличают позицию: тип продукта, предлоги, регионы, общие слова этикетки.
#: Цвет решается отдельно, по месту в названии, в том числе латиницей («chernyy»).
NAME_STOP_WORDS = frozenset(
    word for word in GENERIC_WORDS | LABEL_GENERIC if word not in COLOR_WORDS
)
#: Слова названия винодельни, по которым её не узнать: «винодельня», «усадьба», «estate».
WINERY_STOP_WORDS = GENERIC_WORDS | LABEL_GENERIC


def reader_prefix(reader: str) -> str:
    """Префикс текстовых признаков читателя; пустой читатель — без префикса."""
    return f"{reader}." if reader else ""


def group_of(name: str) -> str:
    """Группа признака по имени: префикс читателя отбрасывается."""
    base = name.rsplit(".", 1)[-1]
    try:
        return _GROUP_OF_BASE[base]
    except KeyError:
        raise KeyError(f"неизвестный признак {name!r}") from None


def feature_sign(name: str) -> int:
    """Требуемый знак веса признака: +1 (≥ 0), −1 (≤ 0) или 0 (свободен)."""
    group_of(name)  # неизвестное имя — KeyError, а не свободный знак
    return _SIGN_OF_BASE.get(name.rsplit(".", 1)[-1], 0)


def feature_signs(names: Sequence[str]) -> list[int]:
    return [feature_sign(name) for name in names]


def readers_of(names: Iterable[str]) -> tuple[str, ...]:
    """Читатели, чьи текстовые признаки есть среди имён (`""` — признаки без префикса)."""
    out: dict[str, None] = {}
    for name in names:
        if group_of(name) in TEXT_GROUPS:
            out[name.rsplit(".", 1)[0] if "." in name else ""] = None
    return tuple(out)


def check_groups(groups: Iterable[str]) -> tuple[str, ...]:
    """Группы в каноническом порядке; неизвестное имя — ошибка, а не тихий пропуск."""
    chosen = set(groups)
    unknown = chosen - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"неизвестные группы признаков: {', '.join(sorted(unknown))}")
    return tuple(group for group in FEATURE_GROUPS if group in chosen)


def feature_names(readers: Sequence[str], groups: Iterable[str] = DEFAULT_GROUPS) -> list[str]:
    """Имена признаков в каноническом порядке: группа за группой, читатель за читателем."""
    names: list[str] = []
    for group in check_groups(groups):
        base = FEATURE_GROUPS[group]
        if group in TEXT_GROUPS:
            for reader in readers:
                names.extend(reader_prefix(reader) + name for name in base)
        else:
            names.extend(base)
    return names


# ------------------------------------------------------------------ вход: CV
@dataclass(frozen=True, slots=True)
class CvCandidate:
    slug: str
    score: float
    rank: int  # место в выдаче после снятия повторов, с единицы


@dataclass(frozen=True, slots=True)
class CvQuery:
    """Выдача CV одного запроса: top-K без повторов slug и отрыв top-1 по всему каталогу."""

    candidates: tuple[CvCandidate, ...]
    margin: float

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(candidate.slug for candidate in self.candidates)

    @property
    def top1_tie(self) -> int:
        """Сколько кандидатов делят счёт CV-top1 до бита (1 — ничьей нет, 0 — выдачи нет).

        При ничьей CV-top1 — первый по порядку файла, то есть жребий, а не решение канала.
        """
        if not self.candidates:
            return 0
        top = self.candidates[0].score
        return sum(candidate.score == top for candidate in self.candidates)


QueryCv = VisualResult | Mapping[str, Any] | Sequence[Candidate | Mapping[str, Any]] | CvQuery


def cv_query(query_cv: QueryCv, *, top_k: int = DEFAULT_TOP_K) -> CvQuery:
    """`VisualResult`, запись `predictions.jsonl` (`top`, `margin`) или список → `CvQuery`.

    Порядок кандидатов берётся как есть (выдача CV уже отсортирована). Повтор slug
    отбрасывается — остаётся лучший ранг, как в `rerank`. Отрыв без явного `margin` — разница
    первых двух счётов. Поле `slug` записи (эталон) не читается.
    """
    if top_k <= 0:
        raise ValueError("top_k должен быть больше нуля")
    if isinstance(query_cv, CvQuery):
        return CvQuery(query_cv.candidates[:top_k], query_cv.margin)
    margin: float | None
    if isinstance(query_cv, VisualResult):
        raw: Sequence[Any] = query_cv.candidates
        margin = query_cv.margin
    elif isinstance(query_cv, Mapping):
        top = query_cv.get("top") or []
        if not isinstance(top, list):
            raise TypeError("запись CV: поле top должно быть списком кандидатов")
        raw = top
        value = query_cv.get("margin")
        margin = None if value is None else float(value)
    else:
        raw = list(query_cv)
        margin = None
    items: list[tuple[str, float]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, Candidate):
            slug, score = item.slug, float(item.score)
        else:
            slug, score = str(item["slug"]), float(item["score"])
        if slug in seen:
            continue
        seen.add(slug)
        items.append((slug, score))
        if len(items) >= top_k:
            break
    candidates = tuple(
        CvCandidate(slug, score, rank) for rank, (slug, score) in enumerate(items, start=1)
    )
    if margin is None:
        margin = max(0.0, items[0][1] - items[1][1]) if len(items) > 1 else 0.0
    return CvQuery(candidates, float(margin))


# ------------------------------------------------------------------ вход: текст
@dataclass(frozen=True, slots=True)
class TextRead:
    """Одно чтение этикетки: поля и, если есть, сырой текст читателя (строки через \\n)."""

    fields: LabelFields | None = None
    raw_text: str | None = None


@dataclass(frozen=True, slots=True)
class WordBag:
    """Прочитанные слова: нормы и скелеты. Слово найдено точно, по скелету или с одной правкой."""

    norms: frozenset[str]
    skeletons: frozenset[str]

    @classmethod
    def of(cls, texts: Iterable[str]) -> WordBag:
        norms: set[str] = set()
        skeletons: set[str] = set()
        for text in texts:
            for word in phrase_words(text):
                norms.add(word)
                if sk := _skeleton(word):
                    skeletons.add(sk)
        return cls(frozenset(norms), frozenset(skeletons))

    def __bool__(self) -> bool:
        return bool(self.norms)

    def has(self, word: str, *, fuzzy: bool = True) -> bool:
        """`fuzzy=False` — только норма или скелет, без правки: для слов, у которых в
        одну правку лежит слово этикетки («белый» — «белое», «красные» — «красное»)."""
        if word in self.norms:
            return True
        sk = _skeleton(word)
        if not sk:
            return False
        if sk in self.skeletons:
            return True
        if not fuzzy or len(sk) < FUZZY_MIN_LEN:
            return False
        return any(
            abs(len(other) - len(sk)) <= 1 and Levenshtein.distance(sk, other, score_cutoff=1) <= 1
            for other in self.skeletons
        )


def read_bag(read: TextRead) -> WordBag:
    """Слова чтения: сырой текст, а без него — значения полей (включая слова вне словаря).

    Сырой текст предпочтительнее: поля — это уже поправленные словарём формы, и «Alma Valley»
    в поле винодельни может стоять на месте прочитанного «River Valley».
    """
    if read.raw_text and read.raw_text.strip():
        return WordBag.of(read.raw_text.splitlines())
    fields = read.fields
    if fields is None:
        return WordBag(frozenset(), frozenset())
    texts = [
        str(evidence.value)
        for group in (fields.producer, fields.cuvee, fields.grapes, fields.serial, fields.unmatched)
        for evidence in group
    ]
    return WordBag.of(texts)


# ------------------------------------------------------------------ каталог
class CatalogStats:
    """Сводки каталога для признаков: размеры групп, слова названий, редкость слов виноделен.

    Считается один раз на таблицу `CatalogAttrs` (`catalog_stats`) и не зависит от кадров.
    """

    def __init__(self, attrs: CatalogAttrs) -> None:
        self.vgroup_sizes: Counter[str] = Counter(w.visual_group for w in attrs if w.visual_group)
        self.cluster_sizes: Counter[str] = Counter(w.cluster for w in attrs if w.cluster)
        by_winery: dict[str, set[str]] = {}
        for wine in attrs:
            by_winery.setdefault(wine.winery, set()).update(self._winery_key_words(wine))
        self._winery_words = {name: frozenset(words) for name, words in by_winery.items()}
        self.winery_df: Counter[str] = Counter(
            word for words in self._winery_words.values() for word in words
        )
        self._name_words: dict[str, tuple[str, ...]] = {}
        self._winery_name_words: dict[str, tuple[str, ...]] = {}
        # Сорта, которые в каталоге различаются цветом в конце названия: «Мускат белый»,
        # «Мускат розовый», «Мускат янтарный». У «Кокур белый» и «Цимлянский чёрный» цвет один
        # на весь каталог, позиции он не отличает, а сорт сравнивает группа `grape`.
        colors: dict[tuple[str, ...], set[frozenset[str]]] = {}
        for wine in attrs:
            norms, skeletons = words_of(wine.name)
            for match in LABEL_TERMS.find(norms, skeletons):
                last = _grape_color_ending(match, norms)
                if last is not None:
                    colors.setdefault(tuple(norms[match.start : last]), set()).add(
                        color_forms(norms[last])
                    )
        self._grapes_split_by_color = frozenset(
            base for base, found in colors.items() if len(found) > 1
        )

    @staticmethod
    def _significant(words: Iterable[str], stop: frozenset[str]) -> list[str]:
        out = []
        for word in words:
            if word in stop or any(ch.isdigit() for ch in word):
                continue
            if len(word) < 3 and not (len(word) == 2 and roman_value(word)):
                continue
            out.append(word)
        return list(dict.fromkeys(out))

    def _winery_key_words(self, wine: WineAttrs) -> list[str]:
        texts = [wine.winery, *sorted(wine.winery_keys)]
        return self._significant(
            (word for text in texts for word in phrase_words(text)), WINERY_STOP_WORDS
        )

    def winery_words(self, wine: WineAttrs) -> frozenset[str]:
        """Все значимые слова винодельни: название, варианты, бренды — по всем её позициям."""
        return self._winery_words.get(wine.winery, frozenset())

    def winery_name_words(self, wine: WineAttrs) -> tuple[str, ...]:
        """Значимые слова канонического названия винодельни («Alma Valley» → alma, valley)."""
        cached = self._winery_name_words.get(wine.winery)
        if cached is None:
            cached = tuple(self._significant(phrase_words(wine.winery), WINERY_STOP_WORDS))
            self._winery_name_words[wine.winery] = cached
        return cached

    def name_words(self, wine: WineAttrs) -> tuple[str, ...]:
        """Значимые слова названия позиции: то, чем она отличается от соседей по каталогу.

        Снимаются фразы сорта, сахара и цвета из таксономии («Каберне Фран»), слова сахара и
        типа продукта в любом роде («Мускат Сухой», «игристый», опечатка «полсусладкое»),
        цвет вина («Купаж красный», «Рислинг белое»), общие слова и слова названия винодельни
        в любом написании («Новый Свет», «merkotan»). Бренды линеек винодельни («ДНК.
        Аллели», «Шато Тамань») остаются: ими позиции одной винодельни и различаются.
        Серия словаря («Резерв», «Гран Резерв», «Блан де Блан») снимается: её с
        канонизацией сравнивает группа `serial`, а сырым словом через письменность она не
        находится («reserve» от «резерв» — две правки). «Премиум» — не серия словаря и
        остаётся словом названия.

        Русское прилагательное цвета остаётся, когда отличает позицию от соседей по линейке:
        последнее слово названия сорта, если этот сорт в каталоге встречается с разными
        цветами («Мускат белый» — «Мускат розовый», но не «Кокур белый» и не «Цимлянский
        чёрный»), и цвет в роде сорта или стиля сразу после него («Мускат Чёрный», «Мускатель
        белый», «Портвейн красный»). Средний род там — цвет вина («Портвейн - белое
        креплёное»).
        """
        cached = self._name_words.get(wine.slug)
        if cached is not None:
            return cached
        norms, skeletons = words_of(wine.name)
        kind_at: dict[int, str] = {}
        grape_final: set[int] = set()  # цвет — последнее слово названия сорта
        follows: set[int] = {i for i in range(1, len(norms)) if norms[i - 1] in STYLE_WORDS}
        for match in LABEL_TERMS.find(norms, skeletons):
            kind_at.update(dict.fromkeys(range(match.start, match.end), match.value.kind))
            if match.value.kind == "grape":
                follows.add(match.end)
                last = _grape_color_ending(match, norms)
                if last is not None and tuple(norms[match.start : last]) in (
                    self._grapes_split_by_color
                ):
                    grape_final.add(last)
        winery = self.winery_name_words(wine)
        winery_skeletons = {sk for word in winery if (sk := _skeleton(word))}
        kept: list[str] = []
        for i, word in enumerate(norms):
            if word in winery or _skeleton(word) in winery_skeletons:
                continue
            if word in COLOR_WORDS:
                agrees = (
                    i in follows
                    and kind_at.get(i, "color") == "color"
                    and _color_adjective(word)
                    and not word.endswith(_NEUTER_ENDINGS)
                )
                if i in grape_final or agrees:
                    kept.append(word)
                continue
            if i in kind_at or word in SUGAR_TYPE_WORDS:
                continue
            kept.append(word)
        cached = tuple(self._significant(kept, NAME_STOP_WORDS))
        self._name_words[wine.slug] = cached
        return cached


#: Средний род прилагательного: «вино белое» — цвет вина, а не слово названия.
_NEUTER_ENDINGS = ("ое", "ее", "oe", "ee")


def _color_adjective(word: str) -> bool:
    """Русское прилагательное цвета в любой форме или его латинское написание («belyj»)."""
    return word in _COLOR_FAMILIES or _skeleton(word) in _COLOR_FAMILY_SKELETONS


def _grape_color_ending(match: PhraseMatch[Term], norms: Sequence[str]) -> int | None:
    """Место цвета, которым кончается название сорта («Мускат белый» → 1), или None."""
    last = match.end - 1
    if match.value.kind == "grape" and last > match.start and _color_adjective(norms[last]):
        return last
    return None


def name_word_found(bag: WordBag, word: str) -> bool:
    """Прочитано ли слово названия кандидата.

    Цвет ищется без правки (одной правкой «белый» становится «белое», а «красные» —
    «красное»), в любом роде и числе, кроме среднего: «белый» в «Мускатель белый» находится
    в «БЕЛЫЙ» и в «белая», но не в «вино белое» — цвет вина сравнивает группа `color`, и
    засчитывать его второй раз словом названия не нужно. Остальные слова — по норме, скелету
    или с одной правкой.
    """
    if word in COLOR_WORDS:
        return any(
            bag.has(form, fuzzy=False)
            for form in color_forms(word)
            if not form.endswith(_NEUTER_ENDINGS)
        )
    return bag.has(word)


_STATS: weakref.WeakKeyDictionary[CatalogAttrs, CatalogStats] = weakref.WeakKeyDictionary()


def catalog_stats(attrs: CatalogAttrs) -> CatalogStats:
    """Сводки каталога, посчитанные один раз на объект таблицы."""
    stats = _STATS.get(attrs)
    if stats is None:
        stats = CatalogStats(attrs)
        _STATS[attrs] = stats
    return stats


# ------------------------------------------------------------------ опции и выход
@dataclass(frozen=True, slots=True)
class FeatureOptions:
    """`published=True` включает признак `unpublished`; множество берётся из разметки каталога."""

    published: bool = False
    unpublished: frozenset[str] = field(default_factory=frozenset)


DEFAULT_OPTIONS = FeatureOptions()


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    """Признаки всех кандидатов одного запроса в порядке выдачи CV."""

    slugs: tuple[str, ...]
    rows: tuple[dict[str, float], ...]

    def matrix(self, names: Sequence[str]) -> np.ndarray:
        """Матрица (кандидаты × признаки). Признака нет в строках — `KeyError`."""
        out = np.zeros((len(self.rows), len(names)), dtype=np.float64)
        for i, row in enumerate(self.rows):
            for j, name in enumerate(names):
                out[i, j] = row[name]
        return out


# ------------------------------------------------------------------ признаки
def _agree(prefix: str, name: str, agree: Agreement) -> dict[str, float]:
    return {
        f"{prefix}{name}_match": float(agree == "match"),
        f"{prefix}{name}_conflict": float(agree == "conflict"),
    }


def _color_agreement(fields: LabelFields | None, wine: WineAttrs | None) -> Agreement:
    read = fields.color.value if fields is not None and fields.color is not None else None
    if wine is None or read is None or wine.color is None:
        return "unknown"
    if read == wine.color:
        return "match"
    # «Белое» и «Оранжевое» не спорят, но и не совпадают (Э4): см. `COMPATIBLE_COLORS`
    return "unknown" if colors_compatible(read, wine.color) else "conflict"


def _cv_block(cv: CvQuery, attrs: CatalogAttrs, stats: CatalogStats) -> list[dict[str, float]]:
    scores = np.array([candidate.score for candidate in cv.candidates], dtype=np.float64)
    top1 = float(scores[0])
    mean, std, low = float(scores.mean()), float(scores.std()), float(scores.min())
    top_wine = attrs.get(cv.candidates[0].slug)
    rows = []
    for candidate in cv.candidates:
        is_top1 = candidate.rank == 1
        wine = attrs.get(candidate.slug)
        group = wine.visual_group if wine else None
        cluster = wine.cluster if wine else None
        rows.append(
            {
                "cv_score": candidate.score,
                "cv_rank": float(candidate.rank),
                "cv_rank_inv": 1.0 / candidate.rank,
                "cv_gap_top1": candidate.score - top1,
                "cv_z": (candidate.score - mean) / std if std > _EPS else 0.0,
                "cv_minmax": (candidate.score - low) / (top1 - low) if top1 - low > _EPS else 1.0,
                "cv_is_top1": float(is_top1),
                "cv_top1_x_margin": cv.margin if is_top1 else 0.0,
                "vgroup_size_log": math.log(stats.vgroup_sizes.get(group, 1) if group else 1),
                "vgroup_mate_of_top1": float(
                    not is_top1
                    and group is not None
                    and top_wine is not None
                    and group == top_wine.visual_group
                ),
                "cluster_size_log": math.log(stats.cluster_sizes.get(cluster, 1) if cluster else 1),
                "cluster_mate_of_top1": float(
                    not is_top1
                    and cluster is not None
                    and top_wine is not None
                    and cluster == top_wine.cluster
                ),
            }
        )
    return rows


def _text_block(
    prefix: str,
    cv: CvQuery,
    read: TextRead,
    attrs: CatalogAttrs,
    stats: CatalogStats,
    rows: list[dict[str, float]],
) -> None:
    keys: ReadKeys = read_fields(read.fields, attrs)
    bag = read_bag(read)
    found_counts: list[int] = []
    for candidate, row in zip(cv.candidates, rows, strict=True):
        wine = attrs.get(candidate.slug)
        for feature in ("winery", "grape", "sugar", "year", "serial", "abv", "cuvee"):
            row.update(_agree(prefix, feature, agreement(feature, wine, keys)))
        row.update(_agree(prefix, "color", _color_agreement(read.fields, wine)))
        row[f"{prefix}winery_match_conf"] = row[f"{prefix}winery_match"] * keys.winery_conf
        row[f"{prefix}winery_conflict_conf"] = row[f"{prefix}winery_conflict"] * keys.winery_conf

        share = idf = 0.0
        found = 0
        if wine is not None and bag:
            name_words = stats.winery_name_words(wine)
            if name_words:
                share = sum(bag.has(word) for word in name_words) / len(name_words)
            hits = [
                1.0 / stats.winery_df[word] for word in stats.winery_words(wine) if bag.has(word)
            ]
            idf = max(hits, default=0.0)
            found = sum(name_word_found(bag, word) for word in stats.name_words(wine))
        row[f"{prefix}winery_text_share"] = share
        row[f"{prefix}winery_text_idf"] = idf
        found_counts.append(found)

    best_found = max(found_counts, default=0)
    for row, found in zip(rows, found_counts, strict=True):
        row[f"{prefix}name_found"] = min(found, WORDS_CAP) / WORDS_CAP
        row[f"{prefix}name_found_rel"] = max(-WORDS_CAP, found - best_found) / WORDS_CAP


def _as_read(value: TextRead | LabelFields | None) -> TextRead:
    if isinstance(value, TextRead):
        return value
    return TextRead(fields=value)


def query_features(
    query_cv: QueryCv,
    reads: Mapping[str, TextRead | LabelFields | None],
    attrs: CatalogAttrs,
    *,
    top_k: int = DEFAULT_TOP_K,
    options: FeatureOptions = DEFAULT_OPTIONS,
) -> QueryFeatures:
    """Признаки всех кандидатов запроса.

    `reads` — чтения по имени читателя; имя становится префиксом текстовых признаков
    (`""` — без префикса). Пустой словарь — запрос без текста: только группы CV. Кандидатов
    нет — пустой результат.
    """
    cv = cv_query(query_cv, top_k=top_k)
    if not cv.candidates:
        return QueryFeatures((), ())
    stats = catalog_stats(attrs)
    rows = _cv_block(cv, attrs, stats)
    for reader in sorted(reads):
        _text_block(reader_prefix(reader), cv, _as_read(reads[reader]), attrs, stats, rows)
    if options.published:
        for candidate, row in zip(cv.candidates, rows, strict=True):
            row["unpublished"] = float(candidate.slug in options.unpublished)
    return QueryFeatures(cv.slugs, tuple(rows))


def pair_features(
    query_cv: QueryCv,
    fields: LabelFields | None,
    cand_slug: str,
    attrs: CatalogAttrs,
    *,
    raw_text: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    options: FeatureOptions = DEFAULT_OPTIONS,
) -> dict[str, float]:
    """Признаки одной пары «запрос — кандидат» (текст одного читателя, без префикса).

    Кандидат должен быть в top-K выдачи: относительные признаки определены только внутри
    неё. Иначе — `KeyError`.
    """
    features = query_features(
        query_cv,
        {"": TextRead(fields=fields, raw_text=raw_text)},
        attrs,
        top_k=top_k,
        options=options,
    )
    try:
        index = features.slugs.index(cand_slug)
    except ValueError:
        raise KeyError(f"{cand_slug} нет среди top-{top_k} кандидатов CV") from None
    return dict(features.rows[index])
