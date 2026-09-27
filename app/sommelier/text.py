"""Текст вопроса сомелье: нормализация, словоформы, отрицание, числа словами.

Перенос вспомогательных модулей «Лозы» — только стандартная библиотека, без словарей:

* `normalize` — `Code/backend/app/domain/taxonomy.py:243-256`: NFKC, «ё» → «е», нижний
  регистр, пунктуация → пробел; `fold_homoglyphs` схлопывает латинских двойников;
* `words_match`, `covered_words` — `core/text_match.py`: словоформы по общему началу и
  закрытому списку окончаний («борщу» — «борщ», «гуся» — «гусь»), без лемматизатора;
* `negated_at`, `refused_after`, `FORMS`, `PREFERENCE_FORMS`, `BOUNDARY` — `core/negation.py`:
  «нет у меня шашлыка» — отказ от блюда, «нет, борщ» — поправка;
* `plural` — `core/russian.py`: три формы слова по числу.

Добавлено здесь: `clean_question` — чистка вопроса по договору (`docs/api-sommelier.md`, §3.1:
до 200 символов после обрезки и схлопывания пробелов, управляющие символы вырезаются),
`with_boundaries` — слова с маркерами границ предложений (`recommend/pairing.py` «Лозы»), и
числительные словами для шаблонов ответа («три вина»).

Сам вопрос гостя нигде здесь не пишется в журнал: модули сомелье получают его только для
разбора (договор, §1 «Журнал»).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence

#: Предел вопроса после чистки (договор, §3.1).
MAX_QUESTION = 200

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)
#: Управляющие символы (категория Unicode C*): их вырезают до обрезки краёв.
_CONTROL_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})

#: Кириллические буквы, визуально неотличимые от латинских (только строчные: таблица
#: применяется после нижнего регистра, как у «Лозы»).
_HOMOGLYPHS = str.maketrans({"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"})


def clean_question(raw: str) -> str:
    """Вопрос после чистки: без управляющих символов, краёв и двойных пробелов.

    Перевод строки и табуляция становятся пробелом, остальные управляющие символы (и
    невидимые форматирующие вроде U+200B) вырезаются. Длину проверяет вызывающий.
    """
    chars = []
    for char in raw or "":
        if char in "\n\r\t\v\f":
            chars.append(" ")
        elif unicodedata.category(char) in _CONTROL_CATEGORIES:
            continue
        else:
            chars.append(char)
    return _SPACE_RE.sub(" ", "".join(chars)).strip()


def normalize(text: str, fold_homoglyphs: bool = False) -> str:
    """Сравнимый вид строки: NFKC, «ё» → «е», нижний регистр, пунктуация → пробел."""
    if not text:
        return ""
    result = unicodedata.normalize("NFKC", text).replace("ё", "е").replace("Ё", "Е")
    result = result.lower()
    if fold_homoglyphs:
        result = result.translate(_HOMOGLYPHS)
    result = _PUNCT_RE.sub(" ", result)
    return _SPACE_RE.sub(" ", result).strip()


# ------------------------------------------------------------------ словоформы
#: Сколько первых букв должны совпасть, чтобы счесть словоформы одним словом.
PREFIX_MATCH_LENGTH = 5
#: На сколько букв слово может быть длиннее общего начала, оставаясь словоформой.
MAX_FORM_TAIL = 3

_CYRILLIC_RE = re.compile(r"^[а-яё]+$")

#: Окончания русских существительных и прилагательных — закрытый список «Лозы» (его читает и
#: замок дескрипторов `entity_lock.py`).
ENDINGS = frozenset(
    (
        "", "а", "е", "ё", "и", "й", "о", "у", "ы", "ь", "э", "ю", "я",
        "ам", "ах", "ев", "ей", "ем", "ов", "ом", "ою", "ям", "ях",
        "ья", "ье", "ью", "ия", "ии", "ие", "ин", "ами", "ями",
        "ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие",
        "ого", "его", "ому", "ему", "ым", "им", "ых", "их", "ую", "юю",
        "ыми", "ими",
    )
)  # fmt: skip


def _russian_ending(tail: str) -> bool:
    """Похож ли остаток слова на окончание, а не на другое слово."""
    if not tail:
        return True
    if not _CYRILLIC_RE.match(tail):
        return True
    return tail in ENDINGS


def _is_tail(tail: str) -> bool:
    """Похож ли остаток на конец окончания: у «шампанское» и «шампанского» — «е» и «го»."""
    if _russian_ending(tail):
        return True
    return any(ending.endswith(tail) for ending in ENDINGS if ending)


def words_match(left: str, right: str) -> bool:
    """Считать ли две словоформы одним словом: «борщу» и «борщ» — да, «паста» и «пастила» — нет."""
    if left == right:
        return True
    if len(left) < PREFIX_MATCH_LENGTH or len(right) < PREFIX_MATCH_LENGTH:
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(shorter) == 4 and longer.startswith(shorter) and len(longer) - len(shorter) <= 2:
            return True
        return (
            len(left) == len(right) >= 4
            and left[:-1] == right[:-1]
            and _russian_ending(left[-1])
            and _russian_ending(right[-1])
        )
    limit = min(len(left), len(right))
    common = 0
    while common < limit and left[common] == right[common]:
        common += 1
    required = max(4, min(PREFIX_MATCH_LENGTH, limit - 1))
    if common < required and (
        common < 4 or not (_russian_ending(left[common:]) and _russian_ending(right[common:]))
    ):
        return False
    if max(len(left), len(right)) - common > MAX_FORM_TAIL:
        return False
    return _is_tail(left[common:]) and _is_tail(right[common:])


def covered_words(phrase_words: Sequence[str], query_words: Sequence[str]) -> set[str]:
    """Слова запроса, покрытые словами фразы, с учётом словоформ."""
    covered: set[str] = set()
    for word in phrase_words:
        for other in query_words:
            if words_match(word, other):
                covered.add(other)
                break
    return covered


def has_word_starting(words: Sequence[str], stems: Sequence[str]) -> bool:
    """Есть ли слово, начинающееся с одной из основ: «цен» не находится в «оцени»."""
    return any(word.startswith(tuple(stems)) for word in words)


def contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


# ------------------------------------------------------------------ отрицание
#: Слова, которыми отказываются: «нет у меня шашлыка», «без сахара».
FORMS = frozenset(("не", "нет", "нету", "без", "кроме", "никакой", "никакого", "никаких", "вместо"))
#: Граница простого предложения: отрицание через неё не переходит.
BOUNDARY = "¶"
#: Отрицания для ограничений вопроса: «нет» и «нету» отвергают прошлую подсказку.
PREFERENCE_FORMS = frozenset(("не", "без", "кроме", "никакой", "никакого", "никаких", "вместо"))
#: Отказ, стоящий после слова: «шашлыка-то у нас и нет».
TRAILING_FORMS = frozenset(("нет", "нету", "кончился", "кончилась", "закончился"))


def negated_at(
    words: Sequence[str],
    index: int,
    window: int = 2,
    transparent: Callable[[str], bool] | None = None,
    forms: frozenset[str] | None = None,
) -> bool:
    """Отрицается ли слово `words[index]`: отрицание слева не дальше `window` слов."""
    if index <= 0:
        return False
    vocabulary = FORMS if forms is None else forms

    def passable(word: str) -> bool:
        if word in vocabulary or word == BOUNDARY:
            return False
        return True if transparent is None else transparent(word)

    for step in range(1, min(window, index) + 1):
        word = words[index - step]
        if word == BOUNDARY:
            return False
        if word in vocabulary:
            return True
        if not passable(word):
            return False
    return False


def refused_after(
    words: Sequence[str],
    index: int,
    window: int = 4,
    bridge: Callable[[str], bool] | None = None,
) -> bool:
    """Отказ после слова: «а шашлыка-то у нас и нет», «борща не будет»."""
    for step in range(1, window + 1):
        position = index + step
        if position >= len(words):
            return False
        word = words[position]
        if word == BOUNDARY:
            return False
        if word in TRAILING_FORMS:
            return True
        if word == "не" and list(words[position + 1 : position + 2]) == ["будет"]:
            return True
        if bridge is not None and not bridge(word):
            return False
    return False


def with_boundaries(text: str) -> list[str]:
    """Слова вопроса с маркером `BOUNDARY` на месте запятых и точек (только для отрицаний)."""
    words: list[str] = []
    for chunk in re.split("[,.;:!?—]+", (text or "").replace("\n", " ")):
        piece = normalize(chunk, fold_homoglyphs=True).split()
        if not piece:
            continue
        if words:
            words.append(BOUNDARY)
        words.extend(piece)
    return words


# ------------------------------------------------------------------ числа словами
def plural(count: int, forms: tuple[str, str, str]) -> str:
    """Форма слова по числу: `("вино", "вина", "вин")` — одно вино, три вина, пять вин."""
    absolute = abs(count) % 100
    if 11 <= absolute <= 14:
        return forms[2]
    last = absolute % 10
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


WINES = ("вино", "вина", "вин")
#: Числа словами для шаблонов — в шаблоне ответа цифр нет (договор, §4.5).
_COUNT_WORDS_NEUTER = {1: "одно", 2: "два", 3: "три", 4: "четыре", 5: "пять", 6: "шесть"}


def count_wines(count: int) -> str:
    """«три вина», «одно вино» — числом словами и согласованным словом."""
    word = _COUNT_WORDS_NEUTER.get(count, str(count))
    return f"{word} {plural(count, WINES)}"


def join_words(items: Sequence[str], *, conjunction: str = "и") -> str:
    """«А», «А и Б», «А, Б и В»."""
    items = [item for item in items if item]
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conjunction} {items[-1]}"


def lower_first(text: str) -> str:
    """Первая буква строчная: «Гусь запечённый» → «гусь запечённый»."""
    return text[:1].lower() + text[1:] if text else text


def upper_first(text: str) -> str:
    """Первая буква прописная: «к борщу» → «К борщу»."""
    return text[:1].upper() + text[1:] if text else text
