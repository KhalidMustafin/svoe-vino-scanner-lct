"""Нормализация текста этикетки.

`norm` приводит строку к сравнимому виду, `fold_homoglyphs` сводит смешанные
алфавиты одного токена к преобладающему. Порядок важен: диакритика снимается
до определения алфавита, иначе «Château» с кириллической «с» не распознаётся
как латиница (на этом ошибалась «Лоза»).
"""

from __future__ import annotations

import re
import unicodedata

# Буквы, у которых «диакритика» — часть самой буквы.
_KEEP_MARKED = frozenset("йЙёЁїЇ")

# Латинские буквы без разложения в NFD.
_UNDECOMPOSABLE = {
    "ß": "ss",
    "ẞ": "SS",
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "ł": "l",
    "Ł": "L",
    "đ": "d",
    "Đ": "D",
    "ı": "i",
}

# Кириллица ↔ латиница, неотличимые на глаз. Строчные м/т/к/н/в — следы
# заглавных: «MACCAHДPA» после нижнего регистра — «maccahдpa».
_HOMOGLYPH_PAIRS = (
    ("А", "A"),
    ("В", "B"),
    ("Е", "E"),
    ("К", "K"),
    ("М", "M"),
    ("Н", "H"),
    ("О", "O"),
    ("Р", "P"),
    ("С", "C"),
    ("Т", "T"),
    ("Х", "X"),
    ("У", "Y"),
    ("І", "I"),
    ("а", "a"),
    ("в", "b"),
    ("е", "e"),
    ("к", "k"),
    ("м", "m"),
    ("н", "h"),
    ("о", "o"),
    ("р", "p"),
    ("с", "c"),
    ("т", "t"),
    ("х", "x"),
    ("у", "y"),
    ("і", "i"),
)
_CYR_TO_LAT = dict(_HOMOGLYPH_PAIRS)
_LAT_TO_CYR = {lat: cyr for cyr, lat in _HOMOGLYPH_PAIRS}

# Цифры, которые OCR ставит вместо букв, и обратно.
_DIGIT_TO_CYR = {"0": "о", "3": "з", "6": "б"}
_DIGIT_TO_LAT = {"0": "o"}
_LETTER_TO_DIGIT = {"o": "0", "O": "0", "о": "0", "О": "0", "з": "3", "З": "3", "б": "6"}

#: Корректное римское число в нижнем регистре (пустая строка тоже совпадает).
ROMAN_RE = re.compile(r"m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})")
_ROMAN_VALUES = {"m": 1000, "d": 500, "c": 100, "l": 50, "x": 10, "v": 5, "i": 1}
_ROMAN_SHORT_RE = re.compile(r"[ivxlc]{2,6}")

_FINAL_HARD_SIGN_RE = re.compile(r"(?<=[^\W\d_])ъ(?![^\W\d_])")
_SPACED_SLASH_RE = re.compile(r"(?<=\d)[ \t]*/[ \t]*(?=\d)")
_NUMBER_JOINERS = frozenset("/.,")


def strip_diacritics(text: str) -> str:
    """Снимает надстрочные знаки: «Château» — «Chateau», «Satèn» — «Saten».

    «й», «ё» и «ї» остаются: для кириллицы это отдельные буквы.
    """
    out: list[str] = []
    for ch in text:
        if ch in _KEEP_MARKED:
            out.append(ch)
        elif ch in _UNDECOMPOSABLE:
            out.append(_UNDECOMPOSABLE[ch])
        elif unicodedata.combining(ch):
            continue  # отдельный знак, который NFKC не прикрепил к букве
        else:
            parts = unicodedata.normalize("NFD", ch)
            if len(parts) > 1 and all(unicodedata.combining(p) for p in parts[1:]):
                out.append(parts[0])
            else:
                out.append(ch)
    return "".join(out)


def norm(text: str) -> str:
    """NFKC, нижний регистр, без диакритики, ё→е, пунктуация → пробел.

    Внутри чисел остаются «,», «.» и «/» («13,5», «0.75», «30/70»), знак «%»
    остаётся всегда. Конечный дореформенный «ъ» снимается («Ведерниковъ»).
    Гомоглифы не сводятся: это делает `fold_homoglyphs` по токенам.
    """
    if not text:
        return ""
    text = strip_diacritics(unicodedata.normalize("NFKC", text).lower()).replace("ё", "е")
    text = _FINAL_HARD_SIGN_RE.sub("", text)
    text = _SPACED_SLASH_RE.sub("/", text)
    last = len(text) - 1
    chars: list[str] = []
    for i, ch in enumerate(text):
        inside_number = (
            ch in _NUMBER_JOINERS
            and 0 < i < last
            and text[i - 1].isdigit()
            and text[i + 1].isdigit()
        )
        chars.append(ch if ch.isalnum() or ch == "%" or inside_number else " ")
    return " ".join("".join(chars).split())


def roman_value(token: str) -> int | None:
    """Значение корректного римского числа или None: «XXIV» — 24, «IIII» — None."""
    token = token.lower()
    if not token or not ROMAN_RE.fullmatch(token):
        return None
    total = 0
    for i, ch in enumerate(token):
        value = _ROMAN_VALUES[ch]
        if i + 1 < len(token) and _ROMAN_VALUES[token[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def _is_cyrillic(ch: str) -> bool:
    return 0x0400 <= ord(ch) <= 0x052F


def _is_latin(ch: str) -> bool:
    code = ord(ch)
    return 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A or 0xC0 <= code <= 0x24F


def _letter_context_digits(token: str, n_letters: int) -> list[int]:
    """Позиции одиночных 0/3/6 между буквами: «М0СКВА», «3ОЛОТАЯ», но не «750мл»."""
    if n_letters < 2:
        return []
    slots = []
    for i, ch in enumerate(token):
        if ch not in _DIGIT_TO_CYR:
            continue
        prev = token[i - 1] if i > 0 else ""
        nxt = token[i + 1] if i + 1 < len(token) else ""
        if prev.isdigit() or nxt.isdigit():
            continue
        if prev.isalpha() or nxt.isalpha():
            slots.append(i)
    return slots


def _dominant_script(token: str, cyr: list[str], lat: list[str], digits: list[int]) -> str:
    if not lat:
        return "cyr"
    if not cyr:
        return "lat"
    # Решают буквы, у которых нет двойника в другом алфавите.
    strong_cyr = sum(ch not in _CYR_TO_LAT for ch in cyr)
    strong_lat = sum(ch not in _LAT_TO_CYR for ch in lat)
    if strong_cyr != strong_lat:
        return "cyr" if strong_cyr > strong_lat else "lat"
    if any(token[i] in "36" for i in digits):
        return "cyr"
    if len(cyr) != len(lat):
        return "cyr" if len(cyr) > len(lat) else "lat"
    # Ничья из одних двойников: «ХI» — римское число, иначе кириллица каталога.
    latin_form = "".join(_CYR_TO_LAT.get(ch, ch) for ch in token).lower()
    if _ROMAN_SHORT_RE.fullmatch(latin_form) and ROMAN_RE.fullmatch(latin_form):
        return "lat"
    return "cyr"


def fold_homoglyphs(token: str) -> str:
    """Сводит смешанный токен к преобладающему алфавиту.

    «MACCAHДPA» → «МАССАНДРА», «сhâtеаu» → «chateau» (после `strip_diacritics`),
    «М0СКВА» → «МОСКВА», «2О23» → «2023». Токен одного алфавита без цифр-двойников
    возвращается как есть — в том числе римские «XXIV» и «III».
    """
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return token
    n_digits = sum(ch.isdigit() for ch in token)
    if n_digits >= 2 and len(letters) < n_digits and all(ch in _LETTER_TO_DIGIT for ch in letters):
        return "".join(_LETTER_TO_DIGIT.get(ch, ch) for ch in token)

    cyr = [ch for ch in letters if _is_cyrillic(ch)]
    lat = [ch for ch in letters if _is_latin(ch)]
    digits = _letter_context_digits(token, len(letters))
    if not (cyr and lat) and not digits:
        return token

    script = _dominant_script(token, cyr, lat, digits)
    letter_table = _LAT_TO_CYR if script == "cyr" else _CYR_TO_LAT
    digit_table = _DIGIT_TO_CYR if script == "cyr" else _DIGIT_TO_LAT
    upper = all(ch.isupper() for ch in letters)
    out = [letter_table.get(ch, ch) for ch in token]
    for i in digits:
        replacement = digit_table.get(token[i])
        if replacement:
            out[i] = replacement.upper() if upper else replacement
    return "".join(out)


def norm_token(text: str) -> str:
    """`norm` и `fold_homoglyphs` для одного токена."""
    return fold_homoglyphs(norm(text).replace(" ", ""))
