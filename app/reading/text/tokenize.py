"""Разбиение чтения на токены: слова, числа, годы, крепость, римские номера, доли купажа.

Токенизатор ничего не выбрасывает и ничего не решает: он размечает, что
написано, а стоп-слова, словарь и поля — дело следующих слоёв. Разорванные
OCR токены склеиваются по общим правилам («2 0 2 3», «13,5 % об.», «30 / 70»,
«Б Р Ю Т»), а слова — только если склейка есть в словаре (`lexicon`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Container
from dataclasses import dataclass
from itertools import pairwise

from app.reading.contracts import Reading, TextLine, TokenKind, TokenSpan
from app.reading.text.normalize import ROMAN_RE, fold_homoglyphs, norm_token, roman_value
from app.reading.text.translit import skeleton

# Надстрочные знаки, которые NFKC не прикрепил к букве.
_COMBINING_CLASS = f"{chr(0x0300)}-{chr(0x036F)}"
# Кусок строки: буквы, цифры, «%» и разделители внутри чисел.
_CHUNK_RE = re.compile(rf"(?:[^\W_]|[{_COMBINING_CLASS}]|%|(?<=\d)[.,/](?=\d))+")
# Части смешанного куска: «12%vol» → «12%», «vol».
_PART_RE = re.compile(rf"\d+(?:[.,/]\d+)*%?|%+|(?:[^\W\d_]|[{_COMBINING_CLASS}])+")
_NUMERIC_RE = re.compile(r"\d+(?:[.,/]\d+)*%?")
_DECIMAL_COMMA_RE = re.compile(r"\d+,\d+%?")
_PLAIN_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_ABV_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)%?")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_RATIO_RE = re.compile(r"\d{1,2}(?:/\d{1,2})+")
_RATIO_PART_RE = re.compile(r"\d{1,2}")
_ROMAN_SHORT_RE = re.compile(r"[ivxlc]{2,6}")

#: Единицы крепости после числа: «13 % об.», «12,5% vol».
ABV_UNITS = frozenset({"об", "vol", "ob"})
#: Правдоподобная крепость вина; «100%» — доля сорта, а не крепость.
ABV_RANGE = (3.0, 25.0)
#: Годы из римских чисел: MMXXII.
_ROMAN_YEAR_RANGE = (1900, 2099)
# Что может стоять в разрыве склеиваемого слова.
_BREAK_GAP_CHARS = " \t-‐‑–—.·'’"


@dataclass(slots=True)
class _Piece:
    start: int
    end: int
    norm: str


def _canon_number(value: str) -> str:
    """Десятичная запятая → точка: «13,5%» → «13.5%», «0,75» → «0.75»."""
    if _DECIMAL_COMMA_RE.fullmatch(value):
        return value.replace(",", ".")
    return value


def _needs_split(parts: list[re.Match[str]]) -> bool:
    """Смешанный кусок режется, если в нём настоящее число, а не цифра-двойник буквы."""
    for part in parts:
        text = part.group()
        if text[0] == "%":
            return True
        if text[0].isdigit() and (len(text) >= 2 or any(ch in ".,/%" for ch in text)):
            return True
    return False


def _line_pieces(text: str) -> list[_Piece]:
    pieces: list[_Piece] = []
    for chunk in _CHUNK_RE.finditer(text):
        whole = _canon_number(norm_token(chunk.group()))
        parts = list(_PART_RE.finditer(chunk.group()))
        if len(parts) > 1 and not _NUMERIC_RE.fullmatch(whole) and _needs_split(parts):
            for part in parts:
                value = _canon_number(norm_token(part.group()))
                if value:
                    start = chunk.start() + part.start()
                    pieces.append(_Piece(start, chunk.start() + part.end(), value))
        elif whole:
            pieces.append(_Piece(chunk.start(), chunk.end(), whole))
    return pieces


def _gap(text: str, left: _Piece, right: _Piece) -> str:
    return text[left.end : right.start]


def _merge_letter_spaced_year(run: list[_Piece]) -> list[_Piece]:
    """Разрядка по одной цифре: четыре цифры подряд, дающие год, — один кусок."""
    out: list[_Piece] = []
    i = 0
    while i < len(run):
        window = run[i : i + 4]
        digits = "".join(p.norm for p in window)
        if len(window) == 4 and _YEAR_RE.fullmatch(digits):
            out.append(_Piece(window[0].start, window[-1].end, digits))
            i += 4
        else:
            out.append(run[i])
            i += 1
    return out


def _merge_spaced_years(pieces: list[_Piece], text: str) -> list[_Piece]:
    """«2 0 2 3» и «20 23» → год.

    Группа — числа, разделённые только пробелами. Разрядка по одной цифре склеивается внутри
    группы, а группа с двузначными кусками — только целиком: иначе куски телефона или адреса
    («8 800 20 23 000») дали бы год.
    """
    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        end = i + 1
        if pieces[i].norm.isdigit():
            while (
                end < len(pieces)
                and pieces[end].norm.isdigit()
                and not _gap(text, pieces[end - 1], pieces[end]).strip()
            ):
                end += 1
        run = pieces[i:end]
        digits = "".join(p.norm for p in run)
        if all(len(p.norm) == 1 for p in run):
            out.extend(_merge_letter_spaced_year(run))
        elif all(len(p.norm) <= 2 for p in run) and _YEAR_RE.fullmatch(digits):
            out.append(_Piece(run[0].start, run[-1].end, digits))
        else:
            out.extend(run)
        i = end
    return out


def _merge_spaced_ratios(pieces: list[_Piece], text: str) -> list[_Piece]:
    """«30 / 70» → «30/70»."""
    out: list[_Piece] = []
    for piece in pieces:
        if (
            out
            and _RATIO_PART_RE.fullmatch(piece.norm)
            and (_RATIO_PART_RE.fullmatch(out[-1].norm) or _RATIO_RE.fullmatch(out[-1].norm))
            and _gap(text, out[-1], piece).strip() == "/"
        ):
            prev = out[-1]
            out[-1] = _Piece(prev.start, piece.end, f"{prev.norm}/{piece.norm}")
        else:
            out.append(piece)
    return out


def _abv_value(value: str) -> float | None:
    match = _ABV_RE.fullmatch(value)
    if not match:
        return None
    number = float(match.group(1))
    return number if ABV_RANGE[0] <= number <= ABV_RANGE[1] else None


def _merge_abv(pieces: list[_Piece], text: str) -> list[_Piece]:
    """Число + «%» и число(%) + «об»/«vol» — один токен. «11 обл» не склеивается."""
    out: list[_Piece] = []
    for piece in pieces:
        if out:
            prev = out[-1]
            gap = _gap(text, prev, piece).strip()
            if piece.norm == "%" and not gap and _PLAIN_NUMBER_RE.fullmatch(prev.norm):
                out[-1] = _Piece(prev.start, piece.end, f"{prev.norm}%")
                continue
            if piece.norm in ABV_UNITS and gap in ("", ".") and _abv_value(prev.norm) is not None:
                out[-1] = _Piece(prev.start, piece.end, prev.norm.rstrip("%") + "%")
                continue
        out.append(piece)
    return out


def _is_word(piece: _Piece) -> bool:
    return piece.norm.isalpha()


def _merge_spaced_letters(pieces: list[_Piece], text: str) -> list[_Piece]:
    """Разрядка «Б Р Ю Т» — одно слово: три и больше одиночных букв подряд."""
    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        j = i
        while (
            j < len(pieces)
            and len(pieces[j].norm) == 1
            and _is_word(pieces[j])
            and (j == i or not _gap(text, pieces[j - 1], pieces[j]).strip())
        ):
            j += 1
        if j - i >= 3:
            joined = fold_homoglyphs("".join(p.norm for p in pieces[i:j]))
            out.append(_Piece(pieces[i].start, pieces[j - 1].end, joined))
            i = j
        else:
            out.append(pieces[i])
            i += 1
    return out


def _merge_lexicon(pieces: list[_Piece], text: str, lexicon: Container[str]) -> list[_Piece]:
    """Склейка слов, разорванных бликом: «ЮЖН ОБЕРЕЖНЫЙ» → «южнобережный».

    Принимается, только если склейка есть в словаре (по норме или скелету):
    без этой проверки соседние слова слипались бы в мусор.
    """
    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        for size in (3, 2):
            group = pieces[i : i + size]
            if len(group) < size or not all(_is_word(p) for p in group):
                continue
            gaps = (_gap(text, a, b) for a, b in pairwise(group))
            if any(gap.strip(_BREAK_GAP_CHARS) for gap in gaps):
                continue
            joined = fold_homoglyphs("".join(p.norm for p in group))
            if joined in lexicon or skeleton(joined) in lexicon:
                out.append(_Piece(group[0].start, group[-1].end, joined))
                i += size
                break
        else:
            out.append(pieces[i])
            i += 1
    return out


def _classify(piece: _Piece, raw: str) -> tuple[TokenKind, str]:
    value = piece.norm
    if _YEAR_RE.fullmatch(value):
        return "year", value
    if value.endswith("%") and _NUMERIC_RE.fullmatch(value):
        return ("abv" if _abv_value(value) is not None else "number"), value
    if _RATIO_RE.fullmatch(value) and sum(int(p) for p in value.split("/")) == 100:
        return "ratio", value
    if _NUMERIC_RE.fullmatch(value):
        return "number", value
    if any(ch.isdigit() for ch in value):
        return "word", value  # бренды вроде «K2», «V2R»
    letters = [ch for ch in raw if ch.isalpha()]
    # Римские номера — только заглавными: строчные «vi», «li» — слова.
    if len(letters) >= 2 and all(ch.isupper() for ch in letters):
        if _ROMAN_SHORT_RE.fullmatch(value) and ROMAN_RE.fullmatch(value):
            return "roman", value
        if "m" in value:
            year = roman_value(value)
            if year is not None and _ROMAN_YEAR_RANGE[0] <= year <= _ROMAN_YEAR_RANGE[1]:
                return "year", str(year)
    return "word", value


def tokenize_line(
    line: TextLine, *, reading: str, lexicon: Container[str] | None = None
) -> list[TokenSpan]:
    """Токены одной строки. `reading` — ключ чтения (`Reading.key`)."""
    text = unicodedata.normalize("NFKC", line.text)
    pieces = _line_pieces(text)
    pieces = _merge_spaced_years(pieces, text)
    pieces = _merge_spaced_ratios(pieces, text)
    pieces = _merge_abv(pieces, text)
    pieces = _merge_spaced_letters(pieces, text)
    if lexicon is not None:
        pieces = _merge_lexicon(pieces, text, lexicon)

    tokens: list[TokenSpan] = []
    for piece in pieces:
        if not piece.norm.strip("%"):
            continue  # одинокий «%» без числа
        raw = text[piece.start : piece.end]
        kind, value = _classify(piece, raw)
        tokens.append(
            TokenSpan(
                text=raw,
                norm=value,
                skeleton=skeleton(value) if kind == "word" else value,
                kind=kind,
                line_id=line.id,
                reading=reading,
                conf=line.conf,
            )
        )
    return tokens


def tokenize(reading: Reading, *, lexicon: Container[str] | None = None) -> list[TokenSpan]:
    """Токены всех строк чтения в порядке строк.

    `lexicon` — множество норм и/или скелетов словаря каталога; без него слова
    не склеиваются, остальные склейки (годы, крепость, доли, разрядка) работают.
    """
    key = reading.key
    return [
        token
        for line in reading.lines
        for token in tokenize_line(line, reading=key, lexicon=lexicon)
    ]
