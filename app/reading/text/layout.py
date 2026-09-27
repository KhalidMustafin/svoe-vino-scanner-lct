"""Соседство строк на кадре: где фраза может продолжиться из одной строки чтения в другую.

Читатели режут текст по-разному. EasyOCR отдаёт каждое слово отдельной строкой («Каберне» и
«Совиньон» в одном ряду), RapidOCR — строку целиком, VLM — строки без рамок. Фраза словаря
или таксономии не должна зависеть от того, как читатель порезал ряд. Но и склеивать слова с
разных мест кадра нельзя: «Пино» с одной бутылки и «Нуар» с соседней — не сорт.

Правило для двух строк, идущих подряд в порядке чтения:
- у одной из них нет рамки — продолжение (VLM, скрипт): порядок чтения — всё, что известно;
- рамки в одном ряду (перекрытие по вертикали не меньше `ROW_OVERLAP` меньшей высоты) и
  разрыв по горизонтали не больше `ROW_GAP_CHARS` ширин буквы — продолжение;
- рамки друг под другом (перекрываются по горизонтали) с зазором по вертикали не больше
  `STACK_GAP` высоты строки — продолжение: название в две строки («АБРАУ» над «ДЮРСО»);
- иначе — разрыв.

Доли кадра по x и y разного масштаба, поэтому по горизонтали мерим ширинами буквы, по
вертикали — высотами строки. Пороги ручные и не подбирались: полевых замеров ещё нет.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.reading.contracts import Reading, TextLine, TokenSpan

#: Одна строка кадра: рамки перекрываются по вертикали хотя бы на эту долю меньшей высоты.
ROW_OVERLAP = 0.5
#: Слова одного ряда: разрыв не больше стольких ширин буквы.
ROW_GAP_CHARS = 3.0
#: Строки друг под другом: зазор не больше стольких высот более высокой строки.
STACK_GAP = 1.0


def _char_width(width: float, text: str) -> float:
    return width / max(1, sum(not ch.isspace() for ch in text))


def continues(before: TextLine, after: TextLine) -> bool:
    """Может ли фраза с конца строки `before` продолжиться в строке `after`."""
    a, b = before.box, after.box
    if a is None or b is None:
        return True
    low, high = sorted((a.y1 - a.y0, b.y1 - b.y0))
    overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
    gap_x = max(a.x0, b.x0) - min(a.x1, b.x1)  # < 0 — рамки перекрываются по горизонтали
    if overlap_y >= ROW_OVERLAP * low:
        char = max(_char_width(a.x1 - a.x0, before.text), _char_width(b.x1 - b.x0, after.text))
        return gap_x <= ROW_GAP_CHARS * char
    return gap_x < 0 and -overlap_y <= STACK_GAP * high


def token_segments(tokens: Sequence[TokenSpan], readings: Sequence[Reading]) -> list[int]:
    """Номер отрезка текста для каждого токена: фраза не выходит за свой отрезок.

    Новый отрезок начинается с новым чтением и там, где строка не продолжает предыдущую
    (`continues`). Строки, не давшие токенов (одна пунктуация), в сравнение не входят:
    сравниваются строки соседних токенов.
    """
    lines = {(reading.key, line.id): line for reading in readings for line in reading.lines}
    segments: list[int] = []
    current = -1
    prev: TokenSpan | None = None
    for token in tokens:
        if prev is None or token.reading != prev.reading:
            current += 1
        elif token.line_id != prev.line_id:
            before = lines.get((prev.reading, prev.line_id))
            after = lines.get((token.reading, token.line_id))
            if before is not None and after is not None and not continues(before, after):
                current += 1
        segments.append(current)
        prev = token
    return segments
