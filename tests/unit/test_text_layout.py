import pytest

from app.reading.contracts import Box, Reading, TextLine
from app.reading.text.layout import continues, token_segments
from app.reading.text.tokenize import tokenize


def line(text: str, box: tuple[float, float, float, float] | None, id: int = 0) -> TextLine:
    x0, y0, x1, y1 = box if box is not None else (0, 0, 1, 1)
    return TextLine(id=id, text=text, box=None if box is None else Box(x0=x0, y0=y0, x1=x1, y1=y1))


def reading(*lines: TextLine, reader: str = "ocr") -> Reading:
    return Reading(
        reader=reader,
        version="1",
        params_hash="p",
        image_sha1="i",
        crop="full",
        crop_px=1024,
        lines=list(lines),
        elapsed_ms=1,
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (None, (0.30, 0.80, 0.60, 0.85)),
        ((0.30, 0.10, 0.60, 0.15), None),
    ],
)
def test_line_without_box_continues(before, after):
    assert continues(line("Каберне", before), line("Совиньон", after))


def test_words_of_one_row_continue():
    # EasyOCR: каждое слово — своя рамка в одном ряду.
    assert continues(
        line("Каберне", (0.10, 0.40, 0.30, 0.45)), line("Совиньон", (0.33, 0.40, 0.55, 0.46))
    )


def test_stacked_title_continues():
    assert continues(
        line("АБРАУ", (0.30, 0.20, 0.60, 0.26)), line("ДЮРСО", (0.32, 0.27, 0.58, 0.33))
    )


@pytest.mark.parametrize(
    "after",
    [
        (0.70, 0.40, 0.80, 0.45),  # тот же ряд, но далеко: соседняя бутылка или колонка
        (0.10, 0.80, 0.30, 0.85),  # под строкой, но далеко по вертикали
        (0.60, 0.47, 0.80, 0.52),  # следующий ряд без перекрытия по горизонтали
    ],
)
def test_distant_lines_are_a_break(after):
    assert not continues(line("Пино", (0.05, 0.40, 0.20, 0.45)), line("Нуар", after))


def test_token_segments_follow_readings_and_breaks():
    boxed = reading(
        line("Пино", (0.20, 0.40, 0.32, 0.45), id=0),
        line("Нуар", (0.34, 0.40, 0.46, 0.45), id=1),
        line("2021", (0.80, 0.90, 0.90, 0.95), id=2),
    )
    plain = reading(line("Блан", None, id=0), line("Нуар", None, id=1), reader="vlm")
    tokens = tokenize(boxed) + tokenize(plain)
    assert [t.norm for t in tokens] == ["пино", "нуар", "2021", "блан", "нуар"]
    assert token_segments(tokens, [boxed, plain]) == [0, 0, 1, 2, 2]


def test_token_segments_without_readings_keep_reading_boundaries():
    tokens = tokenize(reading(line("Пино", None), reader="a")) + tokenize(
        reading(line("Нуар", None), reader="b")
    )
    assert token_segments(tokens, []) == [0, 1]
