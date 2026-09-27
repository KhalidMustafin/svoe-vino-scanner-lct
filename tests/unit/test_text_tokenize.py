import pytest

from app.reading.contracts import Reading, TextLine
from app.reading.text.tokenize import tokenize, tokenize_line
from app.reading.text.translit import skeleton


def make_reading(*lines: str, conf: float | None = 0.9) -> Reading:
    return Reading(
        reader="test",
        version="0",
        params_hash="p",
        image_sha1="i",
        crop="label",
        crop_px=768,
        lines=[TextLine(id=10 + i, text=text, conf=conf) for i, text in enumerate(lines)],
        elapsed_ms=1,
    )


def kinds(line: str, **kwargs) -> list[tuple[str, str]]:
    return [(t.kind, t.norm) for t in tokenize(make_reading(line), **kwargs)]


def test_tokens_carry_reading_line_and_confidence():
    reading = make_reading("Шато Тамань", "Каберне", conf=0.42)
    tokens = tokenize(reading)
    assert [t.text for t in tokens] == ["Шато", "Тамань", "Каберне"]
    assert [t.line_id for t in tokens] == [10, 10, 11]
    assert {t.reading for t in tokens} == {reading.key}
    assert {t.conf for t in tokens} == {0.42}


def test_missing_confidence_stays_missing():
    assert tokenize(make_reading("Брют", conf=None))[0].conf is None


def test_empty_reading_gives_no_tokens():
    assert tokenize(make_reading()) == []
    assert tokenize(make_reading("  —  ")) == []


def test_word_norm_and_skeleton():
    (token,) = tokenize(make_reading("Château"))
    assert (token.text, token.norm, token.skeleton, token.kind) == (
        "Château",
        "chateau",
        skeleton("Шато"),
        "word",
    )


def test_homoglyph_word_is_folded():
    (token,) = tokenize(make_reading("MACCAHДPA"))
    assert token.norm == "массандра"
    assert token.skeleton == skeleton("Massandra")


def test_old_orthography_hard_sign():
    (token,) = tokenize(make_reading("Ведерниковъ"))
    assert token.norm == "ведерников"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("2023", [("year", "2023")]),
        ("1999", [("year", "1999")]),
        ("2 0 2 3", [("year", "2023")]),
        ("20 23", [("year", "2023")]),
        ("урожай 2 0 1 9 года", [("word", "урожай"), ("year", "2019"), ("word", "года")]),
        ("2023г.", [("year", "2023"), ("word", "г")]),
        ("2О23", [("year", "2023")]),
    ],
)
def test_years(line, expected):
    assert kinds(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("1 2 3", [("number", "1"), ("number", "2"), ("number", "3")]),
        ("1850", [("number", "1850")]),  # не 19xx/20xx
        ("12.03.24", [("number", "12.03.24")]),  # дата — не год
        ("2 0 2 3 1", [("year", "2023"), ("number", "1")]),
        ("1 2 0 2 3", [("number", "1"), ("year", "2023")]),
        # Двузначные куски склеиваются в год только всей группой: телефон и адрес — не год.
        (
            "8 800 20 23 000",
            [("number", n) for n in ("8", "800", "20", "23", "000")],
        ),
        ("20 23 45", [("number", "20"), ("number", "23"), ("number", "45")]),
    ],
)
def test_not_years(line, expected):
    assert kinds(line) == expected


def test_spaced_year_keeps_original_text():
    (token,) = tokenize(make_reading("2 0 2 3"))
    assert token.text == "2 0 2 3"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("БРЮТ XXIV", [("word", "брют"), ("roman", "xxiv")]),
        ("III", [("roman", "iii")]),
        ("ХХIV", [("roman", "xxiv")]),  # кириллические «Х»
        ("V", [("word", "v")]),  # одиночная V — не номер
        ("vi", [("word", "vi")]),  # строчные — слово
        ("IIII", [("word", "iiii")]),  # некорректное число
        ("MMXXII", [("year", "2022")]),
        ("DI CASPICO", [("word", "di"), ("word", "caspico")]),
    ],
)
def test_roman_numerals(line, expected):
    assert kinds(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("13,5% об.", [("abv", "13.5%")]),
        ("13,5 % об", [("abv", "13.5%")]),
        ("12% vol.", [("abv", "12%")]),
        ("alc. 11,5%vol", [("word", "alc"), ("abv", "11.5%")]),
        ("12,5 об", [("abv", "12.5%")]),
        ("9%", [("abv", "9%")]),
    ],
)
def test_abv(line, expected):
    assert kinds(line) == expected


def test_abv_keeps_the_whole_surface_text():
    (token,) = tokenize(make_reading("крепость 13,5% об."))[1:]
    assert token.text == "13,5% об"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("11 обл", [("number", "11"), ("word", "обл")]),  # «об» не ловит «обл»
        ("100% Саперави", [("number", "100%"), ("word", "саперави")]),  # доля сорта
        ("выдержка 18 месяцев", [("word", "выдержка"), ("number", "18"), ("word", "месяцев")]),
        ("0,75 л", [("number", "0.75"), ("word", "л")]),
        ("750мл", [("number", "750"), ("word", "мл")]),
    ],
)
def test_numbers_that_are_not_abv(line, expected):
    assert kinds(line) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("30/70", [("ratio", "30/70")]),
        ("30 / 70", [("ratio", "30/70")]),
        ("40/30/30", [("ratio", "40/30/30")]),
        ("15/08", [("number", "15/08")]),  # сумма не 100 — не купаж
    ],
)
def test_ratios(line, expected):
    assert kinds(line) == expected


def test_alphanumeric_brands_are_words():
    assert kinds("Le K2 V2R") == [("word", "le"), ("word", "k2"), ("word", "v2r")]


def test_letter_spaced_word_is_merged():
    assert kinds("Б Р Ю Т") == [("word", "брют")]
    assert kinds("M A C C A H Д P A") == [("word", "массандра")]


def test_two_single_letters_are_not_merged():
    assert kinds("в и") == [("word", "в"), ("word", "и")]


def test_broken_word_merged_only_through_lexicon():
    line = "ЮЖН ОБЕРЕЖНЫЙ"
    assert kinds(line) == [("word", "южн"), ("word", "обережный")]
    assert kinds(line, lexicon={"южнобережный"}) == [("word", "южнобережный")]
    assert kinds(line, lexicon={skeleton("южнобережный")}) == [("word", "южнобережный")]
    assert kinds(line, lexicon={"южный"}) == [("word", "южн"), ("word", "обережный")]


def test_lexicon_merge_of_three_pieces_with_hyphen():
    reading = make_reading("МУС-КА ТЕЛЬ")
    tokens = tokenize(reading, lexicon={"мускатель"})
    assert [(t.kind, t.norm, t.text) for t in tokens] == [("word", "мускатель", "МУС-КА ТЕЛЬ")]


def test_lexicon_does_not_glue_numbers():
    assert kinds("2023 год", lexicon={"2023год"}) == [("year", "2023"), ("word", "год")]


def test_tokenize_line_matches_tokenize():
    reading = make_reading("Пино Нуар 2023")
    assert tokenize_line(reading.lines[0], reading=reading.key) == tokenize(reading)


def test_non_word_skeleton_is_norm():
    tokens = tokenize(make_reading("XXIV 13,5% 30/70 2023"))
    assert [(t.kind, t.skeleton) for t in tokens] == [
        ("roman", "xxiv"),
        ("abv", "13.5%"),
        ("ratio", "30/70"),
        ("year", "2023"),
    ]
