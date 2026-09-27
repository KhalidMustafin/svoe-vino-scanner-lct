"""Строка урожая, объёма и крепости — не начало имени (Э7): списки закрыты PREREG.

Прилагательное цвета или сахара не среднего рода перед словом не из словаря этикетки —
начало имени («Красная Горка»). Строка урожая («ГОД УРОЖАЯ 2023», «Harvest 2021») и строка
объёма и крепости («Алк. 12% об.», «alc. 13% vol») именем не бывают, а на этикетке часто стоят
сразу под цветом. Строки составлены из общих слов этикетки; раскладка «… / БЕЛЫЙ / ГОД УРОЖАЯ /
2023» — как у WebP-оригинала R014 (находка GPU-окна A, `research/2026-09-26_e7/PREREG.md`).
"""

from __future__ import annotations

import pytest
from test_color_sugar_forms import color_sugar, fields_of

from app.reading import fields as fields_module
from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import (
    LABEL_GENERIC,
    NAME_WINE_WORDS,
    VINTAGE_LINE_WORDS,
    VOLUME_LINE_WORDS,
)
from app.reading.text.normalize import norm_token

WHITE, RED, ROSE = Color.WHITE, Color.RED, Color.ROSE
DRY, SWEET = SugarClass.DRY, SugarClass.SWEET


# ------------------------------------------------------------------ Э7-У: строка урожая
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("ИНКЕРМАН", "БЕЛЫЙ", "ГОД УРОЖАЯ", "2021"), (WHITE, [])),
        (("ЗОЛОТАЯ БАЛКА", "КРАСНЫЙ", "ГОД УРОЖАЯ 2020"), (RED, [])),
        (("ВИНО РОССИИ", "СУХАЯ", "Год урожая: 2019"), (None, [DRY])),
        (("РОЗОВЫЙ", "Годы урожая 2018–2020"), (ROSE, [])),
        (("БЕЛЫЙ", "Harvest 2021"), (WHITE, [])),
        (("КРАСНЫЙ", "Vendemmia 2019"), (RED, [])),
        (("ДЕСЕРТНЫЙ", "Выдержка 3 года"), (None, [SWEET])),
        (("БЕЛАЯ", "ГОДА ВЫПУСКА 2022"), (WHITE, [])),
        (("КРАСНЫЙ", "ЛЕТ ВЫДЕРЖКИ 5"), (RED, [])),
    ],
)
def test_vintage_line_after_an_adjective_is_not_a_name(lines, expected):
    assert color_sugar(fields_of(*lines)) == expected


# ------------------------------------------------------------------ Э7-О: строка объёма и крепости
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("РОЗОВЫЙ", "Алк. 12% об."), (ROSE, [])),
        (("БЕЛЫЙ", "alc. 13% vol"), (WHITE, [])),
        (("КРАСНАЯ", "Алкоголь 13,5%"), (RED, [])),
        (("СУХОЙ", "Спирт этиловый 12%"), (None, [DRY])),
        (("БЕЛЫЙ", "Объёма 0,75 л"), (WHITE, [])),
        (("СЛАДКИЙ", "мл 750"), (None, [SWEET])),
    ],
)
def test_volume_or_strength_line_after_an_adjective_is_not_a_name(lines, expected):
    assert color_sugar(fields_of(*lines)) == expected


# ------------------------------------------------------------------ что осталось как было
@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("КРАСНЫЙ", "2019"), (RED, [])),  # год — число, а не слово
        (("БЕЛЫЙ", "0,75 л"), (WHITE, [])),
        (("СУХОЙ", "12% об."), (None, [DRY])),
        (("БЕЛЫЙ", "УРОЖАЙ 2020"), (WHITE, [])),  # «урожай» и раньше был в словаре этикетки
    ],
)
def test_numbers_and_known_label_words_already_kept_the_adjective(lines, expected):
    assert color_sugar(fields_of(*lines)) == expected


@pytest.mark.parametrize(
    "lines",
    [
        ("КРАСНАЯ", "ГОРКА", "ГОД УРОЖАЯ 2020"),
        ("БЕЛАЯ ЛЬВИЦА", "Алк. 12% об."),
        ("СУХОЙ", "ЛИМАН", "Harvest 2021"),
    ],
)
def test_a_name_before_the_vintage_line_is_still_a_name(lines):
    """Правило смотрит на слово сразу после прилагательного: имя перед строкой урожая — имя."""
    assert color_sugar(fields_of(*lines)) == (None, [])


def test_tables_are_the_closed_prereg_lists():
    vintage = (
        "год года году годы лет урожай урожая урожаи выдержка выдержки выдержкой выпуска "
        "vintage millesime millesimato harvest vendemmia annata cosecha jahrgang"
    )
    volume = (
        "объем объема литр литра литров л мл ml cl alc alcohol алк алкоголь крепость спирта "
        "спирт этилового abv vol об"
    )
    assert VINTAGE_LINE_WORDS == {norm_token(w) for w in vintage.split()}
    assert VOLUME_LINE_WORDS == {norm_token(w) for w in volume.split()}
    assert norm_token("Millésime") in VINTAGE_LINE_WORDS


def test_tables_reach_only_the_proper_name_rule():
    """Кюве и винодельня таблиц Э7 не видят: «год» и «л» не стали общими словами этикетки."""
    words = VINTAGE_LINE_WORDS | VOLUME_LINE_WORDS
    assert words <= fields_module._WINE_VOCABULARY
    new = words - LABEL_GENERIC - NAME_WINE_WORDS
    assert new and new.isdisjoint(fields_module._NOT_CUVEE)
