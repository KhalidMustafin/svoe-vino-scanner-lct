"""Полевые написания разбора (Э1, пункты R1 плана точности 25.09): списки закрыты PREREG.

Строки взяты с полевых кадров (`R014`, `M100`, `M162`, `M297`, `M526`, `L09`) и из самих
списков. Написания видит только разбор чтения (`READ_TERMS`): названия каталога, из которых
считаются признаки выбора, разбираются `LABEL_TERMS`, и там этих фраз нет.
"""

from __future__ import annotations

import pytest
from test_color_sugar_forms import color_sugar, fields_of, hit

from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import LABEL_TERMS, READ_TERMS, Term, find_colors, find_sugar

WHITE, RED = Color("Белое"), Color("Красное")
SEMI_SWEET, SWEET = SugarClass("polusladkoe"), SugarClass("sladkoe")
EXTRA_BRUT, BRUT_NATURE = SugarClass("extra_brut"), SugarClass("brut_nature")


# ------------------------------------------------------------------ (а) опечатки цвета
@pytest.mark.parametrize(
    "lines",
    [
        ("ВИНО РОССИИ", "МУСКАТЕЛЬ", "МАССАНДАРА", "БЕЛЬЙ", "ГОД УРОЖАЯ", "2023"),
        ("КУБАНЬ", "СУКАЕ ВЕЛОЕ"),
        ("ПОРТВЕЙН", "бельи"),
        ("Вино белоe",),  # латинская «e»: сводится к «белое» и без списка
    ],
)
def test_ocr_typos_of_white_read_as_white(lines):
    assert color_sugar(fields_of(*lines))[0] == WHITE


def test_typo_of_white_next_to_another_color_abstains():
    """Два цвета в одном чтении — отказ, как у любых равносильных значений."""
    assert color_sugar(fields_of("Красное", "БЕЛЬЙ"))[0] is None


# ------------------------------------------------------------------ (б) semi-dolce, demi-doux
@pytest.mark.parametrize(
    "lines",
    [("Vino Rosso semi-dolce",), ("vin blanc", "MUSCAT BEAU", "demi-doux"), ("ПОЛУ СЛАДКОЕ",)],
)
def test_semi_dolce_and_demi_doux_are_semi_sweet(lines):
    assert color_sugar(fields_of(*lines))[1] == [SEMI_SWEET]


@pytest.mark.parametrize("line", ["Dolce", "Doux", "Moscato dolce"])
def test_dolce_and_doux_alone_stay_sweet(line):
    assert color_sugar(fields_of(line))[1] == [SWEET]


# ------------------------------------------------------------------ (в) extra brut zero dosage
@pytest.mark.parametrize("line", ["EXTRA BRUT • ZERO DOSAGE", "Extra Brut Zero Dosage"])
def test_extra_brut_zero_dosage_gives_both_classes(line):
    assert color_sugar(fields_of("CUVÉE ALEXANDRE", line))[1] == [EXTRA_BRUT, BRUT_NATURE]


@pytest.mark.parametrize(
    ("line", "sugar"),
    [
        ("Brut Zero Dosage", [BRUT_NATURE]),
        ("Zero Dosage", [BRUT_NATURE]),
        ("Extra Brut", [EXTRA_BRUT]),
    ],
)
def test_other_zero_dosage_phrases_are_unchanged(line, sugar):
    assert color_sugar(fields_of(line))[1] == sugar


# ------------------------------------------------------------------ (г) имена с цветом
@pytest.mark.parametrize(
    "lines",
    [
        ("Royal Red Cat", "корейский рыжий кот"),
        ("DENISOV", "САМАРА", "КРАСНАЯ", "СТРЕЛКА", "Рубин"),
        ("ГАЛИЦКИЙ • ГАЛИЦКИЙ", "КРАСНАЯ ГОРКА", "2019"),
        ("КРАСНАЯ ПОЛЯНА",),
        ("Кубань, Красная",),
    ],
)
def test_names_with_a_color_word_give_no_color(lines):
    assert color_sugar(fields_of(*lines))[0] is None


def test_color_next_to_those_names_is_still_read():
    assert color_sugar(fields_of("Red", "Cabernet"))[0] == RED
    assert color_sugar(fields_of("Кубань", "Красное сухое"))[0] == RED


def test_blocker_leaves_the_cuvee_its_words():
    """Имя «Красная стрелка» из словаря остаётся кюве: заглушка цвета слов не отнимает."""
    fields = fields_of("Красная стрелка", hits=[([0, 1], hit("красная стрелка", "cuvee"))])
    assert [e.value for e in fields.cuvee] == ["красная стрелка"]
    assert fields.color is None


# ------------------------------------------------------------------ (д) слова региона
@pytest.mark.parametrize(
    ("lines", "hits"),
    [
        (("КОКУР", "КРЫМ"), [([1], hit("крым", "cuvee"))]),
        (("ДОЛИНА ДОНА",), [([0, 1], hit("долина дона", "cuvee"))]),
        (("Кубани",), [([0], hit("кубани", "cuvee"))]),
        (("Крьм",), [([0], hit("крым", "cuvee", cost=0.5))]),  # опечатка слова региона
    ],
)
def test_region_words_are_not_cuvee(lines, hits):
    assert fields_of(*lines, hits=hits).cuvee == []


def test_cuvee_with_its_own_word_next_to_a_region_stays():
    fields = fields_of("АЛУШТА КРЫМ", hits=[([0, 1], hit("алушта крым", "cuvee"))])
    assert [e.value for e in fields.cuvee] == ["алушта крым"]


# ------------------------------------------------------------------ каталог не меняется
@pytest.mark.parametrize(
    ("text", "term"),
    [
        ("бельй", Term("color", WHITE)),
        ("demi-doux", Term("sugar", SEMI_SWEET)),
        ("extra brut zero dosage", Term("sugar", EXTRA_BRUT, (BRUT_NATURE,))),
        ("red cat", Term("color", None)),
        ("кубань красная", Term("color", None)),
    ],
)
def test_field_spellings_live_only_in_the_reading_index(text, term):
    assert READ_TERMS.lookup(text) == term
    assert LABEL_TERMS.lookup(text) is None


def test_catalog_side_helpers_are_unchanged():
    """Разбор названий каталога и атрибутов (`find_*`) видит прежние фразы."""
    assert find_colors("Royal Red Cat") == [RED]
    assert find_sugar("Extra Brut Zero Dosage") == [BRUT_NATURE]
    assert find_sugar("semi-dolce") == [SWEET]
