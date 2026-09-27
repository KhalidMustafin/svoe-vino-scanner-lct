import pytest

from app.reading.text.normalize import (
    fold_homoglyphs,
    norm,
    norm_token,
    roman_value,
    strip_diacritics,
)


def test_norm_lowercases_and_turns_punctuation_into_spaces():
    assert norm("  Абрау-Дюрсо   Брют! ") == "абрау дюрсо брют"


def test_norm_replaces_yo():
    assert norm("Ёлочка") == "елочка"


def test_norm_strips_latin_diacritics():
    assert norm("Château Satèn Gewürztraminer") == "chateau saten gewurztraminer"


def test_norm_keeps_cyrillic_short_i():
    assert norm("Новый Светлый") == "новый светлый"


def test_norm_drops_final_hard_sign_only():
    # «Лоза», слабое место №5: «ведерниковъ» не находился.
    assert norm("Ведерниковъ") == "ведерников"
    assert norm("Новый Светъ.") == "новый свет"
    assert norm("объём") == "объем"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("13,5% об.", "13,5% об"),
        ("0.75 л", "0.75 л"),
        ("30/70", "30/70"),
        ("30 / 70", "30/70"),
        ("Вино, 2023.", "вино 2023"),
        ("100 %", "100 %"),
    ],
)
def test_norm_keeps_separators_only_inside_numbers(raw, expected):
    assert norm(raw) == expected


def test_norm_of_empty_text():
    assert norm("") == ""


def test_strip_diacritics_keeps_letters_with_marks_in_cyrillic():
    assert strip_diacritics("йЁ ß Øre") == "йЁ ss Ore"


def test_fold_homoglyphs_capital_ocr_mix_to_cyrillic():
    assert fold_homoglyphs("MACCAHДPA") == "МАССАНДРА"


def test_fold_homoglyphs_after_lowercase_folds_m_and_h():
    # «Лоза», слабое место №4: m и h не сводились — «mассаhдра».
    assert norm_token("MACCAHДPA") == "массандра"
    assert fold_homoglyphs("mассаhдра") == "массандра"


def test_fold_homoglyphs_sees_latin_after_diacritics_are_gone():
    # «Лоза», слабое место №6: «с» и «е», «а» кириллические, «â» с диакритикой.
    assert norm_token("сhâtеаu") == "chateau"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("cовиньон", "совиньон"),  # латинская «c» в кириллице
        ("biаncо", "bianco"),  # кириллические «а», «о» в латинице
        ("BIАNCО", "BIANCO"),
        ("pinоt", "pinot"),
    ],
)
def test_fold_homoglyphs_follows_the_dominant_script(token, expected):
    assert fold_homoglyphs(token) == expected


@pytest.mark.parametrize("token", ["XXIV", "III", "xxiv", "Chardonnay", "шардоне", "k2", "v2r"])
def test_fold_homoglyphs_leaves_single_script_tokens(token):
    assert fold_homoglyphs(token) == token


def test_fold_homoglyphs_roman_with_cyrillic_lookalikes_goes_latin():
    assert fold_homoglyphs("ХХIV") == "XXIV"
    assert fold_homoglyphs("ХI") == "XI"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("М0СКВА", "МОСКВА"),
        ("3ОЛОТАЯ", "ЗОЛОТАЯ"),
        ("6ельбек", "бельбек"),
        ("r0se", "rose"),
        ("2О23", "2023"),
        ("2o19", "2019"),
    ],
)
def test_fold_homoglyphs_digit_lookalikes_in_context(token, expected):
    assert fold_homoglyphs(token) == expected


@pytest.mark.parametrize("token", ["750мл", "13,5", "2023г", "0,75л", "3д"])
def test_fold_homoglyphs_keeps_real_numbers(token):
    assert fold_homoglyphs(token) == token


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("XXIV", 24),
        ("iii", 3),
        ("XL", 40),
        ("MMXXII", 2022),
        ("MCMXCIX", 1999),
        ("IIII", None),
        ("", None),
        ("IL", None),
    ],
)
def test_roman_value(token, value):
    assert roman_value(token) == value
