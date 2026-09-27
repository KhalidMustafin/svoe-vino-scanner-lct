import csv
import itertools

import pytest

from app.config import get_settings
from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import (
    GRAPE_SYNONYMS,
    LABEL_TERMS,
    WINE_WORDS,
    PhraseIndex,
    canonical_grape,
    color_of,
    find_colors,
    find_grapes,
    find_serial,
    find_sugar,
    grape_label,
    sugar_class,
)
from app.reading.text.normalize import norm_token
from app.reading.text.translit import skeleton


@pytest.mark.parametrize("text", ["МУСКАТЕЛЬ белый", "Мускатель розовый", "Moscatel Oro"])
def test_muscatel_is_not_muscat(text):
    assert find_grapes(text) == []


@pytest.mark.parametrize(
    ("text", "codes"),
    [
        ("Мускат белый", ["muscat"]),
        ("МУСКАТ", ["muscat"]),
        ("Мускат Розовый", ["muscat_rose"]),
        ("Мускат Оттонель", ["muscat_ottonel"]),
        ("Каберне Фран", ["cabernet_franc"]),
        ("Cabernet Sauvignon", ["cabernet_sauvignon"]),
        ("Совиньон Блан", ["sauvignon_blanc"]),
        ("Траминер Розовый", ["traminer_rose"]),
        ("Красностоп Анапский", ["krasnostop_azos"]),
    ],
)
def test_longest_grape_name_wins(text, codes):
    assert find_grapes(text) == codes


def test_lone_cabernet_is_not_a_grape():
    assert find_grapes("Каберне") == []


def test_grapes_in_order_without_duplicates():
    assert find_grapes("Пино чёрный (Пино нуар) и Мерло") == ["pinot_noir", "merlot"]
    assert find_grapes("Сира (Шираз)") == ["syrah"]


def test_grape_needs_whole_words():
    assert find_grapes("Мерлотта") == []
    assert find_grapes("мерло,шардоне") == ["merlot", "chardonnay"]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("ШAPДOHE", "chardonnay"),  # латинские A, P, O, H, E
        ("Shardone", "chardonnay"),
        ("Pino Nuar", "pinot_noir"),
        ("Мюллер-Тургау", "muller_thurgau"),
        ("Gewürztraminer", "gewurztraminer"),
        ("САПЕРАВИ", "saperavi"),
        ("Sibirkoviy", "sibirkovy"),
    ],
)
def test_homoglyphs_and_transliteration(text, code):
    assert find_grapes(text) == [code]


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("Рислинг Рейнский", "riesling"),
        ("pinot_noir", "pinot_noir"),
        ("Бьянка", "bianca"),
        ("Пино Гриджио", "pinot_gris"),
        ("Цвайгельт Таманский", "zweigelt"),
        ("Олег", None),
        ("Белые сорта винограда", None),
    ],
)
def test_canonical_grape(name, code):
    assert canonical_grape(name) == code


def test_every_synonym_resolves_to_its_code():
    for code, variants in GRAPE_SYNONYMS.items():
        for variant in variants:
            assert canonical_grape(variant) == code, variant


def test_grape_label():
    assert grape_label("pinot_noir") == "Пино Нуар"
    assert grape_label("no_such_code") == "no such code"


# Значения колонки «Сорт винограда», которых не было в словаре «Лозы».
@pytest.mark.parametrize(
    "value",
    [
        "Менье",
        "Рубин Голодриги",
        "Блауфранкиш",
        "Совиньон Зеленый",
        "Мцване Кахетинский",
        "Верментино",
        "Глера",
        "Мускат Гамбургский",
        "Уньи блан",
        "Пино черный",
        "Кумшацкий белый",
        "Саперави Северный",
        "Рубин АЗОС",
        "Москато-джалло",
        "Педро Хименес",
        "Пти Мансен",
        "Гечеи Заматош",
        "Петит арвин",
        "Рисланер",
        "Эким-Кара",
    ],
)
def test_catalog_grape_values_resolve(value):
    assert canonical_grape(value) is not None


def test_catalog_csv_grape_column_resolves():
    path = get_settings().dataset_dir / "strapi_output0709.csv"
    if not path.exists():
        pytest.skip("CSV каталога не распакован")
    generic = {"Белые сорта винограда", "Красные сорта винограда", "Олег"}
    with path.open(encoding="utf-8-sig") as fh:
        values = {
            value.strip()
            for row in csv.DictReader(fh)
            for value in row["Сорт винограда"].split(",")
            if value.strip()
        }
    assert sorted(v for v in values - generic if canonical_grape(v) is None) == []


@pytest.mark.parametrize(
    ("text", "classes"),
    [
        ("Экстра брют", [SugarClass.EXTRA_BRUT]),
        ("Брют натюр", [SugarClass.BRUT_NATURE]),
        ("Полусухое", [SugarClass.SEMI_DRY]),
        ("полу-сладкое", [SugarClass.SEMI_SWEET]),
        ("DEMI-SEC", [SugarClass.SEMI_SWEET]),
        ("Brut Zero Dosage", [SugarClass.BRUT_NATURE]),
        ("Extra Dry", []),
        ("Брют, полусухое", [SugarClass.BRUT, SugarClass.SEMI_DRY]),
        ("Блан Сек", [SugarClass.DRY]),
        ("Кокур десертный", [SugarClass.SWEET]),
    ],
)
def test_find_sugar(text, classes):
    assert find_sugar(text) == classes


@pytest.mark.parametrize(
    ("text", "colors"),
    [
        ("Sauvignon Blanc", []),
        ("Blanc de Blancs", []),
        ("PROSECCO", []),
        ("Rosé Brut", [Color.ROSE]),
        ("Руж Сек", [Color.RED]),
        ("вино белое сухое", [Color.WHITE]),
    ],
)
def test_find_colors(text, colors):
    assert find_colors(text) == colors


def test_find_serial_longest_form():
    assert find_serial("Grand Reserve") == ["гран резерв"]
    assert find_serial("Блан де Нуар") == ["блан де нуар"]


def test_sugar_class_and_color_lookup():
    assert sugar_class("extra_brut") is SugarClass.EXTRA_BRUT
    assert sugar_class("Экстра брют") is SugarClass.EXTRA_BRUT
    assert sugar_class("extra dry") is None
    assert sugar_class("Каберне") is None
    assert color_of("Белое") is Color.WHITE
    assert color_of("rosé") is Color.ROSE
    assert color_of("Шардоне") is None


def test_index_rejects_phrase_with_two_values():
    with pytest.raises(ValueError):
        PhraseIndex([("брют", 1), ("БРЮТ", 2)])
    assert PhraseIndex([("брют", 1), ("БРЮТ", 1)]).lookup("Брют") == 1


def test_index_picks_longest_non_overlapping():
    index = PhraseIndex([("a b", "ab"), ("b c d", "bcd"), ("d", "d"), ("a", "a")])
    found = index.find(["a", "b", "c", "d"])
    assert [(m.start, m.end, m.value) for m in found] == [(0, 1, "a"), (1, 4, "bcd")]


def test_no_skeleton_collisions_between_values():
    for a, b in itertools.combinations(LABEL_TERMS.entries, 2):
        if a.value == b.value or len(a.words) != len(b.words):
            continue
        same = all(
            x == y or (sx is not None and sx == sy)
            for x, y, sx, sy in zip(a.words, b.words, a.skeletons, b.skeletons, strict=True)
        )
        assert not same, (a.words, b.words)


def test_only_grapes_match_by_skeleton():
    assert find_sugar("Новый Свет") == []  # «свет» и «sweet» — один скелет
    assert find_sugar("POLUSUHOE") == [SugarClass.SEMI_DRY]  # транслит — явным списком
    assert find_grapes("Pino Nuar") == ["pinot_noir"]


def test_index_fuzzy_predicate():
    index = PhraseIndex([("шардоне", "grape"), ("sweet", "sugar")], fuzzy=lambda v: v == "grape")
    norms, skeletons = ["shardone", "свет"], [skeleton("shardone"), skeleton("свет")]
    assert [m.value for m in index.find(norms, skeletons)] == ["grape"]


def test_wine_words_are_normalized():
    assert "вино" in WINE_WORDS
    assert all(word == norm_token(word) for word in WINE_WORDS)
