"""Одно слово — одно поле (Э6, гигиена разбора, круг 2): списки закрыты PREREG.

Строки — с полевых кадров v2 (M255, M285, M387, L22, R014), ooc_v2 и студии. Пункты меняют только
списки винодельни, кюве и сорта в полях этикетки: цвет, сахар и серия остаются прежними.
"""

from __future__ import annotations

import pytest
from test_color_sugar_forms import color_sugar, fields_of, hit

from app.reading.contracts import Color, SugarClass
from app.reading.fields import LONGER_CLAIM_FIELDS, NESTED_LEX_FIELDS
from app.reading.taxonomy import (
    LABEL_GENERIC,
    NAME_WINE_WORDS,
    REGION_WORDS_EXTRA,
    STYLE_WORDS,
)


def values(evidence) -> list[str]:
    return [e.value for e in evidence]


# ------------------------------------------------------------------ Э6-Г: сорт внутри сорта
def test_words_of_a_longer_grape_are_not_grapes_of_their_own():
    """«совиньон» из «Каберне Совиньон» resolve прочёл бы как Совиньон Блан."""
    fields = fields_of(
        "КАБЕРНЕ СОВИНЬОН",
        hits=[
            ([0, 1], hit("cabernet_sauvignon", "grape")),
            ([0], hit("каберне", "grape")),
            ([1], hit("совиньон", "grape")),
        ],
    )
    assert values(fields.grapes) == ["cabernet_sauvignon"]


def test_nested_hit_of_the_same_grape_is_dropped_too():
    fields = fields_of(
        "MUSCAT OTTONEL",
        hits=[([0, 1], hit("muscat_ottonel", "grape")), ([0], hit("muscat", "grape"))],
    )
    assert values(fields.grapes) == ["muscat_ottonel"]


def test_separate_and_overlapping_grapes_stay():
    fields = fields_of(
        "ГОЛУБОК",
        "КАБЕРНЕ ФРАН",
        hits=[([0], hit("golubok", "grape")), ([1], hit("каберне", "grape"))],
    )
    assert values(fields.grapes) == ["golubok", "каберне"]


def test_grape_inside_a_longer_hit_of_another_field_stays():
    """Правило — внутри поля сорта: кюве длиннее сорт не отнимает."""
    fields = fields_of(
        "Пино Нуар Джавага",
        hits=[([0, 1], hit("pinot_noir", "grape")), ([0, 1, 2], hit("пино нуар джавага", "cuvee"))],
    )
    assert values(fields.grapes) == ["pinot_noir"]


# ------------------------------------------------------------------ Э6-В: винодельня из чужих слов
@pytest.mark.parametrize(
    ("lines", "hits"),
    [
        (
            ("ПИНО НУАР",),
            [([0, 1], hit("pinot_noir", "grape")), ([0], hit("Шато Пино", "producer"))],
        ),
        (
            ("ЦИМЛЯНСКИЙ ЧЕРНЫЙ",),
            [
                ([0, 1], hit("tsimlyansky_cherny", "grape")),
                ([0], hit("Цимлянские вина", "producer", cost=1.0)),
            ],
        ),
        (
            ("Совиньон Блан",),
            [([0, 1], hit("sauvignon_blanc", "grape")), ([1], hit("Усадьба Маркотх", "producer"))],
        ),
        (("2009 Grand Reserve",), [([1], hit("Château Le Grand Vostock", "producer"))]),
    ],
)
def test_producer_made_of_words_of_a_longer_phrase_is_dropped(lines, hits):
    assert fields_of(*lines, hits=hits).producer == []


def test_producer_with_a_free_word_stays():
    fields = fields_of("CHATEAU", "PINOT", "Пино Нуар", hits=[([1], hit("Шато Пино", "producer"))])
    assert values(fields.producer) == ["Шато Пино"]


def test_producer_on_a_lone_color_word_stays_and_color_is_read():
    """Сравнивается только длина: «Блан» в одиночку — и цвет, и слово винодельни."""
    fields = fields_of("БЛАН", hits=[([0], hit("Усадьба Маркотх", "producer"))])
    assert values(fields.producer) == ["Усадьба Маркотх"]
    assert color_sugar(fields)[0] == Color.WHITE


def test_producer_rule_does_not_touch_color_and_sugar():
    fields = fields_of(
        "СОВИНЬОН БЛАН",
        "БЕЛОЕ СУХОЕ",
        hits=[([0, 1], hit("sauvignon_blanc", "grape")), ([1], hit("Усадьба Маркотх", "producer"))],
    )
    assert fields.producer == []
    assert color_sugar(fields) == (Color.WHITE, [SugarClass.DRY])


# ------------------------------------------------------------------ Э6-С отклонён замером
def test_style_word_stays_cuvee():
    """Пункт Э6-С (слово стиля не кюве) замер отклонил: без «мускатель» ломается R014."""
    fields = fields_of(
        "МУСКАТЕЛЬ",
        "БЕЛЬЙ",
        hits=[([0], hit("мускатель", "cuvee"))],
    )
    assert values(fields.cuvee) == ["мускатель"]
    assert "портвейн" not in NAME_WINE_WORDS and "мускатель" not in NAME_WINE_WORDS


# ------------------------------------------------------------------ Э6-Р: регион латиницей
@pytest.mark.parametrize(
    ("lines", "hits"),
    [
        (("THE TASTE OF CRIMEA",), [([3], hit("crimea", "cuvee"))]),
        (("ЗГУ KUBAN. ANAPA",), [([1], hit("kuban", "cuvee"))]),
        (("вина Крыма",), [([1], hit("крыма", "cuvee"))]),
        (("SEVASTOPOL",), [([0], hit("sevastopol", "cuvee"))]),
    ],
)
def test_latin_region_words_are_not_cuvee(lines, hits):
    assert fields_of(*lines, hits=hits).cuvee == []


# ------------------------------------------------------------------ Э6-Ш: категория и хозяйство
@pytest.mark.parametrize(
    ("lines", "hits"),
    [
        (("ШАТО",), [([0], hit("шато", "cuvee"))]),
        (("Виноград: Чернский Дворец",), [([0], hit("виноград", "cuvee"))]),
        (("ZB WINE SPUMANTE",), [([2], hit("spumante", "cuvee"))]),
        (("ANIMA MILLESIMATO",), [([1], hit("millesimato", "cuvee"))]),
    ],
)
def test_wine_words_are_not_cuvee(lines, hits):
    assert fields_of(*lines, hits=hits).cuvee == []


def test_cuvee_with_its_own_word_next_to_a_wine_word_stays():
    fields = fields_of("ШАТО КАПРИЗ", hits=[([0, 1], hit("шато каприз", "cuvee"))])
    assert values(fields.cuvee) == ["шато каприз"]


# ------------------------------------------------------------------ таблицы
def test_tables_are_closed_by_the_prereg():
    assert NESTED_LEX_FIELDS == frozenset({"grape"})
    assert LONGER_CLAIM_FIELDS == frozenset({"producer"})
    assert not NAME_WINE_WORDS & STYLE_WORDS
    assert {"шато", "chateau", "виноград", "spumante"} <= NAME_WINE_WORDS
    assert REGION_WORDS_EXTRA == frozenset(
        {"crimea", "krym", "kuban", "taman", "anapa", "sevastopol", "крыма"}
    )


def test_catalog_side_generic_words_are_unchanged():
    """Признаки названий каталога (`app.resolve.features`) берут `LABEL_GENERIC`, не таблицы Э6."""
    assert not REGION_WORDS_EXTRA & LABEL_GENERIC
    assert "шато" not in LABEL_GENERIC and "портвейн" not in LABEL_GENERIC
