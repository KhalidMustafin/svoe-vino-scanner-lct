import json

import pytest
from catalog import TOY_RECORDS, toy_attrs, wine_record

from app.reading.contracts import Color, SugarClass
from app.resolve.attrs import (
    CatalogAttrs,
    abv_key,
    color_key,
    colors_compatible,
    grape_key,
    name_words,
    norm_key,
    serial_key,
    sugar_key,
    wine_attrs,
    year_key,
)


@pytest.fixture
def attrs() -> CatalogAttrs:
    return toy_attrs()


# ------------------------------------------------------------------ ключи сравнения
def test_norm_key_matches_lexicon_form():
    # Ключ карточки и каноническая форма словаря этикетки — одна и та же норма.
    assert norm_key("А. Гордиенко & М. Николаев") == "а гордиенко м николаев"
    assert norm_key(None) == "" and norm_key("  ") == ""


@pytest.mark.parametrize(
    ("value", "code"),
    [("aligote", "aligote"), ("Шардоне", "chardonnay"), ("Pinot Noir", "pinot_noir")],
)
def test_grape_key_brings_spellings_to_one_code(value: str, code: str):
    assert grape_key(value) == code


def test_grape_key_keeps_grapes_outside_taxonomy():
    # Сорт вне таксономии остаётся собой: код каталога и есть его каноническая форма.
    assert grape_key("Выдуманный Сорт") == "выдуманный сорт"
    assert grape_key("") == "" and grape_key(None) == ""


@pytest.mark.parametrize(
    ("value", "key"),
    [
        ("Reserve", "резерв"),
        ("riserva", "резерв"),  # написание с этикетки сходится с ключевым словом каталога
        ("Grand Reserve", "гран резерв"),
        ("XXIV", "XXIV"),
        ("xxiv", "XXIV"),  # римский номер приводится к одному виду
        ("30/70", "30/70"),
        ("", ""),
    ],
)
def test_serial_key(value: str, key: str):
    assert serial_key(value) == key


def test_sugar_color_year_abv_keys():
    assert sugar_key("suhoe") is SugarClass.DRY and sugar_key("Brut") is SugarClass.BRUT
    assert sugar_key("") is None and sugar_key("шардоне") is None
    assert color_key("Красное") is Color.RED and color_key("rosé") is Color.ROSE
    assert year_key("2024") == 2024 and year_key(2024) == 2024
    assert year_key("вино") is None and year_key(1899) is None and year_key(None) is None
    assert abv_key("12,5") == 12.5 and abv_key("13%") == 13.0 and abv_key("нет") is None


def test_name_words_drops_generic_words():
    # «Вино» и «российское» не отличают позицию внутри винодельни.
    assert name_words("Вино Российское Гранат") == frozenset({"гранат"})


# ------------------------------------------------------------------ карточка
def test_wine_attrs_collects_keys():
    wine = wine_attrs(TOY_RECORDS[0])
    assert wine.slug == "dolina-aligote-2023"
    assert {"тестовая долина", "тестовая", "долина", "testovaya dolina"} <= wine.winery_keys
    assert wine.grapes == frozenset({"aligote"})
    assert wine.sugar is SugarClass.DRY and wine.year == 2023 and wine.abv == (12.5,)
    assert wine.cuvee == frozenset({"баррель"}) and wine.color is Color.WHITE
    assert wine.visual_group == "1/0" and wine.mates == ("dolina-aligote-2024",)


def test_colors_compatible_only_white_and_orange():
    assert colors_compatible("Белое", "Оранжевое") and colors_compatible(Color.ORANGE, "белое")
    assert not colors_compatible("Белое", "Белое")  # одинаковые — совпадение, а не пара
    assert not colors_compatible("Оранжевое", "Красное") and not colors_compatible("Белое", None)


def test_color_field_decides_over_category():
    """Правка карточки (Э4) обнуляет класс поля, а «Категория» выгрузки остаётся для показа."""
    fixed = wine_record("x", color="Белое")
    fixed["fields"]["color"] = {"class": None, "variants": []}
    assert wine_attrs(fixed).color is None
    legacy = wine_record("y", color="Красное")
    del legacy["fields"]["color"]  # записи без поля цвета — по «Категории», как раньше
    assert wine_attrs(legacy).color is Color.RED


def test_wine_attrs_strips_csv_prefix_of_grape_code():
    # «csv:арени» — сорт из каталога, а не из таксономии: префикс снят, написание сведено.
    record = wine_record("x", grapes=["csv:арени"], grape_values=["Арени"])
    assert wine_attrs(record).grapes == frozenset({"areni"})


def test_wine_attrs_serial_keeps_words_and_roman_numbers():
    wine = wine_attrs(TOY_RECORDS[2])
    assert wine.serial == frozenset({"резерв", "XXIV"})
    assert wine.text_keys >= {"резерв", "XXIV", "chardonnay"}


def test_series_is_cluster_or_slug_and_zero_cluster_survives():
    # cluster_B == 0 — настоящая группа: проверка «если cluster» потеряла бы её.
    zero = wine_attrs(wine_record("zero", cluster=0))
    assert zero.series == "cluster:0"
    assert wine_attrs(wine_record("solo")).series == "slug:solo"


def test_conditional_name_flag():
    assert wine_attrs(TOY_RECORDS[7]).conditional_name  # «Пино Нуар» — описание, не название
    assert not wine_attrs(TOY_RECORDS[0]).conditional_name


# ------------------------------------------------------------------ таблица
def test_members_of_series(attrs: CatalogAttrs):
    assert attrs.members("cluster:1") == ("dolina-aligote-2023", "dolina-aligote-2024")
    assert attrs.members("slug:dolina-merlot") == ("dolina-merlot",)
    assert attrs.members("cluster:99") == ()


def test_series_of_unknown_slug_is_its_own(attrs: CatalogAttrs):
    assert attrs.series_of("dolina-aligote-2023") == "cluster:1"
    assert attrs.series_of("нет-такого") == "slug:нет-такого"


def test_slugs_of_winery_by_any_spelling(attrs: CatalogAttrs):
    expected = (
        "dolina-aligote-2023",
        "dolina-aligote-2024",
        "dolina-reserve-brut",
        "dolina-polusladkoe",
        "dolina-merlot",
    )
    for spelling in ("Тестовая Долина", "тестовая долина", "долина", "testovaya dolina"):
        assert attrs.slugs_of_winery(spelling) == expected
    assert attrs.slugs_of_winery("Неизвестная") == ()


def test_duplicate_slug_is_an_error():
    with pytest.raises(ValueError, match="повтор slug"):
        CatalogAttrs.from_records([wine_record("x"), wine_record("x")])


def test_table_is_a_mapping(attrs: CatalogAttrs):
    assert len(attrs) == 8 and "dolina-merlot" in attrs and "нет" not in attrs
    assert attrs.get("нет") is None
    assert [wine.slug for wine in attrs][:2] == ["dolina-aligote-2023", "dolina-aligote-2024"]


def test_stats(attrs: CatalogAttrs):
    stats = attrs.stats()
    assert stats["slugs"] == 8 and stats["wineries"] == 3
    assert stats["series"] == 6 and stats["clustered_slugs"] == 4
    assert stats["with_year"] == 2 and stats["with_serial"] == 1
    assert stats["conditional_names"] == 2


def test_load_from_file(tmp_path):
    path = tmp_path / "gt_tokens.jsonl"
    lines = [json.dumps(record, ensure_ascii=False) for record in TOY_RECORDS]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")  # пустая строка пропускается
    attrs = CatalogAttrs.load(path)
    assert len(attrs) == 8 and attrs.meta["source"] == "gt_tokens.jsonl"


def test_broken_line_names_the_line(tmp_path):
    path = tmp_path / "gt_tokens.jsonl"
    path.write_text(json.dumps(TOY_RECORDS[0]) + "\n{нет\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: не JSON"):
        CatalogAttrs.load(path)
