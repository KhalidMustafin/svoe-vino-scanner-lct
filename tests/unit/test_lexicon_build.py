import json
import math

import pytest

from app.config import Settings
from app.reading.contracts import Reading, TextLine
from app.reading.lexicon.build import (
    Lexicon,
    build_from_gt,
    build_from_records,
    default_gt_tokens_path,
    default_lexicon_path,
    idf,
)
from app.reading.text.tokenize import tokenize
from app.reading.text.translit import skeleton


def record(
    slug,
    winery,
    *,
    key=(),
    wvariants=(),
    cuvee=(),
    cvariants=(),
    grapes=(),
    gvariants=(),
    sugar=None,
    svariants=(),
    serial=(),
    keywords=(),
    color=None,
    colvariants=(),
):
    """Запись в формате `gt_tokens.jsonl` — только поля, которые читает словарь."""
    return {
        "slug": slug,
        "winery": winery,
        "category": color,
        "fields": {
            "winery": {"key_tokens": list(key), "variants": list(wvariants), "brands": []},
            "cuvee": {"tokens": list(cuvee), "variants": list(cvariants)},
            "grape": {
                "values": [value for value, _ in grapes],
                "codes": [code for _, code in grapes],
                "variants": list(gvariants),
            },
            "sugar": {"class": sugar, "variants": list(svariants)},
            "serial": {"tokens": list(serial), "keywords": list(keywords)},
            "color": {"class": color, "variants": list(colvariants)},
        },
    }


CATALOG = [
    record(
        "abrau-blanc-de-blancs",
        "Абрау-Дюрсо",
        key=["абрау", "дюрсо"],
        wvariants=["abrau durso", "abrau"],
        grapes=[("Шардоне", "chardonnay")],
        gvariants=["shardone", "шардоне"],
        sugar="brut",
        svariants=["брют", "brut"],
        keywords=["blanc de blancs"],
        color="Белое",
    ),
    record(
        "abrau-reserve-extra-brut",
        "Абрау-Дюрсо",
        key=["абрау", "дюрсо"],
        wvariants=["abrau durso", "abrau"],
        grapes=[("Пино Нуар", "pinot_noir")],
        gvariants=["pinot noir", "pino nuar", "пино нуар"],
        sugar="extra_brut",
        keywords=["резерв"],
        color="Белое",
    ),
    record(
        "massandra-muskat",
        "Массандра",
        key=["массандра"],
        wvariants=["massandra"],
        grapes=[("Мускат белый", "muscat")],
        gvariants=["muscat", "мускат", "мускат белый"],
        sugar="sladkoe",
        color="Белое",
    ),
    record(
        "kuban-vino-aristov-xxiv",
        "Кубань-Вино",
        key=["кубань", "вино"],
        wvariants=["aristov", "аристов", "шато тамань", "kuban vino"],
        cuvee=["аристов"],
        cvariants=["aristov"],
        serial=["XXIV"],
        sugar="brut",
        color="Белое",
    ),
    record(
        "fanagoria-saperavi",
        "Фанагория",
        key=["фанагория"],
        wvariants=["fanagoria"],
        grapes=[("Саперави", "saperavi")],
        gvariants=["saperavi", "саперави"],
        sugar="suhoe",
        svariants=["сухое", "dry", "sec"],
        color="Красное",
    ),
    record(
        "gai-kodzor-vexillum",
        "Виноградники Гай-Кодзора",
        key=["гай", "кодзора"],
        wvariants=["gai kodzor"],
        cuvee=["vexillum"],
        cvariants=["вексиллум"],
        grapes=[("Ркацители", "rkatsiteli")],
        gvariants=["rkatsiteli", "ркацители"],
        sugar="polusuhoe",
        svariants=["полусухое", "semi dry", "demi sec"],  # спорный термин каталога
        color="Белое",
    ),
    record(
        "tabiya-pino-nuar-rose",
        "Табия",
        key=["табия"],
        wvariants=["tabiya"],
        grapes=[("Пино Нуар", "pinot_noir")],
        gvariants=["pinot noir", "пино нуар"],
        sugar="polusuhoe",
        color="Розовое",
    ),
    record(
        "shato-pino-blend",
        "Шато Пино",
        key=["шато", "пино"],
        wvariants=["shato pino"],
        grapes=[("Каберне Совиньон", "cabernet_sauvignon"), ("Мерло", "merlot")],
        gvariants=["cabernet sauvignon", "kaberne sovinon", "каберне совиньон", "merlot", "мерло"],
        sugar="polusladkoe",
        svariants=["полусладкое", "demi sec"],
        serial=["30/70"],
        color="Красное",
    ),
    record(
        "alma-pino-blan",
        "Alma Valley",
        key=["alma", "valley"],
        grapes=[("Пино Блан", "pinot_blanc")],
        gvariants=["pinot blanc", "пино блан"],
        sugar="suhoe",
        color="Белое",
    ),
]


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    return build_from_records(CATALOG)


def test_producer_by_name_and_latin_variant(lex):
    by_name = lex.get("producer", "Массандра")
    by_latin = lex.get("producer", "massandra")
    assert by_name.canonical == by_latin.canonical == "Массандра"
    assert by_name.slugs == by_latin.slugs == {"massandra-muskat"}
    assert by_name.skeleton == skeleton("массандра")
    assert lex.get("producer", "Абрау-Дюрсо").slugs == {
        "abrau-blanc-de-blancs",
        "abrau-reserve-extra-brut",
    }


def test_phrase_is_indexed_whole_and_by_words(lex):
    assert lex.get("grape", "каберне совиньон").canonical == "cabernet_sauvignon"
    assert lex.get("grape", "совиньон").canonical == "cabernet_sauvignon"
    assert lex.get("producer", "шато тамань").canonical == "Кубань-Вино"
    assert lex.get("producer", "тамань").slugs == {"kuban-vino-aristov-xxiv"}
    assert lex.max_words >= 3


def test_phrase_words_are_parts_unless_added_on_their_own(lex):
    # Слово только из фраз — кандидат, а не попадание; форма сама по себе — обычная запись.
    assert lex.get("producer", "тамань").part and lex.get("grape", "совиньон").part
    assert not lex.get("producer", "шато тамань").part
    assert not lex.get("grape", "каберне совиньон").part
    assert not lex.get("producer", "абрау").part  # ключевое слово винодельни
    assert not lex.get("grape", "мускат").part  # вариант сорта одним словом
    assert lex.stats()["phrase_parts"]["grape"] >= 1


def test_short_and_generic_words_are_not_entries(lex):
    assert lex.find("de") == []
    assert lex.get("producer", "вино") is None
    assert lex.get("producer", "кубань") is None  # регион, а не винодельня
    assert lex.get("producer", "Кубань-Вино").canonical == "Кубань-Вино"
    assert lex.get("grape", "белый") is None
    assert lex.get("grape", "мускат белый").canonical == "muscat"


def test_variant_is_attached_to_its_grape_in_a_blend(lex):
    assert lex.get("grape", "merlot").canonical == "merlot"
    assert lex.get("grape", "kaberne sovinon").canonical == "cabernet_sauvignon"


def test_word_shared_by_entities_keeps_norm_as_canonical(lex):
    pino = lex.get("grape", "пино")
    assert pino.canonical == "пино"
    assert pino.slugs == {"abrau-reserve-extra-brut", "tabiya-pino-nuar-rose", "alma-pino-blan"}
    assert lex.get("grape", "нуар").canonical == "pinot_noir"


def test_builtin_sugar_term_overrides_catalog_conflict(lex):
    demi_sec = lex.get("sugar", "Demi-Sec")
    assert demi_sec.canonical == "polusladkoe"
    assert demi_sec.slugs == {"shato-pino-blend"}


def test_sugar_terms_exist_without_catalog_positions(lex):
    zero = lex.get("sugar", "zero dosage")
    assert zero.canonical == "brut_nature"
    assert zero.slugs == frozenset()
    assert lex.get("sugar", "pas dosé").canonical == "brut_nature"
    assert lex.get("sugar", "экстра брют").slugs == {"abrau-reserve-extra-brut"}


def test_sugar_phrase_words_are_not_indexed(lex):
    sec = lex.get("sugar", "sec")
    assert sec.canonical == "suhoe"
    assert sec.slugs == {"fanagoria-saperavi", "alma-pino-blan"}
    assert lex.get("sugar", "demi") is None


def test_serial_spellings_share_slugs(lex):
    latin = lex.get("serial", "Blanc de Blancs")
    cyrillic = lex.get("serial", "Блан де Блан")
    assert latin.canonical == cyrillic.canonical == "блан де блан"
    assert latin.slugs == cyrillic.slugs == {"abrau-blanc-de-blancs"}
    assert lex.get("serial", "reserve").slugs == {"abrau-reserve-extra-brut"}


def test_roman_and_ratio_serials(lex):
    roman = lex.get("serial", "XXIV")
    assert (roman.norm, roman.canonical, roman.skeleton) == ("xxiv", "XXIV", "xxiv")
    ratio = lex.get("serial", "30/70")
    assert (ratio.norm, ratio.skeleton, ratio.slugs) == ("30/70", "30/70", {"shato-pino-blend"})


def test_color_terms_by_category(lex):
    rose = lex.get("color", "Rosé")
    assert rose.canonical == "Розовое"
    assert rose.slugs == {"tabiya-pino-nuar-rose"}


def test_idf_is_higher_for_rare_entries(lex):
    common = lex.get("color", "белое")
    rare = lex.get("sugar", "сладкое")
    assert len(common.slugs) > len(rare.slugs)
    assert rare.idf > common.idf
    assert lex.get("sugar", "zero dosage").idf == idf(0, len(CATALOG))
    assert idf(0, 9) == round(math.log(10) + 1, 4)
    assert lex.n_slugs == len(CATALOG)


def test_catalog_only_mode_skips_builtin_terms():
    lex = build_from_records(CATALOG, builtin_terms=False)
    assert lex.get("sugar", "zero dosage") is None
    assert lex.get("sugar", "demi sec") is None  # в каталоге у двух классов — спорный
    assert lex.get("serial", "blanc de blancs").canonical == "blanc de blancs"
    assert lex.get("serial", "блан де блан") is None


def test_save_and_load_roundtrip(lex, tmp_path):
    path = lex.save(tmp_path / "index" / "lexicon.json")
    loaded = Lexicon.load(path)
    assert loaded.entries == lex.entries
    assert any(entry.part for entry in loaded.entries)
    assert loaded.n_slugs == lex.n_slugs
    assert loaded.max_words == lex.max_words
    assert "массандра" in loaded
    assert skeleton("massandra") in loaded


def test_load_rejects_other_version(lex, tmp_path):
    data = lex.to_json()
    data["version"] = 99
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        Lexicon.load(path)


def test_build_from_gt_jsonl(tmp_path):
    path = tmp_path / "gt_tokens.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in CATALOG), encoding="utf-8"
    )
    lex = build_from_gt(path)
    assert lex.entries == build_from_records(CATALOG).entries
    assert lex.meta["source"] == "gt_tokens.jsonl"
    assert len(lex.meta["source_sha1"]) == 40


def test_lexicon_merges_broken_word_in_tokenizer(lex):
    reading = Reading(
        reader="test",
        version="0",
        params_hash="p",
        image_sha1="i",
        crop="label",
        crop_px=768,
        lines=[TextLine(id=0, text="ВЕКСИЛ ЛУМ")],
        elapsed_ms=1,
    )
    assert [t.norm for t in tokenize(reading)] == ["вексил", "лум"]
    assert [t.norm for t in tokenize(reading, lexicon=lex)] == ["вексиллум"]


def test_default_paths_follow_settings(tmp_path):
    settings = Settings(data_dir=tmp_path)
    assert default_lexicon_path(settings) == tmp_path / "index" / "lexicon.json"
    assert default_gt_tokens_path(settings) == tmp_path / "gt" / "gt_tokens.jsonl"


def test_stats_counts_fields(lex):
    stats = lex.stats()
    assert stats["entries"] == len(lex)
    assert sum(stats["by_field"].values()) == len(lex)
    assert stats["without_slugs"]["sugar"] >= 1
