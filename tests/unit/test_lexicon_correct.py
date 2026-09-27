import pytest

from app.reading.contracts import LexHit, Reading, TextLine, TokenSpan
from app.reading.lexicon import correct
from app.reading.lexicon.build import Lexicon, build_from_records
from app.reading.lexicon.correct import (
    SERVICE_SKIP_COST,
    SKELETON_COST,
    SPACE_COST,
    bounded_levenshtein,
    cost_budget,
    fold_key,
    lookup,
    search,
    unmatched,
    weighted_distance,
)
from app.reading.text.tokenize import tokenize
from app.reading.text.translit import skeleton


def record(slug, winery, *, key=(), wvariants=(), cuvee=(), cvariants=(), grapes=(), **extra):
    """Запись в формате `gt_tokens.jsonl`; `extra` — gvariants, sugar, serial, keywords, color."""
    return {
        "slug": slug,
        "winery": winery,
        "category": extra.get("color"),
        "fields": {
            "winery": {"key_tokens": list(key), "variants": list(wvariants), "brands": []},
            "cuvee": {"tokens": list(cuvee), "variants": list(cvariants)},
            "grape": {
                "values": [value for value, _ in grapes],
                "codes": [code for _, code in grapes],
                "variants": list(extra.get("gvariants", ())),
            },
            "sugar": {"class": extra.get("sugar"), "variants": []},
            "serial": {
                "tokens": list(extra.get("serial", ())),
                "keywords": list(extra.get("keywords", ())),
            },
            "color": {"class": extra.get("color"), "variants": []},
        },
    }


CATALOG = [
    record(
        "massandra-muskat",
        "Массандра",
        key=["массандра"],
        wvariants=["massandra"],
        grapes=[("Мускат белый", "muscat")],
        gvariants=["muscat", "мускат"],
        sugar="sladkoe",
        color="Белое",
    ),
    record(
        "kuban-vino-aristov-xxiv",
        "Кубань-Вино",
        key=["кубань", "вино"],
        wvariants=["шато тамань", "aristov", "аристов"],
        cuvee=["аристов"],
        cvariants=["aristov"],
        serial=["XXIV"],
        sugar="brut",
        color="Белое",
    ),
    record(
        "fanagoria-shardone",
        "Фанагория",
        key=["фанагория"],
        wvariants=["fanagoria"],
        grapes=[("Шардоне", "chardonnay")],
        gvariants=["shardone", "шардоне"],  # латинского «chardonnay» нет: только скелет
        sugar="suhoe",
        color="Белое",
    ),
    record(
        "gai-kodzor-vexillum",
        "Гай-Кодзор",
        key=["гай", "кодзор"],
        wvariants=["gai kodzor"],
        cuvee=["vexillum"],
        cvariants=["вексиллум"],
        grapes=[("Ркацители", "rkatsiteli")],
        gvariants=["rkatsiteli", "ркацители"],
        sugar="polusuhoe",
        color="Белое",
    ),
    record(
        "shato-pino-blend",
        "Шато Пино",
        key=["шато", "пино"],
        wvariants=["shato pino"],
        grapes=[("Пино Нуар", "pinot_noir"), ("Мерло", "merlot")],
        gvariants=["pinot noir", "пино нуар", "merlot", "мерло"],
        serial=["30/70"],
        sugar="polusladkoe",
        color="Красное",
    ),
    record(
        "tabiya-blan-de-blan",
        "Табия",
        key=["табия"],
        wvariants=["tabiya"],
        grapes=[("Пино Нуар", "pinot_noir")],
        gvariants=["pinot noir", "пино нуар"],
        keywords=["блан де блан"],
        sugar="brut",
        color="Розовое",
    ),
    record(
        "zolotaya-balka-reserve",
        "Золотая Балка",
        key=["золотая", "балка"],
        grapes=[("Саперави", "saperavi")],
        gvariants=["saperavi", "саперави"],
        keywords=["резерв"],
        sugar="suhoe",
        color="Красное",
    ),
]


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    return build_from_records(CATALOG)


@pytest.fixture(scope="module")
def catalog_only() -> Lexicon:
    return build_from_records(CATALOG, builtin_terms=False)


def make_reading(*lines: str, reader: str = "test") -> Reading:
    return Reading(
        reader=reader,
        version="0",
        params_hash="p",
        image_sha1="i",
        crop="label",
        crop_px=768,
        lines=[TextLine(id=i, text=text) for i, text in enumerate(lines)],
        elapsed_ms=1,
    )


def toks(*lines: str) -> list[TokenSpan]:
    return tokenize(make_reading(*lines))


def found(hits, ids=None, field=None):
    return [
        (h.field, h.canonical, h.cost)
        for span, h in hits
        if (ids is None or span == ids) and (field is None or h.field == field)
    ]


def test_exact_word(lex):
    hits = lookup(toks("Массандра"), lex)
    assert found(hits, (0,), "producer") == [("producer", "Массандра", 0.0)]
    (hit,) = [h for _, h in hits if h.field == "producer"]
    assert isinstance(hit, LexHit)
    assert hit.slugs == {"massandra-muskat"}


def test_homoglyph_word_from_reader(lex):
    hits = lookup(toks("MACCAHДPA"), lex)
    assert ("producer", "Массандра", 0.0) in found(hits, (0,))


def test_homoglyphs_cost_nothing(lex):
    assert weighted_distance("caxap", "сахар") == 0.0
    # Токен с латинской «a», минуя нормализацию токенизатора.
    token = TokenSpan(
        text="Тaбия", norm="тaбия", skeleton=skeleton("тaбия"), kind="word", line_id=0, reading="r"
    )
    assert found(lookup([token], lex), (0,), "producer") == [("producer", "Табия", 0.0)]


def test_transliteration_word_by_skeleton(lex):
    hits = lookup(toks("CHARDONNAY"), lex)
    assert found(hits, (0,), "grape") == [("grape", "chardonnay", SKELETON_COST)]


def test_transliteration_phrase_blanc_de_blancs(catalog_only):
    hits = lookup(toks("Blanc de Blancs"), catalog_only)
    assert ("serial", "блан де блан", SKELETON_COST) in found(hits, (0, 1, 2))


def test_transliteration_producer_phrase(lex):
    hits = lookup(toks("Chateau Tamagne"), lex)
    assert ("producer", "Кубань-Вино", SKELETON_COST) in found(hits, (0, 1))


def test_entity_spelled_twice_gives_one_hit(lex):
    hits = lookup(toks("Шардоне"), lex)
    assert found(hits, (0,), "grape") == [("grape", "chardonnay", 0.0)]


def test_one_typo_in_long_word(lex):
    hits = lookup(toks("Ркацитали"), lex)
    assert found(hits, (0,), "grape") == [("grape", "rkatsiteli", 1.0)]


def test_two_typos_in_long_word_are_rejected(lex):
    assert found(lookup(toks("Ркапитери"), lex), field="grape") == []


def test_ocr_confusions_are_cheap(lex):
    assert weighted_distance("rnerlot", "merlot") == 0.4
    assert weighted_distance("щардоне", "шардоне") == 0.3
    assert weighted_distance("6рют", "брют") == 0.3
    assert weighted_distance("саперави", "сапераеи") == 1.0
    assert found(lookup(toks("Rnerlot"), lex), (0,), "grape") == [("grape", "merlot", 0.4)]


def test_short_words_match_only_exactly(lex):
    assert lookup(toks("sek"), lex) == []
    assert lookup(toks("сек"), lex) == []  # скелет совпадает с «sec», но слово короткое
    assert lookup(toks("брт"), lex) == []
    assert found(lookup(toks("dry"), lex), (0,)) == [("sugar", "suhoe", 0.0)]


def test_two_word_phrase(lex):
    hits = lookup(toks("Пино Нуар"), lex)
    assert hits[0][0] == (0, 1)
    assert found(hits, (0, 1), "grape") == [("grape", "pinot_noir", 0.0)]
    assert found(hits, (1,), "grape") == []  # слово фразы само по себе сорт не даёт


def test_three_word_phrase(lex):
    hits = lookup(toks("БЛАН ДЕ БЛАН"), lex)
    assert found(hits, (0, 1, 2), "serial") == [("serial", "блан де блан", 0.0)]


def test_merged_words_cost_a_space(lex):
    hits = lookup(toks("ПИНОНУАР"), lex)
    assert found(hits, (0,), "grape") == [("grape", "pinot_noir", SPACE_COST)]


def test_numbers_and_romans_match_only_exactly(lex):
    assert found(lookup(toks("30/70"), lex), (0,)) == [("serial", "30/70", 0.0)]
    assert lookup(toks("30/71"), lex) == []
    assert found(lookup(toks("XXIV"), lex), (0,)) == [("serial", "XXIV", 0.0)]
    assert lookup(toks("XXIII"), lex) == []
    assert lookup(toks("2023"), lex) == []


def test_per_span_limit_and_field_filter(lex):
    tokens = toks("Aristov")
    assert len(lookup(tokens, lex)) == 2  # винодельня «Кубань-Вино» и кюве «аристов»
    assert len(lookup(tokens, lex, per_span=1)) == 1
    only_cuvee = lookup(tokens, lex, fields={"cuvee"})
    assert only_cuvee
    assert {h.field for _, h in only_cuvee} == {"cuvee"}


def test_spans_do_not_cross_readings(lex):
    tokens = tokenize(make_reading("Пино", reader="a")) + tokenize(make_reading("Нуар", reader="b"))
    assert all(span != (0, 1) for span, _ in lookup(tokens, lex))


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (("Пино", "Нуар"), ("grape", "pinot_noir", 0.0)),
        (("Блан де", "Нуар"), ("serial", "блан де нуар", 0.0)),
        (("Блан", "Нуар"), ("serial", "блан де нуар", SERVICE_SKIP_COST)),
    ],
)
def test_phrase_split_into_lines_is_found_within_a_segment(lex, lines, expected):
    # Читатель порезал фразу на строки (EasyOCR — по слову на строку), рамок нет: отрезок —
    # всё чтение. Регрессия: правило «одна строка» теряло здесь сорт и серию.
    hits = lookup(toks(*lines), lex)
    assert expected in found(hits, field=expected[0])


@pytest.mark.parametrize("lines", [("Пино", "Нуар"), ("Блан де", "Нуар"), ("Блан", "Нуар")])
def test_phrase_does_not_cross_a_segment_or_a_line_break(lex, lines):
    tokens = toks(*lines)
    apart = [token.line_id for token in tokens]  # каждая строка — свой отрезок
    for hits in (lookup(tokens, lex, segments=apart), lookup(tokens, lex, cross_lines=False)):
        assert found(hits, field="grape") == [] and found(hits, field="serial") == []


def test_segments_must_cover_every_token(lex):
    with pytest.raises(ValueError):
        lookup(toks("Пино Нуар"), lex, segments=[0])


# --- фразы по значимым словам -------------------------------------------------


def test_whole_phrase_with_service_word(lex):
    hits = lookup(toks("Блан де Блан"), lex)
    assert found(hits, (0, 1, 2), "serial") == [("serial", "блан де блан", 0.0)]


@pytest.mark.parametrize("line", ["Блан", "Нуар"])
def test_single_word_of_phrase_gives_no_phrase_hit(lex, line):
    result = search(toks(line), lex)
    # Ни серии «блан де блан» / «блан де нуар», ни сорта pinot_noir по одному слову. Цвет
    # «Белое» у «Блан» остаётся: это однословная запись («blanc»), а не слово фразы.
    assert found(result.hits, field="serial") == [] and found(result.hits, field="grape") == []
    assert result.parts == {0}


def test_lone_phrase_word_is_not_outside_lexicon(lex):
    tokens = toks("Нуар")
    result = search(tokens, lex)
    assert result.hits == []
    assert unmatched(tokens, result.hits) == tokens  # без `known` слово выглядело бы чужим
    assert unmatched(tokens, result.hits, known=result.parts) == []


def test_service_word_may_be_skipped(lex):
    hits = lookup(toks("Блан Нуар"), lex)
    assert found(hits, (0, 1), "serial") == [("serial", "блан де нуар", SERVICE_SKIP_COST)]


def test_other_service_word_and_typo_in_significant_word(lex):
    hits = lookup(toks("Шато ле Тамамь"), lex)  # «м» вместо «н»: одна правка в слове из 6 букв
    assert found(hits, (0, 1, 2), "producer") == [
        ("producer", "Кубань-Вино", pytest.approx(SERVICE_SKIP_COST + 1.0))
    ]


@pytest.mark.parametrize("line", ["Нуар Блан", "Блан де де Нуар", "Блан Мерло Нуар"])
def test_phrase_needs_order_and_at_most_one_service_token_between(lex, line):
    assert found(lookup(toks(line), lex), field="serial") == []


def test_unmatched_strong_tokens(lex):
    tokens = toks("Pinot Noir DONUM", "Вино России", "2023", "13%", "Ай")
    hits = lookup(tokens, lex)
    assert [t.norm for t in unmatched(tokens, hits)] == ["donum"]


def test_unmatched_with_cost_threshold(lex):
    tokens = toks("Ркацитали")
    hits = lookup(tokens, lex)
    assert unmatched(tokens, hits) == []
    assert [t.norm for t in unmatched(tokens, hits, max_cost=0.5)] == ["ркацитали"]


def test_python_fallback_matches_rapidfuzz(lex, monkeypatch):
    pytest.importorskip("rapidfuzz")
    lines = [
        "Chateau Tamagne",
        "Ркацитали Rnerlot",
        "ПИНОНУАР Блан де Блан",
        "Fanagorya",
        "Блан Нуар",
        "Шато ле Тамамь",
    ]

    def run():
        return [
            [(span, h.field, h.canonical, h.cost) for span, h in lookup(toks(line), lex)]
            for line in lines
        ]

    fast = run()
    monkeypatch.setattr(correct, "USE_RAPIDFUZZ", False)
    assert run() == fast
    assert any(fast)


def test_cost_budget_grows_with_length():
    assert cost_budget(3) == 0.0
    assert 0 < cost_budget(4) < cost_budget(6) < cost_budget(9) < cost_budget(13)
    assert cost_budget(20, max_cost=0.7) == 0.7


def test_prefilter_key_never_overestimates():
    pairs = [
        ("rnerlot", "merlot"),
        ("пинонуар", "пино нуар"),
        ("щардоне", "шардоне"),
        ("ркацитали", "ркацители"),
        ("caxap", "сахар"),
        ("fanag0ria", "fanagoria"),
    ]
    for a, b in pairs:
        assert bounded_levenshtein(fold_key(a), fold_key(b), 5) <= weighted_distance(a, b)


def test_empty_inputs(lex):
    assert lookup([], lex) == []
    assert lookup(toks("Массандра"), Lexicon([], n_slugs=0)) == []
    assert unmatched([], []) == []


def test_spellings_of_shared_phrase_word_give_one_hit():
    # «каберне» есть у двух сортов: запись общая, её написания — одно доказательство фразы.
    # Одно «каберне» сорта не даёт: это обрывок «Каберне Совиньон» или «Каберне Фран».
    shared = build_from_records(
        [
            record(
                "lefkadia-cabernet-sauvignon",
                "Лефкадия",
                key=["лефкадия"],
                grapes=[("Каберне Совиньон", "cabernet_sauvignon")],
                gvariants=["kaberne sovinon", "каберне совиньон"],
            ),
            record(
                "divnomorskoe-cabernet-fran",
                "Дивноморское",
                key=["дивноморское"],
                grapes=[("Каберне Фран", "cabernet_franc")],
                gvariants=["kaberne fran", "каберне фран"],
            ),
        ]
    )
    assert shared.get("grape", "каберне").canonical == "каберне"
    assert shared.get("grape", "каберне").part
    assert lookup(toks("Каберне"), shared) == []
    assert search(toks("KABERNE"), shared).parts == {0}
    for line in ("Каберне Фран", "KABERNE FRAN"):
        assert found(lookup(toks(line), shared), (0, 1)) == [("grape", "cabernet_franc", 0.0)]


def test_repeated_lines_give_same_hits(lex):
    line = "Ркацитали Rnerlot Шато Тамань"
    one = lookup(toks(line), lex, cross_lines=False)
    two = lookup(toks(line, line), lex, cross_lines=False)
    width = len(toks(line))
    assert [(ids, h) for ids, h in two if ids[0] < width] == one
    assert [(tuple(i - width for i in ids), h) for ids, h in two if ids[0] >= width] == one


def test_back_label_words_are_not_unmatched(lex):
    tokens = toks("Содержание сахара не более 4 г/дм3", "Краснодарский край", "Агрофирма")
    assert unmatched(tokens, lookup(tokens, lex)) == []
