import pytest

from app.reading.contracts import Box, Color, LabelFields, LexHit, Reading, SugarClass, TextLine
from app.reading.fields import extract_fields
from app.reading.taxonomy import LABEL_GENERIC
from app.reading.text.normalize import norm_token
from app.reading.text.tokenize import tokenize

YEAR_NOW = 2026


def make_reading(*lines: str, reader: str = "vlm", conf: float | None = None) -> Reading:
    return Reading(
        reader=reader,
        version="1",
        params_hash="p",
        image_sha1="img",
        crop="label",
        crop_px=768,
        lines=[TextLine(id=i, text=text, conf=conf) for i, text in enumerate(lines)],
        elapsed_ms=1,
    )


def run(*readings: Reading, hits=None, year_now: int = YEAR_NOW) -> LabelFields:
    tokens = [token for reading in readings for token in tokenize(reading)]
    return extract_fields(tokens, hits or [], readings=list(readings), year_now=year_now)


def fields_of(*lines: str, **kwargs) -> LabelFields:
    return run(make_reading(*lines), **kwargs)


def hit(canonical: str, field: str, cost: float = 0.0) -> LexHit:
    return LexHit(canonical=canonical, field=field, cost=cost, slugs=frozenset({"slug"}))


# --- год урожая -------------------------------------------------------------


@pytest.mark.parametrize(
    "lines",
    [
        ["Урожай 2019"],
        ["ГОД УРОЖАЯ: 2019"],
        ["Vintage 2019"],
        ["Millésime 2019"],
        ["Millesimato 2019"],
        ["2019 года урожая"],
        ["Год урожая", "2019"],
    ],
)
def test_vintage_with_anchor(lines):
    vintage = fields_of(*lines).vintage
    assert vintage is not None and vintage.value == 2019


def test_bare_year_is_vintage():
    reading = make_reading("Шардоне 2021")
    vintage = run(reading).vintage
    assert vintage is not None
    assert (vintage.value, vintage.support, vintage.sources) == (2021, 1, [reading.key])


@pytest.mark.parametrize(
    ("year", "expected"), [(1989, None), (1990, 1990), (2027, 2027), (2028, None)]
)
def test_bare_year_range(year, expected):
    vintage = fields_of(f"Мерло {year}").vintage
    assert (vintage.value if vintage else None) == expected


def test_old_year_needs_anchor():
    assert fields_of("Урожай 1985").vintage.value == 1985
    assert fields_of("Кокур 1985").vintage is None


@pytest.mark.parametrize(
    "line",
    [
        "Основано в 1998 году",
        "Винодельня основана в 2003 г.",
        "Год основания 1999",
        "SINCE 1995",
        "Est. 2001",
        "Founded in 1992",
        "Традиции с 1996 года",
        "Золотая медаль 2018",
    ],
)
def test_founding_and_award_years_are_not_vintage(line):
    assert fields_of(line).vintage is None


@pytest.mark.parametrize(
    "line",
    [
        "Дата розлива 12.03.2024",
        "Дата розлива: 12 03 2024",
        "Розлив 2024",
        "Bottled 2023",
        "14.02.2023",
        "2024-03-12",
        "15 марта 2024",
        "March 2024",
        "ГОСТ 32030-2013",
        "© 2021",
        "Партия 2003",
        "LOT 2011",
        "Дата производства: 2024",
    ],
)
def test_dates_and_documents_are_not_vintage(line):
    assert fields_of(line).vintage is None


def test_anchor_does_not_leak_past_another_number():
    vintage = fields_of("EST. 1907", "2020").vintage
    assert vintage is not None and vintage.value == 2020


def test_phone_number_groups_are_not_vintage():
    assert fields_of("Шардоне", "Тел.: 8 800 20 23 000").vintage is None


def test_anchored_year_beats_founding_and_bare_years():
    assert fields_of("Винодельня основана в 2003 году", "Урожай 2020").vintage.value == 2020
    vintage = fields_of("Урожай 2019", "Выдержка 2016").vintage
    assert vintage.value == 2019


def test_conflicting_equal_years_abstain():
    assert fields_of("Шардоне 2019", "Мерло 2021").vintage is None


def test_equal_values_from_different_readings_follow_reader_order():
    a = make_reading("Шардоне 2019", "Белое", reader="vlm")
    b = make_reading("Мерло 2021", "Красное", reader="ocr")
    first = run(a, b)
    assert (first.vintage.value, first.vintage.sources) == (2019, [a.key])
    assert first.color.value is Color.WHITE
    assert run(b, a).vintage.value == 2021


def test_conflict_inside_one_reading_abstains_with_other_readings():
    a = make_reading("Шардоне 2019", "Мерло 2021", reader="vlm")
    b = make_reading("Брют", reader="ocr")
    assert run(a, b).vintage is None


def test_vintage_support_counts_distinct_readings():
    a = make_reading("Урожай 2019", "2019", reader="vlm")
    b = make_reading("2019", reader="ocr")
    c = make_reading("2021", reader="rapid")
    vintage = run(c, a, b).vintage
    assert vintage.value == 2019
    assert vintage.support == 2
    assert vintage.sources == [a.key, b.key]


def test_roman_year():
    assert fields_of("MMXXII").vintage.value == 2022


# --- сахар -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Экстра брют", SugarClass.EXTRA_BRUT),
        ("БРЮТ НАТЮР", SugarClass.BRUT_NATURE),
        ("Брют", SugarClass.BRUT),
        ("Полусухое", SugarClass.SEMI_DRY),
        ("Сухое", SugarClass.DRY),
        ("Полусладкое", SugarClass.SEMI_SWEET),
        ("Сладкое", SugarClass.SWEET),
    ],
)
def test_longer_sugar_forms_first(line, expected):
    assert [e.value for e in fields_of(f"Вино игристое {line}").sugar] == [expected]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("EXTRA BRUT", SugarClass.EXTRA_BRUT),
        ("Brut Nature", SugarClass.BRUT_NATURE),
        ("Pas Dosé", SugarClass.BRUT_NATURE),
        ("Zero Dosage", SugarClass.BRUT_NATURE),
        ("Demi-Sec", SugarClass.SEMI_SWEET),
        ("Semi-Dry", SugarClass.SEMI_DRY),
        ("Semi-Sweet", SugarClass.SEMI_SWEET),
        ("Sec", SugarClass.DRY),
        ("Secco", SugarClass.DRY),
        ("Dry", SugarClass.DRY),
        ("Doux", SugarClass.SWEET),
        ("Dolce", SugarClass.SWEET),
        ("Sweet", SugarClass.SWEET),
    ],
)
def test_latin_sugar(line, expected):
    assert [e.value for e in fields_of(line).sugar] == [expected]


def test_extra_dry_gives_no_sugar():
    assert fields_of("Prosecco Extra Dry").sugar == []


def test_several_sugar_values():
    sugar = fields_of("Брют", "Полусухое").sugar
    assert {e.value for e in sugar} == {SugarClass.BRUT, SugarClass.SEMI_DRY}


def test_sugar_phrase_across_lines():
    sugar = fields_of("EXTRA", "BRUT").sugar
    assert [(e.value, e.matched) for e in sugar] == [(SugarClass.EXTRA_BRUT, "extra brut")]


def test_sugar_lexicon_hit_merges_with_word():
    a = make_reading("Брют", reader="vlm")
    b = make_reading("6рют", reader="ocr")  # «б» → «6»: путаница OCR в пределах бюджета
    tokens = tokenize(a) + tokenize(b)
    fields = extract_fields(tokens, [([1], hit("brut", "sugar", 0.3))], readings=[a, b])
    assert [(e.value, e.support, e.sources) for e in fields.sugar] == [
        (SugarClass.BRUT, 2, [a.key, b.key])
    ]


def test_lexicon_sugar_by_skeleton_is_not_sugar():
    # Словарь находит «sweet» по скелету «svet», но это не опечатка слова сахара.
    reading = make_reading("Новый Свет", "Пино Нуар")
    fields = extract_fields(
        tokenize(reading), [([1], hit("sladkoe", "sugar", 0.2))], readings=[reading]
    )
    assert fields.sugar == []


@pytest.mark.parametrize(
    ("line", "ids", "canonical", "expected"),
    [
        ("Prosecco Extra Dry", [2], "suhoe", []),
        ("Экстра Брют", [1], "brut", [SugarClass.EXTRA_BRUT]),
        ("Brut Nature", [0], "brut", [SugarClass.BRUT_NATURE]),
    ],
)
def test_lexicon_sugar_inside_longer_phrase_is_dropped(line, ids, canonical, expected):
    reading = make_reading(line)
    hits = [(ids, hit(canonical, "sugar"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.sugar] == expected


def test_longer_lexicon_sugar_typo_beats_shorter_word():
    reading = make_reading("3кстра брют")
    hits = [([0, 1], hit("extra_brut", "sugar", 1.0))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [(e.value, e.matched) for e in fields.sugar] == [(SugarClass.EXTRA_BRUT, "extra_brut")]


# --- серия ---------------------------------------------------------------------


def test_roman_serial():
    serial = fields_of("Аристов", "Серия XXIV").serial
    assert [(e.value, e.matched) for e in serial] == [("XXIV", None)]


@pytest.mark.parametrize(
    "line", ["V", "Vi", "vi xi", "75 CL", "Серия CC", "LI", "[II(", "Ю0 II h", "Партия 12 IX"]
)
def test_not_roman_serial(line):
    assert fields_of(line).serial == []


def test_roman_serial_inside_name():
    assert [e.value for e in fields_of("Александр II Magnum").serial] == ["II"]


def test_roman_month_date_is_not_vintage_or_serial():
    fields = fields_of("12.IX.2021")
    assert fields.vintage is None
    assert fields.serial == []


def test_short_word_does_not_match_sugar_by_skeleton():
    assert fields_of("Новый Свет", "Пино Нуар").sugar == []


def test_extra_sec_gives_no_sugar():
    assert fields_of("Экстра-сек").sugar == []


def test_blend_ratio_serial():
    assert [e.value for e in fields_of("Фантом 30/70").serial] == ["30/70"]
    assert fields_of("Партия 15/08").serial == []


@pytest.mark.parametrize(
    ("line", "value", "matched"),
    [
        ("Гран Резерв", "гран резерв", "гран резерв"),
        ("Grand Reserve", "гран резерв", "гран резерв"),
        ("Reserve", "резерв", "резерв"),
        ("Riserva", "резерв", "резерв"),
        ("Blanc de Noirs", "блан де нуар", "блан де нуар"),
        ("Блан де Блан", "блан де блан", "блан де блан"),
        ("Cuvée Prestige", "кюве", "кюве"),
    ],
)
def test_serial_keywords(line, value, matched):
    assert [(e.value, e.matched) for e in fields_of(line).serial] == [(value, matched)]


def test_serial_from_lexicon_hit():
    reading = make_reading("Фантом")
    fields = extract_fields(tokenize(reading), [([0], hit("Фантом", "serial"))], readings=[reading])
    assert [(e.value, e.matched) for e in fields.serial] == [("Фантом", "Фантом")]


def test_serial_word_and_lexicon_hit_are_one_value():
    a = make_reading("Reserve", reader="vlm")
    b = make_reading("Резерв", reader="ocr")
    tokens = tokenize(a) + tokenize(b)
    fields = extract_fields(
        tokens, [([0], hit("резерв", "serial")), ([1], hit("Резерв", "serial"))], readings=[a, b]
    )
    assert [(e.value, e.support, e.sources) for e in fields.serial] == [
        ("резерв", 2, [a.key, b.key])
    ]


def test_lexicon_roman_serial_is_one_value_with_token():
    reading = make_reading("Серия XXIV")
    fields = extract_fields(tokenize(reading), [([1], hit("XXIV", "serial"))], readings=[reading])
    assert [(e.value, e.matched) for e in fields.serial] == [("XXIV", "XXIV")]


def test_lexicon_ratio_serial_is_one_value_with_token():
    reading = make_reading("Фантом 30/70")
    fields = extract_fields(tokenize(reading), [([1], hit("30/70", "serial"))], readings=[reading])
    assert [(e.value, e.matched) for e in fields.serial] == [("30/70", "30/70")]


@pytest.mark.parametrize("line", ["Ю0 II h", "Вино ii"])
def test_lexicon_roman_serial_keeps_roman_checks(line):
    reading = make_reading(line)
    tokens = tokenize(reading)
    assert tokens[1].norm == "ii"
    fields = extract_fields(tokens, [([1], hit("II", "serial"))], readings=[reading])
    assert fields.serial == []


def test_serial_lexicon_spelling_maps_to_canonical():
    reading = make_reading("Riserva")
    fields = extract_fields(
        tokenize(reading), [([0], hit("Riserva", "serial"))], readings=[reading]
    )
    assert [(e.value, e.matched) for e in fields.serial] == [("резерв", "резерв")]


# --- крепость ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "value"),
    [
        ("13,5% об.", 13.5),
        ("Alc. 12% vol", 12.0),
        ("12,5 % vol", 12.5),
        ("11 об.", 11.0),
        ("Крепость 14%", 14.0),
    ],
)
def test_abv(line, value):
    assert fields_of(line).abv.value == value


@pytest.mark.parametrize(
    "line", ["0,75 л", "750 мл", "Выдержка 18 месяцев", "11 обл", "100%", "3%", "23% vol"]
)
def test_abv_traps(line):
    assert fields_of(line).abv is None


def test_blend_share_is_not_abv():
    assert fields_of("Каберне Совиньон 85%", "15% Мерло").abv is None


def test_grape_on_previous_line_is_not_a_blend():
    assert fields_of("Саперави", "13,5%").abv.value == 13.5
    assert fields_of("Саперави 13,5%").abv is None


def test_unit_makes_percent_next_to_grape_abv():
    assert fields_of("Мерло 13% vol").abv.value == 13.0


def test_percent_with_unit_beats_bare_percent():
    assert fields_of("12%", "13,5% об.").abv.value == 13.5


# --- цвет ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "color"),
    [
        ("Вино сухое красное", Color.RED),
        ("БЕЛОЕ", Color.WHITE),
        ("Rosé", Color.ROSE),
        ("Оранжевое", Color.ORANGE),
        ("Rosso", Color.RED),
    ],
)
def test_color_words(line, color):
    assert fields_of(line).color.value is color


@pytest.mark.parametrize(
    "line", ["Sauvignon Blanc", "Blanc de Blancs", "Мускат Розовый", "Prosecco"]
)
def test_color_not_from_grape_serial_or_other_words(line):
    assert fields_of(line).color is None


def test_color_not_from_named_cuvee():
    reading = make_reading("Rosso Antico")
    tokens = tokenize(reading)
    assert extract_fields(tokens, [], readings=[reading]).color.value is Color.RED
    fields = extract_fields(tokens, [([0, 1], hit("Rosso Antico", "cuvee"))], readings=[reading])
    assert fields.color is None


def test_generic_names_do_not_take_color_word():
    reading = make_reading("Вино сухое красное")
    hits = [
        ([1, 2], hit("Сухое Красное", "cuvee")),  # кюве из одних слов таксономии
        ([2], hit("красные", "cuvee", 1.0)),  # нечёткое кюве на самом слове цвета
    ]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert fields.color.value is Color.RED


def test_conflicting_colors_abstain():
    assert fields_of("Белое", "Красное").color is None


def test_lexicon_color_inside_grape_name_is_dropped():
    reading = make_reading("Traminer Rose")
    fields = extract_fields(tokenize(reading), [([1], hit("Розовое", "color"))], readings=[reading])
    assert fields.color is None


def test_lexicon_color_typo_is_color():
    reading = make_reading("Красвое")  # одна замена «н» → «в»: бюджет словаря для 7 букв
    tokens = tokenize(reading)
    assert fields_of("Красвое").color is None  # таксономия опечатку не видит
    fields = extract_fields(tokens, [([0], hit("Красное", "color", 1.0))], readings=[reading])
    assert fields.color.value is Color.RED


# --- словарь каталога ----------------------------------------------------------


def test_lexicon_fields_from_hits():
    a = make_reading("Массандра", "Пино Нуар", reader="vlm")
    b = make_reading("МАССАНДРА", reader="ocr")
    tokens = tokenize(a) + tokenize(b)  # массандра, пино, нуар | массандра
    hits = [
        ([3], hit("Массандра", "producer", 0.4)),
        ([0], hit("Массандра", "producer")),
        ([1, 2], hit("Пино Нуар", "grape")),
        ([0], hit("Массандра Резерв", "cuvee", 1.0)),
    ]
    fields = extract_fields(tokens, hits, readings=[a, b])

    (producer,) = fields.producer
    assert (producer.value, producer.matched, producer.support) == ("Массандра", "Массандра", 2)
    assert producer.sources == [a.key, b.key]
    assert producer.conf == 1.0
    assert [(e.value, e.support) for e in fields.grapes] == [("Пино Нуар", 1)]
    assert [(e.value, e.conf) for e in fields.cuvee] == [("Массандра Резерв", 0.5)]


def test_lexicon_fields_ranked_by_support():
    a = make_reading("Шардоне Алиготе", reader="vlm")
    b = make_reading("Алиготе", reader="ocr")
    tokens = tokenize(a) + tokenize(b)
    hits = [
        ([0], hit("Шардоне", "grape")),
        ([1], hit("Алиготе", "grape")),
        ([2], hit("Алиготе", "grape")),
    ]
    fields = extract_fields(tokens, hits, readings=[a, b])
    assert [e.value for e in fields.grapes] == ["Алиготе", "Шардоне"]


@pytest.mark.parametrize("ids", [[], [5]])
def test_invalid_hit_indices_raise(ids):
    reading = make_reading("Брют")
    with pytest.raises(ValueError):
        extract_fields(tokenize(reading), [(ids, hit("Брют", "sugar"))], readings=[reading])


# --- кюве против сорта, сахара и серии ------------------------------------------


def test_grape_phrase_takes_its_word_from_cuvee():
    reading = make_reading("Пино нуар")
    hits = [([0, 1], hit("pinot_noir", "grape")), ([1], hit("нуар", "cuvee"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.grapes] == ["pinot_noir"]
    assert fields.cuvee == []


@pytest.mark.parametrize(
    ("line", "ids", "canonical"),
    [("Экстра брют", [1], "брют"), ("Гран Резерв", [1], "резерв"), ("Серия XXIV", [1], "XXIV")],
)
def test_sugar_and_serial_words_are_not_cuvee(line, ids, canonical):
    reading = make_reading(line)
    hits = [(ids, hit(canonical, "cuvee"))]
    assert extract_fields(tokenize(reading), hits, readings=[reading]).cuvee == []


def test_rejected_sugar_hit_does_not_take_cuvee():
    reading = make_reading("Новый Свет")
    hits = [([1], hit("sladkoe", "sugar", 0.2)), ([1], hit("Свет", "cuvee"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert fields.sugar == [] and [e.value for e in fields.cuvee] == ["Свет"]


def test_generic_label_words_are_not_cuvee():
    reading = make_reading("выдержка 18 месяцев в дубе")
    tokens = tokenize(reading)
    assert [t.norm for t in tokens] == ["выдержка", "18", "месяцев", "в", "дубе"]
    hits = [
        ([0], hit("выдержка", "cuvee")),
        ([2], hit("месяцев", "cuvee")),
        ([4], hit("дубе", "cuvee", 0.3)),
    ]
    assert extract_fields(tokens, hits, readings=[reading]).cuvee == []


def test_generic_dictionary_form_is_not_cuvee_even_from_a_typo():
    reading = make_reading("Выдерсжа сталь")
    hits = [([0], hit("выдержка", "cuvee", 1.0)), ([1], hit("сталь", "cuvee"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.cuvee] == ["сталь"]


def test_real_cuvee_next_to_grape_and_serial_word_stays():
    reading = make_reading("Кюве Александр", "Шардоне")
    hits = [
        ([0, 1], hit("Кюве Александр", "cuvee")),
        ([1], hit("Александр", "cuvee")),
        ([2], hit("chardonnay", "grape")),
    ]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.cuvee] == ["Кюве Александр", "Александр"]
    assert [e.value for e in fields.grapes] == ["chardonnay"]
    assert [e.value for e in fields.serial] == ["кюве"]


def test_exact_cuvee_stays_against_fuzzier_serial_and_grape():
    # Регрессия: точное кюве терялось, если слово «заняла» нечёткая серия или сорт за 1,0.
    reading = make_reading("Селекто", "Мерлон")
    hits = [
        ([0], hit("селекто", "cuvee")),
        ([0], hit("селекта", "serial", 1.0)),
        ([1], hit("мерлон", "cuvee")),
        ([1], hit("merlot", "grape", 1.0)),
    ]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.cuvee] == ["селекто", "мерлон"]
    assert [e.value for e in fields.serial] == ["селекта"]
    assert [e.value for e in fields.grapes] == ["merlot"]


def test_fuzzy_cuvee_is_taken_by_exact_serial_and_sugar():
    reading = make_reading("Резерв полусладкое")
    hits = [([0], hit("резервы", "cuvee", 1.0)), ([1], hit("полусладкий", "cuvee", 1.0))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert fields.cuvee == []
    assert [e.value for e in fields.serial] == ["резерв"]


def test_single_word_cuvee_equal_to_grape_goes_to_grape():
    # Известное ограничение: однословное кюве, совпавшее с сортом той же ценой, — сорт.
    reading = make_reading("Сира")
    hits = [([0], hit("syrah", "grape")), ([0], hit("сира", "cuvee"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert [e.value for e in fields.grapes] == ["syrah"]
    assert fields.cuvee == []


def test_longer_grape_phrase_takes_cuvee_word_despite_a_typo_elsewhere():
    reading = make_reading("Пимо нуар")
    hits = [([0, 1], hit("pinot_noir", "grape", 1.0)), ([1], hit("нуар", "cuvee"))]
    fields = extract_fields(tokenize(reading), hits, readings=[reading])
    assert fields.cuvee == [] and [e.value for e in fields.grapes] == ["pinot_noir"]


def test_taxonomy_grape_without_grape_hit_does_not_take_cuvee():
    # Слово занимает поле, которое дало значение: иначе не было бы ни сорта, ни кюве.
    reading = make_reading("Шардоне")
    fields = extract_fields(tokenize(reading), [([0], hit("шардоне", "cuvee"))], readings=[reading])
    assert fields.grapes == [] and [e.value for e in fields.cuvee] == ["шардоне"]


def boxed_reading(*lines: tuple[str, tuple[float, float, float, float]]) -> Reading:
    return make_reading(*(text for text, _ in lines)).model_copy(
        update={
            "lines": [
                TextLine(id=i, text=text, box=Box(x0=x0, y0=y0, x1=x1, y1=y1))
                for i, (text, (x0, y0, x1, y1)) in enumerate(lines)
            ]
        }
    )


def test_taxonomy_phrase_joins_words_of_one_row_but_not_distant_lines():
    row = boxed_reading(("Экстра", (0.10, 0.40, 0.30, 0.45)), ("брют", (0.32, 0.40, 0.45, 0.45)))
    assert [e.value for e in run(row).sugar] == [SugarClass.EXTRA_BRUT]
    apart = boxed_reading(("Экстра", (0.10, 0.10, 0.30, 0.15)), ("брют", (0.10, 0.80, 0.25, 0.85)))
    assert [e.value for e in run(apart).sugar] == [SugarClass.BRUT]


def test_label_generic_words_are_normalized_and_keep_serials():
    assert all(norm_token(word) == word for word in LABEL_GENERIC)
    assert {"защищенным", "объем", "выдержка", "месяцев"} <= LABEL_GENERIC
    assert "резерв" not in LABEL_GENERIC


# --- винная ли этикетка --------------------------------------------------------


def test_not_wine_label_without_vocabulary():
    fields = fields_of("Молоко 3,2%", "Пастеризованное")
    assert fields.is_wine_label is False


@pytest.mark.parametrize(
    "line", ["Вино столовое", "Шардоне", "BRUT", "Chateau", "Херес Амонтильядо", "MADEIRA"]
)
def test_wine_label_by_vocabulary(line):
    assert fields_of(line).is_wine_label is True


def test_wine_label_by_lexicon_hit():
    reading = make_reading("Табия")
    fields = extract_fields(
        tokenize(reading), [([0], hit("Табия", "producer"))], readings=[reading]
    )
    assert fields.is_wine_label is True


def test_empty_input():
    fields = extract_fields([], [], readings=[])
    assert fields.model_dump() == LabelFields(is_wine_label=False).model_dump()


def test_unmatched_left_to_pipeline():
    assert fields_of("Урожай 2019", "Неизвестное Слово").unmatched == []


def test_token_confidence_carried():
    reading = make_reading("Урожай 2019", conf=0.6)
    assert run(reading).vintage.conf == 0.6
