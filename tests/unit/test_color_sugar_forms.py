"""Цвет и сахар во всех родах и числах: таксономия, поля этикетки и слова названия resolve.

Прилагательное согласуется со своим словом: «вино белое», но «Мускатель белый», «Мадера
белая», «Вина белые». Словарь знал только средний род, и «БЕЛЫЙ» под «МУСКАТЕЛЬ» не давал
цвета.

Строки собраны из общих форм слов. Несколько раскладок повторяют чтения VLM с кадров dev
(«МУСКАТЕЛЬ» / «БЕЛЫЙ» / «0,75», «ПОРТВЕЙН» / «КРАСНЫЙ» / «ЛИВАДИЯ», «КРАСНАЯ ГОРКА»):
dev правилами не запрещён. Строк публичных кадров здесь нет, и поведение на них тесты не
проверяют. Названия каталога («Chateau de Talu Блан», «Два Сердца Мускат Сухой», «Красная
стрелка») — из `gt_tokens.jsonl`, а не с кадров.
"""

import pytest
from catalog import wine_record
from learned_synth import cv_record

from app.api.cards import sugar_label
from app.reading.contracts import Box, Color, LabelFields, LexHit, Reading, SugarClass, TextLine
from app.reading.fields import ADJECTIVE_CLAIM_COST, extract_fields
from app.reading.lexicon.correct import SKELETON_COST
from app.reading.taxonomy import (
    adjective_forms,
    color_of,
    find_colors,
    find_grapes,
    find_serial,
    find_sugar,
    sugar_class,
)
from app.reading.text.tokenize import tokenize
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import (
    TextRead,
    WordBag,
    catalog_stats,
    color_forms,
    name_word_found,
    query_features,
)

COLOR_FORMS = [
    (form, color)
    for masculine, color in (
        ("белый", Color.WHITE),
        ("красный", Color.RED),
        ("розовый", Color.ROSE),
        ("оранжевый", Color.ORANGE),
        ("янтарный", Color.ORANGE),
    )
    for form in adjective_forms(masculine)
]
SUGAR_FORMS = [
    (form, sugar)
    for masculine, sugar in (
        ("сухой", SugarClass.DRY),
        ("полусухой", SugarClass.SEMI_DRY),
        ("полусладкий", SugarClass.SEMI_SWEET),
        ("сладкий", SugarClass.SWEET),
        ("десертный", SugarClass.SWEET),
    )
    for form in adjective_forms(masculine)
]


def reading(*lines: str, boxes: list[Box] | None = None) -> Reading:
    return Reading(
        reader="vlm",
        version="1",
        params_hash="p",
        image_sha1="img",
        crop="full",
        crop_px=1024,
        lines=[
            TextLine(id=i, text=text, box=boxes[i] if boxes else None)
            for i, text in enumerate(lines)
        ],
        elapsed_ms=1,
    )


def fields_of(*lines: str, hits=(), boxes: list[Box] | None = None) -> LabelFields:
    read = reading(*lines, boxes=boxes)
    return extract_fields(tokenize(read), list(hits), readings=[read], year_now=2026)


def color_sugar(fields: LabelFields) -> tuple[Color | None, list[SugarClass]]:
    return (fields.color.value if fields.color else None, [e.value for e in fields.sugar])


def hit(canonical: str, field: str, cost: float = 0.0) -> LexHit:
    return LexHit(canonical=canonical, field=field, cost=cost, slugs=frozenset({"slug"}))


# ------------------------------------------------------------------ формы и таксономия
@pytest.mark.parametrize(
    ("masculine", "forms"),
    [
        ("белый", ("белое", "белый", "белая", "белые")),
        ("сухой", ("сухое", "сухой", "сухая", "сухие")),
        ("сладкий", ("сладкое", "сладкий", "сладкая", "сладкие")),
        ("Чёрный", ("черное", "черный", "черная", "черные")),
    ],
)
def test_adjective_forms_neuter_first(masculine, forms):
    assert adjective_forms(masculine) == forms


@pytest.mark.parametrize("word", ["синий", "свежий", "вино", "ый"])
def test_adjective_forms_reject_what_they_cannot_inflect(word):
    with pytest.raises(ValueError):
        adjective_forms(word)


@pytest.mark.parametrize(("form", "color"), COLOR_FORMS)
def test_color_in_every_gender_and_number(form, color):
    assert find_colors(form) == [color]
    assert fields_of(form.upper()).color.value is color
    assert color_of(form) is color  # ключ сравнения resolve (`attrs.color_key`)


@pytest.mark.parametrize(("form", "sugar"), SUGAR_FORMS)
def test_sugar_in_every_gender_and_number(form, sugar):
    assert find_sugar(form) == [sugar]
    assert [e.value for e in fields_of(form.upper()).sugar] == [sugar]
    assert sugar_class(form) is sugar  # ключ сравнения resolve (`attrs.sugar_key`)


@pytest.mark.parametrize("form", adjective_forms("черный"))
def test_black_is_not_a_wine_color(form):
    """«Чёрный принц» в каталоге белое, «Цимлянский чёрный» бывает розовым."""
    assert find_colors(form) == []
    assert fields_of(form).color is None


@pytest.mark.parametrize(
    ("text", "sugar"),
    [
        ("полу-сухая", SugarClass.SEMI_DRY),
        ("Полу-сладкий", SugarClass.SEMI_SWEET),
        ("экстра сухой", None),  # «Extra Dry» по-русски слаще брюта: слово занято, сахара нет
    ],
)
def test_prefixed_sugar_forms(text, sugar):
    assert find_sugar(text) == ([sugar] if sugar else [])


@pytest.mark.parametrize(
    ("text", "color"),
    [
        ("Bianco", Color.WHITE),
        ("Blanc", Color.WHITE),
        ("Blanco", Color.WHITE),
        ("Rosso", Color.RED),
        ("Rouge", Color.RED),
        ("Rosato", Color.ROSE),
        ("Rosé", Color.ROSE),
        ("Orange", Color.ORANGE),
        ("vino beloe", Color.WHITE),  # латиница среднего рода — как в slug каталога
    ],
)
def test_latin_colors(text, color):
    assert fields_of(text).color.value is color


def test_latin_non_neuter_russian_is_a_name_not_a_color():
    """«Krasnaya Gorka» латиницей — транслит имени, а не цвет вина."""
    assert fields_of("Krasnaya Gorka").color is None
    assert fields_of("Belaya").color is None


@pytest.mark.parametrize(
    ("text", "grapes", "serial"),
    [
        ("Sauvignon Blanc", ["sauvignon_blanc"], []),
        ("Pinot Blanc", ["pinot_blanc"], []),
        ("Chenin Blanc", ["chenin_blanc"], []),
        ("Pinot Noir", ["pinot_noir"], []),
        ("Blanc de Blancs", [], ["блан де блан"]),
        ("Мускат белый", ["muscat"], []),
        ("Кокур белый", ["kokur"], []),
        ("Траминер розовый", ["traminer_rose"], []),  # вино из него белое
    ],
)
def test_color_inside_a_longer_grape_or_serial_is_not_a_color(text, grapes, serial):
    assert find_colors(text) == []
    assert find_grapes(text) == grapes and find_serial(text) == serial
    assert fields_of(text).color is None


def test_card_label_stays_neuter():
    """Первое написание сахара — подпись карточки: «сухое», а не «сухой»."""
    labels = [sugar_label(sugar) for sugar in ("suhoe", "polusuhoe", "polusladkoe", "sladkoe")]
    assert labels == ["сухое", "полусухое", "полусладкое", "сладкое"]


# ------------------------------------------------------------------ поля этикетки
@pytest.mark.parametrize(
    ("lines", "color", "sugar"),
    [
        (["МУСКАТЕЛЬ", "БЕЛЫЙ", "0,75"], Color.WHITE, []),
        (["МУСКАТЕЛЬ РОЗОВЫЙ"], Color.ROSE, []),
        (["ПОРТВЕЙН", "КРАСНЫЙ", "ЛИВАДИЯ"], Color.RED, []),
        (["Портвейн белый Алушта"], Color.WHITE, []),  # после стиля — при нём, а не имя
        (["Херес сухой"], None, [SugarClass.DRY]),
        (["Мадера белая сладкая"], Color.WHITE, [SugarClass.SWEET]),
        (["Вина белые сухие"], Color.WHITE, [SugarClass.DRY]),
        (["Кокур белый сухой"], None, [SugarClass.DRY]),
        (["Белый купаж"], Color.WHITE, []),
        (["Красный брют"], Color.RED, [SugarClass.BRUT]),
    ],
)
def test_label_fields_in_any_gender(lines, color, sugar):
    fields = fields_of(*lines)
    assert (fields.color.value if fields.color else None) is color
    assert [e.value for e in fields.sugar] == sugar


@pytest.mark.parametrize(
    "line", ["КРАСНАЯ ГОРКА", "Белая Львица", "Красные Глины", "Сухой Лог", "Розовая пантера"]
)
def test_adjective_before_a_name_word_starts_a_proper_name(line):
    fields = fields_of(line)
    assert fields.color is None and fields.sugar == []


@pytest.mark.parametrize(
    "lines",
    [
        ["КРАСНАЯ", "ГОРКА"],
        ["БЕЛАЯ", "СКАЛА"],
        ["БЕЛЫЙ", "КОЛОДЕЦ"],
        ["СУХОЙ", "ЛИМАН"],
        ["Пино Нуар кларет", "КРАСНАЯ", "СТРЕЛКА"],  # розовое: не «красное» вина
    ],
)
def test_proper_name_across_a_line_break(lines):
    """Имя в две строки — то же имя: сосед ищется в отрезке текста, а не только в строке."""
    assert color_sugar(fields_of(*lines)) == (None, [])
    assert color_sugar(fields_of(" ".join(lines))) == (None, [])


def test_proper_name_rule_stops_at_a_layout_break():
    """Строки в разных местах кадра — разные отрезки: слово с другого места имени не делает."""
    top = Box(x0=0.05, y0=0.05, x1=0.25, y1=0.1)
    far = Box(x0=0.6, y0=0.8, x1=0.9, y1=0.85)
    below = Box(x0=0.05, y0=0.11, x1=0.25, y1=0.16)
    assert fields_of("БЕЛАЯ", "СКАЛА", boxes=[top, far]).color.value is Color.WHITE
    assert fields_of("БЕЛАЯ", "СКАЛА", boxes=[top, below]).color is None  # название в две строки


@pytest.mark.parametrize(
    "lines",
    [
        ["Шардоне Красная Горка"],
        ["Мерло Белая Скала"],
        ["Рислинг Сухая Балка"],
        ["Каберне Красные Холмы"],
        ["ШАРДОНЕ", "КРАСНАЯ", "ГОРКА"],
    ],
)
def test_feminine_or_plural_after_a_grape_starts_a_proper_name(lines):
    """Сорт мужского рода с женским родом и множественным числом не согласуется: это имя."""
    assert color_sugar(fields_of(*lines)) == (None, [])


@pytest.mark.parametrize(
    ("lines", "color", "sugar"),
    [
        (["Кокур десертный Сурож"], None, [SugarClass.SWEET]),  # название каталога
        (["МУСКАТ ДЕСЕРТНЫЙ", "ВЫДЕРЖАНЫЙ"], None, [SugarClass.SWEET]),  # опечатка строкой ниже
        # Известное ограничение: мужской род после сорта перед именем по словам не отличить от
        # «Кокур десертный Сурож» — остаётся признаком вина.
        (["Саперави Белый Колодец"], Color.WHITE, []),
        (["Каберне Совиньон Сухой Лиман"], None, [SugarClass.DRY]),
    ],
)
def test_masculine_after_a_grape_agrees_with_it(lines, color, sugar):
    assert color_sugar(fields_of(*lines)) == (color, sugar)


@pytest.mark.parametrize(
    ("lines", "color", "sugar"),
    [
        (["Рислинг белый"], Color.WHITE, []),  # после сорта в конце — цвет вина
        (["Кокур белая", "Сары Пандас"], Color.WHITE, []),  # после — сорт
        (["ПОРТВЕЙН", "БЕЛЫЙ", "АЛУШТА"], Color.WHITE, []),  # после стиля — при нём
        (["МУСКАТЕЛЬ", "РОЗОВЫЙ", "КРЫМ"], Color.ROSE, []),
        (["ХЕРЕС", "СУХОЙ", "ОЛОРОСО"], None, [SugarClass.DRY]),
        (["Мадера белая Массандра"], Color.WHITE, []),  # стиль женского рода
        (["Купаж красный Абрау"], Color.RED, []),  # купаж — тоже вино
    ],
)
def test_adjective_next_to_its_grape_or_style_stays_a_wine_feature(lines, color, sugar):
    assert color_sugar(fields_of(*lines)) == (color, sugar)


def test_lexicon_typo_of_an_agreeing_form_is_checked_for_a_name_too():
    """Опечатка словаря «Красвая» проходит ту же проверку имени, что и «Красная»."""
    typo = hit("красное", "color", 1.0)
    assert fields_of("КРАСВАЯ ГОРКА", hits=[([0], typo)]).color is None
    assert fields_of("КРАСВАЯ", "ГОРКА", hits=[([0], typo)]).color is None
    assert fields_of("КРАСВАЯ", hits=[([0], typo)]).color.value is Color.RED  # в конце — цвет


@pytest.mark.parametrize(
    ("text", "lex", "expected"),
    [
        # Гомоглифы: «KPACHOE» — это «красное», а не «красные» за одну правку.
        ("KPACHOE CEMEЙНАЯ", hit("красное", "color"), (Color.RED, [])),
        ("CYXOE ЛИМАН", hit("suhoe", "sugar"), (None, [SugarClass.DRY])),
        # Поровну до «красное» и «красная»: форма не выбрана, опечатка остаётся цветом.
        ("КРАСНОЯ ГОРКА", hit("красное", "color", 1.0), (Color.RED, [])),
    ],
)
def test_lexicon_typo_nearest_to_neuter_is_the_wine(text, lex, expected):
    assert color_sugar(fields_of(text)) == (None, [])  # значение даёт только словарь
    assert color_sugar(fields_of(text, hits=[([0], lex)])) == expected


def test_neuter_is_the_wine_even_before_a_name():
    """Средний род согласуется с «вином»: «Белое» перед именем — всё равно цвет."""
    assert fields_of("Белое Львица").color.value is Color.WHITE
    assert fields_of("Белая Львица белое полусладкое").color.value is Color.WHITE


def test_two_colors_in_one_reading_still_abstain():
    assert fields_of("Белый", "Красный").color is None


@pytest.mark.parametrize(
    ("line", "lex"),
    [
        ("Вино красное", hit("красные", "cuvee", 1.0)),  # нечёткое кюве на слове цвета
        ("Rouge", hit("руж", "cuvee", SKELETON_COST)),  # транслит по скелету
        ("Red", hit("ред", "cuvee")),  # точная норма словаря «red», но не написание каталога
        ("Rosso", hit("россо", "cuvee")),
        ("сухое", hit("сухой", "cuvee", 1.0)),
    ],
)
def test_color_or_sugar_word_is_not_a_fuzzy_or_transliterated_cuvee(line, lex):
    fields = fields_of(line, hits=[([0], lex)])
    assert color_sugar(fields) != (None, []) and fields.cuvee == []


def test_adjective_claim_is_between_exact_and_any_inexact_hit():
    assert 0.0 < ADJECTIVE_CLAIM_COST < SKELETON_COST


@pytest.mark.parametrize(
    ("lines", "ids", "cuvee", "color", "sugar"),
    [
        (["Chateau de Talu", "Блан"], [3], "блан", Color.WHITE, []),
        (["Chateau de Talu", "Руж"], [3], "руж", Color.RED, []),
        (["Два Сердца", "Мускат Сухой"], [3], "сухой", None, [SugarClass.DRY]),
        (["Мускат десертный"], [1], "десертный", None, [SugarClass.SWEET]),
        (["КРАСНАЯ"], [0], "красная", Color.RED, []),
    ],
)
def test_exact_catalog_cuvee_stays_next_to_color_and_sugar(lines, ids, cuvee, color, sugar):
    """Каталог называет этим словом саму позицию («Блан» и «Руж» одной винодельни): кюве,
    написанное как в каталоге, остаётся рядом с цветом или сахаром, иначе позицию не отличить
    от соседей по линейке."""
    fields = fields_of(*lines, hits=[(ids, hit(cuvee, "cuvee"))])
    assert [e.value for e in fields.cuvee] == [cuvee]
    assert color_sugar(fields) == (color, sugar)


def test_proper_name_keeps_its_cuvee():
    fields = fields_of("Красная стрелка", hits=[([0], hit("красная", "cuvee"))])
    assert fields.color is None
    assert [e.value for e in fields.cuvee] == ["красная"]


def test_sugar_takes_its_word_from_a_fuzzy_cuvee():
    fields = fields_of("Мускат сухое", hits=[([1], hit("сухой", "cuvee", 1.0))])
    assert [e.value for e in fields.sugar] == [SugarClass.DRY]
    assert fields.cuvee == []


# ------------------------------------------------------------------ слова названия resolve
@pytest.mark.parametrize(
    ("word", "read"),
    [
        ("белый", "БЕЛЫЙ"),
        ("белый", "белая"),
        ("белый", "Вина белые"),
        ("черный", "ЧЁРНЫЙ"),
        ("черный", "черная"),
        ("розовый", "Rozovyy"),
    ],
)
def test_color_name_word_is_found_in_any_gender_but_neuter(word, read):
    assert name_word_found(WordBag.of([read]), word)


@pytest.mark.parametrize(
    ("word", "read"),
    [
        ("белый", "розовое"),
        ("белый", "бельё"),
        ("белый", "вино белое"),  # средний род — цвет вина: его сравнивает группа color
        ("красный", "Красное"),
        ("белый", "vino beloe"),
    ],
)
def test_color_name_word_is_not_found_by_another_color_an_edit_or_the_wine_color(word, read):
    assert not name_word_found(WordBag.of([read]), word)


def test_color_forms_cover_translit_and_leave_foreign_words_alone():
    assert {"белое", "белый", "белая", "белые", "belyy"} <= color_forms("белый")
    assert "черный" in color_forms("chernyj")  # скелет транслита каталога
    assert color_forms("blanc") == {"blanc"}


MUSCAT_RECORDS = [
    wine_record(
        "m-muskat-belyy",
        name="Мускат белый",
        grapes=["muscat"],
        grape_values=["Мускат белый"],
        color="Белое",
    ),
    wine_record(
        "m-muskat-rozovyy",
        name="Мускат розовый",
        grapes=["muscat"],
        grape_values=["Мускат розовый"],
        color="Розовое",
    ),
    wine_record("m-muskatel-belyy", name="Мускатель белый", color="Белое"),
    wine_record("m-muskatel-rozovyy", name="Мускатель розовый", color="Розовое"),
    wine_record("m-muskatel-chernyy", name="Мускатель черный", color="Красное"),
    wine_record("m-portveyn", name="Портвейн - белое креплёное", color="Белое"),
    wine_record("m-kupazh", name="Купаж красный", color="Красное"),
    wine_record("m-sauvignon", name="Sauvignon Blanc", grapes=["sauvignon_blanc"], color="Белое"),
    wine_record("m-kokur", name="Кокур белый", grapes=["kokur"], color="Белое"),
    wine_record(
        "m-tsimlyansky",
        name="Цимлянский чёрный",
        grapes=["tsimlyansky_cherny"],
        color="Красное",
    ),
]


@pytest.fixture(scope="module")
def muscat_catalog() -> CatalogAttrs:
    return CatalogAttrs.from_records(MUSCAT_RECORDS)


@pytest.mark.parametrize(
    ("slug", "words"),
    [
        ("m-muskat-belyy", ("белый",)),  # цвет — последнее слово названия сорта
        ("m-muskat-rozovyy", ("розовый",)),
        ("m-muskatel-belyy", ("мускатель", "белый")),  # цвет в роде стиля — после него
        ("m-muskatel-chernyy", ("мускатель", "черный")),
        ("m-portveyn", ("портвейн",)),  # средний род — цвет вина
        ("m-kupazh", ("купаж",)),  # «красный» после купажа — цвет вина
        ("m-sauvignon", ()),  # «Blanc» — часть сорта, а не слово названия
        # Цвет в конце названия сорта, который в каталоге одного цвета, позицию не отличает:
        # сорт сравнивает группа grape.
        ("m-kokur", ()),
        ("m-tsimlyansky", ()),
    ],
)
def test_color_name_words(muscat_catalog, slug, words):
    assert catalog_stats(muscat_catalog).name_words(muscat_catalog.get(slug)) == words


def rows_for(catalog: CatalogAttrs, slugs, *lines: str) -> dict[str, dict[str, float]]:
    ranked = [(slug, 0.9 - 0.001 * i) for i, slug in enumerate(slugs)]
    read = TextRead(fields_of(*lines), "\n".join(lines))
    features = query_features(cv_record("q", None, ranked), {"": read}, catalog)
    return dict(zip(features.slugs, features.rows, strict=True))


def test_muscatel_colors_differ_by_fields_and_name(muscat_catalog):
    """Выбор между тремя «Мускателями» одной линейки. Это вино есть и в dev, и на публичном
    кадре: строки здесь — общие формы и раскладка dev-снимка, а не чтение публичного кадра, и
    поведение на нём этот тест не подтверждает."""
    slugs = ("m-muskatel-rozovyy", "m-muskatel-chernyy", "m-muskatel-belyy")
    rows = rows_for(muscat_catalog, slugs, "МАССАНДРА", "МУСКАТЕЛЬ", "БЕЛЫЙ")
    white, rose, black = (rows[s] for s in ("m-muskatel-belyy", *slugs[:2]))
    assert white["color_match"] == 1.0
    assert rose["color_conflict"] == 1.0 and black["color_conflict"] == 1.0
    assert white["name_found"] > rose["name_found"] == black["name_found"]
    assert white["name_found_rel"] == 0.0 > rose["name_found_rel"]


@pytest.mark.parametrize(
    ("lines", "winner", "loser"),
    [
        (("МУСКАТ БЕЛЫЙ",), "m-muskat-belyy", "m-muskat-rozovyy"),
        (("МУСКАТ РОЗОВЫЙ",), "m-muskat-rozovyy", "m-muskat-belyy"),
    ],
)
def test_muscat_white_and_rose_differ_by_name_word(muscat_catalog, lines, winner, loser):
    rows = rows_for(muscat_catalog, ("m-muskat-belyy", "m-muskat-rozovyy"), *lines)
    assert rows[winner]["name_found"] > rows[loser]["name_found"]


def test_wine_color_counts_once_by_color_not_again_by_name(muscat_catalog):
    """«вино розовое» решает группа color; слово названия «розовый» им второй раз не найдено."""
    slugs = ("m-muskat-belyy", "m-muskat-rozovyy")
    rows = rows_for(muscat_catalog, slugs, "Мускат", "вино розовое")
    white, rose = rows["m-muskat-belyy"], rows["m-muskat-rozovyy"]
    assert rose["color_match"] == 1.0 and white["color_conflict"] == 1.0
    assert white["name_found"] == rose["name_found"] == 0.0


def test_muscat_white_and_rose_differ_by_label_fields():
    """Сорт таксономии разный, а цвет из названия сорта не берётся ни у того, ни у другого."""
    white, rose = fields_of("МУСКАТ БЕЛЫЙ"), fields_of("МУСКАТ РОЗОВЫЙ")
    assert find_grapes("МУСКАТ БЕЛЫЙ") == ["muscat"]
    assert find_grapes("МУСКАТ РОЗОВЫЙ") == ["muscat_rose"]
    assert white.color is None and rose.color is None
    # С цветом вина на этикетке поля расходятся и цветом.
    assert fields_of("МУСКАТ БЕЛЫЙ", "вино белое").color.value is Color.WHITE
    assert fields_of("МУСКАТ РОЗОВЫЙ", "вино розовое").color.value is Color.ROSE
