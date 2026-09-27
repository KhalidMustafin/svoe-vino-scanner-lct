"""`app.recommend`: справочник выгрузки, правила фактов, подбор похожих фактами, импорты."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from reco_env import ORGANIZER, write_reco

from app.recommend.catalog import (
    PORTAL_LINKS_PATH,
    PORTAL_WINE_URL,
    WRONG_PHOTOS,
    Alcohol,
    RecoCatalog,
    alcohol_of,
    card_sugar,
    degrees_label,
    number_label,
    organizer_row,
    organizer_sparkling,
    organizer_sugar,
    portal_url_of,
    slug_abv,
    style_label,
    sugar_of_name,
    sugar_of_slug,
    sweet_description,
    sweet_name,
    unlisted_slugs,
)
from app.recommend.facts import (
    NOTICE,
    Facts,
    explain,
    fewer_note,
    one_per_winery,
    plain,
    similar,
    sugar_close,
    tile,
    with_contrast,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def catalog(tmp_path) -> RecoCatalog:
    write_reco(tmp_path)
    return RecoCatalog.load(
        tmp_path / "gt_tokens.jsonl",
        wines_path=tmp_path / "wines.jsonl",
        groups_path=tmp_path / "wine_groups.json",
    )


# ------------------------------------------------------------------ импорты
def test_package_imports_no_torch_fastapi_or_api():
    """Пакет — чистый Python. Проверка в отдельном процессе: тесты уже тянут FastAPI."""
    code = (
        "import sys\n"
        "import app.recommend, app.recommend.catalog, app.recommend.facts\n"
        "import app.recommend.content_filter, app.recommend.somm_data\n"
        "import app.recommend.profile, app.recommend.build, app.recommend.shelf\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'fastapi',"
        " 'starlette', 'transformers', 'cv2') or m.startswith('app.api'))\n"
        "print(','.join(bad))\n"
    )
    run = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""},
    )
    assert run.stdout.strip() == ""


# ------------------------------------------------------------------ справочник выгрузки
def test_catalog_is_organizer_positions_only(catalog):
    """Живая карточка портала вне выгрузки есть в словаре групп, но в справочник не входит."""
    assert "delta-merlot" not in catalog and len(catalog) == len(ORGANIZER)
    assert catalog.group_members("W-delta") == ()  # в группе только карточки вне выгрузки
    stats = catalog.stats()
    assert stats["wines"] == len(ORGANIZER) and stats["groups_source"] == "wines.jsonl"
    assert not {"portal_cards", "with_portal", "pool_from_csv"} & set(stats)


def test_pool_is_canonical_with_organizer_photo(catalog):
    pool = {wine.slug for wine in catalog.pool}
    assert "eta-merlot-magnum-suhoe" not in pool  # неканоническое
    assert "theta-merlot-suhoe" not in pool  # фото выгрузки нет
    assert {"beta-merlot", "delta-merlot-suhoe", "eta-merlot-suhoe"} <= pool
    assert catalog.group_members("W-eta") == ("eta-merlot-suhoe", "eta-merlot-magnum-suhoe")


def test_rows_follow_organizer_rules():
    """Факты фейков — те же, что дали бы правила выгрузки по названию и slug."""
    for item in ORGANIZER:
        sugar = organizer_sugar(item["title"], item["slug"])
        assert sugar == item["sugar"], item["slug"]
        assert organizer_sparkling(sugar, item["title"], item["slug"]) == item["sparkling"]


def test_group_without_organizer_canonical_gets_one():
    """Каноническая позиция группы в словаре — живая карточка: её место занимает позиция
    выгрузки (так в пул возвращается `merlo-litavshhuk`), лучше — с фото."""
    records = [
        {"slug": "merlo-a", "name": "Мерло", "category": "Красное", "photo_name": ""},
        {"slug": "merlo-b", "name": "Мерло", "category": "Красное", "photo_name": "b.webp"},
    ]
    grouping = [
        {"slug": s, "wine_id": "W1", "canonical": "merlo-live", "is_canonical": False,
         "winery_norm": "x", "grapes": ["merlot"], "grapes_src": "csv"}
        for s in ("merlo-a", "merlo-b")
    ]  # fmt: skip
    catalog = RecoCatalog.build(
        records, grouping, groups={"W1": ["merlo-live", "merlo-a", "merlo-b"]}
    )
    assert [wine.slug for wine in catalog.pool] == ["merlo-b"]
    assert catalog.get("merlo-a").canonical == "merlo-b"
    assert catalog.group_members("W1") == ("merlo-a", "merlo-b")


def test_catalog_without_group_dictionary_is_one_group_per_wine(tmp_path):
    write_reco(tmp_path)
    catalog = RecoCatalog.load(tmp_path / "gt_tokens.jsonl", wines_path=tmp_path / "missing")
    assert len(catalog) == len(ORGANIZER)
    assert catalog.get("eta-merlot-magnum-suhoe").is_canonical  # группы нет — и не магнум
    assert catalog.get("beta-merlot").grapes == ("merlot",)  # коды разметки из выгрузки


def test_wrong_photo_is_not_shown_but_stays_in_pool():
    """Чужая бутылка на фото выгрузки: фото не показывается, а вино остаётся в пуле (силуэт)."""
    slug = next(iter(WRONG_PHOTOS))
    record = {"slug": slug, "name": "X", "photo_name": "x.webp"}
    row = organizer_row(record)
    assert row["photo"] == "x.webp" and row["photo_wrong"]
    assert not organizer_row({**record, "slug": "other"})["photo_wrong"]
    catalog = RecoCatalog.build([record, {**record, "slug": "other"}])
    wine = catalog.get(slug)
    assert wine.photo_wrong and not wine.photo_shown and wine.in_pool
    assert catalog.get("other").photo_shown
    assert {w.slug for w in catalog.pool} == {slug, "other"}
    stats = catalog.stats()
    assert (stats["with_photo"], stats["photo_wrong"]) == (1, 1)


def test_wrong_photos_are_every_wrong_photo_row_of_the_packshot_fix():
    """Правило одно на все 23 строки `wrong_photo` правки эталонов 23.09, а не на часть."""
    table = REPO / "research" / "2026-09-23_packshot-fix" / "replacements.tsv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="	"))
    wrong = {row["slug"] for row in rows if row["issue"] == "wrong_photo"}
    assert len(wrong) == 23 and WRONG_PHOTOS == wrong


def test_group_canonical_prefers_a_shown_photo():
    """Канонической позицией группы вместо живой карточки становится вино с показанным фото."""
    wrong = next(iter(WRONG_PHOTOS))
    records = [
        {"slug": wrong, "name": "Мерло", "category": "Красное", "photo_name": "a.webp"},
        {"slug": "merlo-b", "name": "Мерло", "category": "Красное", "photo_name": "b.webp"},
    ]
    grouping = [
        {"slug": s, "wine_id": "W1", "canonical": "merlo-live", "winery_norm": "x"}
        for s in (wrong, "merlo-b")
    ]
    catalog = RecoCatalog.build(records, grouping, groups={"W1": [wrong, "merlo-b"]})
    assert [wine.slug for wine in catalog.pool] == ["merlo-b"]


def test_portal_url_only_for_published_positions(catalog):
    """Ссылка — только на страницы из карты сайта вин дампа Strapi организатора."""
    assert all(wine.portal_url == PORTAL_WINE_URL + wine.slug for wine in catalog)
    unlisted = unlisted_slugs()
    assert len(unlisted) == 66 and "merlo-litavshhuk" in unlisted
    assert portal_url_of("merlo-litavshhuk") is None
    assert portal_url_of("daniel-22") == PORTAL_WINE_URL + "daniel-22"
    data = json.loads(PORTAL_LINKS_PATH.read_text(encoding="utf-8"))
    assert (data["catalog"], data["published"]) == (2103, 2037)
    assert data["unlisted"] == sorted(unlisted) and data["sitemap_not_in_catalog"] == 0


def test_portal_links_without_the_list_keep_every_link(tmp_path):
    assert unlisted_slugs(tmp_path / "missing.json") == frozenset()


# ------------------------------------------------------------------ правила фактов
@pytest.mark.parametrize(
    ("name", "slug", "expected"),
    [
        ("Brule Saperavi", "fanagoriya-brule-saperavi-saperavi-igristoe-sladkoe-krasnoe-bryut-125",
         "brut"),  # хвост slug «цвет-сахар-крепость»: последнее слово
        ("Мускатель белый", "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16",
         "sladkoe"),
        ("Brut Zero Dosage", "x-beloe-bryut-12", "brut_nature"),  # название главнее slug
        ("Абрау Экстра Брют", "x", "extra_brut"),  # «экстра брют» поглощает «брют»
        ("Совиньон полусухое", "x-krasnoe-suhoe", "polusuhoe"),
        ("x", "abrau-dyurso-imperial-brut-rose-ekstra-bryut-12", "extra_brut"),
        ("Cru Lermont Рислинг", "cru-lermont-risling", None),  # ни в названии, ни в slug
        # опечатка выгрузки: «полсусладкое» — не «сладкое»; сахар берётся из slug
        ("Абрау Купаж красный полсусладкое",
         "abrau-dyurso-abrau-kupazh-krasnyy-polsusladkoe-kaberne-sovinon-krasnoe-polusladkoe-11",
         "polusladkoe"),
    ],
)  # fmt: skip
def test_sugar_rule(name, slug, expected):
    assert organizer_sugar(name, slug) == expected


def test_sugar_parts():
    assert sugar_of_name("Десертное белое") == "sladkoe"
    assert sugar_of_name("Полусладкое") == "polusladkoe" and sugar_of_name("Мерло") is None
    assert sugar_of_slug("x-suhoe-krasnoe-polusuhoe-12") == "polusuhoe"
    assert sugar_of_slug("brutal-wine") is None  # слово целиком, а не часть


def test_sparkling_rule():
    assert organizer_sparkling("brut", "Мерло", "x")
    assert organizer_sparkling(None, "Игристое розовое", "x")
    assert organizer_sparkling(None, "Мерло", "x-igristoe-rozovoe")
    assert organizer_sparkling(None, "Prosecco", "x")
    assert not organizer_sparkling("suhoe", "Мерло", "merlo-krasnoe-suhoe-13")


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Пет-Нат Рубин", "denisov_pet_nat_rubin"),  # дефис в названии, подчёркивание в slug
        ("Пет-Нат Цитрон-Ркацители", "denisov_pet_nat_citron_rkaciteli"),
        ("Пет Нат Соседи, 2024", "x"),
        ("Петнат Рислинг", "x"),
        ("x", "silvaner-pet-nat-2022"),
    ],
)
def test_pet_nat_is_sparkling_in_every_spelling(name, slug):
    assert organizer_sparkling(None, name, slug)


def test_pet_word_inside_another_word_is_not_pet_nat():
    assert not organizer_sparkling(None, "Букет натуральный", "x")
    assert not organizer_sparkling(None, "Сюжет натюрморт", "x")


def test_abv_from_slug_is_not_the_vintage():
    """`daniel-22` у «Daniel, 2022»: хвост slug — год, а не 22 % об."""
    assert slug_abv([22], 2022) == ()
    assert slug_abv([22], None) == (22.0,)
    assert slug_abv([13.5], 2013) == (13.5,)  # дробное — крепость
    assert slug_abv([12, 13], 2012) == (13.0,)
    assert slug_abv([14], 2022) == (14.0,)
    row = organizer_row(
        {"slug": "daniel-22", "name": "Daniel, 2022",
         "fields": {"abv": {"value": [22]}, "year": {"value": 2022}}}
    )  # fmt: skip
    assert row["abv"] == [] and alcohol_of(row["abv"]) == Alcohol()


def test_sweet_name_needs_unknown_sugar():
    assert sweet_name("Портвейн Крымский", None)
    assert sweet_name("Chateau Tamagne Grand Dessert Nectar", None)
    assert sweet_name("Массандра Мускат", None)
    assert not sweet_name("Массандра Мускат", "suhoe")  # сахар выгрузки главнее названия
    assert not sweet_name("Cru Lermont Рислинг", None)
    # Проверка третьего круга 25.09: это догадка. Её снимают название оранжевого вина и «сухое»,
    # «сухость», «полусухое» или «брют» в «Описании»; «сухофрукты» — не сухость.
    assert not sweet_name("Мускат Оранж", None)
    assert not sweet_name("Muscat Orange", None)
    sweetish = "Приятная легкая сладость, несмотря на абсолютную сухость вина"
    assert not sweet_name("Жемчужная 9 Мускат Розовый, Пино гри", None, sweetish)
    assert not sweet_name("Мускат", None, "Вино сухое, с нотами цветов")
    assert not sweet_name("Мускат", None, "Полусухое, с нотами цветов")
    assert not sweet_name("Мускат", None, "Игристое брют")
    assert sweet_name("Мускат", None, "Оттенки розы, цитрусовых, сухофруктов, пряностей")
    assert sweet_name("Массандра Херес", None, "Тона каленого орешка и горького миндаля")
    # Проверка `acc-somm3` 25.09: «Сладкое розовое вино» в «Описании» — сахар по карточке,
    # а не догадка.
    assert not sweet_name("Мускат позднего сбора розовый", None, "Сладкое розовое вино ЗГУ Крым")


#: Начало «Описания» организатора у «Мускат позднего сбора розовый» (выгрузка 25.09).
LATE_MUSCAT = (
    "«Мускат позднего сбора» розовое Сладкое розовое вино ЗГУ Крым Золото Золото 5 Серебро "
    "Серебро 4 Бронза Бронза 5 Все награды Сладкое вино из сорта Мускат розовый – легенда "
    "виноделия Южного берега Крыма."
)


def test_sweet_description_needs_the_word_wine():
    """«Сладкое», «сладкий», «десертное», «десертный» или «ликёрное» целым словом и «вино» не
    дальше чем через два слова. Без «вина» «сладкий» — о вкусе и аромате, у полусладких и брютов
    тоже (выгрузка 25.09); противоречивое описание фактом не считается."""
    assert sweet_description(LATE_MUSCAT)
    for text in (
        "Сладкое вино из сорта Мускат розовый",
        "Это десертное красное крымское вино",
        "Ликёрное вино Массандры",
        "ликерное вино",
    ):
        assert sweet_description(text), text
    for text in (
        "Сладкий, лакрично-черничный аромат развивается в тонкую пыль чёрного перца",
        "Вкус насыщенный, сладкий, с балансом кислотности.",
        "спелый сладкий персик",
        "тонкое сладкое послевкусие",
        "Полусладкое вино с нотами ягод",
        "Это не сладкое вино, а сухое",
        "Одно из моносортовых сладких вин, которыми славится Массандра",
        "Вкус с десертной сладостью винограда",
        "Сладкое вино, несмотря на абсолютную сухость",  # противоречие — не факт
        "",
    ):
        assert not sweet_description(text), text


def test_card_sugar_takes_the_description_last():
    """Сахар по карточке: выгрузка (название, slug) главнее «Описания»; без неё «сладкое вино»
    в «Описании» — «сладкое»."""
    assert card_sugar("suhoe", LATE_MUSCAT) == "suhoe"
    assert card_sugar(None, LATE_MUSCAT) == "sladkoe"
    assert card_sugar(None, "Ароматы груши") is None
    assert card_sugar(None, None) is None


def test_style_label_rule():
    assert style_label("Красное", "suhoe", False) == "Красное сухое"
    assert style_label("Белое", None, True) == "Белое игристое"
    assert style_label("Белое", "brut", True) == "Белое брют"
    assert style_label("Розовое", None, False) == "Розовое"


# ------------------------------------------------------------------ крепость и подписи
@pytest.mark.parametrize(
    ("abv", "expected"),
    [
        ((13.5,), Alcohol(13.5, None, "catalog")),
        ((10.5, 12.5), Alcohol(10.5, 12.5, "catalog")),  # диапазон
        ((12.0, 12.0), Alcohol(12.0, None, "catalog")),
        ((), Alcohol()),
        ((99.0,), Alcohol()),  # вне 3–25 — неизвестна
        ((135.0, 13.5), Alcohol(13.5, None, "catalog")),
    ],
)
def test_alcohol_rule(abv, expected):
    assert alcohol_of(abv) == expected


def test_labels_without_percent():
    assert number_label(13.5) == "13,5" and number_label(14.0) == "14"
    assert degrees_label(Alcohol(13.5)) == "13,5°"
    assert degrees_label(Alcohol(10.5, 12.5)) == "10,5–12,5°"


# ------------------------------------------------------------------ подбор
def test_sugar_close_is_one_step():
    assert sugar_close("suhoe", "polusuhoe") and sugar_close("brut", "suhoe")
    assert not sugar_close("suhoe", "polusladkoe") and not sugar_close("suhoe", "sladkoe")
    assert sugar_close(None, "sladkoe")


def test_similar_excludes_own_winery_group_and_out_of_pool(catalog):
    anchor = catalog.get("eta-merlot-suhoe")
    picks = similar(catalog, Facts.of_wine(anchor, catalog.alcohol(anchor.slug)), 12)
    slugs = [pick.wine.slug for pick in picks.picks]
    assert "eta-merlot-suhoe" not in slugs and "eta-merlot-magnum-suhoe" not in slugs
    assert "theta-merlot-suhoe" not in slugs and "iota-merlot-sladkoe" not in slugs
    assert all(catalog.get(slug).color == "Красное" for slug in slugs)


def test_unknown_sugar_does_not_narrow(catalog):
    """У Бета Мерло сахара нет ни в названии, ни в slug: сладкое Йоты тоже кандидат."""
    anchor = catalog.get("beta-merlot")
    assert anchor.sugar is None and anchor.style_label == "Красное"
    picks = similar(catalog, Facts.of_wine(anchor, catalog.alcohol(anchor.slug)), 12)
    assert "iota-merlot-sladkoe" in [pick.wine.slug for pick in picks.picks]


def test_one_per_winery_then_fill(catalog):
    wines = [catalog.get(s) for s in ("beta-merlot", "beta-shardone", "eta-merlot-suhoe")]
    assert [w.slug for w in one_per_winery(wines, 2)] == ["beta-merlot", "eta-merlot-suhoe"]
    assert [w.slug for w in one_per_winery(wines, 3)] == [
        "beta-merlot",
        "eta-merlot-suhoe",
        "beta-shardone",
    ]


def test_contrast_replaces_last_when_all_the_same(catalog):
    anchor = catalog.get("delta-merlot-suhoe")
    facts = Facts.of_wine(anchor, catalog.alcohol(anchor.slug))
    same = [catalog.get("eta-merlot-suhoe"), catalog.get("eta-merlot-suhoe")]
    ranked = [*same, catalog.get("zeta-merlot-polusuhoe")]
    assert [w.slug for w in with_contrast(catalog, facts, ranked, same)] == [
        "eta-merlot-suhoe",
        "zeta-merlot-polusuhoe",
    ]


def test_explain_is_facts_only(catalog):
    anchor = catalog.get("delta-merlot-suhoe")
    facts = Facts.of_wine(anchor, catalog.alcohol(anchor.slug))
    epsilon = catalog.get("epsilon-blend-suhoe")
    assert explain(facts, epsilon, catalog.alcohol(epsilon.slug)) == [
        "Общий сорт — Мерло",
        "Тоже сухое",
        "Крым, а не Кубань",
        "Крепость 14° против 13,5°",
    ]
    saperavi = catalog.get("gamma-saperavi")
    assert (
        explain(facts, saperavi, catalog.alcohol(saperavi.slug))[0] == "Сорт — Саперави, а не Мерло"
    )


def test_plain_is_category_by_name(catalog):
    anchor = catalog.get("delta-merlot-suhoe")
    body = plain(catalog, Facts.of_wine(anchor, catalog.alcohol(anchor.slug)), 3)
    names = [pick.wine.title for pick in body.picks]
    assert names == ["Эпсилон Купаж", "Эта Мерло"]  # красное сухое; сахар неизвестен — нет
    assert all(pick.reasons == () for pick in body.picks)


def test_tile_has_link_and_organizer_style(catalog):
    body = tile(catalog.get("kappa-brut"), ["Тоже игристое брют"], photo_url=None)
    assert body["portal_url"] == PORTAL_WINE_URL + "kappa-brut"
    assert (body["style_label"], body["sparkling"]) == ("Белое брют", True)


def test_fewer_note_texts():
    assert fewer_note(2, 3).text == "Нашлось меньше трёх"
    assert fewer_note(0, 3).text == "Похожих в каталоге не нашлось"
    assert fewer_note(1, 12).text == "Нашлось меньше двенадцати"
    assert NOTICE == "Применяются рекомендательные технологии"


def test_facts_without_style():
    assert not Facts().has_style
    assert Facts(sparkling=True).has_style and Facts(color="Белое").has_style
