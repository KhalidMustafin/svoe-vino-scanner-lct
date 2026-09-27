"""Признаки пар для обучаемого resolve: только вход, нормализация внутри запроса, слова названия."""

import copy
import itertools

import numpy as np
import pytest
from catalog import wine_record
from learned_synth import SYNTH_SLUGS, cv_record, label_fields, synth_attrs

from app.features.contracts import Candidate, VisualResult
from app.reading.contracts import Color, Evidence, LabelFields, SugarClass
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import (
    DEFAULT_GROUPS,
    FEATURE_GROUPS,
    FeatureOptions,
    TextRead,
    WordBag,
    catalog_stats,
    cv_query,
    feature_names,
    feature_sign,
    group_of,
    pair_features,
    query_features,
    readers_of,
)


@pytest.fixture(scope="module")
def attrs() -> CatalogAttrs:
    return synth_attrs()


MUSKAT = (("alfa-muskat", 0.91), ("alfa-muskat-chernyj", 0.90), ("beta-merlot", 0.70))
YUZHNAYA = (("alfa-yuzhnaya", 0.93), ("alfa-yuzhnaya-premium", 0.925), ("gamma-saperavi", 0.6))


def rows_by_slug(features) -> dict[str, dict[str, float]]:
    return dict(zip(features.slugs, features.rows, strict=True))


def name_part(row: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in row.items() if group_of(k) == "name"}


# ------------------------------------------------------------------ имена и группы
def test_rows_carry_exactly_the_declared_names(attrs: CatalogAttrs):
    record = cv_record("q1", "alfa-muskat", MUSKAT)
    features = query_features(record, {"": TextRead(label_fields("Альфа Долина"))}, attrs)
    names = feature_names([""], DEFAULT_GROUPS)
    assert set(features.rows[0]) == set(names)
    assert "unpublished" not in names  # слабый признак выключен по умолчанию
    assert all(group_of(name) in FEATURE_GROUPS for name in names)


def test_text_features_are_prefixed_per_reader(attrs: CatalogAttrs):
    record = cv_record("q1", "alfa-muskat", MUSKAT)
    reads = {"vlm35": TextRead(label_fields("Альфа Долина")), "rapid": TextRead(None)}
    row = query_features(record, reads, attrs).rows[0]
    assert set(row) == set(feature_names(["rapid", "vlm35"]))
    assert row["vlm35.winery_match"] == 1.0
    assert row["rapid.winery_match"] == row["rapid.winery_conflict"] == 0.0
    assert "cv_score" in row and "vlm35.cv_score" not in row
    assert readers_of(feature_names(["rapid", "vlm35"])) == ("rapid", "vlm35")
    assert readers_of(feature_names([], ["cv", "cv_group"])) == ()


def test_no_feature_is_constant_within_every_query_and_no_column_repeats(attrs: CatalogAttrs):
    """Счёт линейный, ответ — argmax в запросе: признак, одинаковый у всех кандидатов любого
    запроса, ни на что не влияет, а два одинаковых столбца L2 делит поровну."""
    names = feature_names([""], DEFAULT_GROUPS)
    reads = [
        TextRead(label_fields("Альфа Долина", "Мускат"), "АЛЬФА ДОЛИНА\nМУСКАТ ЧЕРНЫЙ"),
        TextRead(label_fields("Бета Холмы"), "БЕТА ХОЛМЫ\nТЕРРУАР"),
        TextRead(None, "ЮЖНАЯ ВЕРТИКАЛЬ ПРЕМИУМ"),
        TextRead(label_fields("Альфа Долина"), "БЕТА ХОЛМЫ\nТЕРРУАР"),  # поле ≠ сырой текст
        TextRead(None, "АЛЬФА"),  # половина названия винодельни
        TextRead(LabelFields(abv=Evidence[float](value=13.5))),
        TextRead(
            LabelFields(
                producer=[Evidence[str](value="Альфа Долина", conf=0.7)],
                cuvee=[Evidence[str](value="терруар")],
                sugar=[Evidence[SugarClass](value=SugarClass.DRY)],
                vintage=Evidence[int](value=2023),
                serial=[Evidence[str](value="премиум")],
                abv=Evidence[float](value=12.5),
                color=Evidence[Color](value=Color.RED),
            )
        ),
        TextRead(None),
    ]
    rng = np.random.default_rng(0)
    blocks = []
    for i, read in enumerate(itertools.islice(itertools.cycle(reads), 40)):
        order = rng.permutation(len(SYNTH_SLUGS))[:6]
        scores = np.sort(rng.uniform(0.5, 0.9, size=6))[::-1]
        ranked = [(SYNTH_SLUGS[j], float(s)) for j, s in zip(order, scores, strict=True)]
        blocks.append(query_features(cv_record(f"q{i}", None, ranked), {"": read}, attrs))
    varies = {name: any(np.ptp(block.matrix([name])) > 0 for block in blocks) for name in names}
    assert [name for name, ok in varies.items() if not ok] == []
    X = np.vstack([block.matrix(names) for block in blocks])
    repeats = [
        (a, b)
        for i, a in enumerate(names)
        for j, b in enumerate(names)
        if i < j and np.array_equal(X[:, i], X[:, j])
    ]
    assert repeats == []


def test_signs_follow_the_meaning_of_the_feature():
    assert feature_sign("vlm35.grape_match") == 1 and feature_sign("sugar_conflict") == -1
    assert feature_sign("winery_match_conf") == 1 and feature_sign("winery_conflict_conf") == -1
    assert feature_sign("r.name_found") == 1 and feature_sign("r.name_found_rel") == 1
    assert feature_sign("cv_score") == 0 and feature_sign("vgroup_mate_of_top1") == 0
    with pytest.raises(KeyError):
        feature_sign("r.sugar_unknown")  # столбца «не прочитано» больше нет


# ------------------------------------------------------------------ без знания ответа
def test_target_slug_in_record_does_not_change_features(attrs: CatalogAttrs):
    fields = label_fields("Альфа Долина", "Мускат")
    record = cv_record("q1", "alfa-muskat", MUSKAT)
    swapped = copy.deepcopy(record)
    swapped["slug"] = "beta-merlot"
    swapped["hit_rank"] = 3
    reads = {"r": TextRead(fields, "АЛЬФА ДОЛИНА\nМУСКАТ")}
    assert query_features(record, reads, attrs) == query_features(swapped, reads, attrs)


def test_pair_features_reject_candidate_outside_top_k(attrs: CatalogAttrs):
    with pytest.raises(KeyError):
        pair_features(cv_record("q1", None, MUSKAT), None, "gamma-rkatsiteli", attrs)


# ------------------------------------------------------------------ CV
def test_cv_features_are_relative_to_the_same_query(attrs: CatalogAttrs):
    shifted = tuple((slug, score - 0.2) for slug, score in MUSKAT)
    base = rows_by_slug(query_features(cv_record("q", None, MUSKAT), {}, attrs))
    moved = rows_by_slug(query_features(cv_record("q", None, shifted), {}, attrs))
    second = "alfa-muskat-chernyj"
    assert base[second]["cv_gap_top1"] == pytest.approx(-0.01)
    for name in ("cv_gap_top1", "cv_z", "cv_minmax", "cv_rank_inv", "cv_is_top1"):
        assert base[second][name] == pytest.approx(moved[second][name])
    assert base[second]["cv_score"] != pytest.approx(moved[second]["cv_score"])
    assert base["alfa-muskat"]["cv_is_top1"] == 1.0 and base["alfa-muskat"]["cv_minmax"] == 1.0
    assert base["alfa-muskat"]["cv_top1_x_margin"] == pytest.approx(0.01)
    assert base[second]["cv_top1_x_margin"] == 0.0


def test_visual_group_features(attrs: CatalogAttrs):
    ranked = (("alfa-riesling-2023", 0.9), ("alfa-riesling-2024", 0.9), ("alfa-muskat", 0.7))
    rows = rows_by_slug(query_features(cv_record("q", None, ranked), {}, attrs))
    assert rows["alfa-riesling-2024"]["vgroup_mate_of_top1"] == 1.0
    assert rows["alfa-riesling-2023"]["vgroup_mate_of_top1"] == 0.0  # сам top-1 себе не двойник
    assert rows["alfa-muskat"]["vgroup_mate_of_top1"] == 0.0
    assert rows["alfa-muskat"]["cluster_mate_of_top1"] == 0.0
    assert rows["alfa-riesling-2024"]["vgroup_size_log"] > rows["alfa-muskat"]["vgroup_size_log"]


def test_cv_query_accepts_visual_result_and_drops_repeats():
    visual = VisualResult(
        candidates=[
            Candidate(slug="a", score=0.9, view="full", rank=1),
            Candidate(slug="a", score=0.8, view="band", rank=2),
            Candidate(slug="b", score=0.7, view="full", rank=3),
            Candidate(slug="c", score=0.6, view="full", rank=4),
        ],
        margin=0.05,
    )
    query = cv_query(visual, top_k=2)
    assert query.slugs == ("a", "b") and [c.rank for c in query.candidates] == [1, 2]
    assert query.margin == 0.05 and query.top1_tie == 1
    assert cv_query([{"slug": "x", "score": 0.5}, {"slug": "y", "score": 0.4}]).margin == (
        pytest.approx(0.1)
    )


def test_top1_tie_counts_candidates_sharing_the_top_score_to_the_bit():
    tied = cv_query([{"slug": s, "score": v} for s, v in (("a", 0.9), ("b", 0.9), ("c", 0.8))])
    assert tied.top1_tie == 2
    near = cv_query([{"slug": "a", "score": 0.9}, {"slug": "b", "score": 0.8999999}])
    assert near.top1_tie == 1 and cv_query([]).top1_tie == 0


# ------------------------------------------------------------------ текст
def test_agreement_pairs_and_confidence(attrs: CatalogAttrs):
    fields = LabelFields(
        producer=[Evidence[str](value="Бета Холмы", conf=0.5)],
        color=Evidence[Color](value=Color.RED),
        vintage=Evidence[int](value=2024),
    )
    ranked = (("alfa-riesling-2023", 0.9), ("beta-merlot", 0.85), ("alfa-muskat", 0.8))
    rows = rows_by_slug(query_features(cv_record("q", None, ranked), {"": fields}, attrs))
    merlot, riesling, muskat = rows["beta-merlot"], rows["alfa-riesling-2023"], rows["alfa-muskat"]
    assert merlot["winery_match"] == 1.0 and merlot["winery_match_conf"] == 0.5
    assert riesling["winery_conflict"] == 1.0 and riesling["winery_conflict_conf"] == 0.5
    assert merlot["color_match"] == 1.0 and muskat["color_conflict"] == 1.0
    # Год сравнивается, только если он есть у карточки; «не прочитано» — оба нуля.
    assert riesling["year_conflict"] == 1.0
    assert merlot["year_match"] == merlot["year_conflict"] == 0.0
    for row in rows.values():
        assert row["grape_match"] == row["grape_conflict"] == 0.0  # сорт не прочитан


def test_unread_name_words_are_not_penalized(attrs: CatalogAttrs):
    """Случай «Совиньон Блан. Авторская технология. Моно» → «Совиньон Блан»: читатель не прочёл
    лишних слов верного длинного названия — и за это его больше не штрафуют."""
    read = TextRead(label_fields("Альфа Долина", "Мускат"), "АЛЬФА ДОЛИНА\nMUSCAT\nWHITE DRY")
    rows = rows_by_slug(query_features(cv_record("q", None, MUSKAT), {"": read}, attrs))
    assert catalog_stats(attrs).name_words(attrs.get("alfa-muskat-chernyj")) == ("черный",)
    assert name_part(rows["alfa-muskat"]) == name_part(rows["alfa-muskat-chernyj"])
    assert all(value == 0.0 for value in name_part(rows["alfa-muskat-chernyj"]).values())


def test_read_name_word_lifts_its_position(attrs: CatalogAttrs):
    read = TextRead(label_fields("Альфа Долина", "Мускат"), "АЛЬФА ДОЛИНА\nМУСКАТ ЧЁРНЫЙ")
    rows = rows_by_slug(query_features(cv_record("q", None, MUSKAT), {"": read}, attrs))
    black, plain = rows["alfa-muskat-chernyj"], rows["alfa-muskat"]
    assert black["name_found"] == pytest.approx(1 / 3) and black["name_found_rel"] == 0.0
    assert plain["name_found"] == 0.0 and plain["name_found_rel"] == pytest.approx(-1 / 3)


def test_read_name_word_is_found_across_scripts(attrs: CatalogAttrs):
    read = TextRead(None, "ЮЖНАЯ\nВЕРТИКАЛЬ\nPREMIUM")
    rows = rows_by_slug(query_features(cv_record("q", None, YUZHNAYA), {"": read}, attrs))
    premium, plain = rows["alfa-yuzhnaya-premium"], rows["alfa-yuzhnaya"]
    assert premium["name_found"] == 1.0 and premium["name_found_rel"] == 0.0
    # Обе нашли все свои слова, но у премиальной найдено на одно больше.
    assert plain["name_found"] == pytest.approx(2 / 3)
    assert plain["name_found_rel"] == pytest.approx(-1 / 3)
    # Без «Премиум» в прочитанном позиции равны: непрочитанное слово ничего не решает.
    read = TextRead(None, "ЮЖНАЯ ВЕРТИКАЛЬ")
    rows = rows_by_slug(query_features(cv_record("q", None, YUZHNAYA), {"": read}, attrs))
    assert name_part(rows["alfa-yuzhnaya-premium"]) == name_part(rows["alfa-yuzhnaya"])


def test_word_bag_matches_translit_and_one_typo():
    bag = WordBag.of(["APOLLINARI", "Premium"])
    assert bag.has("apollinary") and bag.has("премиум")
    assert not bag.has("терруар") and not WordBag.of([])
    # Слово цвета ищется без правки: «красные» не находится в «красное» с этикетки.
    label = WordBag.of(["Красное сухое"])
    assert label.has("красные") and not label.has("красные", fuzzy=False)


def test_empty_text_gives_no_name_or_agreement_signal(attrs: CatalogAttrs):
    rows = query_features(cv_record("q", None, MUSKAT), {"": TextRead()}, attrs).rows
    text = [name for name in feature_names([""]) if group_of(name) not in ("cv", "cv_group")]
    for row in rows:
        assert all(row[name] == 0.0 for name in text)


def test_winery_text_uses_raw_words_not_corrected_field(attrs: CatalogAttrs):
    # Поле винодельни «поправлено» словарём в Бету, а в сыром тексте — Альфа.
    read = TextRead(label_fields("Бета Холмы"), "АЛЬФА ДОЛИНА\nМУСКАТ")
    rows = rows_by_slug(query_features(cv_record("q", None, MUSKAT), {"": read}, attrs))
    assert rows["alfa-muskat"]["winery_conflict"] == 1.0
    assert rows["alfa-muskat"]["winery_text_share"] == 1.0
    assert rows["beta-merlot"]["winery_match"] == 1.0
    assert rows["beta-merlot"]["winery_text_share"] == 0.0


def test_published_feature_only_when_enabled(attrs: CatalogAttrs):
    record = cv_record("q", None, MUSKAT)
    options = FeatureOptions(published=True, unpublished=frozenset({"beta-merlot"}))
    rows = rows_by_slug(query_features(record, {}, attrs, options=options))
    assert rows["beta-merlot"]["unpublished"] == 1.0 and rows["alfa-muskat"]["unpublished"] == 0.0
    assert "unpublished" not in query_features(record, {}, attrs).rows[0]


# ------------------------------------------------------------------ слова названия
MERKOTAN = {
    "winery": "Усадьба Меркотан",
    "key_tokens": ("меркотан",),
    "variants": ("merkotan",),
    "brands": ("ДНК Аллели", "ДНК Аура", "ДНК Поколение"),
}
NAME_RECORDS = [
    wine_record("dva-muskat-suhoy", name="Мускат Сухой", grapes=["muscat"], sugar="suhoe"),
    wine_record("dva-muskat", name="Мускат", grapes=["muscat"], sugar="suhoe"),
    wine_record("abrau-kupazh", name="Абрау Купаж красный полсусладкое", color="Красное"),
    wine_record("gunko-gliny", name="Красные Глины", grapes=["cabernet_franc"]),
    wine_record("agora-muskat-chernyj", name="Мускат Черный", grapes=["muscat"]),
    wine_record("merkotan-alleli", **MERKOTAN, name="ДНК. Аллели"),
    wine_record("merkotan-aura", **MERKOTAN, name="ДНК. Аура"),
    wine_record("merkotan-latin", **MERKOTAN, name="Merkotan Поколение"),
    wine_record("muskat-igristyy", name="Мускат игристый", grapes=["muscat"]),
    wine_record("shardone-rezerv", name="Шардоне Резерв", grapes=["chardonnay"]),
    wine_record("chardonnay-reserve", name="Chardonnay Reserve Premium", grapes=["chardonnay"]),
]


@pytest.fixture(scope="module")
def name_stats():
    catalog = CatalogAttrs.from_records(NAME_RECORDS)
    stats = catalog_stats(catalog)
    return lambda slug: stats.name_words(catalog.get(slug))


def test_sugar_and_type_words_in_any_gender_are_not_name_words(name_stats):
    assert name_stats("dva-muskat-suhoy") == () == name_stats("dva-muskat")
    assert name_stats("muskat-igristyy") == ()
    # «полсусладкое» — опечатка каталога в сахаре; «красный» после купажа — цвет вина.
    assert name_stats("abrau-kupazh") == ("абрау", "купаж")


def test_color_is_a_name_word_right_after_the_grape_not_at_the_start(name_stats):
    assert name_stats("agora-muskat-chernyj") == ("черный",)
    assert name_stats("gunko-gliny") == ("глины",)


def test_dictionary_serial_is_left_to_the_serial_group(name_stats):
    """«RESERVE» на этикетке не находит «резерв» названия (две правки), а серию и так сравнивает
    группа serial — в словах названия её нет; «Premium» — не серия словаря и остаётся."""
    assert name_stats("shardone-rezerv") == ()
    assert name_stats("chardonnay-reserve") == ("premium",)


def test_line_brands_stay_name_words_winery_name_does_not(name_stats):
    assert name_stats("merkotan-alleli") == ("днк", "аллели")
    assert name_stats("merkotan-aura") == ("днк", "аура")
    assert name_stats("merkotan-latin") == ("поколение",)  # латиница винодельни — тоже она


def test_sugar_on_the_label_does_not_find_a_sugar_word_of_the_name():
    """«Мускат» → «Мускат Сухой»: «Сухое» с этикетки больше не находит «сухой» названия."""
    catalog = CatalogAttrs.from_records(NAME_RECORDS)
    ranked = (("dva-muskat", 0.9), ("dva-muskat-suhoy", 0.89), ("gunko-gliny", 0.6))
    read = TextRead(None, "МУСКАТ\nСухое белое\nКрасное")
    rows = rows_by_slug(query_features(cv_record("q", None, ranked), {"": read}, catalog))
    assert name_part(rows["dva-muskat"]) == name_part(rows["dva-muskat-suhoy"])
    # «Красные» в начале названия — не отличие позиции, а слово цвета: словом названия оно не
    # стало и через «Красное» с этикетки не находится.
    assert rows["gunko-gliny"]["name_found"] == 0.0


def test_white_and_orange_are_neither_match_nor_conflict():
    """Э4: «Белое» и «Оранжевое» не спорят, но и не совпадают — оба признака цвета нули."""
    attrs = CatalogAttrs.from_records(
        [
            wine_record("oranzh", name="Ркацители Оранж", color="Оранжевое"),
            wine_record("beloe", name="Ркацители", color="Белое"),
            wine_record("krasnoe", name="Саперави", color="Красное"),
        ]
    )
    ranked = (("oranzh", 0.9), ("beloe", 0.85), ("krasnoe", 0.8))
    for read, same in ((Color.WHITE, "beloe"), (Color.ORANGE, "oranzh")):
        fields = LabelFields(color=Evidence[Color](value=read))
        rows = rows_by_slug(query_features(cv_record("q", None, ranked), {"": fields}, attrs))
        other = "oranzh" if same == "beloe" else "beloe"
        assert rows[same]["color_match"] == 1.0 and rows[same]["color_conflict"] == 0.0
        assert rows[other]["color_match"] == rows[other]["color_conflict"] == 0.0
        assert rows["krasnoe"]["color_conflict"] == 1.0
