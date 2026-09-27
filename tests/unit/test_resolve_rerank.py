"""Переранжирование кандидатов CV полями этикетки на игрушечном каталоге из восьми позиций.

Косинусы здесь задаются руками: проверяется правило решения, а не модель зрения.
"""

from collections.abc import Sequence

import pytest
from catalog import toy_attrs, wine_record

from app.features.contracts import Candidate, VisualResult
from app.reading.contracts import Color, Evidence, LabelFields, SugarClass
from app.resolve.attrs import CatalogAttrs
from app.resolve.rerank import (
    DEFAULT_CONFIG,
    WEIGHTS_NOTE,
    FeatureWeights,
    RerankConfig,
    agreement,
    read_fields,
    rerank,
)

ABSTAIN = DEFAULT_CONFIG.model_copy(update={"abstain": "ooc_only"})


@pytest.fixture
def attrs() -> CatalogAttrs:
    return toy_attrs()


def visual(pairs: Sequence[tuple[str, float]], *, model: str = "fake/pixel") -> VisualResult:
    """Выдача CV из пар (slug, косинус) — уже в порядке убывания."""
    candidates = [
        Candidate(slug=slug, score=score, view="bottle", rank=rank)
        for rank, (slug, score) in enumerate(pairs, start=1)
    ]
    margin = pairs[0][1] - pairs[1][1] if len(pairs) > 1 else 0.0
    return VisualResult(candidates=candidates, margin=max(0.0, margin), model=model)


def label(
    *,
    winery: str | None = None,
    conf: float | None = 1.0,
    cuvee: Sequence[str] = (),
    grapes: Sequence[str] = (),
    sugar: Sequence[SugarClass] = (),
    year: int | None = None,
    serial: Sequence[str] = (),
    abv: float | None = None,
    color: Color | None = None,
) -> LabelFields:
    """Поля этикетки: пусто — «не прочитано», как и у настоящего модуля чтения."""
    return LabelFields(
        producer=[Evidence[str](value=winery, conf=conf)] if winery else [],
        cuvee=[Evidence[str](value=value, conf=conf) for value in cuvee],
        grapes=[Evidence[str](value=value, conf=conf) for value in grapes],
        sugar=[Evidence[SugarClass](value=value, conf=conf) for value in sugar],
        vintage=Evidence[int](value=year, conf=conf) if year else None,
        serial=[Evidence[str](value=value, conf=conf) for value in serial],
        abv=Evidence[float](value=abv, conf=conf) if abv else None,
        color=Evidence[Color](value=color, conf=conf) if color else None,
    )


TWINS = (("dolina-aligote-2023", 0.82), ("dolina-aligote-2024", 0.81), ("dolina-merlot", 0.60))
BRUT_TWINS = (("dolina-reserve-brut", 0.80), ("dolina-polusladkoe", 0.79))


# ------------------------------------------------------------------ пустой текст
@pytest.mark.parametrize("fields", [None, LabelFields()])
def test_empty_fields_keep_visual_order(attrs: CatalogAttrs, fields: LabelFields | None):
    result = rerank(visual(TWINS), fields, attrs)
    assert [slug for slug, _ in result.top5] == [
        "dolina-aligote-2023",
        "dolina-aligote-2024",
        "dolina-merlot",
    ]
    assert result.slug == "dolina-aligote-2023" and result.evidence["text_empty"] is True


def test_empty_fields_leave_twins_ambiguous(attrs: CatalogAttrs):
    # Двойники получают счёт своей серии: без текста отрыва между ними нет.
    result = rerank(visual(TWINS), None, attrs)
    assert result.margin == 0.0 and result.outcome == "ambiguous"


# ------------------------------------------------------------------ двойники
def test_year_lifts_the_right_twin(attrs: CatalogAttrs):
    result = rerank(visual(TWINS), label(year=2024), attrs)
    assert result.slug == "dolina-aligote-2024" and result.outcome == "matched"
    assert result.evidence["features"]["year"]["agree"] == "match"


def test_sugar_lifts_the_right_twin(attrs: CatalogAttrs):
    result = rerank(visual(BRUT_TWINS), label(sugar=[SugarClass.SEMI_SWEET]), attrs)
    assert result.slug == "dolina-polusladkoe"


def test_serial_word_lifts_the_right_twin(attrs: CatalogAttrs):
    # «Riserva» с этикетки и «reserve» каталога — один ключ «резерв».
    order = (("dolina-polusladkoe", 0.80), ("dolina-reserve-brut", 0.79))
    assert rerank(visual(order), label(serial=["Riserva"]), attrs).slug == "dolina-reserve-brut"


def test_roman_serial_lifts_the_right_twin(attrs: CatalogAttrs):
    order = (("dolina-polusladkoe", 0.80), ("dolina-reserve-brut", 0.79))
    assert rerank(visual(order), label(serial=["xxiv"]), attrs).slug == "dolina-reserve-brut"


def test_series_member_outside_top_k_can_win(attrs: CatalogAttrs):
    # CV поднял только один двойник; второй приходит из серии и выигрывает по году.
    result = rerank(visual([("dolina-aligote-2023", 0.82)]), label(year=2024), attrs)
    assert result.slug == "dolina-aligote-2024"
    assert result.evidence["series"]["from_series_pool"] is True


def test_series_pool_off_keeps_pool_to_candidates(attrs: CatalogAttrs):
    cfg = DEFAULT_CONFIG.model_copy(update={"series_pool": False})
    result = rerank(visual([("dolina-aligote-2023", 0.82)]), label(year=2024), attrs, cfg=cfg)
    assert result.slug == "dolina-aligote-2023" and result.evidence["pool"] == 1


def test_top_k_cuts_candidates(attrs: CatalogAttrs):
    cfg = DEFAULT_CONFIG.model_copy(update={"top_k": 1, "series_pool": False})
    result = rerank(visual(TWINS), label(year=2024), attrs, cfg=cfg)
    assert [slug for slug, _ in result.top5] == ["dolina-aligote-2023"]


# ------------------------------------------------------------------ противоречия
def test_winery_conflict_drops_candidate(attrs: CatalogAttrs):
    order = (("drugaya-kaberne", 0.80), ("dolina-merlot", 0.70))
    result = rerank(visual(order), label(winery="Тестовая Долина"), attrs)
    assert result.slug == "dolina-merlot"
    assert result.evidence["runners_up"][0]["conflict"] == ["winery"]


def test_unknown_winery_conflicts_with_nobody(attrs: CatalogAttrs):
    # Винодельни нет в каталоге: это может быть ошибка чтения, а не «не то вино».
    result = rerank(visual(TWINS), label(winery="Неизвестная Винодельня"), attrs)
    assert result.slug == "dolina-aligote-2023"
    assert result.evidence["features"]["winery"]["agree"] == "unknown"


def test_card_without_year_is_not_punished(attrs: CatalogAttrs):
    # У 1 987 позиций каталога года нет: мягкое правило не должно их топить.
    order = (("dolina-merlot", 0.80), ("dolina-aligote-2023", 0.70))
    result = rerank(visual(order), label(year=2019), attrs)
    assert result.slug == "dolina-merlot"
    assert result.evidence["features"]["year"]["agree"] == "unknown"


def test_grape_conflict_and_match(attrs: CatalogAttrs):
    order = (("drugaya-kaberne", 0.80), ("dolina-merlot", 0.75))
    assert rerank(visual(order), label(grapes=["Мерло"]), attrs).slug == "dolina-merlot"


def test_abv_tolerance(attrs: CatalogAttrs):
    order = (("dolina-aligote-2023", 0.80), ("dolina-aligote-2024", 0.80))
    result = rerank(visual(order), label(abv=13.0), attrs)
    assert result.slug == "dolina-aligote-2024"  # 13,0 против 12,5 у соседа


# ------------------------------------------------------------------ счёт и порядок
def test_score_is_visual_plus_weights(attrs: CatalogAttrs):
    cfg = RerankConfig(
        visual_weight=1.0,
        year=FeatureWeights(match=0.5, conflict=-0.5),
        winery=FeatureWeights(),
        cuvee=FeatureWeights(),
        grape=FeatureWeights(),
        sugar=FeatureWeights(),
        serial=FeatureWeights(),
        abv=FeatureWeights(),
    )
    result = rerank(visual(TWINS), label(year=2024), attrs, cfg=cfg)
    # Счёт серии — максимум по её членам (0,82), плюс совпавший год.
    assert result.slug == "dolina-aligote-2024" and result.score == pytest.approx(1.32)
    # Второй в пуле — «Мерло» (0,60 без признаков): двойник с чужим годом упал до 0,32.
    assert result.margin == pytest.approx(0.72)
    assert dict(result.top5)["dolina-aligote-2023"] == pytest.approx(0.32)


def test_margin_and_series_margin(attrs: CatalogAttrs):
    result = rerank(visual(TWINS), label(year=2024), attrs)
    assert result.series_margin == pytest.approx(0.22)  # 0,82 серии против 0,60 «Мерло»
    assert result.margin > 0.0 and result.top5[0][1] > result.top5[1][1]


def test_ambiguous_when_margin_is_small(attrs: CatalogAttrs):
    cfg = DEFAULT_CONFIG.model_copy(update={"ambiguous_margin": 0.5})
    assert rerank(visual(TWINS), label(year=2024), attrs, cfg=cfg).outcome == "ambiguous"


def test_single_candidate_is_not_ambiguous(attrs: CatalogAttrs):
    result = rerank(visual([("dolina-merlot", 0.9)]), None, attrs)
    assert result.outcome == "matched" and result.margin == 0.0


def test_no_candidates_is_out_of_catalog(attrs: CatalogAttrs):
    result = rerank(VisualResult(), label(year=2024), attrs)
    assert result.slug is None and result.outcome == "out_of_catalog" and result.top5 == []
    assert "reason" in result.evidence


def test_unknown_slug_in_candidates_survives(attrs: CatalogAttrs):
    # Индекс собран по каталогу шире разметки: у чужого slug просто нет признаков.
    result = rerank(visual([("нет-в-разметке", 0.9), ("dolina-merlot", 0.5)]), label(), attrs)
    assert result.slug == "нет-в-разметке"
    assert {item["agree"] for item in result.evidence["features"].values()} == {"unknown"}


def test_top5_is_capped(attrs: CatalogAttrs):
    pairs = [(slug, 0.9 - i / 100) for i, slug in enumerate(toy_attrs().by_slug)]
    assert len(rerank(visual(pairs), None, attrs).top5) == 5


def test_result_is_deterministic(attrs: CatalogAttrs):
    fields = label(winery="Тестовая Долина", year=2024, sugar=[SugarClass.DRY])
    first = rerank(visual(TWINS), fields, attrs)
    second = rerank(visual(TWINS), fields, attrs)
    assert first.model_dump() == second.model_dump()


def test_ties_are_broken_by_visual_then_year_then_slug(attrs: CatalogAttrs):
    # Одинаковый счёт у двух двойников: порядок задан косинусом, годом и slug, а не случаем.
    order = (("dolina-aligote-2024", 0.80), ("dolina-aligote-2023", 0.80))
    result = rerank(visual(order), None, attrs)
    assert [slug for slug, _ in result.top5[:2]] == [
        "dolina-aligote-2024",
        "dolina-aligote-2023",
    ]


# ------------------------------------------------------------------ отказ
def test_abstain_fires_on_three_conditions(attrs: CatalogAttrs):
    order = (("drugaya-kaberne", 0.55), ("drugaya-rislling", 0.50))
    fields = label(winery="Другая Винодельня", grapes=["merlot"])
    result = rerank(visual(order), fields, attrs, cfg=ABSTAIN)
    assert result.slug is None and result.outcome == "out_of_catalog"
    assert result.evidence["abstain"]["fired"] is True
    assert result.top5, "top5 остаётся: это «может быть, вы искали», а не ответ"


def test_abstain_is_off_by_default(attrs: CatalogAttrs):
    order = (("drugaya-kaberne", 0.55), ("drugaya-rislling", 0.50))
    fields = label(winery="Другая Винодельня", grapes=["merlot"])
    assert rerank(visual(order), fields, attrs).slug == "drugaya-kaberne"


@pytest.mark.parametrize(
    ("case", "fields_kwargs", "scores"),
    [
        ("винодельня прочитана неуверенно", {"conf": 0.4}, (0.55, 0.50)),
        ("винодельня не прочитана", {"winery": None}, (0.55, 0.50)),
        ("текста нет — сравнивать нечего", {"grapes": ()}, (0.55, 0.50)),
        ("CV уверен в себе", {}, (0.95, 0.50)),
    ],
)
def test_abstain_needs_all_three_conditions(
    attrs: CatalogAttrs, case: str, fields_kwargs: dict, scores: tuple[float, float]
):
    kwargs = {"winery": "Другая Винодельня", "grapes": ["merlot"], **fields_kwargs}
    order = (("drugaya-kaberne", scores[0]), ("drugaya-rislling", scores[1]))
    result = rerank(visual(order), label(**kwargs), attrs, cfg=ABSTAIN)
    assert result.slug is not None, case
    assert result.evidence["abstain"]["fired"] is False, case


def test_abstain_skipped_when_a_position_matches(attrs: CatalogAttrs):
    order = (("drugaya-kaberne", 0.55), ("drugaya-rislling", 0.50))
    fields = label(winery="Другая Винодельня", grapes=["Рислинг"])
    result = rerank(visual(order), fields, attrs, cfg=ABSTAIN)
    assert result.slug == "drugaya-rislling"
    assert result.evidence["abstain"]["no_position_matches"] is False


def test_abstain_forbidden_for_conditional_names(attrs: CatalogAttrs):
    # У «Третьей Марки» единственное название — «Пино Нуар»: по нему «не то вино» не сказать.
    fields = label(winery="Третья Марка", grapes=["merlot"])
    result = rerank(visual([("tretya-pino-nuar", 0.55)]), fields, attrs, cfg=ABSTAIN)
    assert result.slug == "tretya-pino-nuar"
    assert result.evidence["abstain"]["conditional_names"] is True


# ------------------------------------------------------------------ признаки и доказательства
def test_read_fields_maps_label_to_keys(attrs: CatalogAttrs):
    keys = read_fields(
        label(winery="тестовая долина", grapes=["Шардоне"], serial=["Reserve"], abv=12.0), attrs
    )
    assert keys.grapes == frozenset({"chardonnay"}) and keys.serial == frozenset({"резерв"})
    assert "dolina-merlot" in keys.winery_slugs and keys.abv == 12.0
    assert not keys.empty and read_fields(None, attrs).empty


@pytest.mark.parametrize(
    ("feature", "fields_kwargs", "expected"),
    [
        ("winery", {"winery": "Тестовая Долина"}, "match"),
        ("winery", {"winery": "Другая Винодельня"}, "conflict"),
        ("winery", {}, "unknown"),
        ("grape", {"grapes": ["aligote"]}, "match"),
        ("grape", {"grapes": ["merlot"]}, "conflict"),
        ("sugar", {"sugar": [SugarClass.DRY]}, "match"),
        ("sugar", {"sugar": [SugarClass.BRUT]}, "conflict"),
        ("year", {"year": 2023}, "match"),
        ("year", {"year": 2024}, "conflict"),
        ("cuvee", {"cuvee": ["Баррель"]}, "match"),
        ("cuvee", {"cuvee": ["Терруар"]}, "conflict"),
        # У карточки серии нет — «в каталоге не заполнено», а не «серия другая».
        ("serial", {"serial": ["резерв"]}, "unknown"),
        ("abv", {"abv": 12.5}, "match"),
        ("abv", {"abv": 14.0}, "conflict"),
    ],
)
def test_agreement_table(attrs: CatalogAttrs, feature: str, fields_kwargs: dict, expected: str):
    wine = attrs.get("dolina-aligote-2023")
    keys = read_fields(label(**fields_kwargs), attrs)
    assert agreement(feature, wine, keys) == expected  # type: ignore[arg-type]


def test_serial_conflicts_only_when_both_sides_named_it(attrs: CatalogAttrs):
    wine = attrs.get("dolina-reserve-brut")  # серия каталога: «резерв» и «XXIV»
    assert agreement("serial", wine, read_fields(label(serial=["магнум"]), attrs)) == "conflict"
    assert agreement("serial", wine, read_fields(label(serial=["Reserve"]), attrs)) == "match"


def test_agreement_without_card_is_unknown(attrs: CatalogAttrs):
    keys = read_fields(label(winery="Тестовая Долина", year=2024), attrs)
    assert all(agreement(name, None, keys) == "unknown" for name in ("winery", "year"))  # type: ignore[arg-type]


def test_evidence_carries_note_and_series(attrs: CatalogAttrs):
    result = rerank(visual(TWINS), label(year=2024), attrs)
    assert result.evidence["weights_note"] == WEIGHTS_NOTE
    assert result.evidence["series"] == {
        "id": "cluster:1",
        "size": 2,
        "cv_score": 0.82,
        "is_cv_best": True,
        "from_series_pool": False,
    }
    assert result.evidence["visual"]["top1"] == "dolina-aligote-2023"


def test_catalog_without_clusters_degrades_to_visual_order():
    # Разметки нет: серий нет, признаков нет — resolve повторяет порядок CV.
    attrs = CatalogAttrs(())
    result = rerank(visual(TWINS), label(year=2024), attrs)
    assert [slug for slug, _ in result.top5] == [slug for slug, _ in TWINS]


def test_weights_are_explicit_and_documented():
    assert DEFAULT_CONFIG.winery.conflict < 0 < DEFAULT_CONFIG.winery.match
    assert "не подобраны на данных" in WEIGHTS_NOTE
    # Конфиг замораживается: подкрутить вес по дороге нельзя.
    with pytest.raises(ValueError, match="frozen"):
        DEFAULT_CONFIG.top_k = 5


def test_config_from_record_of_one_position():
    attrs = toy_attrs([wine_record("solo", name="Соло", grapes=["merlot"])])
    result = rerank(visual([("solo", 0.7)]), label(grapes=["merlot"]), attrs)
    assert result.slug == "solo" and result.evidence["series"]["size"] == 1
