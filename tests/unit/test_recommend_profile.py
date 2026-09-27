"""Профиль стиля (`app.recommend.profile`, `build`): перенос `build_profile` «Лозы» и источники осей.

Золотая фикстура — 30 вин join CSV «Лозы» (`tests/fixtures/profile_golden.json`): slug, входы
профиля и оси `derived_*`. Весь join CSV (2 103 вина) сверяет
`research/2026-09-24_after/golden_profile.py`; сам CSV в репозиторий не входит.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reco_env import write_reco

from app.reading.contracts import Color, SugarClass
from app.recommend.build import (
    PRIORS_PATH,
    age_shift,
    build_profile,
    load_priors,
    oak_from_name,
    profile_of,
    profiles_of,
)
from app.recommend.catalog import Alcohol, RecoCatalog, wine_of
from app.recommend.profile import (
    AXES,
    CATALOG,
    GRAPE,
    STYLE,
    STYLE_AXES,
    RussianPGI,
    StylePriors,
    StyleProfile,
    WineKind,
    abv_to_alcohol_axis,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "profile_golden.json"


@pytest.fixture(scope="module")
def priors():
    return load_priors()


def golden() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))["wines"]


def test_priors_file_ships_with_the_code(priors):
    assert PRIORS_PATH.parent.name == "recommend"
    assert len(priors) == 53
    merlot = priors["merlot"]
    assert set(merlot.axes) == {"tannin", "body", "acidity", "aroma_intensity", "oak"}
    assert merlot.descriptors


def test_golden_fixture_has_no_join_secrets():
    """В фикстуре только slug, входы профиля и оси: ни loza_id, ни балла, ни цены."""
    keys = {key for wine in golden() for key in wine}
    assert keys == {"slug", "grapes", "color", "sugar", "kind", "region", "name", "vintage",
                    "basis", "axes"}  # fmt: skip
    text = GOLDEN.read_text(encoding="utf-8")
    for word in ("loza_id", "expert", "price", "rub", "score"):
        assert word not in text


@pytest.mark.parametrize("wine", golden(), ids=lambda wine: wine["slug"][:40])
def test_port_matches_loza_derived_axes(wine, priors):
    profile = build_profile(
        wine["grapes"],
        Color(wine["color"]) if wine["color"] else None,
        SugarClass(wine["sugar"]) if wine["sugar"] else None,
        WineKind(wine["kind"]),
        priors,
        region=RussianPGI(wine["region"]),
        name=wine["name"],
        vintage=wine["vintage"],
    )
    for axis in AXES:
        assert profile.axis(axis) == pytest.approx(wine["axes"][axis], abs=0.05), axis
    assert profile.basis == wine["basis"]


def test_golden_fixture_covers_the_branches():
    wines = golden()
    assert len(wines) == 30
    assert {w["basis"] for w in wines} == {"high", "medium", "low"}
    assert {w["kind"] for w in wines} == {"still", "sparkling", "fortified"}
    assert {w["color"] for w in wines} == {"Белое", "Красное", "Розовое", "Оранжевое"}
    assert any(w["vintage"] for w in wines) and any(w["sugar"] is None for w in wines)


# ------------------------------------------------------------------ источники осей
def test_sources_catalog_vs_grape(priors):
    profile = build_profile(
        ["merlot"], Color.RED, SugarClass.DRY, WineKind.STILL, priors,
        region=RussianPGI.KUBAN, abv=13.5,
    )  # fmt: skip
    assert {axis: profile.source(axis) for axis in AXES} == {
        "sweetness": CATALOG,
        "acidity": GRAPE,
        "tannin": GRAPE,
        "body": GRAPE,
        "alcohol": CATALOG,
        "oak": GRAPE,
        "aroma_intensity": GRAPE,
        "effervescence": CATALOG,
    }
    assert profile.alcohol == pytest.approx(abv_to_alcohol_axis(13.5))
    assert profile.basis == "high"


def test_unknown_facts_have_no_source(priors):
    """Нет сорта в приорах, сахара и крепости — оси есть (как у «Лозы»), источника нет."""
    profile = build_profile(["no_such_grape"], Color.WHITE, None, WineKind.STILL, priors)
    assert profile.basis == "low"
    assert profile.source("acidity") is None and profile.source("tannin") is None
    assert profile.source("sweetness") is None and profile.sweetness == pytest.approx(0.4)
    assert profile.source("alcohol") is None and profile.alcohol == 3.0
    assert profile.source("effervescence") == CATALOG and profile.effervescence == 0.0


def test_loza_rules(priors):
    red = build_profile(["syrah"], Color.RED, SugarClass.DRY, WineKind.STILL, priors)
    white = build_profile(["syrah"], Color.WHITE, SugarClass.DRY, WineKind.STILL, priors)
    rose = build_profile(["syrah"], Color.ROSE, SugarClass.DRY, WineKind.STILL, priors)
    assert red.tannin > 1 and white.tannin <= 0.3 and rose.tannin <= 0.6
    fortified = build_profile([], Color.RED, SugarClass.SWEET, WineKind.FORTIFIED, priors)
    assert fortified.alcohol == 4.6 and fortified.sweetness == 4.5
    sparkling = build_profile([], Color.WHITE, SugarClass.BRUT, WineKind.SPARKLING, priors)
    assert sparkling.effervescence == 4.0 and sparkling.sweetness == 0.7
    assert oak_from_name("Каберне Резерв") == 1.2 and oak_from_name("Молодое вино") == 0.2
    assert oak_from_name("Мерло") is None
    assert age_shift(2025) == {} and age_shift(2014)["acidity"] == pytest.approx(-0.25)
    don = build_profile(["riesling"], Color.WHITE, SugarClass.DRY, WineKind.STILL, priors,
                        region=RussianPGI.DON_VALLEY)  # fmt: skip
    dagestan = build_profile(["riesling"], Color.WHITE, SugarClass.DRY, WineKind.STILL, priors,
                             region=RussianPGI.DAGESTAN)  # fmt: skip
    assert don.acidity - dagestan.acidity == pytest.approx(0.65)


def test_profile_rejects_out_of_scale():
    with pytest.raises(ValueError):
        StyleProfile(acidity=5.5)


# ------------------------------------------------------------------ по справочнику
def test_profile_of_catalog_wine_takes_abv_and_year(priors):
    from reco_env import row

    wine = wine_of(
        {**row("x", "Икс", title="Мерло Резерв 2015", grapes=["merlot"], color="Красное",
               abv=14.0), "year": 2015},
    )  # fmt: skip
    assert wine.year == 2015
    profile = profile_of(wine, priors, alcohol=Alcohol(14.0, None, "catalog"))
    assert profile.alcohol == pytest.approx(abv_to_alcohol_axis(14.0))
    assert profile.source("alcohol") == CATALOG
    assert profile.oak == pytest.approx(max(priors["merlot"].axes["oak"], 1.2))
    no_abv = profile_of(wine, priors)
    assert no_abv.source("alcohol") is None
    younger = profile_of(wine_of({**row("y", "Игрек", title="Мерло Резерв", grapes=["merlot"],
                                        color="Красное")}), priors)  # fmt: skip
    assert profile.acidity < younger.acidity  # год урожая 2015 — минус кислотность


def test_profiles_of_whole_catalog(tmp_path):
    write_reco(tmp_path)
    catalog = RecoCatalog.load(tmp_path / "gt_tokens.jsonl", wines_path=tmp_path / "wines.jsonl")
    profiles = profiles_of(catalog)
    assert set(profiles) == set(catalog.by_slug)
    brut = profiles["kappa-brut"]
    assert brut.effervescence == 4.0 and brut.source("sweetness") == CATALOG
    assert brut.source("alcohol") == CATALOG  # крепость из slug выгрузки
    assert profiles["iota-merlot-sladkoe"].sweetness > profiles["zeta-merlot-polusuhoe"].sweetness
    assert profiles["beta-merlot"].source("sweetness") is None  # сахара нет в названии и slug


# ------------------------------------------------------------------ оценка по стилю
def _grape_wine(priors, grape: str, color: Color, sugar: SugarClass | None, **kw) -> StyleProfile:
    return build_profile([grape], color, sugar, WineKind.STILL, priors, **kw)


def test_style_priors_are_medians_of_grape_axes(priors):
    """Медиана осей «по сорту» у вин того же стиля; вина без сорта в приорах в неё не входят."""
    reds = [
        _grape_wine(priors, grape, Color.RED, SugarClass.DRY)
        for grape in ("merlot", "saperavi", "cabernet_sauvignon", "pinot_noir", "syrah")
    ]
    nameless = build_profile(["no_such_grape"], Color.RED, SugarClass.DRY, WineKind.STILL, priors)
    style = StylePriors.build([(("Красное", False, "suhoe"), p) for p in (*reds, nameless)])
    prior = style.prior(("Красное", False, "suhoe"))
    for axis in STYLE_AXES:
        values = sorted(p.axis(axis) for p in reds)
        assert prior[axis] == pytest.approx(round(values[2], 2)), axis
    assert style.counts[("style", "Красное", False, "suhoe")] == 5  # без вина вне приоров


def test_style_fill_keeps_facts_and_grape_axes(priors):
    """Оценка по стилю закрывает только пустые оси «по сорту»: сахар и крепость — факты."""
    reds = [
        _grape_wine(priors, grape, Color.RED, SugarClass.DRY, abv=13.0)
        for grape in ("merlot", "saperavi", "cabernet_sauvignon", "pinot_noir", "syrah")
    ]
    style = StylePriors.build([(("Красное", False, "suhoe"), p) for p in reds])
    known = reds[0]
    assert style.fill(known, ("Красное", False, "suhoe")) is known  # оси «по сорту» есть
    lone = build_profile(["no_such_grape"], Color.RED, None, WineKind.STILL, priors)
    shown = style.fill(lone, ("Красное", False, None))
    assert {axis: shown.source(axis) for axis in AXES} == {
        "sweetness": None,
        "acidity": STYLE,
        "tannin": STYLE,
        "body": STYLE,
        "alcohol": None,
        "oak": STYLE,
        "aroma_intensity": STYLE,
        "effervescence": CATALOG,
    }
    assert sum(shown.known(axis) for axis in AXES) >= 3  # «роза ветров» рисуется
    prior = style.prior(("Красное", False, None))
    assert all(shown.axis(axis) == pytest.approx(prior[axis]) for axis in STYLE_AXES)
    # Профиль вина не меняется: подбор, пары и подача видят прежние `None`.
    assert lone.source("acidity") is None and lone.acidity == 3.3


def test_style_groups_narrow_to_wide_colour_first(priors):
    """Узкая группа без пяти вин уступает группе цвета; красные в оценку белого не попадают."""
    whites = [
        _grape_wine(priors, grape, Color.WHITE, SugarClass.DRY)
        for grape in ("riesling", "chardonnay", "sauvignon_blanc", "aligote", "rkatsiteli")
    ]
    sweet = _grape_wine(priors, "muscat", Color.WHITE, SugarClass.SWEET)
    reds = [
        _grape_wine(priors, grape, Color.RED, SugarClass.DRY)
        for grape in ("merlot", "saperavi", "cabernet_sauvignon", "pinot_noir", "syrah", "malbec")
    ]
    style = StylePriors.build(
        [
            *((("Белое", False, "suhoe"), p) for p in whites),
            (("Белое", False, "sladkoe"), sweet),
            *((("Красное", False, "suhoe"), p) for p in reds),
        ]
    )
    # Одно сладкое белое — меньше пяти: берётся белое тихое целиком, а не одно вино.
    assert style.prior(("Белое", False, "sladkoe")) == style.groups[("color", "Белое", False)]
    assert (
        style.prior(("Белое", False, "suhoe")) == style.groups[("style", "Белое", False, "suhoe")]
    )
    # Цвет неизвестен — тихие вообще; розового нет — по игристости, но танины белого предела.
    assert style.prior((None, False, None)) == style.groups[("sparkling", False)]
    lone = build_profile(["no_such_grape"], Color.ROSE, SugarClass.DRY, WineKind.STILL, priors)
    shown = style.fill(lone, ("Розовое", False, "suhoe"))
    assert shown.source("tannin") == STYLE and shown.tannin <= 0.6
    small = StylePriors.build([(("Красное", False, "suhoe"), reds[0])])
    assert small.prior(("Белое", False, "suhoe")) == small.groups[("sparkling", False)]
    assert StylePriors.build([]).prior(("Белое", False, None)) == {}
