"""Сборка данных сомелье (`scripts/build_somm.py`) и их загрузчик (`app/recommend/somm_data.py`).

На синтетике — правила фактов выгрузки (сахар, игристость, крепость), таблица подачи, вердикт и
фильтры «к чему подать», портреты сортов, словарь замка, чистка справочника «Лозы». Движок «Лозы»
запускается так же, как в сборке, — отдельным процессом; тест проверяет накладку сканера на
парах, которые винолюб заметит сразу. Загрузчик читает заглушки договора строго и отбрасывает
битое без падения сервиса.

Полная сборка на выгрузке организатора идёт, только если входы есть на машине
(`SVS_DATASET_DIR`, `SVS_DATA_DIR`): приёмка дорожки и сверка с `data/somm` байт в байт.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings
from app.recommend.profile import StyleProfile
from app.recommend.somm_data import (
    FIXTURE_DIR,
    SOMM_FILES,
    load_somm_data,
    match_serve,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_somm.py"
SAPERAVI = "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135"


@pytest.fixture(scope="module")
def somm() -> Any:
    spec = importlib.util.spec_from_file_location("build_somm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference(somm: Any) -> dict[str, Any]:
    return json.loads((somm.REFERENCE / "pairing.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dishes(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {dish["id"]: dish for dish in reference["dishes"]}


@pytest.fixture(scope="module")
def family(somm: Any, reference: dict[str, Any]) -> dict[str, str]:
    return somm.families(reference["dishes"])


def wine(somm: Any, **overrides: Any) -> Any:
    base = {
        "slug": "w",
        "name": "Вино",
        "winery": "Винодельня",
        "region": "Кубань",
        "color": "Красное",
        "grapes": ("Саперави",),
        "codes": ("saperavi",),
        "sugar": "suhoe",
        "sparkling": False,
        "abv": 13.5,
        "year": None,
        "canonical": True,
    }
    base.update(overrides)
    return somm.WineFacts(**base)


def engine_wine(profile: dict[str, float], **overrides: Any) -> dict[str, Any]:
    base = {
        "id": "w",
        "profile": profile,
        "descriptors": [],
        "color": "red",
        "kind": "still",
        "region": "kuban",
        "grapes": [],
        "serve_temp_c": [16, 18],
    }
    base.update(overrides)
    return base


RED = {
    "sweetness": 0.4,
    "acidity": 3.85,
    "tannin": 4.2,
    "body": 4.45,
    "alcohol": 3.93,
    "oak": 2.5,
    "aroma_intensity": 3.6,
    "effervescence": 0.0,
}
SWEET = dict(RED, sweetness=4.5, tannin=0.2, body=2.75, oak=1.0, alcohol=5.0, acidity=3.55)
SEMI_SWEET = dict(SWEET, sweetness=3.2, alcohol=3.0)
BRUT = dict(SWEET, sweetness=0.7, acidity=4.0, body=2.5, alcohol=3.2, effervescence=4.0)


# ------------------------------------------------------------------ факты выгрузки
@pytest.mark.parametrize(
    ("slug", "sugar"),
    [
        ("abrau-dyurso-russkoe-igristoe-sladkoe-krasnoe-bryut-125", "brut"),
        ("fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135", "suhoe"),
        ("x-beloe-polusuhoe-12", "polusuhoe"),
        ("x-beloe-ekstra-bryut-12", "extra_brut"),
        ("x-beloe-bryut-natyur-12", "brut_nature"),
        ("cru-lermont-risling", None),
    ],
)
def test_slug_sugar_takes_last_word(somm: Any, slug: str, sugar: str | None) -> None:
    assert somm.slug_sugar(slug) == sugar


def test_sugar_from_name_first(somm: Any) -> None:
    assert somm.sugar_of("Игристое экстра брют", "x-beloe-sladkoe-12") == "extra_brut"
    assert somm.sugar_of("Мерло", "x-krasnoe-polusladkoe-12") == "polusladkoe"
    assert somm.sugar_of("Мерло", "merlo") is None
    # опечатка выгрузки «полсусладкое» — сахар из slug, как у карточки сервиса
    assert somm.sugar_of("Абрау Купаж красный полсусладкое", "x-krasnoe-polusladkoe-11") == (
        "polusladkoe"
    )


def test_facts_rules_are_the_card_rules(somm: Any) -> None:
    """Сборка и карточка сервиса считают сахар, игристость и крепость одними функциями."""
    from app.recommend import catalog

    assert somm.organizer_sugar is catalog.organizer_sugar
    assert somm.organizer_sparkling is catalog.organizer_sparkling
    assert somm.sweet_name is catalog.sweet_name
    assert somm.card_sugar is catalog.card_sugar


def test_abv_rule(somm: Any) -> None:
    assert somm.abv_of([13.5]) == 13.5
    assert somm.abv_of([11, 13]) == 11
    assert somm.abv_of([135]) is None
    assert somm.abv_of([]) is None
    assert somm.abv_of([22], 2022) is None  # daniel-22: год урожая, а не крепость
    assert somm.abv_of([12], 2022) == 12


def test_load_facts(somm: Any, tmp_path: Path) -> None:
    """Повторы строк выгрузки, заглушки сортов, сахар, игристость, канон группы."""
    rows = [
        ("Cru Lermont Saperavi", "Красное", "Саперави", SAPERAVI),
        ("Cru Lermont Saperavi", "Красное", "Саперави", SAPERAVI),
        (
            "Мускатель белый",
            "Белое",
            "Белые сорта винограда",
            "massandra-muskatel-beloe-sladkoe-16",
        ),
        ("Alveus Ultra Cuvee", "Белое", "Шардоне", "fanagoriya-alveus-beloe-bryut-12"),
        ("Пет-нат", "Розовое", "Каберне Фран, Мерло", "pet-nat-rozovoe"),
        ("Пет-Нат Рубин", "Розовое", "Саперави", "denisov_pet_nat_rubin"),
        ("Daniel, 2022", "Красное", "Красностоп", "daniel-22"),
    ]
    csv_path = tmp_path / "strapi.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Название вина", "Категория", "Регион", "Сорт винограда", "Винодельня", "Slug"]
        )
        for name, color, grapes, slug in rows:
            writer.writerow([name, color, "Кубань", grapes, " Фанагория ", slug])
    gt = [
        {"slug": SAPERAVI, "fields": {"grape": {"codes": ["saperavi"]}, "abv": {"value": [13.5]}}},
        {"slug": "massandra-muskatel-beloe-sladkoe-16", "fields": {"abv": {"value": [16]}}},
        {"slug": "fanagoriya-alveus-beloe-bryut-12", "fields": {"abv": {"value": [12]}}},
        {"slug": "pet-nat-rozovoe", "fields": {"year": {"value": 2023}}},
        {"slug": "denisov_pet_nat_rubin", "fields": {}},
        {"slug": "daniel-22", "fields": {"abv": {"value": [22]}, "year": {"value": 2022}}},
    ]
    gt_path = tmp_path / "gt.jsonl"
    gt_path.write_text("".join(json.dumps(item) + "\n" for item in gt), encoding="utf-8")
    groups = {"W1": {"canonical": SAPERAVI}, "W2": {"canonical": "pet-nat-rozovoe"}}
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps(groups), encoding="utf-8")

    facts = {item.slug: item for item in somm.load_facts(csv_path, gt_path, groups_path)}
    assert len(facts) == 6
    red = facts[SAPERAVI]
    assert (red.sugar, red.sparkling, red.abv, red.codes, red.winery) == (
        "suhoe",
        False,
        13.5,
        ("saperavi",),
        "Фанагория",
    )
    assert red.canonical
    muscat = facts["massandra-muskatel-beloe-sladkoe-16"]
    assert muscat.grapes == () and muscat.sugar == "sladkoe" and not muscat.canonical
    assert facts["fanagoriya-alveus-beloe-bryut-12"].sparkling  # брют — игристое
    petnat = facts["pet-nat-rozovoe"]
    assert petnat.sparkling and petnat.sugar is None and petnat.year == 2023
    assert petnat.grapes == ("Каберне Фран", "Мерло")
    assert facts["denisov_pet_nat_rubin"].sparkling  # «Пет-Нат» и slug через подчёркивание
    daniel = facts["daniel-22"]
    assert daniel.abv is None and daniel.year == 2022  # хвост slug «22» — год


# ------------------------------------------------------------------ подача
@pytest.mark.parametrize(
    ("color", "sugar", "sparkling", "body", "source", "rule", "by_grape"),
    [
        ("Белое", "brut", True, 2.5, "grape", "sparkling", False),
        ("Белое", "sladkoe", False, 2.7, None, "sweet", False),
        ("Оранжевое", "suhoe", False, 3.2, None, "orange", False),
        ("Белое", "suhoe", False, 3.6, "grape", "white_full", True),
        ("Белое", "suhoe", False, 3.6, None, "white", False),
        ("Белое", None, False, 2.5, "grape", "white", True),
        ("Розовое", "polusuhoe", False, 2.5, None, "rose", False),
        ("Красное", "suhoe", False, 4.45, "grape", "red_full", True),
        ("Красное", "suhoe", False, 3.4, None, "red", False),
    ],
)
def test_serve_rules(
    somm: Any,
    color: str,
    sugar: str | None,
    sparkling: bool,
    body: float,
    source: str | None,
    rule: str,
    by_grape: bool,
) -> None:
    """Тело без источника условию `body_min` не отвечает; `by_grape` — сверялись с телом сорта."""
    found = match_serve(
        somm.serve_rules(),
        color=color,
        sugar=sugar,
        sparkling=sparkling,
        body=body,
        body_source=source,
    )
    assert found is not None
    assert (found.rule.id, found.by_grape) == (rule, by_grape)
    assert found.public()["source"] == "rule"


@pytest.mark.parametrize(
    ("name", "color", "sugar", "rule"),
    [
        ("Портвейн Крымский", "Белое", None, "sweet_name"),
        ("Массандра Херес", "Белое", None, "sweet_name"),
        ("Chateau Tamagne Grand Dessert Nectar", "Красное", None, "sweet_name"),
        ("Массандра Мускат", "Белое", None, "sweet_name"),
        # Проверка 25.09, вечер: «поздний сбор» в именительном падеже и ледяное вино.
        ("Поздний сбор, Белое", "Белое", None, "sweet_name"),
        ("Поздний сбор, Красное", "Красное", None, "sweet_name"),
        ("ICE Wine Рислинг", "Белое", None, "sweet_name"),
        ("Мускат", "Белое", "suhoe", "white"),  # сахар выгрузки главнее названия
        ("Мускатель белый", "Белое", "sladkoe", "sweet"),
        ("Cru Lermont Рислинг", "Белое", None, "white"),
    ],
)
def test_fortified_and_dessert_names_without_sugar_are_served_as_sweet(
    somm: Any, name: str, color: str, sugar: str | None, rule: str
) -> None:
    """Позиции без сахара в выгрузке, но с названием креплёного, десертного или мускатного вина
    (24 на 25.09) подаются при 10–14 °C, а не как сухое белое (6–8) или плотное красное (16–18)."""
    found = somm.serve_of(
        wine(somm, name=name, color=color, sugar=sugar),
        StyleProfile(body=2.5, sources={"body": "grape"}),
        somm.serve_rules(),
    )
    assert found.rule.id == rule
    if rule.startswith("sweet"):
        assert found.rule.temperature_c == (10, 14)


def test_every_style_has_a_serve(somm: Any) -> None:
    rules = somm.serve_rules()
    for color in somm.COLOR_CODES:
        for sugar in (None, *somm.SUGAR_WORDS):
            for sparkling in (False, True):
                for body, source in ((None, None), (2.0, "grape"), (4.0, "grape")):
                    found = match_serve(
                        rules,
                        color=color,
                        sugar=sugar,
                        sparkling=sparkling,
                        body=body,
                        body_source=source,
                    )
                    assert found is not None, (color, sugar, sparkling, body)
                    low, high = found.rule.temperature_c
                    assert 4 <= low < high <= 20


def test_serve_table_matches_contract(somm: Any) -> None:
    fixture = json.loads((FIXTURE_DIR / "serve.json").read_text(encoding="utf-8"))
    assert [dict(rule) for rule in somm.SERVE_RULES] == fixture["rules"]


# ------------------------------------------------------------------ вердикт и «к чему подать»
def test_verdict(somm: Any) -> None:
    weights = {"a": 2.5, "soft": -2.2, "hard": -3.0}
    assert somm.verdict_of(0.9, [], [], weights) == "neutral"
    assert somm.verdict_of(0.9, ["a"], ["hard"], weights) == "no"
    assert somm.verdict_of(0.5, ["a"], [], weights) == "no"
    assert somm.verdict_of(0.8, ["a"], ["soft"], weights) == "caveat"
    assert somm.verdict_of(0.8, ["a"], [], weights) == "yes"


def test_verdict_with_hedge_rule(somm: Any) -> None:
    """Правило-оговорка (сладкое только по названию к рыбе, третий круг проверки 25.09) вердикт
    не решает: пара судится без него, и «да», `neutral` и даже «скорее нет» по одной оценке
    становятся оговоркой; «скорее нет» по другому правилу «−» остаётся."""
    hedge = "maybe_sweet_wine_on_fish"
    assert hedge in somm.HEDGE_RULES
    weights = {"a": 2.5, "soft": -2.2, "hard": -3.0, hedge: -1.0}
    # Без оговорки пара — «да» (сумма 3,0), с ней — оговорка, а не «да».
    assert somm.verdict_of(somm.pair_score(2.0), ["a"], [hedge], weights, 2.0) == "caveat"
    # Одна оговорка без других правил — оговорка, а не `neutral`.
    assert somm.verdict_of(somm.pair_score(-1.0), [], [hedge], weights, -1.0) == "caveat"
    # Сумма с оговоркой ниже порога (0,5 → «скорее нет» по оценке), без неё — «да»: оговорка.
    assert somm.pair_score(0.5) < somm.MIN_SCORE <= somm.pair_score(1.5)
    assert somm.verdict_of(somm.pair_score(0.5), ["a"], [hedge], weights, 0.5) == "caveat"
    # Жёсткое правило и оценка без оговорки ниже порога — «скорее нет» по ним.
    assert somm.verdict_of(0.9, ["a"], ["hard", hedge], weights, 3.0) == "no"
    assert somm.verdict_of(somm.pair_score(-0.5), ["a"], ["soft", hedge], weights, -0.5) == "no"
    assert somm.verdict_of(somm.pair_score(1.5), ["a"], ["soft", hedge], weights, 1.5) == "caveat"
    with pytest.raises(ValueError):
        somm.verdict_of(0.9, ["a"], [hedge], weights)
    # В объяснении оговорка — последняя: меньше по модулю любого другого «−».
    assert somm.ordered([hedge, "soft", "hard"], weights) == ["hard", "soft", hedge]


def test_pair_score_is_the_engine_sigmoid(somm: Any, engine: dict[str, Any]) -> None:
    """`pair_score` — та же сигмоида, что у движка: по ней сборка судит пару без оговорки."""
    for row in engine["pairs"].values():
        for total, score, _, _ in row.values():
            assert abs(somm.pair_score(total) - score) < 1e-5


def test_rules_order_by_weight_then_id(somm: Any) -> None:
    weights = {"b": 2.0, "a": 2.0, "c": -3.5, "d": 1.0}
    assert somm.ordered(["d", "b", "c", "a"], weights) == ["c", "a", "b", "d"]


def test_families(family: dict[str, str]) -> None:
    assert family["borsch_s_pampushkami"] == "borsch"
    assert family["borsch"] == "borsch"
    assert family["shashlyk_svinina"] != family["shashlyk_baranina"]
    assert family["ikra_krasnaya"] != family["ikra_chernaya"]


def pair(somm: Any, total: float, verdict: str = "yes", minus: tuple[str, ...] = ()) -> Any:
    return somm.DishPair(total=total, verdict=verdict, plus=("intensity_match",), minus=minus)


def test_top_only_yes_family_and_order(
    somm: Any, dishes: dict[str, Any], family: dict[str, str]
) -> None:
    row = {
        "borsch": pair(somm, 9.0),
        "borsch_s_pampushkami": pair(somm, 10.0),
        "gus": pair(somm, 12.0, "caveat", ("umami_vs_tannin_clash",)),
        "steik_ribay": pair(somm, 8.0),
        "shashlyk_baranina": pair(somm, 8.0),
        "burger": pair(somm, 7.0),
        "buzhenina": pair(somm, 6.0),
        "kholodets": pair(somm, 5.0),
        "solyanka": pair(somm, 4.0, "no", ("tannin_vs_spice_clash",)),
    }
    top = somm.top_dishes(wine(somm), row, dishes, family)
    assert top == [
        "borsch_s_pampushkami",
        "shashlyk_baranina",
        "steik_ribay",
        "burger",
        "buzhenina",
    ]


def test_top_sweet_only_desserts_cheese_spicy(
    somm: Any, dishes: dict[str, Any], family: dict[str, str]
) -> None:
    row = {
        dish_id: pair(somm, 10.0 - index * 0.1)
        for index, dish_id in enumerate(
            ["solenaya_seld", "malosolnaya_semga", "medovik", "cheese_plate", "shaverma", "frukty"]
        )
    }
    # Острое — только когда сработало «Сладость гасит остроту»: шаверма ледяному вину «за
    # кислотность к солёному» в «к чему подать» не идёт (проверка 24.09), харчо с правилом — идёт.
    row["kharcho"] = somm.DishPair(
        total=9.65, verdict="yes", plus=("spice_needs_sweet_and_calm",), minus=()
    )
    for sugar in ("sladkoe", "polusladkoe"):
        top = somm.top_dishes(wine(somm, sugar=sugar), row, dishes, family)
        assert top == ["medovik", "cheese_plate", "kharcho", "frukty"], sugar
    # Сухому вину фильтр сладкого не нужен: шаверма остаётся.
    only = {"shaverma": row["shaverma"]}
    assert somm.top_dishes(wine(somm), only, dishes, family) == ["shaverma"]


def test_top_unknown_sugar(somm: Any, dishes: dict[str, Any], family: dict[str, str]) -> None:
    row = {
        "medovik": pair(somm, 12.0),
        "malosolnaya_semga": pair(somm, 11.0),
        "orehi": pair(somm, 3.0),
        "cheese_plate": pair(somm, 2.0),
    }
    assert somm.top_dishes(wine(somm, sugar=None), row, dishes, family) == [
        "malosolnaya_semga",
        "orehi",
        "cheese_plate",
    ]
    # Портвейн без сахара в выгрузке — для правил сладкое (проверка 25.09, вечер): десерты, сыр и
    # орехи (третий круг проверки 25.09: давние пары сладкого и креплёного), без рыбы.
    for name in ("Портвейн Крымский", "Массандра Херес", "Grand Dessert Muscat", "Мускат"):
        port = wine(somm, sugar=None, name=name)
        assert somm.rule_sugar(port) == "sladkoe", name
        assert somm.top_dishes(port, row, dishes, family) == ["medovik", "orehi", "cheese_plate"]
    # Полусладкому орехи в «к чему подать» не добавлены — только сыр, как было.
    semi = wine(somm, sugar="polusladkoe")
    assert somm.top_dishes(semi, row, dishes, family) == ["medovik", "cheese_plate"]
    # Сахар известен — подсказка стиля не нужна.
    assert "malosolnaya_semga" in somm.top_dishes(
        wine(somm, sugar="suhoe", name="Херес"), row, dishes, family
    )


def test_sweet_by_name_is_a_guess_the_card_can_cancel(somm: Any) -> None:
    """Третий круг проверки 25.09: «сладкое по названию» — догадка. Название оранжевого вина
    («Мускат Оранж») и «сухое», «сухость» или «брют» в «Описании» её снимают; «сухофрукты» —
    нет. Без догадки сахар для правил неизвестен, как у любого вина без сахара."""
    port = wine(somm, sugar=None, name="Портвейн Крымский", description="Ароматы груши.")
    assert somm.sweet_by_name(port) and somm.rule_sugar(port) == "sladkoe"
    orange = wine(somm, sugar=None, name="Мускат Оранж", description="Оттенки сухофруктов.")
    dry = wine(
        somm,
        sugar=None,
        name="Жемчужная 9 Мускат Розовый, Пино гри",
        description="Приятная легкая сладость, несмотря на абсолютную сухость вина",
    )
    for item in (orange, dry):
        assert not somm.sweet_by_name(item) and somm.rule_sugar(item) is None, item.name
    raisins = wine(somm, sugar=None, name="Мускат", description="Тона сухофруктов и мёда.")
    assert somm.sweet_by_name(raisins)
    # Сахар выгрузки главнее и названия, и описания.
    assert not somm.sweet_by_name(wine(somm, sugar="sladkoe", name="Мускат"))
    # Проверка `acc-somm3`: «Сладкое розовое вино» в «Описании» — сахар по карточке, а не догадка.
    late = wine(
        somm,
        sugar=None,
        name="Мускат позднего сбора розовый",
        description="«Мускат позднего сбора» розовое Сладкое розовое вино ЗГУ Крым",
    )
    assert late.sugar is None and somm.sugar_by_card(late) == "sladkoe"
    assert not somm.sweet_by_name(late) and somm.rule_sugar(late) == "sladkoe"
    # Сахар выгрузки главнее «Описания».
    dry = wine(somm, sugar="suhoe", name="Мускат", description="Сладкое вино из сорта Мускат")
    assert somm.sugar_by_card(dry) == "suhoe" == somm.rule_sugar(dry)


def test_sugar_decides_only_sweet_dishes_of_unknown_sugar(
    somm: Any, dishes: dict[str, Any]
) -> None:
    """Сахар неизвестен, блюдо — десерт, варенье или мёд: пару решает сахар (проверка 25.09)."""
    unknown, dry = wine(somm, sugar=None), wine(somm)
    port = wine(somm, sugar=None, name="Портвейн Крымский")
    for dish_id in ("medovik", "shokolad", "frukty", "myod", "varenye"):
        assert somm.sugar_decides(unknown, dishes[dish_id]), dish_id
        assert not somm.sugar_decides(dry, dishes[dish_id]), dish_id
        # Название креплёного вина — не «сахар неизвестен»: правила судят его как сладкое.
        assert not somm.sugar_decides(port, dishes[dish_id]), dish_id
    for dish_id in ("utka_s_yablokami", "grebeshki", "borsch", "cheese_plate"):
        assert not somm.sugar_decides(unknown, dishes[dish_id]), dish_id


def test_top_limit(somm: Any, dishes: dict[str, Any], family: dict[str, str]) -> None:
    row = {dish_id: pair(somm, 1.0) for dish_id in dishes}
    top = somm.top_dishes(wine(somm), row, dishes, family)
    assert len(top) == somm.TOP_LIMIT and top == sorted(top)


# ------------------------------------------------------------------ движок «Лозы» в подпроцессе
@pytest.fixture(scope="module")
def engine(somm: Any) -> dict[str, Any]:
    wines = [
        engine_wine(RED, id="red", descriptors=["black_fruit", "chocolate", "spice"]),
        engine_wine(SWEET, id="sweet", color="white", serve_temp_c=[10, 14]),
        engine_wine(SEMI_SWEET, id="semi", color="white", serve_temp_c=[6, 8]),
        engine_wine(BRUT, id="brut", color="white", kind="sparkling", serve_temp_c=[6, 8]),
        engine_wine(SWEET, id="guess", color="white", serve_temp_c=[10, 14], sugar_by_name=True),
    ]
    return somm.run_engine({"wines": wines, "overlay": True})


def test_engine_rules_are_contract_rules(somm: Any, engine: dict[str, Any]) -> None:
    rules = {rule["id"]: rule for rule in engine["rules"]}
    assert set(rules) == set(somm.RULE_META)
    assert len(rules) == 39 and "portal_editorial_prior" not in rules
    contract = json.loads((FIXTURE_DIR / "dishes.json").read_text(encoding="utf-8"))["rules"]
    for rule in contract:
        assert rules[rule["id"]]["weight"] == rule["weight"], rule["id"]
        assert somm.RULE_META[rule["id"]][0] == rule["chip"], rule["id"]


def test_engine_covers_all_dishes(engine: dict[str, Any], dishes: dict[str, Any]) -> None:
    assert len(dishes) == 82
    for row in engine["pairs"].values():
        assert set(row) == set(dishes)
        for total, score, plus, minus in row.values():
            assert isinstance(total, float) and 0.0 < score <= 0.96
            assert isinstance(plus, list) and isinstance(minus, list)


def test_overlay_meat_does_not_clash_with_tannin(engine: dict[str, Any]) -> None:
    """Накладка: шашлык и стейк связывают танины, а бульон борща — нет."""
    red = engine["pairs"]["red"]
    for dish_id in ("shashlyk_baranina", "steik_ribay", "gus"):
        assert "umami_vs_tannin_clash" not in red[dish_id][3], dish_id
        assert "tannin_meets_protein_fat" in red[dish_id][2], dish_id
    for dish_id in ("borsch", "solyanka", "shchi", "ukha"):
        assert "umami_vs_tannin_clash" in red[dish_id][3], dish_id


def test_overlay_broth_only_for_soups(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09: «Бульон подчёркивает терпкость» — только супам и блюдам на бульоне.

    Без накладки минус «бульон» стоял у солёной сельди и сырной тарелки. Сельдь, икра и
    суши по-прежнему спорят с терпким красным — своим правилом о рыбе.
    """
    red = engine["pairs"]["red"]
    for dish_id in ("solenaya_seld", "cheese_plate", "griby_v_smetane", "ikra_krasnaya", "sushi",
                    "kharcho", "gulyash"):  # fmt: skip
        assert "umami_vs_tannin_clash" not in red[dish_id][3], dish_id
    for dish_id in ("solenaya_seld", "ikra_krasnaya", "sushi"):
        assert "no_tannin_with_oily_fish" in red[dish_id][3], dish_id
    raw = somm.run_engine({"wines": [engine_wine(RED)], "overlay": False})["pairs"]["w"]
    assert "umami_vs_tannin_clash" in raw["cheese_plate"][3]
    assert "umami_vs_tannin_clash" in raw["solenaya_seld"][3]


def test_overlay_dessert_rules_only_for_desserts(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09: утка с яблоками и гребешки (сладость 3) — не десерт ни для сухого, ни
    для сладкого вина; медовик, шоколад и мёд — десерт."""
    red, sweet = engine["pairs"]["red"], engine["pairs"]["sweet"]
    for dish_id in ("utka_s_yablokami", "grebeshki"):
        assert "dry_wine_on_dessert" not in red[dish_id][3], dish_id
        assert "wine_sweeter_than_dish" not in sweet[dish_id][2], dish_id
    for dish_id in ("medovik", "shokolad", "myod"):
        assert "dry_wine_on_dessert" in red[dish_id][3], dish_id
        assert "wine_sweeter_than_dish" in sweet[dish_id][2], dish_id
    raw = somm.run_engine({"wines": [engine_wine(RED)], "overlay": False})["pairs"]["w"]
    assert "dry_wine_on_dessert" in raw["utka_s_yablokami"][3]


def test_overlay_bubbles_only_for_fried(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09: «Пузырьки освежают жареное» — только жареному, а не сёмге, блинам с
    икрой и селёдке под шубой (независимая проверка)."""
    brut = engine["pairs"]["brut"]
    for dish_id in ("kotlety_po_kievski", "kotlety_farsh", "zharenaya_kartoshka_s_gribami"):
        assert "bubbles_cut_fat_and_fry" in brut[dish_id][2], dish_id
    for dish_id in ("malosolnaya_semga", "bliny_ikra", "seledka_pod_shuboy", "olivier", "salo"):
        assert "bubbles_cut_fat_and_fry" not in brut[dish_id][2], dish_id
    raw = somm.run_engine(
        {"wines": [engine_wine(BRUT, color="white", kind="sparkling")], "overlay": False}
    )
    assert "bubbles_cut_fat_and_fry" in raw["pairs"]["w"]["malosolnaya_semga"][2]


def test_unknown_sugar_desserts_are_neutral_and_never_dry(somm: Any) -> None:
    """Сборка на двух винах: без сахара в выгрузке — десерты `neutral` без правил, и ни одно
    правило не называет вино сухим; сухому — «Сухое на фоне десерта резче», как было."""
    result = somm.build_from_facts(
        [wine(somm, slug="dry"), wine(somm, slug="nosugar", sugar=None)], "проба"
    )
    desserts = [
        dish_id
        for dish_id, dish in result.dishes.items()
        if dish["category"] in somm.DESSERT_CATEGORIES
    ]
    assert len(desserts) == 11
    for dish_id in desserts:
        pair = result.pairs["nosugar"][dish_id]
        assert (pair.verdict, pair.plus, pair.minus) == ("neutral", (), ()), dish_id
    assert "dry_wine_on_dessert" in result.pairs["dry"]["medovik"].minus
    report = somm.quality(result)
    assert report["dry_guess"] == [] and report["no_dish"] == []
    rules = {rule["id"]: rule for rule in json.loads(result.files["dishes.json"])["rules"]}
    assert somm.DRY_CLAIM.search(rules["dry_wine_on_dessert"]["chip"])
    # «ощущаются сухо» — о танинах, а не о вине: такое правило вину без сахара можно.
    assert not somm.DRY_CLAIM.search(rules["bare_tannin_on_lean_dish"]["text"])


def test_sweet_name_is_sweet_for_pair_rules(somm: Any) -> None:
    """Проверка 25.09, вечер: портвейн без сахара в выгрузке профиль считал сухим, и к нему
    «подходили» устрицы и сельдь. Теперь правила судят его как возможно сладкое: рыба не «да», а
    оговорка «если вино сладкое…» (третий круг проверки: сахар здесь догадка, а не факт), десерты —
    по правилам о сахаре, а не `neutral`, «к чему подать» — как у сладкого. Сладкое по карточке к
    рыбе — «скорее нет»."""
    port = wine(
        somm,
        slug="port",
        name="Портвейн Крымский",
        color="Белое",
        grapes=("Кокур",),
        codes=("kokur",),
        sugar=None,
    )
    known = wine(
        somm,
        slug="known",
        name="Кокур Десертный Сурож",
        color="Белое",
        grapes=("Кокур",),
        codes=("kokur",),
        sugar="sladkoe",
    )
    result = somm.build_from_facts([port, known, wine(somm, slug="dry")], "проба")
    fish = {dish_id for dish_id, dish in result.dishes.items() if somm.food_of(dish) == "fish"}
    seafood = {d for d, dish in result.dishes.items() if set(dish["flavor_tags"]) & somm.FISH_TAGS}
    row = result.pairs["port"]
    # Сладкое по карточке к рыбе — «скорее нет» своим правилом (третий круг проверки 25.09).
    assert {result.pairs["known"][dish_id].verdict for dish_id in seafood} == {"no"}
    assert result.pairs["known"]["oysters"].minus[0] == "sweet_wine_on_fish"
    # Сладкое только по названию — оговорка с условной фразой; «скорее нет» — только по другим
    # правилам, и тогда первым в объяснении идёт оно, а не догадка о сахаре.
    for dish_id in seafood:
        pair = row[dish_id]
        assert "maybe_sweet_wine_on_fish" in pair.minus, dish_id
        assert "sweet_wine_on_fish" not in pair.minus, dish_id
        assert "sweet_wine_on_savoury_main" not in pair.minus, dish_id
        assert pair.verdict == "caveat" or pair.minus[0] not in somm.HEDGE_RULES, dish_id
    assert {row[dish_id].verdict for dish_id in fish} <= {"caveat", "no"}
    assert row["solenaya_seld"].verdict == "caveat"
    assert row["solenaya_seld"].minus == ("maybe_sweet_wine_on_fish",)
    assert row["ukha"].verdict == "caveat" and row["kulebyaka"].verdict == "caveat"
    # К несладкому горячему без рыбы — как было: «Сладость спорит с блюдом».
    assert "sweet_wine_on_savoury_main" in row["borsch"].minus
    for dish_id in ("cheese_plate", "orehi"):
        assert row[dish_id].verdict == "yes", dish_id
        assert "sweet_wine_with_cheese_and_nuts" in row[dish_id].plus, dish_id
    assert row["medovik"].verdict == "yes" and "wine_sweeter_than_dish" in row["medovik"].plus
    assert result.tops["port"]
    for dish_id in result.tops["port"]:
        dish = result.dishes[dish_id]
        assert (
            dish["category"] in somm.DESSERT_CATEGORIES
            or somm.cheese_or_nuts(dish)
            or (somm.SPICE_RULE in row[dish_id].plus)
        ), dish_id
    report = somm.quality(result)
    assert report["sweet_fish"] == [] and report["no_dish"] == [] and report["dry_guess"] == []
    assert report["sweet_fish_soft"] == [] and report["sweet_cheese"] == []
    assert report["maybe_sweet_fish"] == []


def test_sweet_description_is_sweet_by_card(somm: Any) -> None:
    """Проверка `acc-somm3` 25.09: у «Мускат позднего сбора розовый» сахара нет ни в названии, ни в
    slug, а в «Описании» — «Сладкое розовое вино». Это сахар по карточке, а не догадка: к рыбе и
    морепродуктам — «скорее нет» и первым «Сладость спорит с рыбой», как у сладкого по выгрузке, а
    не оговорка «если вино сладкое…»; подача — как у сладких, сыр и орехи — «да». Тот же мускат
    без таких слов в «Описании» — по-прежнему догадка с оговоркой."""
    base = {
        "name": "Мускат позднего сбора розовый",
        "color": "Розовое",
        "grapes": ("Мускат розовый",),
        "codes": (),
        "sugar": None,
    }
    late = wine(somm, slug="late", description="Сладкое розовое вино ЗГУ Крым", **base)
    guess = wine(somm, slug="guess", description="Ароматы розы и мёда.", **base)
    result = somm.build_from_facts([late, guess, wine(somm, slug="dry")], "проба")
    seafood = {d for d, dish in result.dishes.items() if set(dish["flavor_tags"]) & somm.FISH_TAGS}
    assert len(seafood) == 18
    for dish_id in seafood:
        pair = result.pairs["late"][dish_id]
        assert pair.verdict == "no" and pair.minus[0] == "sweet_wine_on_fish", dish_id
        assert "maybe_sweet_wine_on_fish" not in pair.minus, dish_id
        assert "maybe_sweet_wine_on_fish" in result.pairs["guess"][dish_id].minus, dish_id
    assert result.pairs["guess"]["solenaya_seld"].verdict == "caveat"
    for dish_id in ("cheese_plate", "orehi"):
        assert result.pairs["late"][dish_id].verdict == "yes", dish_id
    assert result.serves["late"].rule.id == "sweet"
    assert result.serves["guess"].rule.id == "sweet_name"
    report = somm.quality(result)
    assert report["sweet_fish"] == [] and report["sweet_fish_soft"] == []
    assert report["maybe_sweet_fish"] == [] and report["sweet_cheese"] == []


def test_hedge_rule_text_is_conditional_and_clean(somm: Any) -> None:
    """Фраза и подпись правила-оговорки — условные («если так»), без утверждения о сахаре вина, и
    чистые по `content_filter`, стоп-листу сборки и стоп-листу голоса."""
    from app.sommelier.guards import stoplist_hits

    rules = {rule["id"]: rule for rule in somm.run_engine({"wines": [], "overlay": True})["rules"]}
    rule = rules["maybe_sweet_wine_on_fish"]
    chip, _, fields = somm.RULE_META["maybe_sweet_wine_on_fish"]
    text = rule["explanation_ru"]
    assert chip == "Если сладкое — спорит с рыбой" and len(chip) <= 30 and fields == ("sweetness",)
    assert "если" in text.lower() and "может быть сладким" in text
    assert rule["weight"] == -1.0 > somm.HARD_CONFLICT
    minus = [
        r["weight"] for rule_id, r in rules.items() if r["weight"] < 0 and rule_id != rule["id"]
    ]
    assert abs(rule["weight"]) < min(abs(weight) for weight in minus)
    assert somm.legal_violations([chip, text, rule["name"]]) == []
    assert stoplist_hits(chip) == [] and stoplist_hits(text) == []
    assert not somm.DRY_CLAIM.search(f"{chip} {text}")


@pytest.fixture(scope="module")
def spread(somm: Any) -> Any:
    """Сборка на винах всех сортов приоров в разных стилях: красное, белое, полусладкое,
    сладкое, брют, без сахара, креплёное название — чтобы сработало как можно больше правил."""
    from app.recommend.build import load_priors

    styles = ("suhoe", "polusladkoe", "sladkoe", "brut", None, "polusuhoe")
    facts = []
    for index, (code, prior) in enumerate(sorted(load_priors().items())):
        sugar = styles[index % len(styles)]
        red = prior.axes.get("tannin", 0.0) >= 1.0
        facts.append(
            wine(
                somm,
                slug=f"w{index}",
                name="Портвейн" if index % 11 == 0 else "Вино",
                color="Красное" if red else "Белое",
                grapes=(code,),
                codes=(code,),
                sugar=sugar,
                sparkling=sugar == "brut",
                abv=None if sugar is None else 13.5,
            )
        )
    return somm.build_from_facts(facts, "проба")


def test_rule_texts_claim_only_what_the_condition_guarantees(somm: Any, spread: Any) -> None:
    """Проверка 25.09, вечер: подпись и фраза правила — правда о каждом блюде, к которому правило
    сработало (`DISH_CLAIMS`): «лёгкое» — нежирное и негромкое, «жирное» — с жиром от 3,5,
    «кислинка» — с кислотностью от 3,5, «рыба» — с рыбным тегом. «Деликатное» у солёных огурцов,
    «сахар спорит с солью и мясом» у грибов в сметане и «дымок с углей» у блинов больше не
    пишутся, а сборка с такими подписями не проходит приёмку."""
    fired = {
        rule for row in spread.pairs.values() for p in row.values() for rule in (*p.plus, *p.minus)
    }
    assert len(fired) >= 30, sorted(set(somm.RULE_META) - fired)
    assert somm.false_claims(spread) == []
    assert somm.quality(spread)["false_claims"] == []
    # Прежние фразы «Лозы» проверка ловит — и сборка упала бы.
    old = {
        "heavy_wine_on_delicate_dish": "Вино слишком мощное для такого деликатного блюда.",
        "sweet_wine_on_savoury_main": "Сладкое вино спорит с солью и мясом.",
        "flat_wine_on_sour_dish": "Уксус и соленья в блюде «съедят» мягкое вино.",
        "acidity_mirror": "Блюдо жирное, вину нужна кислотность.",
    }
    dishes = json.loads(spread.files["dishes.json"])
    for rule in dishes["rules"]:
        rule["text"] = old.get(rule["id"], rule["text"])
    broken = somm.Build(
        files={**spread.files, "dishes.json": somm.dump(dishes)},
        facts=spread.facts,
        pairs=spread.pairs,
        tops=spread.tops,
        dishes=spread.dishes,
        rules=spread.rules,
        serves=spread.serves,
        seconds={},
    )
    found = somm.false_claims(broken)
    never = {(rule_id, dish_id) for rule_id, dish_id, _ in found if dish_id == "*"}
    assert {
        "heavy_wine_on_delicate_dish",
        "sweet_wine_on_savoury_main",
        "flat_wine_on_sour_dish",
    } <= {rule_id for rule_id, _ in never}
    # «Сытное» — тоже не гарантия: «Сладость спорит с …» срабатывает и у овощей на гриле.
    assert somm.DISH_NEVER.search("сладкое вино рядом с несладким сытным блюдом")
    lean = {dish_id for rule_id, dish_id, claim in found if rule_id == "acidity_mirror"}
    assert {"okroshka", "shchi", "borsch"} <= lean


def test_spread_build_keeps_the_quality_rules(somm: Any, spread: Any) -> None:
    """На всех сортах и стилях: у каждого вина блюда, сладкое и креплёное — без рыбы."""
    report = somm.quality(spread)
    assert report["no_dish"] == [] and report["sweet_fish"] == [] and report["dry_guess"] == []


def test_overlay_tannin_spars_with_oysters_and_ukha(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка: у устриц и ухи другие рыбные теги, но терпкому красному они тоже «−»."""
    for dish_id in ("oysters", "ukha", "solenaya_seld", "sushi"):
        assert "no_tannin_with_oily_fish" in engine["pairs"]["red"][dish_id][3], dish_id
        # Без танинов правило молчит, как и у «Лозы».
        assert "no_tannin_with_oily_fish" not in engine["pairs"]["sweet"][dish_id][3], dish_id
    raw = somm.run_engine({"wines": [engine_wine(RED)], "overlay": False})
    assert "no_tannin_with_oily_fish" not in raw["pairs"]["w"]["oysters"][3]


def test_overlay_sweet_wine_spars_with_fish(engine: dict[str, Any]) -> None:
    """Накладка: сладкое и полусладкое к рыбе — «−», а не «подходит». Третий круг проверки 25.09:
    сладкому — своё правило с весом −3 («скорее нет»), полусладкому — оговорка, как было."""
    fish = ("solenaya_seld", "malosolnaya_semga", "seledka_pod_shuboy", "ikra_krasnaya", "rolly",
            "oysters", "ukha", "kulebyaka")  # fmt: skip
    for dish_id in fish:
        assert "sweet_wine_on_fish" in engine["pairs"]["sweet"][dish_id][3], dish_id
        assert "dessert_wine_on_savoury_dish" not in engine["pairs"]["sweet"][dish_id][3]
        assert "dessert_wine_on_savoury_dish" in engine["pairs"]["semi"][dish_id][3], dish_id
        assert "sweet_wine_on_fish" not in engine["pairs"]["semi"][dish_id][3], dish_id
    rules = {rule["id"]: rule for rule in engine["rules"]}
    assert rules["sweet_wine_on_fish"]["weight"] == -3.0
    # Сладкое только по названию (`sugar_by_name`) — оговорка вместо жёсткого правила, и к ухе и
    # кулебяке (рыбные теги, суп и пирог) — без «Сладость спорит с блюдом» (третий круг проверки).
    for dish_id in fish:
        guess = engine["pairs"]["guess"][dish_id][3]
        assert "maybe_sweet_wine_on_fish" in guess, dish_id
        assert "sweet_wine_on_fish" not in guess and "sweet_wine_on_savoury_main" not in guess
        assert "maybe_sweet_wine_on_fish" not in engine["pairs"]["sweet"][dish_id][3], dish_id
    assert "sweet_wine_on_savoury_main" in engine["pairs"]["sweet"]["ukha"][3]
    assert "sweet_wine_on_savoury_main" in engine["pairs"]["guess"]["borsch"][3]
    assert rules["maybe_sweet_wine_on_fish"]["weight"] == -1.0
    # К несладкому не-рыбному полусладкое правило по-прежнему не трогает.
    assert "dessert_wine_on_savoury_dish" not in engine["pairs"]["semi"]["borsch"][3]
    assert "dessert_wine_on_savoury_dish" in engine["pairs"]["sweet"]["steik_ribay"][3]
    # Сладкому плов, паста и винегрет (сладость блюда 2) и горячее «солёное и жирное» — «−»
    # (проверка 24.09); сало и бородинский хлеб с салом — исключение «Лозы», как было.
    for dish_id in ("plov", "pasta_tomat", "vinegret", "pizza", "pasta_carbonara", "shaverma"):
        assert "dessert_wine_on_savoury_dish" in engine["pairs"]["sweet"][dish_id][3], dish_id
        assert "dessert_wine_on_savoury_dish" not in engine["pairs"]["semi"][dish_id][3], dish_id
    for dish_id in ("salo", "borodinsky_hleb_s_salom", "medovik"):
        assert "dessert_wine_on_savoury_dish" not in engine["pairs"]["sweet"][dish_id][3], dish_id


def test_overlay_sweet_wine_with_cheese_and_nuts(engine: dict[str, Any]) -> None:
    """Третий круг проверки 25.09: сладкое и креплёное к сырной тарелке и орехам — давняя пара:
    плюс «Сладкое к сыру и орехам» и ни одного минуса о сахаре, крепости и силе вкуса. Сациви и
    харчо с грецким орехом — не закуски, их правило не трогает; сухому оно не ставится."""
    sweet = engine["pairs"]["sweet"]
    for dish_id in ("cheese_plate", "orehi"):
        assert "sweet_wine_with_cheese_and_nuts" in sweet[dish_id][2], dish_id
        assert not sweet[dish_id][3], (dish_id, sweet[dish_id][3])
    for dish_id in ("satsivi", "kharcho", "shokolad"):
        assert "sweet_wine_with_cheese_and_nuts" not in sweet[dish_id][2], dish_id
    for wine_id in ("red", "semi", "brut"):
        pairs = engine["pairs"][wine_id]
        assert not any("sweet_wine_with_cheese_and_nuts" in pair[2] for pair in pairs.values())


def test_overlay_semisweet_spars_with_savoury(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09, вечер: полусладкое к несладкому неострому солёному — оговорка «Сладость
    спорит с солёным», а не «да» (пицца, карбонара, пельмени, котлеты, оливье).

    Острое, десерты, сыр, орехи, рыба (своё правило), сало и почти несолёное правило не трогает;
    сладкому (от 4) у него свои правила.
    """
    rule = "semisweet_wine_on_savoury_dish"
    semi, sweet = engine["pairs"]["semi"], engine["pairs"]["sweet"]
    for dish_id in ("pizza", "pasta_carbonara", "pelmeni", "kotlety_farsh", "olivier", "borsch",
                    "steik_ribay", "griby_v_smetane", "solenye_ogurcy"):  # fmt: skip
        assert rule in semi[dish_id][3], dish_id
    for dish_id in ("shaverma", "kharcho", "pastroma", "medovik", "frukty", "cheese_plate",
                    "orehi", "satsivi", "salo", "borodinsky_hleb_s_salom", "solenaya_seld",
                    "sushi", "vareniki", "bliny_smetana", "utka_s_yablokami"):  # fmt: skip
        assert rule not in semi[dish_id][3], dish_id
    assert not any(rule in pair[3] for pair in sweet.values())
    assert not any(rule in pair[3] for pair in engine["pairs"]["red"].values())
    raw = somm.run_engine({"wines": [engine_wine(SEMI_SWEET, color="white")], "overlay": False})[
        "pairs"
    ]["w"]
    assert not raw["pizza"][3], "у «Лозы» полусладкому пицца — без минусов"


def test_overlay_sweet_bridge_only_for_cold_salty_fat(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09, вечер: «Сладость оттеняет солёное» — только сало и бородинский хлеб с
    салом; у пиццы, карбонары, бургера, солянки и рыбы этот плюс стоял рядом с минусом о сахаре."""
    for wine_id in ("sweet", "semi"):
        pairs = engine["pairs"][wine_id]
        for dish_id in ("salo", "borodinsky_hleb_s_salom"):
            assert "sweet_wine_bridges_salt_fat" in pairs[dish_id][2], (wine_id, dish_id)
        fired = {d for d, pair in pairs.items() if "sweet_wine_bridges_salt_fat" in pair[2]}
        assert fired == {"salo", "borodinsky_hleb_s_salom"}, (wine_id, fired)
    raw = somm.run_engine({"wines": [engine_wine(SEMI_SWEET)], "overlay": False})["pairs"]["w"]
    for dish_id in ("pizza", "pasta_carbonara", "solenaya_seld"):
        assert "sweet_wine_bridges_salt_fat" in raw[dish_id][2], dish_id


def test_overlay_intensity_mismatch_has_a_direction(somm: Any, engine: dict[str, Any]) -> None:
    """Накладка 25.09, вечер: «разная сила вкуса» «Лозы» — два правила по направлению с тем же
    весом: вино слабее блюда — `intensity_mismatch`, сильнее — `wine_overpowers_dish`. Вместе
    они срабатывают там же, где правило «Лозы»."""
    wines = [engine_wine(RED, id="red"), engine_wine(BRUT, id="brut", color="white")]
    raw = somm.run_engine({"wines": wines, "overlay": False})["pairs"]
    new = somm.run_engine({"wines": wines, "overlay": True})["pairs"]
    for wine_id in ("red", "brut"):
        for dish_id, pair in new[wine_id].items():
            weaker = "intensity_mismatch" in pair[3]
            stronger = "wine_overpowers_dish" in pair[3]
            assert not (weaker and stronger), (wine_id, dish_id)
            assert (weaker or stronger) == ("intensity_mismatch" in raw[wine_id][dish_id][3])
    assert "wine_overpowers_dish" in new["red"]["frukty"][3]
    assert "intensity_mismatch" in new["brut"]["shashlyk_baranina"][3]
    rules = {rule["id"]: rule for rule in somm.run_engine({"wines": [], "overlay": True})["rules"]}
    assert rules["wine_overpowers_dish"]["weight"] == rules["intensity_mismatch"]["weight"]


def test_without_overlay_loza_rules_as_is(somm: Any) -> None:
    raw = somm.run_engine({"wines": [engine_wine(RED)], "overlay": False})
    assert "umami_vs_tannin_clash" in raw["pairs"]["w"]["shashlyk_baranina"][3]


def test_overlay_may_only_change_conditions(somm: Any, tmp_path: Path) -> None:
    bad = tmp_path / "overlay.json"
    bad.write_text(json.dumps({"rules": {"acid_cuts_fat": {"weight": 9}}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="acid_cuts_fat"):
        somm.run_engine({"wines": [], "overlay": str(bad)})
    bad.write_text(json.dumps({"rules": {"no_such_rule": {"dish_condition": "True"}}}))
    with pytest.raises(RuntimeError, match="no_such_rule"):
        somm.run_engine({"wines": [], "overlay": str(bad)})


def test_overlay_added_rules_are_whole_and_new(somm: Any, tmp_path: Path) -> None:
    """Своё правило сканера (`added`) — целиком и под новым `id`: правило «Лозы» им не подменить."""
    rule = {
        "reason": "проба",
        "name": "Проба",
        "dish_condition": "True",
        "wine_condition": "True",
        "weight": -1.0,
        "explanation_ru": "Проба.",
    }
    bad = tmp_path / "overlay.json"
    bad.write_text(json.dumps({"added": {"acid_cuts_fat": rule}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="acid_cuts_fat"):
        somm.run_engine({"wines": [], "overlay": str(bad)})
    partial = {k: v for k, v in rule.items() if k != "weight"}
    bad.write_text(json.dumps({"added": {"my_rule": partial}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="my_rule"):
        somm.run_engine({"wines": [], "overlay": str(bad)})
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"added": {"my_rule": rule}}), encoding="utf-8")
    rules = {r["id"] for r in somm.run_engine({"wines": [], "overlay": str(good)})["rules"]}
    assert "my_rule" in rules and "acid_cuts_fat" in rules


def test_engine_process_does_not_see_scanner_app(somm: Any, tmp_path: Path) -> None:
    """Подпроцесс сборки: пакет «Лозы» не пересекается с `app` сканера."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import importlib.util, json, sys\n"
        "print(json.dumps(importlib.util.find_spec('app') is None))\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(probe)], capture_output=True, check=True, cwd=ROOT
    )
    assert json.loads(done.stdout) is True


# ------------------------------------------------------------------ справочник, сорта, словарь
@pytest.fixture(scope="module")
def topics(somm: Any) -> list[dict[str, Any]]:
    source = json.loads((somm.REFERENCE / "knowledge.json").read_text(encoding="utf-8"))
    return somm.clean_topics(source)


def test_topics_cleaned(somm: Any, topics: list[dict[str, Any]], dishes: dict[str, Any]) -> None:
    ids = {topic["id"] for topic in topics}
    assert len(topics) == 32
    for topic in topics:
        text = f"{topic['answer']} {topic['detail']}"
        assert "Лоз" not in text and "велик" not in text.lower(), topic["id"]
        assert set(topic) == {"id", "name", "triggers", "context", "answer", "detail", "chips"}
        for chip in topic["chips"]:
            args = chip.get("args", {})
            assert chip["id"] in ("term", "dish_check", "serve", "guided"), chip
            assert args.get("topic", topic["id"]) in ids and args.get("topic") != topic["id"]
            assert args.get("dish", "borsch") in dishes
        assert "лучше" not in topic["context"] and "хуже" not in topic["context"]
    assert not somm.legal_violations(somm.strings(topics))


def test_topic_rewrite_must_match(somm: Any) -> None:
    source = json.loads((somm.REFERENCE / "knowledge.json").read_text(encoding="utf-8"))
    for topic in source["topics"]:
        if topic["id"] == "botrytis":
            topic["answer"] = topic["answer"].replace("великие", "большие")
    with pytest.raises(ValueError, match="botrytis"):
        somm.clean_topics(source)


def test_grape_notes(somm: Any) -> None:
    from app.recommend.build import load_priors

    facts = [
        wine(somm, slug="a", region="Кубань"),
        wine(somm, slug="b", region="Кубань"),
        wine(somm, slug="c", region="Крым"),
        wine(somm, slug="d", region="Крым", canonical=False),
        wine(somm, slug="e", codes=("riesling",), color="Белое", region="Крым"),
        wine(somm, slug="f", codes=("no_such_grape",)),
    ]
    notes = somm.grape_notes(facts, load_priors())
    assert set(notes) == {"saperavi", "riesling"}
    saperavi = notes["saperavi"]
    assert saperavi["count"] == 3 and saperavi["regions"] == ["Кубань", "Крым"]
    assert saperavi["label"] == "Саперави"
    assert saperavi["text"].startswith("Саперави по сорту: плотные танины")
    assert "В каталоге 3 вина с этим сортом, чаще всего — Кубань и Крым." in saperavi["text"]
    riesling = notes["riesling"]
    assert "танин" not in riesling["text"]
    assert riesling["text"].endswith("В каталоге одно вино с этим сортом, регион — Крым.")
    assert [somm.plural(n, "вино", "вина", "вин") for n in (1, 2, 5, 11, 21, 104)] == [
        "вино",
        "вина",
        "вин",
        "вин",
        "вино",
        "вина",
    ]


def test_winery_words(somm: Any) -> None:
    words = somm.winery_words(
        [
            "Фанагория",
            "Ароматное",
            "Долина Лефкадия",
            "Солнечная долина",
            "Шато Пино",
            "Гусевъ",
            "А. Гордиенко & М. Николаев",
            "Инкерманский ЗМВ",
        ],
        taken={"пино нуар", "мускат"},
    )
    assert words == ["гордиенко", "гусев", "гусевъ", "лефкадия", "николаев", "фанагория"]


# ------------------------------------------------------------------ запись файлов
def test_dump_pairs_is_json(somm: Any) -> None:
    obj = {
        "version": 1,
        "built": "проба",
        "wines": {
            "b": {"top": ["borsch"], "dishes": {"borsch": ["yes", ["acid_cuts_fat"], []]}},
            "a": {"top": [], "dishes": {}},
        },
    }
    text = somm.dump_pairs(obj)
    assert json.loads(text) == obj
    assert text == somm.dump_pairs(obj) and text.endswith(b"}\n")
    assert text.count(b"\n") == len(obj["wines"]) + 6  # одна строка на вино


def test_write_and_check(somm: Any, tmp_path: Path) -> None:
    files = {name: f'{{"name": "{name}"}}\n'.encode() for name in SOMM_FILES}
    assert somm.write_or_check(files, tmp_path, check_only=True)
    assert somm.write_or_check(files, tmp_path, check_only=False) == []
    assert somm.write_or_check(files, tmp_path, check_only=True) == []
    (tmp_path / "serve.json").write_bytes(b"{}\n")
    assert somm.write_or_check(files, tmp_path, check_only=True) == [
        "serve.json: байты расходятся с пересборкой"
    ]


# ------------------------------------------------------------------ загрузчик
def test_loader_reads_contract_fixtures_strictly() -> None:
    data = load_somm_data(None, strict=True)
    assert set(data.sources.values()) == {"fixture"}
    assert data.stats()["dishes.json"] == {"source": "fixture", "dishes": 24, "rules": 39}
    red = data.wine(SAPERAVI)
    assert red is not None and red.top[0] == "borodinsky_hleb_s_salom"
    assert red.pair("borsch").verdict == "caveat"
    assert red.pair("no_such_dish").verdict == "neutral"
    found = data.match_serve(
        color="Красное", sugar="suhoe", sparkling=False, body=4.45, body_source="grape"
    )
    assert found is not None and found.public() == {
        "temperature_c": [16, 18],
        "source": "rule",
        "rule": "red_full",
        "by_grape": True,
    }
    assert "solenaya_seld" in data.food_dishes("fish")
    assert data.topics["brut_scale"].chips and data.grapes["saperavi"].count > 0
    assert "фанагория" in data.vocab.winery_words


def test_loader_fallback_and_missing(tmp_path: Path) -> None:
    serve = json.loads((FIXTURE_DIR / "serve.json").read_text(encoding="utf-8"))
    serve["rules"] = serve["rules"][:2]
    (tmp_path / "serve.json").write_text(json.dumps(serve, ensure_ascii=False), encoding="utf-8")
    data = load_somm_data(tmp_path)
    assert data.sources["serve.json"] == "data" and len(data.serve) == 2
    assert data.sources["pairs.json"] == "fixture"
    empty = load_somm_data(tmp_path, fallback=None)
    assert empty.sources["pairs.json"] == "missing" and not empty.pairs and not empty.dishes
    assert empty.stats()["pairs.json"] == {"source": "missing", "wines": 0}


def test_loader_broken_file(tmp_path: Path) -> None:
    (tmp_path / "vocab.json").write_text("{", encoding="utf-8")
    pairs = json.loads((FIXTURE_DIR / "pairs.json").read_text(encoding="utf-8"))
    pairs["wines"][SAPERAVI]["dishes"]["borsch"] = [
        "yes",
        ["acid_cuts_fat"],
        ["umami_vs_tannin_clash"],
    ]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    data = load_somm_data(tmp_path)
    assert data.sources["vocab.json"] == "missing" and not data.vocab.grapes
    assert data.sources["pairs.json"] == "missing" and not data.pairs
    assert data.sources["dishes.json"] == "fixture"
    with pytest.raises(ValueError, match="vocab.json"):
        load_somm_data(tmp_path, strict=True)


def test_loader_drops_foreign_links(tmp_path: Path) -> None:
    """Пары из данных, блюда из заглушки: чужие блюда отброшены, сервис не падает."""
    pairs = json.loads((FIXTURE_DIR / "pairs.json").read_text(encoding="utf-8"))
    pairs["wines"][SAPERAVI]["dishes"]["plov"] = ["yes", ["acid_cuts_fat"], []]
    pairs["wines"][SAPERAVI]["top"] = ["plov", "gus"]
    (tmp_path / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
    data = load_somm_data(tmp_path)
    red = data.wine(SAPERAVI)
    assert red is not None and red.top == ("gus",) and "plov" not in red.dishes
    with pytest.raises(ValueError, match="plov"):
        load_somm_data(tmp_path, strict=True)


def test_loader_shares_pairs() -> None:
    """Равные пары разных вин — один объект: 172 тысячи записей держат единицы мегабайт."""
    data = load_somm_data(None, strict=True)
    objects: dict[Any, set[int]] = {}
    for entry in data.pairs.values():
        for value in entry.dishes.values():
            objects.setdefault(value, set()).add(id(value))
    assert len(objects) < sum(len(entry.dishes) for entry in data.pairs.values())
    assert all(len(ids) == 1 for ids in objects.values())


# ------------------------------------------------------------------ полная сборка
def _inputs() -> tuple[Path, Path, Path] | None:
    settings = get_settings()
    paths = (
        settings.dataset_dir / "strapi_output0709.csv",
        settings.data_dir / "gt" / "gt_tokens.jsonl",
        settings.data_dir / "catalog" / "wine_groups.json",
    )
    return paths if all(path.is_file() for path in paths) else None


@pytest.fixture(scope="module")
def full(somm: Any) -> Any:
    inputs = _inputs()
    if inputs is None:
        pytest.skip("нет выгрузки организатора: SVS_DATASET_DIR и SVS_DATA_DIR")
    started = time.perf_counter()
    result = somm.build(*inputs)
    result.seconds["total"] = time.perf_counter() - started
    return result


def test_full_build_acceptance(somm: Any, full: Any) -> None:
    """Приёмка дорожки: блюда и подача у каждого вина, конфликтов в тройке нет, право чисто."""
    report = somm.quality(full)
    assert full.seconds["total"] < 180
    assert len(full.facts) == 2103
    assert report["no_dish"] == [] and report["no_dish_pool"] == [] and report["top_short"] == []
    # Третий круг проверки 25.09: сладкое к рыбе — «скорее нет», к сыру и орехам — «да»; сладкое
    # только по названию к рыбе — оговорка.
    assert report["sweet_fish_soft"] == [] and report["sweet_cheese"] == []
    assert report["maybe_sweet_fish"] == []
    assert len(full.serves) == len(full.facts)
    assert report["conflicts_top3"] == [] and report["sweet_fish"] == []
    assert report["dry_guess"] == [] and report["false_claims"] == []
    assert report["forbidden"] == {
        "expert_score": 0,
        "price": 0,
        "руб": 0,
        "%": 0,
        "typical_price": 0,
    }
    assert report["legal"] == []
    # Сахар, игристость и крепость — как в договоре «после поиска», §3: пет-нат через дефис и
    # подчёркивание — игристое (+2), год урожая в хвосте slug — не крепость (daniel-22, −1).
    assert sum(item.sugar is not None for item in full.facts) == 1727
    assert sum(item.sparkling for item in full.facts) == 297
    assert sum(item.abv is not None for item in full.facts) == 1567
    by_slug = {item.slug: item for item in full.facts}
    assert by_slug["denisov_pet_nat_rubin"].sparkling
    assert by_slug["daniel-22"].abv is None
    typo = "abrau-dyurso-abrau-kupazh-krasnyy-polsusladkoe-kaberne-sovinon-krasnoe-polusladkoe-11"
    assert by_slug[typo].sugar == "polusladkoe"
    # Креплёные, десертные и мускатные названия без сахара — подача 10–14 °C (24 вина из 27: у
    # «Мускат Оранж» название оранжевого, у «Жемчужная 9 … Пино гри» — «сухость» в описании, а
    # «Мускат позднего сбора розовый» — «Сладкое розовое вино» в описании, сладкое по карточке).
    sweet_name = [slug for slug, serve in full.serves.items() if serve.rule.id == "sweet_name"]
    assert len(sweet_name) == 24 and {"portvejn-krymskij", "pozdnij-sbor-beloe"} <= set(sweet_name)
    assert {
        "perovskih_muskat_orange",
        "zhemchuzhnaya-9-muskat-rozovyj-pino-gri",
        "muskat-pozdnego-sbora-rozovyj",
    }.isdisjoint(sweet_name)
    guessed = {item.slug for item in full.facts if somm.sweet_by_name(item)}
    assert guessed == set(sweet_name)
    # Сахар по «Описанию» — у одного вина выгрузки; карточка «после поиска» его не видит (1 727).
    described = [item for item in full.facts if item.sugar != somm.sugar_by_card(item)]
    assert [item.slug for item in described] == ["muskat-pozdnego-sbora-rozovyj"]
    assert somm.sugar_by_card(described[0]) == "sladkoe" == somm.rule_sugar(described[0])
    assert full.serves["muskat-pozdnego-sbora-rozovyj"].rule.id == "sweet"
    assert full.serves["chateau-tamagne-grand-dessert-nectar"].rule.temperature_c == (10, 14)
    # Демо-вино: мясо и птица, а не бородинский хлеб с салом (проверка 24.09).
    categories = {full.dishes[dish_id]["category"] for dish_id in full.tops[SAPERAVI][:3]}
    assert categories <= {"main_meat", "main_poultry", "fast_food"}, full.tops[SAPERAVI]


def test_full_build_sweet_wines_never_get_fish(somm: Any, full: Any) -> None:
    """Сладкое и полусладкое — и название креплёного или десертного вина без сахара в выгрузке
    (`rule_sugar`, проверка 25.09, вечер: портвейну «подходили» устрицы) — без рыбы."""
    fish = {dish_id for dish_id, dish in full.dishes.items() if somm.food_of(dish) == "fish"}
    port = next(item for item in full.facts if item.slug == "portvejn-krymskij")
    assert port.sugar is None and somm.rule_sugar(port) == "sladkoe"
    # Проверка `acc-somm3`: «Сладкое розовое вино» в «Описании» — сладкое по карточке: ко всем 18
    # рыбным блюдам «скорее нет» и первым «Сладость спорит с рыбой», а не оговорка (было 17 из 18).
    late = full.pairs["muskat-pozdnego-sbora-rozovyj"]
    seafood = [d for d, dish in full.dishes.items() if set(dish["flavor_tags"]) & somm.FISH_TAGS]
    assert len(seafood) == 18
    for dish_id in seafood:
        assert late[dish_id].verdict == "no", dish_id
        assert late[dish_id].minus[0] == "sweet_wine_on_fish", dish_id
        assert "maybe_sweet_wine_on_fish" not in late[dish_id].minus, dish_id
    for item in full.facts:
        if somm.rule_sugar(item) not in somm.SWEET_SUGARS:
            continue
        assert not fish & set(full.tops[item.slug]), item.slug
        for dish_id in fish:
            assert full.pairs[item.slug][dish_id].verdict != "yes", (item.slug, dish_id)
        # Несладкое блюдо сладкому вину — только острое по правилу «Сладость гасит остроту».
        for dish_id in full.tops[item.slug]:
            if full.dishes[dish_id]["category"] in somm.DESSERT_CATEGORIES:
                continue
            if not somm.cheese_or_nuts(full.dishes[dish_id]):
                assert somm.SPICE_RULE in full.pairs[item.slug][dish_id].plus, (item.slug, dish_id)


def test_full_build_rules_tell_the_truth_about_the_dish(somm: Any, full: Any) -> None:
    """Подпись правила — правда о блюде, к которому её ставит шаблон (независимая проверка
    25.09): «бульон» — только супам и блюдам на бульоне, «жареное» — только жареному, «десерт» —
    только десертам, варенью и мёду; вину без сахара в выгрузке сладкие блюда — `neutral` без
    правил."""
    dessert_rules = {"dry_wine_on_dessert", "wine_sweeter_than_dish", "semisweet_wine_on_dessert"}
    by_slug = {item.slug: item for item in full.facts}
    wrong = []
    for slug, row in full.pairs.items():
        for dish_id, pair in row.items():
            dish = full.dishes[dish_id]
            tags, category = set(dish["flavor_tags"]), dish["category"]
            rules = {*pair.plus, *pair.minus}
            broth = category == "soup" or "meat_broth" in tags
            if "umami_vs_tannin_clash" in rules and not broth:
                wrong.append((slug, dish_id, "бульон"))
            if "bubbles_cut_fat_and_fry" in rules and "fried" not in tags:
                wrong.append((slug, dish_id, "жареное"))
            sweet_dish = category in somm.DESSERT_CATEGORIES
            if rules & dessert_rules and not sweet_dish:
                wrong.append((slug, dish_id, "десерт"))
            unknown = somm.rule_sugar(by_slug[slug]) is None
            if sweet_dish and unknown and (pair.verdict != "neutral" or rules):
                wrong.append((slug, dish_id, "сахар"))
    assert not wrong, wrong[:10]
    assert somm.false_claims(full) == []


def test_full_build_matches_data_dir(somm: Any, full: Any) -> None:
    """`--check`: `data/somm` на машине — ровно эта сборка, байт в байт, и читается строго."""
    out = get_settings().data_dir / "somm"
    if not all((out / name).is_file() for name in SOMM_FILES):
        pytest.skip(f"нет {out}: соберите scripts/build_somm.py")
    assert somm.write_or_check(full.files, out, check_only=True) == []
    started = time.perf_counter()
    data = load_somm_data(out, fallback=None, strict=True)
    assert time.perf_counter() - started < 3.0
    assert set(data.sources.values()) == {"data"}
    assert len(data.pairs) == 2103 and len(data.dishes) == 82 and len(data.rules) == 39


@pytest.mark.parametrize(
    ("name", "broken"),
    [
        ("serve.json", {"version": 1, "source": "x", "rules": [{"id": "a"}]}),
        ("serve.json", {"version": 1, "source": "x", "rules": "не список"}),
        ("knowledge.json", {"version": 1, "source": "x", "topics": [], "grapes": []}),
        ("dishes.json", []),
    ],
)
def test_loader_survives_unexpected_shapes(tmp_path: Path, name: str, broken: Any) -> None:
    """Сервис стартует всегда: неожиданная форма — `missing`, а в строгом режиме — ошибка."""
    (tmp_path / name).write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    data = load_somm_data(tmp_path)
    assert data.sources[name] == "missing"
    with pytest.raises(ValueError, match=name):
        load_somm_data(tmp_path, strict=True)


def test_digest_ignores_line_endings(somm: Any, tmp_path: Path) -> None:
    """`built` не зависит от `core.autocrlf`: справочник с CRLF — тот же вход."""
    lf, crlf = tmp_path / "lf" / "a.json", tmp_path / "crlf" / "a.json"
    for path, body in ((lf, b'{\n  "a": 1\n}\n'), (crlf, b'{\r\n  "a": 1\r\n}\r\n')):
        path.parent.mkdir()
        path.write_bytes(body)
    assert somm.digest([lf]) == somm.digest([crlf])
    crlf.write_bytes(b'{\r\n  "a": 2\r\n}\r\n')
    assert somm.digest([lf]) != somm.digest([crlf])
