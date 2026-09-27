"""«Не тупик» и подсказка экрана на записанных ответах поля (план, Д2, п. 7).

Кадры — настоящие ответы сервиса на бутылки вне каталога (`tests/fixtures/after_recorded/`):
строки и поля чтения VLM, top-5 после H5 и счёт CV. Справочник — двенадцать позиций каталога
с теми же винодельнями, сортами, цветом и сахаром, что в `wines.jsonl` и `gt_tokens.jsonl`:
винодельни этих кадров, три «шато» для родового токена и чужие игристые брют и оранжевое для
похожих.

    R030  Alveus Ультра Кюве Оранж Брют (Фанагория): винодельни на этикетке нет, только линия
    R002  Шато Пино «Беленькое»: винодельня каталога, позиции нет
    R058  Усадьба Дивноморское «Вечерница»: винодельня каталога, сканер уверенно ошибся
    R086  Bel Colle Barolo: ни одного поля не прочитано
    M054  Château de Châtaignier: словарь дал только «chateau», сканер предлагал Le Grand Vostock

Счёт CV лучшей серии (zmax) у кадров: R030 0,777, R002 0,807, R058 0,949, R086 0,692, M054 0,760.
Подсказка `suggest_not_found` (порог 0,8024) поднята у R030, R086 и M054.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from catalog import wine_record
from reco_env import row
from test_after_fixtures import ABS_PATH, NOTICE, STOP, strings

from app.api.after_layer import (
    SUGGEST_NOT_FOUND_VISUAL_MAX,
    AfterSearch,
    ByLabelBody,
    after_block,
)
from app.api.cards import CatalogCards
from app.api.service import Confidence, ScanResult, TopItem
from app.recommend.catalog import RecoCatalog, wine_of
from app.recommend.content_filter import check
from app.resolve.attrs import CatalogAttrs

FRAMES = Path(__file__).resolve().parents[1] / "fixtures" / "after_recorded" / "ooc_frames.json"

FANAGORIA = {"winery": "Фанагория", "key_tokens": ("фанагория",),
             "variants": ("fanagoria", "fanagoriya")}  # fmt: skip
SHATO_PINO = {"winery": "Шато Пино", "key_tokens": ("шато", "пино"),
              "variants": ("shato pino", "пино")}  # fmt: skip
DIVNOMORSKOE = {"winery": "Усадьба Дивноморское", "key_tokens": ("дивноморское",),
                "variants": ("divnomorskoe", "дивноморское")}  # fmt: skip
VOSTOCK = {"winery": "Château Le Grand Vostock",
           "key_tokens": ("chateau", "le", "grand", "vostock"),
           "variants": ("восток", "восток гран", "гранд", "ле", "чатеау", "шато")}  # fmt: skip
ANDRE = {"winery": "Chateau Andre", "key_tokens": ("chateau", "andre"),
         "variants": ("andre", "андре", "чатеау", "шато")}  # fmt: skip
TALU = {"winery": "Chateau de Talu", "key_tokens": ("chateau", "de", "talu"),
        "variants": ("talu", "де", "талу", "талю", "чатеау", "шато")}  # fmt: skip
ABRAU = {"winery": "Абрау-Дюрсо", "key_tokens": ("абрау", "дюрсо"),
         "variants": ("abrau dyurso", "абрау")}  # fmt: skip
BELBEK = {"winery": "Бельбек", "key_tokens": ("бельбек",), "variants": ("belbek",)}

#: (slug, винодельня, название, сорта, цвет, сахар, игристое, крепость, флаги разметки)
WINES: list[tuple[Any, ...]] = [
    ("fanagoriya-alveus-ultra-cuvee-brut-shardone-beloe-bryut-12", FANAGORIA,
     "Alveus Ultra Cuvee Brut", ["riesling", "chardonnay"], "Белое", "brut", True, 12.0, ()),
    ("fanagoriya-alveus-ultra-cuvee-bryut-rozovoe-merlo-igristoe-bryut-rozovoe-12", FANAGORIA,
     "Alveus Ultra Cuvee. Брют розовое", ["merlot", "pinot_noir"], "Розовое", "brut", True,
     12.0, ()),
    ("shato-pino-sovinon-blan-semilon-beloe-suhoe-115", SHATO_PINO,
     "Совиньон Блан - Семильон", ["sauvignon_blanc"], "Белое", "suhoe", False, 11.5,
     ("name_grape_only",)),
    ("shato-pino-shiraz-krasnoe-suhoe-14", SHATO_PINO, "Шираз", ["syrah"], "Красное", "suhoe",
     False, 14.0, ("name_grape_only",)),
    ("usadba-divnomorskoe-yuzhnyy-les-merlo-krasnoe-suhoe-137", DIVNOMORSKOE, "Южный Лес",
     ["merlot"], "Красное", "suhoe", False, 13.7, ()),
    ("usadba-divnomorskoe-solnechnyy-veter-shardone-beloe-suhoe-125", DIVNOMORSKOE,
     "Солнечный Ветер", ["chardonnay"], "Белое", "suhoe", False, 12.5, ()),
    ("chteau-le-grand-vostock-cabernet-sauvignon-kaberne-sovinon-krasnoe-suhoe-14", VOSTOCK,
     "Cabernet Sauvignon", ["cabernet_sauvignon"], "Красное", "suhoe", False, 14.0,
     ("name_grape_only",)),
    ("chateau-andre-merlo-krasnoe-suhoe-14", ANDRE, "Мерло", ["merlot"], "Красное", "suhoe",
     False, 14.0, ("name_grape_only",)),
    ("chateau-de-talu-uroki-frantsuzskogo-kaberne-sovinon-krasnoe-suhoe-14", TALU,
     "Уроки французского. Каберне Совиньон", ["cabernet_sauvignon"], "Красное", "suhoe", False,
     14.0, ()),
    ("abrau-dyurso-abrau-durso-reserve-brut-shardone-beloe-bryut-115", ABRAU,
     "Abrau-Durso Reserve Brut", ["pinot_blanc", "riesling", "chardonnay"], "Белое", "brut",
     True, 11.5, ()),
    ("abrau-dyurso-brut-dor-rose-pino-nuar-rozovoe-bryut-12", ABRAU, "Brut d'Or Rose",
     ["pinot_noir"], "Розовое", "brut", True, 12.0, ()),
    ("belbek-muskat-oranzh-muskat-belyy-oranzhevoe-suhoe-125", BELBEK, "Мускат Оранж",
     ["muscat"], "Оранжевое", "suhoe", False, 12.5, ("name_grape_only",)),
]  # fmt: skip


@pytest.fixture(scope="module")
def search() -> AfterSearch:
    records = []
    rows = []
    for slug, winery, title, grapes, color, sugar, sparkling, abv, flags in WINES:
        facts = {"grapes": grapes, "color": color, "sugar": sugar}
        region = "Крым" if winery is BELBEK else "Кубань"
        records.append(wine_record(slug, **winery, name=title, abv=[abv], flags=flags, **facts))
        name = winery["winery"]
        item = row(slug, name, title=title, sparkling=sparkling, abv=abv, region=region, **facts)
        item["photo"] = f"{slug}.webp"  # в пуле похожих только вина с фото; файла нет
        rows.append(item)
    catalog = RecoCatalog([wine_of(item) for item in rows])
    return AfterSearch(catalog, CatalogCards.build(records), CatalogAttrs.from_records(records))


def frames() -> dict[str, dict[str, Any]]:
    data = json.loads(FRAMES.read_text(encoding="utf-8"))
    return {frame["query_id"].split("-")[0]: frame for frame in data["frames"]}


def result_of(frame: dict[str, Any]) -> ScanResult:
    """Ответ сервиса на записанный кадр: top-5, вероятность, чтение и счёт CV."""
    top5 = [TopItem(slug=slug, score=p) for slug, p in frame["top5"]]
    return ScanResult(
        slug=top5[0].slug,
        confidence=Confidence(top1=top5[0].score),
        top5=top5,
        outcome=frame["outcome"],
        evidence={
            "vlm": frame["vlm"],
            "cv": {"top5": [{"slug": top5[0].slug, "score": frame["cv_top1"]}]},
        },
    )


def by_label(search: AfterSearch, read: dict[str, Any]) -> dict[str, Any]:
    body = search.by_label(ByLabelBody(**read), limit=3, order="reco")
    for text in strings(body):
        assert not STOP.search(text) and not ABS_PATH.search(text), text
    assert "%" not in json.dumps(body, ensure_ascii=False)
    assert all(check(note["text"]).clean for note in body["notes"])
    assert body["notice"] == NOTICE
    return body


def codes(body: dict[str, Any]) -> list[str]:
    return [note["code"] for note in body["notes"]]


def test_fixture_has_no_paths_and_all_plan_cases():
    text = FRAMES.read_text(encoding="utf-8")
    assert not ABS_PATH.search(text)
    assert set(frames()) == {"R030", "R002", "R058", "R086", "M054"}


def test_alveus_orange_brut_winery_is_not_on_the_label(search):
    """ALVEUS / УЛЬТРА КЮВЕ / ОРАНЖ / БРЮТ: винодельни на этикетке нет, остальное прочитано."""
    after = after_block(result_of(frames()["R030"]), search)
    assert (after["state"], after["reasons"]) == ("check", ["ambiguous", "visual_low"])
    assert after["suggest_not_found"] is True
    assert after["read"] == {"winery": None, "color": "Оранжевое", "sugar": "brut", "grapes": [],
                             "abv": None, "sparkling": True}  # fmt: skip
    assert after["read_label"] == "игристое · оранжевое · брют"
    assert after["winery_in_catalog"] is None and after["winery_slugs"] == []

    body = by_label(search, after["read"])
    assert body["winery"] == {"name": None, "in_catalog": None, "count": 0}
    assert body["same_winery"] == []
    assert body["notes"][0] == {
        "code": "winery_unknown",
        "text": "Винодельню по этикетке определить не удалось",
    }
    assert "relaxed_color" in codes(body)
    # оранжевых игристых в каталоге нет — брют других цветов; линия Alveus находится фактами
    similar = body["similar"]
    assert len(similar) == 3
    assert all(tile["sparkling"] and "брют" in tile["style_label"] for tile in similar)
    assert all(tile["reasons"][0] == "Тоже игристое брют" for tile in similar)
    assert "Alveus Ultra Cuvee Brut" in [tile["name"] for tile in similar]


def test_chateau_pinot_belenkoe_reads_the_catalog_winery(search):
    """«CHÂTEAU / PINOT / Беленькое»: Шато Пино в каталоге, «Беленького» у неё нет."""
    frame = frames()["R002"]
    after = after_block(result_of(frame), search)
    assert (after["state"], after["reasons"]) == ("check", ["ambiguous"])  # счёт CV 0,807
    assert after["suggest_not_found"] is False
    assert after["read"]["winery"] == "Шато Пино" and after["winery_in_catalog"] is True
    assert after["winery_slugs"] == [
        "shato-pino-sovinon-blan-semilon-beloe-suhoe-115",
        "shato-pino-shiraz-krasnoe-suhoe-14",
    ]
    # Родовой «chateau» рядом не подтверждает чужую винодельню, даже если её предлагал сканер.
    match = search.resolve_winery(frame["vlm"]["fields"]["winery"], prefer=["chateau de talu"])
    assert (match.name, match.in_catalog) == ("Шато Пино", True)

    body = by_label(search, after["read"])
    assert body["winery"] == {"name": "Шато Пино", "in_catalog": True, "count": 2}
    assert {tile["winery"] for tile in body["same_winery"]} == {"Шато Пино"}
    assert codes(body) == ["no_facts"] and body["similar"] == []


def test_divnomorskoe_vechernitsa_same_winery(search):
    """Сканер уверенно (p = 0,92) отдал «Южный Лес»; винодельня прочитана верно."""
    after = after_block(result_of(frames()["R058"]), search)
    assert (after["state"], after["reasons"]) == ("found", [])
    assert after["suggest_not_found"] is False  # счёт CV 0,95: подсказка ошибку не ловит
    assert after["read"]["winery"] == "Усадьба Дивноморское"
    assert after["winery_in_catalog"] is True and len(after["winery_slugs"]) == 2

    body = by_label(search, after["read"])
    assert body["winery"] == {"name": "Усадьба Дивноморское", "in_catalog": True, "count": 2}
    assert [tile["name"] for tile in body["same_winery"]] == ["Солнечный Ветер", "Южный Лес"]
    assert codes(body) == ["no_facts"]


def test_barolo_nothing_read(search):
    """BELCOLLE / BAROLO / DOCG: ни винодельни, ни цвета, ни сорта — честные фразы."""
    after = after_block(result_of(frames()["R086"]), search)
    # p = 0,68: по правилу Д1 это `found` на чужом вине; счёт CV 0,69 ниже порога подсказки
    assert (after["state"], after["reasons"]) == ("check", ["visual_low"])
    assert after["suggest_not_found"] is True
    assert after["read"] == {"winery": None, "color": None, "sugar": None, "grapes": [],
                             "abv": None, "sparkling": None}  # fmt: skip
    assert after["read_label"] is None and after["winery_in_catalog"] is None

    body = by_label(search, after["read"])
    assert codes(body) == ["winery_unknown", "no_facts"]
    assert body["notes"][0]["text"] == "Винодельню на этикетке не прочитали"
    assert body["same_winery"] == [] and body["similar"] == []


def test_chateau_de_chataignier_generic_token_is_not_le_grand_vostock(search):
    """Правка (a) и (b): «chateau» не подтверждает винодельню кандидата, имя — из строк."""
    frame = frames()["M054"]
    assert frame["vlm"]["fields"]["winery"] == ["chateau", "чатеау"]
    assert frame["top5"][0][0].startswith("chteau-le-grand-vostock")
    after = after_block(result_of(frame), search)
    assert after["suggest_not_found"] is True and "visual_low" in after["reasons"]
    assert after["read"]["winery"] == "CHÂTEAU de CHÂTAIGNIER"
    assert after["winery_in_catalog"] is False and after["winery_slugs"] == []
    assert after["read_label"] == "CHÂTEAU de CHÂTAIGNIER · сухое"

    body = by_label(search, after["read"])
    assert body["winery"] == {"name": "CHÂTEAU de CHÂTAIGNIER", "in_catalog": False, "count": 0}
    assert body["notes"][0] == {"code": "winery_unknown", "text": "Этой винодельни в каталоге нет"}
    assert body["same_winery"] == [] and body["similar"]


def test_suggest_not_found_on_recorded_frames(search):
    """Порог калибровки Д3: подсказка на трёх кадрах вне каталога из пяти, и ни разу не экран
    `not_found`; slug ответа слой не трогает."""
    assert search.suggest_max == SUGGEST_NOT_FOUND_VISUAL_MAX
    flagged = set()
    for name, frame in frames().items():
        result = result_of(frame)
        before = result.predict_body()
        after = after_block(result, search)
        assert after["state"] != "not_found"
        assert ("visual_low" in after["reasons"]) is after["suggest_not_found"]
        assert result.predict_body() == before
        if after["suggest_not_found"]:
            flagged.add(name)
            assert after["state"] == "check"
    assert flagged == {"R030", "R086", "M054"}


def test_not_found_is_off_by_default(search):
    """Калибровка Д2 дала полноту ниже 50 %: сервер `not_found` не возвращает."""
    assert search.not_found_max is None
    for frame in frames().values():
        assert after_block(result_of(frame), search)["state"] != "not_found"


def test_not_found_rule_when_switched_on(search, monkeypatch):
    """Правило целиком, если порог задан: обе ветки и то, что его не включает."""
    monkeypatch.setattr(search, "not_found_max", 0.85)
    recorded = frames()
    # винодельни нет в каталоге, счёт CV 0,76
    after = after_block(result_of(recorded["M054"]), search)
    assert (after["state"], after["reasons"]) == (
        "not_found",
        ["winery_not_in_catalog", "visual_low"],
    )
    # Шато Пино прочитана, но спорить с её позициями нечем — остаётся «проверьте»
    assert after_block(result_of(recorded["R002"]), search)["state"] == "check"
    # с «брют» спорят обе сухие позиции, но их названия условные («Шираз») — судить нельзя
    frame = json.loads(json.dumps(recorded["R002"]))
    frame["vlm"]["fields"]["sugar"] = ["brut"]
    assert after_block(result_of(frame), search)["state"] == "check"
    # Дивноморское с «брют»: обе сухие позиции спорят, названия свои — ни одна не сходится
    frame = json.loads(json.dumps(recorded["R058"]))
    frame["vlm"]["fields"]["sugar"] = ["brut"]
    after = after_block(result_of(frame), search)
    assert (after["state"], after["reasons"]) == ("found", [])  # счёт CV 0,95 — выше порога
    frame["cv_top1"] = 0.8
    after = after_block(result_of(frame), search)
    # 0,8 ниже и порога подсказки (0,8024): причина `visual_low` добавляется и к `not_found`
    assert (after["state"], after["reasons"]) == ("not_found", ["winery_no_match", "visual_low"])
