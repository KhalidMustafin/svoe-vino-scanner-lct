"""Заглушки «после поиска» (`tests/fixtures/after/`) соответствуют договору `docs/api-after-search.md`.

Страница (дорожка B) верстается на этих файлах до того, как готов бэкенд, поэтому расхождение
заглушки с договором — это расхождение страницы с будущим API. Здесь же — правовые проверки
договора: без стоп-слов, без `%` в блоках рекомендаций, без абсолютных путей машины.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.api.after_layer import SUGGEST_NOT_FOUND_VISUAL_MAX
from app.reading.contracts import Color, SugarClass

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "after"
SCANS = ("scan_found", "scan_check", "scan_not_found", "scan_suggest", "scan_error")
RECO = ("similar", "similar_plain", "by_label", "shelf", "shelf_empty")
ALL = (*SCANS, "wine_card", *RECO)

PREDICT_KEYS = {
    "slug",
    "confidence",
    "margin",
    "top5",
    "outcome",
    "degraded",
    "timings_ms",
    "error",
}
AFTER_KEYS = {
    "state",
    "reasons",
    "read",
    "read_label",
    "winery_in_catalog",
    "winery_slugs",
    "suggest_not_found",
    "question",
}
READ_KEYS = {"winery", "color", "sugar", "grapes", "abv", "sparkling"}
#: Вопрос сомелье экрана `check` (договор сомелье, §5).
QUESTION_FIELDS = ("sugar", "color", "sparkling", "grapes", "abv", "name")
CANDIDATE_KEYS = {"slug", "name", "winery", "photo_url"}
TILE_KEYS = {
    "slug",
    "name",
    "winery",
    "region",
    "style_label",
    "sparkling",
    "grapes",
    "photo_url",
    "portal_url",
    "reasons",
}
#: Карточка без портала (договор, §3): только выгрузка организатора и наши правила.
CARD_KEYS = {
    "slug",
    "name",
    "winery",
    "region",
    "grapes",
    "category",
    "color",
    "sugar",
    "sugar_class",
    "photo_name",
    "color_label",
    "sugar_label",
    "sparkling",
    "style_label",
    "description",
    "description_src",
    "photo_url",
    "portal_url",
    "alcohol",
    "alcohol_max",
    "alcohol_src",
}
#: Поля снимка портала и живого портала: после решения 24.09 их в ответах нет.
PORTAL_KEYS = {
    "dishes",
    "temperature",
    "category_gradient",
    "live_category",
    "published",
    "icon_url",
}
SUGAR_WORDS = {
    "brut_nature": "брют натюр",
    "extra_brut": "экстра брют",
    "brut": "брют",
    "suhoe": "сухое",
    "polusuhoe": "полусухое",
    "polusladkoe": "полусладкое",
    "sladkoe": "сладкое",
}
NOTICE = "Применяются рекомендательные технологии"
PORTAL = "https://vino-svoe.ru/wines/"
STOP = re.compile(
    r"купи|цен[аы]|₽|руб\.|лучш|вино недели|рейтинг|publicrating|скидк", re.IGNORECASE
)
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|/Users/|/home/")
ERROR_CODES = {"no_image", "bad_request", "too_large", "decode", "cv", "not_ready", "internal"}


def load(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in strings(v)]
    return []


def day1_state(body: dict[str, Any]) -> tuple[str, list[str]]:
    """Правило `after.state` на Д1 из договора (§2), сверху вниз."""
    if body["outcome"] == "error":
        return "error", [str(body["error"]).split(":", 1)[0]]
    if body["outcome"] == "out_of_catalog":
        return "check", ["abstain"]
    top1 = body["confidence"]["top1"]
    if top1 is None:
        return "check", ["no_confidence"]
    if body["outcome"] == "ambiguous":
        return "check", ["ambiguous"]
    if top1 < 0.5:
        return "check", ["low_confidence"]
    return "found", []


def best_visual(body: dict[str, Any]) -> float | None:
    """Счёт CV лучшей серии: `evidence.abstain.best_visual`, иначе счёт CV top-1 (договор, §2)."""
    evidence = body.get("evidence") or {}
    value = (evidence.get("abstain") or {}).get("best_visual")
    if value is None:
        top5 = (evidence.get("cv") or {}).get("top5") or []
        value = top5[0].get("score") if top5 else None
    return value


def suggested(body: dict[str, Any]) -> bool:
    """`after.suggest_not_found` по договору: счёт CV ниже порога, экран не `error`."""
    visual = best_visual(body)
    return (
        body["outcome"] != "error"
        and visual is not None
        and SUGGEST_NOT_FOUND_VISUAL_MAX is not None
        and visual < SUGGEST_NOT_FOUND_VISUAL_MAX
    )


def expected_state(body: dict[str, Any]) -> tuple[str, list[str]]:
    """Экран и причины по договору (§2): правило Д1 плюс подсказка `visual_low`."""
    state, reasons = day1_state(body)
    if suggested(body):
        return ("check" if state == "found" else state), [*reasons, "visual_low"]
    return state, reasons


def test_fixture_set_is_complete() -> None:
    assert {p.stem for p in FIXTURES.glob("*.json")} == set(ALL)


@pytest.mark.parametrize("name", ALL)
def test_no_stop_words_or_machine_paths(name: str) -> None:
    for text in strings(load(name)):
        assert not STOP.search(text), (name, text)
        assert not ABS_PATH.search(text), (name, text)


@pytest.mark.parametrize("name", SCANS)
def test_scan_body_adds_fields_without_touching_predict(name: str) -> None:
    body = load(name)
    assert PREDICT_KEYS <= set(body)
    assert set(body) - PREDICT_KEYS == {"evidence", "card", "candidates", "after"}
    after = body["after"]
    assert set(after) == AFTER_KEYS
    assert set(after["read"]) == READ_KEYS
    assert after["suggest_not_found"] is suggested(body)
    assert ("visual_low" in after["reasons"]) is after["suggest_not_found"]
    read = after["read"]
    assert read["color"] in {c.value for c in Color} | {None}
    assert read["sugar"] in {s.value for s in SugarClass} | {None}
    assert read["sparkling"] in (True, None)  # false сервис не ставит
    assert isinstance(read["grapes"], list)
    assert len(body["candidates"]) <= 5
    for cand in body["candidates"]:
        assert set(cand) == CANDIDATE_KEYS
        assert cand["photo_url"] == f"/v1/wines/{cand['slug']}/photo"
    assert [c["slug"] for c in body["candidates"]] == [t["slug"] for t in body["top5"]]
    if body["slug"]:
        assert body["candidates"][0]["slug"] == body["slug"]
        assert body["card"]["slug"] == body["slug"]
    else:
        assert body["card"] is None
    if body["outcome"] == "error":
        assert body["candidates"] == []
    if after["winery_in_catalog"] is not True:
        assert after["winery_slugs"] == []


@pytest.mark.parametrize("name", ["scan_found", "scan_check", "scan_suggest", "scan_error"])
def test_state_follows_day1_rule(name: str) -> None:
    body = load(name)
    state, reasons = expected_state(body)
    assert (body["after"]["state"], body["after"]["reasons"]) == (state, reasons)
    if state == "error":
        assert set(reasons) <= ERROR_CODES


def test_not_found_keeps_slug_and_card() -> None:
    """Д2: подсказка «нет в каталоге» не трогает slug — predict прежний."""
    body = load("scan_not_found")
    assert body["after"]["state"] == "not_found"
    assert body["after"]["reasons"] == ["winery_no_match"]
    assert body["slug"] and body["card"]
    assert body["after"]["winery_in_catalog"] is True and body["after"]["winery_slugs"]


@pytest.mark.parametrize("name", SCANS)
def test_check_question_only_on_check(name: str) -> None:
    """`after.question` (договор сомелье, §5): объект только на экране `check`, варианты — из
    кандидатов, от двух до четырёх, текст — вопрос без стоп-слов."""
    from app.recommend.content_filter import check

    body = load(name)
    question = body["after"]["question"]
    if body["after"]["state"] != "check":
        assert question is None
        return
    assert question is not None
    assert set(question) == {"field", "text", "options"}
    assert question["field"] in QUESTION_FIELDS and question["text"].endswith("?")
    assert check(question["text"]).clean
    slugs = [cand["slug"] for cand in body["candidates"]]
    assert 2 <= len(question["options"]) <= 4
    assert all(option["slug"] in slugs and option["label"] for option in question["options"])
    field = question["field"]
    if field in ("sugar", "color", "sparkling", "abv"):
        assert body["after"]["read"][field] is None, "прочитанное поле не спрашивают"


def test_suggest_not_found_is_a_check_with_slug_and_card() -> None:
    """Д3: чужое вино (R086, Bel Colle Barolo) сканер отдал с p = 0,68, счёт CV 0,69 — ниже
    порога. Экран `check` с причиной `visual_low`, а slug, карточка и top-5 прежние: predict
    от подсказки не меняется, `not_found` она не открывает."""
    body = load("scan_suggest")
    assert day1_state(body) == ("found", [])
    after = body["after"]
    assert after["suggest_not_found"] is True
    assert (after["state"], after["reasons"]) == ("check", ["visual_low"])
    assert body["slug"] and body["card"]["slug"] == body["slug"]
    assert body["outcome"] == "matched" and body["top5"][0]["slug"] == body["slug"]
    for name in ("scan_found", "scan_check", "scan_not_found", "scan_error"):
        assert load(name)["after"]["suggest_not_found"] is False, name


CARDS = [load("wine_card"), *(load(name)["card"] for name in SCANS if load(name)["card"])]


@pytest.mark.parametrize("card", CARDS, ids=lambda card: card["slug"][:40])
def test_card_fields(card: dict[str, Any]) -> None:
    """Карточка без портала: ровно поля договора, ссылка у каждой позиции, правила сахара."""
    assert set(card) == CARD_KEYS and not PORTAL_KEYS & set(card)
    assert card["photo_url"] in (f"/v1/wines/{card['slug']}/photo", None)
    assert card["portal_url"] == PORTAL + card["slug"]
    assert card["color_label"] == card["category"] and card["color_label"] in {
        c.value for c in Color
    }
    assert card["sugar"] == card["sugar_label"] == SUGAR_WORDS.get(card["sugar_class"], "")
    assert card["alcohol_src"] in ("catalog", None)
    assert card["description_src"] in ("catalog", None)
    assert (card["alcohol"] is None) == (card["alcohol_src"] is None)


def tiles(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [*body.get("items", []), *body.get("same_winery", []), *body.get("similar", [])]


@pytest.mark.parametrize("name", RECO)
def test_recommendation_blocks(name: str) -> None:
    body = load(name)
    assert "%" not in json.dumps(body, ensure_ascii=False)
    assert body["order"] in ("reco", "plain")
    assert body["notice"] == (NOTICE if body["order"] == "reco" else None)
    for note in body["notes"]:
        assert set(note) == {"code", "text"} and note["text"]
    for tile in tiles(body):
        assert TILE_KEYS <= set(tile)
        assert tile["photo_url"] == f"/v1/wines/{tile['slug']}/photo"
        assert tile["portal_url"] == PORTAL + tile["slug"]
        if body["order"] == "plain":
            assert tile["reasons"] == []
        else:
            assert tile["reasons"]


def test_similar_excludes_anchor_and_plain_is_by_name() -> None:
    reco, plain = load("similar"), load("similar_plain")
    for body in (reco, plain):
        assert body["slug"] not in {t["slug"] for t in body["items"]}
        assert len(body["items"]) == body["limit"] == 3
    names = [t["name"].lower() for t in plain["items"]]
    assert names == sorted(names)
    assert {t["style_label"] for t in plain["items"]} == {plain["category_label"]}


def test_by_label_splits_wineries() -> None:
    body = load("by_label")
    winery = body["read"]["winery"]
    assert body["winery"]["name"] == winery and body["winery"]["in_catalog"]
    assert body["same_winery"] and all(t["winery"] == winery for t in body["same_winery"])
    assert body["similar"] and all(t["winery"] != winery for t in body["similar"])
    assert len({t["winery"] for t in body["similar"]}) == len(body["similar"])


def test_shelf_answers_three_or_honest_phrase() -> None:
    full, empty = load("shelf"), load("shelf_empty")
    assert len(full["items"]) == 3 and full["notes"] == []
    for tile in full["items"]:
        assert tile["want_source"] in ("catalog", "grape")
        assert tile["dishes"]  # вино попало по блюду — значит, пары у него есть
        assert tile["reasons"][0].endswith("по правилам сочетаний")
    assert empty["items"] == [] and empty["notes"]
    for body in (full, empty):
        assert body["food"] in ("meat", "fish", "cheese", "none")
        assert body["want"] in ("fresher", "softer", "sweeter", "none")
        assert body["pool"] in ("near", "catalog")
