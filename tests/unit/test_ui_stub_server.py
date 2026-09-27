"""Заглушка API для вёрстки (`scripts/ui_stub_server.py`) отвечает по договорам.

Страница и лист сомелье верстаются на ней до готовности бэкенда, поэтому заглушка обязана
говорить на языке `docs/api-after-search.md` и `docs/api-sommelier.md`: те же маршруты, те же
404 и 422, `plain` без плашки и объяснений, карточка без полей портала, поток NDJSON сомелье.
Сервер поднимается на свободном порту без каталога данных: сеть наружу, модели и видеокарта
не нужны.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.recommend.content_filter import check

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ui_stub_server.py"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "after"
SOMM = Path(__file__).resolve().parents[1] / "fixtures" / "somm"
MASSANDRA = "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"
LERMONT = "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135"
PORTAL = "https://vino-svoe.ru/wines/"
#: 66 slug выгрузки вне карты сайта вин организатора: ссылки на портал у них нет (как у сервиса).
LINKS = Path(__file__).resolve().parents[2] / "app" / "recommend" / "portal_links.json"
UNLISTED = frozenset(json.loads(LINKS.read_text(encoding="utf-8"))["unlisted"])
PORTAL_KEYS = ("dishes", "temperature", "category_gradient", "live_category", "published")
NOTICE = "Применяются рекомендательные технологии"
ALGO = "Текст и подбор — алгоритм"
AXES = (
    "sweetness",
    "acidity",
    "tannin",
    "body",
    "alcohol",
    "oak",
    "aroma_intensity",
    "effervescence",
)
JSON_HEADERS = {"Content-Type": "application/json"}
# Без прокси: переменные окружения машины не должны уводить запросы к 127.0.0.1 наружу.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture(scope="module")
def stub() -> Any:
    spec = importlib.util.spec_from_file_location("ui_stub_server", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base(stub: Any) -> Iterator[str]:
    server = stub.make_server(port=0, data_dir=None)
    stub.serve_in_thread(server)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(
    url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None
) -> tuple[int, str, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with OPENER.open(request, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def get_json(url: str, **kwargs: Any) -> tuple[int, Any]:
    status, _, body = call(url, **kwargs)
    return status, json.loads(body.decode("utf-8"))


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def somm(name: str) -> Any:
    return json.loads((SOMM / name).read_text(encoding="utf-8"))


def portal(slug: str) -> str | None:
    """Ссылка на карточку портала — только у slug из карты сайта вин (договор «после поиска», §1)."""
    return None if slug in UNLISTED else PORTAL + slug


def noportal(card: dict[str, Any]) -> dict[str, Any]:
    """Карточка заглушки по договору «после поиска», §3: полей портала нет, ссылка — по карте."""
    out = {k: v for k, v in copy.deepcopy(card).items() if k not in PORTAL_KEYS}
    out["portal_url"] = portal(out["slug"])
    if out.get("alcohol_src") not in (None, "catalog"):
        out.update(alcohol=None, alcohol_max=None, alcohol_src=None)
    out["description_src"] = "catalog" if out.get("description") else None
    return out


def tiles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**item, "portal_url": portal(item["slug"])} for item in items]


def expected_scan(name: str) -> dict[str, Any]:
    """Заглушка скана так, как её отдаёт сервер: карточка без портала и вопрос экрана check."""
    body = copy.deepcopy(fixture(name))
    if body.get("card"):
        body["card"] = noportal(body["card"])
    question = somm("scan_check_question.json")["cases"][0]["question"]
    body["after"]["question"] = question if body["after"]["state"] == "check" else None
    return body


def test_page_is_served_and_driver_only_on_shot(base: str) -> None:
    status, ctype, body = call(base + "/")
    assert status == 200 and ctype.startswith("text/html")
    assert b"ui_stub_driver" not in body and b"/__stub/driver.js" not in body
    _, _, shot = call(base + "/?shot=found")
    assert b'<script src="/__stub/driver.js"></script>' in shot
    status, ctype, _ = call(base + "/__stub/driver.js")
    assert status == 200 and ctype.startswith("text/javascript")


@pytest.mark.parametrize(
    ("state", "name"),
    [
        ("found", "scan_found"),
        ("check", "scan_check"),
        ("not_found", "scan_not_found"),
        ("error", "scan_error"),
    ],
)
def test_scan_state_from_query_or_referer(base: str, state: str, name: str) -> None:
    want = expected_scan(name)
    assert get_json(f"{base}/v1/scan?state={state}", data=b"x") == (200, want)
    referer = {"Referer": f"{base}/?state={state}"}
    assert get_json(base + "/v1/scan", data=b"x", headers=referer) == (200, want)


def test_scan_check_asks_the_contract_question(base: str) -> None:
    """Экран check у заглушки — кадр L17: вопрос о цвете (договор сомелье, §5, случай 1)."""
    _, body = get_json(f"{base}/v1/scan?state=check", data=b"x")
    question = body["after"]["question"]
    assert question["field"] == "color"
    assert question["text"] == "Какого цвета вино: красное или белое?"
    assert [o["label"] for o in question["options"]] == ["красное", "белое"]
    assert check(question["text"]).clean


def test_scan_defaults_to_found(base: str) -> None:
    status, body = get_json(base + "/v1/scan", data=b"x")
    assert status == 200 and body["after"]["state"] == "found"
    assert body["after"]["question"] is None


@pytest.mark.parametrize(("state", "name"), [("found", "scan_found"), ("check", "scan_check")])
def test_scan_suggest_flag(base: str, state: str, name: str) -> None:
    """`suggest=1` — подсказка «нет в каталоге»: found → check, причина visual_low; slug тот же."""
    plain = expected_scan(name)
    for kwargs in (
        {"url": f"{base}/v1/scan?state={state}&suggest=1"},
        {"url": base + "/v1/scan", "headers": {"Referer": f"{base}/?state={state}&suggest=1"}},
        {"url": base + "/v1/scan", "headers": {"Referer": f"{base}/?shot=suggest&state={state}"}},
    ):
        status, body = get_json(kwargs.pop("url"), data=b"x", **kwargs)
        after = body["after"]
        assert status == 200 and after["suggest_not_found"] is True
        assert after["state"] == "check"
        assert after["reasons"] == [*plain["after"]["reasons"], "visual_low"]
        assert (body["slug"], body["top5"], body["card"]) == (
            plain["slug"],
            plain["top5"],
            plain["card"],
        )


def test_scan_suggest_leaves_error_alone(base: str) -> None:
    assert get_json(f"{base}/v1/scan?state=error&suggest=1", data=b"x") == (
        200,
        expected_scan("scan_error"),
    )


def test_card_without_portal(base: str) -> None:
    """Карточка — договор «после поиска», §3: состав заглушки `wine_card_noportal.json`."""
    keys = set(somm("wine_card_noportal.json"))
    status, card = get_json(f"{base}/v1/wines/{LERMONT}")
    assert status == 200 and card == somm("wine_card_noportal.json")
    status, card = get_json(f"{base}/v1/wines/{MASSANDRA}")
    assert status == 200 and card == noportal(fixture("wine_card"))
    assert set(card) == keys and "photo_path" not in card
    status, body = get_json(base + "/v1/wines/no-such-wine")
    assert status == 404 and body["detail"] == "нет карточки 'no-such-wine'"


def test_no_portal_fields_anywhere(base: str) -> None:
    """Ни скан, ни карточки вариантов, ни плитки не несут полей портала; ссылка — по карте."""
    for state in ("found", "check", "not_found"):
        _, body = get_json(f"{base}/v1/scan?state={state}", data=b"x")
        assert not set(body["card"]) & set(PORTAL_KEYS)
        for cand in body["candidates"]:
            _, card = get_json(f"{base}/v1/wines/{cand['slug']}")
            assert set(card) == set(somm("wine_card_noportal.json")), cand["slug"]
            assert card["portal_url"] == portal(card["slug"])
    _, similar = get_json(f"{base}/v1/wines/{MASSANDRA}/similar")
    _, by_label = get_json(
        base + "/v1/similar/by-label",
        data=json.dumps(fixture("by_label")["read"]).encode("utf-8"),
        headers=JSON_HEADERS,
    )
    for item in similar["items"] + by_label["same_winery"] + by_label["similar"]:
        assert item["portal_url"] == portal(item["slug"])
    assert call(base + "/v1/icons/dishes/syr.webp")[0] == 404  # иконок портала нет


def test_similar_reco_and_plain(base: str) -> None:
    slug = fixture("similar")["slug"]
    status, reco = get_json(f"{base}/v1/wines/{slug}/similar?limit=3&order=reco")
    want = fixture("similar")
    want["items"] = tiles(want["items"])
    assert status == 200 and reco == want
    status, plain = get_json(f"{base}/v1/wines/{slug}/similar?limit=2&order=plain")
    assert status == 200 and plain["notice"] is None and len(plain["items"]) == 2
    assert all(item["reasons"] == [] for item in plain["items"])


@pytest.mark.parametrize("query", ["order=best", "limit=0", "limit=13", "limit=x"])
def test_similar_rejects_bad_query(base: str, query: str) -> None:
    status, _ = get_json(f"{base}/v1/wines/{MASSANDRA}/similar?{query}")
    assert status == 422


def test_by_label_echo_plain_and_unknown_winery(base: str) -> None:
    read = fixture("by_label")["read"]
    payload = json.dumps(read).encode("utf-8")
    status, body = get_json(base + "/v1/similar/by-label", data=payload, headers=JSON_HEADERS)
    want = fixture("by_label")
    want["same_winery"], want["similar"] = tiles(want["same_winery"]), tiles(want["similar"])
    assert status == 200 and body == want
    _, plain = get_json(
        base + "/v1/similar/by-label?order=plain", data=payload, headers=JSON_HEADERS
    )
    assert plain["notice"] is None
    assert all(t["reasons"] == [] for t in plain["same_winery"] + plain["similar"])
    names = [t["name"].lower() for t in plain["same_winery"]]
    assert names == sorted(names)
    _, nobody = get_json(
        base + "/v1/similar/by-label",
        data=json.dumps({**read, "winery": None}).encode(),
        headers=JSON_HEADERS,
    )
    assert nobody["same_winery"] == [] and nobody["winery"]["in_catalog"] is False
    assert [n["code"] for n in nobody["notes"]] == ["winery_unknown"]
    status, _ = get_json(base + "/v1/similar/by-label", data=b"[1]", headers=JSON_HEADERS)
    assert status == 422


def test_by_label_exclude(base: str) -> None:
    """`exclude` убирает вина из обоих списков, в эхо `read` не попадает; больше 20 — 422."""
    body = fixture("by_label")
    read = body["read"]
    gone = [body["same_winery"][0]["slug"], body["similar"][0]["slug"], "no-such-wine"]
    status, got = get_json(
        base + "/v1/similar/by-label",
        data=json.dumps({**read, "exclude": gone}).encode("utf-8"),
        headers=JSON_HEADERS,
    )
    assert status == 200 and got["read"] == read
    assert got["same_winery"] == tiles(body["same_winery"][1:])
    assert got["similar"] == tiles(body["similar"][1:])
    for bad in (["x"] * 21, "slug", [1]):
        status, _ = get_json(
            base + "/v1/similar/by-label",
            data=json.dumps({**read, "exclude": bad}).encode("utf-8"),
            headers=JSON_HEADERS,
        )
        assert status == 422, bad


@pytest.mark.parametrize(
    "name",
    [
        "sommelier_red.json",
        "sommelier_sweet.json",
        "sommelier_sparkling.json",
        "sommelier_nosugar.json",
    ],
)
def test_sommelier_fixtures(base: str, name: str) -> None:
    """`GET …/sommelier` у вин заглушек — сама заглушка; при `plain` — без плашки 149-ФЗ."""
    want = somm(name)
    status, body = get_json(f"{base}/v1/wines/{want['slug']}/sommelier?order=reco")
    assert status == 200 and body == want
    status, plain = get_json(f"{base}/v1/wines/{want['slug']}/sommelier?order=plain")
    assert status == 200 and plain["order"] == "plain" and plain["notice_149"] is None
    names = [dish["name"].lower() for dish in plain["dishes"]]
    assert names == sorted(names)


def test_sommelier_from_card_facts(base: str) -> None:
    """Вино без заглушки: ответ по правилам договора из фактов карточки, без осей «по сорту»."""
    slug = "abrau-dyurso-russkoe-igristoe-polusladkoe-shardone-beloe-12"
    status, body = get_json(f"{base}/v1/wines/{slug}/sommelier")
    assert status == 200 and body["slug"] == slug and body["notice_149"] == NOTICE
    assert [axis["axis"] for axis in body["profile"]] == list(AXES)
    for axis in body["profile"]:
        assert (axis["value"] is None) == (axis["source"] is None)
        assert axis["source"] in (None, "catalog")
    assert body["serve"] == {
        "temperature_c": [6, 8],
        "source": "rule",
        "rule": "sparkling",
        "by_grape": False,
    }
    note = body["note"]
    assert note["label"] == ALGO and note["generated"] is False
    assert "Подают при 6–8 °C." in note["text"] and check(note["text"]).clean
    assert [chip["id"] for chip in body["chips"]][-1] == "guided"
    assert get_json(f"{base}/v1/wines/no-such-wine/sommelier")[0] == 404
    assert get_json(f"{base}/v1/wines/{slug}/sommelier?order=top")[0] == 422


def ask(base: str, payload: Any, query: str = "") -> tuple[int, str, list[dict[str, Any]]]:
    status, ctype, raw = call(
        f"{base}/v1/sommelier/ask{query}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=JSON_HEADERS,
    )
    text = raw.decode("utf-8")
    events = [json.loads(line) for line in text.split("\n") if line] if status == 200 else []
    return status, ctype, events


def grammar(events: list[dict[str, Any]]) -> str:
    """Поток по договору (§3.2): `stage* ( facts stage* text | error ) done`."""
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done" and kinds.count("done") == 1
    body = [k for k in kinds[:-1] if k != "stage"]
    assert body in (["facts", "text"], ["error"]), kinds
    return body[0]


@pytest.mark.parametrize(
    ("payload", "query", "intent", "reason"),
    [
        ({"question": "где купить подешевле?"}, "", "refuse", "not_voiced"),
        ({"question": "а к борщу подойдёт?"}, "", "dish_check", None),
        ({"chip": "dish_check", "args": {"dish": "borsch"}}, "", "dish_check", "guard"),
        ({"chip": "what_to_eat"}, "", "what_to_eat", "busy"),
        ({"chip": "guided"}, "", "guided", "not_voiced"),
        ({"chip": "softer"}, "?order=plain", "softer", "plain"),
    ],
)
def test_ask_streams_contract_ndjson(
    base: str, payload: dict[str, Any], query: str, intent: str, reason: str | None
) -> None:
    status, ctype, events = ask(base, {"slug": LERMONT, **payload}, query)
    assert status == 200 and ctype.startswith("application/x-ndjson")
    assert grammar(events) == "facts"
    facts = next(e for e in events if e["type"] == "facts")
    text = next(e for e in events if e["type"] == "text")
    assert facts["slug"] == LERMONT and facts["intent"] == intent
    assert text["reason"] == reason
    assert (text["label"] == ALGO) == (text["generated"] is False)
    assert check(text["text"]).clean


def test_ask_error_and_bad_requests(base: str) -> None:
    status, _, events = ask(base, {"slug": LERMONT, "chip": "what_to_eat"}, "?somm=error")
    assert status == 200 and grammar(events) == "error"
    for bad in (
        {"slug": LERMONT},
        {"slug": LERMONT, "chip": "serve", "question": "как подать?"},
        {"slug": LERMONT, "chip": "buy"},
        {"slug": LERMONT, "question": "   "},
        {"slug": LERMONT, "question": "а" * 201},
        {"question": "как подать?"},
    ):
        assert ask(base, bad)[0] == 422, bad
    assert ask(base, {"slug": "no-such-wine", "chip": "serve"})[0] == 404
    assert ask(base, {"slug": LERMONT, "chip": "serve"}, "?order=top")[0] == 422


def test_files_without_data_and_path_escape(base: str) -> None:
    # Без каталога данных фото нет: страница должна показать свой силуэт бутылки.
    assert call(f"{base}/v1/wines/{MASSANDRA}/photo")[0] == 404
    assert call(base + "/static/..%2F..%2F..%2Fpyproject.toml")[0] == 404
    # Как у сервиса: под /static шрифт, картинки и лист сомелье, страницы — нет.
    assert call(base + "/static/index.html")[0] == 404
    assert call(base + "/static/field.html")[0] == 404
    status, ctype, font = call(base + "/static/fonts/PlayfairDisplay-Medium-cyrillic.woff2")
    assert status == 200 and ctype == "font/woff2" and font[:4] == b"wOF2"
    assert call(base + "/static/fonts/PlayfairDisplay-Bold.woff2")[0] == 404  # начертания нет
    assert call(base + "/static/fonts/OFL.txt")[0] == 404  # лицензия рядом, но не отдаётся
    assert call(base + "/nowhere")[0] == 404


def test_stub_static_matches_service(stub: Any) -> None:
    """Заглушка `app` не импортирует, поэтому список расширений повторён — и сверяется здесь."""
    pytest.importorskip("fastapi")
    from app.api.page import ASSET_SUFFIXES

    assert stub.ASSET_SUFFIXES == ASSET_SUFFIXES


def test_stub_photos_and_links_match_service(stub: Any) -> None:
    """Чужие фото (23 строки `wrong_photo`) и slug вне карты сайта — те же, что у сервиса.

    Заглушка `app` не импортирует: список фото повторён, карта сайта читается тем же файлом.
    Вина вне карты сайта отвечают без ссылки на портал, остальные — со ссылкой.
    """
    from app.recommend.catalog import WRONG_PHOTOS, unlisted_slugs

    assert stub.WRONG_PHOTO == WRONG_PHOTOS and len(WRONG_PHOTOS) == 23
    assert stub.UNLISTED == unlisted_slugs() and len(stub.UNLISTED) == 66
    hidden = min(stub.UNLISTED)
    assert stub.portal_url(hidden) is None
    assert stub.portal_url(MASSANDRA) == PORTAL + MASSANDRA
    assert stub.load_unlisted(Path("__нет__.json")) == frozenset()


def test_stub_rules_follow_card_contract(stub: Any) -> None:
    """Правила сахара, игристости и фото заглушки — те же, что в договоре «после поиска», §3–4."""
    assert stub.sugar_of("Alveus Ultra Cuvee Brut", "x-shardone-beloe-bryut-12") == "brut"
    assert stub.sugar_of("Мускатель белый", MASSANDRA) == "sladkoe"
    assert stub.sugar_of("Аристов", "kuban-vino-aristov-shardone-beloe-ekstra-bryut-12") == (
        "extra_brut"
    )
    assert stub.sugar_of("Brut Zero Dosage", "x") == "brut_nature"
    assert stub.sugar_of("Cru Lermont Рислинг", "cru-lermont-risling") is None
    assert stub.sparkling_of("Русское Игристое", "x", None) is True
    assert stub.sparkling_of("Мускатель белый", MASSANDRA, "sladkoe") is False
    fresh = stub.Stub()
    for slug in stub.WRONG_PHOTO:
        assert fresh.photo_url(slug, f"/v1/wines/{slug}/photo") is None
    assert fresh.photo_url(MASSANDRA, None) == f"/v1/wines/{MASSANDRA}/photo"
