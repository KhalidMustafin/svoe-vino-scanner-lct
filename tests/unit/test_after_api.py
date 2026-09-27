"""Слой «после поиска» через HTTP: ответы соответствуют договору `docs/api-after-search.md`.

Проверки договора — те же, что у заглушек (`test_after_fixtures.py`): ключи, отсутствие
стоп-слов и путей машины, `%` только вне блоков рекомендаций, плашка и обычная сортировка.
Плюс то, чего по заглушкам не проверить: правило подбора на живом справочнике, фото выгрузки с
диска, 404 и 422. Снимка портала нет (решение 24.09): маршрута иконок блюд тоже.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import BLUE, RED, FakeClock, FakeOllama, image_bytes, make_service, settings_for
from fastapi.testclient import TestClient
from predict_cases import GOLDEN, cases
from reco_env import FOREIGN_PHOTO, make_reco_service, write_reco
from test_after_fixtures import (
    ABS_PATH,
    AFTER_KEYS,
    CANDIDATE_KEYS,
    CARD_KEYS,
    NOTICE,
    PORTAL,
    PORTAL_KEYS,
    PREDICT_KEYS,
    READ_KEYS,
    STOP,
    TILE_KEYS,
    expected_state,
    strings,
)

from app.api.after_layer import (
    EXCLUDE_MAX,
    SUGGEST_NOT_FOUND_VISUAL_MAX,
    AfterSearch,
    ByLabelBody,
    winery_of_lines,
)
from app.api.main import create_app
from app.api.service import failure
from app.recommend.content_filter import check

SCAN = "/v1/scan"


def client_for(tmp_path, text: str = "Бета Холмы\nМерло", **overrides: Any) -> TestClient:
    service = make_reco_service(tmp_path, text, **overrides)
    return TestClient(create_app(service=service, warm=False))


def post(client: TestClient, color=RED, data: bytes | None = None):
    payload = data if data is not None else image_bytes(color)
    return client.post(SCAN, files={"image": ("q.png", payload, "application/octet-stream")})


def assert_clean(body: Any, *, recommendation: bool) -> None:
    """Правовые проверки договора на любом ответе."""
    for text in strings(body):
        assert not STOP.search(text), text
        assert not ABS_PATH.search(text), text
    if recommendation:
        assert "%" not in json.dumps(body, ensure_ascii=False)


# ------------------------------------------------------------------ /v1/scan
def test_scan_found_carries_card_candidates_and_after(tmp_path):
    with client_for(tmp_path) as client:
        body = post(client).json()
    assert body["slug"] == "beta-merlot" and body["outcome"] == "matched"
    assert set(body) - PREDICT_KEYS == {"evidence", "card", "candidates", "after"}
    assert [c["slug"] for c in body["candidates"]] == [t["slug"] for t in body["top5"]]
    assert body["candidates"][0]["slug"] == body["slug"]
    for cand in body["candidates"]:
        assert set(cand) == CANDIDATE_KEYS
        assert "score" not in cand
        assert cand["photo_url"] == f"/v1/wines/{cand['slug']}/photo"  # фото есть у всех
    assert body["candidates"][0]["name"] == "Мерло Терруар"
    after = body["after"]
    assert set(after) == AFTER_KEYS and set(after["read"]) == READ_KEYS
    assert (after["state"], after["reasons"]) == expected_state(body) == ("found", [])
    assert after["read"]["winery"] == "Бета Холмы" and after["read"]["grapes"] == ["merlot"]
    assert after["read_label"] == "Бета Холмы · Мерло"
    assert after["winery_in_catalog"] is True
    assert after["winery_slugs"] == ["beta-merlot", "beta-shardone"]  # по названию
    assert after["suggest_not_found"] is False  # счёт CV красного кадра 0,98
    assert body["card"]["slug"] == "beta-merlot" and "photo_path" not in body["card"]
    assert body["evidence"]["abstain"]["mode"] == "off"
    assert_clean({k: v for k, v in body.items() if k != "evidence"}, recommendation=False)


def test_scan_check_when_ambiguous(tmp_path):
    # Строка прочитана, слов каталога в ней нет: решает слой выбора, и «Мерло» вплотную.
    with client_for(tmp_path, text="Урожай позапрошлого года") as client:
        body = post(client).json()
    assert body["outcome"] == "ambiguous"
    assert (body["after"]["state"], body["after"]["reasons"]) == ("check", ["ambiguous"])
    assert body["after"]["read"] == dict.fromkeys(READ_KEYS) | {"grapes": []}
    assert body["after"]["read_label"] is None and body["after"]["winery_in_catalog"] is None
    assert len(body["candidates"]) == 5


def test_scan_check_without_confidence_when_reader_gave_nothing(tmp_path):
    """Э2: VLM ничего не прочитал — ответ CV top-1 без вероятности, экран `check`."""
    with client_for(tmp_path, text="") as client:
        body = post(client).json()
    assert body["slug"] == body["evidence"]["cv"]["top5"][0]["slug"]
    assert body["outcome"] == "matched" and body["confidence"]["top1"] is None
    assert body["evidence"]["resolve"]["fallback"] == "reader_empty"
    assert (body["after"]["state"], body["after"]["reasons"]) == ("check", ["no_confidence"])
    assert body["after"]["read"] == dict.fromkeys(READ_KEYS) | {"grapes": []}
    assert [c["slug"] for c in body["candidates"]] == [
        c["slug"] for c in body["evidence"]["cv"]["top5"]
    ]


def test_scan_abstain_is_check_and_keeps_candidates(tmp_path):
    with client_for(tmp_path, text="Гамма Берег\nМерло", abstain="ooc_only") as client:
        body = post(client, BLUE).json()
    assert body["slug"] is None and body["card"] is None
    # у синего кадра все косинусы ниже 0,75 — к отказу добавляется подсказка
    after = body["after"]
    assert (after["state"], after["reasons"]) == ("check", ["abstain", "visual_low"])
    assert after["suggest_not_found"] is True
    assert body["candidates"] and body["after"]["winery_in_catalog"] is True
    assert body["after"]["read"]["winery"] == "Гамма Берег"


@pytest.mark.parametrize(("data", "code"), [(b"junk", "decode"), (b"", "decode")])
def test_scan_error_state(tmp_path, data, code):
    with client_for(tmp_path) as client:
        body = post(client, data=data).json()
    assert body["candidates"] == [] and body["card"] is None
    assert (body["after"]["state"], body["after"]["reasons"]) == ("error", [code])


def test_scan_without_file_is_error_no_image(tmp_path):
    with client_for(tmp_path) as client:
        body = client.post(
            SCAN, files={"file": ("q.png", image_bytes(RED), "application/octet-stream")}
        ).json()
    assert (body["after"]["state"], body["after"]["reasons"]) == ("error", ["no_image"])


def test_scan_survives_broken_after_layer(tmp_path, monkeypatch):
    """Ошибка слоя «после поиска» не прячет ответ сканера: slug и карточка остаются."""
    service = make_reco_service(tmp_path)

    def boom(result):
        raise RuntimeError("слой упал")

    monkeypatch.setattr(service.after, "candidates", boom)
    with TestClient(create_app(service=service, warm=False)) as client:
        body = post(client).json()
    assert body["slug"] == "beta-merlot" and body["error"] is None
    assert body["card"]["slug"] == "beta-merlot" and "photo_path" not in body["card"]
    assert body["candidates"] == [] and body["after"]["state"] == "found"


def test_not_ready_scan_body_has_after():
    body = failure("not_ready: сервис ещё не собран").scan_body(None)
    assert body["after"]["state"] == "error" and body["after"]["reasons"] == ["not_ready"]
    assert body["candidates"] == [] and set(body["after"]) == AFTER_KEYS
    assert body["after"]["suggest_not_found"] is False


# ------------------------------------------------------------------ подсказка suggest_not_found
def test_suggest_threshold_is_calibrated():
    """Порог из калибровки `research/2026-09-24_after/hint.md`: подсказка включена."""
    assert SUGGEST_NOT_FOUND_VISUAL_MAX is not None and 0.5 < SUGGEST_NOT_FOUND_VISUAL_MAX < 1


def test_suggest_turns_found_into_check_and_keeps_slug(tmp_path):
    """Счёт CV ниже порога: `found` → `check` с причиной `visual_low`, но slug, карточка и
    кандидаты прежние, а `not_found` подсказка не открывает."""
    service = make_reco_service(tmp_path)
    with TestClient(create_app(service=service, warm=False)) as client:
        plain = post(client).json()
        service.after.suggest_max = 1.0  # красный кадр — 0,98: теперь ниже порога
        flagged = post(client).json()
    assert (plain["after"]["state"], plain["after"]["suggest_not_found"]) == ("found", False)
    after = flagged["after"]
    assert (after["state"], after["reasons"], after["suggest_not_found"]) == (
        "check",
        ["visual_low"],
        True,
    )
    for key in ("slug", "outcome", "top5", "card", "candidates"):
        assert flagged[key] == plain[key], key
    rest = ("read", "read_label", "winery_in_catalog", "winery_slugs")
    assert {k: after[k] for k in rest} == {k: plain["after"][k] for k in rest}


def test_suggest_off_when_threshold_is_none(tmp_path):
    with client_for(tmp_path, text="Гамма Берег\nМерло", abstain="ooc_only") as client:
        client.app.state.service.after.suggest_max = None
        after = post(client, BLUE).json()["after"]
    assert (after["state"], after["reasons"], after["suggest_not_found"]) == (
        "check",
        ["abstain"],
        False,
    )


@pytest.mark.parametrize("name", ["matched_by_text", "abstain_ooc_only"])
def test_suggest_never_reaches_predict(tmp_path, name):
    """Подсказка на каждом кадре (порог выше любого счёта): тело predict байт в байт прежнее."""
    case = cases()[name]
    write_reco(tmp_path)
    settings = settings_for(tmp_path, abstain=case.get("abstain", "off"))
    service = make_service(FakeOllama(case["text"]), settings=settings, clock=FakeClock())
    service.after = AfterSearch.load(settings, service.cards, service.attrs)
    service.after.suggest_max = 2.0
    data = case.get("raw") or image_bytes(case.get("color", RED))
    files = {"image": ("q.png", data, "application/octet-stream")}
    with TestClient(create_app(service=service, warm=False)) as client:
        predict = client.post("/v1/eval/predict", files=files)
        scan = client.post(SCAN, files=files).json()
    assert predict.content.decode("utf-8") == json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    assert scan["after"]["suggest_not_found"] is True and scan["after"]["state"] == "check"
    assert scan["slug"] == predict.json()["slug"]


def test_suggest_page_phrase_passes_content_filter():
    """Текст для страницы (договор, §2) — фраза, а не код причины, и она чистая."""
    text = "Похоже, этой бутылки может не быть в каталоге"
    assert check(text).clean and not STOP.search(text) and "%" not in text


def test_sparkling_is_read_from_sugar_and_words(tmp_path):
    with client_for(tmp_path, text="Гамма Берег\nИгристое вино\nбрют\n12%") as client:
        body = post(client).json()
    read = body["after"]["read"]
    assert read["sparkling"] is True and read["sugar"] == "brut"
    assert body["after"]["read_label"].startswith("Гамма Берег · игристое · брют")


def test_winery_resolution_prefers_scanner_candidate_then_exact_name(tmp_path):
    after = make_reco_service(tmp_path).after
    both = ["Гамма Берег", "Бета Холмы"]
    assert after.resolve_winery(both).name == "Гамма Берег"  # первое точное имя
    picked = after.resolve_winery(both, prefer=["бета холмы"])  # его ранжировал сканер
    assert (picked.name, picked.in_catalog, picked.keys) == ("Бета Холмы", True, ("бета холмы",))
    unknown = after.resolve_winery(["Barolo"])
    assert (unknown.name, unknown.in_catalog) == ("Barolo", False)
    assert after.resolve_winery([]).in_catalog is None


def test_ambiguous_winery_token_is_not_a_winery(tmp_path, monkeypatch):
    """«долина» указывает на несколько виноделен: «прочитали: долина» человеку ничего не даёт."""
    after = make_reco_service(tmp_path).after
    spread = {"долина": {"a": "Долина Лефкадия", "b": "Солнечная долина"}}
    monkeypatch.setattr(after, "_winery_keys", lambda text: spread.get(text, {}))
    match = after.resolve_winery(["долина"])
    assert (match.name, match.in_catalog, match.keys) == (None, None, ())
    body = after.by_label(ByLabelBody(winery="долина", color="Красное"), limit=3, order="reco")
    assert body["winery"] == {"name": "долина", "in_catalog": None, "count": 0}
    assert body["notes"][0] == {
        "code": "winery_unknown",
        "text": "По прочитанному названию винодельню однозначно не определить",
    }
    assert body["same_winery"] == [] and body["similar"]


def test_generic_token_does_not_confirm_scanner_winery(tmp_path):
    """«долина» в фейках есть только у Альфа Долины, но слово родовое: винодельней оно не
    станет — ни потому, что эту винодельню предлагал сканер, ни потому, что она одна."""
    after = make_reco_service(tmp_path).after
    assert after.specific_key("долина") is None
    for prefer in ([], ["альфа долина"]):
        match = after.resolve_winery(["долина"], prefer=prefer)
        assert (match.name, match.in_catalog, match.keys) == (None, None, ())
    # полное имя рядом с родовым токеном однозначно, чья бы винодельня ни была у сканера
    match = after.resolve_winery(["долина", "Альфа Долина"], prefer=["бета холмы"])
    assert (match.name, match.in_catalog, match.keys) == ("Альфа Долина", True, ("альфа долина",))


def test_winery_name_of_generic_words_is_specific(tmp_path, monkeypatch):
    """«Кубань-Вино» — имя винодельни, хотя оба слова родовые; «Кубань» и «villa» — не имя."""
    after = make_reco_service(tmp_path).after
    spread = {
        "Кубань-Вино": {"кубань вино": "Кубань-Вино"},
        "Кубань": {"кубань вино": "Кубань-Вино"},
        "villa": {"villa di alma": "Villa di Alma"},
    }
    monkeypatch.setattr(after, "_winery_keys", lambda text: spread.get(text, {}))
    assert after.specific_key("Кубань-Вино") == "кубань вино"
    assert after.specific_key("Кубань") is None and after.specific_key("villa") is None
    match = after.resolve_winery(["Кубань", "villa"], prefer=["кубань вино", "villa di alma"])
    assert (match.name, match.in_catalog) == (None, None)
    match = after.resolve_winery(["villa", "Кубань-Вино"], prefer=["villa di alma"])
    assert (match.name, match.in_catalog) == ("Кубань-Вино", True)


def test_single_winery_generic_or_grape_word_is_not_specific(tmp_path, monkeypatch):
    """В словаре каталога «хозяйство» и «кооператив» — у одной винодельни, «блан» — написание
    Усадьбы Маркотх. Слова родовые или сорта: винодельню они не называют, её имя — называет."""
    after = make_reco_service(tmp_path).after
    elbuzd = {"донское винодельческое хозяйство эльбузд": "Донское винодельческое хозяйство"}
    spread = {
        "хозяйство": elbuzd,
        "Донское винодельческое хозяйство": elbuzd,
        "кооператив": {"винный кооператив интуиция": "Винный кооператив Интуиция"},
        "блан": {"усадьба маркотх": "Усадьба Маркотх"},
        "Маркотх": {"усадьба маркотх": "Усадьба Маркотх"},
    }
    monkeypatch.setattr(after, "_winery_keys", lambda text: spread.get(text, {}))
    for text in ("хозяйство", "кооператив", "блан"):
        assert after.specific_key(text) is None, text
    assert after.specific_key("Донское винодельческое хозяйство") == next(iter(elbuzd))
    assert after.specific_key("Маркотх") == "усадьба маркотх"
    match = after.resolve_winery(["блан", "хозяйство"], prefer=["усадьба маркотх"])
    assert (match.name, match.in_catalog) == (None, None)


@pytest.mark.parametrize(
    ("lines", "name"),
    [
        (["CHÂTEAU", "de CHÂTAIGNIER", "КРАСНОЕ", "СУХОЕ"], "CHÂTEAU de CHÂTAIGNIER"),
        (["Muzaradi", "Winery", "АЛАЗАНСКАЯ ДОЛИНА"], "Muzaradi Winery"),
        (["DOMAINE RENAUD", "tradition", "SEMI-SWEET"], "DOMAINE RENAUD"),
        (["ЛАЗИ", "DUGLADZE WINE COMPANY", "АЛАЗАНИ"], "DUGLADZE WINE COMPANY"),
        (["ВИННАЯ", "УСАДЬБА", "ДОМАШНЕЕ"], None),  # одни родовые слова
        (["WINE GUIDE", "RESERVE", "Single Vineyard"], None),  # серия, а не имя
        (["ШАТО", "КРАСНОЕ СУХОЕ"], None),  # под словом хозяйства — цвет и сахар
        (["ВИНОДЕЛЬНЯ AZUR", "2024"], None),  # за «винодельней» у Криницы — кюве
        (["MASSIMO VISCONTI", "Lambrusco"], None),  # без слова хозяйства имя не отличить
        (["ШАТО", "СКИДКИ"], None),  # строка не прошла бы content_filter
    ],
)
def test_winery_of_lines(lines, name):
    assert winery_of_lines(lines) == name


def test_house_name_close_to_catalog_winery_is_not_claimed(tmp_path):
    """«Усадьба Бета Холми» — Бета Холмы с ошибкой чтения: «в каталоге нет» было бы неправдой."""
    after = make_reco_service(tmp_path).after
    assert after.near_catalog_winery("Усадьба Бета Холми")
    assert after.resolve_winery([], house="Усадьба Бета Холми").in_catalog is None
    match = after.resolve_winery([], house="Усадьба Совсем Другая")
    assert (match.name, match.in_catalog, match.keys) == ("Усадьба Совсем Другая", False, ())


def test_scan_reads_winery_outside_catalog_from_lines(tmp_path):
    """Правка (b): винодельня прочитана, но её нет в словаре — `winery_in_catalog: false`."""
    text = "CHÂTEAU\nde CHÂTAIGNIER\nКрасное\nСухое"
    with client_for(tmp_path, text=text) as client:
        after = post(client).json()["after"]
        body = by_label(client, after["read"])
    assert after["read"]["winery"] == "CHÂTEAU de CHÂTAIGNIER"
    assert after["winery_in_catalog"] is False and after["winery_slugs"] == []
    assert after["read_label"] == "CHÂTEAU de CHÂTAIGNIER · красное · сухое"
    assert after["state"] != "not_found"  # автоэкрана нет (калибровка Д2)
    assert body["winery"] == {"name": "CHÂTEAU de CHÂTAIGNIER", "in_catalog": False, "count": 0}
    assert body["notes"][0] == {"code": "winery_unknown", "text": "Этой винодельни в каталоге нет"}


@pytest.mark.parametrize(
    ("payload", "text"),
    [
        ({}, "Винодельню на этикетке не прочитали"),
        ({"color": "Красное"}, "Винодельню по этикетке определить не удалось"),
        ({"sparkling": True}, "Винодельню по этикетке определить не удалось"),
        ({"winery": "Совсем Другая"}, "Этой винодельни в каталоге нет"),
    ],
)
def test_by_label_winery_note_is_truthful(tmp_path, payload, text):
    """«Не прочитали» — только когда с этикетки не пришло ничего: чужую винодельню словарь не
    знает, и `winery: null` при прочитанном цвете значит «не узнали», а не «не прочитали»."""
    with client_for(tmp_path) as client:
        body = by_label(client, payload)
    note = body["notes"][0]
    assert note == {"code": "winery_unknown", "text": text}
    assert check(note["text"]).clean


# ------------------------------------------------------------------ карточка
def test_card_follows_contract(tmp_path):
    """Карточка без портала: поля выгрузки, сахар и крепость по правилам, ссылка у всех."""
    with client_for(tmp_path) as client:
        card = client.get("/v1/wines/delta-merlot-suhoe").json()
        beta = client.get("/v1/wines/beta-merlot").json()
    assert set(card) == CARD_KEYS and not PORTAL_KEYS & set(card)
    assert card["photo_url"] == "/v1/wines/delta-merlot-suhoe/photo"
    assert card["portal_url"] == PORTAL + "delta-merlot-suhoe"
    assert (card["sugar_class"], card["sugar"], card["sugar_label"]) == ("suhoe", "сухое", "сухое")
    assert (card["style_label"], card["color_label"], card["category"]) == (
        "Красное сухое",
        "Красное",
        "Красное",
    )
    assert (card["alcohol"], card["alcohol_max"], card["alcohol_src"]) == (13.5, None, "catalog")
    assert card["sparkling"] is False and card["description_src"] is None
    # сахара нет ни в названии, ни в slug: сахар неизвестен, стиль — один цвет
    assert (beta["sugar_class"], beta["sugar"], beta["style_label"]) == (None, "", "Красное")
    assert beta["portal_url"] == PORTAL + "beta-merlot"
    assert_clean(card, recommendation=False)


def test_card_description_is_the_organizer_text_as_is(tmp_path):
    """«Описание» выгрузки — как написал организатор: абзацы остаются, края без пробелов."""
    service = make_reco_service(tmp_path)
    raw = "  Первый абзац.\n\nВторой абзац.  "
    object.__setattr__(service.cards.by_slug["beta-merlot"], "description", raw)
    card = service.card("beta-merlot")
    assert card["description"] == "Первый абзац.\n\nВторой абзац."
    assert card["description_src"] == "catalog"


def test_live_portal_wine_outside_markup_has_no_card(tmp_path):
    """73 живые карточки портала вне выгрузки в продукт не входят: ни карточки, ни фото."""
    with client_for(tmp_path) as client:
        assert client.get("/v1/wines/delta-merlot").status_code == 404
        assert client.get("/v1/wines/delta-merlot/photo").status_code == 404
        assert client.get("/v1/wines/no-such-wine").status_code == 404


def test_card_never_leaks_machine_paths(tmp_path):
    """photo_map указывает на пути машины: в карточке этого пути быть не должно."""
    service = make_reco_service(tmp_path)
    for slug in [*service.cards.by_slug, *service.after.catalog.by_slug]:
        card = service.card(slug)
        assert card is not None
        for text in strings(card):
            assert not ABS_PATH.search(text), (slug, text)


# ------------------------------------------------------------------ фото выгрузки
def test_photo_route_serves_file_with_cache(tmp_path):
    with client_for(tmp_path) as client:
        answer = client.get("/v1/wines/beta-merlot/photo")
        assert answer.status_code == 200
        assert answer.headers["cache-control"] == "public, max-age=86400"
        assert answer.headers["content-type"] == "image/webp"  # не octet-stream, как на Windows
        assert answer.content == b"RIFF-small-beta-merlot"
        assert client.get("/v1/wines/no-such-wine/photo").status_code == 404
        assert client.get("/v1/wines/..%2F..%2Fwines.jsonl/photo").status_code == 404


def test_photo_is_only_the_organizer_one(tmp_path):
    """Лёгкая копия, затем путь карты выгрузки; путь правки эталонов (`packshot_fix`,
    `packshots_fixed`) не берётся, и без копии у такого вина фото нет."""
    service = make_reco_service(tmp_path)
    after = service.after
    assert after.foreign_photos == {"beta-shardone"}
    light = tmp_path / "photos" / "beta-shardone.webp"
    assert service.photo_file("beta-shardone") == light  # копия выгрузки, а не правка
    light.unlink()
    after._photos.clear()
    assert service.photo_file("beta-shardone") is None  # путь карты — не фото выгрузки
    # путь карты выгрузки берётся, когда лёгкой копии нет
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "beta-merlot.webp").write_bytes(b"dump")
    (tmp_path / "photos" / "beta-merlot.webp").unlink()
    after._photos.clear()
    assert service.photo_file("beta-merlot") == dump / "beta-merlot.webp"
    # каталог правки узнаётся и без колонки method
    card = service.cards.by_slug["gamma-saperavi"]
    fixed = tmp_path / FOREIGN_PHOTO / "gamma-saperavi.png"
    fixed.write_bytes(b"PNG")
    object.__setattr__(card, "photo_path", str(fixed))
    (tmp_path / "photos" / "gamma-saperavi.webp").unlink()
    after._photos.clear()
    assert service.photo_file("gamma-saperavi") is None


def test_photo_showing_another_wine_is_not_served(tmp_path):
    """На фото выгрузки другое вино (правка 23.09, `wrong_photo`): фото нет, `photo_url: null`."""
    service = make_reco_service(tmp_path)
    service.after.wrong_photos = frozenset({"beta-merlot"})
    service.after._photos.clear()
    assert service.photo_file("beta-merlot") is None
    assert service.card("beta-merlot")["photo_url"] is None


def test_dish_icon_route_is_gone(tmp_path):
    """Иконок блюд портала нет: маршрут снят вместе со снимком (договор, §4)."""
    with client_for(tmp_path) as client:
        for name in ("syr_820e71a7c0.webp", "..%2Fdetails.json"):
            assert client.get(f"/v1/icons/dishes/{name}").status_code == 404


# ------------------------------------------------------------------ похожие
def similar(client: TestClient, slug: str, **params: Any) -> dict[str, Any]:
    answer = client.get(f"/v1/wines/{slug}/similar", params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_similar_reco_follows_rules(tmp_path):
    with client_for(tmp_path) as client:
        body = similar(client, "delta-merlot-suhoe")
    assert body["order"] == "reco" and body["notice"] == NOTICE and body["limit"] == 3
    assert body["category_label"] == "Красное сухое"
    slugs = [item["slug"] for item in body["items"]]
    # своя винодельня, вино без фото, магнум и сладкое — нет; у Бета Мерло сахар неизвестен:
    # он подбор не сужает, но тот же сахар Эты ближе
    assert slugs == ["eta-merlot-suhoe", "beta-merlot", "zeta-merlot-polusuhoe"]
    for item in body["items"]:
        assert TILE_KEYS <= set(item)
        assert item["photo_url"] == f"/v1/wines/{item['slug']}/photo"
        assert item["portal_url"] == PORTAL + item["slug"]
        assert item["reasons"] and all(check(reason).clean for reason in item["reasons"])
    assert body["items"][0]["reasons"] == [
        "Тот же сорт — Мерло",
        "Тоже сухое",
        "Тоже Кубань",
        "Крепость та же — 13,5°",
    ]
    # сахара у Бета Мерло нет — и строки о сахаре тоже
    assert body["items"][1]["reasons"] == [
        "Тот же сорт — Мерло",
        "Тоже Кубань",
        "Крепость та же — 13,5°",
    ]
    assert body["items"][2]["reasons"][1] == "Полусухое, а не сухое"
    assert body["notes"] == []
    assert_clean(body, recommendation=True)


def test_similar_for_wine_without_sugar(tmp_path):
    """Сахар неизвестен (376 позиций выгрузки): стиль — цвет, сахар подбор не сужает."""
    with client_for(tmp_path) as client:
        body = similar(client, "beta-merlot", limit=12)
    assert body["category_label"] == "Красное"
    slugs = [item["slug"] for item in body["items"]]
    assert "iota-merlot-sladkoe" in slugs and "theta-merlot-suhoe" not in slugs
    assert not any("сухое" in reason for item in body["items"] for reason in item["reasons"])


def test_similar_plain_is_same_category_by_name(tmp_path):
    with client_for(tmp_path) as client:
        body = similar(client, "delta-merlot-suhoe", order="plain", limit=6)
    assert body["notice"] is None
    names = [item["name"] for item in body["items"]]
    assert names == sorted(names, key=str.casefold)
    assert all(item["reasons"] == [] for item in body["items"])
    assert {item["style_label"] for item in body["items"]} == {"Красное сухое"}
    # та же категория без своей винодельни: полусухое Зеты, сладкое Йоты и Бета Мерло и
    # Саперави без сахара сюда не входят
    assert {item["slug"] for item in body["items"]} == {"eta-merlot-suhoe", "epsilon-blend-suhoe"}
    assert body["notes"] == [{"code": "fewer_than_three", "text": "Нашлось меньше шести"}]
    assert_clean(body, recommendation=True)


@pytest.mark.parametrize("params", [{"order": "best"}, {"limit": 0}, {"limit": 13}])
def test_similar_rejects_bad_params(tmp_path, params):
    with client_for(tmp_path) as client:
        assert client.get("/v1/wines/beta-merlot/similar", params=params).status_code == 422


def test_similar_unknown_slug_is_404(tmp_path):
    with client_for(tmp_path) as client:
        answer = client.get("/v1/wines/no-such-wine/similar")
    assert answer.status_code == 404 and answer.json()["detail"] == "нет карточки 'no-such-wine'"


def test_similar_without_recommendation_data_is_honest():
    """Справочника нет (сервис без data/): карточка есть, похожих нет — честная фраза."""
    service = make_service(FakeOllama(""), settings=settings_for())
    with TestClient(create_app(service=service, warm=False)) as client:
        body = similar(client, "beta-merlot")
    assert body["items"] == [] and body["notes"][0]["code"] == "fewer_than_three"


# ------------------------------------------------------------------ «Не тупик»
def by_label(client: TestClient, payload: dict[str, Any], **params: Any) -> dict[str, Any]:
    answer = client.post("/v1/similar/by-label", json=payload, params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_by_label_splits_winery_and_others(tmp_path):
    payload = {"winery": "Бета Холмы", "color": "Красное", "sugar": "suhoe", "grapes": [],
               "abv": 13.5, "sparkling": None}  # fmt: skip
    with client_for(tmp_path) as client:
        body = by_label(client, payload)
    assert body["winery"] == {"name": "Бета Холмы", "in_catalog": True, "count": 2}
    assert body["read"]["winery"] == "Бета Холмы" and body["notice"] == NOTICE
    same = body["same_winery"]
    assert [t["slug"] for t in same] == ["beta-merlot", "beta-shardone"]  # красное первым
    assert same[1]["reasons"] == ["Белое, а не красное"]  # сахара у Шардоне в выгрузке нет
    others = body["similar"]
    assert others and all(t["winery"] != "Бета Холмы" for t in others)
    assert len({t["winery"] for t in others}) == len(others)
    assert "Тоже Кубань" in others[0]["reasons"]  # регион — от винодельни
    assert_clean(body, recommendation=True)


def test_by_label_relaxes_color_for_orange_sparkling(tmp_path):
    payload = {"winery": "Гамма Берег", "color": "Оранжевое", "sugar": "brut", "grapes": [],
               "abv": 12, "sparkling": True}  # fmt: skip
    with client_for(tmp_path) as client:
        body = by_label(client, payload)
    assert body["same_winery"][0]["slug"] == "gamma-brut"
    assert [t["slug"] for t in body["similar"]] == ["kappa-brut", "lambda-rose-brut"]
    assert body["similar"][0]["reasons"][:2] == ["Тоже игристое брют", "Белое, а не оранжевое"]
    codes = [note["code"] for note in body["notes"]]
    assert codes == ["relaxed_color", "fewer_than_three"]
    assert body["notes"][0]["text"] == (
        "Оранжевых игристых в каталоге нет — показываем игристые брют других цветов"
    )
    assert_clean(body, recommendation=True)


def test_by_label_unknown_winery_and_empty_read(tmp_path):
    with client_for(tmp_path) as client:
        unknown = by_label(client, {"winery": "Совсем Другая", "color": "Красное"})
        empty = by_label(client, {})
    assert unknown["winery"] == {"name": "Совсем Другая", "in_catalog": False, "count": 0}
    assert unknown["same_winery"] == [] and unknown["similar"]
    assert unknown["notes"][0] == {
        "code": "winery_unknown",
        "text": "Этой винодельни в каталоге нет",
    }
    assert empty["winery"]["in_catalog"] is None and empty["similar"] == []
    assert [note["code"] for note in empty["notes"]] == ["winery_unknown", "no_facts"]


def test_by_label_plain(tmp_path):
    payload = {"winery": "Бета Холмы", "color": "Красное", "sugar": "suhoe"}
    with client_for(tmp_path) as client:
        body = by_label(client, payload, order="plain")
    assert body["notice"] is None
    for tile in [*body["same_winery"], *body["similar"]]:
        assert tile["reasons"] == []
    names = [t["name"] for t in body["similar"]]
    assert names == sorted(names, key=str.casefold)


@pytest.mark.parametrize(
    "payload", [{"abv": "крепкое"}, {"grapes": "merlot"}, {"sparkling": "может быть"}]
)
def test_by_label_rejects_bad_body(tmp_path, payload):
    with client_for(tmp_path) as client:
        assert client.post("/v1/similar/by-label", json=payload).status_code == 422


def test_by_label_echo_accepts_scan_read_as_is(tmp_path):
    """Страница шлёт `after.read` из ответа `/v1/scan` без правок."""
    with client_for(tmp_path) as client:
        read = post(client).json()["after"]["read"]
        body = by_label(client, read)
    assert body["read"] == read
    assert body["winery"]["in_catalog"] is True


BETA_RED = {"winery": "Бета Холмы", "color": "Красное", "sugar": "suhoe", "grapes": [],
            "abv": 13.5, "sparkling": None}  # fmt: skip


def slugs_of(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    return [t["slug"] for t in body["same_winery"]], [t["slug"] for t in body["similar"]]


def test_by_label_exclude_drops_slug_and_its_wine_group(tmp_path):
    """`exclude` снимает вино и всю его группу `wine_id` из обоих списков, а места добираются."""
    with client_for(tmp_path) as client:
        base = by_label(client, BETA_RED)
        body = by_label(client, BETA_RED | {"exclude": ["beta-merlot", "eta-merlot-magnum-suhoe"]})
    same, similar = slugs_of(base)
    assert same == ["beta-merlot", "beta-shardone"] and "eta-merlot-suhoe" in similar
    same, similar = slugs_of(body)
    assert same == ["beta-shardone"]
    # магнума Эты в пуле нет, но его группа — это и «Эта Мерло»
    assert "eta-merlot-suhoe" not in similar and "eta-merlot-magnum-suhoe" not in similar
    assert len(similar) == len(slugs_of(base)[1]) == 3
    assert body["winery"] == base["winery"]  # число вин винодельни — по всему справочнику
    assert body["read"] == base["read"]  # эхо — без `exclude`
    assert_clean(body, recommendation=True)


def test_by_label_exclude_in_plain_order(tmp_path):
    payload = BETA_RED | {"exclude": ["beta-shardone", "delta-merlot-suhoe"]}
    with client_for(tmp_path) as client:
        body = by_label(client, payload, order="plain")
    same, similar = slugs_of(body)
    assert same == ["beta-merlot"]
    assert similar and "delta-merlot-suhoe" not in similar


def test_by_label_exclude_candidates_of_a_scan(tmp_path):
    """Кнопка «Моего вина здесь нет»: страница шлёт `after.read` и отвергнутые кандидаты."""
    with client_for(tmp_path) as client:
        scan = post(client).json()
        rejected = [c["slug"] for c in scan["candidates"]]
        body = by_label(client, scan["after"]["read"] | {"exclude": rejected})
    shown = {t["slug"] for t in [*body["same_winery"], *body["similar"]]}
    assert rejected and not shown & set(rejected)


@pytest.mark.parametrize("exclude", [[], ["no-such-wine"], ["no-such-wine", "../wines.jsonl", ""]])
def test_by_label_exclude_empty_or_unknown_is_unchanged(tmp_path, exclude):
    with client_for(tmp_path) as client:
        base = by_label(client, BETA_RED)
        body = by_label(client, BETA_RED | {"exclude": exclude})
    assert body == base


def test_by_label_exclude_limit(tmp_path):
    ok = [f"no-such-wine-{i}" for i in range(EXCLUDE_MAX)]
    with client_for(tmp_path) as client:
        assert by_label(client, BETA_RED | {"exclude": ok}) == by_label(client, BETA_RED)
        answer = client.post("/v1/similar/by-label", json=BETA_RED | {"exclude": [*ok, "x"]})
        assert answer.status_code == 422
        answer = client.post("/v1/similar/by-label", json=BETA_RED | {"exclude": "beta-merlot"})
        assert answer.status_code == 422
        # null и строка длиннее 200 символов — тоже 422 (договор, §6), а не «неизвестный slug»
        for bad in (None, ["x" * 201]):
            answer = client.post("/v1/similar/by-label", json=BETA_RED | {"exclude": bad})
            assert answer.status_code == 422, bad


def test_by_label_exclude_keeps_color_note_truthful(tmp_path):
    """Единственное оранжевое вино исключено: «…в каталоге нет» было бы неправдой — оно есть."""
    payload = {"color": "Оранжевое"}
    with client_for(tmp_path) as client:
        base = by_label(client, payload)
        body = by_label(client, payload | {"exclude": ["mu-orange-suhoe"]})
    assert "mu-orange-suhoe" in slugs_of(base)[1]
    assert "mu-orange-suhoe" not in slugs_of(body)[1]
    note = next(n for n in body["notes"] if n["code"] == "relaxed_color")
    assert note["text"] == (
        "Оранжевых вин такого стиля в каталоге мало — добавили близкие по стилю других цветов"
    )
    assert check(note["text"]).clean


def test_excluded_slugs_expand_groups(tmp_path):
    after = make_reco_service(tmp_path).after
    assert after.excluded_slugs(["eta-merlot-suhoe"]) == {
        "eta-merlot-suhoe",
        "eta-merlot-magnum-suhoe",
    }
    assert after.excluded_slugs(["delta-merlot"]) == frozenset()  # вне выгрузки
    assert after.excluded_slugs(["beta-merlot", "no-such-wine"]) == {"beta-merlot"}
    assert after.excluded_slugs([]) == frozenset()


def test_routes_are_registered_once(tmp_path):
    service = make_reco_service(tmp_path)
    app = create_app(service=service, warm=False)
    paths = [getattr(route, "path", "") for route in app.routes]
    assert paths.count("/v1/wines/{slug}") == 1
    for path in ("/v1/wines/{slug}/photo", "/v1/wines/{slug}/similar", "/v1/similar/by-label"):
        assert path in paths
    assert "/v1/icons/dishes/{name}" not in paths


def test_every_recommendation_string_passes_content_filter(tmp_path):
    service = make_reco_service(tmp_path)
    texts: list[str] = []
    for wine in service.after.catalog.pool:
        for order in ("reco", "plain"):
            body = service.after.similar(wine.slug, limit=5, order=order)
            texts.extend(strings(body))
            assert "%" not in json.dumps(body, ensure_ascii=False)
    assert texts
    bad = [text for text in texts if not check(text).clean or re.search(STOP, text)]
    assert bad == []
