"""`GET /v1/wines/{slug}/shelf` — «Сомелье у полки» через HTTP по договору (§7).

Справочник — фейки `reco_env` (Бета Холмы, Дельта Сад, Эта, Зета, Йота…), блюда чипа «К
чему?» — фейковые правила сочетаний `reco_env.FOODS` (пары `yes` к блюду группы чипа). Правила
подбора проверяет `test_recommend_shelf.py`; здесь — ключи ответа, 404 и 422, плашка и
обычная сортировка, честная фраза без данных пар, право на всех сочетаниях чипов.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import FakeOllama, make_service, settings_for
from fastapi.testclient import TestClient
from reco_env import make_reco_service
from test_after_fixtures import ABS_PATH, NOTICE, PORTAL, STOP, TILE_KEYS, strings

from app.api.main import create_app
from app.recommend.content_filter import check
from app.recommend.shelf import FOODS, WANTS, Shelf

SHELF_KEYS = {"slug", "food", "want", "order", "notice", "pool", "items", "notes"}
SHELF_TILE_KEYS = TILE_KEYS | {"dishes", "want_source"}


def shelf(client: TestClient, slug: str, **params: Any) -> dict[str, Any]:
    answer = client.get(f"/v1/wines/{slug}/shelf", params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def assert_contract(body: dict[str, Any]) -> None:
    assert set(body) == SHELF_KEYS
    assert body["pool"] in ("near", "catalog")
    assert body["notice"] == (NOTICE if body["order"] == "reco" else None)
    assert "%" not in json.dumps(body, ensure_ascii=False)
    for text in strings(body):
        assert not STOP.search(text) and not ABS_PATH.search(text), text
        assert check(text).clean, text
    for note in body["notes"]:
        assert set(note) == {"code", "text"} and note["text"]
    if len(body["items"]) < 3:
        assert {n["code"] for n in body["notes"]} & {
            "fewer_than_three",
            "no_difference",
            "not_evaluable",
            "no_pairs",
        }
    for tile in body["items"]:
        assert set(tile) == SHELF_TILE_KEYS
        assert tile["photo_url"] == f"/v1/wines/{tile['slug']}/photo"
        assert tile["portal_url"] == PORTAL + tile["slug"]
        assert isinstance(tile["dishes"], list)
        if body["food"] != "none":
            assert tile["dishes"]  # вино попало по блюду — значит, пары у него есть
        if body["order"] == "plain" or body["want"] == "none":
            assert tile["want_source"] is None
        else:
            assert tile["want_source"] in ("catalog", "grape")
        assert (tile["reasons"] == []) == (body["order"] == "plain")


@pytest.fixture
def client(tmp_path):
    service = make_reco_service(tmp_path)
    with TestClient(create_app(service=service, warm=False)) as test_client:
        yield test_client


def test_shelf_meat_sweeter(client):
    body = shelf(client, "delta-merlot-suhoe", food="meat", want="sweeter")
    assert_contract(body)
    assert (body["food"], body["want"], body["order"]) == ("meat", "sweeter", "reco")
    # к мясу слаще сухого — полусухое Зеты и сладкое Йоты; у Бета Мерло сахара нет — не «слаще»
    assert [t["slug"] for t in body["items"]] == ["zeta-merlot-polusuhoe", "iota-merlot-sladkoe"]
    assert body["pool"] == "near"
    assert body["notes"] == [{"code": "fewer_than_three", "text": "Нашлось меньше трёх"}]
    zeta = body["items"][0]
    assert zeta["reasons"][:2] == [
        "К мясу — по правилам сочетаний",
        "Послаще: полусухое, а не сухое",
    ]
    assert zeta["dishes"] == ["Шашлык", "Борщ"] and zeta["want_source"] == "catalog"


def test_shelf_defaults_are_none_chips(client):
    body = shelf(client, "beta-merlot")
    assert_contract(body)
    assert (body["food"], body["want"], body["order"]) == ("none", "none", "reco")
    assert body["items"] and all(t["want_source"] is None for t in body["items"])


def test_shelf_sweeter_for_wine_without_sugar_is_honest(client):
    """Сахар якоря неизвестен: «послаще» считать не от чего — так и сказано, без «разницы нет».

    «По описаниям разницы нет» было бы неправдой: слаще вина в каталоге есть, сравнить не с чем.
    """
    for food in ("none", "meat"):
        body = shelf(client, "beta-merlot", food=food, want="sweeter")
        assert_contract(body)
        assert body["items"] == []
        assert body["notes"] == [
            {
                "code": "not_evaluable",
                "text": "Сахар этого вина в каталоге не указан — послаще подобрать не по чему",
            }
        ]


def test_shelf_plain_is_category_by_name(client):
    body = shelf(client, "delta-merlot-suhoe", food="meat", want="sweeter", order="plain")
    assert_contract(body)
    names = [t["name"] for t in body["items"]]
    assert names == sorted(names, key=str.casefold)
    # направление при обычной сортировке не применяется: красное сухое с мясом
    assert {t["style_label"] for t in body["items"]} == {"Красное сухое"}
    assert [t["slug"] for t in body["items"]] == ["eta-merlot-suhoe"]


def test_shelf_no_difference_for_sweet_anchor(client):
    body = shelf(client, "iota-merlot-sladkoe", want="sweeter")
    assert_contract(body)
    assert body["items"] == []
    assert body["notes"] == [
        {"code": "no_difference",
         "text": "Это вино уже сладкое: послаще в каталоге по описаниям не найти"},
    ]  # fmt: skip


def test_shelf_without_pairing_data_says_so(tmp_path):
    """Данных правил сочетаний нет: при выбранном блюде — пусто и нота `no_pairs`."""
    service = make_reco_service(tmp_path)
    service.after.sommelier = Shelf(service.after.catalog)
    with TestClient(create_app(service=service, warm=False)) as client:
        body = shelf(client, "delta-merlot-suhoe", food="fish", want="fresher")
        free = shelf(client, "delta-merlot-suhoe", want="fresher")
    assert_contract(body)
    assert body["items"] == [] and body["notes"] == [
        {"code": "no_pairs", "text": "Подбор к блюдам недоступен: нет данных правил сочетаний"}
    ]
    assert_contract(free)
    assert all(tile["dishes"] == [] for tile in free["items"])


@pytest.mark.parametrize(
    "params", [{"food": "pizza"}, {"want": "better"}, {"order": "best"}, {"food": ""}]
)
def test_shelf_rejects_values_outside_chips(client, params):
    assert client.get("/v1/wines/beta-merlot/shelf", params=params).status_code == 422


def test_shelf_unknown_slug_is_404(client):
    answer = client.get("/v1/wines/no-such-wine/shelf")
    assert answer.status_code == 404 and answer.json()["detail"] == "нет карточки 'no-such-wine'"
    assert client.get("/v1/wines/delta-merlot/shelf").status_code == 404  # вне выгрузки


def test_shelf_without_recommendation_data_is_honest():
    service = make_service(FakeOllama(""), settings=settings_for())
    with TestClient(create_app(service=service, warm=False)) as client:
        body = shelf(client, "beta-merlot", food="fish", want="fresher")
    assert body["items"] == [] and body["notes"][0]["code"] == "fewer_than_three"


def test_shelf_route_registered_once(tmp_path):
    app = create_app(service=make_reco_service(tmp_path), warm=False)
    paths = [getattr(route, "path", "") for route in app.routes]
    assert paths.count("/v1/wines/{slug}/shelf") == 1


def test_every_chip_combination_follows_contract(tmp_path):
    service = make_reco_service(tmp_path)
    for wine in service.after.catalog.pool:
        for food in FOODS:
            for want in (*WANTS, "none"):
                for order in ("reco", "plain"):
                    body = service.after.shelf(wine.slug, food=food, want=want, order=order)
                    assert_contract(body)
