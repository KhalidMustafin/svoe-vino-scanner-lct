"""Страница продукта на `/` и её файлы на `/static` (`app.api.page`).

Без `SVS_FIELD` на `/` — страница продукта, полевого стенда нет вовсе. Под `/static` отдаются
только шрифт, картинки и лист сомелье (`.js`, `.css`): HTML оттуда не отдаётся, иначе страница
стенда открывалась бы в обход флага и ключа.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import RED, FakeOllama, image_bytes, make_service, settings_for
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reco_env import make_reco_service
from test_product_page import ASSET_SUFFIXES as PAGE_TEST_SUFFIXES
from test_product_page import FONT_FACES, PORTAL_FIELDS, SHEET_FILES, examples_of

from app.api.cards import CatalogCards
from app.api.field import FieldSettings
from app.api.main import create_app
from app.api.page import ASSET_SUFFIXES, ASSET_TYPES, PAGE_PATH, AssetFiles

SOMM = Path(__file__).resolve().parents[1] / "fixtures" / "somm"
SCRIPT_TYPE = "text/javascript; charset=utf-8"
STYLE_TYPE = "text/css; charset=utf-8"
FIELD_ROUTES = (
    "/field",
    "/v1/field/stats",
    "/v1/field/job/x",
    "/v1/field/photo/beta-merlot",
)


def build(tmp_path: Path, *, field: bool = False) -> TestClient:
    service = make_service(FakeOllama(""), settings=settings_for(tmp_path))
    settings = FieldSettings(directory=tmp_path / "field", enabled=field)
    return TestClient(create_app(service=service, warm=False, field=settings))


def test_root_serves_product_page_without_field_flag(tmp_path):
    with build(tmp_path) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert page.headers["cache-control"] == "no-cache"
        assert page.text == PAGE_PATH.read_text(encoding="utf-8")
        assert 'id="age"' in page.text, "лист 18+ приходит вместе со страницей"
        # Без контура ключ в адресе ничего не значит: та же страница, без перенаправления.
        same = client.get("/", params={"k": "s3cret"}, follow_redirects=False)
        assert same.status_code == 200 and same.text == page.text


def test_field_routes_absent_without_flag(tmp_path):
    with build(tmp_path) as client:
        for path in FIELD_ROUTES:
            assert client.get(path).status_code == 404, path
        for path in ("/v1/field/scan", "/v1/field/submit", "/v1/field/verdict"):
            assert client.post(path).status_code == 404, path
        paths = {getattr(route, "path", "") for route in client.app.routes}
    assert not any(path.startswith(("/field", "/v1/field")) for path in paths), paths


def test_product_page_stays_on_root_with_field_on(tmp_path):
    with build(tmp_path, field=True) as client:
        page = client.get("/", follow_redirects=False)
        assert page.status_code == 200
        assert page.text == PAGE_PATH.read_text(encoding="utf-8")
        assert client.get("/field").status_code == 200


def test_static_gives_no_html_and_no_escape(tmp_path):
    with build(tmp_path) as client:
        for path in (
            "/static/index.html",
            "/static/field.html",
            "/static/",
            "/static/fonts/PlayfairDisplay-Bold.woff2",  # такого начертания нет — честный 404
            "/static/fonts/OFL.txt",  # лицензия лежит рядом со шрифтом, но не отдаётся
            "/static/..%2F..%2F..%2Fpyproject.toml",
            "/static/../main.py",
        ):
            assert client.get(path).status_code == 404, path


def test_static_serves_page_fonts_as_woff2(tmp_path):
    """Шрифт страницы отдаётся `/static` с типом font/woff2 — не octet-stream.

    Тип задан таблицей `ASSET_TYPES`: `mimetypes` Python 3.12 своего `.woff2` не знает.
    """
    with build(tmp_path) as client:
        for name in FONT_FACES:
            font = client.get(f"/static/fonts/{name}")
            assert font.status_code == 200, name
            assert font.headers["content-type"] == "font/woff2", name
            assert font.content[:4] == b"wOF2"
            again = client.get(
                f"/static/fonts/{name}", headers={"If-None-Match": font.headers["etag"]}
            )
            assert again.status_code == 304  # кэш браузера работает, тип 304 не ломает


def page_fields(script: str, name: str) -> set[str]:
    """Поля, которые скрипт страницы читает у объекта `name`: `card.region` → `region`."""
    return set(re.findall(rf"\b{name}\.([a-z_]+)\b", script))


def somm_fixture(name: str) -> dict:
    return json.loads((SOMM / name).read_text(encoding="utf-8"))


def test_page_reads_only_fields_the_live_api_gives(tmp_path):
    """Страница верстана на заглушках: каждое поле, которое она читает, есть в ответе.

    Проход — тот же, что у человека: скан → карточка и кандидаты → сомелье → похожие → «Моего
    вина здесь нет» с прочитанным. Сервис на фейках (`reco_env`), без моделей и сети.

    Карточка — только поля, которые есть и в живом ответе, и в карточке без портала (договор
    «после поиска», §3). Ответ сомелье и `after.question` — и по заглушкам договора сомелье, и
    по живому сервису: поле, которое страница читает, должно быть в обоих.
    """
    html = PAGE_PATH.read_text(encoding="utf-8")
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.DOTALL))
    service = make_reco_service(tmp_path)
    with TestClient(create_app(service=service, warm=False)) as client:
        scan = client.post(
            "/v1/scan", files={"image": ("q.png", image_bytes(RED), "application/octet-stream")}
        ).json()
        card = client.get(f"/v1/wines/{scan['slug']}").json()
        similar = client.get(f"/v1/wines/{scan['slug']}/similar?limit=3&order=reco").json()
        by_label = client.post(
            "/v1/similar/by-label?limit=3&order=reco", json=scan["after"]["read"]
        ).json()
        live_somm = client.get(f"/v1/wines/{scan['slug']}/sommelier?order=reco")

    assert scan["after"]["state"] == "found" and card == scan["card"]
    assert similar["items"] and by_label["same_winery"]
    # блюд, температуры и градиента портала в карточке нет (решение 24.09)
    assert not {"dishes", "temperature", "category_gradient"} & set(card)
    assert live_somm.status_code == 200 and "question" in scan["after"]
    somm = somm_fixture("sommelier_red.json")
    somms = [somm, live_somm.json()]
    dishes = [s["dishes"][0] for s in somms if s["dishes"]]
    question = somm_fixture("scan_check_question.json")["cases"][0]["question"]
    noportal = set(somm_fixture("wine_card_noportal.json"))
    wants = {
        "card": set(card) & noportal,
        "cand": set(scan["candidates"][0]),  # «Не то вино?»
        "after": set(scan["after"]),  # с полем договора сомелье `question` (§5)
        "read": set(scan["after"]["read"]),
        "item": set(similar["items"][0]) | set(by_label["similar"][0]),  # плитки
        "winery": set(by_label["winery"]),
        # `note` — и нота подборки {code, text}, и заметка сомелье.
        "note": {"code", "text"} | set.intersection(*(set(s["note"]) for s in somms)),
        "somm": set.intersection(*(set(s) for s in somms)),
        "axis": set.intersection(*(set(s["profile"][0]) for s in somms)),
        "serve": set.intersection(*(set(s["serve"]) for s in somms)),
        "dish": set.intersection(*(set(dish) for dish in dishes)),
        "rule": set.intersection(*(set(dish["plus"][0]) for dish in dishes if dish["plus"])),
        "q": set(question),
        "opt": set(question["options"][0]),
        # У `body` разные ответы: скан, похожие, «Не тупик» и 404.
        "body": set(scan) | set(similar) | set(by_label) | {"detail"},
    }
    for name, given in wants.items():
        read = page_fields(script, name)
        assert read, f"страница не читает {name}.* — проверка устарела"
        assert read <= given, f"{name}: страница читает {sorted(read - given)}, API их не даёт"
    # Полей портала страница не читает, даже пока сервис ещё их отдаёт.
    assert not page_fields(script, "card") & set(PORTAL_FIELDS)


REAL = Path(os.environ.get("SVS_DATA_DIR") or "__нет__")


@pytest.mark.skipif(
    not (REAL / "catalog" / "wines.jsonl").is_file()
    or not (REAL / "somm" / "pairs.json").is_file(),
    reason="нужны данные сервиса: SVS_DATA_DIR",
)
def test_examples_show_full_radar_on_real_data():
    """«Посмотреть на примере» на настоящем справочнике: «роза ветров» по 8 осям из 8.

    Карточка — та же, что у сервиса: название, винодельня и цвет, как на плитке главной, и
    фото выгрузки. Профиль — `GET …/sommelier`: не меньше шести известных осей (у всех трёх
    сейчас восемь), три блюда и шаблон заметки.
    """
    from test_somm_api import FakeVoice, settings

    from app.api.after_layer import AfterSearch
    from app.api.service import load_catalog
    from app.api.somm import register_somm_routes
    from app.recommend.catalog import RecoCatalog
    from app.recommend.somm_data import load_somm_data

    html = PAGE_PATH.read_text(encoding="utf-8")
    script = "\n".join(re.findall(r"<script>(.*?)</script>", html, re.DOTALL))
    records, attrs = load_catalog(REAL / "gt" / "gt_tokens.jsonl")
    cards = CatalogCards.build(records, photo_map=REAL / "catalog" / "slug_photo_map.csv")
    catalog = RecoCatalog.load(
        REAL / "gt" / "gt_tokens.jsonl",
        wines_path=REAL / "catalog" / "wines.jsonl",
        groups_path=REAL / "catalog" / "wine_groups.json",
    )
    after = AfterSearch(catalog, cards, attrs)
    app = FastAPI()
    app.state.service = SimpleNamespace(after=after)
    data = load_somm_data(REAL / "somm")
    register_somm_routes(app, settings=settings(), data=data, voice=FakeVoice(), preload=False)
    client = TestClient(app)
    examples = examples_of(script)
    assert len(examples) == 3
    for example in examples:
        card = after.card(example["slug"])
        assert card is not None, example
        assert card["name"] == example["name"] and card["winery"] == example["winery"]
        assert card["color_label"] == example["color"]
        assert card["photo_url"] == f"/v1/wines/{example['slug']}/photo"
        somm = client.get(f"/v1/wines/{example['slug']}/sommelier").json()
        known = [axis for axis in somm["profile"] if axis["value"] is not None]
        assert len(known) >= 6, (example["slug"], len(known))
        assert len(somm["dishes"]) == 3 and somm["note"]["text"]


def test_page_test_knows_the_same_asset_types():
    """`test_product_page.py` повторяет список расширений, чтобы не тянуть FastAPI."""
    assert set(PAGE_TEST_SUFFIXES) == ASSET_SUFFIXES == set(ASSET_TYPES)


def test_static_serves_sheet_files_with_types(tmp_path):
    """Лист сомелье под `/static`: `.js` и `.css` с явным типом и кодировкой.

    Реестр Windows отдаёт `.js` как `text/plain`, и браузер с `nosniff` его бы не запустил.
    Файлов листа до слияния дорожки может не быть — тогда честный 404.
    """
    assert ASSET_TYPES[".js"] == SCRIPT_TYPE and ASSET_TYPES[".css"] == STYLE_TYPE
    with build(tmp_path) as client:
        for name in sorted(SHEET_FILES):
            got = client.get(f"/static/{name}")
            if (PAGE_PATH.parent / name).is_file():
                assert got.status_code == 200, name
                assert got.headers["content-type"] == ASSET_TYPES[Path(name).suffix], name
            else:
                assert got.status_code == 404, name


def test_asset_files_serve_only_design_files(tmp_path):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts" / "Face.woff2").write_bytes(b"wOF2")
    (tmp_path / "logo.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "somm.js").write_text("window.SommUI = {};", encoding="utf-8")
    (tmp_path / "somm.css").write_text(".somm-x{}", encoding="utf-8")
    (tmp_path / "page.html").write_text("<h1>стенд</h1>", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("заметки", encoding="utf-8")
    app = FastAPI()
    app.mount("/static", AssetFiles(directory=tmp_path), name="static")
    with TestClient(app) as client:
        font = client.get("/static/fonts/Face.woff2")
        assert font.status_code == 200 and font.content == b"wOF2"
        assert font.headers["content-type"] == "font/woff2"
        logo = client.get("/static/logo.svg")
        assert logo.status_code == 200 and logo.headers["content-type"] == "image/svg+xml"
        code = client.get("/static/somm.js")
        assert code.status_code == 200 and code.headers["content-type"] == SCRIPT_TYPE
        assert code.text == "window.SommUI = {};"
        css = client.get("/static/somm.css")
        assert css.status_code == 200 and css.headers["content-type"] == STYLE_TYPE
        assert client.get("/static/page.html").status_code == 404
        assert client.get("/static/notes.txt").status_code == 404
        assert client.get("/static/fonts/").status_code == 404
