"""Полевой контур: кадр и ответ ложатся в архив, отметка пишется, боевой контракт не тронут."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import RED, FakeOllama, image_bytes, make_service, settings_for
from fastapi.testclient import TestClient

from app.api.field import FieldSettings
from app.api.main import create_app

PAGE = "/field"
SCAN = "/v1/field/scan"
VERDICT = "/v1/field/verdict"
STATS = "/v1/field/stats"
#: Метка страницы стенда: её нет на странице продукта (`/`).
FIELD_MARK = "<title>Полевой сканер"


def build(tmp_path: Path, *, token: str = "", enabled: bool = True) -> TestClient:
    """Приложение с полевым контуром; контур включён, как на VPS (`SVS_FIELD=1`)."""
    service = make_service(FakeOllama("Бета Холмы\nМерло"), settings=settings_for(tmp_path))
    field = FieldSettings(directory=tmp_path / "field", token=token, enabled=enabled)
    return TestClient(create_app(service=service, warm=False, field=field))


def post_frame(client: TestClient, data: bytes | None = None, **params: str):
    return client.post(
        SCAN,
        params=params,
        files={"image": ("q.jpg", data or image_bytes(RED), "application/octet-stream")},
    )


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frame_and_answer_land_in_archive(tmp_path):
    with build(tmp_path) as client:
        body = post_frame(client, who="Съёмка").json()

    assert body["slug"] == "beta-merlot"
    assert body["archived"] is True
    scan_id = body["field_id"]

    rows = read_lines(tmp_path / "field" / "index.jsonl")
    assert len(rows) == 1
    assert rows[0]["id"] == scan_id
    assert rows[0]["slug"] == "beta-merlot"
    assert rows[0]["client"] == "Съёмка"

    frame = tmp_path / "field" / rows[0]["file"]
    assert frame.is_file()
    assert frame.read_bytes() == image_bytes(RED), "кадр должен лежать байт в байт, как прислан"
    assert frame.suffix == ".png", "расширение — по сигнатуре байтов, а не по имени файла"

    saved = json.loads(frame.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["response"]["slug"] == "beta-merlot"
    assert saved["response"]["card"] is not None


def test_battle_routes_write_nothing(tmp_path):
    """`/v1/eval/predict` и `/v1/scan` про архив не знают: контракт организатора не меняем."""
    with build(tmp_path) as client:
        for path in ("/v1/eval/predict", "/v1/scan"):
            answer = client.post(
                path, files={"image": ("q.jpg", image_bytes(RED), "application/octet-stream")}
            )
            assert answer.status_code == 200
            assert "field_id" not in answer.json()

    assert not (tmp_path / "field").exists()


def test_verdict_appends_and_counts(tmp_path):
    with build(tmp_path) as client:
        scan_id = post_frame(client).json()["field_id"]
        assert client.post(VERDICT, json={"id": scan_id, "verdict": "ok"}).status_code == 200
        # Передумали: строка дописывается, прежняя остаётся.
        assert (
            client.post(
                VERDICT, json={"id": scan_id, "verdict": "wrong", "correct_slug": "alma-rose"}
            ).status_code
            == 200
        )
        stats = client.get(STATS).json()

    rows = read_lines(tmp_path / "field" / "verdicts.jsonl")
    assert [row["verdict"] for row in rows] == ["ok", "wrong"]
    assert rows[1]["correct_slug"] == "alma-rose"
    assert stats["scans"] == 1
    assert stats["marked"] == 1, "две отметки на один кадр — это один размеченный кадр"
    assert stats["verdicts"]["wrong"] == 1, "считается последняя отметка"
    assert stats["accuracy"] == 0.0


def test_unknown_verdict_refused(tmp_path):
    with build(tmp_path) as client:
        scan_id = post_frame(client).json()["field_id"]
        assert client.post(VERDICT, json={"id": scan_id, "verdict": "ага"}).status_code == 400
        assert client.post(VERDICT, json={"verdict": "ok"}).status_code == 400
    assert not (tmp_path / "field" / "verdicts.jsonl").exists()


def test_token_closes_page_and_routes(tmp_path):
    with build(tmp_path, token="s3cret") as client:
        assert client.get(PAGE).status_code == 401
        assert post_frame(client).status_code == 401
        assert client.get(STATS).status_code == 401

        assert client.get(PAGE, params={"k": "s3cret"}).status_code == 200
        assert post_frame(client, k="s3cret").json()["archived"] is True
        assert client.get(STATS, headers={"X-Field-Key": "s3cret"}).status_code == 200

        # Боевые маршруты ключом не закрыты: скрипт организатора о нём не знает.
        assert client.get("/v1/health").status_code == 200
        # Страница продукта — тоже: она ничего не пишет и ключа не знает.
        product = client.get("/")
        assert product.status_code == 200 and FIELD_MARK not in product.text


def test_old_link_with_key_goes_to_field_page(tmp_path):
    """Старые ссылки стенда `/?k=КЛЮЧ`: при включённом контуре они ведут на `/field`."""
    with build(tmp_path, token="s3cret") as client:
        moved = client.get("/", params={"k": "s3cret", "who": "Съёмка"}, follow_redirects=False)
        assert moved.status_code == 307, "временно: постоянное перенаправление браузер запомнит"
        assert (
            moved.headers["location"] == "/field?k=s3cret&who=%D0%A1%D1%8A%D1%91%D0%BC%D0%BA%D0%B0"
        )

        page = client.get("/", params={"k": "s3cret"})
        assert page.status_code == 200
        assert FIELD_MARK in page.text
        # Неверный ключ перенаправляется так же, а закрывает его уже сам стенд.
        assert client.get("/", params={"k": "мимо"}).status_code == 401


def test_page_offers_camera_and_gallery(tmp_path):
    with build(tmp_path) as client:
        page = client.get(PAGE)

    assert page.status_code == 200
    assert FIELD_MARK in page.text
    assert 'capture="environment"' in page.text, "съёмка — нативной камерой, без видоискателя"
    assert page.text.count('<input type="file"') >= 2, "нужен и выбор из галереи"
    assert "http://" not in page.text.replace("http://адрес", ""), "страница не ходит наружу"


def wait_for(client: TestClient, job_id: str, *, tries: int = 100) -> dict:
    """Опрос до конца разбора — так же, как это делает страница."""
    for _ in range(tries):
        body = client.get(f"/v1/field/job/{job_id}").json()
        if body["status"] in {"done", "error"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"разбор {job_id} не кончился: {body}")


def test_submit_returns_at_once_and_result_comes_by_polling(tmp_path):
    """Через туннель Cloudflare синхронный ответ не проходит: 100 с — и обрыв."""
    with build(tmp_path) as client:
        accepted = client.post(
            "/v1/field/submit",
            params={"who": "Съёмка"},
            files={"image": ("q.jpg", image_bytes(RED), "application/octet-stream")},
        )
        assert accepted.status_code == 202
        submitted = accepted.json()
        assert submitted["status"] in {"queued", "running", "done"}
        assert submitted["job_id"]
        assert "slug" not in submitted, "ответа ещё нет — только номер очереди"

        body = wait_for(client, submitted["job_id"])

    assert body["status"] == "done"
    assert body["slug"] == "beta-merlot"
    assert body["archived"] is True
    assert body["card"]["name"]
    rows = read_lines(tmp_path / "field" / "index.jsonl")
    assert len(rows) == 1 and rows[0]["client"] == "Съёмка"


def test_verdict_works_for_queued_scan(tmp_path):
    with build(tmp_path) as client:
        job = client.post(
            "/v1/field/submit",
            files={"image": ("q.jpg", image_bytes(RED), "application/octet-stream")},
        ).json()
        body = wait_for(client, job["job_id"])
        answer = client.post("/v1/field/verdict", json={"id": body["field_id"], "verdict": "ok"})
        assert answer.status_code == 200
    assert read_lines(tmp_path / "field" / "verdicts.jsonl")[0]["verdict"] == "ok"


def test_unknown_job_answers_404(tmp_path):
    with build(tmp_path) as client:
        assert client.get("/v1/field/job/нетакого").status_code == 404


def test_missing_photo_answers_404(tmp_path):
    with build(tmp_path) as client:
        assert client.get("/v1/field/photo/beta-merlot").status_code == 404
        assert client.get("/v1/field/photo/нет-такого").status_code == 404


def test_field_photo_comes_from_field_photo_dir(tmp_path):
    photos = tmp_path / "small"
    photos.mkdir()
    (photos / "beta-merlot.webp").write_bytes(b"RIFF....WEBPVP8 ")
    service = make_service(FakeOllama(""), settings=settings_for(tmp_path))
    field = FieldSettings(directory=tmp_path / "field", photo_dir=photos, enabled=True)
    with TestClient(create_app(service=service, warm=False, field=field)) as client:
        answer = client.get("/v1/field/photo/beta-merlot")
    assert answer.status_code == 200
    assert answer.content == b"RIFF....WEBPVP8 "


def test_without_flag_field_routes_do_not_exist(tmp_path):
    """152-ФЗ: без `SVS_FIELD` продукт не пишет кадры — маршрутов записи нет вовсе."""
    with build(tmp_path, enabled=False) as client:
        assert client.get(PAGE).status_code == 404
        assert client.get("/static/field.html").status_code == 404, "и в обход через статику"
        # На `/` — страница продукта, и старая ссылка стенда тоже открывает её, а не `/field`.
        for params in ({}, {"k": "s3cret"}):
            product = client.get("/", params=params, follow_redirects=False)
            assert product.status_code == 200 and FIELD_MARK not in product.text
        assert post_frame(client).status_code == 404
        submit = client.post(
            "/v1/field/submit",
            files={"image": ("q.jpg", image_bytes(RED), "application/octet-stream")},
        )
        assert submit.status_code == 404
        assert client.get(STATS).status_code == 404
        assert client.get("/v1/field/photo/beta-merlot").status_code == 404
        # боевые маршруты и слой после поиска на месте
        scan = client.post(
            "/v1/scan", files={"image": ("q.jpg", image_bytes(RED), "application/octet-stream")}
        )
        assert scan.status_code == 200 and scan.json()["slug"] == "beta-merlot"
        assert client.get("/v1/wines/beta-merlot").status_code == 200
    assert not (tmp_path / "field").exists()


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("", False), ("0", False), ("no", False), ("1", True), ("true", True), ("ON", True)],
)
def test_flag_is_read_from_environment(tmp_path, value, enabled):
    settings = FieldSettings.from_env({"SVS_FIELD": value}, tmp_path)
    assert settings.enabled is enabled
    assert FieldSettings.from_env({}, tmp_path).enabled is False


def test_create_app_without_field_settings_follows_env(tmp_path, monkeypatch):
    service = make_service(FakeOllama(""), settings=settings_for(tmp_path))
    monkeypatch.delenv("SVS_FIELD", raising=False)
    with TestClient(create_app(service=service, warm=False)) as client:
        assert client.get(STATS).status_code == 404
    monkeypatch.setenv("SVS_FIELD", "1")
    monkeypatch.setenv("SVS_FIELD_DIR", str(tmp_path / "field"))
    with TestClient(create_app(service=service, warm=False)) as client:
        assert client.get(STATS).status_code == 200


def test_env_example_turns_field_on():
    """Полевой VPS живёт по `deploy/svs-field.env.example`: там контур должен быть включён."""
    example = Path(__file__).resolve().parents[2] / "deploy" / "svs-field.env.example"
    lines = [line.strip() for line in example.read_text(encoding="utf-8").splitlines()]
    assert "SVS_FIELD=1" in lines
