"""Контракт HTTP: `/v1/eval/predict` всегда 200 и `slug` наверху — так, как читает скрипт организатора."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import (
    RED,
    FakeOllama,
    image_bytes,
    jq_slug,
    make_service,
    settings_for,
    write_files,
)
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.service import StartupError

PREDICT = "/v1/eval/predict"
#: Фильтр jq из participant_test.sh — слово в слово.
JQ_FILTER = """
      if type == "object" then .slug
      elif type == "array" and length > 0 then .[0].slug
      else empty
      end
      | select(type == "string" and length > 0)
"""


@pytest.fixture
def client():
    service = make_service(FakeOllama("Бета Холмы\nМерло"))
    with TestClient(create_app(service=service, warm=False)) as test_client:
        yield test_client


def post_image(client, data: bytes, filename: str = "q.jpg", field: str = "image"):
    return client.post(PREDICT, files={field: (filename, data, "application/octet-stream")})


def test_predict_returns_slug_on_top_level(client):
    response = post_image(client, image_bytes(RED))
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "beta-merlot"
    assert set(body) >= {
        "slug",
        "confidence",
        "margin",
        "top5",
        "outcome",
        "degraded",
        "timings_ms",
    }
    assert set(body["confidence"]) == {"top1", "top5"}
    assert all(set(item) == {"slug", "score"} for item in body["top5"])
    assert "evidence" not in body
    assert jq_slug(response.text) == "beta-merlot"


@pytest.mark.parametrize(
    ("fmt", "filename", "save"),
    [
        ("PNG", "photo.jpg", {}),  # PNG под расширением .jpg
        ("WEBP", "photo.jpg", {"lossless": True}),  # WebP под .jpg
        ("JPEG", "photo.png", {"quality": 95}),  # JPEG под .png
    ],
)
def test_format_is_sniffed_not_taken_from_name(client, fmt, filename, save):
    response = post_image(client, image_bytes(RED, fmt, **save), filename)
    assert response.status_code == 200
    assert response.json()["slug"] == "beta-merlot"


def test_png_with_alpha_is_accepted(client):
    response = post_image(client, image_bytes((255, 0, 0, 200), "PNG"), "a.png")
    assert response.status_code == 200 and response.json()["slug"]


@pytest.mark.parametrize(
    "data", [b"", b"GIF89a-broken", b"\xff\xd8\xff\xe0" + b"\x00" * 100, os.urandom(2048)]
)
def test_broken_or_empty_file_is_200_with_null_slug(client, data):
    response = post_image(client, data)
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] is None and body["outcome"] == "error" and body["error"]
    assert jq_slug(response.text) is None


def test_empty_body_is_200_with_null_slug(client):
    response = client.post(PREDICT)
    assert response.status_code == 200
    assert response.json()["slug"] is None
    assert response.json()["error"].startswith("bad_request")


def test_wrong_field_name_is_200_with_null_slug(client):
    response = post_image(client, image_bytes(RED), field="file")
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] is None and body["error"].startswith("no_image")


def test_text_field_instead_of_file_is_200_with_null_slug(client):
    response = client.post(PREDICT, data={"image": "not a file"})
    assert response.status_code == 200 and response.json()["slug"] is None


def test_huge_file_is_200_with_null_slug():
    settings = settings_for(max_upload_bytes=4096)
    service = make_service(FakeOllama(""), settings=settings)
    with TestClient(create_app(service=service, warm=False)) as client:
        big = image_bytes(RED, "PNG", size=(64, 64)) + b"\0" * 5000  # чуть больше предела
        response = post_image(client, big)
        assert response.status_code == 200
        assert response.json()["slug"] is None
        assert response.json()["error"].startswith("too_large")
        # запрос сильно больше предела отклоняется по Content-Length, не разбирая multipart
        huge = post_image(client, b"\xff\xd8\xff" + b"\0" * 200_000)
        assert huge.status_code == 200 and huge.json()["error"].startswith("too_large")


def test_exception_inside_service_is_still_200(client, monkeypatch):
    service = client.app.state.service

    def boom(data):
        raise RuntimeError("не должно было случиться")

    monkeypatch.setattr(service, "scan", boom)
    response = post_image(client, image_bytes(RED))
    assert response.status_code == 200
    assert response.json()["slug"] is None and response.json()["error"].startswith("internal")


def test_scan_returns_evidence_and_card(client):
    response = client.post("/v1/scan", files={"image": ("q.png", image_bytes(RED), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "beta-merlot"
    assert body["card"]["slug"] == "beta-merlot" and body["card"]["winery"] == "Бета Холмы"
    assert {"image", "cv", "vlm", "resolve"} <= set(body["evidence"])
    assert body["evidence"]["cv"]["top5"][0]["slug"] == "alfa-muskat"


def test_scan_error_has_no_card(client):
    response = client.post("/v1/scan", files={"image": ("q.png", b"junk", "image/png")})
    assert response.status_code == 200
    assert response.json()["slug"] is None and response.json()["card"] is None


def test_wine_card_and_unknown_slug(client):
    response = client.get("/v1/wines/alfa-muskat")
    assert response.status_code == 200
    card = response.json()
    assert card["name"] == "Мускат" and card["grapes"] == ["Мускат"]
    # сахар — только из названия и slug выгрузки: у «Мускат» (alfa-muskat) его нет
    assert card["sugar"] == "" and card["sugar_class"] is None
    assert card["portal_url"] == "https://vino-svoe.ru/wines/alfa-muskat"
    assert client.get("/v1/wines/no-such-wine").status_code == 404


def test_health_starting_then_ready_after_lifespan_warm():
    service = make_service(FakeOllama("CHATEAU WARMUP"))
    with TestClient(create_app(service=service, warm=False)) as client:
        assert client.get("/v1/health").json()["status"] == "starting"
    service = make_service(FakeOllama("CHATEAU WARMUP"))
    with TestClient(create_app(service=service)) as client:
        health = client.get("/v1/health").json()
        assert health["status"] == "ready"
        assert health["warmed_at"]
        assert {"model", "index", "vlm", "warmed_at"} <= set(health)


def test_start_refused_when_index_model_differs(tmp_path):
    write_files(tmp_path, index_model="fake/other")
    app = create_app(settings=settings_for(tmp_path), warm=False)
    with pytest.raises(StartupError), TestClient(app):
        pass


def _jq() -> str | None:
    for candidate in (shutil.which("jq"), str(Path.home() / "bin" / "jq.exe")):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


@pytest.mark.parametrize("ok", [True, False])
def test_real_jq_filter_reads_the_same_slug(client, ok):
    """Если jq есть на машине — фильтр скрипта организатора на настоящем ответе."""
    jq = _jq()
    if jq is None:
        pytest.skip("jq не найден")
    response = post_image(client, image_bytes(RED) if ok else b"broken")
    binary = ["-b"] if os.name == "nt" else []  # jq.exe без -b пишет CRLF
    run = subprocess.run(
        [jq, *binary, "-er", JQ_FILTER],
        input=response.content,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if ok:
        assert run.returncode == 0 and run.stdout.decode("utf-8") == "beta-merlot\n"
    else:
        assert run.returncode != 0 and run.stdout == b""
    assert jq_slug(response.text) == (run.stdout.decode("utf-8").strip() or None)


def test_body_is_plain_json_without_nan(client):
    response = post_image(client, image_bytes(RED))
    json.loads(response.text, parse_constant=lambda name: pytest.fail(f"{name} в ответе"))


def _asgi_post(app, chunks: list[bytes], headers: list[tuple[bytes, bytes]]) -> tuple[dict, int]:
    """POST прямо в ASGI-приложение: тело отдаётся кусками, счётчик — сколько их прочитано.

    TestClient читает тело запроса целиком до вызова приложения, поэтому, сколько сервис
    успел прочитать, через него не увидеть.
    """
    import asyncio

    sent = {"chunks": 0}
    messages: list[dict] = []

    async def receive() -> dict:
        i = sent["chunks"]
        if i < len(chunks):
            sent["chunks"] += 1
            return {"type": "http.request", "body": chunks[i], "more_body": i + 1 < len(chunks)}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": PREDICT,
        "raw_path": PREDICT.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8080),
        "state": {},
    }
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    assert status == 200
    return json.loads(body), sent["chunks"]


def test_chunked_upload_over_the_limit_is_cut_while_reading():
    """Без Content-Length (chunked) предел держит счёт байт, а не заголовок.

    Раньше тело 10 МБ разбиралось целиком во временный файл и только потом давало too_large.
    """
    from app.api.main import MULTIPART_SLACK

    service = make_service(FakeOllama(""), settings=settings_for(max_upload_bytes=4096))
    app = create_app(service=service, warm=False)
    app.state.service = service
    boundary = "svsboundary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="q.jpg"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    chunk = 16 * 1024
    chunks = [head] + [b"\0" * chunk] * 640 + [f"\r\n--{boundary}--\r\n".encode()]  # 10 МБ
    headers = [
        (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
        (b"transfer-encoding", b"chunked"),
    ]
    body, read = _asgi_post(app, chunks, headers)
    assert body["slug"] is None and body["error"].startswith("too_large")
    assert read * chunk <= 4096 + MULTIPART_SLACK + 2 * chunk  # оборвано у предела
    assert read < len(chunks) // 10
    # а chunked-кадр в пределах предела разбирается как обычно
    ok = make_service(FakeOllama("Бета Холмы\nМерло"))
    ok_app = create_app(service=ok, warm=False)
    ok_app.state.service = ok
    data = image_bytes(RED)
    small = [
        head,
        data[: len(data) // 2],
        data[len(data) // 2 :],
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body, read = _asgi_post(ok_app, small, headers)
    assert body["slug"] == "beta-merlot" and read == len(small)


def test_health_reports_formats_warnings_reasons_and_scan_counters(client):
    post_image(client, image_bytes(RED))
    post_image(client, b"broken")
    client.post(PREDICT)  # не multipart: до сканера не дошло, но в счётчики попало
    health = client.get("/v1/health").json()
    assert health["formats"]["jpeg"] is True and "heic" in health["formats"]
    # у фейков VLM не та, что в замере: предупреждение настроек видно и в health
    assert any("SVS_VLM_MODEL=fake:vlm" in warning for warning in health["warnings"])
    assert health["degraded_reasons"] == []
    scans = health["scans"]
    assert scans["total"] == 3 and scans["errors"] == 2 and scans["null_slug"] == 2
    assert scans["text_read"] == 1


def test_jpeg_over_the_pixel_limit_gets_a_slug(client, monkeypatch):
    """Кадр больше предела пикселей — JPEG уменьшается при разжатии, slug есть."""
    import app.api.service as service_module

    monkeypatch.setattr(service_module, "MAX_PIXELS", 10_000)
    response = client.post(
        "/v1/scan", files={"image": ("big.jpg", image_bytes(RED, "JPEG", (400, 300)), "image/jpeg")}
    )
    body = response.json()
    assert body["slug"] == "beta-merlot" and body["error"] is None
    assert (body["evidence"]["image"]["width"], body["evidence"]["image"]["height"]) == (100, 75)
    png = post_image(client, image_bytes(RED, "PNG", (400, 300)), "big.png").json()
    assert png["slug"] is None and "только у JPEG" in png["error"]
