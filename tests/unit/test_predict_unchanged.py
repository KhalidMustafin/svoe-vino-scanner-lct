"""`/v1/eval/predict` не изменился ни на байт: ключи, значения и сами байты тела ответа.

Эталон `tests/fixtures/predict_golden.json` снят `predict_cases.py` на коде до слоя «после
поиска». Слой добавляет в `/v1/scan` поля `candidates` и `after` и всегда пишет в `evidence`
условия правила отказа, но в тело predict ничего из этого попадать не должно — ни с
загруженным справочником рекомендаций, ни без него. Значения двух случаев сбоя читателя
пересняты намеренно после Э2 (`predict_cases.py`); ключи и их порядок прежние.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import RED, FakeClock, FakeOllama, image_bytes, make_service, settings_for
from fastapi.testclient import TestClient
from predict_cases import GOLDEN, cases, predict_bodies
from reco_env import write_reco

from app.api.after_layer import AfterSearch
from app.api.main import create_app

PREDICT_KEYS = [
    "slug",
    "confidence",
    "margin",
    "top5",
    "outcome",
    "degraded",
    "timings_ms",
    "error",
]


def golden() -> dict[str, str]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_covers_every_case():
    assert set(golden()) == set(cases())


@pytest.mark.parametrize("name", sorted(cases()))
def test_predict_body_is_byte_identical(name):
    assert predict_bodies()[name] == golden()[name]


def test_predict_keys_and_order_are_the_old_ones():
    for name, body in golden().items():
        assert list(json.loads(body)) == PREDICT_KEYS, name


@pytest.mark.parametrize("name", ["matched_by_text", "abstain_ooc_only", "decode_error"])
def test_predict_is_the_same_with_recommendation_data_loaded(tmp_path, name):
    """Справочник, снимок портала и фото на диске не меняют тело predict."""
    case = cases()[name]
    write_reco(tmp_path)
    settings = settings_for(tmp_path, abstain=case.get("abstain", "off"))
    service = make_service(FakeOllama(case["text"]), settings=settings, clock=FakeClock())
    service.after = AfterSearch.load(settings, service.cards, service.attrs)
    assert service.after.catalog.pool, "справочник должен был загрузиться"
    data = case.get("raw") or image_bytes(case.get("color", RED))
    with TestClient(create_app(service=service, warm=False)) as client:
        predict = client.post(
            "/v1/eval/predict", files={"image": ("q.png", data, "application/octet-stream")}
        )
        scan = client.post("/v1/scan", files={"image": ("q.png", data, "application/octet-stream")})
    assert predict.content.decode("utf-8") == golden()[name]
    # а /v1/scan тем временем несёт новые поля
    assert {"candidates", "after", "card", "evidence"} <= set(scan.json())


def test_scan_body_is_predict_plus_four_fields(tmp_path):
    service = make_service(FakeOllama("Бета Холмы\nМерло"), clock=FakeClock())
    result = service.scan(image_bytes(RED))
    scan = service.scan_body(result)
    predict = result.predict_body()
    assert set(scan) - set(predict) == {"evidence", "card", "candidates", "after"}
    assert {key: scan[key] for key in predict} == predict
