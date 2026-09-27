"""Кадры, на которых сверяется тело `/v1/eval/predict` до и после слоя «после поиска».

Скрипт организатора читает только `/v1/eval/predict`, поэтому его ответ не должен меняться ни
на байт. Эталон `tests/fixtures/predict_golden.json` снят этим модулем на коде до слоя (ветка
after-search, a89998c): тело HTTP-ответа как есть, байтами. Часы подставные и стоят на месте,
поэтому `timings_ms` детерминированы.

Намеренно пересняты два случая сбоя читателя — `ambiguous_cv_only` (пустой ответ VLM) и
`vlm_timeout`. После Э2 (`research/2026-09-25_acc/PREREG_E2_reader_fallback.md`) там ответ —
CV top-1: `top5` в порядке CV со счётами CV, `confidence` null, `outcome` `matched`. Ключи и
остальные пять случаев — байт в байт прежние. Имя `ambiguous_cv_only` оставлено, чтобы эталон
сравнивался по тем же ключам.

Пересобрать эталон (только если predict меняют намеренно):

    python tests/unit/predict_cases.py

Модуль не собирается pytest (имя не начинается с `test_`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GOLDEN = HERE.parent / "fixtures" / "predict_golden.json"

BETA_TEXT = "Бета Холмы\nМерло"
GAMMA_TEXT = "Гамма Берег\nМерло"


def cases() -> dict[str, dict[str, Any]]:
    """Имя случая → что отправить: текст VLM, цвет кадра, настройки, ошибка транспорта."""
    from api_env import BLUE, RED

    return {
        "matched_by_text": {"text": BETA_TEXT, "color": RED},
        "ambiguous_cv_only": {"text": "", "color": RED},
        "weak_picture_abstain_off": {"text": GAMMA_TEXT, "color": BLUE},
        "abstain_ooc_only": {"text": GAMMA_TEXT, "color": BLUE, "abstain": "ooc_only"},
        "vlm_timeout": {"text": BETA_TEXT, "color": RED, "timeout": True},
        "decode_error": {"text": BETA_TEXT, "raw": b"junk"},
        "no_image_field": {"text": BETA_TEXT, "field": "file"},
    }


def predict_bodies() -> dict[str, str]:
    """Случай → тело ответа `/v1/eval/predict` строкой (UTF-8), как его получил бы скрипт."""
    from api_env import RED, FakeClock, FakeOllama, image_bytes, make_service, settings_for
    from fastapi.testclient import TestClient

    from app.api.main import create_app

    out: dict[str, str] = {}
    for name, case in cases().items():
        settings = settings_for(abstain=case.get("abstain", "off"))
        error = TimeoutError("timed out") if case.get("timeout") else None
        service = make_service(
            FakeOllama(case["text"], error=error), settings=settings, clock=FakeClock()
        )
        data = case.get("raw") or image_bytes(case.get("color", RED))
        field = case.get("field", "image")
        with TestClient(create_app(service=service, warm=False)) as client:
            response = client.post(
                "/v1/eval/predict", files={field: ("q.png", data, "application/octet-stream")}
            )
        assert response.status_code == 200
        out[name] = response.content.decode("utf-8")
    return out


def main() -> int:
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parents[1]))
    bodies = predict_bodies()
    GOLDEN.write_text(json.dumps(bodies, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{GOLDEN}: {len(bodies)} случаев")
    return 0


if __name__ == "__main__":
    sys.exit(main())
