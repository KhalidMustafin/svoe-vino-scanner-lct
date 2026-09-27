"""Преполёт полевого стенда: не пинги, а живые ответы.

    python3 scripts/preflight_field.py http://127.0.0.1:8080 кадр.jpg [ключ]

Только стандартная библиотека — запускается на голом сервере, где venv ещё не собран.
Проверяется то, что видно снаружи: страница открывается, кадр доходит до модели, ответ
содержит slug и карточку, кадр лёг в архив, отметка записалась. Отдельно — то, на чём
полевой стенд обычно и ломается: чтение этикетки выброшено по бюджету (`degraded`), HEIC не
читается, фото каталога не отдаётся.

Код выхода 1, если хоть одна проверка упала.
"""

from __future__ import annotations

import io
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✓' if ok else '✗'} {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class CallFailed(Exception):
    """Запрос не дошёл или ответил не тем."""


def call(
    url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = 900.0
):
    request = urllib.request.Request(url, data=data, headers=headers or {})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return body, time.time() - started
    except urllib.error.HTTPError as exc:
        raise CallFailed(f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise CallFailed(f"нет связи: {exc}") from exc


def multipart(path: Path) -> tuple[bytes, str]:
    """Тело multipart с полем `image` — руками, чтобы не тянуть requests на сервер."""
    boundary = uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    return head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(), boundary


def main(argv: list[str]) -> int:
    base = (argv[1] if len(argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
    frame = Path(argv[2]) if len(argv) > 2 else None
    key = argv[3] if len(argv) > 3 else ""
    suffix = f"?k={key}" if key else ""

    print(f"\nПолевой стенд: {base}")

    health: dict = {}
    try:
        body, _ = call(f"{base}/v1/health", timeout=30)
        health = json.loads(body)
    except (CallFailed, json.JSONDecodeError) as exc:
        check("сервис отвечает", False, str(exc))
        print("\nНЕ ГОТОВО: сервис не поднялся")
        return 1

    settings = health.get("settings") or {}
    # Стенд без чтения этикетки — законный режим (`SVS_VLM_TIMEOUT_MS=0`), когда модель
    # чтения на сервер не привезти. Тогда жалобы про VLM ожидаемы и провалом не считаются,
    # но об этом надо сказать вслух: ответы слабее замеров.
    vlm_off = str(settings.get("vlm_timeout_ms")) == "0"
    # Имена ровно те, что ставит сервис: `vlm_warm_disabled` в причинах прогрева
    # (`service.py:691`) и `vlm_disabled` во флагах каждого ответа (`service.py:874`).
    expected = {"vlm_disabled", "vlm_warm_disabled", "warm_scan:vlm_disabled"} if vlm_off else set()

    status = health.get("status")
    check("сервис отвечает", status in {"ready", "degraded"}, f"status={status}")
    if vlm_off:
        print("  ! чтение этикетки выключено: вино выбирается по одной картинке")
    reasons = [r for r in (health.get("degraded_reasons") or []) if r not in expected]
    check("прогрев без жалоб", not reasons, ", ".join(reasons) or "чисто")
    device = (health.get("model") or {}).get("cv_device")
    check("устройство CV названо честно", bool(device), f"cv_device={device}")
    formats = health.get("formats") or {}
    check(
        "HEIC читается (айфон)",
        bool(formats.get("heic")),
        "нет pillow-heif" if not formats.get("heic") else "",
    )

    budget = settings.get("budget_ms")
    vlm_budget = settings.get("vlm_timeout_ms")
    if device == "cpu" and not vlm_off:
        check(
            "бюджеты подняты под процессор",
            bool(budget and budget >= 60_000 and vlm_budget and vlm_budget >= 30_000),
            f"budget_ms={budget}, vlm_timeout_ms={vlm_budget} (на CPU чтение идёт десятки секунд)",
        )

    try:
        # Стенд живёт на `/field`; на `/` — страница продукта, у неё тоже есть `capture=`,
        # поэтому метка — полевой маршрут отправки кадра.
        page, _ = call(f"{base}/field{suffix}", timeout=30)
        check("страница съёмки открывается", b"/v1/field/submit" in page, f"{len(page)} байт")
    except CallFailed as exc:
        check("страница съёмки открывается", False, str(exc))

    if frame is None or not frame.is_file():
        print("\nКадр не передан — проверка живым фото пропущена (это главная проверка).")
        return finish()

    print(f"\nЖивой кадр: {frame.name} ({frame.stat().st_size / 1024:.0f} КБ)")
    # Тем же путём, что и страница: отправка плюс опрос. Синхронный `/v1/field/scan` за
    # туннелем Cloudflare оборвался бы на сотой секунде.
    try:
        data, boundary = multipart(frame)
        body, submit_seconds = call(
            f"{base}/v1/field/submit{suffix}",
            data=data,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=120,
        )
        job = json.loads(body)
        check("кадр принят сразу", bool(job.get("job_id")), f"за {submit_seconds:.1f} с")
        started = time.time()
        answer = {}
        while time.time() - started < 900:
            answer = json.loads(call(f"{base}/v1/field/job/{job['job_id']}{suffix}", timeout=30)[0])
            if answer.get("status") in {"done", "error"}:
                break
            time.sleep(2)
        seconds = time.time() - started
        if answer.get("status") != "done":
            check(
                "разбор доходит до конца",
                False,
                f"status={answer.get('status')} за {seconds:.0f} с",
            )
            return finish()
    except (CallFailed, json.JSONDecodeError, KeyError) as exc:
        check("кадр проходит цепочку", False, str(exc))
        return finish()

    check(
        "кадр проходит цепочку",
        bool(answer.get("slug")),
        f"slug={answer.get('slug')} за {seconds:.0f} с",
    )
    check(
        "карточка вина собрана",
        bool(answer.get("card")),
        (answer.get("card") or {}).get("name", "нет"),
    )
    check("уверенность посчитана", (answer.get("confidence") or {}).get("top1") is not None)
    degraded = [f for f in (answer.get("degraded") or []) if f not in expected]
    check(
        "ответ не урезан бюджетом",
        not degraded,
        ", ".join(degraded) or "флагов нет",
    )
    check("кадр лёг в архив", bool(answer.get("archived")), f"id={answer.get('field_id')}")

    if answer.get("slug"):
        try:
            photo, _ = call(f"{base}/v1/field/photo/{answer['slug']}{suffix}", timeout=30)
            check("фото каталога отдаётся", len(photo) > 500, f"{len(photo) / 1024:.0f} КБ")
        except CallFailed as exc:
            check("фото каталога отдаётся", False, f"{exc} — нужен scripts/make_photo_pack.py")

    if answer.get("field_id"):
        try:
            call(
                f"{base}/v1/field/verdict{suffix}",
                data=json.dumps({"id": answer["field_id"], "verdict": "unknown"}).encode(),
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            stats = json.loads(call(f"{base}/v1/field/stats{suffix}", timeout=30)[0])
            check(
                "отметка записывается",
                stats.get("marked", 0) >= 1,
                f"размечено {stats.get('marked')}",
            )
        except (CallFailed, json.JSONDecodeError) as exc:
            check("отметка записывается", False, str(exc))

    return finish()


def finish() -> int:
    if failures:
        print(f"\nНЕ ГОТОВО: упало {len(failures)} — {', '.join(failures)}")
        return 1
    print("\nСтенд готов: можно начинать съёмку с телефонов.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
