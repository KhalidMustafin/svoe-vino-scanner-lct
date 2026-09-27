"""Э2, пункт (3) PREREG: живая проверка запасного ответа в GPU-окне — Ollama остановлена.

Запускается вручную в окне с видеокартой, против уже поднятого сервиса ветки с Э2. Скрипт
сам ничего не останавливает и не запускает: только HTTP к сервису. Порядок:

    1. сервис с Э2 поднят и прогрет (`/v1/health` отвечает);
    2. Ollama остановлена (например, `ollama stop` модели и выход из трея, или остановка службы);
    3. python research/2026-09-25_acc/e2_window_check.py --images-dir <каталог кадров> \
           [--url http://127.0.0.1:8080] [--n 20] [--out runs/e2_window]

Приёмка (PREREG_E2_reader_fallback.md, пункт (3), без изменений):
    20 кадров → 0 null; ответы = CV top-1; p95 < 3 с.
Как это проверяется здесь:
    - каждый кадр идёт по очереди в `/v1/eval/predict`, как у скрипта организатора; время — на
      стороне клиента, от отправки до последнего байта; p95 — по ближайшему рангу
      (19-е значение из 20);
    - тот же кадр затем идёт в `/v1/scan`: slug должен совпасть с predict и с
      `evidence.cv.top5[0].slug`, а `evidence.resolve.fallback` — начинаться с `reader_`;
    - ключи тела predict — прежние восемь, по порядку;
    - счётчик `/v1/health` `scans.null_slug` за прогон не растёт.
Если ни один кадр не ушёл на запасной путь, Ollama, видимо, не остановлена: проверять нечего,
код выхода 2. Остановить её можно и после прогрева сервиса — `/v1/health` тогда ещё `ready`.

Коды выхода: 0 — (3) выполнено; 1 — не выполнено; 2 — окружение не то.
Только стандартная библиотека: запускается любым Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif"}
P95_LIMIT_MS = 3000.0
TIMEOUT_S = 30.0


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def post_image(url: str, path: Path) -> tuple[bytes, float]:
    """multipart с полем `image`, как `curl -F image=@кадр`; время — до последнего байта ответа."""
    boundary = uuid.uuid4().hex
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        raw = response.read()
    return raw, (time.perf_counter() - started) * 1000


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, help="каталог для e2_window.json (по желанию)")
    args = ap.parse_args()

    images = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[: args.n]
    if len(images) < args.n:
        print(f"ОШИБКА: в {args.images_dir} {len(images)} кадров, нужно {args.n}")
        return 2
    base = args.url.rstrip("/")
    try:
        health_before = get_json(f"{base}/v1/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"ОШИБКА: сервис не отвечает на {base}/v1/health: {exc}")
        return 2
    print(f"health: {health_before.get('status')}, причины {health_before.get('degraded_reasons')}")

    rows: list[dict[str, Any]] = []
    for path in images:
        raw, ms = post_image(f"{base}/v1/eval/predict", path)
        body = json.loads(raw.decode("utf-8"))
        scan, _ = post_image(f"{base}/v1/scan", path)
        full = json.loads(scan.decode("utf-8"))
        evidence = full.get("evidence") or {}
        cv_top = ((evidence.get("cv") or {}).get("top5") or [{}])[0].get("slug")
        fallback = (evidence.get("resolve") or {}).get("fallback")
        row = {
            "image": path.name,
            "ms": round(ms, 1),
            "slug": body.get("slug"),
            "scan_slug": full.get("slug"),
            "cv_top1": cv_top,
            "fallback": fallback,
            "degraded": body.get("degraded"),
            "keys_ok": list(body) == PREDICT_KEYS,
        }
        row["ok"] = (
            row["slug"] is not None
            and row["slug"] == row["scan_slug"] == cv_top
            and isinstance(fallback, str)
            and fallback.startswith("reader_")
            and row["keys_ok"]
        )
        rows.append(row)
        print(
            f"{path.name}: {ms:7.0f} мс  slug={row['slug']}  cv_top1={cv_top}  "
            f"fallback={fallback}  degraded={row['degraded']}  {'ok' if row['ok'] else 'НЕ ТАК'}"
        )
    health_after = get_json(f"{base}/v1/health")
    null_before = (health_before.get("scans") or {}).get("null_slug", 0)
    null_after = (health_after.get("scans") or {}).get("null_slug", 0)
    times = [r["ms"] for r in rows]
    summary = {
        "frames": len(rows),
        "null": sum(r["slug"] is None for r in rows),
        "answer_eq_cv_top1": sum(r["slug"] is not None and r["slug"] == r["cv_top1"] for r in rows),
        "fallback_marked": sum(bool(r["fallback"]) for r in rows),
        "keys_ok": sum(r["keys_ok"] for r in rows),
        "p95_ms": round(p95(times), 1),
        "max_ms": round(max(times), 1),
        "null_slug_counter_delta": null_after - null_before,
    }
    passed = (
        summary["null"] == 0
        and all(r["ok"] for r in rows)
        and summary["p95_ms"] < P95_LIMIT_MS
        and summary["null_slug_counter_delta"] == 0
    )
    summary["passed"] = passed
    if not summary["fallback_marked"]:
        print(json.dumps(summary, ensure_ascii=False))
        print("ОШИБКА: ни один кадр не ушёл на запасной путь — Ollama, похоже, не остановлена")
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    print("(3) ВЫПОЛНЕНО" if passed else "(3) НЕ ВЫПОЛНЕНО")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "e2_window.json").write_text(
            json.dumps(
                {
                    "summary": summary,
                    "rows": rows,
                    "health_before": health_before,
                    "health_after": health_after,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
