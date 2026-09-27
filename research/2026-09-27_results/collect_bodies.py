"""Полные ответы `/v1/eval/predict` по кадрам манифеста — для top-5, уверенности и отрыва.

Скрипт организатора сохраняет только slug и время. Этот клиент шлёт те же кадры тем же
запросом (multipart `image`) и пишет тело ответа целиком: `top5`, `confidence`, `margin`.

    python research/2026-09-27_results/collect_bodies.py --images-dir <папка кадров> \
        --manifest data/gt/real_photos_queries.tsv --out <папка прогона>/bodies.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/eval/predict")
    args = ap.parse_args()

    with args.manifest.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (
        httpx.Client(timeout=15) as client,
        args.out.open("w", encoding="utf-8", newline="\n") as out,
    ):
        for row in rows:
            path = args.images_dir / row["image_path"]
            t0 = time.perf_counter()
            resp = client.post(args.endpoint, files={"image": (path.name, path.read_bytes())})
            ms = round((time.perf_counter() - t0) * 1000, 1)
            body = resp.json() if resp.status_code == 200 else None
            out.write(
                json.dumps(
                    {
                        "query_id": row["query_id"],
                        "status": resp.status_code,
                        "latency_ms": ms,
                        "body": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(rows)} кадров → {args.out}")


if __name__ == "__main__":
    main()
