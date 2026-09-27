"""Цена флага кандидатов в сервисе (CPU): поиск по индексу, слой выбора, загрузка и память.

SigLIP и VLM флаг не трогает, поэтому меряется только то, что меняется: `VisualIndex._rank` (у сервиса
он идёт на CPU numpy и на видеокарточной сборке), проекция индекса при загрузке, память карты и слой
выбора (`FundReplay.resolve`: признаки, ранкер, H5 / P1). Кадры — v2 + kr-dev пула (661), векторы
запросов — `trainpool.npz`; по 3 прохода, берётся медиана и p95 по кадрам третьего прохода.

    PYTHONPATH=. python research/2026-09-26_fund/final/latency.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import protocol as P

SETS = ("catalog_v2", "kr_dev")


def pct(xs: list[float]) -> dict[str, float]:
    return {"p50": round(statistics.median(xs), 3), "p95": round(float(np.percentile(xs, 95)), 3),
            "max": round(max(xs), 3)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    import replay as R

    from app.api.config import CV_PER_SLUG
    from app.features.adapter import LinearAdapter
    from app.features.embedder import unit_rows

    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in SETS]
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    qemb = np.load(P.PROTOCOL / "trainpool.npz")["qemb"]
    adapter_file = P.FUND / "final" / "cv-adapter-lw.npz"
    out: dict[str, object] = {"frames": len(rows), "threads": os.environ.get("OMP_NUM_THREADS", "default")}
    variants = {
        "off": {},
        "adapter-lw": {"SVS_CANDIDATE": "adapter-lw", "SVS_CV_ADAPTER": str(adapter_file)},
        "adapter-lw-ranker": {"SVS_CANDIDATE": "adapter-lw-ranker", "SVS_CV_ADAPTER": str(adapter_file)},
    }
    for name, env in variants.items():
        t = time.perf_counter()
        rp = R.FundReplay(extra_env=env) if env else R.FundReplay()
        load_s = time.perf_counter() - t
        idx = rp.svc.index
        Qs = [unit_rows(np.asarray(qemb[r["row"]], dtype=np.float32)) for r in rows]
        for _ in range(2):
            for Q in Qs:
                idx._rank(Q, 20, CV_PER_SLUG)
        rank_ms, resolve_ms = [], []
        for Q, r in zip(Qs, rows, strict=True):
            t = time.perf_counter()
            cands, margin = idx._rank(Q, 20, CV_PER_SLUG)
            rank_ms.append(1000 * (time.perf_counter() - t))
            v = R.visual_of([[c.slug, c.score, c.view, c.rank] for c in cands], margin, idx.meta.model)
            t = time.perf_counter()
            rp.resolve(v, R.read_of(r["read"]), vlm_status=r["vlm_status"], vlm_lines=r["vlm_lines"])
            resolve_ms.append(1000 * (time.perf_counter() - t))
        out[name] = {"service_load_s": round(load_s, 2), "rank_ms": pct(rank_ms), "resolve_ms": pct(resolve_ms)}
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
    # проекция индекса при загрузке и память карты
    rp = R.FundReplay()
    idx = rp.svc.index
    tracemalloc.start()
    t = time.perf_counter()
    ad = LinearAdapter.load(adapter_file)
    load_ms = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    idx2 = idx.with_adapter(ad)
    project_ms = 1000 * (time.perf_counter() - t)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    Q = unit_rows(np.asarray(qemb[rows[0]["row"]], dtype=np.float32))
    ts = []
    for _ in range(500):
        t = time.perf_counter()
        ad.apply(Q)
        ts.append(1000 * (time.perf_counter() - t))
    out["adapter"] = {
        "file_mb": round(adapter_file.stat().st_size / 2**20, 2),
        "load_ms": round(load_ms, 1),
        "index_projection_ms": round(project_ms, 1),
        "query_projection_ms_4_windows": pct(ts),
        "resident_mb_after_load_and_projection": round(cur / 2**20, 1),
        "peak_mb_during": round(peak / 2**20, 1),
        "note": "проекция индекса — новая матрица 6312×1152 float32 (29 МБ); исходная у сервиса не хранится",
    }
    del idx2
    dst = P.FUND / "final" / "latency.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["adapter"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
