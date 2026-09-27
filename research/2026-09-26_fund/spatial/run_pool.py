"""SIFT + RANSAC для CV top-20 каждого кадра пула разработки → `spatial/pool_sift.jsonl`.

    PYTHONPATH=. python research/2026-09-26_fund/spatial/run_pool.py [--workers 10] [--sets ...]

Пул — `protocol/trainpool.jsonl` (kr-test в нём нет; страж `assert_no_test` проверяет каждый id
и каждую картинку до работы и ещё раз в каждом процессе). Строка на кадр: число точек и время
по видам кадра (`full`, `label`), по кандидату top-20 — `good`, `inliers`, `raw_inliers`,
`valid`, `area_ref`, `area_label`, `area_q`, `ms` по каждому виду; в режиме `shortlist`
(по умолчанию, `pool_sift_sl.jsonl`) ещё `uniq_good`, `uniq_inl` — различимость внутри top-20
(`sift_core.match_shortlist`), и у первых пяти `<вид>5` = [uniq_good, uniq_inl, inliers] внутри
top-5. Режим `per_cand` — прежний проход по кандидату (`pool_sift.jsonl`). Дописывается: готовые id
пропускаются. Только CPU, один поток на процесс.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import protocol as P
import sift_core as S

OUT = P.FUND / "spatial"
SETS = ["catalog_v2", "kr_dev", "kr_dev_sp", "pairs", "pairs_phone", "ooc_v2"]

_STORE: S.RefStore | None = None


def _init() -> None:
    global _STORE
    S.set_threads(1)
    _STORE = S.RefStore(OUT / "refs")


def _m(m: S.Match) -> list[Any]:
    return [m.good, m.inliers, m.raw_inliers, int(m.valid), round(m.area_ref, 5), round(m.area_label, 5), round(m.area_q, 5), round(m.ms, 2)]


MATCH_FIELDS = ["good", "inliers", "raw_inliers", "valid", "area_ref", "area_label", "area_q", "ms"]


def process(job: dict[str, Any]) -> dict[str, Any]:
    P.assert_no_test([job["id"]], [job["image"]])
    assert _STORE is not None
    t0 = time.perf_counter()
    gray = S.query_gray(Path(job["image"]).read_bytes())
    t1 = time.perf_counter()
    views = S.query_views(gray)
    qv: dict[str, S.QueryView] = {}
    sift_ms: dict[str, float] = {}
    sift = S._sift(S.Q_NFEAT)
    for name, img in views.items():
        ts = time.perf_counter()
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None or not kps:
            qv[name] = S.QueryView(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), img.shape[1], img.shape[0])
        else:
            qv[name] = S.QueryView(np.array([k.pt for k in kps], dtype=np.float32), S.root_sift(desc), img.shape[1], img.shape[0])
        sift_ms[name] = round((time.perf_counter() - ts) * 1e3, 1)
    cands = []
    slugs = [c[0] for c in job["cv"]]
    if job.get("mode") == "shortlist":
        # один проход по всему списку: те же поля + различимость по top-20 и по top-5
        t_sl = {}
        per_view = {}
        for v in S.QUERY_VIEWS:
            ts = time.perf_counter()
            per_view[v] = S.match_shortlist(qv[v], _STORE, slugs)
            t_sl[v] = round((time.perf_counter() - ts) * 1e3, 1)
            ts = time.perf_counter()
            per_view[v + "5"] = S.match_shortlist(qv[v], _STORE, slugs[:5])
            t_sl[v + "5"] = round((time.perf_counter() - ts) * 1e3, 1)
        for i, slug in enumerate(slugs):
            c: dict[str, Any] = {"slug": slug}
            for v in S.QUERY_VIEWS:
                m = per_view[v][i]
                c[v] = _m(m) + [m.extra.get("uniq_good", 0), m.extra.get("uniq_inl", 0)]
                if i < 5:
                    m5 = per_view[v + "5"][i]
                    c[v + "5"] = [m5.extra.get("uniq_good", 0), m5.extra.get("uniq_inl", 0), m5.inliers]
            cands.append(c)
    else:
        t_sl = {}
        for slug in slugs:
            mm = S.match_slug(qv, _STORE, slug)
            cands.append({"slug": slug, **{v: _m(mm[v]) for v in S.QUERY_VIEWS}})
    return {
        "id": job["id"],
        "set": job["set"],
        "decode_ms": round((t1 - t0) * 1e3, 1),
        "sift_ms": sift_ms,
        "n_kp": {k: len(v.kp) for k, v in qv.items()},
        "view_wh": {k: [v.w, v.h] for k, v in qv.items()},
        "wall_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "match_ms": t_sl,
        "cands": cands,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=["per_cand", "shortlist"], default="shortlist")
    args = ap.parse_args()
    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in args.sets]
    n = P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    print(f"кадров {len(rows)}, страж проверил {n} id и картинок", flush=True)
    out = Path(args.out or OUT / ("pool_sift_sl.jsonl" if args.mode == "shortlist" else "pool_sift.jsonl"))
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.open(encoding="utf-8") if line.strip()}
    jobs = [{"id": r["id"], "set": r["set"], "image": r["image"], "cv": r["cv"], "mode": args.mode} for r in rows if r["id"] not in done]
    print(f"уже готово {len(done)}, осталось {len(jobs)}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex, out.open("a", encoding="utf-8") as fh:
        futs = [ex.submit(process, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            fh.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
            if k % 100 == 0:
                fh.flush()
                el = time.perf_counter() - t0
                print(f"{k}/{len(jobs)} {el:.0f} s, ~{el / k * (len(jobs) - k):.0f} s осталось", flush=True)
    print(f"готово за {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
