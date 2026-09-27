"""SIFT исходных фото каталога (2 103 slug + второй эталон `merlo-litavshhuk`) → `spatial/refs/`.

    PYTHONPATH=. python research/2026-09-26_fund/spatial/build_refs.py [--workers 8]

Выход: `refs_desc.npy` (n, 128) uint8, `refs_kp.npy` (n, 4) float32, `refs_meta.npz`
(slugs, offsets, wh, paths, ms). Только CPU, только снимок `frozen_data`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import protocol as P
import sift_core as S

OUT = P.FUND / "spatial" / "refs"


def _one(args: tuple[str, str]) -> tuple[str, str, np.ndarray, np.ndarray, int, int, float]:
    S.set_threads(1)
    slug, path = args
    t = time.perf_counter()
    r = S.extract_reference(Path(path))
    return slug, path, r.kp, r.desc, r.w, r.h, (time.perf_counter() - t) * 1e3


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pm = S.photo_map(P.FROZEN)
    jobs = [(slug, str(p)) for slug, ps in sorted(pm.items()) for p in ps]
    # эталоны — только фото каталога, не krasnostop: страж на всякий случай
    P.assert_no_test((), [p for _, p in jobs])
    print(f"эталонов: {len(jobs)} ({len(pm)} slug)", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_one, jobs, chunksize=8))
    kps, descs, slugs, paths, wh, ms = [], [], [], [], [], []
    offsets = [0]
    for slug, path, kp, desc, w, h, t in res:
        slugs.append(slug)
        paths.append(path)
        kps.append(kp)
        descs.append(desc)
        wh.append((w, h))
        ms.append(t)
        offsets.append(offsets[-1] + len(kp))
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "refs_desc.npy", np.concatenate(descs).astype(np.uint8))
    np.save(OUT / "refs_kp.npy", np.concatenate(kps).astype(np.float32))
    np.savez(
        OUT / "refs_meta.npz",
        slugs=np.array(slugs),
        paths=np.array(paths),
        offsets=np.array(offsets, dtype=np.int64),
        wh=np.array(wh, dtype=np.int32),
        ms=np.array(ms, dtype=np.float32),
    )
    n = np.diff(offsets)
    summary = {
        "refs": len(slugs),
        "slugs": len(pm),
        "keypoints_total": int(offsets[-1]),
        "kp_per_ref": {"mean": float(n.mean()), "median": float(np.median(n)), "p05": float(np.percentile(n, 5)), "min": int(n.min())},
        "ms_per_ref_1thread": {"mean": float(np.mean(ms)), "p95": float(np.percentile(ms, 95))},
        "desc_mb": round(int(offsets[-1]) * 128 / 2**20, 1),
        "kp_mb": round(int(offsets[-1]) * 16 / 2**20, 1),
        "wall_s": round(time.perf_counter() - t0, 1),
        "params": {"REF_MAX_W": S.REF_MAX_W, "REF_MAX_H": S.REF_MAX_H, "REF_NFEAT": S.REF_NFEAT},
    }
    (OUT / "refs_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
