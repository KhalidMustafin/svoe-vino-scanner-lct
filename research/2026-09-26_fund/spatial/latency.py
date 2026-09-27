"""Задержка и память пространственной проверки на CPU: один процесс, выборка кадров пула.

    PYTHONPATH=. python research/2026-09-26_fund/spatial/latency.py [--threads 1] [--per-set 40]

Меряется то, что добавилось бы к сервису (разбор кадра у сервиса уже есть — пишется отдельно):
SIFT кадра по видам `full` и `label`, сопоставление с эталонами CV top-5 и top-20
(`match_shortlist`: inliers + различимость; оба вида и только `label`), пик рабочего набора процесса. Кадры — пул разработки (без kr-test).
"""

from __future__ import annotations

import os
import sys

_threads = "1"
for i, a in enumerate(sys.argv):
    if a == "--threads" and i + 1 < len(sys.argv):
        _threads = sys.argv[i + 1]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _threads

import argparse
import ctypes
import json
import random
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import protocol as P
import sift_core as S

OUT = P.FUND / "spatial"


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem_mb() -> tuple[float, float]:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return float("nan"), float("nan")
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs)
    return {"mean": round(float(a.mean()), 1), "p50": round(float(np.percentile(a, 50)), 1), "p95": round(float(np.percentile(a, 95)), 1), "max": round(float(a.max()), 1)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--per-set", type=int, default=40)
    args = ap.parse_args()
    S.set_threads(args.threads)
    m0 = mem_mb()
    rows = P.jsonl(P.PROTOCOL / "trainpool.jsonl")
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    random.seed(7)
    sample = []
    for st in ("catalog_v2", "kr_dev", "pairs_phone"):
        rs = [r for r in rows if r["set"] == st]
        sample += random.sample(rs, min(args.per_set, len(rs)))
    t = time.perf_counter()
    store = S.RefStore(OUT / "refs", mmap=False)  # как в сервисе: всё в памяти
    load_s = time.perf_counter() - t
    m1 = mem_mb()
    res: dict[str, dict[str, list[float]]] = {}
    for r in sample:
        d = res.setdefault(r["set"], {k: [] for k in ("decode", "sift_full", "sift_label", "match5_both", "match20_both", "match5_label", "total5_both", "total5_label", "total20_both")})
        t0 = time.perf_counter()
        gray = S.query_gray(Path(r["image"]).read_bytes())
        t1 = time.perf_counter()
        views = S.query_views(gray)
        sift = S._sift(S.Q_NFEAT)
        qv = {}
        tt = {}
        for name, img in views.items():
            ts = time.perf_counter()
            kps, desc = sift.detectAndCompute(img, None)
            qv[name] = S.QueryView(np.array([k.pt for k in kps], np.float32).reshape(-1, 2), S.root_sift(desc) if desc is not None else np.zeros((0, 128), np.float32), img.shape[1], img.shape[0])
            tt[name] = (time.perf_counter() - ts) * 1e3
        slugs = [c[0] for c in r["cv"]]
        # холодный кэш RootSIFT эталонов: как у нового кадра (эталоны кандидатов другие)
        store._root.clear()
        ts = time.perf_counter()
        for v in qv:
            S.match_shortlist(qv[v], store, slugs[:5])
        m5 = (time.perf_counter() - ts) * 1e3
        store._root.clear()
        ts = time.perf_counter()
        S.match_shortlist(qv["label"], store, slugs[:5])
        m5l = (time.perf_counter() - ts) * 1e3
        store._root.clear()
        ts = time.perf_counter()
        for v in qv:
            S.match_shortlist(qv[v], store, slugs[:20])
        m20 = (time.perf_counter() - ts) * 1e3
        d["decode"].append((t1 - t0) * 1e3)
        d["sift_full"].append(tt["full"])
        d["sift_label"].append(tt["label"])
        d["match5_both"].append(m5)
        d["match20_both"].append(m20)
        d["match5_label"].append(m5l)
        d["total5_both"].append(tt["full"] + tt["label"] + m5)
        d["total5_label"].append(tt["label"] + m5l)
        d["total20_both"].append(tt["full"] + tt["label"] + m20)
    m2 = mem_mb()
    out = {
        "threads": args.threads,
        "frames": len(sample),
        "store_load_s": round(load_s, 2),
        "mem_mb": {"start": round(m0[0]), "after_store": round(m1[0]), "end": round(m2[0]), "peak": round(m2[1])},
        "per_set_ms": {st: {k: pct(v) for k, v in d.items()} for st, d in res.items()},
        "note": "total* = SIFT кадра + сопоставление, без разбора кадра (у сервиса он уже есть)",
    }
    (OUT / f"latency_t{args.threads}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
