"""Таблица признаков пространственной проверки «кадр × кандидат» для трека ранкера.

    PYTHONPATH=. python research/2026-09-26_fund/spatial/table.py

Вход — `spatial/pool_sift_sl.jsonl` (`run_pool.py`, режим `shortlist`). Выход:
- `spatial/sift_features.csv` — строка на пару (id, slug): набор, ранг CV, верный ли, признаки;
- `spatial/sift_features.npz` — `ids` (N), `sift` (N, 20, F) float32 в порядке строк
  `trainpool.npz` и кандидатов CV (как `features`), `sift_names`, `feat_slugs`. Нет кадра или
  кандидата — NaN.

Признаки кандидата (по видам кадра `full` и `label` берётся лучший по числу inliers):
- `sift_inl_log` — log1p(inliers) — основной, объявлен до замера;
- `sift_inl_full_log`, `sift_inl_label_log` — то же по каждому виду;
- `sift_good_log` — log1p(пар после теста отношения);
- `sift_inl_ratio` — inliers / good того же вида;
- `sift_area_label` — доля полосы этикетки эталона под оболочкой inliers;
- `sift_area_q` — доля вида кадра под оболочкой inliers;
- `sift_inl_rel` — inliers / максимум inliers по кандидатам кадра (0, если у всех 0);
- `sift_inl_gap` — log1p(inliers) − максимум log1p(inliers) остальных кандидатов кадра;
- `sift_inl_norm` — inliers / число точек эталона (у эталона высокого разрешения точек больше);
- `sift_uniq_log` — log1p(различимых inliers внутри top-20, `sift_core.match_shortlist`);
- `sift_inl5_log`, `sift_uniq5_log` — то же, но проверка только у top-5 (различимость внутри
  top-5), у рангов 6–20 — 0: вариант, укладывающийся в бюджет задержки.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import protocol as P

OUT = P.FUND / "spatial"
MATCH_FIELDS = ["good", "inliers", "raw_inliers", "valid", "area_ref", "area_label", "area_q", "ms"]
SIFT_NAMES = [
    "sift_inl_log",
    "sift_inl_full_log",
    "sift_inl_label_log",
    "sift_good_log",
    "sift_inl_ratio",
    "sift_area_label",
    "sift_area_q",
    "sift_inl_rel",
    "sift_inl_gap",
    "sift_inl_norm",
    "sift_uniq_log",
    "sift_inl5_log",
    "sift_uniq5_log",
]
UNIQ_GOOD, UNIQ_INL = 8, 9  # позиции в строке вида режима shortlist


def load_sift(path: Path = OUT / "pool_sift_sl.jsonl") -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def cand_features(cands: Sequence[Mapping[str, Any]], ref_kp: Mapping[str, int] | None = None) -> list[dict[str, float]]:
    """Признаки кандидатов одного кадра (порядок — как во входе, то есть CV)."""
    base = []
    for i, c in enumerate(cands):
        f = dict(zip(MATCH_FIELDS, c["full"][:8], strict=True))
        lab = dict(zip(MATCH_FIELDS, c["label"][:8], strict=True))
        uniq = max(c["full"][UNIQ_INL], c["label"][UNIQ_INL]) if len(c["full"]) > UNIQ_INL else 0
        inl5 = max(c["full5"][2], c["label5"][2]) if "full5" in c else 0
        uniq5 = max(c["full5"][1], c["label5"][1]) if "full5" in c else 0
        nkp = (ref_kp or {}).get(c["slug"], 0)
        best = f if (f["inliers"], f["good"]) >= (lab["inliers"], lab["good"]) else lab
        inl = max(f["inliers"], lab["inliers"])
        base.append(
            {
                "inl": float(inl),
                "sift_inl_log": math.log1p(inl),
                "sift_inl_full_log": math.log1p(f["inliers"]),
                "sift_inl_label_log": math.log1p(lab["inliers"]),
                "sift_good_log": math.log1p(max(f["good"], lab["good"])),
                "sift_inl_ratio": best["inliers"] / best["good"] if best["good"] else 0.0,
                "sift_area_label": float(max(f["area_label"], lab["area_label"])),
                "sift_area_q": float(max(f["area_q"], lab["area_q"])),
                "sift_inl_norm": inl / nkp if nkp else 0.0,
                "sift_uniq_log": math.log1p(uniq),
                "sift_inl5_log": math.log1p(inl5) if i < 5 else 0.0,
                "sift_uniq5_log": math.log1p(uniq5) if i < 5 else 0.0,
                "uniq": float(uniq),
            }
        )
    top = max((b["inl"] for b in base), default=0.0)
    logs = [b["sift_inl_log"] for b in base]
    for i, b in enumerate(base):
        b["sift_inl_rel"] = b["inl"] / top if top > 0 else 0.0
        others = logs[:i] + logs[i + 1 :]
        b["sift_inl_gap"] = b["sift_inl_log"] - (max(others) if others else 0.0)
    return base


def ref_keypoints() -> dict[str, int]:
    """slug → число точек SIFT его эталона (у двух эталонов — больше из двух)."""
    meta = np.load(OUT / "refs" / "refs_meta.npz")
    n = np.diff(meta["offsets"])
    out: dict[str, int] = {}
    for s, k in zip(meta["slugs"], n, strict=True):
        out[str(s)] = max(out.get(str(s), 0), int(k))
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    rows = P.jsonl(P.PROTOCOL / "trainpool.jsonl")
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    z = np.load(P.PROTOCOL / "trainpool.npz")
    feat_slugs = z["feat_slugs"]
    sift = load_sift()
    P.assert_no_test(sift.keys())
    ref_kp = ref_keypoints()
    N = len(z["ids"])
    arr = np.full((N, 20, len(SIFT_NAMES)), np.nan, dtype=np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with (OUT / "sift_features.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["id", "set", "slug", "cv_rank", "correct", "full_good", "full_inliers", "full_valid", "label_good", "label_inliers", "label_valid", *SIFT_NAMES]
        )
        for r in rows:
            d = sift.get(r["id"])
            if d is None:
                continue
            ok = {r["slug"], *(r.get("acceptable") or [])}
            feats = cand_features(d["cands"], ref_kp)
            fs = [str(s) for s in feat_slugs[r["row"]]]
            for k, (c, f) in enumerate(zip(d["cands"], feats, strict=True)):
                assert c["slug"] == r["cv"][k][0]
                # у кадров Э2 (сбой читателя) признаков ранкера нет: feat_slugs пустые, порядок — CV
                if fs[k] and fs[k] != c["slug"]:
                    raise AssertionError(f"порядок кандидатов не совпал: {r['id']} {k}")
                arr[r["row"], k] = [f[n] for n in SIFT_NAMES]
                w.writerow(
                    [r["id"], r["set"], c["slug"], k + 1, int(c["slug"] in ok), c["full"][0], c["full"][1], c["full"][3], c["label"][0], c["label"][1], c["label"][3]]
                    + [round(f[n], 5) for n in SIFT_NAMES]
                )
                n_pairs += 1
    np.savez_compressed(OUT / "sift_features.npz", ids=z["ids"], sift=arr, sift_names=np.array(SIFT_NAMES), feat_slugs=feat_slugs)
    print(f"кадров с признаками: {len(sift)}, пар: {n_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
