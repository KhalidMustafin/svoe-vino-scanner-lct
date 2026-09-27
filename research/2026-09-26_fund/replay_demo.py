"""Как повторить конвейер на CPU с другой выдачей CV или другим слоем выбора — проверка и пример.

Только пул разработки (`trainpool.jsonl` + `trainpool.npz`, без kr-test). Три проверки:

1. `resolve_row(запись)` = ответ продукта на каждом кадре (слой выбора по записи без картинки);
2. `cv_scores(Q)` → `candidates_from_scores` = выдача сервиса `index._rank` бит в бит — значит,
   любой свой счёт по всем slug можно подать в тот же слой выбора;
3. пример подмены: CV только по окну `full` запроса (без bottle/label/band) — ответы и счёт.

    PYTHONPATH=. python research/2026-09-26_fund/replay_demo.py [--sets catalog_v2 kr_dev]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P
import replay as R
from app.features.embedder import unit_rows


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["catalog_v2", "kr_dev"])
    args = ap.parse_args()
    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in args.sets]
    z = np.load(P.PROTOCOL / "trainpool.npz")
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    qemb = z["qemb"]
    names = [str(n) for n in z["qemb_names"]]
    rp = R.FundReplay()
    t0 = time.perf_counter()
    same_resolve = same_cv = 0
    ok_prod = ok_full = changed = 0
    for r in rows:
        # 1. слой выбора по записи
        same_resolve += rp.resolve_row(r) == r["answer"]
        # 2. счёт по всем slug -> та же выдача
        Q = unit_rows(qemb[r["row"]].astype(np.float32))
        scores, best_row = rp.cv_scores(Q)
        cands, margin = rp.candidates_from_scores(scores, best_row)
        ref, ref_m = rp.default_cv(Q)
        same_cv += [(c.slug, c.score, c.view) for c in cands] == [
            (c.slug, c.score, c.view) for c in ref
        ] and margin == ref_m
        # 3. подмена: только окно full запроса
        Qf = Q[[names.index("full")]]
        s_full, b_full = rp.cv_scores(Qf)
        c_full, m_full = rp.candidates_from_scores(s_full, b_full)
        ans = rp.resolve(
            R.VisualResult(candidates=c_full, margin=m_full),
            R.read_of(r["read"]),
            vlm_status=r["vlm_status"],
            vlm_lines=r["vlm_lines"],
        )["answer"]
        lab = {"slug": r["slug"], "acceptable": r["acceptable"]}
        ok_prod += P.correct(r["answer"], lab)
        ok_full += P.correct(ans, lab)
        changed += ans != r["answer"]
    n = len(rows)
    print(
        json.dumps(
            {
                "frames": n,
                "resolve_row_equal": same_resolve,
                "cv_from_scores_equal": same_cv,
                "product_same_wine": ok_prod,
                "full_window_only_same_wine": ok_full,
                "answers_changed": changed,
                "ms_per_frame": round(1000 * (time.perf_counter() - t0) / n, 1),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
