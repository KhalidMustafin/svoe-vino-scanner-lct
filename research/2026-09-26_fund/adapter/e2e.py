"""Сквозной счёт «то же вино» с выдачей CV адаптера вне фолда: слой выбора сервиса как есть.

Берёт `adapter/screen/<имя>.npz` (top-20 вне фолда), строит `VisualResult` ровно как
`FundReplay.candidates_from_scores`, подаёт записанное чтение кадра в `FundReplay.resolve`
(ранкер `-goal`, H5, P1, Э2) и сравнивает с ответом продукта кадр в кадр: починки, поломки,
тест знаков по каждому набору. Пул — `trainpool.*` (kr-test нет, `assert_no_test`).

    PYTHONPATH=. python research/2026-09-26_fund/adapter/e2e.py none lw-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import adapter_core as C  # noqa: E402
import protocol as P  # noqa: E402

SETS = ("catalog_v2", "kr_dev", "kr_dev_sp", "pairs", "pairs_phone")


def visual_from(z: Any, i: int, ix: C.IndexData, model: str = "") -> Any:
    import replay as R

    cands = [
        R.Candidate(slug=ix.slug_order[int(s)], score=float(max(0.0, sc)), view=ix.views[int(row)], rank=r)
        for r, (s, sc, row) in enumerate(zip(z["top_idx"][i], z["top_score"][i], z["top_row"][i]), start=1)
    ]
    return R.VisualResult(candidates=cands, margin=float(z["margin"][i]), timings_ms={}, model=model)


def answers(rp: Any, pool: C.Pool, z: Any, ix: C.IndexData, idx: np.ndarray) -> dict[int, str | None]:
    import replay as R

    out = {}
    for i in idx:
        r = pool.rows[i]
        v = visual_from(z, i, ix, rp.svc.index.meta.model)
        out[int(i)] = rp.resolve(v, R.read_of(r["read"]), vlm_status=r["vlm_status"],
                                 vlm_lines=r["vlm_lines"])["answer"]
    return out


def compare(pool: C.Pool, ans: dict[int, str | None], ref: dict[int, str | None] | None = None,
            sets: tuple[str, ...] = SETS) -> dict[str, Any]:
    """Счёт по наборам; `ref` — ответы сравнения (по умолчанию продукт из пула)."""
    res: dict[str, Any] = {}
    tot = {"n": 0, "ok": 0, "ref_ok": 0, "fix": 0, "brk": 0}
    for s in sets:
        ids = [i for i in ans if pool.sets[i] == s and pool.rows[i]["in_catalog"]]
        ok = fix = brk = rok = 0
        for i in ids:
            lab = pool.rows[i]
            a = P.correct(ans[i], lab)
            b = P.correct((ref or {}).get(i, lab["answer"]) if ref is not None else lab["answer"], lab)
            ok += a
            rok += b
            fix += a and not b
            brk += b and not a
        res[s] = {"n": len(ids), "ok": ok, "ref_ok": rok, "fix": fix, "brk": brk,
                  "p": round(P.sign_test(fix, brk), 4)}
        for k, v in zip(("n", "ok", "ref_ok", "fix", "brk"), (len(ids), ok, rok, fix, brk), strict=True):
            tot[k] += v
    tot["p"] = round(P.sign_test(tot["fix"], tot["brk"]), 4)
    res["total"] = tot
    return res


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    import replay as R

    rp = R.FundReplay()
    ix = C.index_data(rp.svc)
    pool = C.load_pool(ix.slug_order)
    idx = np.array([i for i in range(len(pool)) if pool.rows[i]["in_catalog"] and pool.sets[i] in SETS])
    for name in args.names:
        z = np.load(C.OUT / "screen" / f"{name}.npz")
        assert [str(x) for x in z["ids"]] == [r["id"] for r in pool.rows]
        ans = answers(rp, pool, z, ix, idx)
        res = compare(pool, ans)
        # пары dev: на них учился ранкер -goal — отдельно
        dev = {i: a for i, a in ans.items() if pool.rows[i].get("goal_split") != "dev"}
        res_nodev = compare(pool, dev)
        out = {"name": name, "e2e_goal_ranker": res, "e2e_goal_ranker_no_pairs_dev": res_nodev,
               "answers": {pool.rows[i]["id"]: a for i, a in ans.items()}}
        (C.OUT / "screen" / f"{name}.e2e.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
        short = {s: f"{v['ok']}/{v['n']} ({v['ref_ok']}) +{v['fix']}-{v['brk']}" for s, v in res.items() if s != "total"}
        t = res["total"]
        print(f"{name}: {json.dumps(short, ensure_ascii=False)} | total {t['ok']} vs {t['ref_ok']} "
              f"+{t['fix']}-{t['brk']} p={t['p']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
