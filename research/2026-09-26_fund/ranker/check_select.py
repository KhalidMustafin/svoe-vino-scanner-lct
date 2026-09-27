"""Проверка: `common.select` по записанным признакам = слой выбора сервиса (`FundReplay.resolve`).

1. модель `-goal` сервиса: `select` = ответ продукта (`answer` в trainpool) на всех кадрах пула;
2. другая модель (listwise на всём пуле, L2 0,01): `select` = `FundReplay.resolve` с этой моделью
   в сервисе (`svc.model` подменён) — на всех кадрах пула;
3. фолды: размеры по срезам, ни одна группа вина не делится.

    PYTHONPATH=. python research/2026-09-26_fund/ranker/check_select.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import protocol as P  # noqa: E402
import replay as R  # noqa: E402

from app.resolve.learned import LogisticRanker  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    frames, names = C.load_pool()
    rows = {r["id"]: r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in C.LABELLED_SETS}
    rp = R.FundReplay()
    attrs = rp.svc.attrs
    goal = LogisticRanker.load(C.GOAL_MODEL)
    assert list(goal.feature_names) == names
    t0 = time.perf_counter()
    eq_goal = sum(C.select(goal, f, attrs, names)["answer"] == f.goal_answer for f in frames)
    ms_select = 1000 * (time.perf_counter() - t0) / len(frames)

    other = C.fit_ranker(frames, names, l2=0.01)
    other.temperature_ = 1.3  # любая: проверяется совпадение пути H5 / P1, а не качество
    orig = rp.svc.model
    rp.svc.model = other
    eq_other = 0
    t0 = time.perf_counter()
    diff_from_goal = 0
    for f in frames:
        a = C.select(other, f, attrs, names)["answer"]
        b = rp.resolve_row(rows[f.id])
        eq_other += a == b
        diff_from_goal += a != f.goal_answer
    ms_service = 1000 * (time.perf_counter() - t0) / len(frames)
    rp.svc.model = orig

    folds = C.group_folds(frames)
    split_groups = Counter()
    per = Counter()
    for f in frames:
        per[(folds[f.group], f.source)] += 1
    for g in {f.group for f in frames}:
        split_groups[len({folds[x.group] for x in frames if x.group == g})] += 1
    table = {
        k: {s: per[(k, s)] for s in C.SOURCES} for k in range(C.N_FOLDS)
    }
    out = {
        "frames": len(frames),
        "select_eq_product_goal": eq_goal,
        "select_eq_service_other_model": eq_other,
        "other_model_answers_differ_from_goal": diff_from_goal,
        "ms_per_frame_select": round(ms_select, 2),
        "ms_per_frame_service_resolve_plus_select": round(ms_service, 2),
        "folds_by_source": table,
        "groups": len({f.group for f in frames}),
        "groups_in_one_fold": dict(split_groups),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "check_select.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    assert eq_goal == len(frames) and eq_other == len(frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
