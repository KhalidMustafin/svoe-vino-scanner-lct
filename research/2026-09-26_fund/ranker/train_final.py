"""Итоговая модель трека «ранкер» на всём пуле, устойчивость весов и цена в сервисе (CPU).

1. `pool`-рецепт (`oof.VARIANTS`, по умолчанию `pool`) на всех 1 533 кадрах пула с меткой: L2 и
   температура — вложенным групповым 4-fold, как внутри OOF. Модель пишется в формате сервиса
   (`LogisticRanker.save`, `svs-resolve-logistic/1`) с метой и sha1 — её можно подать в
   `FundReplay(model_path=...)` и в сервис (`SVS_RESOLVE_MODEL`).
2. Веса: стандартизованные, рядом с `-goal`; 95 % интервал — бутстрэп по группам вин (200) при
   том же L2; доля повторов, где вес на границе знака (ровно 0).
3. Сервис: `FundReplay(model_path=модель)` отвечает на кадрах пула так же, как `common.select`
   (проверка переноса); время слоя выбора на кадр — у модели и у `-goal`; размер файла.

Ответы на пуле здесь — в выборке обучения и для оценки не годятся (оценка — `oof.py`).

    PYTHONPATH=. python research/2026-09-26_fund/ranker/train_final.py [--variant pool]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402
import oof as O  # noqa: E402
import protocol as P  # noqa: E402

from app.resolve.learned import LogisticRanker  # noqa: E402

N_BOOT = 200


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pool")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    O.register_extra_variants(args.extra)
    v = O.VARIANTS[args.variant]
    frames, names = C.load_pool()
    if args.extra:
        import extras

        extras.attach(frames, args.extra)
    attrs = C.load_attrs()
    goal = LogisticRanker.load(C.GOAL_MODEL)
    t0 = time.perf_counter()
    model, info = O.fit_nested(frames, v, names, attrs, seed=C.FOLD_SEED)
    fit_s = time.perf_counter() - t0
    pool = [f for f in frames if v.train(f)]
    model.meta = {
        **{k: goal.meta[k] for k in ("reader_keys", "readers", "top_k", "variant")},
        "feature_version": goal.meta["feature_version"],
        "gt_tokens_sha1": attrs.meta.get("sha1"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "track": "fund/ranker (26.09)",
        "recipe": f"{args.variant}: listwise, знаки {'да' if v.signed else 'нет'}, L2 и T — вложенный групповой 4-fold",
        "sets": sorted({f.set for f in pool}),
        "queries": len(pool),
        "l2_selection": info["grid"],
        "kr_test": "нет: страж protocol.assert_no_test на всех кадрах",
        "trainpool_sha1": P.sha1_file(P.PROTOCOL / "trainpool.jsonl"),
    }
    if args.extra:
        import extras

        model.meta["extra_tables_sha1"] = extras.table_sha1(args.extra)
        model.meta["extra_features"] = list(v.extra)
    out_dir = C.OUT / "candidate"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"s2so400m-vlm35-{args.variant}.json"
    model.save(path)
    sha1 = hashlib.sha1(path.read_bytes()).hexdigest()
    print(f"модель {path.name}: L2 {info['l2']}, T {info['T']}, sha1 {sha1}, обучение {fit_s:.1f} с", flush=True)

    # ---- веса и их устойчивость
    rng = np.random.default_rng(0)
    groups = sorted({f.group for f in pool})
    by_group: dict[str, list[C.Frame]] = {}
    for f in pool:
        by_group.setdefault(f.group, []).append(f)
    boots = []
    for b in range(args.boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        sample = []
        for n, gi in enumerate(draw):
            # копии кадров под своим id: listwise требует сплошные блоки запросов, не id
            sample.extend(by_group[groups[gi]])
        m = C.fit_ranker(sample, names, l2=info["l2"], signed=v.signed, extra=v.extra)
        boots.append(m.coef_.copy())
    W = np.asarray(boots)
    use = list(model.feature_names)
    goal_w = goal.weights()
    weights = []
    for j, n in enumerate(use):
        weights.append(
            {
                "feature": n,
                "w": round(float(model.coef_[j]), 4),
                "ci95": [round(float(np.percentile(W[:, j], 2.5)), 4), round(float(np.percentile(W[:, j], 97.5)), 4)],
                "zero_share": round(float(np.mean(np.abs(W[:, j]) < 1e-12)), 3),
                "goal_w": round(goal_w[n], 4) if n in goal_w else None,
                "raw_mean": round(float(model.mean_[j]), 5),
                "raw_scale": round(float(model.scale_[j]), 5),
            }
        )
    weights.sort(key=lambda r: -abs(r["w"]))

    # ---- перенос в сервис и цена
    import replay as R

    rows = {r["id"]: r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in C.LABELLED_SETS}
    eq = None
    lat: dict[str, float] = {}
    if not args.extra:
        rp_new = R.FundReplay(model_path=path)
        rp_goal = R.FundReplay()
        eq = 0
        for f in frames:
            eq += rp_new.resolve_row(rows[f.id]) == C.select(model, f, attrs, names)["answer"]
        for name, rp in (("goal", rp_goal), ("candidate", rp_new)):
            ts = []
            for f in frames:
                t = time.perf_counter()
                rp.resolve_row(rows[f.id])
                ts.append(1000 * (time.perf_counter() - t))
            lat[f"{name}_ms_median"] = round(statistics.median(ts), 3)
            lat[f"{name}_ms_p95"] = round(float(np.percentile(ts, 95)), 3)
        tracemalloc.start()
        LogisticRanker.load(path)
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        lat["model_load_peak_kib"] = round(peak / 1024, 1)
    report = {
        "variant": args.variant,
        "model": str(path),
        "sha1": sha1,
        "file_bytes": path.stat().st_size,
        "l2": info["l2"],
        "temperature": info["T"],
        "l2_selection": info["grid"],
        "fit_info": model.fit_info,
        "fit_seconds_nested": round(fit_s, 1),
        "service_equal_select": eq,
        "frames": len(frames),
        "latency": lat,
        "weights": weights,
        "n_boot": args.boot,
    }
    (C.OUT / f"final_{args.variant}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("l2", "temperature", "service_equal_select", "latency", "file_bytes")}, ensure_ascii=False))
    for r in weights:
        print(f"{r['feature']:28s} {r['w']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] zero {r['zero_share']:.2f}  goal {r['goal_w']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
