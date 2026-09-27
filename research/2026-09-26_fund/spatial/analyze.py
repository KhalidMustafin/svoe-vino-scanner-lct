"""Насколько inliers отделяют верную карточку от соседей винодельни и от чужих — по пулу.

    PYTHONPATH=. python research/2026-09-26_fund/spatial/analyze.py

Только пул разработки (без kr-test). Кадр учитывается, если верный slug (или допустимый)
есть в CV top-20. Кандидаты кадра делятся на:
- `correct` — slug кадра или допустимый («то же вино»);
- `twin` — не верный, но в той же `visual_group`, что верный (та же этикетка, другой год или объём);
- `sibling` — та же винодельня (`WineAttrs.winery`), не двойник;
- `other` — другая винодельня.

Для каждого набора: доля кадров, где лучший верный набрал inliers строго больше лучшего
кандидата класса (и ничьи), медианы, «SIFT top-1» (argmax inliers, ничья — лучший ранг CV)
в top-5 и top-20 против CV top-1, и что SIFT говорит на кадрах, где продукт ошибся или прав.
Отдельно — вне каталога (`ooc_v2`): максимум inliers против кадров каталога и против счёта и
отрыва CV top-1 (что SIFT добавил бы к отказу «нет в каталоге»).
Выход — `spatial/separation.json` и таблица в stdout.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import protocol as P
import table as T

OUT = P.FUND / "spatial"
SETS = ["catalog_v2", "kr_dev", "kr_dev_sp", "pairs", "pairs_phone"]
#: Что сравнивается: inliers (лучший вид, `full`, `label`), различимые inliers внутри top-20,
#: inliers на точку эталона.
VIEWS = {"best": "inl", "full": "sift_inl_full_log", "label": "sift_inl_label_log", "uniq": "uniq", "norm": "sift_inl_norm"}


def winery_key(attrs: Any, slug: str) -> str | None:
    w = attrs.get(slug)
    return (w.winery or "").strip().lower() or None if w is not None else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    from app.resolve.attrs import CatalogAttrs

    attrs = CatalogAttrs.load(P.FROZEN / "gt" / "gt_tokens.jsonl")
    rows = P.jsonl(P.PROTOCOL / "trainpool.jsonl")
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    sift = T.load_sift()
    P.assert_no_test(sift.keys())
    report: dict[str, Any] = {}
    ref_kp = T.ref_keypoints()
    for view in VIEWS:
        key = VIEWS[view]
        per_set: dict[str, Any] = {}
        for st in [*SETS, "v2+kr_dev", "all"]:
            members = SETS if st == "all" else (["catalog_v2", "kr_dev"] if st == "v2+kr_dev" else [st])
            stats: dict[str, list[Any]] = defaultdict(list)
            n = 0
            for r in rows:
                if r["set"] not in members or r["id"] not in sift or not r["true_rank_cv"]:
                    continue
                n += 1
                feats = T.cand_features(sift[r["id"]]["cands"], ref_kp)
                val = [f[key] if key in ("inl", "uniq", "sift_inl_norm") else float(np.expm1(f[key])) for f in feats]
                slugs = [c[0] for c in r["cv"]]
                ok = {r["slug"], *(r.get("acceptable") or [])}
                true_w = winery_key(attrs, r["slug"])
                vgroups = {a.visual_group for s in ok if (a := attrs.get(s)) is not None and a.visual_group}
                cls = []
                for s in slugs:
                    a = attrs.get(s)
                    if s in ok:
                        cls.append("correct")
                    elif a is not None and a.visual_group and a.visual_group in vgroups:
                        cls.append("twin")
                    elif true_w and winery_key(attrs, s) == true_w:
                        cls.append("sibling")
                    else:
                        cls.append("other")
                best_c = max(v for v, c in zip(val, cls, strict=True) if c == "correct")
                stats["correct_inl"].append(best_c)
                for k in ("twin", "sibling", "other"):
                    vs = [v for v, c in zip(val, cls, strict=True) if c == k]
                    if vs:
                        m = max(vs)
                        stats[f"{k}_max"].append(m)
                        stats[f"{k}_win"].append(best_c > m)
                        stats[f"{k}_tie"].append(best_c == m)
                    # в top-5
                    vs5 = [v for v, c in zip(val[:5], cls[:5], strict=True) if c == k]
                    if vs5 and "correct" in cls[:5]:
                        m5 = max(vs5)
                        b5 = max(v for v, c in zip(val[:5], cls[:5], strict=True) if c == "correct")
                        stats[f"{k}_win5"].append(b5 > m5)
                for K in (5, 20):
                    vv = val[:K]
                    top = max(vv)
                    arg = vv.index(top)  # ничья — лучший ранг CV
                    stats[f"sift_top1_{K}"].append(cls[arg] == "correct")
                    stats[f"sift_top1_{K}_zero"].append(top == 0)
                stats["cv_top1"].append(cls[0] == "correct")
                stats["prod"].append(bool(r["correct"]))
                arg20 = val.index(max(val))
                if r["correct"]:
                    stats["prod_ok_sift_other"].append(cls[arg20] != "correct" and max(val) > 0)
                else:
                    stats["prod_bad_sift_ok"].append(cls[arg20] == "correct" and max(val) > 0)
            res: dict[str, Any] = {"frames": n}
            res["correct_inl_median"] = float(np.median(stats["correct_inl"])) if n else None
            res["correct_inl_zero"] = float(np.mean(np.asarray(stats["correct_inl"]) == 0)) if n else None
            for k in ("twin", "sibling", "other"):
                wins = stats[f"{k}_win"]
                res[k] = {
                    "frames_with": len(wins),
                    "correct_gt_max": round(float(np.mean(wins)), 4) if wins else None,
                    "tie": round(float(np.mean(stats[f"{k}_tie"])), 4) if wins else None,
                    "max_median": float(np.median(stats[f"{k}_max"])) if wins else None,
                    "correct_gt_max_top5": round(float(np.mean(stats[f"{k}_win5"])), 4) if stats[f"{k}_win5"] else None,
                    "frames_with_top5": len(stats[f"{k}_win5"]),
                }
            for K in (5, 20):
                res[f"sift_top1_{K}"] = round(float(np.mean(stats[f"sift_top1_{K}"])), 4)
                res[f"sift_top1_{K}_allzero"] = round(float(np.mean(stats[f"sift_top1_{K}_zero"])), 4)
            res["cv_top1"] = round(float(np.mean(stats["cv_top1"])), 4)
            res["product"] = round(float(np.mean(stats["prod"])), 4)
            res["product_wrong_sift_top1_correct"] = f"{sum(stats['prod_bad_sift_ok'])}/{len(stats['prod_bad_sift_ok'])}"
            res["product_right_sift_top1_wrong"] = f"{sum(stats['prod_ok_sift_other'])}/{len(stats['prod_ok_sift_other'])}"
            per_set[st] = res
        report[view] = per_set
    # вне каталога: максимум inliers по top-20 (и top-5) против счёта и отрыва CV top-1
    def auc(a: np.ndarray, b: np.ndarray) -> float:
        return round(float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()), 4)

    sig: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["id"] not in sift:
            continue
        feats = T.cand_features(sift[r["id"]]["cands"], ref_kp)
        grp = "ooc_v2" if r["set"] == "ooc_v2" else ("catalog_field" if r["set"] in ("catalog_v2", "kr_dev") else None)
        if grp:
            sig[grp]["max_inl20"].append(max(f["inl"] for f in feats))
            sig[grp]["max_inl5"].append(max(f["inl"] for f in feats[:5]))
            sig[grp]["inl_top1"].append(feats[0]["inl"])
            sig[grp]["cv_top1_score"].append(r["cv"][0][1])
            sig[grp]["cv_margin"].append(r["margin"])
    report["ooc"] = {
        g: {k: {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))} for k, v in d.items()} | {"n": len(d["max_inl20"])}
        for g, d in sig.items()
    }
    report["ooc"]["auc_catalog_vs_ooc"] = {
        k: auc(np.asarray(sig["catalog_field"][k]), np.asarray(sig["ooc_v2"][k])) for k in sig["catalog_field"]
    }
    (OUT / "separation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for view in VIEWS:
        print(f"\n== вид: {view}")
        print(f"{'набор':12s} {'кадр':>5s} {'медиана верн':>12s} {'верн=0':>7s} | {'>двойн':>7s} {'>сосед':>7s} {'ничья':>6s} {'>чужих':>7s} | {'SIFT@5':>7s} {'SIFT@20':>7s} {'CV@1':>6s} {'прод':>6s} | прод✗→SIFT✓  прод✓→SIFT✗")
        for st, res in report[view].items():
            print(
                f"{st:12s} {res['frames']:5d} {res['correct_inl_median']:12.0f} {res['correct_inl_zero']:7.3f} | "
                f"{res['twin']['correct_gt_max'] if res['twin']['correct_gt_max'] is not None else float('nan'):7.3f} "
                f"{res['sibling']['correct_gt_max'] if res['sibling']['correct_gt_max'] is not None else float('nan'):7.3f} {res['sibling']['tie'] or 0:6.3f} "
                f"{res['other']['correct_gt_max'] if res['other']['correct_gt_max'] is not None else float('nan'):7.3f} | "
                f"{res['sift_top1_5']:7.3f} {res['sift_top1_20']:7.3f} {res['cv_top1']:6.3f} {res['product']:6.3f} | "
                f"{res['product_wrong_sift_top1_correct']:>11s}  {res['product_right_sift_top1_wrong']:>11s}"
            )
    print("\nвне каталога, AUC «каталог против вне каталога»:", json.dumps(report["ooc"]["auc_catalog_vs_ooc"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
