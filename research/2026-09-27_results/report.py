"""Отчёт по прогону на публичных данных: top-1, top-5, F1@1/F1@5, время, отрыв, срезы.

Входы — выход `scripts/run_eval.sh` (скрипт организатора: `predictions.jsonl`, `judge.json`) и
полные ответы `collect_bodies.py` (`bodies.jsonl`), эталон — `data/gt/*_gt.tsv`.

    python research/2026-09-27_results/report.py

Определения (одна правильная карточка на кадр):
- top-1 — slug ответа совпал; top-5 — правильная карточка есть среди пяти `top5` ответа;
- «строго» — только slug эталона; «то же вино» — ещё и карточки того же вина из колонки
  `acceptable` эталона (другой год урожая или дубль карточки);
- F1@k = 2·P·R/(P+R), где P@k — доля попаданий среди данных ответов, R@k — доля попаданий среди
  кадров с вином из каталога. Сервис отвечает всегда (null — 0), поэтому на кадрах из каталога
  P = R и F1@k равна доле попаданий. На всех 100 кадрах ответы на вина вне каталога — промахи P;
- интервал — 95 % бутстрэп по винам (кадры одного вина берутся вместе), 10 000 повторов.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SETS = {
    "real100": ROOT / "data/gt/real_photos_gt.tsv",
    "public": ROOT / "data/gt/public_gt.tsv",
}


def jl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def gt_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out = {}
    for r in rows:
        acc = [s for s in (r.get("acceptable") or "").split("|") if s]
        out[r["query_id"]] = {"slug": r["slug"], "acceptable": acc, "scene": r.get("scene") or ""}
    return out


def boot_ci(hits: list[bool], groups: list[str], reps: int = 10_000, seed: int = 26) -> list[float]:
    by: dict[str, list[bool]] = defaultdict(list)
    for h, g in zip(hits, groups):
        by[g].append(h)
    keys = sorted(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        pick = [by[rng.choice(keys)] for _ in keys]
        n = sum(len(p) for p in pick)
        vals.append(100 * sum(sum(p) for p in pick) / n)
    vals.sort()
    return [round(vals[int(0.025 * reps)], 1), round(vals[int(0.975 * reps) - 1], 1)]


def pct(a: int, b: int) -> float:
    return round(100 * a / b, 1) if b else 0.0


def latency(preds: list[dict]) -> dict:
    ms = sorted(float(p["latency_ms"]) for p in preds)
    rank = lambda q: ms[max(0, min(len(ms) - 1, int(-(-q * len(ms) // 1)) - 1))]  # ближайший ранг
    return {
        "n": len(ms),
        "p50_ms": rank(0.5),
        "p95_ms": rank(0.95),
        "max_ms": ms[-1],
        "over_3000": sum(m > 3000 for m in ms),
        "over_10000": sum(m > 10000 for m in ms),
    }


def winery_sizes() -> Counter:
    sizes: Counter = Counter()
    for w in jl(ROOT / "data/catalog/wines.jsonl"):
        if w.get("in_csv", True):
            sizes[w.get("winery_norm") or ""] += 1
    return sizes


def winery_of() -> dict[str, str]:
    return {w["slug"]: w.get("winery_norm") or "" for w in jl(ROOT / "data/catalog/wines.jsonl")}


def evaluate(name: str) -> dict:
    gt = gt_rows(SETS[name])
    preds = {p["query_id"]: p for p in jl(HERE / name / "predictions.jsonl")}
    bodies = {b["query_id"]: b for b in jl(HERE / name / "bodies.jsonl")}
    inside = [q for q, g in gt.items() if g["slug"] != "__none__"]
    res: dict = {
        "frames": len(gt),
        "in_catalog": len(inside),
        "out_of_catalog": len(gt) - len(inside),
        "null": sum(preds[q].get("predicted_slug") in (None, "") for q in gt),
        "latency_script": latency(list(preds.values())),
    }
    stable = sum(preds[q]["predicted_slug"] == (bodies[q]["body"] or {}).get("slug") for q in gt)
    res["same_answer_second_pass"] = f"{stable} из {len(gt)}"
    if not inside:
        return res
    groups = [gt[q]["slug"] for q in inside]
    for mode in ("same_wine", "strict"):
        ok = {
            q: {gt[q]["slug"], *(gt[q]["acceptable"] if mode == "same_wine" else [])}
            for q in inside
        }
        h1 = [preds[q]["predicted_slug"] in ok[q] for q in inside]
        h5 = [bool({t["slug"] for t in bodies[q]["body"]["top5"]} & ok[q]) for q in inside]
        answered_all = sum(preds[q].get("predicted_slug") not in (None, "") for q in gt)
        block = {}
        for k, hits in (("1", h1), ("5", h5)):
            t = sum(hits)
            p_in, r_in = t / len(inside), t / len(inside)
            p_all = t / answered_all if answered_all else 0.0
            f_all = 2 * p_all * r_in / (p_all + r_in) if p_all + r_in else 0.0
            block[f"top{k}"] = {
                "hits": t,
                "n": len(inside),
                "pct": pct(t, len(inside)),
                "ci95": boot_ci(hits, groups),
                f"F1@{k}_in_catalog": round(100 * 2 * p_in * r_in / (p_in + r_in), 1) if t else 0.0,
                f"F1@{k}_all_frames": round(100 * f_all, 1),
            }
        block["misses_top1"] = [q for q, h in zip(inside, h1) if not h]
        res[mode] = block
    # отрыв 1-го от 2-го и уверенность
    ok_sw = {q: {gt[q]["slug"], *gt[q]["acceptable"]} for q in inside}
    marg = [(bodies[q]["body"]["margin"], preds[q]["predicted_slug"] in ok_sw[q]) for q in inside]
    big = [ok for m, ok in marg if m is not None and m >= 0.5]
    res["margin"] = {
        "ge_0_5": f"{len(big)} из {len(inside)}",
        "ge_0_5_correct": f"{sum(big)} из {len(big)}",
        "median_correct": round(statistics.median(m for m, ok in marg if ok), 3),
        "median_wrong": round(statistics.median(m for m, ok in marg if not ok), 3)
        if any(not ok for _, ok in marg)
        else None,
    }
    conf = [bodies[q]["body"]["confidence"]["top1"] for q in inside]
    res["mean_confidence_top1"] = round(100 * statistics.mean(conf), 1)
    # срезы: сцена съёмки и «двойники» (винодельни с 6+ винами в выгрузке)
    scenes: dict[str, list[bool]] = defaultdict(list)
    for q in inside:
        scenes[gt[q]["scene"] or "—"].append(preds[q]["predicted_slug"] in ok_sw[q])
    res["by_scene_same_wine"] = {s: f"{sum(v)} из {len(v)}" for s, v in sorted(scenes.items())}
    sizes, wof = winery_sizes(), winery_of()

    def winery(q: str) -> str:
        # Эталон бывает карточкой живого портала вне выгрузки: винодельню берём по первой карточке
        # «того же вина», которая в выгрузке есть.
        return next((wof[s] for s in (gt[q]["slug"], *gt[q]["acceptable"]) if s in wof), "")

    big_w = [q for q in inside if sizes[winery(q)] >= 6]
    res["wineries_6plus"] = {
        "same_wine": f"{sum(preds[q]['predicted_slug'] in ok_sw[q] for q in big_w)} из {len(big_w)}",
        "strict": f"{sum(preds[q]['predicted_slug'] == gt[q]['slug'] for q in big_w)} из {len(big_w)}",
    }
    # Кадры, у которых вино есть в выгрузке 07.09 (эталон или карточка «того же вина» в индексе).
    # У остальных эталон — карточка живого портала, и верный ответ по индексу невозможен.
    in_export = [q for q in inside if any(s in wof for s in (gt[q]["slug"], *gt[q]["acceptable"]))]
    res["in_export"] = {
        "n": len(in_export),
        "not_in_export": [q for q in inside if q not in in_export],
    }
    if in_export:
        g_ex = [gt[q]["slug"] for q in in_export]
        for mode in ("same_wine", "strict"):
            hits = [
                preds[q]["predicted_slug"]
                in (
                    {gt[q]["slug"], *gt[q]["acceptable"]}
                    if mode == "same_wine"
                    else {gt[q]["slug"]}
                )
                for q in in_export
            ]
            res["in_export"][mode] = {
                "top1": f"{sum(hits)} из {len(in_export)}",
                "pct": pct(sum(hits), len(in_export)),
                "ci95": boot_ci(hits, g_ex),
            }
    return res


def main() -> None:
    out = {name: evaluate(name) for name in SETS}
    (HERE / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
