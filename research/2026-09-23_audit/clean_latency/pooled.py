"""Задача B: сводка по двум прогонам каждой модели (порядок 4b, 8b, 8b_r2, 4b_r2), повторяемость по кадрам,
сверка ответов сервиса с дампами bigcat (8b) и bigcat4b_live (4b) под H5."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
CL = Path(__file__).resolve().parent
RUNS = Path(r"<корень>\svoe-vino-scanner\runs\field25\iters\runs")
FD = Path(r"<корень>\field_dataset")
labels = {d["id"]: d for d in (json.loads(x) for x in open(FD / "labels.jsonl", encoding="utf-8"))}
sel = {s["q"]: s for s in json.load(open(CL / "selection.json", encoding="utf-8"))}


def preds(cfg: str) -> dict:
    return {d["query_id"]: d for d in (json.loads(x) for x in open(CL / cfg / "predictions.jsonl", encoding="utf-8"))}


def nr(v, q):
    s = sorted(v)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def h5(run: str) -> dict:
    p = {d["query_id"]: d for d in (json.loads(x) for x in open(RUNS / run / "iter20/predictions.jsonl", encoding="utf-8"))}
    wi = {d["query_id"]: d for d in json.load(open(RUNS / run / "iter20/whatif.json", encoding="utf-8"))}
    out = {}
    for q in sel:
        d = p[q]
        out[q] = wi[q]["no_cluster+filter"][0] if (d.get("p_top1") is not None and d["p_top1"] < 0.5) else d["slug"]
    return out


def soft(q, slug):
    lab = labels[sel[q]["photo"]]
    return slug in {lab["gt_slug"], *lab["acceptable"]}


P = {c: preds(c) for c in ("4b", "4b_r2", "8b", "8b_r2")}
for m in ("4b", "8b"):
    a, b = P[m], P[m + "_r2"]
    lat = [a[q]["latency_ms"] for q in sel] + [b[q]["latency_ms"] for q in sel]
    d = [b[q]["latency_ms"] - a[q]["latency_ms"] for q in sel]
    same = sum(a[q]["predicted_slug"] == b[q]["predicted_slug"] for q in sel)
    print(f"{m}: 40 запросов (2 прогона × 20): p50 {np.percentile(lat, 50):.0f}, p95 {np.percentile(lat, 95):.0f} (линейн.) / {nr(lat, .95)} (ближ. ранг), "
          f"max {max(lat)}, >3 с {sum(x > 3000 for x in lat)}/40, >5 с {sum(x > 5000 for x in lat)}/40; "
          f"повтор минус первый по кадрам: медиана {np.median(d):.0f} мс, |разн.| max {max(abs(x) for x in d)}; одинаковый slug {same}/20")
    for c in (m, m + "_r2"):
        print(f"   {c}: мягко {sum(soft(q, P[c][q]['predicted_slug']) for q in sel)}/20")
d8, d4 = h5("bigcat"), h5("bigcat4b_live")
for m, dump, name in (("8b", d8, "bigcat (8b)"), ("4b", d4, "bigcat4b_live (4b)")):
    eq = [q for q in sel if P[m][q]["predicted_slug"] == dump[q]]
    print(f"сервис {m} = H5 дампа {name}: {len(eq)}/20; расхождения: {[(q, P[m][q]['predicted_slug'], dump[q]) for q in sel if q not in eq]}; "
          f"дамп мягко {sum(soft(q, dump[q]) for q in sel)}/20")
print("кадры с ошибкой (мягко):")
for q in sel:
    r = {c: soft(q, P[c][q]["predicted_slug"]) for c in P}
    if not all(r.values()):
        print(f"   {q} gt={labels[sel[q]['photo']]['gt_slug']} acc={labels[sel[q]['photo']]['acceptable']} -> "
              + "; ".join(f"{c}:{P[c][q]['predicted_slug']}({'ok' if r[c] else 'x'})" for c in P))
# по типам кадров
for kind in ("portal_webp", "heic", "jpg"):
    qs = [q for q in sel if sel[q]["kind"] == kind]
    print(kind, len(qs), {c: int(np.median([P[c][q]["latency_ms"] for q in qs])) for c in P})
