"""Сводка замера задачи B по папкам clean_latency/<cfg>: латентность скрипта организатора, обрывы,
точность на 20 кадрах (мягко: gt_slug ∪ acceptable из labels.jsonl; строго: gt_slug), VRAM, CPU, прогрев.

python summarize.py 4b [8b]
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
CL = Path(__file__).resolve().parent
FD = Path(r"<корень>\field_dataset")
labels = {d["id"]: d for d in (json.loads(x) for x in open(FD / "labels.jsonl", encoding="utf-8"))}
sel = {s["q"]: s for s in json.load(open(CL / "selection.json", encoding="utf-8"))}
SCAN = re.compile(r"^(\S+ \S+) INFO app\.api\.service: скан: slug=(\S+) outcome=(\S+) p=(\S+) total=(\d+) мс degraded=(\S+)")


def t(s: str) -> datetime:
    return datetime.strptime(s, "%H:%M:%S.%f")


def nearest_rank(v: list[float], q: float) -> float:
    s = sorted(v)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def summarize(cfg: str) -> dict:
    d = CL / cfg
    meta = json.load(open(d / "run_meta.json", encoding="utf-8"))
    preds = [json.loads(x) for x in open(d / "predictions.jsonl", encoding="utf-8")]
    lat = [p["latency_ms"] for p in preds]
    rows = []
    for p in preds:
        lab = labels[sel[p["query_id"]]["photo"]]
        ok_soft = p["predicted_slug"] in {lab["gt_slug"], *lab["acceptable"]}
        ok_strict = p["predicted_slug"] == lab["gt_slug"]
        rows.append(dict(q=p["query_id"], kind=sel[p["query_id"]]["kind"], lat=p["latency_ms"], slug=p["predicted_slug"],
                         soft=ok_soft, strict=ok_strict))
    # журнал сервиса: прогрев и сканы
    log = (d / "service.log").read_text(encoding="utf-8", errors="replace").splitlines()
    warm_i = next(i for i, l in enumerate(log) if "Прогрев:" in l)
    warm_scans = [SCAN.match(l).groups() for l in log[:warm_i] if SCAN.match(l)]
    warm_line = log[warm_i]
    after = [SCAN.match(l).groups() for l in log[warm_i:] if SCAN.match(l)]
    pt_scans = after[:len(preds)]
    for r, s in zip(rows, pt_scans):
        assert (s[1] if s[1] != "None" else None) == r["slug"], (r, s)
        r["svc_total"] = int(s[4]); r["degraded"] = s[5]; r["p"] = s[3]
    n_to = sum("vlm_timeout" in r["degraded"] for r in rows)
    n_cut = sum("vlm_budget_cut" in r["degraded"] for r in rows)
    n_bud = sum("budget_exceeded" in r["degraded"] for r in rows)
    h = json.load(open(d / "health_after_pt.json", encoding="utf-8"))
    # CPU и nvidia-smi в окне прогона
    m = meta["marks"]
    cpu_file = d / "cpu.tsv"
    if not cpu_file.exists() or "\t" not in cpu_file.read_text():
        cpu_file = d / "cpu_manual.tsv"  # 4b: логгер драйвера не стартовал, досэмплировано вручную с 12:53:46
    cpu = [l.split("\t") for l in cpu_file.read_text().splitlines() if "\t" in l]
    cpu_pt = [float(v) for ts, v in cpu if t(m["pt_start"]) <= t(ts) <= t(m["pt_end"])]
    cpu_pre = [float(v) for ts, v in cpu if t(ts) < t(m["svc_start"])]
    smi = []
    for l in (d / "smi.csv").read_text().splitlines():
        parts = [x.strip() for x in l.split(",")]
        if len(parts) < 4:
            continue
        ts = parts[0].split(" ")[1]
        smi.append((t(ts), int(parts[1].split()[0]), int(parts[2].split()[0]), parts[3]))
    smi_pt = [x for x in smi if t(m["pt_start"]) <= x[0] <= t(m["pt_end"])]
    smi_svc = [x for x in smi if t(m["svc_start"]) <= x[0] <= t(m["svc_stopped"])]
    smi_pre = [x for x in smi if x[0] < t(m["svc_start"])]
    scans = json.load(open(d / "scans.json", encoding="utf-8"))
    out = dict(
        cfg=cfg, model=meta["model"], cold_start_s=meta["cold_start_s"], port_open_s=meta["port_open_s"], pt_code=meta["pt_code"],
        pt_seconds=meta["pt_seconds"],
        n=len(preds), lat_sorted=sorted(lat), p50=float(np.percentile(lat, 50)), p95_linear=float(np.percentile(lat, 95)),
        p95_nearest=nearest_rank(lat, 0.95), max=max(lat), mean=round(float(np.mean(lat))), over3s=sum(x > 3000 for x in lat),
        over5s=sum(x > 5000 for x in lat), over8s=sum(x > 8000 for x in lat), over10s=sum(x >= 10000 for x in lat),
        null_slugs=sum(r["slug"] is None for r in rows),
        soft=sum(r["soft"] for r in rows), strict=sum(r["strict"] for r in rows),
        vlm_timeout_log=n_to, vlm_budget_cut_log=n_cut, budget_exceeded_log=n_bud,
        health_scans=h.get("scans"), health_status=h.get("status"), health_reasons=h.get("degraded_reasons"),
        warm_scans_before_ready=warm_scans, warm_line=warm_line[:900],
        cpu_pre_mean=round(float(np.mean(cpu_pre)), 1) if cpu_pre else None,
        cpu_pt_mean=round(float(np.mean(cpu_pt)), 1) if cpu_pt else None, cpu_pt_max=max(cpu_pt) if cpu_pt else None,
        cpu_pt_n=len(cpu_pt),
        smi_pre_max=max(x[1] for x in smi_pre) if smi_pre else None,
        smi_pt_min=min(x[1] for x in smi_pt) if smi_pt else None, smi_pt_max=max(x[1] for x in smi_pt) if smi_pt else None,
        smi_svc_max=max(x[1] for x in smi_svc) if smi_svc else None, smi_n_pt=len(smi_pt),
        ps_ready=[{k: x.get(k) for k in ("name", "size", "size_vram", "context_length")} for x in meta["ps_ready"].get("models", [])],
        scans=[{k: s[k] for k in ("q", "kind", "wall_ms", "slug", "degraded", "timings_ms", "p")} for s in scans],
        rows=rows,
    )
    return out


res = {cfg: summarize(cfg) for cfg in sys.argv[1:]}
for cfg, o in res.items():
    print(f"\n==== {cfg} ({o['model']})")
    print(f"холодный старт до ready {o['cold_start_s']} с (порт {o['port_open_s']} с); participant_test код {o['pt_code']}, {o['pt_seconds']} с на 20 кадров")
    print(f"latency_ms (скрипт организатора): p50 {o['p50']:.0f}, p95 {o['p95_linear']:.0f} (линейн.) / {o['p95_nearest']} (ближ. ранг), max {o['max']}, "
          f"среднее {o['mean']}; >3 с: {o['over3s']}, >5 с: {o['over5s']}, >8 с: {o['over8s']}, >=10 с: {o['over10s']}; null slug {o['null_slugs']}")
    print("  по возрастанию:", o["lat_sorted"])
    print(f"точность на {o['n']}: мягко {o['soft']}/{o['n']}, строго {o['strict']}/{o['n']}")
    print(f"обрывы по журналу: vlm_timeout {o['vlm_timeout_log']}, vlm_budget_cut {o['vlm_budget_cut_log']}, budget_exceeded {o['budget_exceeded_log']}")
    print(f"/v1/health после прогона: status {o['health_status']} {o['health_reasons']} scans {json.dumps(o['health_scans'], ensure_ascii=False)}")
    print(f"CPU: до старта сервиса среднее {o['cpu_pre_mean']} %; в окне прогона среднее {o['cpu_pt_mean']} %, max {o['cpu_pt_max']} % ({o['cpu_pt_n']} отсчётов)")
    print(f"nvidia-smi: до старта max {o['smi_pre_max']} МиБ; в окне прогона {o['smi_pt_min']}–{o['smi_pt_max']} МиБ ({o['smi_n_pt']} отсчётов); "
          f"max за жизнь сервиса {o['smi_svc_max']} МиБ")
    print("/api/ps после ready:", o["ps_ready"])
    print("прогревочные сканы до ready:", o["warm_scans_before_ready"])
    print("строка прогрева:", o["warm_line"])
    print("покадрово: q | вид | latency | total сервиса | degraded | мягко/строго")
    for r in o["rows"]:
        print(f"  {r['q']:12s} {r['kind']:11s} {r['lat']:5d} {r.get('svc_total', ''):>5} {r.get('degraded', ''):28s} {int(r['soft'])}/{int(r['strict'])}  p={r.get('p')}")
    print("/v1/scan после прогона (разбор стадий):")
    for s in o["scans"]:
        print(f"  {s['q']:12s} {s['kind']:17s} wall {s['wall_ms']:5d} degraded={s['degraded']} timings={s['timings_ms']}")
if len(res) == 2:
    a, b = (res[k] for k in sys.argv[1:3])
    ra = {r["q"]: r for r in a["rows"]}; rb = {r["q"]: r for r in b["rows"]}
    diff = [rb[q]["lat"] - ra[q]["lat"] for q in ra]
    print(f"\n{sys.argv[2]} минус {sys.argv[1]} по кадрам: медиана {np.median(diff):.0f} мс, min {min(diff)}, max {max(diff)}; "
          f"{sys.argv[2]} медленнее на {sum(x > 0 for x in diff)}/{len(diff)}")
    print("ответы различаются:", [(q, ra[q]["slug"], rb[q]["slug"], ra[q]["soft"], rb[q]["soft"]) for q in ra if ra[q]["slug"] != rb[q]["slug"]])
json.dump(res, open(CL / ("summary_" + "_".join(sys.argv[1:]) + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
