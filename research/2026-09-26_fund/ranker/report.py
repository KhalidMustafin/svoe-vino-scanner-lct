"""Сводка трека «ранкер» из выходов `oof.py`, `errors.py`, `train_final.py` — один JSON.

Все числа — вне фолда (OOF) на пуле разработки (1 533 кадра с меткой, kr-test нет). Срезы:
`field_main` = v2 (353) + kr-dev (308); `all` = всё с `studio`, «телефоном» и same_packshot.

    PYTHONPATH=. python research/2026-09-26_fund/ranker/report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

KEYS = ("studio_all", "studio_goaltest", "v2", "v2_R", "v2_M", "kr_dev", "kr_dev_sp", "field_main", "all")


def load(tag: str) -> dict | None:
    p = C.OUT / f"{tag}_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def row(t: dict, ci: dict | None = None) -> dict:
    out = {k: f"{t[k]['ok']}/{t[k]['n']} ({t[k]['pct']:.1f} %)" for k in KEYS if k in t}
    if ci:
        out["ci95"] = ci
    return out


def seeds(s: dict, v: str) -> dict:
    reps = s.get("seed_repeats", {}).get(v)
    if not reps:
        return {}
    out = {}
    for k in ("studio_all", "v2", "v2_R", "kr_dev", "field_main", "all"):
        vals = [s["tallies"][v][k]["ok"]] + [r["service"][k]["ok"] for r in reps]
        out[k] = {"mean": round(sum(vals) / len(vals), 1), "min": min(vals), "max": max(vals), "n_seeds": len(vals)}
    return out


def paired(s: dict, name: str) -> dict:
    d = s["paired"].get(name, {})
    return {k: {"a": d[k]["a"], "b": d[k]["b"], "fix/break": f"+{d[k]['fixes']}/-{d[k]['breaks']}",
                "sign_p": d[k]["sign_p"], "diff_pp": d[k]["diff_pp"], "diff_ci95_pp": d[k]["diff_ci95"]}
            for k in KEYS if k in d}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    rep: dict = {}
    oof = load("oof")
    if oof:
        T, CI = oof["tallies"], oof["ci95"]
        rep["oof_main"] = {v: row(T[v], CI.get(v)) for v in T}
        rep["oof_main_paired"] = {c: paired(oof, c) for c in (
            "pool_vs_goal", "studio_refit_vs_goal", "pool_vs_studio_refit", "field_only_vs_goal",
            "pool_unsigned_vs_goal", "pool_ranker_only_vs_pool")}
        rep["oof_main_seeds"] = {v: seeds(oof, v) for v in oof.get("seed_repeats", {})}
        rep["oof_main_l2"] = {v: {k: f["l2"] for k, f in fo.items()} for v, fo in oof["folds"].items()}
        rep["paths"] = oof["paths"]
    abl = load("ablation")
    if abl:
        base = abl["tallies"]["pool"]
        rep["ablation_delta_vs_pool"] = {
            v: {k: abl["tallies"][v][k]["ok"] - base[k]["ok"] for k in ("v2", "kr_dev", "field_main", "all")}
            for v in abl["tallies"] if v.startswith("drop_") and not v.endswith("_ranker_only")
        }
    cur = load("curve")
    if cur:
        rep["learning_curve"] = {
            v: {"seed0": {k: cur["tallies"][v][k]["ok"] for k in ("field_main", "kr_dev", "all")}, "seeds": seeds(cur, v)}
            for v in ("studio_refit", "pool_f25", "pool_f50", "pool_f75", "pool")
        }
    for tag in ("sift1", "siftall"):
        s = load(tag)
        if s:
            rep[tag] = {
                "extra": s["meta"]["extra"], "extra_sha1": s["meta"].get("extra_sha1"),
                "tallies": {v: row(s["tallies"][v]) for v in ("goal", "pool", "pool_extra", "pool_extra_ranker_only")},
                "paired_vs_pool": paired(s, "pool_extra_vs_pool"),
                "paired_vs_goal": paired(s, "pool_extra_vs_goal"),
                "seeds": {"pool_extra": seeds(s, "pool_extra")},
            }
    for tag in ("oof", "siftall"):
        p = C.OUT / f"{tag}_errors.json"
        if p.exists():
            e = json.loads(p.read_text(encoding="utf-8"))
            rep[f"errors_{tag}"] = {v: {"all": e[v]["all"], "field_main": e[v]["field_main"]} for v in e}
    for v in ("pool", "pool_extra"):
        p = C.OUT / f"final_{v}.json"
        if p.exists():
            f = json.loads(p.read_text(encoding="utf-8"))
            rep[f"final_{v}"] = {k: f[k] for k in ("model", "sha1", "file_bytes", "l2", "temperature",
                                                    "service_equal_select", "latency", "fit_seconds_nested")}
            rep[f"final_{v}"]["weights_top"] = f["weights"][:16]
            rep[f"final_{v}"]["weights_all"] = f["weights"]
    out = C.OUT / "ranker_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
