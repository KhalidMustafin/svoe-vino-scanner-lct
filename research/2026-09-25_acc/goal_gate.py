"""Совместные ворота итоговой сборки против `acc-freeze` @e266d2a — свой счёт, только CPU.

Входы — выходы `e3_measure.py measure` двух снимков кода на одних данных (CPU-векторы запросов,
чтения из кэша стенда, разбор и resolve — код снимка):

    # база: git archive e266d2a -> <снимок>; e3_measure из снимка (git show — из рабочего дерева)
    python research/2026-09-25_acc/e3_measure.py measure --set catalog_v2    --out base_v2.json
    python research/2026-09-25_acc/e3_measure.py measure --set krasnostop_v1 --out base_kr.json
    # итог: то же из рабочего дерева сборки -> final_v2.json, final_kr.json
    python research/2026-09-25_acc/goal_gate.py --dir <каталог с четырьмя файлами> --out goal_gate.json

Принятое правило: на v2 (353) 0 мягких поломок в каждом срезе и строго не ниже; на kr
(617 основных, без `same_packshot`) нетто мягко >= 0 в обоих счётах — как есть и без водяного
знака. Это ворота регрессии, а не замер прироста: параметров здесь нет.

Счёт: «то же вино» — ответ в {slug, acceptable} меток; строго — ответ = slug; макро — среднее по
винам `wine_groups_final.json` (226 вин v2), 95 % интервал — бутстрэп по винам (2000, seed 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

FD = Path(r"<корень>\field_dataset")


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def boot_ci(per: list[float], seed: int = 0, n: int = 2000) -> list[float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(per, dtype=float)
    bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)]
    return [
        round(100 * float(np.percentile(bs, 2.5)), 1),
        round(100 * float(np.percentile(bs, 97.5)), 1),
    ]


def v2_metrics(
    rows: dict[str, Any], key: str, meta: dict[str, Any], wine: dict[str, str]
) -> dict[str, Any]:
    soft = {q: rows[q][key] in {meta[q]["slug"], *(meta[q].get("acceptable") or [])} for q in rows}
    strict = {q: rows[q][key] == meta[q]["slug"] for q in rows}
    n_s, s_ok, st_ok = Counter(), Counter(), Counter()
    by_soft: dict[str, list[float]] = defaultdict(list)
    by_strict: dict[str, list[float]] = defaultdict(list)
    for q in rows:
        s = meta[q]["photo"][0]
        n_s[s] += 1
        s_ok[s] += soft[q]
        st_ok[s] += strict[q]
        w = wine.get(meta[q]["slug"], meta[q]["slug"])
        by_soft[w].append(float(soft[q]))
        by_strict[w].append(float(strict[q]))
    per_soft = [float(np.mean(v)) for v in by_soft.values()]
    n = len(rows)
    return {
        "frames": n,
        "wines": len(by_soft),
        "soft": sum(soft.values()),
        "soft_micro": round(100 * sum(soft.values()) / n, 2),
        "soft_macro": round(100 * float(np.mean(per_soft)), 2),
        "soft_macro_ci95": boot_ci(per_soft),
        "strict": sum(strict.values()),
        "strict_micro": round(100 * sum(strict.values()) / n, 2),
        "strict_macro": round(100 * float(np.mean([np.mean(v) for v in by_strict.values()])), 2),
        "slices_soft": {k: f"{s_ok[k]}/{n_s[k]}" for k in sorted(n_s)},
        "slices_strict": {k: f"{st_ok[k]}/{n_s[k]}" for k in sorted(n_s)},
        "_soft": soft,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def load(name: str) -> dict[str, Any]:
        return json.loads((args.dir / name).read_text(encoding="utf-8"))

    groups = json.loads((FD / "catalog" / "wine_groups_final.json").read_text(encoding="utf-8"))
    wine = {slug: wid for wid, g in groups.items() for slug in g.get("members") or []}
    out: dict[str, Any] = {}

    # v2
    meta = {m["query_id"]: m for m in jsonl(FD / "sets" / "catalog_v2" / "meta.jsonl")}
    bv, fv = load("base_v2.json"), load("final_v2.json")
    b, f = bv["rows"], fv["rows"]
    assert set(b) == set(f) == set(meta)
    v2: dict[str, Any] = {}
    for key, tag in (("off", "flag_off"), ("on", "flag_on_reference")):
        mb, mf = v2_metrics(b, key, meta, wine), v2_metrics(f, key, meta, wine)
        fixes = sorted(q for q in b if not mb["_soft"][q] and mf["_soft"][q])
        breaks = sorted(q for q in b if mb["_soft"][q] and not mf["_soft"][q])
        v2[tag] = {
            "base": {k: v for k, v in mb.items() if not k.startswith("_")},
            "final": {k: v for k, v in mf.items() if not k.startswith("_")},
            "answers_changed": sorted(q for q in b if b[q][key] != f[q][key]),
            "breaks_by_slice": {s: [q for q in breaks if meta[q]["photo"][0] == s] for s in "RML"},
            "fixes_by_slice": {s: [q for q in fixes if meta[q]["photo"][0] == s] for s in "RML"},
            "strict_not_lower": mf["strict"] >= mb["strict"],
        }
    off = v2["flag_off"]
    v2["pass"] = not any(off["breaks_by_slice"].values()) and off["strict_not_lower"]
    out["catalog_v2"] = v2

    # kr
    kmeta = {m["query_id"]: m for m in jsonl(FD / "sets" / "krasnostop_v1" / "meta.jsonl")}
    kb, kf = load("base_kr.json")["rows"], load("final_kr.json")["rows"]
    assert set(kb) == set(kf)
    main_q = sorted(q for q in kb if not kmeta[q].get("same_packshot"))
    rest = sorted(q for q in kb if kmeta[q].get("same_packshot"))

    def ok(q: str, ans: str) -> bool:
        return ans in {kmeta[q]["slug"], *(kmeta[q].get("acceptable") or [])}

    kr: dict[str, Any] = {"main_frames": len(main_q), "same_packshot_frames": len(rest)}
    counts = (
        ("as_is", "off"),
        ("no_watermark", "wm_off"),
        ("flag_on_as_is", "on"),
        ("flag_on_no_watermark", "wm_on"),
    )
    for tag, key in counts:
        fixes = [q for q in main_q if not ok(q, kb[q][key]) and ok(q, kf[q][key])]
        breaks = [q for q in main_q if ok(q, kb[q][key]) and not ok(q, kf[q][key])]
        base_ok = sum(ok(q, kb[q][key]) for q in main_q)
        fin_ok = sum(ok(q, kf[q][key]) for q in main_q)
        kr[tag] = {
            "base": base_ok,
            "final": fin_ok,
            "final_micro": round(100 * fin_ok / len(main_q), 2),
            "strict_base": sum(kb[q][key] == kmeta[q]["slug"] for q in main_q),
            "strict_final": sum(kf[q][key] == kmeta[q]["slug"] for q in main_q),
            "answers_changed": [q for q in main_q if kb[q][key] != kf[q][key]],
            "fixes": fixes,
            "breaks": breaks,
            "net": len(fixes) - len(breaks),
            "same_packshot_changed": [q for q in rest if kb[q][key] != kf[q][key]],
        }
    kr["K0225_correct"] = {
        "base": ok("K0225-orig-0", kb["K0225-orig-0"]["off"]),
        "final": ok("K0225-orig-0", kf["K0225-orig-0"]["off"]),
    }
    kr["pass"] = kr["as_is"]["net"] >= 0 and kr["no_watermark"]["net"] >= 0
    out["krasnostop_v1"] = kr

    out["e3_code_gate"] = {
        name: {
            "gate_failures": load(name)["gate"],
            "frames": load(name)["gate_frames"],
            "failed": load(name)["failed"],
            "reads": load(name)["reads"],
        }
        for name in ("base_v2.json", "final_v2.json", "base_kr.json", "final_kr.json")
    }
    out["files_sha1"] = fv["files"]["sha1"]
    out["provenance_flag_on"] = load("final_kr.json").get("provenance_on")
    out["pass"] = v2["pass"] and kr["pass"]
    return_out = {"what": "совместные ворота итоговой сборки против acc-freeze @e266d2a", **out}
    args.out.write_text(
        json.dumps(return_out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: out[k]["pass"] for k in ("catalog_v2", "krasnostop_v1")} | {"pass": out["pass"]},
            ensure_ascii=False,
        )
    )
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
