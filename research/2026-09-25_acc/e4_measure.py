"""Замер Э4 строго по `PREREG_E4_card_attrs.md` (§7–8) и дополнению `PREREG_E4_addendum.md`.

Только CPU и записанные кэши стенда: видеокарта и Ollama не нужны. Здесь правила только
исполняются; ответы кадров посчитаны стендом (`e4_stand.py`) и кодом снимков (`e4_redecide.py`).

Входы:
    runs/field25/iters/runs/{pfix_final,kr_holdout}/iter20 — дампы (ворота 0)
    runs/field25/iters/runs/e1/pkg_{v2,kr}                 — стенд принятого Э1 (ворота 1)
    runs/field25/iters/runs/e4/<прогон>_{v2,kr}             — стенд Э4 и `redecide.json`
    field_dataset/sets/{catalog_v2,krasnostop_v1}/meta.jsonl — эталон

Шаги (`python e4_measure.py <шаг>`):
    gate   — ворота 0 (снимок @7b76247 + bf6a88 = H5 дампа) и 1 (база после Э1 = стенд Э1 pkg)
    step1  — 13 единиц против базы; какие прошли (0 мягких поломок на v2 и kr в обоих счётах)
    report — единицы и пакет против базы; приёмка пакета; итог — в e4_results.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "2026-09-24_h6"))

import measure_h6 as mh

ROOT = Path(r"<корень>")
FD = ROOT / "field_dataset"
RUNS = mh.RUNS
E4 = RUNS / "e4"
E1 = RUNS / "e1"
SETS = {"v2": ("pfix_final", "catalog_v2", 353), "kr": ("kr_holdout", "krasnostop_v1", 903)}
SLICES = ("R", "M", "L")
BASE_SHA1 = "bf6a8823a45004b1e13e586e5a92e05b8beb07b0"
#: Кадр вина, правленного по slug товара krasnostop (строка 10) — отдельный отчёт, не в итоге kr.
KR_ASIDE = "K0225"
ROWS = [f"row{i:02d}" for i in range(1, 13)]
UNITS = ["compat", *ROWS]


# ------------------------------------------------------------------ входы
def meta_of(set_key: str) -> list[dict[str, Any]]:
    return mh.jsonl(FD / "sets" / SETS[set_key][1] / "meta.jsonl")


def stand(run: str, set_key: str, root: Path = E4) -> dict[str, dict[str, Any]]:
    return {r["query_id"]: r for r in mh.jsonl(root / f"{run}_{set_key}" / "predictions.jsonl")}


def run_info(run: str, set_key: str) -> dict[str, Any]:
    return json.loads((E4 / f"{run}_{set_key}" / "run_info.json").read_text(encoding="utf-8"))


def redecided(run: str, set_key: str) -> dict[str, Any]:
    path = E4 / f"{run}_{set_key}" / "redecide.json"
    return json.loads(path.read_text(encoding="utf-8"))


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def run_health(run: str, set_key: str) -> list[str]:
    """Прогон целый: все кадры, 0 промахов кэша, ключ CV дампов, решение снимка = стенд."""
    bad = []
    info = run_info(run, set_key)
    rows = stand(run, set_key)
    n = SETS[set_key][2]
    if len(rows) != n:
        bad.append(f"{run}_{set_key}: кадров {len(rows)} ≠ {n}")
    cache = info["cache"]
    if cache.get("cv_miss") or cache.get("read_miss") or cache.get("failed"):
        bad.append(f"{run}_{set_key}: кэш {cache}")
    if info["cv_key"] != "ae2c4db886d1b1e9":
        bad.append(f"{run}_{set_key}: ключ CV {info['cv_key']}")
    red = redecided(run, set_key) if (E4 / f"{run}_{set_key}" / "redecide.json").is_file() else None
    if run != "gate0":
        if red is None:
            bad.append(f"{run}_{set_key}: нет redecide.json")
        elif red["asis_mismatch"] or red["frames"] != n:
            bad.append(f"{run}_{set_key}: решение снимка ≠ стенд {red['asis_mismatch'][:5]}")
    return bad


# ------------------------------------------------------------------ ворота
def gate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_key, (dump_name, _, _) in SETS.items():
        frames = {f.qid: f for f in mh.load_run(dump_name)[0]}
        g0 = stand("gate0", set_key)
        both = sorted(set(g0) & set(frames))
        out[f"gate0_{set_key}"] = {
            "stand": len(g0),
            "dump": len(frames),
            "slug_eq_h5": sum(g0[q]["slug"] == frames[q].h5 for q in both),
            "fields_eq": sum(
                (g0[q].get("text_read") or {}).get("fields")
                == (frames[q].record.get("text_read") or {}).get("fields")
                for q in both
            ),
            "health": run_health("gate0", set_key),
        }
        base, e1 = stand("base", set_key), stand("pkg", set_key, E1)
        both = sorted(set(base) & set(e1))
        out[f"gate1_{set_key}"] = {
            "base": len(base),
            "e1_pkg": len(e1),
            "slug_eq": sum(base[q]["slug"] == e1[q]["slug"] for q in both),
            "text_read_eq": sum(base[q].get("text_read") == e1[q].get("text_read") for q in both),
            "resolve_eq": sum(base[q].get("resolve") == e1[q].get("resolve") for q in both),
            "lexicon_eq_e1": sha1(E4 / f"base_{set_key}" / "lexicon.json")
            == sha1(E1 / f"pkg_{set_key}" / "lexicon.json"),
            "health": run_health("base", set_key),
        }
    out["nofix_sha1"] = sha1(E4 / "gt" / "nofix" / "gt_tokens.jsonl")
    out["nofix_is_bf6a88"] = out["nofix_sha1"] == BASE_SHA1
    for key, val in out.items():
        print(key, json.dumps(val, ensure_ascii=False))
    return out


def gate_ok(g: Mapping[str, Any]) -> bool:
    ok = g["nofix_is_bf6a88"]
    for set_key, (_, _, n) in SETS.items():
        g0, g1 = g[f"gate0_{set_key}"], g[f"gate1_{set_key}"]
        ok &= g0["stand"] == g0["dump"] == g0["slug_eq_h5"] == g0["fields_eq"] == n
        ok &= not g0["health"] and not g1["health"] and g1["lexicon_eq_e1"]
        ok &= g1["base"] == g1["e1_pkg"] == g1["slug_eq"] == g1["text_read_eq"] == n
        ok &= g1["resolve_eq"] == n
    return bool(ok)


# ------------------------------------------------------------------ счёт
def score(rows: Sequence[Mapping[str, Any]], answers: Mapping[str, str]) -> dict[str, Any]:
    by_wine: dict[str, list[bool]] = defaultdict(list)
    sl: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    soft_n = strict_n = 0
    for r in rows:
        a = answers[r["query_id"]]
        soft = a in {r["slug"], *(r.get("acceptable") or [])}
        strict = a == r["slug"]
        by_wine[r["source"]].append(soft)
        s = sl[r["query_id"][0]]
        s[0] += soft
        s[1] += strict
        s[2] += 1
        soft_n += soft
        strict_n += strict
    n = len(rows)
    return {
        "n": n,
        "soft_ok": soft_n,
        "strict_ok": strict_n,
        "soft_micro": round(100 * soft_n / n, 2),
        "soft_macro": round(100 * sum(sum(v) / len(v) for v in by_wine.values()) / len(by_wine), 2),
        "strict_micro": round(100 * strict_n / n, 2),
        "slices": {k: {"soft": v[0], "strict": v[1], "n": v[2]} for k, v in sorted(sl.items())},
    }


def compare(
    rows: Sequence[Mapping[str, Any]], base: Mapping[str, str], new: Mapping[str, str]
) -> dict[str, Any]:
    """Метрики нового ответа и парный счёт починок и поломок против базы, мягко и строго."""
    res = score(rows, new)
    fixes, breaks, sfix, sbrk, changed = [], [], [], [], []
    by_slice: dict[str, dict[str, int]] = {}
    for r in rows:
        q = r["query_id"]
        ok = {r["slug"], *(r.get("acceptable") or [])}
        b, n = base[q] in ok, new[q] in ok
        s = by_slice.setdefault(q[0], {"fix": 0, "brk": 0})
        if not b and n:
            fixes.append(r["photo"])
            s["fix"] += 1
        if b and not n:
            breaks.append(r["photo"])
            s["brk"] += 1
        if base[q] != r["slug"] and new[q] == r["slug"]:
            sfix.append(r["photo"])
        if base[q] == r["slug"] and new[q] != r["slug"]:
            sbrk.append(r["photo"])
        if base[q] != new[q]:
            changed.append(r["photo"])
    res.update(
        fixes=fixes,
        breaks=breaks,
        strict_fixes=sfix,
        strict_breaks=sbrk,
        by_slice=dict(sorted(by_slice.items())),
        changed=changed,
    )
    return res


def kr_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """617 основных кадров kr (без `same_packshot`); K0225 — отдельно, итог — 616."""
    main = [r for r in meta_of("kr") if not r.get("same_packshot")]
    aside = next(r for r in main if r["photo"] == KR_ASIDE)
    return [r for r in main if r["photo"] != KR_ASIDE], aside


def answers(run: str, set_key: str, *, wm: bool = False) -> dict[str, str]:
    if not wm:
        return {q: r["slug"] for q, r in stand(run, set_key).items()}
    return dict(redecided(run, set_key)["wm"])


def aside(
    row: Mapping[str, Any], base: Mapping[str, str], new: Mapping[str, str]
) -> dict[str, Any]:
    ok = {row["slug"], *(row.get("acceptable") or [])}
    q = row["query_id"]
    return {"base": base[q], "base_ok": base[q] in ok, "new": new[q], "new_ok": new[q] in ok}


def measure(run: str, base: Mapping[str, Any]) -> dict[str, Any]:
    v2_rows = meta_of("v2")
    kr_main, kr_aside = kr_rows()
    v2 = answers(run, "v2")
    kr = answers(run, "kr")
    kr_wm = answers(run, "kr", wm=True)
    res = {
        "health": run_health(run, "v2") + run_health(run, "kr"),
        "v2": compare(v2_rows, base["v2_ans"], v2),
        "kr": compare(kr_main, base["kr_ans"], kr),
        "kr_wm": compare(kr_main, base["kr_wm_ans"], kr_wm),
        "k0225": aside(kr_aside, base["kr_ans"], kr),
        "k0225_wm": aside(kr_aside, base["kr_wm_ans"], kr_wm),
        # справочно: кадры same_packshot (вне 617), у которых сменился ответ
        "kr_other_changed": sorted(
            q for q, a in kr.items() if q not in base["main_ids"] and base["kr_ans"][q] != a
        ),
    }
    res["soft_breaks"] = (
        len(res["v2"]["breaks"]) + len(res["kr"]["breaks"]) + len(res["kr_wm"]["breaks"])
    )
    # справочно (в приёмку не входит): у скольких кадров сменились поля чтения и порядок слоя
    for set_key in SETS:
        old, new = stand("base", set_key), stand(run, set_key)
        res[f"{set_key}_fields_changed"] = sorted(
            q
            for q in new
            if (new[q].get("text_read") or {}).get("fields")
            != (old[q].get("text_read") or {}).get("fields")
        )
        res[f"{set_key}_resolve_changed"] = sum(
            new[q].get("resolve") != old[q].get("resolve") for q in new
        )
    return res


def base_answers() -> dict[str, Any]:
    kr_main, kr_aside = kr_rows()
    return {
        "v2_ans": answers("base", "v2"),
        "kr_ans": answers("base", "kr"),
        "kr_wm_ans": answers("base", "kr", wm=True),
        "main_ids": {r["query_id"] for r in kr_main} | {kr_aside["query_id"]},
    }


def base_scores(b: Mapping[str, Any]) -> dict[str, Any]:
    kr_main, kr_aside = kr_rows()
    return {
        "v2": score(meta_of("v2"), b["v2_ans"]),
        "kr": score(kr_main, b["kr_ans"]),
        "kr_wm": score(kr_main, b["kr_wm_ans"]),
        "k0225": aside(kr_aside, b["kr_ans"], b["kr_ans"]),
        "k0225_wm": aside(kr_aside, b["kr_wm_ans"], b["kr_wm_ans"]),
        "health": run_health("base", "v2") + run_health("base", "kr"),
    }


def package_verdict(base: Mapping[str, Any], pkg: Mapping[str, Any]) -> list[str]:
    """Пустой список — пакет принят (PREREG, §8, п. 2–3); иначе — что не выполнено."""
    bad = []
    for s, cnt in pkg["v2"]["by_slice"].items():
        if cnt["brk"]:
            bad.append(f"v2 срез {s}: мягких поломок {cnt['brk']}")
    for label in ("kr", "kr_wm"):
        if pkg[label]["breaks"]:
            bad.append(
                f"{label}: мягких поломок {len(pkg[label]['breaks'])} {pkg[label]['breaks']}"
            )
    for label in ("v2", "kr", "kr_wm"):
        if pkg[label]["strict_ok"] < base[label]["strict_ok"]:
            bad.append(f"{label}: строго {pkg[label]['strict_ok']} < {base[label]['strict_ok']}")
    if pkg["health"]:
        bad.append(f"прогон: {pkg['health']}")
    return bad


def step1() -> dict[str, Any]:
    b = base_answers()
    units = {u: measure(u, b) for u in UNITS}
    kept = [u for u in UNITS if units[u]["soft_breaks"] == 0 and not units[u]["health"]]
    out = {
        "base": base_scores(b),
        "units": units,
        "kept": kept,
        "dropped": [u for u in UNITS if u not in kept],
        "pkg_rows": [int(u[3:]) for u in kept if u.startswith("row")],
        "pkg_compat": "compat" in kept,
    }
    return out


def line(name: str, r: Mapping[str, Any]) -> str:
    v2, kr, wm = r["v2"], r["kr"], r["kr_wm"]
    s = v2["slices"]
    return (
        f"{name:7s} v2 +{len(v2['fixes'])}/-{len(v2['breaks'])} {v2['fixes']} {v2['breaks']} "
        f"строго +{len(v2['strict_fixes'])}/-{len(v2['strict_breaks'])} смен {len(v2['changed'])} "
        f"R/M/L {s['R']['soft']}/{s['M']['soft']}/{s['L']['soft']} | "
        f"kr +{len(kr['fixes'])}/-{len(kr['breaks'])} {kr['fixes']} {kr['breaks']} смен {len(kr['changed'])} | "
        f"wm +{len(wm['fixes'])}/-{len(wm['breaks'])} {wm['fixes']} {wm['breaks']} | "
        f"K0225 {r['k0225']['base_ok']}->{r['k0225']['new_ok']} | прочие kr {len(r['kr_other_changed'])} | "
        f"{'ЦЕЛ' if not r['health'] else r['health']}"
    )


def main() -> int:
    step = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if step == "gate":
        g = gate()
        g["ok"] = gate_ok(g)
        print("ворота", "сошлись" if g["ok"] else "НЕ СОШЛИСЬ")
        (HERE / "e4_gate.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), "utf-8")
        return 0 if g["ok"] else 1
    res = step1()
    b = res["base"]
    print(
        f"база: v2 {b['v2']['soft_ok']}/353 строго {b['v2']['strict_ok']} макро {b['v2']['soft_macro']} "
        f"R/M/L {b['v2']['slices']['R']['soft']}/{b['v2']['slices']['M']['soft']}/{b['v2']['slices']['L']['soft']} | "
        f"kr {b['kr']['soft_ok']}/{b['kr']['n']} строго {b['kr']['strict_ok']} | "
        f"wm {b['kr_wm']['soft_ok']} строго {b['kr_wm']['strict_ok']} | K0225 {b['k0225']['base_ok']}"
    )
    for u in UNITS:
        print(line(u, res["units"][u]))
    print("шаг 1: прошли", res["kept"], "откат", res["dropped"])
    if step == "step1":
        (HERE / "e4_step1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        return 0
    b_ans = base_answers()
    pkg = measure("pkg", b_ans)
    pkg["verdict"] = package_verdict(res["base"], pkg)
    res["pkg"] = pkg
    print(line("pkg", pkg))
    print("пакет:", "ПРИНЯТ" if not pkg["verdict"] else f"ОТКЛОНЁН {pkg['verdict']}")
    (HERE / "e4_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
