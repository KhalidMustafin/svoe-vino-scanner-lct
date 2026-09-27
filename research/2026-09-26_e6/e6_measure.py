"""Замер Э6 строго по `PREREG.md` (§3–4): ворота способа, пункты по одному, пакет.

Только CPU и записанные кэши стенда: видеокарта, Ollama и порт 8080 не нужны. Правила только
исполняются; ответы кадров посчитаны стендом (`e6_stand.py`) и кодом снимков (`e6_redecide.py`).

Входы:
    runs/field25/iters/runs/e6/<прогон>_{v2,kr,ooc}         — стенд Э6 и `redecide.json` (kr)
    runs/field25/iters/runs/e4/pkg_{v2,kr}, e1/pkg_ooc       — стенды Э4 и Э1 (ворота базы)
    acc_plan/integrate/expected_v2_freeze.json               — офлайн-ответы acc-freeze на v2
    field_dataset/sets/{catalog_v2,krasnostop_v1,ooc_v2}/meta.jsonl, labels_v2.jsonl

Шаги (`python e6_measure.py <шаг>`):
    gate   — база = стенд Э4/Э1 и офлайн-ответы acc-freeze; снимок off = база; счёт базы
    step1  — пять пунктов против базы; какие прошли (0 мягких поломок на v2 и kr в обоих счётах)
    report — step1 + пакет `pkg` против базы, приёмка пакета; итог — в e6_results.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.reading.contracts import LabelFields
from app.resolve.attrs import color_key, sugar_key

ROOT = Path(r"<корень>")
FD = ROOT / "field_dataset"
RUNS = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters" / "runs"
E6, E4, E1 = RUNS / "e6", RUNS / "e4", RUNS / "e1"
ACC = ROOT / "svs-logs" / "somm-2409" / "acc_plan"
SETS = {"v2": ("catalog_v2", 353), "kr": ("krasnostop_v1", 903), "ooc": ("ooc_v2", 408)}
SLICES = ("R", "M", "L")
#: kr: кадр вина, чьи атрибуты правил Э4 по slug товара krasnostop — вне ворот (§5 плана).
KR_ASIDE = "K0225"
UNITS = ["g", "v", "s", "r", "sh"]
UNIT_NAMES = {
    "g": "Э6-Г сорт внутри сорта",
    "v": "Э6-В винодельня из чужих слов",
    "s": "Э6-С слово стиля не кюве",
    "r": "Э6-Р регион латиницей",
    "sh": "Э6-Ш слово категории/хозяйства не кюве",
}
#: Ожидаемый счёт базы (acc-freeze, сводка 25.09 вне репозитория).
BASE_EXPECTED = {"v2": (315, 307, {"R": 63, "M": 233, "L": 19}), "kr617": 543}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------ входы
def meta_of(set_key: str) -> list[dict[str, Any]]:
    return jsonl(FD / "sets" / SETS[set_key][0] / "meta.jsonl")


def stand(run: str, set_key: str, root: Path = E6) -> dict[str, dict[str, Any]]:
    return {r["query_id"]: r for r in jsonl(root / f"{run}_{set_key}" / "predictions.jsonl")}


def run_info(run: str, set_key: str) -> dict[str, Any]:
    return json.loads((E6 / f"{run}_{set_key}" / "run_info.json").read_text(encoding="utf-8"))


def redecided(run: str) -> dict[str, Any]:
    return json.loads((E6 / f"{run}_kr" / "redecide.json").read_text(encoding="utf-8"))


def fields_of(rec: Mapping[str, Any]) -> Any:
    return (rec.get("text_read") or {}).get("fields")


def run_health(run: str) -> list[str]:
    """Прогон целый: все кадры, 0 промахов кэша, ключ CV дампов, решение снимка = стенд (kr)."""
    bad = []
    for set_key, (_, n) in SETS.items():
        info = run_info(run, set_key)
        rows = stand(run, set_key)
        if len(rows) != n:
            bad.append(f"{run}_{set_key}: кадров {len(rows)} ≠ {n}")
        cache = info["cache"]
        failed_ok = set_key == "ooc" and cache.get("failed") == 1  # один кадр ooc — всегда сбой
        if (
            cache.get("cv_miss")
            or cache.get("read_miss")
            or (cache.get("failed") and not failed_ok)
        ):
            bad.append(f"{run}_{set_key}: кэш {cache}")
        if info["cv_key"] != "ae2c4db886d1b1e9":
            bad.append(f"{run}_{set_key}: ключ CV {info['cv_key']}")
        if not info["provenance"]["gt_tokens_sha1"].startswith("899d5db3"):
            bad.append(f"{run}_{set_key}: эталон {info['provenance']['gt_tokens_sha1']}")
    red = redecided(run)
    if red["asis_mismatch"] or red["frames"] != SETS["kr"][1]:
        bad.append(f"{run}_kr: решение снимка ≠ стенд {red['asis_mismatch'][:5]}")
    return bad


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


def kr_rows() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Ворота kr — 616 (617 основных без K0225); K0225 отдельно; справочно — все 617."""
    main = [r for r in meta_of("kr") if not r.get("same_packshot")]
    aside = next(r for r in main if r["photo"] == KR_ASIDE)
    return [r for r in main if r["photo"] != KR_ASIDE], aside, main


def answers(run: str, set_key: str, *, wm: bool = False) -> dict[str, str]:
    if wm:
        return dict(redecided(run)["wm"])
    return {q: r["slug"] for q, r in stand(run, set_key).items()}


# ------------------------------------------------------------------ ooc: чтение против разметки
LABELS = {r["id"]: r for r in jsonl(FD / "labels_v2.jsonl")}
#: Как в `acc_plan/reading/ooc_precision.py` и `e1_measure.py`: пометки «не на этикетке».
NOT_ON = re.compile(
    r"карточк|не указ|не чита|не видно|catalog|not on|по дизайну|по линейке|по карточке|"
    r"не напечат|нет на",
    re.IGNORECASE,
)


def truth(photo: str) -> dict[str, str | None]:
    """Сахар и цвет разметки — функция `truth()` Э1 без изменений."""
    rd = (LABELS.get(photo) or {}).get("reading") or {}
    out: dict[str, str | None] = {}
    for fld, keyf in (("sugar", sugar_key), ("color", color_key)):
        t = str(rd.get(fld) or "")
        if not t or NOT_ON.search(t):
            out[fld] = None
            continue
        head = re.split(r"[(;,]", t)[0].strip()
        k = (
            keyf(head)
            or (keyf(" ".join(head.split()[:2])) if head.split() else None)
            or (keyf(head.split()[0]) if head.split() else None)
        )
        out[fld] = k.value if k else None
    return out


def ooc_reading(run: str) -> dict[str, Any]:
    meta = {r["query_id"]: r for r in meta_of("ooc")}
    c: Counter[str] = Counter()
    wrong_color: dict[str, str] = {}
    for q, rec in stand(run, "ooc").items():
        t = truth(meta[q]["photo"])
        raw = fields_of(rec)
        fl = LabelFields.model_validate(raw) if raw else None
        sugar = {str(e.value) for e in fl.sugar} if fl else set()
        col = str(fl.color.value) if fl and fl.color else None
        if t["sugar"] and sugar:
            c["sugar_read"] += 1
            c["sugar_ok" if t["sugar"] in sugar else "sugar_bad"] += 1
        if t["color"] and col:
            c["color_read"] += 1
            if t["color"] == col:
                c["color_ok"] += 1
            else:
                c["color_bad"] += 1
                wrong_color[q] = col
    return {"counts": dict(c), "wrong_color": wrong_color}


def ooc_verdict(base: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    bad = []
    b, n = base["counts"], new["counts"]
    for fld in ("sugar", "color"):
        br, nr = b.get(f"{fld}_read", 0), n.get(f"{fld}_read", 0)
        bo, no = b.get(f"{fld}_ok", 0), n.get(f"{fld}_ok", 0)
        if nr and br and no * br < bo * nr:  # доля верных ниже базы
            bad.append(f"ooc {fld} {no}/{nr} < {bo}/{br}")
    fresh = sorted(q for q, col in new["wrong_color"].items() if base["wrong_color"].get(q) != col)
    if fresh:
        bad.append(f"ooc новых ложных цветов {len(fresh)}: {fresh}")
    return bad


# ------------------------------------------------------------------ ворота
def gate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_key in ("v2", "kr"):
        base, e4 = stand("base", set_key), stand("pkg", set_key, E4)
        both = sorted(set(base) & set(e4))
        out[f"base_vs_e4pkg_{set_key}"] = {
            "base": len(base),
            "e4": len(e4),
            "slug_eq": sum(base[q]["slug"] == e4[q]["slug"] for q in both),
            "text_read_eq": sum(base[q].get("text_read") == e4[q].get("text_read") for q in both),
            "resolve_eq": sum(base[q].get("resolve") == e4[q].get("resolve") for q in both),
        }
    base, e1 = stand("base", "ooc"), stand("pkg", "ooc", E1)
    out["base_vs_e1pkg_ooc"] = {
        "base": len(base),
        "fields_eq": sum(fields_of(base[q]) == fields_of(e1[q]) for q in base if q in e1),
    }
    expected = json.loads(
        (ACC / "integrate" / "expected_v2_freeze.json").read_text(encoding="utf-8")
    )["answers"]
    v2 = answers("base", "v2")
    out["base_vs_expected_v2_freeze"] = sum(v2[q] == a["off"] for q, a in expected.items())
    for set_key in SETS:
        off, base = stand("off", set_key), stand("base", set_key)
        both = sorted(set(off) & set(base))
        out[f"off_vs_base_{set_key}"] = {
            "off": len(off),
            "base": len(base),
            "slug_eq": sum(off[q]["slug"] == base[q]["slug"] for q in both),
            "fields_eq": sum(fields_of(off[q]) == fields_of(base[q]) for q in both),
            "resolve_eq": sum(off[q].get("resolve") == base[q].get("resolve") for q in both),
        }
    out["off_vs_base_kr_wm"] = sum(
        a == redecided("base")["wm"][q] for q, a in redecided("off")["wm"].items()
    )
    bs = base_scores()
    out["base_scores"] = {
        "v2": [bs["v2"]["soft_ok"], bs["v2"]["strict_ok"], bs["v2"]["slices"]],
        "kr617": bs["kr617"]["soft_ok"],
        "kr616": bs["kr"]["soft_ok"],
        "kr616_wm": bs["kr_wm"]["soft_ok"],
    }
    out["health"] = {"base": run_health("base"), "off": run_health("off")}
    for key, val in out.items():
        print(key, json.dumps(val, ensure_ascii=False))
    return out


def gate_ok(g: Mapping[str, Any]) -> bool:
    ok = True
    for set_key in ("v2", "kr"):
        n = SETS[set_key][1]
        v = g[f"base_vs_e4pkg_{set_key}"]
        ok &= v["base"] == v["e4"] == v["slug_eq"] == v["text_read_eq"] == v["resolve_eq"] == n
    ok &= g["base_vs_e1pkg_ooc"]["fields_eq"] == g["base_vs_e1pkg_ooc"]["base"] == 408
    ok &= g["base_vs_expected_v2_freeze"] == 353
    for set_key, (_, n) in SETS.items():
        v = g[f"off_vs_base_{set_key}"]
        ok &= v["off"] == v["base"] == v["slug_eq"] == v["fields_eq"] == v["resolve_eq"] == n
    ok &= g["off_vs_base_kr_wm"] == SETS["kr"][1]
    soft, strict, slices = BASE_EXPECTED["v2"]
    s = g["base_scores"]
    ok &= s["v2"][0] == soft and s["v2"][1] == strict
    ok &= all(s["v2"][2][k]["soft"] == v for k, v in slices.items())
    ok &= s["kr617"] == BASE_EXPECTED["kr617"]
    ok &= not g["health"]["base"] and not g["health"]["off"]
    return bool(ok)


# ------------------------------------------------------------------ пункты и пакет
def base_answers() -> dict[str, Any]:
    return {
        "v2": answers("base", "v2"),
        "kr": answers("base", "kr"),
        "kr_wm": answers("base", "kr", wm=True),
        "ooc": ooc_reading("base"),
    }


def base_scores() -> dict[str, Any]:
    b = base_answers()
    kr_gate, _, kr_all = kr_rows()
    return {
        "v2": score(meta_of("v2"), b["v2"]),
        "kr": score(kr_gate, b["kr"]),
        "kr_wm": score(kr_gate, b["kr_wm"]),
        "kr617": score(kr_all, b["kr"]),
        "kr617_wm": score(kr_all, b["kr_wm"]),
        "ooc": b["ooc"]["counts"],
    }


def measure(run: str, b: Mapping[str, Any]) -> dict[str, Any]:
    kr_gate, aside, kr_all = kr_rows()
    v2, kr, kr_wm = answers(run, "v2"), answers(run, "kr"), answers(run, "kr", wm=True)
    ooc = ooc_reading(run)
    q = aside["query_id"]
    ok = {aside["slug"], *(aside.get("acceptable") or [])}
    res: dict[str, Any] = {
        "health": run_health(run),
        "v2": compare(meta_of("v2"), b["v2"], v2),
        "kr": compare(kr_gate, b["kr"], kr),
        "kr_wm": compare(kr_gate, b["kr_wm"], kr_wm),
        "kr617": compare(kr_all, b["kr"], kr),
        "kr617_wm": compare(kr_all, b["kr_wm"], kr_wm),
        "k0225": {"base_ok": b["kr"][q] in ok, "new_ok": kr[q] in ok},
        "kr_same_packshot_changed": sorted(
            x for x in kr if x not in {r["query_id"] for r in kr_all} and kr[x] != b["kr"][x]
        ),
        "ooc": ooc["counts"],
        "ooc_verdict": ooc_verdict(b["ooc"], ooc),
    }
    res["soft_breaks"] = (
        len(res["v2"]["breaks"]) + len(res["kr"]["breaks"]) + len(res["kr_wm"]["breaks"])
    )
    for set_key in SETS:  # справочно: у скольких кадров сменились поля чтения
        old, new = stand("base", set_key), stand(run, set_key)
        res[f"{set_key}_fields_changed"] = sum(fields_of(new[x]) != fields_of(old[x]) for x in new)
    return res


def package_verdict(base: Mapping[str, Any], pkg: Mapping[str, Any]) -> list[str]:
    """Пустой список — пакет принят (PREREG, §4, п. 2); иначе — что не выполнено."""
    bad = []
    v2 = pkg["v2"]
    if len(v2["fixes"]) < 4 * len(v2["breaks"]):
        bad.append(f"v2 починок {len(v2['fixes'])} < 4 × поломок {len(v2['breaks'])}")
    if v2["by_slice"].get("R", {}).get("brk", 0):
        bad.append("v2 поломка на R")
    for s in SLICES:
        if v2["slices"][s]["soft"] < base["v2"]["slices"][s]["soft"]:
            bad.append(f"v2 верных в {s} меньше")
    if v2["strict_ok"] < base["v2"]["strict_ok"]:
        bad.append(f"v2 строго {v2['strict_ok']} < {base['v2']['strict_ok']}")
    for label in ("kr", "kr_wm"):
        net = len(pkg[label]["fixes"]) - len(pkg[label]["breaks"])
        if net < 0:
            bad.append(f"{label}: нетто {net}")
    bad += pkg["ooc_verdict"]
    if pkg["health"]:
        bad.append(f"прогон: {pkg['health']}")
    return bad


def line(name: str, r: Mapping[str, Any]) -> str:
    v2, kr, wm = r["v2"], r["kr"], r["kr_wm"]
    s = v2["slices"]
    return (
        f"{name:3s} v2 +{len(v2['fixes'])}/-{len(v2['breaks'])} {v2['fixes']} {v2['breaks']} "
        f"строго +{len(v2['strict_fixes'])}/-{len(v2['strict_breaks'])} смен {len(v2['changed'])} "
        f"→ {v2['soft_ok']}/{v2['n']} строго {v2['strict_ok']} "
        f"R/M/L {s['R']['soft']}/{s['M']['soft']}/{s['L']['soft']} | "
        f"kr616 +{len(kr['fixes'])}/-{len(kr['breaks'])} {kr['fixes']} {kr['breaks']} "
        f"смен {len(kr['changed'])} | wm +{len(wm['fixes'])}/-{len(wm['breaks'])} "
        f"{wm['fixes']} {wm['breaks']} | K0225 {r['k0225']['base_ok']}->{r['k0225']['new_ok']} | "
        f"ooc {r['ooc']} {r['ooc_verdict'] or ''} | полей изм. v2/kr/ooc "
        f"{r['v2_fields_changed']}/{r['kr_fields_changed']}/{r['ooc_fields_changed']} | "
        f"{'ЦЕЛ' if not r['health'] else r['health']}"
    )


def main() -> int:
    step = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if step == "gate":
        g = gate()
        g["ok"] = gate_ok(g)
        print("ворота", "сошлись" if g["ok"] else "НЕ СОШЛИСЬ")
        (HERE / "e6_gate.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), "utf-8")
        return 0 if g["ok"] else 1
    b = base_answers()
    bs = base_scores()
    units = {u: measure(u, b) for u in UNITS}
    kept = [u for u in UNITS if units[u]["soft_breaks"] == 0 and not units[u]["health"]]
    res: dict[str, Any] = {
        "base": bs,
        "units": units,
        "kept": kept,
        "dropped": [u for u in UNITS if u not in kept],
    }
    print(
        f"база: v2 {bs['v2']['soft_ok']}/353 строго {bs['v2']['strict_ok']} "
        f"R/M/L {bs['v2']['slices']['R']['soft']}/{bs['v2']['slices']['M']['soft']}/"
        f"{bs['v2']['slices']['L']['soft']} | kr616 {bs['kr']['soft_ok']} строго "
        f"{bs['kr']['strict_ok']} | wm {bs['kr_wm']['soft_ok']} | kr617 {bs['kr617']['soft_ok']} | "
        f"ooc {bs['ooc']}"
    )
    for u in UNITS:
        print(line(u, units[u]), "—", UNIT_NAMES[u])
    print("шаг 1: прошли", kept, "выброшены", res["dropped"])
    if step == "step1" or not kept:
        if not kept:
            res["pkg"] = None
            print("пакет пуст: ни один пункт не прошёл шаг 1 — правок нет (PREREG, §4, п. 3)")
        name = "e6_step1.json" if step == "step1" else "e6_results.json"
        (HERE / name).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        return 0
    pkg_spec = (E6 / "pkg_v2" / "run_info.json").read_text(encoding="utf-8")
    snap_note = Path(json.loads(pkg_spec)["snapshot"]) / "E6_SNAPSHOT.txt"
    res["pkg_snapshot"] = snap_note.read_text(encoding="utf-8").strip()
    pkg = measure("pkg", b)
    pkg["verdict"] = package_verdict(bs, pkg)
    res["pkg"] = pkg
    print(res["pkg_snapshot"])
    print(line("pkg", pkg))
    print("пакет:", "ПРИНЯТ" if not pkg["verdict"] else f"ОТКЛОНЁН {pkg['verdict']}")
    (HERE / "e6_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
