"""Замер Э1 строго по `PREREG_E1_rules.md`: правило P1 и пункты разбора R1.

Только CPU и записанные прогоны: видеокарта, Ollama и SigLIP не нужны. Правила, счёт и приёмка —
из PREREG (коммит 8a7496d), здесь они только исполняются.

Входы:
    runs/field25/iters/runs/{pfix_final,kr_holdout,ooc_v2prod}/iter20 — дампы базы
    runs/field25/iters/runs/e1/<снимок>_<набор>                     — стенд `e1_stand.py`
    field_dataset/sets/{catalog_v2,krasnostop_v1,ooc_v2}/meta.jsonl  — эталон
    field_dataset/labels_v2.jsonl                                    — разметка чтения (ooc)

Шаги (`python e1_measure.py <шаг>`):
    gate      — стенд на коде базы = дампы: slug = H5 дампа, поля чтения равны (PREREG, §4)
    report    — все снимки против базы: v2 (R/M/L), kr (как есть и без водяного знака), ooc;
                приёмка по PREREG, §5; итог — в results.json

Решение кадра вне стенда (`decide`) — тем же кодом, что `ScannerService._resolve`: порядок модели,
при p < 0,5 — H5, иначе (для снимков с P1) — `block_bonus_flip`. Для счёта «как есть» берётся slug
стенда, и `decide` обязан его повторить на каждом кадре; для счёта без водяного знака — `decide`
по чтению после `strip_wm` (`acc_plan/errors_holdout/wm_whatif.py`).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "2026-09-24_h6"))

import measure_h6 as mh

from app.features.contracts import VisualResult
from app.reading.contracts import LabelFields
from app.resolve.ambiguous import block_bonus_flip, contradicts, rerank_ambiguous
from app.resolve.attrs import color_key, sugar_key
from app.resolve.features import TextRead, readers_of
from app.resolve.learned import LogisticRanker, rank_query_detailed

ROOT = Path(r"<корень>")
FD = ROOT / "field_dataset"
E1_RUNS = mh.RUNS / "e1"
ACC = ROOT / "svs-logs" / "somm-2409" / "acc_plan"
SETS = {
    "v2": ("pfix_final", "catalog_v2"),
    "kr": ("kr_holdout", "krasnostop_v1"),
    "ooc": ("ooc_v2prod", "ooc_v2"),
}
SLICES = ("R", "M", "L")
#: kr: K0638 — отдельно, в итог не входит (PREREG).
KR_ASIDE = "K0638"
#: Снимки стенда: имя → есть ли в коде сервиса P1.
SNAPSHOTS: dict[str, bool] = {
    "base": False, "off": False, "p1": True,
    "r1a": False, "r1b": False, "r1c": False, "r1g": False, "r1d": False,
    "r1": False, "pkg": True,
}  # fmt: skip
ITEMS = ("r1a", "r1b", "r1c", "r1g", "r1d")

MODEL = LogisticRanker.load(mh.MODEL)
READER = readers_of(MODEL.feature_names)[0]
ATTRS = mh.service_source().attrs

WM_SCRIPT = ACC / "errors_holdout" / "wm_whatif.py"


def _load_strip_wm() -> Callable[[dict[str, Any]], tuple[dict[str, Any], bool]]:
    """`strip_wm` и помощники из wm_whatif.py как есть — без его заголовка (он грузит svs-h6)."""
    from rapidfuzz.distance import Levenshtein

    source = WM_SCRIPT.read_text(encoding="utf-8")
    body = (
        "HOMO = str.maketrans" + source.split("HOMO = str.maketrans", 1)[1].split("def main()")[0]
    )
    ns: dict[str, Any] = {"re": re, "json": json, "Levenshtein": Levenshtein}
    exec(compile(body, str(WM_SCRIPT), "exec"), ns)  # noqa: S102 — инструмент плана без правок
    return ns["strip_wm"]


strip_wm = _load_strip_wm()


# ------------------------------------------------------------------ входы
def meta_of(set_key: str) -> dict[str, dict[str, Any]]:
    rows = mh.jsonl(FD / "sets" / SETS[set_key][1] / "meta.jsonl")
    return {r["query_id"]: r for r in rows}


def stand(name: str, set_key: str) -> dict[str, dict[str, Any]]:
    path = E1_RUNS / f"{name}_{set_key}" / "predictions.jsonl"
    return {r["query_id"]: r for r in mh.jsonl(path)}


def dump(set_key: str) -> dict[str, mh.Frame]:
    frames, _ = mh.load_run(SETS[set_key][0])
    return {f.qid: f for f in frames}


def fields_of(rec: Mapping[str, Any]) -> LabelFields | None:
    tr = rec.get("text_read") or {}
    return LabelFields.model_validate(tr["fields"]) if tr.get("fields") else None


def softmax_p1(scores: tuple[float, ...], temperature: float) -> float:
    """p лидера — формула `service.softmax` (сервис не импортируем: он тянет модели)."""
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return float((e / e.sum())[0])


def decide(rec: Mapping[str, Any], p1_rule: bool) -> str:
    """Ответ кадра так же, как `ScannerService._resolve`: H5 при p < 0,5, иначе P1 (если есть)."""
    vis = VisualResult.model_validate(rec["visual"])
    tr = rec.get("text_read") or {}
    fields = fields_of(rec)
    read = TextRead(fields=fields, raw_text=tr.get("raw_text"))
    ranking, feats = rank_query_detailed(MODEL, vis, {READER: read}, ATTRS)
    ranked = list(ranking.slugs)
    if softmax_p1(ranking.scores, MODEL.temperature_) < mh.AMBIGUOUS_P_TOP1:
        return rerank_ambiguous(MODEL, feats, fields, ATTRS)[0]
    if p1_rule:
        return block_bonus_flip(MODEL, feats, ranked, fields, ATTRS)[0]
    return ranked[0]


# ------------------------------------------------------------------ ворота
def gate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_key in SETS:
        base, frames = stand("base", set_key), dump(set_key)
        both = sorted(set(base) & set(frames))
        slug_bad = [q for q in both if base[q]["slug"] != frames[q].h5]
        fields_bad = [
            q
            for q in both
            if (base[q].get("text_read") or {}).get("fields")
            != (frames[q].record.get("text_read") or {}).get("fields")
        ]
        out[set_key] = {
            "stand": len(base),
            "dump": len(frames),
            "common": len(both),
            "slug_eq": len(both) - len(slug_bad),
            "fields_eq": len(both) - len(fields_bad),
            "slug_bad": slug_bad[:10],
            "fields_bad": fields_bad[:10],
        }
        print(set_key, json.dumps(out[set_key], ensure_ascii=False))
    return out


# ------------------------------------------------------------------ ответы снимков
def answers_of(name: str, set_key: str, *, strip: bool = False) -> dict[str, str]:
    """Ответы снимка: slug стенда; `strip` — решение по чтению без водяного знака."""
    recs = stand(name, set_key)
    if not strip:
        return {q: r["slug"] for q, r in recs.items()}
    out = {}
    for q, rec in recs.items():
        new, changed = strip_wm(rec)
        out[q] = decide(new, SNAPSHOTS[name]) if changed else rec["slug"]
    return out


def offline_mismatch(name: str, set_key: str, qids: list[str]) -> list[str]:
    """Кадры, где решение `decide` по входам стенда ≠ slug стенда (должно быть пусто)."""
    recs = stand(name, set_key)
    return [q for q in qids if decide(recs[q], SNAPSHOTS[name]) != recs[q]["slug"]]


def p1_offline_from_dump(set_key: str) -> dict[str, str]:
    """P1 по дампу базы, как `selection/kr_p1.py`: порядок модели и `whatif["no_cluster"]`."""
    out = {}
    for q, f in dump(set_key).items():
        answer = f.h5
        if f.p_top1 >= mh.AMBIGUOUS_P_TOP1 and f.whatif is not None:
            leader, plain = f.slug, f.whatif["no_cluster"][0]
            if (
                leader != plain
                and contradicts(leader, f.fields, ATTRS)
                and not contradicts(plain, f.fields, ATTRS)
            ):
                answer = plain
        out[q] = answer
    return out


# ------------------------------------------------------------------ счёт
def compare(
    rows: list[dict[str, Any]], base: Mapping[str, str], new: Mapping[str, str]
) -> dict[str, Any]:
    """Метрики нового ответа и парный счёт починок и поломок против базы."""
    m = mh.metrics(rows, new)
    fixes, breaks, sfix, sbrk = [], [], [], []
    by_slice: dict[str, list[int]] = {}
    for r in rows:
        q = r["query_id"]
        ok = {r["slug"], *(r.get("acceptable") or [])}
        b, n = base[q] in ok, new[q] in ok
        s = by_slice.setdefault(q[0], [0, 0])
        if not b and n:
            fixes.append(r["photo"])
            s[0] += 1
        if b and not n:
            breaks.append(r["photo"])
            s[1] += 1
        if base[q] != r["slug"] and new[q] == r["slug"]:
            sfix.append(r["photo"])
        if base[q] == r["slug"] and new[q] != r["slug"]:
            sbrk.append(r["photo"])
    return {
        "soft_ok": sum(
            new[r["query_id"]] in {r["slug"], *(r.get("acceptable") or [])} for r in rows
        ),
        "strict_ok": sum(new[r["query_id"]] == r["slug"] for r in rows),
        "n": len(rows),
        "soft_macro": round(m["soft_macro"], 2),
        "soft_micro": round(m["soft_micro"], 2),
        "strict_micro": round(m["strict_micro"], 2),
        "slices": m["slices"],
        "fixes": fixes,
        "breaks": breaks,
        "strict_fixes": sfix,
        "strict_breaks": sbrk,
        "by_slice": {k: {"fix": v[0], "brk": v[1]} for k, v in sorted(by_slice.items())},
        "changed": sum(base[r["query_id"]] != new[r["query_id"]] for r in rows),
    }


def kr_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """617 основных кадров kr без K0638 — в итог; K0638 — отдельно."""
    main = [r for r in meta_of("kr").values() if not r.get("same_packshot")]
    aside = next(r for r in main if r["photo"] == KR_ASIDE)
    return [r for r in main if r["photo"] != KR_ASIDE], aside


def aside_line(row: Mapping[str, Any], base: Mapping[str, str], new: Mapping[str, str]) -> str:
    ok = {row["slug"], *(row.get("acceptable") or [])}
    q = row["query_id"]
    return f"{row['photo']}: база {'верно' if base[q] in ok else 'неверно'}, " + (
        f"кандидат {'верно' if new[q] in ok else 'неверно'}"
    )


# ------------------------------------------------------------------ ooc: чтение против разметки
LABELS = {r["id"]: r for r in mh.jsonl(FD / "labels_v2.jsonl")}
#: Как в `acc_plan/reading/ooc_precision.py`: пометки разметки «не на этикетке».
NOT_ON = re.compile(
    r"карточк|не указ|не чита|не видно|catalog|not on|по дизайну|по линейке|по карточке|"
    r"не напечат|нет на",
    re.IGNORECASE,
)


def truth(photo: str) -> dict[str, str | None]:
    """Сахар и цвет разметки — функция `truth()` из `ooc_precision.py` без изменений."""
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


def ooc_reading(name: str) -> dict[str, Any]:
    meta = meta_of("ooc")
    c: Counter[str] = Counter()
    wrong_color: dict[str, str] = {}
    color_no_label: dict[str, str] = {}
    for q, rec in stand(name, "ooc").items():
        t = truth(meta[q]["photo"])
        fl = fields_of(rec)
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
        elif col and not t["color"]:
            color_no_label[q] = col
    return {"counts": dict(c), "wrong_color": wrong_color, "color_no_label": color_no_label}


# ------------------------------------------------------------------ приёмка (PREREG, §5)
SUGAR_MIN, COLOR_MIN = 290 / 295, 288 / 289


def v2_verdict(base: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Пустой список — критерии v2 плана выполнены; иначе — какие нет."""
    bad = []
    if len(new["fixes"]) < 4 * len(new["breaks"]):
        bad.append(f"v2 починок {len(new['fixes'])} < 4 × поломок {len(new['breaks'])}")
    if new["by_slice"].get("R", {}).get("brk", 0):
        bad.append("v2 поломка на R")
    for s in SLICES:
        if new["slices"][s]["soft"] < base["slices"][s]["soft"]:
            bad.append(f"v2 верных в {s} меньше")
    if new["strict_ok"] < base["strict_ok"]:
        bad.append("v2 строго микро ниже")
    return bad


def kr_verdict(asis: dict[str, Any], wm: dict[str, Any]) -> list[str]:
    bad = []
    for label, r in (("как есть", asis), ("без водяного знака", wm)):
        if len(r["fixes"]) - len(r["breaks"]) < 0:
            bad.append(f"kr {label}: нетто {len(r['fixes']) - len(r['breaks'])}")
    return bad


def ooc_verdict(base: dict[str, Any], new: dict[str, Any]) -> list[str]:
    bad = []
    c = new["counts"]
    if c.get("sugar_read") and c.get("sugar_ok", 0) / c["sugar_read"] < SUGAR_MIN - 1e-12:
        bad.append(f"ooc сахар {c.get('sugar_ok')}/{c['sugar_read']} < 290/295")
    if c.get("color_read") and c.get("color_ok", 0) / c["color_read"] < COLOR_MIN - 1e-12:
        bad.append(f"ooc цвет {c.get('color_ok')}/{c['color_read']} < 288/289")
    fresh = new_false_colors(base, new)
    if fresh:
        bad.append(f"ooc новых ложных цветов {len(fresh)}: {fresh}")
    return bad


def new_false_colors(base: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return sorted(q for q, col in new["wrong_color"].items() if base["wrong_color"].get(q) != col)


def fields_changed(name: str, set_key: str) -> int:
    """Сколько кадров набора разобрано снимком иначе, чем базой."""
    base, new = stand("base", set_key), stand(name, set_key)
    return sum(
        (r.get("text_read") or {}).get("fields") != (base[q].get("text_read") or {}).get("fields")
        for q, r in new.items()
    )


def report() -> dict[str, Any]:
    v2_rows = list(meta_of("v2").values())
    kr_main, kr_aside = kr_rows()
    base_v2 = answers_of("base", "v2")
    base_kr = answers_of("base", "kr")
    base_kr_wm = answers_of("base", "kr", strip=True)
    base_ooc = ooc_reading("base")
    base_v2_score = compare(v2_rows, base_v2, base_v2)
    base_kr_score = compare(kr_main, base_kr, base_kr)
    out: dict[str, Any] = {
        "base": {
            "v2": base_v2_score,
            "kr": base_kr_score,
            "kr_wm": compare(kr_main, base_kr, base_kr_wm),
            "ooc": base_ooc["counts"],
        }
    }
    kr_ids = [r["query_id"] for r in kr_main] + [kr_aside["query_id"]]
    v2_ids = [r["query_id"] for r in v2_rows]
    for name in SNAPSHOTS:
        if name == "base" or not (E1_RUNS / f"{name}_ooc" / "predictions.jsonl").is_file():
            continue
        v2 = answers_of(name, "v2")
        kr = answers_of(name, "kr")
        kr_wm = answers_of(name, "kr", strip=True)
        ooc = ooc_reading(name)
        res = {
            "v2": compare(v2_rows, base_v2, v2),
            "kr": compare(kr_main, base_kr, kr),
            "kr_wm": compare(kr_main, base_kr_wm, kr_wm),
            "kr_aside": aside_line(kr_aside, base_kr, kr),
            "kr_aside_wm": aside_line(kr_aside, base_kr_wm, kr_wm),
            "ooc": ooc["counts"],
            "ooc_new_false_colors": new_false_colors(base_ooc, ooc),
            "ooc_color_no_label": ooc["color_no_label"],
            "ooc_fields_changed": fields_changed(name, "ooc"),
            "v2_fields_changed": fields_changed(name, "v2"),
            "kr_fields_changed": fields_changed(name, "kr"),
            "offline_mismatch": {
                "v2": offline_mismatch(name, "v2", v2_ids),
                "kr": offline_mismatch(name, "kr", kr_ids),
            },
        }
        res["verdict"] = (
            v2_verdict(base_v2_score, res["v2"])
            + kr_verdict(res["kr"], res["kr_wm"])
            + ooc_verdict(base_ooc, ooc)
        )
        res["soft_breaks_anywhere"] = (
            len(res["v2"]["breaks"]) + len(res["kr"]["breaks"]) + len(res["kr_wm"]["breaks"])
        )
        out[name] = res
    p1_dump = {k: p1_offline_from_dump(k) for k in ("v2", "kr")}
    if "p1" in out:
        p1_stand = {k: answers_of("p1", k) for k in p1_dump}
        out["p1_dump_vs_stand"] = {
            k: [q for q, a in p1_dump[k].items() if p1_stand[k].get(q, a) != a] for k in p1_dump
        }
    items_kept = [i for i in ITEMS if i in out and out[i]["soft_breaks_anywhere"] == 0]
    out["step1_items_kept"] = items_kept
    out["step1_items_dropped"] = [i for i in ITEMS if i in out and i not in items_kept]
    return out


def summary(res: Mapping[str, Any]) -> None:
    b = res["base"]
    print(
        f"база: v2 мягко {b['v2']['soft_ok']}/353 строго {b['v2']['strict_ok']} "
        f"макро {b['v2']['soft_macro']} | kr {b['kr']['soft_ok']}/{b['kr']['n']} "
        f"wm {b['kr_wm']['soft_ok']} | ooc {b['ooc']}"
    )
    for name, r in res.items():
        if not isinstance(r, dict) or "kr_wm" not in r or name == "base":
            continue
        v2, kr, wm = r["v2"], r["kr"], r["kr_wm"]
        print(
            f"{name:5s} v2 +{len(v2['fixes'])}/-{len(v2['breaks'])} {v2['fixes']} {v2['breaks']} "
            f"строго +{len(v2['strict_fixes'])}/-{len(v2['strict_breaks'])} "
            f"R/M/L {v2['slices']['R']['soft']}/{v2['slices']['M']['soft']}/{v2['slices']['L']['soft']} "
            f"макро {v2['soft_macro']} | kr +{len(kr['fixes'])}/-{len(kr['breaks'])} {kr['fixes']} {kr['breaks']} "
            f"wm +{len(wm['fixes'])}/-{len(wm['breaks'])} {wm['fixes']} {wm['breaks']} | {r['kr_aside']} | "
            f"ooc {r['ooc']} новые ложные {r['ooc_new_false_colors']} полей изм. {r['ooc_fields_changed']} | "
            f"офлайн≠стенд {sum(map(len, r['offline_mismatch'].values()))} | вердикт {r['verdict'] or 'OK'}"
        )
    print("шаг 1: оставить", res["step1_items_kept"], "выбросить", res["step1_items_dropped"])
    if "p1_dump_vs_stand" in res:
        print("P1 офлайн по дампу ≠ стенд:", res["p1_dump_vs_stand"])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if step == "gate":
        result = gate()
        (HERE / "e1_gate.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    elif step == "report":
        result = report()
        (HERE / "e1_results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        summary(result)
