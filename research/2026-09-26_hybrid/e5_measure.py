"""Замер Э5 строго по `PREREG.md` этой папки. Только CPU и записанные кэши стенда.

Входы:
    runs/field25/iters/runs/e5/base_{v2,kr} — стенд `acc-freeze` (`e5_chain.sh`): CV, чтение
                                              и ответ базы; `redecide.json` — ответы кода снимка
                                              «как есть» и «без водяного знака»
    field_dataset/sets/{catalog_v2,krasnostop_v1}/meta.jsonl — эталон
    configs/resolve/s2so400m-vlm35-goal.json — замороженная -goal (калитка)
    configs/resolve/s2so400m-vlm35-a3.json   — итоговая A3 (`bench.train_hybrid`, шаг kr)

Ответ кандидата на кадре — `ScannerService._resolve` этой ветки с моделью A3 (код сборки);
на каждом кадре он сверяется с офлайн-ответом `bench.resolve_equality.offline_answer`.

    python e5_measure.py gate      ворота базы (§2) и сверка признаков с augment() 25.09 (§3)
    python e5_measure.py oof       OOF на v2, 5 разбиений, приёмка (§6) → e5_oof.json
    python e5_measure.py kr        итоговая A3 один раз на kr 617, приёмка (§7) → e5_kr.json
    python e5_measure.py service   сервис = ответы замера на входах стенда (§8) и справочно
                                   v2 в выборке (§9) → e5_service.json
    python e5_measure.py insample  только справочно, после отказа по OOF: итоговая A3 (студия +
                                   353 v2) в файл этой папки, не в configs; v2 в выборке;
                                   kr не открывается → e5_insample.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from rapidfuzz.distance import Levenshtein

from app.api.config import ServiceSettings
from app.api.service import ScannerService, _Run, load_catalog
from app.features.contracts import IndexMeta, VisualResult
from app.features.index import VisualIndex
from app.resolve.features import readers_of
from app.resolve.learned import LogisticRanker, order_by_model
from app.resolve.rerank import read_fields
from bench.resolve_equality import CLUSTER_FEATURE, Weights, offline_answer
from bench.train_hybrid import (
    Item,
    a3_names,
    field_items,
    fit_a3,
    read_meta,
    studio_items,
    text_read_of,
)

ROOT = Path(r"<корень>")
SCANNER = ROOT / "svoe-vino-scanner"
RUNS = SCANNER / "runs" / "field25" / "iters" / "runs"
E5 = RUNS / "e5"
E4_PKG = RUNS / "e4"
FD = ROOT / "field_dataset"
GT = RUNS / "e4" / "gt" / "pkg" / "gt_tokens.jsonl"
GT_SHA1 = "899d5db3747fd315b7c8bfedf121f6ae79799061"
GOAL = REPO / "configs" / "resolve" / "s2so400m-vlm35-goal.json"
GOAL_SHA1 = "be51adb713e1b7b7b564a41c73dce384066d22be"
A3 = REPO / "configs" / "resolve" / "s2so400m-vlm35-a3.json"
GT_FIXES = REPO / "data" / "gt" / "gt_fixes.tsv"
WM_SCRIPT = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "errors_holdout" / "wm_whatif.py"
SETS = {"v2": ("catalog_v2", 353), "kr": ("krasnostop_v1", 903)}
SLICES = ("R", "M", "L")
SPLIT_SEEDS = (0, 1, 2, 3, 4)
K = 5
#: Счёт базы (сводка замеров 25.09, вне репозитория): ворота §2.
BASE_V2 = {"soft": 315, "strict": 307, "R": 63, "M": 233, "L": 19}
BASE_KR = {"asis": 543, "wm": 545}


# ------------------------------------------------------------------ входы
def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stand(set_key: str) -> list[dict[str, Any]]:
    return jsonl(E5 / f"base_{set_key}" / "predictions.jsonl")


def redecided(set_key: str) -> dict[str, Any]:
    return json.loads((E5 / f"base_{set_key}" / "redecide.json").read_text(encoding="utf-8"))


def meta_of(set_key: str) -> dict[str, dict[str, Any]]:
    return read_meta(FD / "sets" / SETS[set_key][0] / "meta.jsonl")


def main_kr(meta: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [q for q, m in meta.items() if not m.get("same_packshot")]


def load_strip_wm() -> Callable[[dict[str, Any]], tuple[dict[str, Any], bool]]:
    """`strip_wm` из wm_whatif.py как есть — без его заголовка (как `e4_redecide.py`)."""
    source = WM_SCRIPT.read_text(encoding="utf-8")
    body = (
        "HOMO = str.maketrans" + source.split("HOMO = str.maketrans", 1)[1].split("def main()")[0]
    )
    ns: dict[str, Any] = {"re": re, "json": json, "Levenshtein": Levenshtein}
    exec(compile(body, str(WM_SCRIPT), "exec"), ns)  # noqa: S102 — инструмент плана без правок
    return ns["strip_wm"]


def s1_slugs() -> set[str]:
    """Вина строк `gt_fixes.tsv`, найденных по slug товара krasnostop (attr_audit S1)."""
    out = set()
    with GT_FIXES.open(encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(head, line.rstrip("\n").split("\t"), strict=False))
            if "attr_audit S1" in row.get("flagged_by", ""):
                out.add(row["slug"])
    return out


assert sha1(GT) == GT_SHA1, "эталон стенда не 899d5db3"
assert sha1(GOAL) == GOAL_SHA1, "модель -goal не та"
_, ATTRS = load_catalog(GT)
GOAL_MODEL = LogisticRanker.load(GOAL)
(READER,) = readers_of(GOAL_MODEL.feature_names)
NAMES = a3_names(GOAL_MODEL)
GOAL_W = Weights.load(GOAL)


# ------------------------------------------------------------------ сервис и офлайн
class _NoEmbedder:
    dim = 1

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        raise RuntimeError("решение по записанным входам: CV не пересчитывается")


def service(hybrid: LogisticRanker | None) -> ScannerService:
    """Сервис ветки: настоящие только -goal, A3 (или без неё) и разметка каталога."""
    settings = ServiceSettings(resolve_model=GOAL, hybrid_model=None)
    meta = IndexMeta(model=settings.cv_model, dim=1, views=["bottle"], n_slugs=1, n_vectors=1)
    index = VisualIndex(["__e5__"], ["bottle"], np.ones((1, 1), dtype=np.float32), meta)
    return ScannerService(
        settings,
        index=index,
        embedder=_NoEmbedder(settings.cv_model),
        lexicon=None,
        attrs=ATTRS,
        model=GOAL_MODEL,
        vlm=None,
        hybrid=hybrid,
    )


def decide(svc: ScannerService, rec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    visual = VisualResult.model_validate(rec["visual"])
    read = text_read_of(rec)
    now = time.perf_counter()
    run = _Run(clock=time.perf_counter, started=now, deadline=now + 3600.0)
    answer = svc._resolve(visual, {READER: read}, read.fields, run)
    return answer["slug"], run.evidence["resolve"]


def weights_of(model: LogisticRanker) -> Weights:
    assert model.coef_ is not None and model.mean_ is not None and model.scale_ is not None
    names = tuple(model.feature_names)
    coef = np.asarray(model.coef_, dtype=np.float64)
    plain = coef.copy()
    if CLUSTER_FEATURE in names:
        plain[names.index(CLUSTER_FEATURE)] = 0.0
    return Weights(
        names=names,
        coef=coef,
        coef_plain=plain,
        mean=np.asarray(model.mean_, dtype=np.float64),
        scale=np.asarray(model.scale_, dtype=np.float64),
        intercept=float(model.intercept_),
        temperature=float(model.temperature_),
        top_k=int(model.meta.get("top_k") or 20),
    )


def offline(rec: Mapping[str, Any], hybrid: Weights | None) -> str:
    visual = VisualResult.model_validate(rec["visual"])
    return offline_answer(GOAL_W, visual, text_read_of(rec), ATTRS, READER, hybrid)


# ------------------------------------------------------------------ счёт
def slice_of(meta: Mapping[str, Any]) -> str:
    return str(meta["photo"])[0]


def score(
    qids: Sequence[str],
    ans: Mapping[str, str],
    base: Mapping[str, str],
    meta: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    def ok(q: str, a: str) -> bool:
        m = meta[q]
        return a == m["slug"] or a in (m.get("acceptable") or [])

    per_wine: dict[str, list[bool]] = defaultdict(list)
    slices: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    for q in qids:
        per_wine[meta[q]["source"]].append(ok(q, ans[q]))
        slices[slice_of(meta[q])] += ok(q, ans[q])
        sizes[slice_of(meta[q])] += 1
    fixes = [q for q in qids if ok(q, ans[q]) and not ok(q, base[q])]
    breaks = [q for q in qids if not ok(q, ans[q]) and ok(q, base[q])]
    strict = sum(ans[q] == meta[q]["slug"] for q in qids)
    return {
        "n": len(qids),
        "soft": sum(ok(q, ans[q]) for q in qids),
        "strict": strict,
        "micro": round(100 * sum(ok(q, ans[q]) for q in qids) / len(qids), 2),
        "macro": round(100 * float(np.mean([np.mean(v) for v in per_wine.values()])), 2),
        "strict_micro": round(100 * strict / len(qids), 2),
        "slices": {s: slices[s] for s in sorted(sizes)},
        "sizes": {s: sizes[s] for s in sorted(sizes)},
        "fixes": [meta[q]["photo"] for q in fixes],
        "breaks": [meta[q]["photo"] for q in breaks],
        "strict_fixes": sum(ans[q] == meta[q]["slug"] != base[q] for q in qids),
        "strict_breaks": sum(base[q] == meta[q]["slug"] != ans[q] for q in qids),
        "changed": sum(ans[q] != base[q] for q in qids),
    }


def line(name: str, s: Mapping[str, Any]) -> str:
    sl = " ".join(f"{k} {v}/{s['sizes'][k]}" for k, v in s["slices"].items())
    return (
        f"{name:28s} мягко {s['soft']}/{s['n']} ({s['micro']:.2f}, макро {s['macro']:.2f}) "
        f"строго {s['strict']} ({s['strict_micro']:.2f}) | {sl} | "
        f"+{len(s['fixes'])}/−{len(s['breaks'])} строго +{s['strict_fixes']}/−{s['strict_breaks']}"
        f" | смен {s['changed']} | починки {s['fixes']} поломки {s['breaks']}"
    )


def accept_v2(cand: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, bool]:
    breaks_r = [p for p in cand["breaks"] if p.startswith("R")]
    return {
        "fixes>=4xbreaks": len(cand["fixes"]) >= 4 * len(cand["breaks"]),
        "breaks_R=0": not breaks_r,
        **{f"{s}>=base": cand["slices"][s] >= base["slices"][s] for s in SLICES},
        "strict>=base": cand["strict"] >= base["strict"],
    }


def accept_kr(cand: Mapping[str, Any]) -> dict[str, bool]:
    fixes, breaks = len(cand["fixes"]), len(cand["breaks"])
    return {"net>=0": fixes - breaks >= 0, "fixes>=2xbreaks": fixes >= 2 * breaks}


def dump(name: str, obj: Any) -> None:
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ фолды (common.py 25.09)
def group_folds(groups: list[str], seed: int, k: int = K) -> np.ndarray:
    """Группа целиком в одном фолде; порядок групп — перестановка по seed, каждая — в самый
    лёгкий (по числу запросов) фолд, при равенстве — в меньший номер."""
    counts = Counter(groups)
    names = sorted(counts)
    perm = np.random.default_rng(seed).permutation(len(names))
    load = [0] * k
    fold_of: dict[str, int] = {}
    for i in perm:
        g = names[int(i)]
        f = min(range(k), key=lambda j: (load[j], j))
        fold_of[g] = f
        load[f] += counts[g]
    return np.asarray([fold_of[g] for g in groups], dtype=np.int64)


# ------------------------------------------------------------------ augment() 25.09 — копия
PREFIX = f"{READER}."
NEW_BASE = ("grape", "sugar", "color", "winery", "cuvee")
NEW_NAMES = [f"{PREFIX}{f}_match_unique" for f in NEW_BASE] + [f"{PREFIX}grape_jaccard"]


def augment_2509(item: Item) -> list[dict[str, float]]:
    """Шесть признаков A3 кодом `acc_plan/selection/common.py::augment` (25.09)."""
    rows = [dict(r) for r in item.features.rows]
    for f in NEW_BASE:
        col = f"{PREFIX}{f}_match"
        n = sum(r[col] for r in rows)
        for r in rows:
            r[f"{PREFIX}{f}_match_unique"] = r[col] / n if r[col] > 0 else 0.0
    read = read_fields(item.fields, ATTRS).grapes
    for slug, r in zip(item.slugs, rows, strict=True):
        wine = ATTRS.get(slug)
        if read and wine is not None:
            union = read | wine.grapes
            r[f"{PREFIX}grape_jaccard"] = len(read & wine.grapes) / len(union) if union else 0.0
        else:
            r[f"{PREFIX}grape_jaccard"] = 0.0
    return rows


def features_match_2509(items: Sequence[Item]) -> dict[str, int]:
    bad = 0
    for item in items:
        for new, got in zip(augment_2509(item), item.features.rows, strict=True):
            if any(new[n] != got[n] for n in NEW_NAMES):
                bad += 1
                break
    return {"items": len(items), "differ": bad}


# ------------------------------------------------------------------ шаги
def step_gate() -> int:
    out: dict[str, Any] = {"gt_sha1": GT_SHA1, "goal_sha1": GOAL_SHA1}
    problems: list[str] = []
    strip_wm = load_strip_wm()
    off = service(None)
    for set_key, (_, n) in SETS.items():
        info = json.loads((E5 / f"base_{set_key}" / "run_info.json").read_text(encoding="utf-8"))
        recs = stand(set_key)
        red = redecided(set_key)
        cache = info["cache"]
        health = {
            "frames": len(recs),
            "cache": cache,
            "cv_key": info["cv_key"],
            "gt": info["provenance"]["gt_tokens_sha1"],
            "redecide_mismatch": len(red["asis_mismatch"]),
        }
        if (
            len(recs) != n
            or cache.get("cv_miss")
            or cache.get("read_miss")
            or info["cv_key"] != "ae2c4db886d1b1e9"
            or health["gt"] != GT_SHA1
            or red["asis_mismatch"]
        ):
            problems.append(f"{set_key}: стенд {health}")
        # Выключатель: ветка без A3 = код acc-freeze (как есть и без водяного знака).
        asis_eq = sum(decide(off, r)[0] == r["slug"] for r in recs)
        wm_eq = 0
        for r in recs:
            new, did = strip_wm(r)
            wm_eq += (decide(off, new)[0] if did else r["slug"]) == red["wm"][r["query_id"]]
        health["branch_off_eq_stand"] = asis_eq
        health["branch_off_eq_wm"] = wm_eq
        if asis_eq != n or wm_eq != n:
            problems.append(f"{set_key}: ветка без A3 ≠ acc-freeze ({asis_eq}, {wm_eq} из {n})")
        e4_path = E4_PKG / f"pkg_{set_key}" / "predictions.jsonl"
        e4 = {r["query_id"]: r["slug"] for r in jsonl(e4_path)}
        health["eq_e4_pkg_stand"] = sum(e4.get(r["query_id"]) == r["slug"] for r in recs)
        out[set_key] = health
        print(f"{set_key}: {json.dumps(health, ensure_ascii=False)}")

    meta = meta_of("v2")
    recs = stand("v2")
    base = {r["query_id"]: r["slug"] for r in recs}
    s = score(list(meta), base, base, meta)
    out["base_v2"] = s
    print(line("база v2", s))
    want = {"soft": s["soft"], "strict": s["strict"], **s["slices"]}
    if want != BASE_V2:
        problems.append(f"база v2 {want} ≠ {BASE_V2}")
    kmeta = meta_of("kr")
    qids = main_kr(kmeta)
    red = redecided("kr")
    for count in ("asis", "wm"):
        ks = score(qids, red[count], red[count], kmeta)
        out[f"base_kr_{count}"] = ks
        print(line(f"база kr {count}", ks))
        if ks["soft"] != BASE_KR[count]:
            problems.append(f"база kr {count} {ks['soft']} ≠ {BASE_KR[count]}")

    # §3: признаки кода = augment() 25.09 на v2, kr (903) и студии.
    v2_items = field_items(recs, meta, ATTRS, READER)
    kr_items = field_items(stand("kr"), kmeta, ATTRS, READER)
    studio, sinfo = studio_items(GOAL_MODEL, SCANNER, ATTRS, GT)
    feats = {
        "v2": features_match_2509(v2_items),
        "kr": features_match_2509(kr_items),
        "studio": features_match_2509(studio),
    }
    out["features_vs_2509"] = feats
    out["studio"] = {"queries": sinfo["queries"], "items": len(studio)}
    print(f"признаки vs augment() 25.09: {feats}; студия {out['studio']}")
    if any(v["differ"] for v in feats.values()):
        problems.append(f"признаки ≠ augment(): {feats}")
    if len(studio) != 298:
        problems.append(f"студия {len(studio)} ≠ 298")
    out["problems"] = problems
    dump("e5_gate.json", out)
    print("ВОРОТА:", "пройдены" if not problems else f"НЕ пройдены: {problems}")
    return 1 if problems else 0


def step_oof() -> int:
    t0 = time.perf_counter()
    meta = meta_of("v2")
    recs = stand("v2")
    by_q = {r["query_id"]: r for r in recs}
    base = {q: r["slug"] for q, r in by_q.items()}
    qids = list(meta)
    items = field_items(recs, meta, ATTRS, READER)
    studio, _ = studio_items(GOAL_MODEL, SCANNER, ATTRS, GT)
    base_s = score(qids, base, base, meta)
    print(line("база acc-freeze", base_s))
    groups = [it.group for it in items]
    print(f"v2 {len(items)} кадров, групп {len(set(groups))}, студия {len(studio)}", flush=True)
    out: dict[str, Any] = {"base": base_s, "groups": len(set(groups)), "splits": {}}
    passed_all = True
    for seed in SPLIT_SEEDS:
        folds = group_folds(groups, seed)
        ans: dict[str, str] = {}
        gate_open = 0
        mismatch = []
        fit_info = []
        for k in range(K):
            train = [it for it, f in zip(items, folds, strict=True) if f != k] + studio
            model = fit_a3(train, NAMES)
            fit_info.append(model.fit_info)
            svc = service(model)
            w = weights_of(model)
            for it, f in zip(items, folds, strict=True):
                if f != k:
                    continue
                slug, ev = decide(svc, by_q[it.key])
                ans[it.key] = slug
                gate_open += bool(ev["hybrid"])
                if offline(by_q[it.key], w) != slug:
                    mismatch.append(it.key)
        s = score(qids, ans, base, meta)
        acc = accept_v2(s, base_s)
        passed = all(acc.values())
        passed_all &= passed
        out["splits"][seed] = {
            "score": s,
            "accept": acc,
            "passed": passed,
            "gate_open": gate_open,
            "offline_mismatch": mismatch,
            "folds": [int((folds == k).sum()) for k in range(K)],
            "converged": all(i.get("converged") for i in fit_info),
            "answers": ans,
        }
        print(line(f"seed {seed}", s), flush=True)
        print(
            f"   калитка открыта {gate_open}/{len(qids)}; сервис ≠ офлайн {len(mismatch)}; "
            f"фолды {out['splits'][seed]['folds']}; сошлись {out['splits'][seed]['converged']}; "
            f"приёмка {acc} → {'ПРОШЛО' if passed else 'НЕ ПРОШЛО'}",
            flush=True,
        )
        if mismatch:
            print(f"   ОШИБКА: сервис ≠ офлайн на {mismatch[:10]}")
            passed_all = False
    out["decision_v2"] = "принято" if passed_all else "отклонено"
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    dump("e5_oof.json", out)
    print(f"ИТОГ v2 (OOF): {out['decision_v2']} ({out['wall_s']} с)")
    return 0 if passed_all else 1


def step_kr() -> int:
    model = LogisticRanker.load(A3)
    a3_sha1 = sha1(A3)
    print(f"A3 {A3.name} sha1 {a3_sha1}; gt {model.meta.get('gt_tokens_sha1')}")
    assert model.meta.get("gt_tokens_sha1") == GT_SHA1
    assert model.feature_names == NAMES
    strip_wm = load_strip_wm()
    kmeta = meta_of("kr")
    recs = stand("kr")
    by_q = {r["query_id"]: r for r in recs}
    red = redecided("kr")
    main = main_kr(kmeta)
    s1 = s1_slugs()
    aside = [q for q in main if {kmeta[q]["slug"], *(kmeta[q].get("acceptable") or [])} & s1]
    gate = [q for q in main if q not in aside]
    svc = service(model)
    w = weights_of(model)
    cand = {"asis": {}, "wm": {}}
    mismatch = []
    wm_changed = 0
    for q in main:
        rec = by_q[q]
        slug, _ = decide(svc, rec)
        cand["asis"][q] = slug
        if offline(rec, w) != slug:
            mismatch.append(q)
        new, did = strip_wm(rec)
        wm_changed += did
        cand["wm"][q] = decide(svc, new)[0] if did else slug
    out: dict[str, Any] = {
        "a3": str(A3),
        "a3_sha1": a3_sha1,
        "s1_slugs": sorted(s1),
        "aside": [kmeta[q]["photo"] for q in aside],
        "gate_frames": len(gate),
        "wm_changed": wm_changed,
        "offline_mismatch": mismatch,
    }
    passed = not mismatch
    for count in ("asis", "wm"):
        base = red[count]
        s = score(gate, cand[count], base, kmeta)
        total = score(main, cand[count], base, kmeta)
        side = score(aside, cand[count], base, kmeta) if aside else None
        acc = accept_kr(s)
        passed &= all(acc.values())
        out[count] = {
            "gate": s,
            "total_617": total,
            "aside": side,
            "base_gate": score(gate, base, base, kmeta),
            "base_total_617": score(main, base, base, kmeta),
            "accept": acc,
        }
        print(line(f"kr {count}: база (616)", out[count]["base_gate"]))
        print(line(f"kr {count}: A3 (616)", s))
        print(line(f"kr {count}: A3 (617)", total))
        if side:
            print(line(f"kr {count}: S1 отдельно", side))
        print(f"   приёмка {count}: {acc}")
    out["answers"] = cand
    out["decision_kr"] = "принято" if passed else "отклонено"
    dump("e5_kr.json", out)
    print(f"сервис ≠ офлайн: {len(mismatch)}; ИТОГ kr: {out['decision_kr']}")
    return 0 if passed else 1


def step_service() -> int:
    """§8: сервис ветки с A3 = офлайн на всех кадрах стенда; §9: v2 в выборке, время A3."""
    model = LogisticRanker.load(A3)
    svc = service(model)
    w = weights_of(model)
    out: dict[str, Any] = {"a3_sha1": sha1(A3)}
    for set_key, (_, n) in SETS.items():
        recs = stand(set_key)
        eq = sum(decide(svc, r)[0] == offline(r, w) for r in recs)
        out[f"service_eq_offline_{set_key}"] = f"{eq}/{n}"
        print(f"{set_key}: сервис с A3 = офлайн {eq}/{n}")
    meta = meta_of("v2")
    recs = stand("v2")
    base = {r["query_id"]: r["slug"] for r in recs}
    ans = {r["query_id"]: decide(svc, r)[0] for r in recs}
    s = score(list(meta), ans, base, meta)
    out["insample_v2"] = s
    print(line("v2 в выборке (справочно)", s))
    items = field_items(recs, meta, ATTRS, READER)
    times = []
    for it in items:
        t = time.perf_counter()
        order_by_model(model, it.features)
        times.append(1000 * (time.perf_counter() - t))
    out["a3_order_ms"] = {
        "mean": round(float(np.mean(times)), 3),
        "p95": round(float(np.percentile(times, 95)), 3),
    }
    print(f"время порядка A3 на кадр (CPU): {out['a3_order_ms']}")
    assert model.coef_ is not None
    weights = dict(zip(model.feature_names, model.coef_, strict=True))
    out["weights"] = {k: round(float(v), 4) for k, v in weights.items()}
    out["fit_info"] = model.fit_info
    print(
        "веса A3 (|w| > 0,05): "
        + ", ".join(
            f"{k.replace(PREFIX, '')} {v:+.2f}" for k, v in weights.items() if abs(v) > 0.05
        )
    )
    out["gate_open_v2"] = sum(decide(svc, r)[1]["hybrid"] for r in recs)
    dump("e5_service.json", out)
    equal = [v for k, v in out.items() if k.startswith("service_eq")]
    return 0 if all(v.split("/")[0] == v.split("/")[1] for v in equal) else 1


def step_insample() -> int:
    """Справочно (§9): итоговая A3 на всех 353 кадрах v2, которые она видела при обучении."""
    from bench.train_hybrid import main as train_main

    path = HERE / "a3_final_insample.json"
    pred = E5 / "base_v2" / "predictions.jsonl"
    meta_file = FD / "sets" / "catalog_v2" / "meta.jsonl"
    code = train_main(
        ["--field", f"{pred}={meta_file}", "--gt", str(GT), "--goal", str(GOAL),
         "--root", str(SCANNER), "--out", str(path)]
    )  # fmt: skip
    model = LogisticRanker.load(path)
    svc = service(model)
    w = weights_of(model)
    meta = meta_of("v2")
    recs = stand("v2")
    base = {r["query_id"]: r["slug"] for r in recs}
    ans, gate_open, mismatch = {}, 0, 0
    for r in recs:
        slug, ev = decide(svc, r)
        ans[r["query_id"]] = slug
        gate_open += bool(ev["hybrid"])
        mismatch += offline(r, w) != slug
    s = score(list(meta), ans, base, meta)
    print(line("v2 в выборке (справочно)", s))
    assert model.coef_ is not None
    weights = dict(zip(model.feature_names, model.coef_, strict=True))
    out = {
        "model": str(path),
        "model_sha1": sha1(path),
        "train_exit": code,
        "fit_info": model.fit_info,
        "insample_v2": s,
        "gate_open": gate_open,
        "service_ne_offline": mismatch,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "kr": "не открывался: Э5 отклонён по OOF (PREREG §6)",
    }
    print(f"калитка открыта {gate_open}/{len(recs)}; сервис ≠ офлайн {mismatch}")
    print(
        "веса A3 (|w| > 0,05): "
        + ", ".join(
            f"{k.replace(PREFIX, '')} {v:+.2f}" for k, v in weights.items() if abs(v) > 0.05
        )
    )
    dump("e5_insample.json", out)
    return 0


STEPS = {
    "gate": step_gate,
    "oof": step_oof,
    "kr": step_kr,
    "service": step_service,
    "insample": step_insample,
}

if __name__ == "__main__":
    raise SystemExit(STEPS[sys.argv[1]]())
