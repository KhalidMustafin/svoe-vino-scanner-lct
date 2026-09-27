"""Э8 по `PREREG.md` §4–6: прогон сервиса ветки на базе и кандидатах, ворота М4–М5, счёт. Только CPU.

Сервис грузится несколькими комплектами (`SVS_INDEX_PATH`, флаг `SVS_LIVE_CARDS`), каждый кадр
проходит `scan()` целиком со вставками Э3 (`research/2026-09-25_acc/e3_measure.py`): поиск —
`index._rank` по CPU-векторам запросов (`acc_plan/retrieval/qemb_*.npz`), чтение — из кэша стенда
(промах — сбой кадра, Ollama не вызывается: адрес — порт 9). Разбор, resolve и H5 — код ветки.

Комплекты: `base` (общий индекс), `reorder` (ворота М5), `e8` (кандидат); с флагом Э3 —
`base_on` и `e8_on` (живой кандидат, справочно, только v2).

    PYTHONPATH=. python research/2026-09-26_e8/e8_measure.py run --set catalog_v2 --out <каталог>
    PYTHONPATH=. python research/2026-09-26_e8/e8_measure.py run --set krasnostop_v1 --out <каталог>
    PYTHONPATH=. python research/2026-09-26_e8/e8_measure.py run --set ooc_v2 --out <каталог>
    PYTHONPATH=. python research/2026-09-26_e8/e8_measure.py gates --out <каталог>   # без меток
    PYTHONPATH=. python research/2026-09-26_e8/e8_measure.py score --out <каталог>   # просмотр kr

`run` пишет только ответы (без меток); `gates` сверяет ответы между комплектами и с записанными
ответами сборки; метки kr читает только `score`, и только после прохода ворот.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research" / "2026-09-25_acc"))

import e3_measure as e3  # noqa: E402  (ставит CUDA_VISIBLE_DEVICES=-1 и HF offline до импорта app)
import numpy as np  # noqa: E402

from app.api.config import CV_PER_SLUG, ServiceSettings  # noqa: E402
from app.api.service import ScannerService, _Run  # noqa: E402
from app.features.contracts import VisualResult  # noqa: E402
from app.features.embedder import unit_rows  # noqa: E402
from app.reading.contracts import Reading  # noqa: E402

E8_DIR = e3.ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "e8"
CAND = E8_DIR / "visual-s2so400m.e8.npz"
CAND_LIVE = E8_DIR / "visual-s2so400m-live71.e8.npz"
REORDER = E8_DIR / "visual-s2so400m.reorder.npz"
#: записанные ответы сборки 030c62e (сводка 25.09, вне репозитория) — ворота М4
REF_DIR = Path(
    r"<tmp>\goal26"
)
CARDS = [
    "abrau-dyurso-abrau-kupazh-tyomnyy-suhoe-kaberne-sovinon-krasnoe-13",
    "derbent-vino-endemy-saperavi-krasnoe-suhoe-13",
    "derbent-vino-endemy-shardone-beloe-suhoe-13",
    "legato-legato-sovinon-blan-beloe-suhoe-125",
    "skalistyy-bereg-shyopot-tsvetov-risling-beloe-suhoe-109",
    "sober-bash-krasnostop-krasnostop-zolotovskiy-krasnoe-suhoe-14",
    "vibes-vermentino-viognier-barrel-fermented-2022",
    "vinodelnya-vedernikov-gubernatorskoe-golubok-krasnoe-suhoe-11",
]
#: PREREG §5: основные кадры kr вин 8 карточек и K0225 (Э4 S1) — вне ворот, отдельной строкой
KR_EXCLUDED = [
    "K0007", "K0320", "K0321", "K0342", "K0570", "K0592", "K0693", "K0704", "K0780", "K0853", "K0855",
]  # fmt: skip
KR_E4_S1 = "K0225-orig-0"
SETS = {
    "catalog_v2": ["base", "reorder", "e8", "base_on", "e8_on"],
    "krasnostop_v1": ["base", "reorder", "e8"],
    "ooc_v2": ["base", "e8"],
}
KITS = {
    "base": (False, None),
    "reorder": (False, REORDER),
    "e8": (False, CAND),
    "base_on": (True, None),
    "e8_on": (True, CAND_LIVE),
}


class Replay:
    """`e3_measure.Replay` с выбором индекса: CV по векторам запросов, чтения из кэша стенда."""

    def __init__(self, live: bool, index: Path | None) -> None:
        env = {
            "SVS_DATA_DIR": str(e3.DATA),
            "SVS_DATASET_DIR": str(e3.DATASET),
            "SVS_LIVE_CARDS": "1" if live else "0",
            "SVS_OLLAMA_URL": "http://127.0.0.1:9",
            "SVS_DEVICE": "cpu",
            "SVS_BUDGET_MS": "30000",
            "SVS_VLM_TIMEOUT_MS": "15000",
        }
        if index is not None:
            env["SVS_INDEX_PATH"] = str(index)
        self.settings = ServiceSettings.from_env(env)
        self.svc = svc = ScannerService.load(self.settings)
        self.Q: np.ndarray | None = None
        self.cap: dict[str, Any] = {}
        self.hits = self.misses = 0
        locked = svc.vlm
        assert locked is not None
        ident = f"{locked.id}|{locked.version}|{locked.params_hash}"
        assert ident == e3.READER_IDENT, ident

        def cached_read(image: np.ndarray, *, crop: str, budget_ms: int) -> Reading:
            key = e3.sha1_bytes(image.tobytes(), str(image.shape).encode(), crop.encode(), ident.encode())
            path = e3.READS / f"{key}.json"
            if not path.exists():
                self.misses += 1
                raise RuntimeError("промах кэша чтений")
            self.hits += 1
            return Reading.model_validate_json(path.read_text(encoding="utf-8"))

        locked.read = cached_read  # type: ignore[method-assign]

        def search(image: np.ndarray, run: _Run) -> VisualResult:
            assert self.Q is not None
            candidates, margin = svc.index._rank(self.Q, self.settings.top_k, CV_PER_SLUG)
            visual = VisualResult(candidates=candidates, margin=margin,
                                  timings_ms={"embed": 0.0, "match": 0.0, "total": 0.0},
                                  model=svc.index.meta.model)
            self.cap["visual"] = visual
            return visual

        svc._search = search  # type: ignore[method-assign]
        orig_read = svc._read

        def spy_read(image: np.ndarray, run: _Run) -> Any:
            reads, fields = orig_read(image, run)
            self.cap["reads"], self.cap["fields"] = reads, fields
            return reads, fields

        svc._read = spy_read  # type: ignore[method-assign]

    def scan(self, data: bytes, Q: np.ndarray) -> tuple[Any, dict[str, Any], bool]:
        self.Q = Q
        self.cap = {}
        before = self.misses
        result = self.svc.scan(data)
        ok = self.misses == before and result.outcome != "error" and "reads" in self.cap
        return result, dict(self.cap), ok

    def resolve(self, visual: VisualResult, read: Any) -> str:
        t = time.perf_counter()
        run = _Run(time.perf_counter, t, t + 30)
        return self.svc._resolve(visual, {self.svc.reader_key: read}, read.fields, run)["slug"]


# ------------------------------------------------------------------ прогон (без меток)
def run(set_name: str, out: Path, limit: int | None) -> int:
    logging.getLogger("app").setLevel(logging.WARNING)
    meta = {m["query_id"]: m for m in e3.jsonl(e3.FD / "sets" / set_name / "meta.jsonl")}
    qids, vectors = e3.load_queries(set_name)
    if limit:
        qids, vectors = qids[:limit], vectors[:limit]
    t0 = time.perf_counter()
    kits = {k: Replay(*KITS[k]) for k in SETS[set_name]}
    files = {k: {"index": str(r.settings.index_path), "sha1": e3.sha1_file(r.settings.index_path),
                 "rows": len(r.svc.index), "base_rows": r.svc.index.n_base} for k, r in kits.items()}
    print(f"комплекты за {time.perf_counter() - t0:.1f} с: "
          + ", ".join(f"{k} {Path(f['index']).name} {f['sha1'][:8]} ({f['rows']})" for k, f in files.items()),
          flush=True)
    wm = set_name == "krasnostop_v1"
    rows: dict[str, dict[str, Any]] = {}
    bad: list[str] = []
    for n, (q, qv) in enumerate(zip(qids, vectors, strict=True)):
        Q = unit_rows(qv.astype(np.float32))
        data = e3.image_path(meta[q]).read_bytes()
        row: dict[str, Any] = {}
        ok_all = True
        for k, r in kits.items():
            res, cap, ok = r.scan(data, Q)
            if not ok:
                ok_all = False
                break
            cands = cap["visual"].candidates
            row[k] = res.slug
            row[f"p_{k}"] = res.confidence.top1
            row[f"cv_{k}"] = [[c.slug, round(c.score, 6)] for c in cands]
            if wm:
                read = cap["reads"][r.svc.reader_key]
                wr, ch = e3.strip_wm(read)
                row["wm_changed"] = ch
                row[f"wm_{k}"] = r.resolve(cap["visual"], wr) if ch else res.slug
        if not ok_all:
            bad.append(q)
            continue
        rows[q] = row
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(qids)} за {time.perf_counter() - t0:.0f} с; сбоев {len(bad)}", flush=True)
    res = {
        "set": set_name, "frames": len(qids), "replayed": len(rows), "failed": bad,
        "reads": {k: {"hits": r.hits, "misses": r.misses} for k, r in kits.items()},
        "files": files, "rows": rows, "wall_s": round(time.perf_counter() - t0, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"run_{set_name}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False)[:3000])
    return 0 if not bad else 1


def load(out: Path, set_name: str) -> dict[str, Any]:
    return json.loads((out / f"run_{set_name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ ворота М4–М5 (без меток)
def kr_gate_frames(rows: dict[str, Any]) -> list[str]:
    """PREREG §5: 617 основных без 11 кадров вин 8 карточек и без K0225 — 605 кадров."""
    kmeta = {m["query_id"]: m for m in e3.jsonl(e3.FD / "sets" / "krasnostop_v1" / "meta.jsonl")}
    primary = sorted(q for q in rows if not kmeta[q].get("same_packshot"))
    excl = {f"{k}-orig-0" for k in KR_EXCLUDED}
    assert excl <= set(primary), sorted(excl - set(primary))
    return [q for q in primary if q not in excl and q != KR_E4_S1]


def gates(out: Path) -> dict[str, Any]:
    v2, kr = load(out, "catalog_v2"), load(out, "krasnostop_v1")
    ref_v2 = json.loads((REF_DIR / "final_v2.json").read_text(encoding="utf-8"))["rows"]
    ref_kr = json.loads((REF_DIR / "final_kr.json").read_text(encoding="utf-8"))["rows"]
    g: dict[str, Any] = {}

    def diff(rows: dict[str, Any], a: str, ref: dict[str, Any] | None, b: str) -> list[str]:
        return sorted(q for q in rows if rows[q][a] != (ref[q][b] if ref is not None else rows[q][b]))

    whole = {
        name: {"frames": r["frames"], "replayed": r["replayed"], "failed": r["failed"],
               "misses": sum(x["misses"] for x in r["reads"].values())}
        for name, r in (("catalog_v2", v2), ("krasnostop_v1", kr))
    }
    g["whole_runs"] = whole
    g["M4_stand"] = {
        "v2_off_vs_030c62e": diff(v2["rows"], "base", ref_v2, "off"),
        "v2_on_vs_030c62e": diff(v2["rows"], "base_on", ref_v2, "on"),
        "kr_off_vs_030c62e": diff(kr["rows"], "base", ref_kr, "off"),
        "kr_wm_off_vs_030c62e": diff(kr["rows"], "wm_base", ref_kr, "wm_off"),
        "ref_frames": [len(ref_v2), len(ref_kr)],
    }
    g["M5_reorder"] = {
        "v2": diff(v2["rows"], "reorder", None, "base"),
        "kr": diff(kr["rows"], "reorder", None, "base"),
        "kr_wm": diff(kr["rows"], "wm_reorder", None, "wm_base"),
    }
    # PREREG_addendum.md: М5 засчитывается на кадрах, которые решают (v2 353, kr 605 ворот);
    # членство в воротах — по `same_packshot` и списку исключений, без меток.
    gate_q = set(kr_gate_frames(kr["rows"]))
    g["M5_on_gate_sets"] = {
        "kr_gate_frames": len(gate_q),
        "v2": g["M5_reorder"]["v2"],
        "kr": [q for q in g["M5_reorder"]["kr"] if q in gate_q],
        "kr_wm": [q for q in g["M5_reorder"]["kr_wm"] if q in gate_q],
    }
    g["M4_passed"] = (
        not any(g["M4_stand"][k] for k in g["M4_stand"] if k != "ref_frames")
        and len(ref_v2) == 353 and len(ref_kr) == 903
        and all(w["replayed"] == w["frames"] and not w["misses"] for w in whole.values())
        and v2["frames"] == 353 and kr["frames"] == 903
    )
    g["M5_passed"] = not any(g["M5_reorder"].values())
    g["passed"] = g["M4_passed"] and g["M5_passed"]
    g["M5_on_gate_sets_passed"] = not any(
        g["M5_on_gate_sets"][k] for k in ("v2", "kr", "kr_wm")
    ) and len(gate_q) == 605
    g["passed_addendum"] = g["M4_passed"] and g["M5_on_gate_sets_passed"]
    (out / "gates.json").write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(g, ensure_ascii=False, indent=1))
    return g


# ------------------------------------------------------------------ счёт (просмотр kr)
def cv_info(rows: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    """Справочно: сдвиг счёта CV у slug вне 8 карточек и новые векторы в CV top-1."""
    cards = set(CARDS)
    shift = 0.0
    top1_to_card, top1_from_card = [], []
    for q, r in rows.items():
        sa = {s: x for s, x in r[f"cv_{a}"]}
        sb = {s: x for s, x in r[f"cv_{b}"]}
        for s in sa.keys() & sb.keys():
            if s not in cards:
                shift = max(shift, abs(sa[s] - sb[s]))
        ta, tb = r[f"cv_{a}"][0][0], r[f"cv_{b}"][0][0]
        if tb in cards and ta != tb:
            top1_to_card.append(q)
        if ta in cards and ta != tb:
            top1_from_card.append(q)
    return {"max_abs_score_shift_other_slugs": round(shift, 6), "cv_top1_became_card": sorted(top1_to_card),
            "cv_top1_left_card": sorted(top1_from_card),
            "cv_top1_changed": sorted(q for q, r in rows.items() if r[f"cv_{a}"][0][0] != r[f"cv_{b}"][0][0])}


def v2_score(rows: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    meta = {m["query_id"]: m for m in e3.jsonl(e3.FD / "sets" / "catalog_v2" / "meta.jsonl")}
    wine = e3.wine_of()

    def metrics(key: str) -> dict[str, Any]:
        soft = {q: rows[q][key] in {meta[q]["slug"], *(meta[q].get("acceptable") or [])} for q in rows}
        strict = {q: rows[q][key] == meta[q]["slug"] for q in rows}
        n_s, s_ok, st_ok = Counter(), Counter(), Counter()
        byw: dict[str, list[float]] = defaultdict(list)
        for q in rows:
            s = meta[q]["photo"][0]
            n_s[s] += 1
            s_ok[s] += soft[q]
            st_ok[s] += strict[q]
            byw[wine.get(meta[q]["slug"], meta[q]["slug"])].append(float(soft[q]))
        return {"soft": sum(soft.values()), "soft_micro": round(100 * sum(soft.values()) / len(rows), 2),
                "soft_macro": round(100 * float(np.mean([np.mean(v) for v in byw.values()])), 2),
                "strict": sum(strict.values()), "strict_micro": round(100 * sum(strict.values()) / len(rows), 2),
                "slices_soft": {k: s_ok[k] for k in "RML"}, "slices_strict": {k: st_ok[k] for k in "RML"},
                "slices_n": {k: n_s[k] for k in "RML"}, "_soft": soft}

    mb, mf = metrics(a), metrics(b)
    fixes = sorted(q for q in rows if not mb["_soft"][q] and mf["_soft"][q])
    breaks = sorted(q for q in rows if mb["_soft"][q] and not mf["_soft"][q])
    strict_fix = sorted(q for q in rows if rows[q][a] != meta[q]["slug"] and rows[q][b] == meta[q]["slug"])
    strict_brk = sorted(q for q in rows if rows[q][a] == meta[q]["slug"] and rows[q][b] != meta[q]["slug"])
    return {
        "base": {k: v for k, v in mb.items() if not k.startswith("_")},
        "candidate": {k: v for k, v in mf.items() if not k.startswith("_")},
        "answers_changed": [[q, rows[q][a], rows[q][b]] for q in sorted(rows) if rows[q][a] != rows[q][b]],
        "fixes": fixes, "breaks": breaks, "strict_fixes": strict_fix, "strict_breaks": strict_brk,
        "breaks_R": [q for q in breaks if q.startswith("R")],
        "slices_not_fewer": all(mf["slices_soft"][k] >= mb["slices_soft"][k] for k in "RML"),
        "strict_not_lower": mf["strict"] >= mb["strict"],
    }


def score(out: Path) -> int:
    g = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    if not g["passed_addendum"]:
        print("ворота способа не пройдены — счёта нет (PREREG §4, PREREG_addendum.md)")
        return 2
    res: dict[str, Any] = {"what": "Э8 — замер по research/2026-09-26_e8/PREREG.md", "gates": g}
    v2 = load(out, "catalog_v2")
    res["catalog_v2"] = {"flag_off": v2_score(v2["rows"], "base", "e8"),
                         "flag_on_reference": v2_score(v2["rows"], "base_on", "e8_on"),
                         "cv": cv_info(v2["rows"], "base", "e8")}
    off = res["catalog_v2"]["flag_off"]
    acc_v2 = {
        "fix>=4*break": len(off["fixes"]) >= 4 * len(off["breaks"]),
        "fix>=1": len(off["fixes"]) >= 1,
        "no_break_R": not off["breaks_R"],
        "slices_R_M_L_not_fewer": off["slices_not_fewer"],
        "strict_not_lower": off["strict_not_lower"],
    }
    # kr — единственный просмотр меток
    kr = load(out, "krasnostop_v1")
    kmeta = {m["query_id"]: m for m in e3.jsonl(e3.FD / "sets" / "krasnostop_v1" / "meta.jsonl")}
    rows = kr["rows"]
    primary = sorted(q for q in rows if not kmeta[q].get("same_packshot"))
    excl = {f"{k}-orig-0" for k in KR_EXCLUDED}
    gate_q = kr_gate_frames(rows)
    assert len(gate_q) == 605

    def ok(q: str, ans: str) -> bool:
        return ans in {kmeta[q]["slug"], *(kmeta[q].get("acceptable") or [])}

    def count(qs: list[str], a: str, b: str) -> dict[str, Any]:
        fixes = [q for q in qs if not ok(q, rows[q][a]) and ok(q, rows[q][b])]
        breaks = [q for q in qs if ok(q, rows[q][a]) and not ok(q, rows[q][b])]
        return {"frames": len(qs), "base": sum(ok(q, rows[q][a]) for q in qs),
                "candidate": sum(ok(q, rows[q][b]) for q in qs),
                "strict_base": sum(rows[q][a] == kmeta[q]["slug"] for q in qs),
                "strict_candidate": sum(rows[q][b] == kmeta[q]["slug"] for q in qs),
                "answers_changed": [[q, rows[q][a], rows[q][b]] for q in qs if rows[q][a] != rows[q][b]],
                "fixes": fixes, "breaks": breaks, "net": len(fixes) - len(breaks)}

    res["krasnostop_v1"] = {
        "gate_605": {"as_is": count(gate_q, "base", "e8"), "no_watermark": count(gate_q, "wm_base", "wm_e8")},
        "excluded_e8_wines": {"as_is": count(sorted(excl), "base", "e8"),
                              "no_watermark": count(sorted(excl), "wm_base", "wm_e8")},
        "K0225": {"as_is": count([KR_E4_S1], "base", "e8"), "no_watermark": count([KR_E4_S1], "wm_base", "wm_e8")},
        "primary_617_reference": {"as_is": count(primary, "base", "e8"),
                                  "no_watermark": count(primary, "wm_base", "wm_e8")},
        "same_packshot_changed": sorted(q for q in rows if kmeta[q].get("same_packshot") and rows[q]["base"] != rows[q]["e8"]),
        "cv": cv_info({q: rows[q] for q in gate_q}, "base", "e8"),
    }
    gk = res["krasnostop_v1"]["gate_605"]
    acc_kr = {"net_as_is>=0": gk["as_is"]["net"] >= 0, "net_no_wm>=0": gk["no_watermark"]["net"] >= 0}
    ooc = load(out, "ooc_v2")
    changed = sorted(q for q, r in ooc["rows"].items() if r["base"] != r["e8"])
    res["ooc_v2"] = {"frames": ooc["frames"], "replayed": ooc["replayed"], "failed": ooc["failed"],
                     "answers_changed": [[q, ooc["rows"][q]["base"], ooc["rows"][q]["e8"]] for q in changed],
                     "to_e8_card": [q for q in changed if ooc["rows"][q]["e8"] in CARDS],
                     "cv": cv_info(ooc["rows"], "base", "e8")}
    res["acceptance"] = {"v2": acc_v2, "kr": acc_kr}
    res["accepted"] = all(acc_v2.values()) and all(acc_kr.values())
    res["decision"] = "принят" if res["accepted"] else "отклонён"
    (out / "e8_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    brief = {
        "v2": {k: off[k] for k in ("fixes", "breaks", "strict_fixes", "strict_breaks", "answers_changed")}
        | {"base": {k: off["base"][k] for k in ("soft", "strict", "slices_soft")},
           "candidate": {k: off["candidate"][k] for k in ("soft", "strict", "slices_soft")}},
        "v2_flag_on": {k: res["catalog_v2"]["flag_on_reference"][k] for k in ("fixes", "breaks")},
        "kr_gate": {t: {k: gk[t][k] for k in ("frames", "base", "candidate", "fixes", "breaks", "net")} for t in gk},
        "kr_excluded": {t: {k: res["krasnostop_v1"]["excluded_e8_wines"][t][k] for k in ("base", "candidate", "fixes", "breaks")}
                        for t in ("as_is", "no_watermark")},
        "ooc_changed": len(changed),
        "acceptance": res["acceptance"], "decision": res["decision"],
    }
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "gates", "score"])
    ap.add_argument("--set", choices=sorted(SETS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.mode == "run":
        return run(args.set, args.out, args.limit)
    if args.mode == "gates":
        return 0 if gates(args.out)["passed_addendum"] else 1
    return score(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
