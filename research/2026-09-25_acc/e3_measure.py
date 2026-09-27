"""Э3 строго по `PREREG_E3.md`: ворота кода и замер живых карточек. Только CPU, без Ollama.

Сервис ветки грузится дважды — флагом `SVS_LIVE_CARDS` 0 и 1 (`ScannerService.load`), и каждый
кадр проходит `scan()` целиком. Две вставки, как у стенда `iterbench.py --no-gpu`:
- поиск по картинке — `index._rank` загруженного индекса по CPU-векторам запросов
  (`acc_plan/retrieval/qemb_*.npz`) вместо SigLIP;
- чтение этикетки — из кэша стенда (`runs/field25/iters/cache/reads`, читатель
  `vlm|qwen3.5:4b|f3a017317f04`); промах — сбой кадра, а не вызов Ollama (адрес Ollama — порт 9).
Разбор чтения (`read_label`), resolve и H5 — код сервиса со словарём и gt своего комплекта.

Ворота кода (PREREG, п. 4): на каждом кадре top-20 (slug, score, view, rank), отрыв и ответ H5
кода ветки при выключенном флаге и на индексе с маской «все строки базовые» — бит в бит как у
`app/features/index.py` @7b76247 (модуль берётся из git).

    PYTHONPATH=. python research/2026-09-25_acc/e3_measure.py gate --set catalog_v2
    PYTHONPATH=. python research/2026-09-25_acc/e3_measure.py measure --set krasnostop_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.api.config import CV_PER_SLUG, ServiceSettings  # noqa: E402
from app.api.service import ScannerService, _Run  # noqa: E402
from app.features.contracts import VisualResult  # noqa: E402
from app.features.embedder import unit_rows  # noqa: E402
from app.features.index import VisualIndex, align_pairs  # noqa: E402
from app.reading.contracts import LabelFields, Reading  # noqa: E402
from app.resolve.features import TextRead  # noqa: E402

ROOT = Path(r"<корень>")
DATA = Path(os.environ.get("SVS_DATA_DIR") or ROOT / "svoe-vino-scanner" / "data")
DATASET = Path(
    os.environ.get("SVS_DATASET_DIR")
    or ROOT / "Датасет и подробное задание" / "unpacked" / "Датасет"
)
ITERS = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters"
READS = ITERS / "cache" / "reads"
RUNS = ITERS / "runs"
QEMB = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "retrieval"
FD = ROOT / "field_dataset"
READER_IDENT = "vlm|qwen3.5:4b|f3a017317f04"
BASE_COMMIT = "7b76247"
DUMPS = {"catalog_v2": "pfix_final", "krasnostop_v1": "kr_holdout", "ooc_v2": "ooc_v2prod"}
QFILES = {
    "catalog_v2": "qemb_catalog_v2.npz",
    "krasnostop_v1": "qemb_krasnostop_v1.npz",
    "ooc_v2": "qemb_ooc_v2.npz",
}


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sha1_bytes(*parts: bytes) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ код @7b76247
def base_index_module() -> types.ModuleType:
    """`app/features/index.py` из коммита базы — эталон ворот кода."""
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:app/features/index.py"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    mod = types.ModuleType("index_at_base")
    mod.__file__ = f"{BASE_COMMIT}:app/features/index.py"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)  # noqa: S102 — свой код из git
    return mod


# ------------------------------------------------------------------ водяной знак (wm_whatif.py)
# Правило `strip_wm` из acc_plan/errors_holdout/wm_whatif.py — копией: модуль при импорте
# подключает чужое рабочее дерево (svs-h6).
HOMO = str.maketrans(
    {
        "K": "к", "P": "р", "A": "а", "C": "с", "H": "н", "O": "о", "T": "т", "E": "е",
        "M": "м", "B": "в", "X": "х", "Y": "у", "k": "к", "p": "р", "a": "а", "c": "с",
        "o": "о", "e": "е", "x": "х", "y": "у",
    }
)  # fmt: skip
WM = "красностоп"


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fold(tok: str) -> str:
    t = tok.strip()
    if t.lower() == "krasnostop":
        return WM
    return t.translate(HOMO).lower()


def wm_token(tok: str) -> bool:
    t = re.sub(r"(\.ru|ru)$", "", re.sub(r"[^\w.]", "", tok.strip()), flags=re.I).strip(".")
    if not t:
        return False
    f = _fold(t)
    return len(f) >= 8 and _levenshtein(f, WM) <= 2


def wm_line(line: str) -> bool:
    toks = [x for x in re.split(r"\s+", line.strip()) if x]
    return len(toks) == 1 and wm_token(toks[0])


def has_krasno_token(lines: list[str]) -> bool:
    return any(
        tok and wm_token(tok) for ln in lines for tok in re.split(r"[\s\-+/,;:]+", ln)
    )


def strip_wm(read: TextRead) -> tuple[TextRead, bool]:
    raw = read.raw_text or ""
    lines = raw.splitlines()
    kept = [ln for ln in lines if not wm_line(ln)]
    changed = len(kept) != len(lines)
    if not changed:
        return read, False
    fields = read.fields.model_dump(mode="json") if read.fields is not None else None
    if fields and not has_krasno_token(kept):
        for grp in ("grapes", "cuvee"):
            fields[grp] = [e for e in (fields.get(grp) or []) if e.get("value") != "krasnostop"]
        fields["unmatched"] = [
            e for e in (fields.get("unmatched") or []) if not wm_token(str(e.get("value")))
        ]
    new_fields = LabelFields.model_validate(fields) if fields is not None else None
    return TextRead(fields=new_fields, raw_text="\n".join(kept)), True


# ------------------------------------------------------------------ сервис на записанных входах
class Replay:
    """Сервис ветки с CV по векторам запросов и чтением из кэша стенда."""

    def __init__(self, live: bool) -> None:
        env = {
            "SVS_DATA_DIR": str(DATA),
            "SVS_DATASET_DIR": str(DATASET),
            "SVS_LIVE_CARDS": "1" if live else "0",
            # Порт 9 (discard): промах кэша не может дойти до Ollama на 11434.
            "SVS_OLLAMA_URL": "http://127.0.0.1:9",
            "SVS_DEVICE": "cpu",
            "SVS_BUDGET_MS": "30000",
            "SVS_VLM_TIMEOUT_MS": "15000",
        }
        self.settings = ServiceSettings.from_env(env)
        self.svc = svc = ScannerService.load(self.settings)
        self.Q: np.ndarray | None = None
        self.cap: dict[str, Any] = {}
        self.hits = self.misses = 0
        locked = svc.vlm
        assert locked is not None
        ident = f"{locked.id}|{locked.version}|{locked.params_hash}"
        assert ident == READER_IDENT, ident

        def cached_read(image: np.ndarray, *, crop: str, budget_ms: int) -> Reading:
            key = sha1_bytes(image.tobytes(), str(image.shape).encode(), crop.encode(), ident.encode())
            path = READS / f"{key}.json"
            if not path.exists():
                self.misses += 1
                raise RuntimeError("промах кэша чтений")
            self.hits += 1
            return Reading.model_validate_json(path.read_text(encoding="utf-8"))

        locked.read = cached_read  # type: ignore[method-assign]

        def search(image: np.ndarray, run: _Run) -> VisualResult:
            assert self.Q is not None
            candidates, margin = svc.index._rank(self.Q, self.settings.top_k, CV_PER_SLUG)
            visual = VisualResult(
                candidates=candidates,
                margin=margin,
                timings_ms={"embed": 0.0, "match": 0.0, "total": 0.0},
                model=svc.index.meta.model,
            )
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

    def resolve(self, visual: VisualResult, read: TextRead) -> str:
        """Ответ H5 сервиса на готовых CV и чтении (без разжатия кадра)."""
        t = time.perf_counter()
        run = _Run(time.perf_counter, t, t + 30)
        key = self.svc.reader_key
        return self.svc._resolve(visual, {key: read}, read.fields, run)["slug"]


def cand_tuple(visual_or_cands: Any) -> list[tuple[str, float, str, int]]:
    cands = getattr(visual_or_cands, "candidates", visual_or_cands)
    return [(c.slug, c.score, c.view, c.rank) for c in cands]


def load_queries(set_name: str) -> tuple[list[str], np.ndarray]:
    z = np.load(QEMB / QFILES[set_name])
    return [str(q) for q in z["qids"]], z["vectors"]


def image_path(m: dict[str, Any]) -> Path:
    p = Path(m["image"])
    return p if p.is_absolute() else FD / p


def dump_h5(set_name: str) -> dict[str, str]:
    """Ответ H5 записанного прогона: при p_top1 < 0,5 — `no_cluster+filter` из whatif."""
    base = RUNS / DUMPS[set_name] / "iter20"
    wi = {r["query_id"]: r for r in json.loads((base / "whatif.json").read_text(encoding="utf-8"))}
    out = {}
    for rec in jsonl(base / "predictions.jsonl"):
        q, w = rec["query_id"], wi.get(rec["query_id"])
        p = float(rec.get("p_top1") or 0)
        out[q] = (w.get("no_cluster+filter") or [rec["slug"]])[0] if p < 0.5 and w else rec["slug"]
    return out


def dump_fields(set_name: str) -> dict[str, Any]:
    base = RUNS / DUMPS[set_name] / "iter20"
    return {
        rec["query_id"]: (rec.get("text_read") or {}).get("fields")
        for rec in jsonl(base / "predictions.jsonl")
    }


# ------------------------------------------------------------------ прогон
def run_set(
    set_name: str, *, gate: bool, measure: bool, wm: bool, limit: int | None
) -> dict[str, Any]:
    """`gate` — ворота кода (без ответов при включённом флаге); `measure` — ответы обоих комплектов."""
    logging.getLogger("app").setLevel(logging.WARNING)
    meta = {m["query_id"]: m for m in jsonl(FD / "sets" / set_name / "meta.jsonl")}
    qids, vectors = load_queries(set_name)
    if limit:
        qids, vectors = qids[:limit], vectors[:limit]
    t0 = time.perf_counter()
    off, on = Replay(live=False), Replay(live=True)
    print(f"сервисы: {time.perf_counter() - t0:.1f} с; off {off.settings.index_path.name} "
          f"({len(off.svc.index)} строк), on {on.settings.index_path.name} "
          f"({len(on.svc.index)} строк, базовых {on.svc.index.n_base})", flush=True)
    assert off.svc.index.base_rows is None and on.svc.index.base_rows is not None
    base_slugs = set(off.svc.index.slug_order)
    old = noadd = None
    if gate:
        mod = base_index_module()
        old = mod.VisualIndex.load(off.settings.index_path)
        with np.load(off.settings.index_path, allow_pickle=False) as z:
            noadd = VisualIndex(
                list(off.svc.index.slugs),
                list(off.svc.index.views),
                np.asarray(z["vectors"], dtype=np.float32),
                off.svc.index.meta,
                base_rows=np.ones(len(off.svc.index), dtype=bool),
            )
        assert noadd.base_rows is None
    rows: dict[str, dict[str, Any]] = {}
    bad: list[str] = []
    gate_fail: Counter[str] = Counter()
    full_mask = np.ones(len(off.svc.index), dtype=bool)
    csv_delta_max = 0.0
    for n, (q, qv) in enumerate(zip(qids, vectors, strict=True)):
        Q = unit_rows(qv.astype(np.float32))
        data = image_path(meta[q]).read_bytes()
        r_off, c_off, ok_off = off.scan(data, Q)
        if measure:
            r_on, c_on, ok_on = on.scan(data, Q)
        else:
            ok_on = True
        if not (ok_off and ok_on):
            bad.append(q)
            continue
        read_off = c_off["reads"][off.svc.reader_key]
        row: dict[str, Any] = {
            "off": r_off.slug,
            "p_off": r_off.confidence.top1,
            "cv_off": c_off["visual"].candidates[0].slug,
            "fields_off": read_off.fields.model_dump(mode="json") if read_off.fields else None,
        }
        if measure:
            read_on = c_on["reads"][on.svc.reader_key]
            row.update(
                on=r_on.slug,
                p_on=r_on.confidence.top1,
                cv_on=c_on["visual"].candidates[0].slug,
                fields_same=read_off.fields == read_on.fields,
            )
        if gate:
            assert old is not None and noadd is not None
            cands_old, m_old = old._rank(Q, 20, CV_PER_SLUG)
            cands_na, m_na = noadd._rank(Q, 20, CV_PER_SLUG)
            v_off = c_off["visual"]
            if cand_tuple(cands_old) != cand_tuple(v_off) or m_old != v_off.margin:
                gate_fail["a_top20"] += 1
            if cand_tuple(cands_old) != cand_tuple(cands_na) or m_old != m_na:
                gate_fail["b_top20"] += 1
            v_old = VisualResult(
                candidates=cands_old, margin=m_old, timings_ms={}, model=v_off.model
            )
            a_old = off.resolve(v_old, read_off)
            if a_old != r_off.slug:
                gate_fail["a_h5"] += 1
            v_na = VisualResult(candidates=cands_na, margin=m_na, timings_ms={}, model=v_off.model)
            if off.resolve(v_na, read_off) != a_old:
                gate_fail["b_h5"] += 1
            # (б') путь с маской «все строки» против пути без маски — сами числа align_pairs
            sims = off.svc.index.vectors @ Q.T
            groups = off.svc.index.view_groups
            if not np.array_equal(align_pairs(sims, groups), align_pairs(sims, groups, full_mask)):
                gate_fail["b_masked_path_bits"] += 1
            # справочно: счёт карточек CSV в живом индексе = счёт в базовом (только CV)
            live_cands, _ = on.svc.index._rank(Q, 20, CV_PER_SLUG)
            live_base = [c for c in live_cands if c.slug in base_slugs]
            ref = v_off.candidates[: len(live_base)]
            if [(c.slug, c.view) for c in live_base] != [(c.slug, c.view) for c in ref]:
                gate_fail["info_csv_order"] += 1
            else:
                deltas = [abs(a.score - b.score) for a, b in zip(live_base, ref, strict=True)]
                csv_delta_max = max([csv_delta_max, *deltas])
        if wm and measure:
            wr_off, ch = strip_wm(read_off)
            wr_on, _ = strip_wm(read_on)
            row["wm_changed"] = ch
            row["wm_off"] = off.resolve(c_off["visual"], wr_off) if ch else r_off.slug
            row["wm_on"] = on.resolve(c_on["visual"], wr_on) if ch else r_on.slug
        rows[q] = row
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(qids)} за {time.perf_counter() - t0:.0f} с; сбоев {len(bad)}; "
                  f"ворота {dict(gate_fail)}", flush=True)
    return {
        "set": set_name,
        "frames": len(qids),
        "replayed": len(rows),
        "failed": bad,
        "reads": {"off_hits": off.hits, "off_misses": off.misses, "on_hits": on.hits, "on_misses": on.misses},
        "gate": dict(gate_fail) if gate else None,
        "gate_frames": len(rows) if gate else 0,
        "csv_score_delta_max": csv_delta_max if gate else None,
        "files": {
            "off": {k: str(getattr(off.settings, k)) for k in ("index_path", "attrs_path", "lexicon_path")},
            "on": {k: str(getattr(on.settings, k)) for k in ("index_path", "attrs_path", "lexicon_path")},
            "sha1": {
                p.name: sha1_file(p)
                for p in (
                    off.settings.index_path, off.settings.attrs_path, off.settings.lexicon_path,
                    on.settings.index_path, on.settings.attrs_path, on.settings.lexicon_path,
                )
            },
        },
        "provenance_on": on.svc.provenance,
        "live_slugs": sorted(set(on.svc.index.slug_order) - base_slugs),
        "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


# ------------------------------------------------------------------ счёт
def wine_of() -> dict[str, str]:
    groups = json.loads((FD / "catalog" / "wine_groups_final.json").read_text(encoding="utf-8"))
    return {slug: wid for wid, g in groups.items() for slug in g.get("members") or []}


def same_wine(a: str | None, b: str | None, wines: dict[str, str]) -> bool:
    return a == b or (a in wines and wines.get(a) == wines.get(b))


def score_v2(res: dict[str, Any]) -> dict[str, Any]:
    meta = {m["query_id"]: m for m in jsonl(FD / "sets" / "catalog_v2" / "meta.jsonl")}
    base_gt = {json.loads(x)["slug"] for x in (DATA / "gt" / "gt_tokens.jsonl").open(encoding="utf-8")}
    rows = res["rows"]
    ok = {q: {meta[q]["slug"], *(meta[q].get("acceptable") or [])} for q in rows}
    in_csv = {q for q in rows if ok[q] & base_gt}

    def metrics(key: str) -> dict[str, Any]:
        soft = {q: rows[q][key] in ok[q] for q in rows}
        strict = {q: rows[q][key] == meta[q]["slug"] for q in rows}
        sl, slo = Counter(), Counter()
        byw: dict[str, list[bool]] = defaultdict(list)
        for q in rows:
            sl[meta[q]["photo"][0]] += 1
            slo[meta[q]["photo"][0]] += soft[q]
            byw[meta[q]["slug"]].append(soft[q])
        return {
            "micro": round(100 * np.mean(list(soft.values())), 2),
            "macro": round(100 * np.mean([np.mean(v) for v in byw.values()]), 2),
            "strict": round(100 * np.mean(list(strict.values())), 2),
            "slices": {k: f"{slo[k]}/{sl[k]}" for k in sorted(sl)},
        }

    fix = sorted(q for q in rows if rows[q]["off"] not in ok[q] and rows[q]["on"] in ok[q])
    brk = sorted(q for q in rows if rows[q]["off"] in ok[q] and rows[q]["on"] not in ok[q])
    brk_csv = [q for q in brk if q in in_csv]
    brk_r = [q for q in brk if meta[q]["photo"].startswith("R")]
    accept = not brk_csv and len(fix) >= 4 * len(brk) and not brk_r
    return {
        "frames_gt_in_csv": len(in_csv),
        "fixes": fix,
        "breaks": brk,
        "breaks_gt_in_csv": brk_csv,
        "breaks_R": brk_r,
        "changed": sorted(q for q in rows if rows[q]["off"] != rows[q]["on"]),
        "off": metrics("off"),
        "on": metrics("on"),
        "accept_v2": accept,
    }


def score_kr(res: dict[str, Any]) -> dict[str, Any]:
    meta = {m["query_id"]: m for m in jsonl(FD / "sets" / "krasnostop_v1" / "meta.jsonl")}
    wines = wine_of()
    live = set(res["live_slugs"])
    rows = res["rows"]
    main = [q for q in rows if not meta[q].get("same_packshot")]
    ok = {q: {meta[q]["slug"], *(meta[q].get("acceptable") or [])} for q in rows}
    out: dict[str, Any] = {"main_frames": len(main)}
    for tag, a, b in (("as_is", "off", "on"), ("no_wm", "wm_off", "wm_on")):
        changed = [q for q in main if rows[q][a] != rows[q][b]]
        other = [q for q in changed if not same_wine(rows[q][a], rows[q][b], wines)]
        out[tag] = {
            "changed": changed,
            "changed_other_wine": other,
            "to_live_card": [q for q in other if rows[q][b] in live],
            "fixes": [q for q in main if rows[q][a] not in ok[q] and rows[q][b] in ok[q]],
            "breaks": [q for q in main if rows[q][a] in ok[q] and rows[q][b] not in ok[q]],
            "micro_off": round(100 * np.mean([rows[q][a] in ok[q] for q in main]), 2),
            "micro_on": round(100 * np.mean([rows[q][b] in ok[q] for q in main]), 2),
        }
    n = max(len(out["as_is"]["changed_other_wine"]), len(out["no_wm"]["changed_other_wine"]))
    out["decision_count"] = n
    out["decision"] = "flag_on_default" if n == 0 else "flag_off_default" if n == 1 else "reject"
    rest = [q for q in rows if meta[q].get("same_packshot")]
    out["same_packshot_changed_other_wine"] = [
        q for q in rest if not same_wine(rows[q]["off"], rows[q]["on"], wines)
    ]
    return out


def score_ooc(res: dict[str, Any]) -> dict[str, Any]:
    live = set(res["live_slugs"])
    rows = res["rows"]
    to_live = sorted(q for q in rows if rows[q]["on"] in live)
    return {"frames": len(rows), "to_live_card": len(to_live), "share": round(len(to_live) / max(1, len(rows)), 4),
            "qids": to_live}


def sanity(set_name: str, res: dict[str, Any]) -> dict[str, Any]:
    """Справочно: ответ H5 базового комплекта против дампа (CPU- против GPU-векторов) и поля."""
    h5 = dump_h5(set_name)
    fields = dump_fields(set_name)
    rows = res["rows"]
    diff = sorted(q for q in rows if q in h5 and rows[q]["off"] != h5[q])
    fdiff = sorted(q for q in rows if q in fields and rows[q]["fields_off"] != fields[q])
    return {"h5_same_as_dump": len(rows) - len(diff), "h5_differs": diff,
            "fields_same_as_dump": len(rows) - len(fdiff), "fields_differ": fdiff[:20],
            "fields_same_off_on": sum(bool(r.get("fields_same")) for r in rows.values())}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "measure"])
    ap.add_argument("--set", required=True, choices=sorted(QFILES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    measure = args.mode == "measure"
    # Замер повторяет ворота кода (дёшево) — кроме ooc_v2, где ворот нет.
    gate = not measure or args.set != "ooc_v2"
    res = run_set(
        args.set, gate=gate, measure=measure, wm=args.set == "krasnostop_v1", limit=args.limit
    )
    res["sanity"] = sanity(args.set, res)
    if measure:
        res["score"] = {"catalog_v2": score_v2, "krasnostop_v1": score_kr, "ooc_v2": score_ooc}[
            args.set
        ](res)
    for row in res["rows"].values():  # поля нужны только сверке с дампом — в файл не идут
        row.pop("fields_off", None)
    out = args.out or HERE / f"e3_{args.mode}_{args.set}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: v for k, v in res.items() if k not in ("rows",)}
    print(json.dumps(brief, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
