"""Э2: замер строго по `PREREG_E2_reader_fallback.md` — `ScannerService._resolve` на записанных входах.

Только CPU и дампы: видеокарта, Ollama и SigLIP не нужны. VLM сервиса выключен, вместо него —
поддельный читатель в процессе. Правило, формы пустого чтения и ворота заданы в PREREG, здесь
они только исполняются.

Два шага: ответы считаются отдельно кодом базы и кодом ветки, потом сравниваются.

    # ответы (в одном процессе — один код: `app` импортируется из --code)
    python e2_replay.py answers --code <снимок @7b76247> --out base.json
    python e2_replay.py answers --code <корень ветки>    --out new.json
    # ворота (1) и (2)
    python e2_replay.py compare --base base.json --new new.json --out results.json

Входы:
    runs/field25/iters/runs/pfix_final/iter20  — catalog_v2, 353 кадра
    runs/field25/iters/runs/kr_holdout/iter20  — krasnostop_v1, 903 кадра, из них 617 основных
    field_dataset/sets/{catalog_v2,krasnostop_v1}/meta.jsonl — эталон, slug ∪ acceptable
    SVS_DATA_DIR/gt/gt_tokens.jsonl, configs/resolve/s2so400m-vlm35-goal.json — как у сервиса

Формы входа `_resolve`:
- `asis` — чтение из дампа: `TextRead(fields, raw_text)`; `evidence.vlm` — статус и строки дампа;
- `timeout` (основной счёт), `unavailable`, `error`, `empty` — `_read` сервиса с поддельным
  читателем, вернувшим этот статус без строк;
- `exception` — `read_label` бросил исключение (ветка `vlm_exception` в `_read`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(r"<корень>")
RUNS = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters" / "runs"
FD = ROOT / "field_dataset" / "sets"
SETS = {
    "pfix_final": FD / "catalog_v2" / "meta.jsonl",
    "kr_holdout": FD / "krasnostop_v1" / "meta.jsonl",
}
FORMS = ("timeout", "unavailable", "error", "empty", "exception")
PRIMARY = "timeout"
#: Порог спорного кадра сервиса (`AMBIGUOUS_P_TOP1`) — для офлайн-ответа H5 из дампа.
AMBIGUOUS_P_TOP1 = 0.5


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------ ответы
def answers(code: Path, out: Path, limit: int | None) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["SVS_DEVICE"] = "cpu"
    os.environ["SVS_VLM_TIMEOUT_MS"] = "0"  # OllamaVlmReader не создаётся вовсе
    if not os.environ.get("SVS_DATA_DIR"):
        raise SystemExit("нужен SVS_DATA_DIR: gt_tokens и индекс сервиса")
    sys.path.insert(0, str(code))

    import numpy as np

    import app.api.service as service_module
    from app.api.config import ServiceSettings
    from app.api.service import LockedReader, ScannerService, _Run
    from app.features.contracts import VisualResult
    from app.reading.contracts import LabelFields
    from app.reading.readers.base import make_reading
    from app.resolve.features import TextRead

    assert Path(service_module.__file__).resolve().is_relative_to(code.resolve()), (
        f"app импортирован не из {code}: {service_module.__file__}"
    )
    # Исключение формы `exception` сервис пишет в журнал со стеком: здесь оно ожидаемо.
    logging.getLogger("app.api.service").setLevel(logging.CRITICAL)
    settings = ServiceSettings.from_env()
    svc = ScannerService.load(settings)
    key = svc.reader_key
    assert key == "vlm35", key

    class FailingReader:
        """Читатель без Ollama: каждый вызов — чтение с заданным статусом и без строк."""

        id = "vlm"
        version = settings.vlm_model
        params_hash = "e2-replay"

        def __init__(self, status: str) -> None:
            self.status = status

        def available(self) -> bool:
            return True

        def read(self, image: Any, *, crop: str, budget_ms: int) -> Any:
            return make_reading(
                self, image, params=self.params_hash, crop=crop, status=self.status, elapsed_ms=0
            )

    def new_run() -> Any:
        now = svc._clock()
        return _Run(svc._clock, now, now + settings.budget_ms / 1000)

    # Входы пустых форм не зависят от кадра: чтения нет, поля пусты.
    image = np.full((64, 48, 3), 200, dtype=np.uint8)
    reading_settings = dataclasses.replace(settings, vlm_timeout_ms=5000)
    forms: dict[str, tuple[dict[str, Any], Any, dict[str, Any], list[str]]] = {}
    orig_read_label = service_module.read_label
    for form in FORMS:
        svc.settings = reading_settings
        svc.vlm = LockedReader(
            FailingReader("empty" if form == "exception" else form), svc.gpu_lock, svc._clock
        )
        if form == "exception":

            def boom(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("e2-replay: read_label упал")

            service_module.read_label = boom
        run = new_run()
        try:
            reads, fields = svc._read(image, run)
        finally:
            service_module.read_label = orig_read_label
            svc.settings = settings
            svc.vlm = None
        forms[form] = (reads, fields, dict(run.evidence["vlm"]), list(run.degraded))

    def answer(visual: Any, reads: Any, fields: Any, vlm: Mapping[str, Any], flags: Sequence[str]):
        run = new_run()
        run.evidence["vlm"] = dict(vlm)
        run.flag(*flags)
        out = svc._resolve(visual, reads, fields, run)
        return {
            "slug": out["slug"],
            "confidence": out["confidence"].model_dump(mode="json"),
            "margin": out["margin"],
            "top5": [item.model_dump(mode="json") for item in out["top5"]],
            "outcome": out["outcome"],
            "degraded": list(run.degraded),
            "resolve": run.evidence.get("resolve"),
        }

    result: dict[str, Any] = {
        "code": str(code),
        "service_file": str(service_module.__file__),
        "model": svc.model_name,
        "attrs_sha1": svc.attrs.meta.get("sha1"),
        "forms": {
            f: {"vlm": v, "degraded": d, "fields_none": fl is None, "raw": r[key].raw_text}
            for f, (r, fl, v, d) in forms.items()
        },
        "runs": {},
    }
    t0 = time.perf_counter()
    for name in SETS:
        rows = jsonl(RUNS / name / "iter20" / "predictions.jsonl")
        if limit:
            rows = rows[:limit]
        per: dict[str, Any] = {}
        for rec in rows:
            visual = VisualResult.model_validate(rec["visual"])
            tr = rec.get("text_read") or {}
            fields = LabelFields.model_validate(tr["fields"]) if tr.get("fields") else None
            reads = {key: TextRead(fields=fields, raw_text=tr.get("raw_text"))}
            vlm = {"status": rec["vlm"]["status"], "lines": rec["vlm"]["lines"]}
            row = {"asis": answer(visual, reads, fields, vlm, [])}
            for form, (f_reads, f_fields, f_vlm, f_flags) in forms.items():
                row[form] = answer(visual, f_reads, f_fields, f_vlm, f_flags)
            per[rec["query_id"]] = row
        result["runs"][name] = per
        print(f"{name}: {len(per)} кадров, {time.perf_counter() - t0:.0f} с", flush=True)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ сравнение
def offline_h5(name: str) -> dict[str, str]:
    """Офлайн-ответ H5 дампа, как `Frame.h5` в `research/2026-09-24_h6/measure_h6.py`."""
    base = RUNS / name / "iter20"
    whatif = {
        row["query_id"]: row
        for row in json.loads((base / "whatif.json").read_text(encoding="utf-8"))
    }
    out = {}
    for rec in jsonl(base / "predictions.jsonl"):
        slug = rec["slug"]
        wi = whatif.get(rec["query_id"])
        if float(rec.get("p_top1") or 0.0) < AMBIGUOUS_P_TOP1 and wi is not None:
            alt = wi.get("no_cluster+filter") or []
            slug = alt[0] if alt else slug
        out[rec["query_id"]] = slug
    return out


def score(
    meta: Sequence[Mapping[str, Any]], base: Mapping[str, str], new: Mapping[str, str], sl
) -> dict[str, Any]:
    """«То же вино» и строго, починки и поломки нового против базы — по срезам `sl(meta)`."""
    rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "base": 0,
            "new": 0,
            "base_strict": 0,
            "new_strict": 0,
            "fix": [],
            "brk": [],
        }
    )
    for m in meta:
        q = m["query_id"]
        ok = {m["slug"], *(m.get("acceptable") or [])}
        b, n = base[q], new[q]
        for key in (sl(m), "all"):
            r = rows[key]
            r["n"] += 1
            r["base"] += b in ok
            r["new"] += n in ok
            r["base_strict"] += b == m["slug"]
            r["new_strict"] += n == m["slug"]
            if n in ok and b not in ok:
                r["fix"].append(m["photo"])
            if b in ok and n not in ok:
                r["brk"].append(m["photo"])
    for r in rows.values():
        r["net"] = r["new"] - r["base"]
    return dict(sorted(rows.items()))


def compare(base_path: Path, new_path: Path, out: Path) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "base_code": base["code"],
        "new_code": new["code"],
        "model": [base["model"], new["model"]],
        "attrs_sha1": [base["attrs_sha1"], new["attrs_sha1"]],
        "forms": {"base": base["forms"], "new": new["forms"]},
    }
    passed = True
    for name, meta_path in SETS.items():
        meta_all = [m for m in jsonl(meta_path) if m["query_id"] in base["runs"][name]]
        primary = [m for m in meta_all if not m.get("same_packshot")]
        b, n = base["runs"][name], new["runs"][name]
        h5 = offline_h5(name)

        def eq(qs: Sequence[str], left: Mapping[str, Any], right: Mapping[str, Any]):
            bad = [q for q in qs if left[q] != right[q]]
            return len(qs) - len(bad), bad[:10]

        def body(side: Mapping[str, Any], form: str) -> dict[str, Any]:
            """Ответ `_resolve` без доказательств — то, что уходит в тело predict."""
            return {
                q: {k: v for k, v in row[form].items() if k != "resolve"} for q, row in side.items()
            }

        base_slug = {q: row["asis"]["slug"] for q, row in b.items()}
        base_body, new_body = body(b, "asis"), body(n, "asis")
        gates: dict[str, Any] = {}
        for label, meta in (("primary", primary), ("all", meta_all)):
            qs = [m["query_id"] for m in meta]
            ctrl, ctrl_bad = eq(qs, base_slug, h5)
            same, same_bad = eq(qs, base_body, new_body)
            gates[label] = {
                "frames": len(qs),
                "control_base_eq_offline_h5": ctrl,
                "control_bad": ctrl_bad,
                "asis_new_eq_base": same,
                "asis_bad": same_bad,
            }
            passed &= ctrl == len(qs) and same == len(qs)
        # Контроль: база «как есть» воспроизводит точность дампа (только основные кадры).
        base_asis = {q: b[q]["asis"]["slug"] for q in b}
        gates["base_asis_soft"] = sum(
            base_asis[m["query_id"]] in {m["slug"], *(m.get("acceptable") or [])} for m in primary
        )

        def slice_of(m: Mapping[str, Any]) -> str:
            return m["query_id"][0]

        forms: dict[str, Any] = {}
        for form in FORMS:
            bb = {q: b[q][form]["slug"] for q in b}
            nn = {q: n[q][form]["slug"] for q in n}
            s = score(primary, bb, nn, slice_of)
            cv_top1 = sum(nn[q] == n[q][form]["resolve"]["cv_top1"] for q in nn)
            forms[form] = {
                "slices": s,
                "new_eq_cv_top1": cv_top1,
                "new_null": sum(v is None for v in nn.values()),
                "gate_net_ge_0": all(r["net"] >= 0 for r in s.values()),
            }
            passed &= forms[form]["gate_net_ge_0"] and forms[form]["new_null"] == 0
        report[name] = {"gates": gates, "forms": forms}
    report["passed"] = passed
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print_report(report)
    return 0 if passed else 1


def print_report(r: Mapping[str, Any]) -> None:
    print(
        f"база: {r['base_code']}\nветка: {r['new_code']}\nмодель {r['model']}, gt {r['attrs_sha1']}"
    )
    for name in SETS:
        g = r[name]["gates"]
        for label in ("primary", "all"):
            x = g[label]
            print(
                f"{name} {label}: кадров {x['frames']}; база = офлайн H5 {x['control_base_eq_offline_h5']}"
                f"; ветка = база (как есть, весь ответ) {x['asis_new_eq_base']}"
                + (
                    f"; расхождения {x['control_bad']} {x['asis_bad']}"
                    if x["control_bad"] or x["asis_bad"]
                    else ""
                )
            )
        print(f"{name}: база как есть, «то же вино» на основных — {g['base_asis_soft']}")
        for form, f in r[name]["forms"].items():
            parts = []
            for key, s in f["slices"].items():
                parts.append(
                    f"{key} {s['base']}→{s['new']}/{s['n']} (нетто {s['net']:+d}, +{len(s['fix'])}/−{len(s['brk'])}"
                    f", строго {s['base_strict']}→{s['new_strict']})"
                )
            mark = "PRIMARY " if form == PRIMARY else ""
            print(
                f"  {mark}{form}: "
                + "; ".join(parts)
                + f"; ответ = CV top-1 {f['new_eq_cv_top1']}, null {f['new_null']}"
            )
    print("ПРИНЯТО (1)+(2)" if r["passed"] else "НЕ ПРИНЯТО")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("answers")
    a.add_argument("--code", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--limit", type=int)
    c = sub.add_parser("compare")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--new", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "answers":
        return answers(args.code, args.out, args.limit)
    return compare(args.base, args.new, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
