"""Срез R на оригиналах организатора (PREREG_R_originals.md): 65 WebP через весь `scan()` кода дерева.

Тот же способ, что `research/2026-09-25_acc/e3_measure.py` (класс `Replay` берётся из дерева, чей
код меряем): CV — `index._rank` по CPU-векторам запросов `acc_plan/retrieval/qemb_r_orig.npz`,
чтение — кэш стенда (ключи пополнены GPU-окном C), разбор, resolve и H5 — код дерева. Флаг Э3 0 и
1 (1 — справочно). Кадры и query_id — `runs/field25/iters/runs/r_orig/meta.jsonl` (поле `image` —
абсолютный путь к WebP). Только CPU, Ollama не вызывается (адрес — порт 9).

    python goal2_rorig.py --tree <корень кода> --out base_r.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"<корень>")
R_ORIG = ROOT / "svoe-vino-scanner" / "runs" / "field25" / "iters" / "runs" / "r_orig"
QFILE = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "retrieval" / "qemb_r_orig.npz"
LIVE = ROOT / "svs-logs" / "somm-2409" / "gpu_C" / "0925_1934" / "r_orig" / "predictions.jsonl"


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tree = args.tree.resolve()
    sys.path.insert(0, str(tree))
    spec = importlib.util.spec_from_file_location(
        "e3_measure", tree / "research" / "2026-09-25_acc" / "e3_measure.py"
    )
    assert spec is not None and spec.loader is not None
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    import app  # noqa: PLC0415 — после вставки дерева

    assert Path(app.__file__).resolve().is_relative_to(tree), app.__file__
    np = e3.np

    meta = {m["query_id"]: m for m in jsonl(R_ORIG / "meta.jsonl")}
    z = np.load(QFILE)
    qids, vectors = [str(q) for q in z["qids"]], z["vectors"]
    assert set(qids) == set(meta) and len(qids) == 65
    t0 = time.perf_counter()
    off, on = e3.Replay(live=False), e3.Replay(live=True)
    rows: dict[str, dict] = {}
    bad: list[str] = []
    for q, qv in zip(qids, vectors, strict=True):
        Q = e3.unit_rows(qv.astype(np.float32))
        data = Path(meta[q]["image"]).read_bytes()
        r_off, c_off, ok_off = off.scan(data, Q)
        r_on, c_on, ok_on = on.scan(data, Q)
        if not (ok_off and ok_on):
            bad.append(q)
            continue
        read_off = c_off["reads"][off.svc.reader_key]
        rows[q] = {
            "off": r_off.slug,
            "p_off": r_off.confidence.top1,
            "cv_off": c_off["visual"].candidates[0].slug,
            "fields_off": read_off.fields.model_dump(mode="json") if read_off.fields else None,
            "on": r_on.slug,
            "p_on": r_on.confidence.top1,
        }
    live = {r["query_id"]: r.get("predicted_slug") for r in jsonl(LIVE)}
    stand = {r["query_id"]: r for r in jsonl(R_ORIG / "iter20" / "predictions.jsonl")}
    res = {
        "what": "срез R на WebP-оригиналах, scan() кода дерева на записанных входах",
        "tree": str(tree),
        "frames": len(qids),
        "replayed": len(rows),
        "failed": bad,
        "reads": {"off_hits": off.hits, "off_misses": off.misses, "on_hits": on.hits, "on_misses": on.misses},
        "sanity": {
            "off_equals_live_gpu_c": sum(rows[q]["off"] == live.get(q) for q in rows),
            "off_differs_from_live": sorted(q for q in rows if rows[q]["off"] != live.get(q)),
            "fields_equal_stand_dump": sum(
                rows[q]["fields_off"] == (stand[q].get("text_read") or {}).get("fields") for q in rows
            ),
            "fields_differ_from_stand": sorted(
                q for q in rows if rows[q]["fields_off"] != (stand[q].get("text_read") or {}).get("fields")
            ),
        },
        "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
