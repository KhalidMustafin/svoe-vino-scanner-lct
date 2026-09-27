"""Э7, R по WebP-оригиналам (PREREG.md, §3): повтор по записанным входам кодом снимка.

Прогон 65 кадров среза R по оригиналам снимается на машине с видеокартой
(`runs/field25/iters/runs/r_orig`, манифест и gt — `gpu_A/0925_1720/r_webp/`). Здесь он не повторяется: строки читателя каждого кадра
разбираются кодом снимка тем же `read_label`, что в сервисе (читатель подменён записанным чтением),
а кадр решает `ScannerService._resolve` снимка на записанном CV — как `e7_redecide.py`. Кадр со
сбоем читателя решает запасной ответ Э2 (CV top-1) — одинаково у базы и пакета.

Запуск с cwd = снимок (`snap/e7_base`, `snap/e7_pkg`), эталон — тот же, что у стенда:

    cd <снимок>; python e7_rorig.py <папка прогона r_orig> <gt_tokens.jsonl> <выход.json>

Проверка способа: у базы поля разбора = записанным, а ответ = записанному slug прогона (если
прогон снят кодом `after-search`). Счёт — мягко по разметке v2 (id кадров R совпадают с v2).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

SNAP = Path.cwd()
sys.path.insert(0, str(SNAP))

import numpy as np

from app.api.config import ServiceSettings
from app.api.service import (
    ScannerService,
    _Run,
    load_catalog,
    load_lexicon,
    load_resolve_model,
    text_read_of,
)
from app.features.contracts import IndexMeta, VisualResult
from app.features.index import VisualIndex
from app.reading.contracts import Reading, TextLine
from app.reading.pipeline import read_label

ROOT = Path(r"<корень>")
MODEL = (
    ROOT / "svoe-vino-scanner" / "runs" / "goal" / "iter20" / "resolve" / "models" / "vlm35.json"
)
LEXICON = ROOT / "svoe-vino-scanner" / "data" / "index" / "lexicon.json"
META = ROOT / "field_dataset" / "sets" / "catalog_v2" / "meta.jsonl"
COMPARED = ("producer", "cuvee", "grapes", "sugar", "vintage", "serial", "abv", "color")


class _NoEmbedder:
    dim = 1

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        raise RuntimeError("решение по записанным входам: CV не пересчитывается")


class _Recorded:
    """Читатель, который возвращает записанное чтение кадра."""

    id = "vlm"

    def __init__(self, reading: Reading) -> None:
        self.reading = reading
        self.version = reading.version
        self.params_hash = reading.params_hash

    def read(self, image: np.ndarray, *, crop: str, budget_ms: int) -> Reading:
        return self.reading


def parse_key(key: str) -> dict[str, Any]:
    head, params, image, crop, px = key.rsplit("|", 4)
    reader, _, version = head.partition("@")
    return {
        "reader": reader,
        "version": version,
        "params_hash": params,
        "image_sha1": image,
        "crop": crop,
        "crop_px": int(px),
    }


def first_source(fields: dict[str, Any] | None) -> str | None:
    for name in (*COMPARED, "unmatched"):
        value = (fields or {}).get(name)
        items = value if isinstance(value, list) else [value] if value else []
        for item in items:
            if item.get("sources"):
                return str(item["sources"][0])
    return None


def service(gt: Path) -> ScannerService:
    settings = ServiceSettings(resolve_model=MODEL)
    _, attrs = load_catalog(gt)
    meta = IndexMeta(model=settings.cv_model, dim=1, views=["bottle"], n_slugs=1, n_vectors=1)
    index = VisualIndex(["__e7__"], ["bottle"], np.ones((1, 1), dtype=np.float32), meta)
    return ScannerService(
        settings,
        index=index,
        embedder=_NoEmbedder(settings.cv_model),
        lexicon=load_lexicon(LEXICON),
        attrs=attrs,
        model=load_resolve_model(MODEL),
        vlm=None,
    )


def decide(svc: ScannerService, rec: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    visual = VisualResult.model_validate(rec["visual"])
    vlm = rec.get("vlm") or {}
    now = time.perf_counter()
    run = _Run(clock=time.perf_counter, started=now, deadline=now + 3600.0)
    run.evidence["vlm"] = {"status": vlm.get("status"), "lines": vlm.get("lines") or []}
    assert svc.reader_key is not None
    if vlm.get("status") != "ok" or not vlm.get("lines"):
        return svc._resolve(visual, {}, None, run)["slug"], None
    recorded = (rec.get("text_read") or {}).get("fields")
    key = first_source(recorded) or "vlm@qwen3.5:4b|f3a017317f04|rorig|full|1024"
    reading = Reading(
        **parse_key(key),
        lines=[TextLine(id=i, text=text) for i, text in enumerate(vlm["lines"])],
        elapsed_ms=0,
    )
    result = read_label(
        np.zeros((8, 8, 3), dtype=np.uint8),
        readers=[_Recorded(reading)],
        lexicon=svc.lexicon,
        crop="full",
        budget_ms=3_600_000,
    )
    read = text_read_of(result)
    fields = read.fields.model_dump(mode="json") if read.fields is not None else None
    return svc._resolve(visual, {svc.reader_key: read}, read.fields, run)["slug"], fields


def main() -> int:
    run_dir, gt, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    import app

    if not Path(app.__file__).resolve().is_relative_to(SNAP.resolve()):
        raise SystemExit(f"код не из снимка: {app.__file__}")
    svc = service(gt)
    meta = {}
    with META.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                meta[row["query_id"]] = row
    preds = sorted(run_dir.rglob("predictions.jsonl"))
    if len(preds) != 1:
        raise SystemExit(f"в {run_dir} не один predictions.jsonl: {preds}")
    with preds[0].open(encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    answers: dict[str, str] = {}
    fields_eq: list[str] = []
    fields_ne: list[str] = []
    slug_ne: list[str] = []
    ok = 0
    for rec in recs:
        q = rec["query_id"]
        answers[q], fields = decide(svc, rec)
        recorded = (rec.get("text_read") or {}).get("fields")
        if fields is not None and recorded is not None:
            same = all(fields.get(k) == recorded.get(k) for k in COMPARED)
            (fields_eq if same else fields_ne).append(q)
        if answers[q] != rec.get("slug"):
            slug_ne.append(q)
        m = meta.get(q)
        if m is not None:
            ok += answers[q] in {m["slug"], *(m.get("acceptable") or [])}
    out = {
        "snapshot": str(SNAP),
        "run": str(preds[0]),
        "frames": len(recs),
        "soft_ok": ok,
        "fields_eq_recorded": len(fields_eq),
        "fields_ne_recorded": fields_ne,
        "answer_ne_recorded": slug_ne,
        "answers": answers,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"{SNAP.name}: {len(recs)} кадров, мягко {ok}; поля = записанным {len(fields_eq)}, "
        f"≠ {len(fields_ne)}; ответ ≠ записанному {len(slug_ne)} {slug_ne[:8]}"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
