"""Решение кадров прогона Э4 кодом самого снимка: `ScannerService._resolve` на входах стенда.

Запускается с cwd = снимок (`snap/e4_*`), тем же эталоном, что у прогона. Сервис — только модель
выбора и разметка каталога (как `bench/resolve_equality.resolve_only_service`), CV и чтение берутся
из `predictions.jsonl` прогона. Два ответа на кадр:
- «как есть» — чтение стенда; обязан совпасть со slug стенда на каждом кадре (проверка способа);
- «без водяного знака» — чтение после `strip_wm` (`acc_plan/errors_holdout/wm_whatif.py`, как у
  Э1); кадр, чьё чтение `strip_wm` не меняет, сохраняет ответ «как есть».

    cd <снимок>; python e4_redecide.py <папка прогона> <gt_tokens.jsonl>

Выход — `<папка прогона>/redecide.json`.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SNAP = Path.cwd()
sys.path.insert(0, str(SNAP))

import numpy as np
from rapidfuzz.distance import Levenshtein

from app.api.config import ServiceSettings
from app.api.service import ScannerService, _Run, load_catalog, load_resolve_model
from app.features.contracts import IndexMeta, VisualResult
from app.features.index import VisualIndex
from app.reading.contracts import LabelFields
from app.resolve.features import TextRead

ROOT = Path(r"<корень>")
MODEL = (
    ROOT / "svoe-vino-scanner" / "runs" / "goal" / "iter20" / "resolve" / "models" / "vlm35.json"
)
WM_SCRIPT = ROOT / "svs-logs" / "somm-2409" / "acc_plan" / "errors_holdout" / "wm_whatif.py"


def load_strip_wm() -> Callable[[dict[str, Any]], tuple[dict[str, Any], bool]]:
    """`strip_wm` и помощники из wm_whatif.py как есть — без его заголовка (он грузит svs-h6)."""
    source = WM_SCRIPT.read_text(encoding="utf-8")
    body = (
        "HOMO = str.maketrans" + source.split("HOMO = str.maketrans", 1)[1].split("def main()")[0]
    )
    ns: dict[str, Any] = {"re": re, "json": json, "Levenshtein": Levenshtein}
    exec(compile(body, str(WM_SCRIPT), "exec"), ns)  # noqa: S102 — инструмент плана без правок
    return ns["strip_wm"]


class _NoEmbedder:
    dim = 1

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        raise RuntimeError("решение по записанным входам: CV не пересчитывается")


def service(gt: Path) -> ScannerService:
    settings = ServiceSettings(resolve_model=MODEL)
    _, attrs = load_catalog(gt)
    meta = IndexMeta(model=settings.cv_model, dim=1, views=["bottle"], n_slugs=1, n_vectors=1)
    index = VisualIndex(["__e4__"], ["bottle"], np.ones((1, 1), dtype=np.float32), meta)
    return ScannerService(
        settings,
        index=index,
        embedder=_NoEmbedder(settings.cv_model),
        lexicon=None,
        attrs=attrs,
        model=load_resolve_model(MODEL),
        vlm=None,
    )


def decide(svc: ScannerService, rec: dict[str, Any]) -> str:
    visual = VisualResult.model_validate(rec["visual"])
    tr = rec.get("text_read") or {}
    fields = LabelFields.model_validate(tr["fields"]) if tr.get("fields") else None
    read = TextRead(fields=fields, raw_text=tr.get("raw_text"))
    now = time.perf_counter()
    run = _Run(clock=time.perf_counter, started=now, deadline=now + 3600.0)
    assert svc.reader_key is not None
    return svc._resolve(visual, {svc.reader_key: read}, fields, run)["slug"]


def main() -> int:
    run_dir, gt = Path(sys.argv[1]), Path(sys.argv[2])
    import app

    if not Path(app.__file__).resolve().is_relative_to(SNAP.resolve()):
        raise SystemExit(f"код не из снимка: {app.__file__}")
    strip_wm = load_strip_wm()
    svc = service(gt)
    asis: dict[str, str] = {}
    wm: dict[str, str] = {}
    changed: list[str] = []
    mismatch: list[str] = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    for rec in recs:
        q = rec["query_id"]
        asis[q] = decide(svc, rec)
        if asis[q] != rec["slug"]:
            mismatch.append(q)
        new, did = strip_wm(rec)
        wm[q] = decide(svc, new) if did else asis[q]
        if did:
            changed.append(q)
    out = {
        "snapshot": str(SNAP),
        "gt": str(gt),
        "gt_sha1": svc.attrs.meta.get("sha1"),
        "frames": len(recs),
        "asis_mismatch": mismatch,
        "wm_changed": changed,
        "asis": asis,
        "wm": wm,
    }
    (run_dir / "redecide.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{run_dir.name}: {len(recs)} кадров, решение ≠ стенд {len(mismatch)}, wm {len(changed)}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
