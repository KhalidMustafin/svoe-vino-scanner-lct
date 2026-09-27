"""Векторы видов запроса студийных пар на CPU — тот же путь, что `acc_plan/retrieval/qemb.py`.

decode_on_backgrounds(байты, [BACKGROUND]) → from_query(кадр, None) → SigLIP2 so400m (float32,
CPU) — как сервис. Запросы: 358 студийных снимков Роскачества (`pairs`, webp) и их «телефонные»
копии (`pairs_phone`) — сохранённые JPEG `runs/pairs-phone-images`, те же байты, что читал OCR
(сервис получил бы именно их). Прежний прогон CV пар (`runs/goal/iter20/cvall-pairs*`) шёл по
индексу до правки эталонов 23.09, поэтому для нынешнего индекса векторы считаются заново.

Пишет `fund/protocol/qemb_pairs.npz`: qids (`pairs:<wine_id>`, `pairs_phone:<wine_id>`), vectors
(N, 4, 1152) float16, names. Контрольная точка каждые 20 кадров; перезапуск продолжает.

    NT=12 PYTHONPATH=. python research/2026-09-26_fund/qemb_pairs.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import protocol as P
from app.features.embedder import SiglipEmbedder
from app.features.views import BACKGROUND, from_query, order_views
from app.normalize.decode import decode_on_backgrounds

PAIRS_JSON = P.ROOT / "Code" / "data" / "raw" / "pairs" / "benchmark.json"
PAIRS_IMG = P.ROOT / "Code" / "data" / "raw" / "pairs" / "roskachestvo"
PHONE_IMG = P.ROOT / "svoe-vino-scanner" / "runs" / "pairs-phone-images"
OUT = P.PROTOCOL / "qemb_pairs.npz"
PART = P.PROTOCOL / "qemb_pairs.partial.npz"


def queries() -> list[tuple[str, Path]]:
    recs = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    out = [(f"pairs:{r['wine_id']}", PAIRS_IMG / f"{r['wine_id']}.webp") for r in recs]
    out += [(f"pairs_phone:{r['wine_id']}", PHONE_IMG / f"{r['wine_id']}.jpg") for r in recs]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    torch.set_num_threads(int(os.environ.get("NT", "12")))
    order = queries()
    n_checked = P.assert_no_test([q for q, _ in order], [p for _, p in order])
    print("страж:", n_checked, "проверено", flush=True)
    qids: list[str] = []
    vecs: list[np.ndarray] = []
    names: list[str] | None = None
    if PART.exists():
        z = np.load(PART)
        qids, vecs, names = [str(q) for q in z["qids"]], list(z["vectors"]), [str(n) for n in z["names"]]
        print("продолжаю с", len(qids), flush=True)
    done = set(qids)

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, qids=np.array(qids), vectors=np.stack(vecs), names=np.array(names))
        os.replace(tmp, path)

    emb = SiglipEmbedder(
        "google/siglip2-so400m-patch14-384", device="cpu", dtype="float32", batch_size=8
    )
    t0 = time.time()
    for q, path in order:
        if q in done:
            continue
        img = decode_on_backgrounds(path.read_bytes(), [BACKGROUND])[0]
        views = from_query(img, None)
        vnames = order_views(views)
        if names is None:
            names = vnames
        assert vnames == names, (q, vnames, names)
        qids.append(q)
        vecs.append(emb.embed([views[n] for n in vnames]).astype(np.float16))
        if len(qids) % 20 == 0:
            save(PART)
            print(len(qids), q, f"{time.time() - t0:.0f} с", flush=True)
    save(PART)
    save(OUT)
    print("готово", len(qids), f"{time.time() - t0:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
