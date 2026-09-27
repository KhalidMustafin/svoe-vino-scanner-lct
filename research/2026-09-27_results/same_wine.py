"""Доля совпадений по `predictions.jsonl` скрипта организатора: строго и «то же вино».

`bench.judge` считает строгий slug. Здесь рядом — счёт «то же вино»: верен и slug из колонки
`acceptable` эталона (другой год урожая или дубль карточки того же вина).

    python research/2026-09-27_results/same_wine.py --pred runs/eval-real100/predictions.jsonl \
        --gt data/gt/real_photos_gt.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("data/gt/real_photos_gt.tsv"))
    args = ap.parse_args()

    with args.gt.open(encoding="utf-8") as fh:
        gt = {r["query_id"]: r for r in csv.DictReader(fh, delimiter="\t")}
    preds = {}
    for line in args.pred.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            preds[row["query_id"]] = row.get("predicted_slug")
    inside = [q for q, r in gt.items() if r["slug"] != "__none__"]
    strict = sum(preds.get(q) == gt[q]["slug"] for q in inside)
    same = sum(
        preds.get(q)
        in {gt[q]["slug"], *[s for s in (gt[q].get("acceptable") or "").split("|") if s]}
        for q in inside
    )
    n = len(inside)
    print(
        f"кадров с вином из каталога: {n}, вне каталога: {len(gt) - n}, null: "
        f"{sum(preds.get(q) in (None, '') for q in gt)}"
    )
    print(f"top-1 «то же вино»: {same} из {n} = {100 * same / n:.1f} %")
    print(f"top-1 строго по slug: {strict} из {n} = {100 * strict / n:.1f} %")


if __name__ == "__main__":
    main()
