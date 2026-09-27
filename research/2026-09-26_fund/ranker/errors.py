"""Классы ошибок вне фолда: что ранкер вообще может починить на нынешних признаках.

Только пул разработки (ответы `oof_answers.jsonl` из `oof.py`, kr-test нет). Ошибка кадра:
- `not_in_top20` — верного нет в выдаче CV (ранкер бессилен);
- `e2` — сбой читателя, ответ CV top-1;
- соседнее вино той же винодельни (`same_winery`) или другой (`other_winery`) × совпадают ли
  22 текстовых признака у выбранного и у верного (`text_identical`): при совпадении прочитанное
  их не различает, и решает только CV — чинить такое может лишь новый сигнал (картинка
  точнее, слово этикетки, которого нет в признаках), а не другой вес.

    PYTHONPATH=. python research/2026-09-26_fund/ranker/errors.py [--tag oof] [--variants goal pool]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402


def classify(f: C.Frame, ans: str | None, attrs, text_cols) -> str | None:
    if f.ok(ans):
        return None
    if f.failure is not None:
        return "e2"
    if f.pos is None:
        return "not_in_top20"
    j = f.slugs.index(ans) if ans in f.slugs else None
    wa, wt = attrs.get(ans), attrs.get(f.slugs[f.pos])
    same = wa is not None and wt is not None and wa.winery == wt.winery
    tie = j is not None and np.allclose(f.X[j, text_cols], f.X[f.pos, text_cols])
    return f"{'same' if same else 'other'}_winery/{'text_identical' if tie else 'text_differs'}"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="oof")
    ap.add_argument("--variants", nargs="*", default=["goal", "studio_refit", "pool"])
    args = ap.parse_args()
    frames, names = C.load_pool()
    attrs = C.load_attrs()
    answers = {r["id"]: r for r in map(json.loads, open(C.OUT / f"{args.tag}_answers.jsonl", encoding="utf-8"))}
    text_cols = [j for j, n in enumerate(names) if n.startswith("vlm35.")]
    out = {}
    for v in args.variants:
        by_slice: dict[str, Counter] = {}
        for name, pred in C.SLICES.items():
            cnt = Counter()
            for f in frames:
                if pred(f):
                    c = classify(f, answers[f.id][v], attrs, text_cols)
                    if c:
                        cnt[c] += 1
            by_slice[name] = dict(cnt.most_common())
        out[v] = by_slice
        print(v, json.dumps(by_slice["all"], ensure_ascii=False))
        print("   field_main", json.dumps(by_slice["field_main"], ensure_ascii=False))
    (C.OUT / f"{args.tag}_errors.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
