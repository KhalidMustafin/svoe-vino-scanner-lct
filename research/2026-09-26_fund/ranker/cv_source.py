"""Другой источник кандидатов для ранкера: выдача CV вне фолда из трека adapter.

`adapter/screen/<имя>.npz` (трек adapter, `research/2026-09-26_fund/adapter/screen.py`): на каждый
кадр пула с меткой — top-20 slug (номера `index.slug_order`), счёт, строка индекса, отрыв и фолд
(`group_kfold` трека, 5 групп вин). Здесь из этой выдачи собирается `VisualResult` ровно как в
`FundReplay.candidates_from_scores`, и 34 признака ранкера пересчитываются кодом сервиса
(`app.resolve.features.query_features`) с записанным чтением кадра. Получаются те же `Frame`,
что у `common.load_pool`, только с другими кандидатами — ранкер учится и отвечает на них.

Внешние фолды OOF для такого источника — фолды самого адаптера: счёт кадра получен адаптером,
не видевшим его группу, и ранкер учится только на кадрах других фолдов той же разбивки, так что
вложенной утечки через адаптер нет.
"""

from __future__ import annotations


import numpy as np

import common as C

SCREEN = C.P.FUND / "adapter" / "screen"


def index_views_and_slugs() -> tuple[list[str], list[str]]:
    from app.api.config import CV_DTYPE, ServiceSettings  # noqa: F401
    from app.api.service import load_index

    settings = ServiceSettings.from_env(
        {"SVS_DATA_DIR": str(C.P.FROZEN), "SVS_DATASET_DIR": str(C.P.FROZEN_DATASET),
         "SVS_CACHE_DIR": str(C.P.FROZEN / "cache"), "SVS_LIVE_CARDS": "0", "SVS_DEVICE": "cpu"}
    )
    idx = load_index(settings.index_path, settings.cv_model)
    return list(idx.slug_order), list(idx.views)


def frames_from_screen(name: str, *, log: bool = True) -> tuple[list[C.Frame], list[str], dict[str, int]]:
    """Кадры пула с кандидатами из выдачи адаптера и фолд каждого кадра (по id)."""
    from app.features.contracts import Candidate, VisualResult
    from app.resolve.features import query_features
    from replay import read_of

    frames, names = C.load_pool(log=log)
    z = np.load(SCREEN / f"{name}.npz")
    ids = [str(x) for x in z["ids"]]
    C.P.assert_no_test(ids)
    pos = {q: i for i, q in enumerate(ids)}
    slug_order, views = index_views_and_slugs()
    attrs = C.load_attrs()
    rows = {r["id"]: r for r in C.P.jsonl(C.P.PROTOCOL / "trainpool.jsonl") if r["id"] in pos}
    folds: dict[str, int] = {}
    for f in frames:
        i = pos[f.id]
        folds[f.id] = int(z["fold"][i])
        cands = [
            Candidate(slug=slug_order[int(s)], score=float(max(0.0, sc)), view=views[int(row)], rank=r)
            for r, (s, sc, row) in enumerate(zip(z["top_idx"][i], z["top_score"][i], z["top_row"][i]), start=1)
        ]
        f.cv_first = cands[0].slug
        if f.failure is not None:  # Э2: признаков нет, ответ — CV top-1 новой выдачи
            f.slugs, f.X = (), np.zeros((0, len(names)))
            continue
        v = VisualResult(candidates=cands, margin=float(z["margin"][i]), timings_ms={}, model="")
        qf = query_features(v, {"vlm35": read_of(rows[f.id]["read"])}, attrs, top_k=20)
        f.slugs = qf.slugs
        f.X = qf.matrix(names)
    return frames, names, folds


def check_none() -> dict[str, int]:
    """Выдача `none` (без адаптера) = выдача продукта: признаки и ответы `-goal` совпадают."""
    from app.resolve.learned import LogisticRanker

    base, names = C.load_pool(log=False)
    new, _, _ = frames_from_screen("none", log=False)
    goal = LogisticRanker.load(C.GOAL_MODEL)
    attrs = C.load_attrs()
    pairs = list(zip(base, new, strict=True))
    return {
        "frames": len(base),
        "same_slugs": sum(a.slugs == b.slugs for a, b in pairs),
        # счёт адаптера — свой путь float32: разница признаков CV до 1,3e-4 (cv_z)
        "same_features_2e-4": sum(a.X.shape == b.X.shape and np.allclose(a.X, b.X, atol=2e-4) for a, b in pairs),
        "goal_answer_equal": sum(C.select(goal, b, attrs, names)["answer"] == a.goal_answer for a, b in pairs),
    }


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(json.dumps(check_none(), ensure_ascii=False))
