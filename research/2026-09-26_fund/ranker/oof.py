"""Ранкер на 4× данных: групповой K-fold вне фолда (OOF) по винам против замороженного `-goal`.

Пул — кадры с меткой каталога из `trainpool` (студия 358 + «телефон» 358 + v2 353 + kr-dev 308
+ same_packshot kr-dev 156 = 1 533; kr-test нет, страж в `common.load_pool`). Внешние фолды —
5 групп вин (`common.group_folds`, seed 20260926): все кадры вина в одном фолде.

Каждый вариант учится на 4 фолдах и отвечает на пятом тем же слоем выбора, что сервис
(`common.select`: Э2, ранкер, H5 / P1; и отдельно «только ранкер»). Внутри обучения — вложенный
групповой 4-fold: он выбирает L2 из сетки (критерий — «то же вино» слоя выбора вне внутреннего
фолда, ничья — больший L2) и температуру softmax по счётам вне внутреннего фолда. Порог H5 —
константа сервиса 0,5, её никто не трогает. Ничего не подбирается по внешнему фолду.

Варианты:
- `goal` — модель продукта как есть (без обучения); на 298 студийных кадрах `goal_split=dev`
  она в выборке обучения;
- `studio_refit` — рецепт `-goal` (listwise, знаки) только на студийных кадрах обучающих фолдов:
  отделяет «переобучить на нынешнем индексе» от «больше данных»;
- `pool` — тот же рецепт на всём пуле обучающих фолдов (студия + v2 + kr-dev + same_packshot);
- `field_only` — только полевые кадры (v2 + kr-dev + same_packshot);
- `pool_unsigned` — `pool` без ограничений знаков;
- `drop_<группа>`, `drop_text` — `pool` без группы признаков (важность группы);
- `pool_f25/50/75` — студия + четверть / половина / три четверти полевых групп (кривая обучения);
- `pool_extra` — `pool` + признаки таблиц других треков (`--extra`, `extras.py`, снимок таблицы).

`--cv screen:<имя>` — кандидаты из выдачи трека adapter (`cv_source.py`), внешние фолды — его же.
Там у счёта обучающих кадров вторичная утечка (адаптер их фолда видел тестовый фолд), поэтому
строгий счёт «адаптер + ранкер» — `adapter/ranker_folds.py` (перекрёстная подгонка).

    PYTHONPATH=. python research/2026-09-26_fund/ranker/oof.py --seeds 5 --tag oof
    PYTHONPATH=. python research/2026-09-26_fund/ranker/oof.py --variants pool --extra sift --seeds 3 \
        --seed-variants pool_extra --tag siftall
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402

from app.resolve.learned import LogisticRanker  # noqa: E402

L2_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
INNER_K = 4
STUDIO = ("studio", "studio_phone")


@dataclass(frozen=True)
class Variant:
    name: str
    train: Callable[[C.Frame], bool]
    signed: bool = True
    drop_groups: tuple[str, ...] = ()
    extra: tuple[str, ...] = ()
    l2_grid: tuple[float, ...] = L2_GRID


def _all(f: C.Frame) -> bool:
    return True


def _studio(f: C.Frame) -> bool:
    return f.source in STUDIO


def _field(f: C.Frame) -> bool:
    return f.source not in STUDIO


VARIANTS = {
    "studio_refit": Variant("studio_refit", _studio),
    "pool": Variant("pool", _all),
    "field_only": Variant("field_only", _field),
    "pool_unsigned": Variant("pool_unsigned", _all, signed=False),
}
#: Абляции `pool`: без одной группы признаков (важность группы — потеря OOF без неё).
_TEXT = ("winery", "winery_text", "grape", "sugar", "year", "serial", "color", "abv", "cuvee", "name")
for _g in ("cv_group", *_TEXT):
    VARIANTS[f"drop_{_g}"] = Variant(f"drop_{_g}", _all, drop_groups=(_g,))
VARIANTS["drop_text"] = Variant("drop_text", _all, drop_groups=_TEXT)


def _field_share(frac: float) -> Callable[[C.Frame], bool]:
    """Студия + доля полевых групп вин (детерминированно по md5 группы): кривая обучения."""
    import hashlib

    def pred(f: C.Frame) -> bool:
        if f.source in STUDIO:
            return True
        h = int(hashlib.md5(f.group.encode()).hexdigest()[:8], 16) / 16**8
        return h < frac

    return pred


for _p in (25, 50, 75):
    VARIANTS[f"pool_f{_p}"] = Variant(f"pool_f{_p}", _field_share(_p / 100))


# ------------------------------------------------------------------ вложенное обучение
_STATE: dict[str, Any] = {}


def _init(extra_tables: Sequence[str], cv: str = "product") -> None:
    """Пул в процессе-работнике: кандидаты продукта или выдача адаптера (`screen:<имя>`)."""
    folds_by_id = None
    if cv == "product":
        frames, names = C.load_pool(log=False)
    else:
        import cv_source

        frames, names, folds_by_id = cv_source.frames_from_screen(cv.split(":", 1)[1], log=False)
    if extra_tables:
        import extras

        extras.attach(frames, extra_tables)
    _STATE.update(frames=frames, names=names, attrs=C.load_attrs(), folds_by_id=folds_by_id, cv=cv)


def fit_nested(
    train: Sequence[C.Frame], v: Variant, names: Sequence[str], attrs: Any, seed: int
) -> tuple[LogisticRanker, dict[str, Any]]:
    """L2 и температура — вложенным групповым K-fold только по обучающим кадрам."""
    cols = C.cols_without(names, v.drop_groups) if v.drop_groups else None
    pool = [f for f in train if v.train(f)]
    inner = C.group_folds(pool, k=INNER_K, seed=seed + 1)
    table = []
    best = None
    for l2 in v.l2_grid:
        models, scores = {}, [np.empty(0)] * len(pool)
        for j in range(INNER_K):
            tr = [f for f in pool if inner[f.group] != j]
            m = C.fit_ranker(tr, names, l2=l2, signed=v.signed, cols=cols, extra=v.extra)
            models[j] = m
            for q, f in enumerate(pool):
                if inner[f.group] == j and f.failure is None:
                    scores[q] = C.scores_of(m, f, cols, v.extra)
        T = C.calibrate(models[0], pool, scores)
        ok_s = ok_r = 0
        for q, f in enumerate(pool):
            m = models[inner[f.group]]
            m.temperature_ = T
            sel = C.select(m, f, attrs, names, cols=cols, extra=v.extra,
                           scores=scores[q] if f.failure is None else None)
            ok_s += f.ok(sel["answer"])
            ok_r += f.ok(sel.get("ranker", sel["answer"]))
        row = {"l2": l2, "T": round(T, 4), "inner_service": ok_s, "inner_ranker": ok_r, "n": len(pool)}
        table.append(row)
        key = (ok_s, ok_r, l2)
        if best is None or key > best[0]:
            best = (key, l2, T)
    assert best is not None
    _, l2, T = best
    model = C.fit_ranker(pool, names, l2=l2, signed=v.signed, cols=cols, extra=v.extra)
    model.temperature_ = T
    return model, {"l2": l2, "T": round(T, 4), "grid": table, "train_frames": len(pool),
                   "fit": model.fit_info}


def outer_fold_of(frames: Sequence[C.Frame], seed: int, folds_by_id: dict[str, int] | None) -> dict[str, int]:
    """Внешний фолд кадра: свои групповые фолды (seed) или фолды адаптера для его выдачи."""
    if folds_by_id is not None:
        return dict(folds_by_id)
    g = C.group_folds(frames, seed=seed)
    return {f.id: g[f.group] for f in frames}


def run_fold(args: tuple[Any, ...]) -> dict[str, Any]:
    vname, k, seed, extra_tables = args[:4]
    cv = args[4] if len(args) > 4 else "product"
    if not _STATE:
        _init(extra_tables, cv)
    frames, names, attrs = _STATE["frames"], _STATE["names"], _STATE["attrs"]
    register_extra_variants(extra_tables)
    v = VARIANTS[vname]
    fold = outer_fold_of(frames, seed, _STATE.get("folds_by_id"))
    train = [f for f in frames if fold[f.id] != k]
    test = [f for f in frames if fold[f.id] == k]
    assert not {f.group for f in train} & {f.group for f in test}, "группа вина в двух фолдах"
    t0 = time.perf_counter()
    model, info = fit_nested(train, v, names, attrs, seed=seed + 100 * (k + 1))
    cols = C.cols_without(names, v.drop_groups) if v.drop_groups else None
    out = {}
    for f in test:
        sel = C.select(model, f, attrs, names, cols=cols, extra=v.extra)
        out[f.id] = {"service": sel["answer"], "ranker": sel.get("ranker", sel["answer"]),
                     "p1": sel["p1"], "path": sel["path"]}
    info["wall_s"] = round(time.perf_counter() - t0, 1)
    info["weights"] = model.weights()
    return {"variant": vname, "fold": k, "seed": seed, "answers": out, "info": info}


def register_extra_variants(specs: Sequence[str]) -> None:
    """`pool_extra` — рецепт `pool` + признаки таблиц других треков (`extras.py`)."""
    if specs:
        import extras

        VARIANTS["pool_extra"] = Variant("pool_extra", _all, extra=tuple(extras.extra_names(specs)))


def run_oof(
    variants: Sequence[str], seed: int, workers: int, extra_tables: Sequence[str], cv: str = "product"
) -> dict[str, Any]:
    jobs = [(v, k, seed, tuple(extra_tables), cv) for v in variants for k in range(C.N_FOLDS)]
    results: dict[str, dict[str, Any]] = {v: {"answers": {}, "folds": {}} for v in variants}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(tuple(extra_tables), cv)) as ex:
        for res in ex.map(run_fold, jobs):
            results[res["variant"]]["answers"].update(res["answers"])
            results[res["variant"]]["folds"][res["fold"]] = res["info"]
            print(f"  {res['variant']} seed {seed} fold {res['fold']}: L2 {res['info']['l2']} "
                  f"T {res['info']['T']} ({res['info']['wall_s']} s)", flush=True)
    return results


# ------------------------------------------------------------------ отчёт
def summarize(
    frames: list[C.Frame], results: dict[str, Any], names: Sequence[str], attrs: Any, cv: str = "product"
) -> dict[str, Any]:
    from app.resolve.learned import LogisticRanker as LR

    goal = LR.load(C.GOAL_MODEL)
    ok: dict[str, list[bool]] = {}
    goal_sel = [C.select(goal, f, attrs, names) for f in frames]
    if cv == "product":  # слой выбора по записи = ответ продукта кадр в кадр
        assert all(s["answer"] == f.goal_answer for s, f in zip(goal_sel, frames, strict=True))
    else:  # на выдаче адаптера: «goal» — -goal + H5/P1 на новой выдаче, «product» — ответ продукта
        ok["product"] = [f.ok(f.goal_answer) for f in frames]
    ok["goal"] = [f.ok(s["answer"]) for s, f in zip(goal_sel, frames, strict=True)]
    ok["goal_ranker_only"] = [f.ok(s.get("ranker", s["answer"])) for s, f in zip(goal_sel, frames, strict=True)]
    ok["cv_top1"] = [f.ok(f.cv_top1) for f in frames]
    ok["ceiling_top20"] = [f.pos is not None for f in frames]
    for v, res in results.items():
        a = res["answers"]
        ok[v] = [f.ok(a[f.id]["service"]) for f in frames]
        ok[f"{v}_ranker_only"] = [f.ok(a[f.id]["ranker"]) for f in frames]
    tallies = {k: C.tally(frames, o) for k, o in ok.items()}
    cis = {k: {s: C.ci_micro(frames, o, s) for s in ("all", "field_main", "kr_dev", "v2", "studio_all")}
           for k, o in ok.items()}
    comps = {}
    for v in results:
        comps[f"{v}_vs_goal"] = C.paired(frames, ok["goal"], ok[v])
        comps[f"{v}_ranker_only_vs_{v}"] = C.paired(frames, ok[v], ok[f"{v}_ranker_only"])
    if "pool" in results and "studio_refit" in results:
        comps["pool_vs_studio_refit"] = C.paired(frames, ok["studio_refit"], ok["pool"])
        comps["studio_refit_vs_goal"] = C.paired(frames, ok["goal"], ok["studio_refit"])
    if "pool_extra" in results and "pool" in results:
        comps["pool_extra_vs_pool"] = C.paired(frames, ok["pool"], ok["pool_extra"])
    if "product" in ok:
        comps["goal_vs_product"] = C.paired(frames, ok["product"], ok["goal"])
        for v in results:
            comps[f"{v}_vs_product"] = C.paired(frames, ok["product"], ok[v])
    folds = {v: {str(k): {kk: vv for kk, vv in inf.items() if kk != "weights"} for k, inf in res["folds"].items()}
             for v, res in results.items()}
    paths = {}
    for v, res in results.items():
        from collections import Counter

        paths[v] = dict(Counter(res["answers"][f.id]["path"] for f in frames))
    paths["goal"] = dict(__import__("collections").Counter(s["path"] for s in goal_sel))
    return {"tallies": tallies, "ci95": cis, "paired": comps, "folds": folds, "paths": paths}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=1, help="повторы разбиения на фолды (seed + i)")
    ap.add_argument("--seed-variants", nargs="*", default=["studio_refit", "pool"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--extra", nargs="*", default=[], help="таблицы признаков других треков (extras.py)")
    ap.add_argument("--tag", default="oof")
    ap.add_argument("--cv", default="product", help="product или screen:<имя> (выдача трека adapter)")
    args = ap.parse_args()
    register_extra_variants(args.extra)
    variants = list(args.variants) + (["pool_extra"] if args.extra else [])
    folds_by_id = None
    if args.cv == "product":
        frames, names = C.load_pool()
    else:
        import cv_source

        frames, names, folds_by_id = cv_source.frames_from_screen(args.cv.split(":", 1)[1])
    if args.extra:
        import extras

        extras.attach(frames, args.extra)
    attrs = C.load_attrs()
    C.OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"OOF seed {C.FOLD_SEED}: {variants}", flush=True)
    results = run_oof(variants, C.FOLD_SEED, args.workers, args.extra, args.cv)
    summary = summarize(frames, results, names, attrs, args.cv)
    summary["meta"] = {
        "frames": len(frames),
        "fold_seed": C.FOLD_SEED,
        "n_folds": C.N_FOLDS,
        "inner_k": INNER_K,
        "l2_grid": L2_GRID,
        "variants": variants,
        "extra": args.extra,
        "extra_sha1": __import__("extras").table_sha1(args.extra) if args.extra else {},
        "cv": args.cv,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    # повторы разбиения: разброс OOF от выбора фолдов
    if args.seeds > 1:
        rep: dict[str, list[dict[str, Any]]] = {}
        for i in range(1, args.seeds):
            seed = C.FOLD_SEED + i
            print(f"OOF seed {seed}: {args.seed_variants}", flush=True)
            if folds_by_id is not None:
                break  # у выдачи адаптера фолды свои, повтор разбиения не имеет смысла
            res = run_oof(args.seed_variants, seed, args.workers, args.extra, args.cv)
            for v, r in res.items():
                okv = [f.ok(r["answers"][f.id]["service"]) for f in frames]
                okr = [f.ok(r["answers"][f.id]["ranker"]) for f in frames]
                rep.setdefault(v, []).append({"seed": seed, "service": C.tally(frames, okv),
                                              "ranker_only": C.tally(frames, okr)})
        summary["seed_repeats"] = rep
    with (C.OUT / f"{args.tag}_answers.jsonl").open("w", encoding="utf-8") as fh:
        folds = outer_fold_of(frames, C.FOLD_SEED, folds_by_id)
        for f in frames:
            row = {"id": f.id, "set": f.set, "source": f.source, "group": f.group,
                   "fold": folds[f.id], "goal": f.goal_answer, "cv_top1": f.cv_top1,
                   "ok_goal": f.ok(f.goal_answer)}
            for v, r in results.items():
                a = r["answers"][f.id]
                row[v] = a["service"]
                row[f"{v}_ranker_only"] = a["ranker"]
                row[f"{v}_p1"] = a["p1"]
                row[f"{v}_path"] = a["path"]
                row[f"ok_{v}"] = f.ok(a["service"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    weights = {v: {str(k): inf["weights"] for k, inf in r["folds"].items()} for v, r in results.items()}
    (C.OUT / f"{args.tag}_fold_weights.json").write_text(json.dumps(weights, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (C.OUT / f"{args.tag}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    t = summary["tallies"]
    for k in t:
        row = t[k]
        print(f"{k:28s} " + "  ".join(f"{s} {row[s]['ok']}/{row[s]['n']}" for s in
                                        ("studio_all", "v2", "v2_R", "kr_dev", "kr_dev_sp", "field_main", "all") if s in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
