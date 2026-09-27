"""Быстрая OOF-оценка: ранкер продукта + inliers SIFT одним признаком (групповой K-fold по винам).

    PYTHONPATH=. python research/2026-09-26_fund/spatial/oof.py [--seeds 5]

Пул — наборы с метками каталога (`catalog_v2`, `kr_dev`, `kr_dev_sp`, `pairs`, `pairs_phone`),
kr-test нет (`assert_no_test`). Фолды — 5 по группе вина (все кадры вина в одном фолде),
жадно по числу кадров и срезам; несколько сидов разбиения.

Варианты (объявлены до замера; основной — `goal+sift`):
- `product` — ответы продукта из пула; `goal` — тот же `-goal` через свой слой выбора (сверка);
- `goal+sift` — счёт `-goal` как есть + `sift_inl_log` (log1p inliers): listwise-логистика на двух
  признаках (знаки +, L2 0,01, как у `-goal`) по фолдам обучения, затем модель собирается в
  35-признаковый `LogisticRanker` (веса `-goal` × a + вес SIFT), чтобы H5 и P1 работали как в
  сервисе (снятие бонуса соседу, отсев спорящих по сахару и цвету). Температура — по фолдам обучения;
- `refit34` / `refit34+sift` — контроль: те же 34 признака `-goal` переобучены по фолдам с нуля
  (знаки и L2 как у `-goal`) без и с `sift_inl_log` — чистое сравнение «тот же рецепт ± SIFT»;
- разведка (не основное, вилка выбора — добавлены после разбора ошибок SIFT на пуле):
  `+sift4` — 4 признака (`inl_log`, `inl_gap`, `inl_ratio`, `area_label`); `+uniq` — различимые
  inliers внутри top-20 (`sift_uniq_log`); `+norm` — inliers на точку эталона и различимые;
  `+top5` — проверка только у top-5 (`sift_inl5_log`, `sift_uniq5_log`), вариант по бюджету.

Ответ — слой выбора сервиса по записи (Э2 → CV top-1, ранкер, softmax с T, H5 при p1 < 0,5, иначе
P1). Счёт «то же вино» по срезам, знаковый тест по сменившимся кадрам против продукта и против
контроля. Выход — `spatial/oof.json`, `spatial/oof_frames.jsonl`.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import protocol as P

from app.api.service import AMBIGUOUS_P_TOP1, reader_failure, softmax
from app.reading.contracts import LabelFields
from app.resolve.ambiguous import block_bonus_flip, rerank_ambiguous
from app.resolve.features import QueryFeatures
from app.resolve.learned import LogisticRanker, QueryBlocks, fit_temperature

OUT = P.FUND / "spatial"
GOAL_MODEL = P.HERE.parents[1] / "configs" / "resolve" / "s2so400m-vlm35-goal.json"
SETS = ("catalog_v2", "kr_dev", "kr_dev_sp", "pairs", "pairs_phone")
K = 5
SEED = 20260926
L2 = 0.01
#: Наборы признаков SIFT: основной (`sift`, объявлен до замера) и разведка.
EXTRAS: dict[str, list[str]] = {
    "sift": ["sift_inl_log"],
    "sift4": ["sift_inl_log", "sift_inl_gap", "sift_inl_ratio", "sift_area_label"],
    "uniq": ["sift_uniq_log"],
    "norm": ["sift_inl_log", "sift_uniq_log", "sift_inl_norm"],
    "top5": ["sift_inl5_log", "sift_uniq5_log"],
}


@dataclass
class Frame:
    i: int
    id: str
    set: str
    source: str
    group: str
    labels: set[str]
    slug: str
    slugs: tuple[str, ...]
    X: np.ndarray  # (n, 34)
    S: np.ndarray  # (n, F) SIFT
    fields: LabelFields | None
    failure: str | None
    cv_first: str | None
    product: str | None
    goal_split: str | None

    def ok(self, a: str | None) -> bool:
        return a is not None and a in self.labels

    @property
    def pos(self) -> int | None:
        if self.slug in self.slugs:
            return self.slugs.index(self.slug)
        for j, s in enumerate(self.slugs):
            if s in self.labels:
                return j
        return None


def source_of(r: dict[str, Any]) -> str:
    if r["set"] == "catalog_v2":
        return {"portal_real": "v2_R", "phone_m": "v2_M", "set_l": "v2_L"}[r["source"]]
    return {"pairs": "studio", "pairs_phone": "studio_phone"}.get(r["set"], r["set"])


def load_frames() -> tuple[list[Frame], list[str], list[str]]:
    from replay import read_of

    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in SETS]
    n = P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    print(f"[страж] assert_no_test: {n} id и картинок, kr-test нет", flush=True)
    z = np.load(P.PROTOCOL / "trainpool.npz")
    sz = np.load(OUT / "sift_features.npz")
    assert (sz["ids"] == z["ids"]).all()
    names = [str(x) for x in z["feature_names"]]
    snames = [str(x) for x in sz["sift_names"]]
    frames = []
    for r in rows:
        fs = z["feat_slugs"][r["row"]]
        n = sum(1 for s in fs if str(s))
        slugs = tuple(str(s) for s in fs[:n])
        S = sz["sift"][r["row"], :n].astype(np.float64)
        if n and not np.isfinite(S).all():
            raise AssertionError(f"нет SIFT у {r['id']}")
        frames.append(
            Frame(
                i=r["row"], id=r["id"], set=r["set"], source=source_of(r), group=r["group"],
                labels={r["slug"], *(r["acceptable"] or [])}, slug=r["slug"], slugs=slugs,
                X=z["features"][r["row"], :n].astype(np.float64), S=S,
                fields=read_of(r["read"]).fields,
                failure=reader_failure({"status": r["vlm_status"], "lines": ["x"] * r["vlm_lines"]}),
                cv_first=r["cv"][0][0] if r["cv"] else None, product=r["answer"],
                goal_split=r.get("goal_split"),
            )
        )
    return frames, names, snames


def group_folds(frames: Sequence[Frame], k: int, seed: int) -> dict[str, int]:
    """Группа вина → фолд: жадно, крупные первыми, выравнивая кадры по срезам."""
    by: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in frames:
        by[f.group][f.source] += 1
    rng = np.random.default_rng(seed)
    keys = sorted(by)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -sum(by[g].values()))
    srcs = sorted({f.source for f in frames})
    load = np.zeros((k, len(srcs)))
    out = {}
    for g in keys:
        v = np.array([by[g].get(s, 0) for s in srcs], float)
        costs = []
        for j in range(k):
            new = load.copy()
            new[j] += v
            costs.append(np.abs(new - new.mean(0)).sum() + abs(new.sum(1) - new.sum(1).mean()).sum())
        j = int(np.argmin(costs))
        load[j] += v
        out[g] = j
    return out


# ------------------------------------------------------------------ модели
def design(frames: Sequence[Frame], mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, qs = [], [], []
    for q, f in enumerate(frames):
        pos = f.pos
        if pos is None or f.failure is not None or not f.slugs:
            continue
        keep = [j for j, s in enumerate(f.slugs) if j == pos or s not in f.labels]
        X = mat(f)[keep]
        y = np.zeros(len(keep))
        y[keep.index(pos)] = 1.0
        Xs.append(X)
        ys.append(y)
        qs.append(np.full(len(keep), q))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)


def calibrate(model: LogisticRanker, frames: Sequence[Frame], mat) -> None:
    flat, ids, correct = [], [], []
    for q, f in enumerate(frames):
        if f.failure is not None or not f.slugs:
            continue
        s = model.decision_function(mat(f))
        flat.append(s)
        ids.append(np.full(len(s), q))
        correct.append(float(f.ok(f.slugs[int(np.argmax(s))])))
    model.temperature_ = fit_temperature(np.concatenate(flat), QueryBlocks.from_ids(np.concatenate(ids)), np.asarray(correct))


def select(model: LogisticRanker, f: Frame, attrs: Any, mat, names: Sequence[str], rules: bool = True) -> str | None:
    """Слой выбора сервиса по записи (как `ScannerService._resolve`)."""
    if f.failure is not None:
        return f.cv_first
    if not f.slugs:
        return f.cv_first
    X = mat(f)
    s = model.decision_function(X)
    order = sorted(range(len(s)), key=lambda i: (-float(s[i]), i))
    ranked = [f.slugs[i] for i in order]
    if not rules:
        return ranked[0]
    p1 = float(softmax([float(s[i]) for i in order], model.temperature_)[0])
    qf = QueryFeatures(f.slugs, tuple({n: float(v) for n, v in zip(names, X[i], strict=True)} for i in range(len(X))))
    if p1 < AMBIGUOUS_P_TOP1:
        return list(rerank_ambiguous(model, qf, f.fields, attrs))[0]
    return list(block_bonus_flip(model, qf, ranked, f.fields, attrs))[0]


def compose(goal: LogisticRanker, stack: LogisticRanker, extra: Sequence[str]) -> LogisticRanker:
    """`-goal` × a + веса SIFT → один `LogisticRanker` на 34 + len(extra) признаках (для H5/P1)."""
    assert goal.coef_ is not None and goal.mean_ is not None and goal.scale_ is not None
    assert stack.coef_ is not None and stack.mean_ is not None and stack.scale_ is not None
    a = stack.coef_[0] / stack.scale_[0]
    m = LogisticRanker([*goal.feature_names, *extra], l2=L2, loss="listwise", signs=[*goal.signs, *([1] * len(extra))])
    m.mean_ = np.concatenate([goal.mean_, stack.mean_[1:]])
    m.scale_ = np.concatenate([goal.scale_, stack.scale_[1:]])
    m.coef_ = np.concatenate([a * goal.coef_, stack.coef_[1:]])
    m.intercept_ = 0.0
    m.meta = {"a_goal": float(a), "sift_coef": [float(c) for c in stack.coef_[1:]]}
    return m


def run(frames: list[Frame], names: list[str], snames: list[str], attrs: Any, seed: int) -> dict[str, dict[str, str | None]]:
    goal = LogisticRanker.load(GOAL_MODEL)
    assert list(goal.feature_names) == names
    folds = group_folds(frames, K, seed)
    idx = {n: j for j, n in enumerate(snames)}
    out: dict[str, dict[str, str | None]] = defaultdict(dict)
    info: dict[str, list[Any]] = defaultdict(list)

    def m34(f: Frame) -> np.ndarray:
        return f.X

    def m_ext(extra: Sequence[str]):
        cols = [idx[e] for e in extra]
        return lambda f: np.hstack([f.X, f.S[:, cols]])

    def m_goal_stack(extra: Sequence[str]):
        cols = [idx[e] for e in extra]
        return lambda f: np.column_stack([goal.decision_function(f.X), f.S[:, cols]])

    for f in frames:
        out["goal"][f.id] = select(goal, f, attrs, m34, names)
        out["goal_norules"][f.id] = select(goal, f, attrs, m34, names, rules=False)
    for k in range(K):
        tr = [f for f in frames if folds[f.group] != k]
        te = [f for f in frames if folds[f.group] == k]
        for tag, extra in EXTRAS.items():
            # goal + SIFT: стек на двух (пяти) признаках, затем сборка в 35 (38)
            X, y, q = design(tr, m_goal_stack(extra))
            st = LogisticRanker(["goal_score", *extra], l2=L2, loss="listwise", signs=[1] * (1 + len(extra))).fit(X, y, q)
            model = compose(goal, st, extra)
            mat = m_ext(extra)
            calibrate(model, tr, mat)
            info[f"goal+{tag}"].append({"fold": k, **model.meta, "T": round(model.temperature_, 4)})
            for f in te:
                out[f"goal+{tag}"][f.id] = select(model, f, attrs, mat, [*names, *extra])
                out[f"goal+{tag}_norules"][f.id] = select(model, f, attrs, mat, [*names, *extra], rules=False)
        # refit с нуля: 34 и 34 + SIFT
        for tag, extra in (("refit34", []), *((f"refit34+{t}", e) for t, e in EXTRAS.items())):
            mat = m_ext(extra) if extra else m34
            X, y, q = design(tr, mat)
            model = LogisticRanker([*names, *extra], l2=L2, loss="listwise", signs=[*goal.signs, *([1] * len(extra))]).fit(X, y, q)
            calibrate(model, tr, mat)
            w = model.weights()
            info[tag].append({"fold": k, "T": round(model.temperature_, 4), **{e: round(w[e], 4) for e in extra}})
            for f in te:
                out[tag][f.id] = select(model, f, attrs, mat, [*names, *extra])
                out[f"{tag}_norules"][f.id] = select(model, f, attrs, mat, [*names, *extra], rules=False)
    out["_info"] = info  # type: ignore[assignment]
    return out


# ------------------------------------------------------------------ счёт
SLICES = {
    "v2": lambda f: f.set == "catalog_v2",
    "v2_R": lambda f: f.source == "v2_R",
    "kr_dev": lambda f: f.set == "kr_dev",
    "kr_dev_sp": lambda f: f.set == "kr_dev_sp",
    "field(v2+kr_dev)": lambda f: f.set in ("catalog_v2", "kr_dev"),
    "studio": lambda f: f.set == "pairs",
    "studio_phone": lambda f: f.set == "pairs_phone",
    "studio_goaltest": lambda f: f.set in ("pairs", "pairs_phone") and f.goal_split == "test",
    "all": lambda f: True,
}


def sign_p(fix: int, brk: int) -> float:
    n = fix + brk
    if n == 0:
        return 1.0
    k = min(fix, brk)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def compare(frames: Sequence[Frame], a: dict[str, str | None], b: dict[str, str | None]) -> dict[str, Any]:
    fix = sum(1 for f in frames if f.ok(a[f.id]) and not f.ok(b[f.id]))
    brk = sum(1 for f in frames if not f.ok(a[f.id]) and f.ok(b[f.id]))
    return {"fix": fix, "break": brk, "p": round(sign_p(fix, brk), 4), "changed": sum(1 for f in frames if a[f.id] != b[f.id])}


def load_attrs() -> Any:
    """Таблица каталога сервиса на снимке (как `FundReplay().svc.attrs`, без индекса и модели)."""
    from app.api.config import ServiceSettings
    from app.api.service import load_catalog

    settings = ServiceSettings.from_env(
        {
            "SVS_DATA_DIR": str(P.FROZEN),
            "SVS_DATASET_DIR": str(P.FROZEN_DATASET),
            "SVS_CACHE_DIR": str(P.FROZEN / "cache"),
            "SVS_LIVE_CARDS": "0",
            "SVS_DEVICE": "cpu",
        }
    )
    return load_catalog(settings.attrs_path)[1]


_W: dict[str, Any] = {}


def _seed_job(seed: int) -> dict[str, Any]:
    if not _W:
        frames, names, snames = load_frames()
        _W.update(frames=frames, names=names, snames=snames, attrs=load_attrs())
    return run(_W["frames"], _W["names"], _W["snames"], _W["attrs"], seed)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--tag", default="", help="суффикс файлов вывода")
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor

    frames, _, _ = load_frames()
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(_seed_job, [SEED + s for s in range(args.seeds)]))
    prod = {f.id: f.product for f in frames}
    report: dict[str, Any] = {"frames": len(frames), "seeds": []}
    per_frame: dict[str, dict[str, Any]] = {f.id: {"set": f.set, "product_ok": f.ok(f.product)} for f in frames}
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s, res in enumerate(results):
        seed = SEED + s
        info = res.pop("_info")
        mism = [f.id for f in frames if res["goal"][f.id] != f.product]
        assert not mism, f"свой слой выбора ≠ продукт на {len(mism)} кадрах: {mism[:5]}"
        seed_rep: dict[str, Any] = {"seed": seed, "goal_equals_product": True, "fold_info": info, "slices": {}}
        variants = ["product", *[v for v in res if v != "goal"]]
        for sl, pred in SLICES.items():
            fr = [f for f in frames if pred(f)]
            row: dict[str, Any] = {"n": len(fr)}
            for v in variants:
                ans = prod if v == "product" else res[v]
                c = sum(f.ok(ans[f.id]) for f in fr)
                row[v] = c
                agg[sl][v].append(c)
            for t in EXTRAS:
                row[f"goal+{t}_vs_product"] = compare(fr, res[f"goal+{t}"], prod)
                row[f"refit34+{t}_vs_refit34"] = compare(fr, res[f"refit34+{t}"], res["refit34"])
            row["refit34_vs_product"] = compare(fr, res["refit34"], prod)
            row["goal+sift_norules_vs_goal_norules"] = compare(fr, res["goal+sift_norules"], res["goal_norules"])
            seed_rep["slices"][sl] = row
        report["seeds"].append(seed_rep)
        if s == 0:
            for f in frames:
                per_frame[f.id].update({v: res[v][f.id] for v in ("goal+sift", "refit34", "refit34+sift")})
                per_frame[f.id]["goal+sift_ok"] = f.ok(res["goal+sift"][f.id])
        print(f"сид {seed}: готов", flush=True)
    report["mean_over_seeds"] = {sl: {v: round(float(np.mean(x)), 2) for v, x in d.items()} for sl, d in agg.items()}
    report["range_over_seeds"] = {sl: {v: [min(x), max(x)] for v, x in d.items()} for sl, d in agg.items()}
    (OUT / f"oof{args.tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / f"oof_frames{args.tag}.jsonl").open("w", encoding="utf-8") as fh:
        for fid, d in per_frame.items():
            fh.write(json.dumps({"id": fid, **d}, ensure_ascii=False) + "\n")
    mean = report["mean_over_seeds"]
    cols = ["product", "goal_norules", *(f"goal+{t}" for t in EXTRAS), "refit34", *(f"refit34+{t}" for t in EXTRAS)]
    print(f"\nсреднее «то же вино» по {args.seeds} сидам разбиения:")
    print(f"{'срез':18s} {'n':>5s} " + " ".join(f"{c:>13s}" for c in cols))
    for sl, d in mean.items():
        n = report["seeds"][0]["slices"][sl]["n"]
        print(f"{sl:18s} {n:5d} " + " ".join(f"{d[c]:13.1f}" for c in cols))
    print("\nсид 0, знаковый тест по сменившимся кадрам (fix/break p):")
    for sl, row in report["seeds"][0]["slices"].items():
        parts = [f"goal+{t} vs прод {row[f'goal+{t}_vs_product']['fix']}/{row[f'goal+{t}_vs_product']['break']} p={row[f'goal+{t}_vs_product']['p']}" for t in EXTRAS]
        parts += [f"refit+{t} vs refit {row[f'refit34+{t}_vs_refit34']['fix']}/{row[f'refit34+{t}_vs_refit34']['break']} p={row[f'refit34+{t}_vs_refit34']['p']}" for t in EXTRAS]
        print(f"{sl:18s} " + "; ".join(parts))
    print("\nвеса goal+sift по фолдам (сид 0):", json.dumps(report["seeds"][0]["fold_info"]["goal+sift"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
