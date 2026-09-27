"""Трек «ранкер» фундаментального трека: пул, групповые фолды, слой выбора сервиса на записях.

Только пул обучения и разработки (`fund/protocol/trainpool.*`, EVAL_PROTOCOL.md §3), kr-test
здесь не бывает: `load_pool` зовёт `protocol.assert_no_test` на всех кадрах и картинках и
пишет число проверенных. Ранкер — `app.resolve.learned.LogisticRanker` того же формата, что
`-goal` сервиса: 34 признака `resolve-features/3`, listwise, L2, знаки весов.

Слой выбора (`select`) повторяет `ScannerService._resolve` на записанных признаках: Э2 (сбой
читателя → CV top-1), ранкер, softmax с температурой модели, H5 при p_top1 < 0,5, иначе P1.
Равенство с `FundReplay.resolve` проверяет `check_select` (все кадры пула, модель `-goal`).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
sys.path.insert(0, str(FUND_DIR))

import protocol as P  # noqa: E402

from app.api.service import AMBIGUOUS_P_TOP1, reader_failure, softmax  # noqa: E402
from app.reading.contracts import LabelFields  # noqa: E402
from app.resolve.ambiguous import block_bonus_flip, rerank_ambiguous  # noqa: E402
from app.resolve.features import FEATURE_GROUPS, QueryFeatures, feature_signs  # noqa: E402
from app.resolve.learned import (  # noqa: E402
    LogisticRanker,
    QueryBlocks,
    fit_temperature,
)

OUT = P.FUND / "ranker"
GOAL_MODEL = P.HERE.parents[1] / "configs" / "resolve" / "s2so400m-vlm35-goal.json"
#: Наборы пула с метками каталога — на них учится и меряется ранкер. ooc_v2 — без верного.
LABELLED_SETS = ("pairs", "pairs_phone", "catalog_v2", "kr_dev", "kr_dev_sp")
FOLD_SEED = 20260926
N_FOLDS = 5
#: Знак веса новых признаков других треков (сходство — вес ≥ 0); задаётся при подключении таблицы.
EXTRA_SIGNS: dict[str, int] = {}


# ------------------------------------------------------------------ пул
@dataclass
class Frame:
    """Кадр пула: метка, записанные признаки кандидатов (порядок CV) и всё для слоя выбора."""

    i: int  # номер строки в trainpool
    id: str
    set: str
    source: str  # срез отчёта: studio / studio_phone / v2_R / v2_M / v2_L / kr_dev / kr_dev_sp
    group: str
    slug: str | None
    acceptable: tuple[str, ...]
    slugs: tuple[str, ...]  # кандидаты в порядке CV
    X: np.ndarray  # (кандидаты × 34) float64
    fields: LabelFields | None
    failure: str | None  # сбой читателя (Э2) или None
    goal_answer: str | None  # ответ продукта (-goal + H5/P1/Э2)
    goal_split: str | None
    kr_test_wine: bool
    cv_first: str | None = None  # CV top-1 по записи (у кадров Э2 признаков нет)
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # новые признаки других треков

    @property
    def labels(self) -> set[str]:
        return {self.slug, *self.acceptable} if self.slug else set()

    def ok(self, answer: str | None) -> bool:
        return answer is not None and answer in self.labels

    @property
    def pos(self) -> int | None:
        """Индекс верного кандидата для обучения: slug, иначе лучший по CV из acceptable."""
        if self.slug in self.slugs:
            return self.slugs.index(self.slug)
        for j, s in enumerate(self.slugs):
            if s in self.labels:
                return j
        return None

    @property
    def cv_top1(self) -> str | None:
        return self.cv_first


def source_of(r: Mapping[str, Any]) -> str:
    if r["set"] == "catalog_v2":
        return {"portal_real": "v2_R", "phone_m": "v2_M", "set_l": "v2_L"}[r["source"]]
    return {"pairs": "studio", "pairs_phone": "studio_phone"}.get(r["set"], r["set"])


def load_pool(sets: Sequence[str] = LABELLED_SETS, *, log: bool = True) -> tuple[list[Frame], list[str]]:
    """Кадры пула с метками каталога; страж kr-test — до любой работы."""
    from replay import read_of

    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in sets]
    checked = P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    if log:
        print(f"[страж] assert_no_test: проверено {checked} (кадров {len(rows)}) — kr-test нет", flush=True)
    z = np.load(P.PROTOCOL / "trainpool.npz")
    names = [str(n) for n in z["feature_names"]]
    feats, fslugs = z["features"], z["feat_slugs"]
    frames = []
    for r in rows:
        n = sum(1 for s in fslugs[r["row"]] if str(s))
        slugs = tuple(str(s) for s in fslugs[r["row"]][:n])
        assert slugs == tuple(c[0] for c in r["cv"][:n]), r["id"]
        X = feats[r["row"], :n].astype(np.float64)
        read = read_of(r["read"])
        failure = reader_failure({"status": r["vlm_status"], "lines": ["x"] * r["vlm_lines"]})
        frames.append(
            Frame(
                i=r["row"],
                id=r["id"],
                set=r["set"],
                source=source_of(r),
                group=r["group"],
                slug=r["slug"],
                acceptable=tuple(r["acceptable"] or ()),
                slugs=slugs,
                X=X,
                fields=read.fields,
                failure=failure,
                goal_answer=r["answer"],
                goal_split=r.get("goal_split"),
                kr_test_wine=bool(r.get("kr_test_wine")),
                cv_first=r["cv"][0][0] if r["cv"] else None,
            )
        )
    return frames, names


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


# ------------------------------------------------------------------ групповые фолды
def group_folds(frames: Sequence[Frame], k: int = N_FOLDS, seed: int = FOLD_SEED) -> dict[str, int]:
    """Группа вина → фолд. Жадно, крупные группы первыми, штраф — перекос по срезам.

    Все кадры группы (студия, её «телефонная» копия, v2, kr-dev, same_packshot) — в одном фолде.
    """
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in frames:
        by_group[f.group][f.source] += 1
    rng = np.random.default_rng(seed)
    keys = sorted(by_group)
    rng.shuffle(keys)
    keys.sort(key=lambda g: -sum(by_group[g].values()))
    sources = sorted({f.source for f in frames})
    load = np.zeros((k, len(sources)))
    total = np.zeros(k)
    out: dict[str, int] = {}
    for g in keys:
        vec = np.array([by_group[g].get(s, 0) for s in sources], dtype=float)
        best, best_cost = 0, math.inf
        for j in range(k):
            new = load.copy()
            new[j] += vec
            tot = total.copy()
            tot[j] += vec.sum()
            cost = float(np.abs(new - new.mean(axis=0)).sum() + np.abs(tot - tot.mean()).sum())
            if cost < best_cost - 1e-9:
                best, best_cost = j, cost
        load[best] += vec
        total[best] += vec.sum()
        out[g] = best
    return out


# ------------------------------------------------------------------ дизайн обучения
def design(
    frames: Sequence[Frame], cols: Sequence[int] | None = None, extra: Sequence[str] = ()
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X, y, qid для listwise: кадры без верного в top-20 выпадают (как в `LogisticRanker.fit`).

    Прочие верные (acceptable, кроме выбранного положительного) из списка кадра убираются: они
    не отрицательные, а у listwise положительный один.
    """
    Xs, ys, qs = [], [], []
    for q, f in enumerate(frames):
        pos = f.pos
        if pos is None or f.failure is not None:  # нет верного в top-20 или Э2 (признаков нет)
            continue
        keep = [j for j, s in enumerate(f.slugs) if j == pos or s not in f.labels]
        X = f.X if cols is None else f.X[:, cols]
        if extra:
            X = np.hstack([X, np.column_stack([f.extra[e] for e in extra])])
        Xs.append(X[keep])
        y = np.zeros(len(keep))
        y[keep.index(pos)] = 1.0
        ys.append(y)
        qs.append(np.full(len(keep), q))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)


def fit_ranker(
    frames: Sequence[Frame],
    names: Sequence[str],
    *,
    l2: float,
    signed: bool = True,
    cols: Sequence[int] | None = None,
    extra: Sequence[str] = (),
    all_names: Sequence[str] | None = None,
) -> LogisticRanker:
    base = [names[c] for c in cols] if cols is not None else list(names)
    use = base + list(extra)
    # Признаки других треков — со знаком из EXTRA_SIGNS (по умолчанию свободны).
    signs = (feature_signs(base) + [EXTRA_SIGNS.get(e, 0) for e in extra]) if signed else [0] * len(use)
    X, y, q = design(frames, cols, extra)
    model = LogisticRanker(use, l2=l2, loss="listwise", signs=signs)
    model.fit(X, y, q)
    return model


def frame_matrix(f: Frame, cols: Sequence[int] | None = None, extra: Sequence[str] = ()) -> np.ndarray:
    X = f.X if cols is None else f.X[:, cols]
    if extra:
        X = np.hstack([X, np.column_stack([f.extra[e] for e in extra])])
    return X


def scores_of(model: LogisticRanker, f: Frame, cols=None, extra=()) -> np.ndarray:
    return model.decision_function(frame_matrix(f, cols, extra))


def calibrate(model: LogisticRanker, frames: Sequence[Frame], score_list: Sequence[np.ndarray]) -> float:
    """Температура по счётам вне фолда (как `bench/train_resolve.temperature_from`)."""
    flat, ids, correct = [], [], []
    for q, (f, s) in enumerate(zip(frames, score_list, strict=True)):
        if not len(s):
            continue
        flat.append(s)
        ids.append(np.full(len(s), q))
        correct.append(float(f.ok(f.slugs[int(np.argmax(s))])))
    T = fit_temperature(np.concatenate(flat), QueryBlocks.from_ids(np.concatenate(ids)), np.asarray(correct))
    model.temperature_ = T
    return T


# ------------------------------------------------------------------ слой выбора
def query_features_of(f: Frame, names: Sequence[str], cols=None, extra=()) -> QueryFeatures:
    use = [names[c] for c in cols] if cols is not None else list(names)
    use += list(extra)
    X = frame_matrix(f, cols, extra)
    rows = tuple({n: float(v) for n, v in zip(use, X[i], strict=True)} for i in range(len(X)))
    return QueryFeatures(f.slugs, rows)


def select(
    model: LogisticRanker,
    f: Frame,
    attrs: Any,
    names: Sequence[str],
    *,
    rules: bool = True,
    cols=None,
    extra=(),
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Ответ сервиса по записанным признакам: Э2, ранкер, H5 / P1 (при `rules`)."""
    if f.failure is not None:
        return {"answer": f.cv_first, "p1": None, "path": "e2"}
    if not f.slugs:
        return {"answer": None, "p1": None, "path": "empty"}
    s = scores if scores is not None else scores_of(model, f, cols, extra)
    order = sorted(range(len(s)), key=lambda i: (-float(s[i]), i))
    ranked = [f.slugs[i] for i in order]
    probs = softmax([float(s[i]) for i in order], model.temperature_)
    p1 = float(probs[0])
    if not rules:
        return {"answer": ranked[0], "p1": p1, "path": "ranker"}
    qf = query_features_of(f, names, cols, extra)
    if p1 < AMBIGUOUS_P_TOP1:
        out = list(rerank_ambiguous(model, qf, f.fields, attrs))
        return {"answer": out[0], "p1": p1, "path": "h5", "ranker": ranked[0]}
    guarded = list(block_bonus_flip(model, qf, ranked, f.fields, attrs))
    return {"answer": guarded[0], "p1": p1, "path": "p1" if guarded[0] != ranked[0] else "ranker",
            "ranker": ranked[0]}


# ------------------------------------------------------------------ счёт
SOURCES = ("studio", "studio_phone", "v2_R", "v2_M", "v2_L", "kr_dev", "kr_dev_sp")


def _src(*names: str):
    return lambda f: f.source in names


#: Срезы отчёта. `studio_goaltest` — студийные кадры, которых `-goal` не видел при обучении
#: (goal_split = test); на `goal_split = dev` его ответы в выборке обучения.
SLICES = {
    **{s: _src(s) for s in SOURCES},
    "studio_all": _src("studio", "studio_phone"),
    "studio_goaltest": lambda f: f.source in ("studio", "studio_phone") and f.goal_split == "test",
    "v2": _src("v2_R", "v2_M", "v2_L"),
    "field_main": _src("v2_R", "v2_M", "v2_L", "kr_dev"),
    "all": lambda f: True,
}


def tally(frames: Sequence[Frame], ok: Sequence[bool]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, pred in SLICES.items():
        idx = [i for i, f in enumerate(frames) if pred(f)]
        if not idx:
            continue
        k = sum(bool(ok[i]) for i in idx)
        out[name] = {"n": len(idx), "ok": k, "pct": round(100 * k / len(idx), 2)}
    return out


def paired(
    frames: Sequence[Frame], a: Sequence[bool], b: Sequence[bool], *, n_boot: int = 10000, seed: int = 0
) -> dict[str, dict[str, Any]]:
    """B против A по срезам: починки / поломки, тест знаков и 95 % интервал разности (п.п.)
    бутстрэпом по группам вин (кадры группы — вместе)."""
    out: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(seed)
    for name, pred in SLICES.items():
        idx = [i for i, f in enumerate(frames) if pred(f)]
        if not idx:
            continue
        fixes = sum(1 for i in idx if b[i] and not a[i])
        breaks = sum(1 for i in idx if a[i] and not b[i])
        keys = sorted({frames[i].group for i in idx})
        pos = {g: j for j, g in enumerate(keys)}
        d = np.zeros(len(keys))
        c = np.zeros(len(keys))
        for i in idx:
            j = pos[frames[i].group]
            d[j] += float(b[i]) - float(a[i])
            c[j] += 1
        draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
        diff = 100 * d[draws].sum(axis=1) / c[draws].sum(axis=1)
        out[name] = {
            "n": len(idx),
            "a": sum(bool(a[i]) for i in idx),
            "b": sum(bool(b[i]) for i in idx),
            "fixes": fixes,
            "breaks": breaks,
            "sign_p": round(P.sign_test(fixes, breaks), 4),
            "diff_pp": round(100 * (fixes - breaks) / len(idx), 2),
            "diff_ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)],
        }
    return out


def ci_micro(frames: Sequence[Frame], ok: Sequence[bool], name: str) -> list[float]:
    pred = SLICES[name]
    idx = [i for i, f in enumerate(frames) if pred(f)]
    return P.boot_ci_by_group([ok[i] for i in idx], [frames[i].group for i in idx])


def group_names() -> dict[str, tuple[str, ...]]:
    return dict(FEATURE_GROUPS)


def cols_without(names: Sequence[str], drop_groups: Iterable[str]) -> list[int]:
    from app.resolve.features import group_of

    drop = set(drop_groups)
    return [j for j, n in enumerate(names) if group_of(n) not in drop]
