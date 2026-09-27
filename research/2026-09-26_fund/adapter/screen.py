"""Адаптеры поиска по GroupKFold пула: выдача CV вне фолда и её полнота (top-1/5/20).

Каждый метод учится только на кадрах обучающих фолдов (в каталоге, наборы `sets`), индекс и
запрос проецируются им, счёт — zmax сервиса. Выход — `adapter/screen/<имя>.npz` (top-20 slug,
счёт, строка индекса, отрыв по каталогу для каждого кадра пула вне фолда) и `<имя>.json`
(полнота по наборам). Пул — `trainpool.*`, kr-test нет (`assert_no_test` в `load_pool`).

    PYTHONPATH=. python research/2026-09-26_fund/adapter/screen.py --method lw --params '{"shrink":0.1}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import adapter_core as C  # noqa: E402
import linear_maps as L  # noqa: E402

SEED = 20260926
K = 5
TOPK = 20
EVAL_SETS = ("catalog_v2", "kr_dev", "kr_dev_sp", "pairs", "pairs_phone")


def scores_from(Qa: np.ndarray, Ia: np.ndarray, ix: C.IndexData) -> np.ndarray:
    Ivm = torch.from_numpy(np.ascontiguousarray(Ia[ix.perm], dtype=np.float32))
    slug_vm = torch.from_numpy(ix.slug_ids[ix.perm])
    out = np.zeros((len(Qa), ix.n_slugs), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(Qa), 128):
            q = torch.from_numpy(np.ascontiguousarray(Qa[a : a + 128], dtype=np.float32))
            out[a : a + 128] = C.zmax_torch(q, Ivm, ix.n_per_view, slug_vm, ix.n_slugs).numpy()
    return out


def best_rows(Qa: np.ndarray, Ia: np.ndarray, ix: C.IndexData, slugs: np.ndarray) -> np.ndarray:
    """Номер строки индекса, давшей счёт slug (как `cv_scores` сервиса) — для поля `view`."""
    from app.features.index import align_pairs

    sims = align_pairs(Ia @ Qa.T, ix.view_rows, None)
    best = sims.max(axis=1)
    out = []
    for s in slugs:
        rows = np.flatnonzero(ix.slug_ids == s)
        out.append(int(rows[np.argmax(best[rows])]))
    return np.asarray(out)


def matching_pairs(pool: C.Pool, idx: np.ndarray, ix: C.IndexData, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Пары «окно кадра — вид эталона верной карточки» для обученного выбеливания."""
    V = ix.vectors
    qs, ps = [], []
    for i in idx:
        rows = np.flatnonzero(np.isin(ix.slug_ids, pool.pos[i]))
        S = pool.Q[i] @ V[rows].T  # (4, rows)
        if mode == "best":
            w, r = np.unravel_index(int(np.argmax(S)), S.shape)
            qs.append(pool.Q[i, w])
            ps.append(V[rows[r]])
        elif mode == "per_window":
            for w in range(S.shape[0]):
                r = int(np.argmax(S[w]))
                qs.append(pool.Q[i, w])
                ps.append(V[rows[r]])
        else:
            raise ValueError(mode)
    return np.asarray(qs), np.asarray(ps)


def fit_method(method: str, params: dict[str, Any], pool: C.Pool, train_idx: np.ndarray,
               ix: C.IndexData, fold_seed: int, log: Any) -> tuple[Any, dict[str, Any]]:
    sets = tuple(params.get("sets") or C.Config().sets)
    keep = np.array([i for i in train_idx if len(pool.pos[i]) and pool.sets[i] in sets], dtype=np.int64)
    if method == "none":
        return None, {}
    if method == "pcaw":
        m = L.pca_whitening(ix.vectors, alpha=params.get("alpha", 0.5), shrink=params.get("shrink", 0.1),
                            dims=params.get("dims"))
        return m, {"n_train": 0}
    if method == "lw":
        q, p = matching_pairs(pool, keep, ix, params.get("pairs", "best"))
        m = L.learned_whitening(q, p, ix.vectors, shrink=params.get("shrink", 0.1), dims=params.get("dims"),
                                rotate=params.get("rotate", True))
        return m, {"n_train": len(keep), "n_pairs": len(q)}
    if method == "lw_nested":
        # усадка LW выбирается внутренним 4-фолдом по группам обучающей части (полнота CV top-1),
        # тестовый фолд в выборе не участвует
        grid = params.get("grid", [0.3, 1.0, 2.0, 3.0, 5.0, 10.0])
        base = {k: v for k, v in params.items() if k != "grid"}
        inner = C.group_kfold(pool.groups[train_idx], pool.sets[train_idx], 4, fold_seed + 500)
        hits = {g: 0 for g in grid}
        for j in range(4):
            itr, ite = train_idx[inner != j], train_idx[inner == j]
            ite = np.array([i for i in ite if len(pool.pos[i]) and pool.sets[i] in sets], dtype=np.int64)
            for g in grid:
                mj, _ = fit_method("lw", {**base, "shrink": g}, pool, itr, ix, fold_seed, log)
                Qa, Ia = transform(mj, pool.Q[ite], ix.vectors)
                top = scores_from(Qa, Ia, ix).argmax(1)
                hits[g] += int(sum(top[k] in pool.pos[i] for k, i in enumerate(ite)))
        best = max(grid, key=lambda g: (hits[g], g))
        m, fi = fit_method("lw", {**base, "shrink": best}, pool, train_idx, ix, fold_seed, log)
        return m, {**fi, "shrink": best, "inner_top1": {str(g): h for g, h in hits.items()}}
    if method == "infonce":
        pre = params.get("pre")  # {"shrink": .., "pairs": ..}: сначала LW на том же фолде, адаптер — поверх
        cfg = C.cfg_from({**{k: v for k, v in params.items() if k != "pre"}, "seed": fold_seed})
        if pre:
            lw, _ = fit_method("lw", pre, pool, train_idx, ix, fold_seed, log)
            ix2, pool2 = project(lw, pool, ix)
            model, tl = C.train_adapter(pool2, train_idx, ix2, cfg, log=None)
            m = Chain(lw, model)
        else:
            model, tl = C.train_adapter(pool, train_idx, ix, cfg, log=None)
            m = model
        return m, {"n_train": tl.n_train, "n_val": tl.n_val, "best_epoch": tl.best_epoch,
                   "seconds": round(tl.seconds, 1), "history": tl.history}
    raise ValueError(method)


class Chain:
    """Сначала линейная карта (LW), затем обученный адаптер."""

    def __init__(self, first: Any, second: Any) -> None:
        self.first, self.second = first, second

    def np_q(self, X: np.ndarray) -> np.ndarray:
        return self.second.np_q(self.first.np_q(X))

    def np_i(self, X: np.ndarray) -> np.ndarray:
        return self.second.np_i(self.first.np_i(X))


def project(m: Any, pool: C.Pool, ix: C.IndexData) -> tuple[C.IndexData, C.Pool]:
    """Пул и индекс в пространстве линейной карты (для адаптера поверх неё)."""
    from dataclasses import replace

    Qa, Ia = transform(m, pool.Q, ix.vectors)
    return replace(ix, vectors=Ia), replace(pool, Q=np.ascontiguousarray(Qa, dtype=np.float32))


def transform(m: Any, pool_Q: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if m is None:
        return pool_Q, V
    Qa = m.np_q(pool_Q.reshape(-1, pool_Q.shape[-1])).reshape(len(pool_Q), pool_Q.shape[1], -1)
    return Qa, m.np_i(V)


def recall_table(pool: C.Pool, top_idx: np.ndarray, which: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sel = np.ones(len(pool), bool) if which is None else which
    for s in EVAL_SETS:
        ids = [i for i in np.flatnonzero((pool.sets == s) & sel) if pool.rows[i]["in_catalog"]]
        ranks = []
        for i in ids:
            hit = np.flatnonzero(np.isin(top_idx[i], pool.pos[i]))
            ranks.append(int(hit[0]) + 1 if len(hit) else 99)
        r = np.asarray(ranks)
        out[s] = {"n": len(ids), "top1": int((r <= 1).sum()), "top5": int((r <= 5).sum()),
                  "top20": int((r <= 20).sum())}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--name", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--split-seed", type=int, default=SEED, help="seed разреза по группам (устойчивость)")
    args = ap.parse_args()
    params = json.loads(args.params)
    torch.set_num_threads(args.threads)
    params.setdefault("threads", args.threads) if args.method == "infonce" else None
    name = args.name or f"{args.method}-" + "-".join(f"{k}{v}" for k, v in sorted(params.items()) if k != "threads")
    out_dir = C.OUT / "screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / f"{name}.log").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    import replay as R

    rp = R.FundReplay()
    ix = C.index_data(rp.svc)
    pool = C.load_pool(ix.slug_order, log=log)
    fold = C.group_kfold(pool.groups, pool.sets, K, args.split_seed)
    N = len(pool)
    top_idx = np.zeros((N, TOPK), np.int64)
    top_sc = np.zeros((N, TOPK), np.float32)
    top_row = np.zeros((N, TOPK), np.int64)
    margin = np.zeros(N, np.float32)
    info: dict[str, Any] = {"method": args.method, "params": params, "folds": []}
    t0 = time.perf_counter()
    for f in range(K):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        assert not set(pool.groups[tr]) & set(pool.groups[te])
        m, fi = fit_method(args.method, params, pool, tr, ix, args.split_seed + f, log)
        Qa, Ia = transform(m, pool.Q[te], ix.vectors)
        S = scores_from(Qa, Ia, ix)
        order = np.argsort(-S, axis=1, kind="stable")
        for k, i in enumerate(te):
            o = order[k, :TOPK]
            top_idx[i] = o
            top_sc[i] = S[k, o]
            top_row[i] = best_rows(Qa[k], Ia, ix, o)
            margin[i] = max(0.0, float(S[k, order[k, 0]] - S[k, order[k, 1]]))
        fi = {k: v for k, v in fi.items() if k != "history"} | ({"history": fi["history"]} if "history" in fi else {})
        info["folds"].append({"fold": f, "n_test": len(te), **fi})
        rt = recall_table(pool, top_idx, fold == f)
        log(f"fold {f}: {json.dumps({s: v['top1'] for s, v in rt.items()})} {json.dumps({k: v for k, v in fi.items() if k != 'history'})}")
    info["recall"] = recall_table(pool, top_idx)
    info["seconds"] = round(time.perf_counter() - t0, 1)
    tot = {k: sum(v[k] for v in info["recall"].values()) for k in ("n", "top1", "top5", "top20")}
    info["recall_total"] = tot
    log(f"OOF {name}: {json.dumps(info['recall'])} total {json.dumps(tot)} ({info['seconds']} s)")
    np.savez_compressed(out_dir / f"{name}.npz", ids=np.array([r["id"] for r in pool.rows]), fold=fold,
                        top_idx=top_idx, top_score=top_sc, top_row=top_row, margin=margin)
    (out_dir / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
