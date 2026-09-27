"""Честный сквозной счёт адаптера с ранкером, переобученным внутри фолдов (вложенная перекрёстная подгонка).

Ранкер `-goal` учился на счёте CV без адаптера; у адаптированного CV другое распределение счёта
(уровень, отрывы), поэтому сравниваются четыре конфигурации на одних и тех же внешних фолдах:

    A  CV продукта  + ранкер -goal        (= ответы продукта)
    B  CV адаптера  + ранкер -goal
    C  CV продукта  + ранкер, обученный в фолде
    D  CV адаптера  + ранкер, обученный в фолде
    E  то же, что D, но без правил H5 / P1 (ответ — top-1 ранкера; сбой читателя — CV top-1, Э2);
       справочно: кандидат PREREG_final оставляет правила как в продукте

Внешний фолд f (GroupKFold по винам, как `screen.py`): адаптер учится на обучающей части, даёт CV
тестовой. Ранкер для f учится на признаках обучающей части, посчитанных по CV **вне** адаптера —
внутренний 4-фолд по группам обучающей части (адаптер внутреннего фолда не видел кадр). Температура
ранкера — по его собственным вне-фолдовым счётам на той же части. Тестовые кадры f не участвуют
ни в чём, кроме итогового ответа. Так же будет собран кандидат: адаптер на всём пуле, ранкер — на
перекрёстно подогнанном CV пула.

Рецепт ранкера — как у `-goal`: 34 признака `resolve-features/3`, listwise, l2 = 0,01, знаки
`-goal`; обучается на всех кадрах каталога обучающей части с верным в top-20.

    PYTHONPATH=. python research/2026-09-26_fund/adapter/ranker_folds.py --method lw \
        --params '{"shrink":2.0,"pairs":"per_window"}' --name lw2pw
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
import e2e as E  # noqa: E402
import protocol as P  # noqa: E402
import screen as SC  # noqa: E402

K_INNER = 4


def cv_for(m: Any, pool: C.Pool, idx: np.ndarray, ix: C.IndexData) -> dict[int, tuple[list, float]]:
    """Выдача CV (кандидаты, отрыв) кадров `idx` под моделью `m` — как `candidates_from_scores`."""
    import replay as R

    Qa, Ia = SC.transform(m, pool.Q[idx], ix.vectors)
    S = SC.scores_from(Qa, Ia, ix)
    out = {}
    for k, i in enumerate(idx):
        o = np.argsort(-S[k], kind="stable")[:SC.TOPK]
        rows = SC.best_rows(Qa[k], Ia, ix, o)
        cands = [R.Candidate(slug=ix.slug_order[int(s)], score=float(max(0.0, S[k, s])),
                             view=ix.views[int(rw)], rank=r)
                 for r, (s, rw) in enumerate(zip(o, rows, strict=True), start=1)]
        out[int(i)] = (cands, max(0.0, float(S[k, o[0]] - S[k, o[1]])))
    return out


def features_of(rp: Any, pool: C.Pool, i: int, cv: tuple[list, float], names: list[str]) -> tuple[np.ndarray, list[str]]:
    import replay as R
    from app.resolve.features import query_features

    v = R.VisualResult(candidates=cv[0], margin=cv[1], timings_ms={}, model=rp.svc.index.meta.model)
    qf = query_features(v, {rp.svc.reader_key: R.read_of(pool.rows[i]["read"])}, rp.svc.attrs, top_k=20)
    return qf.matrix(names), list(qf.slugs)


def train_ranker(rp: Any, pool: C.Pool, idx: np.ndarray, cvs: dict[int, tuple[list, float]],
                 groups_seed: int, meta: dict[str, Any]) -> Any:
    """Ранкер рецепта -goal на кадрах `idx`; температура — по его вне-фолдовым счётам (внутренний разрез)."""
    from app.resolve.learned import LogisticRanker, QueryBlocks, fit_temperature

    goal = rp.svc.model
    names = list(goal.feature_names)
    Xs, ys, qids, keep = [], [], [], []
    for i in idx:
        if not len(pool.pos[i]):
            continue
        X, slugs = features_of(rp, pool, int(i), cvs[int(i)], names)
        pos = {pool.rows[i]["slug"], *(pool.rows[i].get("acceptable") or [])}
        hits = [k for k, s in enumerate(slugs) if s in pos]
        if not hits:
            continue
        rows = [k for k in range(len(slugs)) if k == hits[0] or k not in hits]  # один верный
        y = np.zeros(len(rows))
        y[rows.index(hits[0])] = 1
        Xs.append(X[rows])
        ys.append(y)
        qids.append(np.full(len(rows), int(i)))
        keep.append(int(i))

    def fit(sel: list[int]) -> Any:
        m = LogisticRanker(names, l2=goal.l2, loss="listwise", signs=goal.signs)
        m.fit(np.vstack([Xs[k] for k in sel]), np.concatenate([ys[k] for k in sel]),
              np.concatenate([qids[k] for k in sel]), meta=meta)
        return m

    # температура: вне-фолдовые счёта ранкера по внутренним группам
    grp = pool.groups[np.asarray(keep)]
    inner = C.group_kfold(grp, pool.sets[np.asarray(keep)], K_INNER, groups_seed)
    sc_all = [None] * len(keep)
    for j in range(K_INNER):
        tr = [k for k in range(len(keep)) if inner[k] != j]
        m = fit(tr)
        for k in range(len(keep)):
            if inner[k] == j:
                sc_all[k] = m.decision_function(Xs[k])
    scores = np.concatenate(sc_all)
    q = np.concatenate([qids[k] for k in range(len(keep))])
    correct = np.array([float(np.argmax(sc_all[k]) == int(np.argmax(ys[k]))) for k in range(len(keep))])
    T = fit_temperature(scores, QueryBlocks.from_ids(q), correct)
    model = fit(list(range(len(keep))))
    model.temperature_ = T
    return model


def resolve_with(rp: Any, model: Any, pool: C.Pool, idx: np.ndarray, cvs: dict[int, tuple[list, float]]) -> dict[int, str | None]:
    import replay as R

    old = rp.svc.model
    rp.svc.model = model
    try:
        out = {}
        for i in idx:
            r = pool.rows[i]
            c, mg = cvs[int(i)]
            v = R.VisualResult(candidates=c, margin=mg, timings_ms={}, model=rp.svc.index.meta.model)
            out[int(i)] = rp.resolve(v, R.read_of(r["read"]), vlm_status=r["vlm_status"],
                                     vlm_lines=r["vlm_lines"])["answer"]
        return out
    finally:
        rp.svc.model = old


def resolve_norules(rp: Any, model: Any, pool: C.Pool, idx: np.ndarray,
                    cvs: dict[int, tuple[list, float]]) -> dict[int, str | None]:
    """Ответ ранкера без H5 / P1: Э2 при сбое читателя, иначе top-1 ранкера (сбой ранкера — CV top-1)."""
    import replay as R
    from app.api.service import reader_failure
    from app.resolve.learned import rank_query_detailed

    out = {}
    for i in idx:
        r = pool.rows[i]
        c, mg = cvs[int(i)]
        if reader_failure({"status": r["vlm_status"], "lines": ["x"] * r["vlm_lines"]}) is not None:
            out[int(i)] = c[0].slug
            continue
        v = R.VisualResult(candidates=c, margin=mg, timings_ms={}, model=rp.svc.index.meta.model)
        ranking, _ = rank_query_detailed(model, v, {rp.svc.reader_key: R.read_of(r["read"])}, rp.svc.attrs)
        out[int(i)] = ranking.slugs[0] if ranking.slugs else c[0].slug
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--name", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--split-seed", type=int, default=SC.SEED, help="seed разреза по группам (устойчивость)")
    args = ap.parse_args()
    seed = args.split_seed
    params = json.loads(args.params)
    torch.set_num_threads(args.threads)
    if args.method == "infonce":
        params.setdefault("threads", args.threads)
    out_dir = C.OUT / "ranker_folds"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / f"{args.name}.log").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    import replay as R

    t0 = time.perf_counter()
    rp = R.FundReplay()
    ix = C.index_data(rp.svc)
    pool = C.load_pool(ix.slug_order, log=log)
    fold = C.group_kfold(pool.groups, pool.sets, SC.K, seed)
    goal = rp.svc.model
    ev = np.array([i for i in range(len(pool)) if pool.rows[i]["in_catalog"] and pool.sets[i] in E.SETS])
    base_cv_all = cv_for(None, pool, np.arange(len(pool)), ix)
    ans: dict[str, dict[int, Any]] = {k: {} for k in "ABCDE"}
    info: dict[str, Any] = {"method": args.method, "params": params, "split_seed": seed, "folds": []}
    for f in range(SC.K):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        te_ev = np.array([i for i in te if i in set(ev.tolist())])
        m, fi = SC.fit_method(args.method, params, pool, tr, ix, seed + f, log)
        cv_te = cv_for(m, pool, te_ev, ix)
        # внутренний разрез обучающей части: CV вне адаптера для признаков ранкера
        inner = C.group_kfold(pool.groups[tr], pool.sets[tr], K_INNER, seed + 100 + f)
        cv_tr: dict[int, tuple[list, float]] = {}
        for j in range(K_INNER):
            itr, ite = tr[inner != j], tr[inner == j]
            assert not set(pool.groups[itr]) & set(pool.groups[ite])
            mj, _ = SC.fit_method(args.method, params, pool, itr, ix, seed + 1000 + 10 * f + j, log)
            cv_tr.update(cv_for(mj, pool, ite, ix))
        meta = {"fold": f, "method": args.method, "params": params, "top_k": 20}
        rk_adapt = train_ranker(rp, pool, tr, cv_tr, seed + 200 + f, meta)
        rk_base = train_ranker(rp, pool, tr, base_cv_all, seed + 200 + f, meta | {"method": "none"})
        base_te = {int(i): base_cv_all[int(i)] for i in te_ev}
        ans["A"].update(resolve_with(rp, goal, pool, te_ev, base_te))
        ans["B"].update(resolve_with(rp, goal, pool, te_ev, cv_te))
        ans["C"].update(resolve_with(rp, rk_base, pool, te_ev, base_te))
        ans["D"].update(resolve_with(rp, rk_adapt, pool, te_ev, cv_te))
        ans["E"].update(resolve_norules(rp, rk_adapt, pool, te_ev, cv_te))
        fi = {k: v for k, v in fi.items() if k != "history"}
        info["folds"].append({"fold": f, "n_test": len(te_ev), **fi, "T_adapt": rk_adapt.temperature_,
                              "T_base": rk_base.temperature_})
        part = {k: sum(P.correct(ans[k][int(i)], pool.rows[i]) for i in te_ev) for k in "ABCDE"}
        log(f"fold {f}: n={len(te_ev)} {part} T_adapt={rk_adapt.temperature_:.3f} T_base={rk_base.temperature_:.3f} "
            f"({time.perf_counter() - t0:.0f} s)")
    prod = {int(i): pool.rows[i]["answer"] for i in ev}
    assert all(ans["A"][i] == prod[i] for i in prod), "A должен совпасть с продуктом"
    res = {
        "B_vs_A": E.compare(pool, ans["B"], ans["A"]),
        "C_vs_A": E.compare(pool, ans["C"], ans["A"]),
        "D_vs_A": E.compare(pool, ans["D"], ans["A"]),
        "D_vs_C": E.compare(pool, ans["D"], ans["C"]),
        "D_vs_B": E.compare(pool, ans["D"], ans["B"]),
        "E_vs_A": E.compare(pool, ans["E"], ans["A"]),
        "E_vs_D": E.compare(pool, ans["E"], ans["D"]),
    }
    nodev = {k: {i: a for i, a in v.items() if pool.rows[i].get("goal_split") != "dev"} for k, v in ans.items()}
    res_nodev = {
        "B_vs_A": E.compare(pool, nodev["B"], nodev["A"]),
        "D_vs_A": E.compare(pool, nodev["D"], nodev["A"]),
        "D_vs_C": E.compare(pool, nodev["D"], nodev["C"]),
    }
    for key, r in res.items():
        short = {s: f"{v['ok']} vs {v['ref_ok']} +{v['fix']}-{v['brk']}" for s, v in r.items() if s != "total"}
        t = r["total"]
        log(f"{key}: {json.dumps(short, ensure_ascii=False)} | total {t['ok']} vs {t['ref_ok']} +{t['fix']}-{t['brk']} p={t['p']}")
    info.update(results=res, results_no_pairs_dev=res_nodev, seconds=round(time.perf_counter() - t0, 1),
                answers={k: {pool.rows[i]["id"]: a for i, a in v.items()} for k, v in ans.items()})
    (out_dir / f"{args.name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
