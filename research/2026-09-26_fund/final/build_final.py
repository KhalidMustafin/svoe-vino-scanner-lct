"""Итоговые артефакты двух кандидатов PREREG_final — на всём пуле обучения и разработки, kr-test нет.

1. Карта `adapter-lw` (оба кандидата). Обученное выбеливание (Radenović LW) на 1 529 кадрах каталога
   пула — матрица трека adapter `adapter/candidate/lw_candidate.npz` (sha1 `ae96acfb…`), собранная
   `build_candidate.py`. Здесь она пересобирается тем же рецептом (`screen.fit_method("lw_nested")`,
   seed 20260926) для проверки воспроизводимости и кладётся в формат сервиса
   (`app/features/adapter.py`, `svs-cv-adapter/1`) с sha1 индекса снимка и двумя порогами шкалы
   счёта (ниже). Массивы `mean` и `W` — байт в байт из `lw_candidate.npz`.
2. Пороги шкалы счёта CV с адаптером (на «то же вино» не влияют: `SVS_ABSTAIN=off`) — теми же
   правилами, что у продукта, по выдаче адаптера вне фолда (`adapter/screen/lwn-pw.npz`):
   - `suggest_not_found_visual_max` (у продукта 0,8024 — «наибольший порог, при котором флаг стоит не
     больше чем на 7 из 353 кадров каталога v2»): 8-й снизу счёт top-1 v2 с адаптером, вниз до 1e-4;
   - `abstain_visual_floor` (у продукта 0,75): та же доля кадров v2 + ooc_v2 ниже порога, что у 0,75
     на счёте продукта (сопоставление по рангу), вниз до 1e-4.
3. Ранкер `adapter-lw-ranker`. Рецепт `-goal` (34 признака `resolve-features/3`, listwise, L2 0,01,
   знаки `-goal`) на всех кадрах пула с меткой и верным в top-20, по выдаче адаптера, подогнанной
   перекрёстно: пул делится на 4 группы вин (seed 20260926 + 105), карта каждой группы учится на
   трёх других (`lw_nested`, seed 20260926 + 1050 + j) — ни один кадр не видит карту, обученную на
   нём. Температура — по вне-фолдовым счётам ранкера внутреннего 4-фолда (seed 20260926 + 205).
   Ровно `adapter/ranker_folds.py::train_ranker` — рецепт, оценённый вне фолда (конфигурация D).

Выход: `svs-logs/somm-2409/fund/final/cv-adapter-lw.npz`, `configs/resolve/s2so400m-vlm35-lw-pool.json`
(и копия рядом с картой), `final/build_final.json` — sha1, пороги, выбор усадки, время.

    PYTHONPATH=. python research/2026-09-26_fund/final/build_final.py
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
sys.path.insert(0, str(FUND_DIR / "adapter"))
sys.path.insert(0, str(FUND_DIR))

import adapter_core as AC
import build_candidate as BC
import protocol as P
import ranker_folds as RF
import screen as SC

OUT = P.FUND / "final"
REPO = FUND_DIR.parents[1]
ADAPTER_FILE = OUT / "cv-adapter-lw.npz"
RANKER_NAME = "s2so400m-vlm35-lw-pool.json"
RANKER_REPO = REPO / "configs" / "resolve" / RANKER_NAME
LW_SOURCE_SHA1 = "ae96acfb7c02b9074b2adcf6c697993157f67abf"  # adapter/PREREG.md
SEED = 20260926
XFIT_SEED = SEED + 105
LW_SEED = SEED + 1050
T_SEED = SEED + 205
K_XFIT = 4
PRODUCT_SUGGEST = 0.8024  # app/api/after_layer.py: SUGGEST_NOT_FOUND_VISUAL_MAX
PRODUCT_FLOOR = 0.75  # app/resolve/rerank.py: RerankConfig.visual_floor
SUGGEST_MAX_FLAGGED = 7  # правило продукта: не больше 7 из 353 кадров каталога v2


def floor4(x: float) -> float:
    return math.floor(x * 1e4) / 1e4


def thresholds(pool: AC.Pool, log: Any) -> dict[str, Any]:
    """Пороги шкалы счёта с адаптером по выдаче вне фолда (`screen/lwn-pw.npz`)."""
    z = np.load(AC.OUT / "screen" / "lwn-pw.npz")
    assert [str(x) for x in z["ids"]] == [r["id"] for r in pool.rows]
    ad_top = z["top_score"][:, 0].astype(np.float64)
    pr_top = np.array([r["cv"][0][1] for r in pool.rows], dtype=np.float64)
    v2 = pool.sets == "catalog_v2"
    ooc = pool.sets == "ooc_v2"
    both = v2 | ooc
    # подсказка «нет в каталоге»: наибольший порог с ≤ 7 флагами на 353 кадрах v2
    s = np.sort(ad_top[v2])
    suggest = floor4(float(s[SUGGEST_MAX_FLAGGED]))
    # S10: та же доля кадров v2 + ooc_v2 ниже порога, что у 0,75 на счёте продукта
    k = int((pr_top[both] < PRODUCT_FLOOR).sum())
    floor = floor4(float(np.sort(ad_top[both])[k]))
    out = {
        "source": "adapter/screen/lwn-pw.npz (выдача адаптера вне фолда, 5 групп вин, seed 20260926)",
        "suggest_not_found_visual_max": suggest,
        "abstain_visual_floor": floor,
        "check": {
            "v2_n": int(v2.sum()),
            "ooc_v2_n": int(ooc.sum()),
            "suggest_flags_v2": {"product@0.8024": int((pr_top[v2] < PRODUCT_SUGGEST).sum()),
                                 "adapter": int((ad_top[v2] < suggest).sum())},
            "suggest_flags_ooc_v2": {"product@0.8024": int((pr_top[ooc] < PRODUCT_SUGGEST).sum()),
                                     "adapter": int((ad_top[ooc] < suggest).sum())},
            "floor_below_v2_ooc": {"product@0.75": k, "adapter": int((ad_top[both] < floor).sum())},
            "floor_below_ooc_v2": {"product@0.75": int((pr_top[ooc] < PRODUCT_FLOOR).sum()),
                                   "adapter": int((ad_top[ooc] < floor).sum())},
            "top1_mean": {"v2_product": round(float(pr_top[v2].mean()), 4),
                          "v2_adapter": round(float(ad_top[v2].mean()), 4),
                          "ooc_product": round(float(pr_top[ooc].mean()), 4),
                          "ooc_adapter": round(float(ad_top[ooc].mean()), 4)},
        },
    }
    log(f"пороги: {json.dumps(out, ensure_ascii=False)}")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    import replay as R
    import torch

    from app.features.adapter import LinearAdapter, file_sha1
    from app.resolve.learned import LogisticRanker

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = (OUT / "build_final.log").open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.perf_counter()
    rp = R.FundReplay()  # флаг выключен: индекс и ранкер продукта
    assert rp.settings.candidate == "off" and rp.svc.index.adapter is None
    ix = AC.index_data(rp.svc)
    pool = AC.load_pool(ix.slug_order, log=log)  # страж kr-test внутри
    allidx = np.arange(len(pool))
    kr_test = set(P.load_split()["kr_test"])
    assert not kr_test & {r["id"] for r in pool.rows}
    index_sha1 = P.sha1_file(rp.settings.index_path)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    # ---- 1. карта: матрица трека adapter и её пересборка тем же рецептом
    src = BC.CAND / "lw_candidate.npz"
    assert P.sha1_file(src) == LW_SOURCE_SHA1, "lw_candidate.npz не та, что в adapter/PREREG.md"
    lw = BC.load_candidate(src)
    t1 = time.perf_counter()
    lw2, fi = SC.fit_method("lw_nested", BC.PARAMS, pool, allidx, ix, SEED, log)
    rebuild = {
        "shrink": fi["shrink"],
        "inner_top1_by_shrink": fi["inner_top1"],
        "n_train_frames": fi["n_train"],
        "n_pairs": fi["n_pairs"],
        "W_equal": bool(np.array_equal(lw2.W, lw.W)),
        "mean_equal": bool(np.array_equal(lw2.mean, lw.mean)),
        "W_max_abs_diff": float(np.abs(lw2.W - lw.W).max()),
        "seconds": round(time.perf_counter() - t1, 1),
    }
    log(f"пересборка карты: {json.dumps(rebuild, ensure_ascii=False)}")

    # ---- 2. пороги шкалы, 3. файл сервиса
    th = thresholds(pool, log)
    z = np.load(src)
    meta = {
        "kind": "lw",
        "name": str(z["name"]),
        "shrink": float(z["shrink"]),
        "model": rp.svc.index.meta.model,
        "index_sha1": index_sha1,
        "source_file": "svs-logs/somm-2409/fund/adapter/candidate/lw_candidate.npz",
        "source_sha1": LW_SOURCE_SHA1,
        "trainpool_sha1": {"jsonl": P.sha1_file(P.PROTOCOL / "trainpool.jsonl"),
                           "npz": P.sha1_file(P.PROTOCOL / "trainpool.npz")},
        "train": "1 529 кадров каталога пула (pairs, pairs_phone, catalog_v2, kr_dev, kr_dev_sp), "
                 "пары per_window, 6 116 пар; kr-test нет",
        "suggest_not_found_visual_max": th["suggest_not_found_visual_max"],
        "abstain_visual_floor": th["abstain_visual_floor"],
        "thresholds_rule": "build_final.py: правила продукта на выдаче адаптера вне фолда",
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": commit,
    }
    adapter = LinearAdapter(lw.mean, lw.W, meta)
    adapter.save(ADAPTER_FILE)
    back = LinearAdapter.load(ADAPTER_FILE)
    assert np.array_equal(back.W, lw.W) and np.array_equal(back.mean, lw.mean)
    log(f"карта сервиса: {ADAPTER_FILE} sha1 {file_sha1(ADAPTER_FILE)}, содержимое {back.sha1}")

    # ---- 4. ранкер по перекрёстно подогнанной выдаче адаптера
    t2 = time.perf_counter()
    xfit = AC.group_kfold(pool.groups, pool.sets, K_XFIT, XFIT_SEED)
    cv_all: dict[int, tuple[list, float]] = {}
    xinfo = []
    for j in range(K_XFIT):
        itr, ite = allidx[xfit != j], allidx[xfit == j]
        assert not set(pool.groups[itr]) & set(pool.groups[ite])
        mj, fij = SC.fit_method("lw_nested", {"pairs": "per_window"}, pool, itr, ix, LW_SEED + j, log)
        cv_all.update(RF.cv_for(mj, pool, ite, ix))
        xinfo.append({"part": j, "n_fit": int(fij["n_train"]), "n_scored": len(ite),
                      "shrink": fij["shrink"]})
        log(f"перекрёстная карта {j}: {xinfo[-1]}")
    goal = rp.svc.model
    model = RF.train_ranker(rp, pool, allidx, cv_all, T_SEED, {"top_k": 20})
    fit_s = time.perf_counter() - t2
    model.meta = {
        **{k: goal.meta[k] for k in ("reader_keys", "readers", "top_k", "variant", "feature_version",
                                    "groups", "signed")},
        "gt_tokens_sha1": rp.svc.attrs.meta.get("sha1"),
        "cv_adapter_sha1": adapter.sha1,
        "cv_adapter_file": "cv-adapter-lw.npz",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "track": "fund/final (26.09): кандидат adapter-lw-ranker, research/2026-09-26_fund/PREREG_final.md",
        "recipe": "как -goal: 34 признака resolve-features/3, listwise, L2 0,01, знаки -goal; выдача CV — "
                  "адаптер LW, перекрёстно подогнанный 4 группами вин; T — вне-фолдовые счёта внутреннего 4-фолда",
        "sets": ["pairs", "pairs_phone", "catalog_v2", "kr_dev", "kr_dev_sp"],
        "queries": int(model.fit_info.get("queries", 0)),
        "seeds": {"xfit_split": XFIT_SEED, "lw_nested": [LW_SEED + j for j in range(K_XFIT)], "temperature": T_SEED},
        "xfit": xinfo,
        "trainpool_sha1": meta["trainpool_sha1"]["jsonl"],
        "kr_test": "нет: страж protocol.assert_no_test на всех кадрах пула",
        "code_commit": commit,
    }
    model.save(RANKER_REPO)
    shutil.copyfile(RANKER_REPO, OUT / RANKER_NAME)
    again = LogisticRanker.load(RANKER_REPO)
    assert again.temperature_ == model.temperature_
    log(f"ранкер: {RANKER_REPO} sha1 {P.sha1_file(RANKER_REPO)}, T {model.temperature_:.4f}, "
        f"fit {json.dumps(model.fit_info, ensure_ascii=False)} ({fit_s:.0f} с)")

    report = {
        "what": "итоговые артефакты кандидатов PREREG_final (adapter-lw, adapter-lw-ranker), весь пул, kr-test нет",
        "adapter": {
            "file": str(ADAPTER_FILE),
            "file_sha1": file_sha1(ADAPTER_FILE),
            "content_sha1": adapter.sha1,
            "source_sha1": LW_SOURCE_SHA1,
            "index_sha1": index_sha1,
            "rebuild_same_recipe": rebuild,
            "thresholds": th,
        },
        "ranker": {
            "file": str(RANKER_REPO),
            "sha1": P.sha1_file(RANKER_REPO),
            "l2": model.l2,
            "temperature": model.temperature_,
            "fit_info": model.fit_info,
            "goal_temperature": goal.temperature_,
            "weights": {n: round(float(w), 4) for n, w in zip(model.feature_names, model.coef_, strict=True)},
            "goal_weights": {n: round(float(w), 4) for n, w in zip(goal.feature_names, goal.coef_, strict=True)},
            "xfit": xinfo,
            "fit_seconds": round(fit_s, 1),
        },
        "code_commit": commit,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT / "build_final.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: report[k] for k in ("adapter", "code_commit", "wall_s")}, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
