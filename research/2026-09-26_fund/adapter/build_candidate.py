"""Кандидат «адаптер LW»: обученное выбеливание на всём пуле обучения и разработки (без kr-test).

Рецепт — ровно тот, что оценён вне фолда (`screen.py --method lw_nested`, `ranker_folds.py`):
пары «окно кадра — вид эталона верной карточки» (`per_window`: для каждого из 4 окон — лучший вид
верной карточки), `C_S` разностей, усадка из сетки {0,3; 1; 2; 3; 5; 10} — внутренним 4-фолдом по
группам вин пула (полнота CV top-1), поворот на главные оси индекса. Слой выбора не меняется:
ранкер `-goal`, H5, P1, Э2 — как в продукте.

Пишет `adapter/candidate/lw_candidate.npz` (mean, W, усадка) и `lw_candidate.json` (sha1 файлов
пула, индекса, матрицы; выбор усадки; проверки). Затем — проверка пути сервиса: на кадрах kr-dev
`FundReplay.scan` с подменой поиска (`cv_fn`) даёт те же ответы, что слой выбора по записи.

    PYTHONPATH=. python research/2026-09-26_fund/adapter/build_candidate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import adapter_core as C  # noqa: E402
import linear_maps as L  # noqa: E402
import protocol as P  # noqa: E402
import screen as SC  # noqa: E402

CAND = C.OUT / "candidate"
PARAMS = {"pairs": "per_window", "grid": [0.3, 1.0, 2.0, 3.0, 5.0, 10.0]}
SEED = 20260926


def load_candidate(path: Path = CAND / "lw_candidate.npz") -> L.LinearMap:
    z = np.load(path)
    return L.LinearMap(z["mean"].astype(np.float32), z["W"].astype(np.float32), str(z["name"]))


def make_cv_fn(rp: Any, lw: L.LinearMap) -> Any:
    """Поиск сервиса поверх карты: запрос и индекс проецируются, дальше zmax и выдача как в `_rank`."""
    ix = C.index_data(rp.svc)
    Ia = lw.np_i(ix.vectors)

    def cv_fn(Q: np.ndarray) -> Any:
        scores, best_row = C.cv_scores_np(lw.np_q(Q), Ia, ix)
        return rp.candidates_from_scores(scores, best_row)

    return cv_fn


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    import replay as R

    CAND.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rp = R.FundReplay()
    ix = C.index_data(rp.svc)
    pool = C.load_pool(ix.slug_order)
    allidx = np.arange(len(pool))
    lw, fi = SC.fit_method("lw_nested", PARAMS, pool, allidx, ix, SEED, print)
    print(f"усадка {fi['shrink']}, пар {fi['n_pairs']}, внутренний top-1 {fi['inner_top1']}")
    dst = CAND / "lw_candidate.npz"
    np.savez(dst, mean=lw.mean, W=lw.W, name=np.array(lw.name), shrink=np.array(fi["shrink"]))
    # проверка пути сервиса на kr-dev: scan с cv_fn == resolve по той же выдаче
    rp.cv_fn = make_cv_fn(rp, lw)
    zq = np.load(P.PROTOCOL / "trainpool.npz")["qemb"]
    kr = [i for i in allidx if pool.sets[i] == "kr_dev"]
    same = ok = 0
    t1 = time.perf_counter()
    for i in kr:
        r = pool.rows[i]
        row = rp.scan(Path(r["image"]).read_bytes(), zq[r["row"]])
        assert row["ok"], (r["id"], row.get("error"))
        v = R.visual_of(row["cv"], row["margin"], rp.svc.index.meta.model)
        again = rp.resolve(v, R.read_of(r["read"]), vlm_status=r["vlm_status"], vlm_lines=r["vlm_lines"])
        same += again["answer"] == row["answer"]
        ok += P.correct(row["answer"], r)
    ms_scan = 1000 * (time.perf_counter() - t1) / len(kr)
    # цена: проекция 4 окон запроса и индекса
    Q = pool.Q[0]
    t2 = time.perf_counter()
    for _ in range(200):
        lw.np_q(Q)
    ms_q = 1000 * (time.perf_counter() - t2) / 200
    t3 = time.perf_counter()
    lw.np_i(ix.vectors)
    s_index = time.perf_counter() - t3
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    meta = {
        "what": "кандидат adapter-lw: обученное выбеливание (Radenović LW) поверх SigLIP2 so400m, обе стороны",
        "recipe": PARAMS,
        "shrink": fi["shrink"],
        "inner_top1_by_shrink": fi["inner_top1"],
        "n_train_frames": fi["n_train"],
        "n_pairs": fi["n_pairs"],
        "code_commit_at_build": commit,
        "sha1": {
            "lw_candidate.npz": P.sha1_file(dst),
            "trainpool.jsonl": P.sha1_file(P.PROTOCOL / "trainpool.jsonl"),
            "trainpool.npz": P.sha1_file(P.PROTOCOL / "trainpool.npz"),
            "index": P.sha1_file(rp.settings.index_path),
        },
        "check_kr_dev_scan_eq_resolve": f"{same}/{len(kr)}",
        "kr_dev_in_sample_same_wine": f"{ok}/{len(kr)} (в выборке обучения — не оценка)",
        "cost": {
            "W_shape": list(lw.W.shape),
            "W_mb_float32": round(lw.W.nbytes / 2**20, 2),
            "query_projection_ms_4_windows": round(ms_q, 3),
            "index_projection_s_6312_rows": round(s_index, 3),
            "scan_replay_ms_per_frame_cpu": round(ms_scan, 1),
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (CAND / "lw_candidate.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
