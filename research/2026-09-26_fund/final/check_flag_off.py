"""Флаг кандидатов выключен — сервис ветки `fund` = продукт `after-search` бит в бит (v2 + kr-dev, 661 кадр).

Каждый вариант гоняется в своём процессе полным `FundReplay.scan` (разжатие кадра, виды, CV по
CPU-векторам, чтение из кэша стенда, разбор, признаки, ранкер, H5 / P1 / Э2, правило S10) и пишет
по строке на кадр: запись реплея (top-20 CV со счётами, отрыв, чтение, признаки, счёт ранкера,
ответ) и тело `/v1/eval/predict` с `evidence` — без времён (`timings_ms`, `elapsed_ms`, бюджет VLM,
которые от прогона к прогону разные).

Варианты:
- `after` — код `after-search` @1d5b4e9 (`git archive` в `--after-root`): пакет `app` берётся оттуда;
- `off` — код этой ветки без `SVS_CANDIDATE`; `off-explicit` — с `SVS_CANDIDATE=off`;
- `svc-lw`, `svc-lw-ranker` — флаг включён (`SVS_CANDIDATE`, `SVS_CV_ADAPTER` — карта `final/`);
- `res-lw`, `res-lw-ranker` — путь стенда, на котором кандидаты оценивались: флаг выключен, выдача
  CV — `adapter/build_candidate.make_cv_fn` по `lw_candidate.npz`, ранкер подменён в `svc.model`.

Проверки (`compare`): `off` и `off-explicit` = `after` байт в байт по всем строкам; ответы `after` =
ответы продукта из пула (`trainpool.jsonl`); `svc-*` = `res-*` по выдаче CV, признакам, счёту ранкера
и телу predict (кроме имени модели и порогов S10 в `evidence`, которые у адаптера свои). Ответы
флага на пуле — в выборке обучения кандидатов и для оценки не годятся. kr-test здесь нет:
`assert_no_test` на всех кадрах и картинках.

    PYTHONPATH=. python research/2026-09-26_fund/final/check_flag_off.py all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
REPO = FUND_DIR.parents[1]
AFTER_COMMIT = "1d5b4e9"
SETS = ("catalog_v2", "kr_dev")
VARIANTS = ("after", "off", "off-explicit", "svc-lw", "svc-lw-ranker", "res-lw", "res-lw-ranker")
DROP_KEYS = {"timings_ms", "elapsed_ms", "budget_ms"}


def strip(obj: Any) -> Any:
    """Без времён: они меняются от прогона к прогону."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def dump(variant: str, out: Path, after_root: Path | None, limit: int | None) -> None:
    if variant == "after":
        assert after_root is not None
        sys.path.insert(0, str(after_root))
        import app

        assert Path(app.__file__).resolve().parent == (after_root / "app").resolve(), app.__file__
    sys.path.insert(0, str(FUND_DIR))
    sys.path.insert(0, str(FUND_DIR / "adapter"))
    import numpy as np
    import protocol as P
    import replay as R

    import app as app_pkg

    rows = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in SETS]
    n = P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    rows = rows[:limit] if limit else rows
    qemb = np.load(P.PROTOCOL / "trainpool.npz")["qemb"]
    adapter_file = P.FUND / "final" / "cv-adapter-lw.npz"
    env: dict[str, str] = {}
    if variant == "off-explicit":
        env = {"SVS_CANDIDATE": "off"}
    elif variant == "svc-lw":
        env = {"SVS_CANDIDATE": "adapter-lw", "SVS_CV_ADAPTER": str(adapter_file)}
    elif variant == "svc-lw-ranker":
        env = {"SVS_CANDIDATE": "adapter-lw-ranker", "SVS_CV_ADAPTER": str(adapter_file)}
    rp = R.FundReplay(extra_env=env) if env else R.FundReplay()
    if variant.startswith("res-"):
        import build_candidate as BC

        rp.cv_fn = BC.make_cv_fn(rp, BC.load_candidate(BC.CAND / "lw_candidate.npz"))
        if variant == "res-lw-ranker":
            from app.resolve.learned import LogisticRanker

            rp.svc.model = LogisticRanker.load(REPO / "configs" / "resolve" / "s2so400m-vlm35-lw-pool.json")
    cap: dict[str, Any] = {}
    orig = rp.svc.scan

    def spy(data: bytes) -> Any:
        cap["result"] = res = orig(data)
        return res

    rp.svc.scan = spy  # type: ignore[method-assign]
    t0 = time.perf_counter()
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            row = rp.scan(Path(r["image"]).read_bytes(), qemb[r["row"]])
            assert row["ok"], (r["id"], row.get("error"))
            res = cap["result"]
            ev = res.evidence
            rec = {
                "id": r["id"],
                "row": row,
                "predict": strip(res.predict_body()),
                "evidence": strip({k: ev.get(k) for k in ("image", "cv", "vlm", "resolve", "abstain")}),
            }
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    meta = {
        "variant": variant,
        "app": str(Path(app_pkg.__file__).resolve().parent),
        "frames": len(rows),
        "assert_no_test": n,
        "reads": {"hits": rp.hits, "misses": rp.misses},
        "candidate": getattr(rp.settings, "candidate", None),
        "resolve_model": str(rp.settings.resolve_model),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {json.loads(line)["id"]: json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()}


def compare(out_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(FUND_DIR))
    import protocol as P

    files = {v: out_dir / f"{v}.jsonl" for v in VARIANTS}
    sha = {v: hashlib.sha1(f.read_bytes()).hexdigest() for v, f in files.items() if f.exists()}
    recs = {v: load(f) for v, f in files.items() if f.exists()}
    pool = {r["id"]: r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in SETS}
    res: dict[str, Any] = {"files_sha1": sha}
    after = recs["after"]
    ids = list(after)
    # 1. флаг выключен = after-search байт в байт
    for v in ("off", "off-explicit"):
        res[f"{v}_eq_after"] = {
            "files_identical": sha.get(v) == sha["after"],
            "frames_identical": sum(json.dumps(recs[v][q], sort_keys=True) == json.dumps(after[q], sort_keys=True)
                                    for q in ids),
            "answers_equal": sum(recs[v][q]["row"]["answer"] == after[q]["row"]["answer"] for q in ids),
            "frames": len(ids),
        }
    # 2. after-search = ответы и выдача продукта из пула
    res["after_eq_trainpool"] = {
        "answers_equal": sum(after[q]["row"]["answer"] == pool[q]["answer"] for q in ids),
        "cv_equal": sum(after[q]["row"]["cv"] == pool[q]["cv"] for q in ids),
        "rank_scores_equal": sum(after[q]["row"].get("rank_scores") == pool[q].get("rank_scores") for q in ids),
        "same_wine": {s: sum(P.correct(after[q]["row"]["answer"], pool[q]) for q in ids if pool[q]["set"] == s)
                      for s in SETS},
        "frames": len(ids),
    }

    # 3. флаг включён = путь стенда (выдача, признаки, ранкер, predict), кроме имени модели и порогов
    def core(rec: dict[str, Any]) -> str:
        row = {k: v for k, v in rec["row"].items() if k != "resolve"}
        resolve = {k: v for k, v in (rec["row"].get("resolve") or {}).items() if k != "model"}
        ev = {k: v for k, v in rec["evidence"].items() if k != "abstain"}
        ev["resolve"] = {k: v for k, v in (ev.get("resolve") or {}).items() if k != "model"}
        return json.dumps({"row": row, "resolve": resolve, "predict": rec["predict"], "ev": ev}, sort_keys=True)

    for c in ("lw", "lw-ranker"):
        a, b = recs.get(f"svc-{c}"), recs.get(f"res-{c}")
        if a is None or b is None:
            continue
        res[f"svc_eq_res_{c}"] = {
            "core_identical": sum(core(a[q]) == core(b[q]) for q in ids),
            "answers_equal": sum(a[q]["row"]["answer"] == b[q]["row"]["answer"] for q in ids),
            "cv_equal": sum(a[q]["row"]["cv"] == b[q]["row"]["cv"] for q in ids),
            "frames": len(ids),
            "in_sample_same_wine_not_an_estimate": {
                s: sum(P.correct(a[q]["row"]["answer"], pool[q]) for q in ids if pool[q]["set"] == s) for s in SETS
            },
            "answers_changed_vs_product": sum(a[q]["row"]["answer"] != after[q]["row"]["answer"] for q in ids),
        }
    return res


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dump", "compare", "all"))
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--out-dir", type=Path, default=Path(r"<корень>\svs-logs"
                                                          r"\somm-2409\fund\final\flag_check"))
    ap.add_argument("--after-root", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    after_root = args.after_root or (args.out_dir / "after_search_src")
    if args.cmd == "dump":
        dump(args.variant, args.out_dir / f"{args.variant}.jsonl", after_root, args.limit)
        return 0
    if args.cmd == "all":
        if not (after_root / "app").is_dir():
            after_root.mkdir(parents=True, exist_ok=True)
            tar = subprocess.run(["git", "-C", str(REPO), "archive", AFTER_COMMIT, "app", "configs"],
                                 capture_output=True, check=True).stdout
            import io
            import tarfile

            tarfile.open(fileobj=io.BytesIO(tar)).extractall(after_root, filter="data")
        env = {**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2"}
        procs = []
        for v in args.variants:
            cmd = [sys.executable, str(Path(__file__).resolve()), "dump", "--variant", v, "--out-dir",
                   str(args.out_dir), "--after-root", str(after_root)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            log = (args.out_dir / f"{v}.log").open("w", encoding="utf-8")
            procs.append((v, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO))))
        for v, p in procs:
            code = p.wait()
            print(f"{v}: код {code}", flush=True)
            assert code == 0, v
    res = compare(args.out_dir)
    res["after_commit"] = AFTER_COMMIT
    res["code_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    (args.out_dir / "check_flag_off.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
