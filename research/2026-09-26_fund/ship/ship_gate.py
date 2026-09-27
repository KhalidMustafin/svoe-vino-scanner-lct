"""Ворота выпуска: кандидат 2 `adapter-lw-ranker` — путь по умолчанию после слияния `fund` в `after-search`.

kr-test здесь нет: `protocol.assert_no_test` на всех кадрах и картинках, `test_frames` не
вызывается (kr-test израсходован, `fund/protocol/test_ledger.md`). Каждый вариант — свой процесс
и полный `FundReplay.scan` на снимке данных `fund/frozen_data` (разжатие кадра, виды, CV по
CPU-векторам запросов, чтение из кэша стенда, разбор, признаки, ранкер, H5 / P1 / Э2, S10):

    набор        кадры  откуда
    catalog_v2    353   строки пула `trainpool.jsonl` + векторы `trainpool.npz` (R — JPEG-копии)
    kr_dev        308   то же (основные кадры kr-dev, без same_packshot)
    ooc_v2        408   то же (все вина вне каталога: ответ не засчитывается, только смены)
    r_orig         65   R-оригиналы организатора (WebP), `qemb_r_orig.npz`, метка — catalog_v2

Варианты:
- `fund-on`    — код `fund` @9bf10c8 (`git archive` app configs), `SVS_CANDIDATE=adapter-lw-ranker`,
                 `SVS_CV_ADAPTER` — карта `fund/final/cv-adapter-lw.npz`: то, что принято по kr-test
                 и проверено живьём в GPU-окне E;
- `merged-on`  — код этого дерева **без** `SVS_CANDIDATE` (умолчание), та же карта через
                 `SVS_CV_ADAPTER`;
- `merged-off` — код этого дерева с `SVS_CANDIDATE=off`;
- `after`      — код `after-search` @070cab4 (`git archive`), без переменных.

BLAS — один поток во всех вариантах: у карты счёт зависит от порядка сложения в последних битах
(PREREG_final §3). Запись кадра — запись реплея (top-20 CV со счётами, отрыв, чтение, признаки,
счёт ранкера, ответ) и тело `/v1/eval/predict` с `evidence`, без времён.

Проверки (`compare`):
1. `merged-off` = `after` байт в байт на всех 1 134 кадрах;
2. `merged-on` против `fund-on`: каждый кадр, где запись различается, должен иметь другие поля
   чтения (Э7 правит только разбор, `app/reading`); список таких кадров — с ответами и верностью.
   Поля чтения от флага не зависят, поэтому тот же список даёт `after` против `fund-on`;
3. «то же вино» и строго по наборам: R-оригиналы `merged-on` ≥ 62 (критерий 2 PREREG_final);
   поломок «того же вина» `merged-on` против `fund-on` на v2, kr-dev и R — только на кадрах Э7;
4. офлайн = сервис: независимый счёт модели и правил (`bench.resolve_equality.offline_answer`) по
   записанным выдаче CV и чтению = ответ сервиса (Э2 — CV top-1), для `merged-on` (модель
   `-lw-pool` на выдаче карты) и `merged-off` (`-goal`).

    PYTHONPATH=. python research/2026-09-26_fund/ship/ship_gate.py all
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
REPO = FUND_DIR.parents[1]
FUND_COMMIT = "9bf10c8"
AFTER_COMMIT = "070cab4"
POOL_SETS = ("catalog_v2", "kr_dev", "ooc_v2")
SETS = (*POOL_SETS, "r_orig")
SCORED = ("catalog_v2", "kr_dev", "r_orig")
VARIANTS = ("fund-on", "merged-on", "merged-off", "after")
DROP_KEYS = {"timings_ms", "elapsed_ms", "budget_ms"}
#: = PREREG_final.md, §2 (те же, что в final_look.py и r_orig_check.py)
ADAPTER_FILE_SHA1 = "206627c7a95a759a781db21b8ee5c70d9e0c758b"
ADAPTER_CONTENT_SHA1 = "5d25e5c66ecd0034a018a7a6e43f092aba60daf8"
INDEX_SHA1 = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5"
R_ORIG_FLOOR = 62
LOG_DIR = Path(r"<корень>\svs-logs\somm-2409\fund\ship\gate")


def strip(obj: Any) -> Any:
    """Без времён: они меняются от прогона к прогону."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def code_root(variant: str, out_dir: Path) -> Path | None:
    commit = {"fund-on": FUND_COMMIT, "after": AFTER_COMMIT}.get(variant)
    return out_dir / f"src_{commit}" if commit else None


def frames(P: Any) -> list[dict[str, Any]]:
    """Кадры ворот: строки пула трёх наборов и R-оригиналы — с меткой, картинкой и вектором."""
    import numpy as np

    pool = [r for r in P.jsonl(P.PROTOCOL / "trainpool.jsonl") if r["set"] in POOL_SETS]
    qemb = np.load(P.PROTOCOL / "trainpool.npz")["qemb"]
    out = [
        {"key": f"{r['set']}:{r['id']}", "set": r["set"], "image": r["image"],
         "label": {"slug": r["slug"], "acceptable": r.get("acceptable") or []},
         "qvec": qemb[r["row"]]}
        for r in pool
    ]  # fmt: skip
    run = P.ITERS / "runs" / "r_orig"
    meta_r = {m["query_id"]: m for m in P.jsonl(run / "meta.jsonl")}
    v2 = {m["query_id"]: m for m in P.jsonl(P.FROZEN_FD / "sets" / "catalog_v2" / "meta.jsonl")}
    z = np.load(P.QEMB / "qemb_r_orig.npz")
    assert [str(n) for n in z["names"]] == ["bottle", "label", "band", "full"]
    for q, vec in zip((str(q) for q in z["qids"]), z["vectors"], strict=True):
        label = v2[q]
        assert meta_r[q]["slug"] == label["slug"], q
        out.append(
            {"key": f"r_orig:{q}", "set": "r_orig", "image": meta_r[q]["image"],
             "label": {"slug": label["slug"], "acceptable": label.get("acceptable") or []},
             "qvec": vec}
        )  # fmt: skip
    counts = {s: sum(f["set"] == s for f in out) for s in SETS}
    assert counts == {"catalog_v2": 353, "kr_dev": 308, "ooc_v2": 408, "r_orig": 65}, counts
    return out


def dump(variant: str, out_dir: Path, limit: int | None) -> None:
    root = code_root(variant, out_dir)
    if root is not None:  # пакет `app` — из архива коммита, до любого импорта реплея
        sys.path.insert(0, str(root))
        import app

        assert Path(app.__file__).resolve().parent == (root / "app").resolve(), app.__file__
    sys.path.insert(0, str(FUND_DIR))
    import protocol as P
    import replay as R

    import app as app_pkg

    rows = frames(P)
    n_checked = P.assert_no_test(
        [f["key"].split(":", 1)[1] for f in rows], [f["image"] for f in rows]
    )
    rows = rows[:limit] if limit else rows
    adapter_file = P.FUND / "final" / "cv-adapter-lw.npz"
    assert P.sha1_file(adapter_file) == ADAPTER_FILE_SHA1
    env = {
        "fund-on": {"SVS_CANDIDATE": "adapter-lw-ranker", "SVS_CV_ADAPTER": str(adapter_file)},
        "merged-on": {"SVS_CV_ADAPTER": str(adapter_file)},  # SVS_CANDIDATE не задан: умолчание
        "merged-off": {"SVS_CANDIDATE": "off"},
        "after": {},
    }[variant]
    rp = R.FundReplay(extra_env=env)
    assert P.sha1_file(rp.settings.index_path) == INDEX_SHA1
    candidate = getattr(rp.settings, "candidate", None)
    adapter = getattr(rp.svc, "cv_adapter", None)
    if variant.endswith("-on"):
        assert candidate == "adapter-lw-ranker", candidate
        assert adapter is not None and adapter.sha1 == ADAPTER_CONTENT_SHA1
        assert Path(rp.settings.resolve_model).name == "s2so400m-vlm35-lw-pool.json"
    else:
        assert adapter is None and candidate in (None, "off"), candidate
        assert Path(rp.settings.resolve_model).name == "s2so400m-vlm35-goal.json"
    health = rp.svc.health()
    cap: dict[str, Any] = {}
    orig = rp.svc.scan

    def spy(data: bytes) -> Any:
        cap["result"] = res = orig(data)
        return res

    rp.svc.scan = spy  # type: ignore[method-assign]
    t0 = time.perf_counter()
    with (out_dir / f"{variant}.jsonl").open("w", encoding="utf-8") as fh:
        for f in rows:
            row = rp.scan(Path(f["image"]).read_bytes(), f["qvec"])
            assert row["ok"], (f["key"], row.get("error"))
            res = cap["result"]
            ev = res.evidence
            rec = {
                "key": f["key"],
                "row": row,
                "predict": strip(res.predict_body()),
                "evidence": strip(
                    {k: ev.get(k) for k in ("image", "cv", "vlm", "resolve", "abstain")}
                ),
            }
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    meta = {
        "variant": variant,
        "app": str(Path(app_pkg.__file__).resolve().parent),
        "frames": len(rows),
        "assert_no_test": n_checked,
        "reads": {"hits": rp.hits, "misses": rp.misses},
        "candidate": candidate,
        "resolve_model": Path(rp.settings.resolve_model).name,
        "cv_adapter": adapter.public() if adapter is not None else None,
        "provenance": health.get("provenance"),
        "warnings": health.get("warnings"),
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / f"{variant}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "provenance"}, ensure_ascii=False))


def load(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        out[rec["key"]] = rec
    return out


def canon(rec: dict[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, sort_keys=True)


def fields_of(rec: dict[str, Any]) -> Any:
    return (rec["row"].get("read") or {}).get("fields")


def offline_equality(recs: dict[str, dict[str, Any]], model_path: Path, P: Any) -> dict[str, Any]:
    """Ответ сервиса = независимый офлайн-счёт по записанным выдаче CV и чтению (Э2 — CV top-1)."""
    sys.path.insert(0, str(REPO))
    from app.api.service import load_catalog
    from bench import resolve_equality as RE
    from replay import read_of, visual_of

    _, attrs = load_catalog(P.FROZEN / "gt" / "gt_tokens.jsonl")
    weights = RE.Weights.load(model_path)
    reader = RE.readers_of(list(weights.names))[0]
    differ, fallback = [], 0
    for key, rec in recs.items():
        row = rec["row"]
        if "fallback" in (row.get("resolve") or {}):
            fallback += 1
            want = row["cv"][0][0]
        else:
            visual = visual_of(row["cv"], row["margin"])
            want = RE.offline_answer(weights, visual, read_of(row["read"]), attrs, reader)
        if want != row["answer"]:
            differ.append({"key": key, "service": row["answer"], "offline": want})
    return {
        "model": model_path.name,
        "frames": len(recs),
        "equal": len(recs) - len(differ),
        "reader_fallback_cv_top1": fallback,
        "differ": differ,
    }


def compare(out_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(FUND_DIR))
    import protocol as P

    files = {v: out_dir / f"{v}.jsonl" for v in VARIANTS}
    sha = {v: hashlib.sha1(f.read_bytes()).hexdigest() for v, f in files.items()}
    recs = {v: load(f) for v, f in files.items()}
    labels = {f["key"]: f for f in frames(P)}
    keys = list(recs["after"])
    assert all(list(recs[v]) == keys for v in VARIANTS)
    set_of = {k: labels[k]["set"] for k in keys}
    res: dict[str, Any] = {"files_sha1": sha, "frames": {s: sum(set_of[k] == s for k in keys) for s in SETS}}

    # 1. путь off = after-search @070cab4 байт в байт
    res["merged_off_eq_after"] = {
        "files_identical": sha["merged-off"] == sha["after"],
        "frames_identical": {
            s: sum(canon(recs["merged-off"][k]) == canon(recs["after"][k]) for k in keys if set_of[k] == s)
            for s in SETS
        },
    }

    # 2. путь по умолчанию после слияния против fund с флагом: разница — только кадры Э7
    def answer(v: str, k: str) -> str | None:
        return recs[v][k]["row"]["answer"]

    def ok(v: str, k: str) -> bool:
        return P.correct(answer(v, k), labels[k]["label"])

    def ok_strict(v: str, k: str) -> bool:
        return P.strict(answer(v, k), labels[k]["label"])

    diff = [k for k in keys if canon(recs["merged-on"][k]) != canon(recs["fund-on"][k])]
    e7_fields = [k for k in keys if fields_of(recs["merged-on"][k]) != fields_of(recs["fund-on"][k])]
    e7_fields_off = [k for k in keys if fields_of(recs["after"][k]) != fields_of(recs["fund-on"][k])]
    unexplained = [k for k in diff if k not in set(e7_fields)]
    listed = []
    for k in diff:
        scored = set_of[k] in SCORED
        listed.append({
            "key": k,
            "set": set_of[k],
            "fields_differ": k in set(e7_fields),
            "raw_text_same": (recs["merged-on"][k]["row"].get("read") or {}).get("raw_text")
            == (recs["fund-on"][k]["row"].get("read") or {}).get("raw_text"),
            "fund_on": answer("fund-on", k),
            "merged_on": answer("merged-on", k),
            "answer_changed": answer("fund-on", k) != answer("merged-on", k),
            "label": labels[k]["label"]["slug"],
            "fund_on_same_wine": ok("fund-on", k) if scored else None,
            "merged_on_same_wine": ok("merged-on", k) if scored else None,
            "fund_on_strict": ok_strict("fund-on", k) if scored else None,
            "merged_on_strict": ok_strict("merged-on", k) if scored else None,
        })  # fmt: skip
    res["merged_on_vs_fund_on"] = {
        "frames_identical": {s: sum(set_of[k] == s for k in keys) - sum(set_of[k] == s for k in diff) for s in SETS},
        "records_differ": len(diff),
        "fields_differ": len(e7_fields),
        "fields_differ_after_vs_fund_on": len(e7_fields_off),
        "same_e7_frames_on_and_off": sorted(e7_fields) == sorted(e7_fields_off),
        "unexplained": unexplained,
        "answers_changed": sum(x["answer_changed"] for x in listed),
        "same_wine_breaks": [x["key"] for x in listed if x["fund_on_same_wine"] and not x["merged_on_same_wine"]],
        "same_wine_fixes": [x["key"] for x in listed if x["merged_on_same_wine"] and not x["fund_on_same_wine"]],
        "frames": listed,
    }  # fmt: skip

    # 3. счёт по наборам
    score: dict[str, Any] = {}
    for v in VARIANTS:
        score[v] = {
            s: {
                "same_wine": sum(ok(v, k) for k in keys if set_of[k] == s),
                "strict": sum(ok_strict(v, k) for k in keys if set_of[k] == s),
            }
            for s in SCORED
        }
        v2r = [k for k in keys if set_of[k] == "catalog_v2" and k.split(":", 1)[1].startswith("R")]
        score[v]["catalog_v2_R_jpeg"] = sum(ok(v, k) for k in v2r)
    res["score"] = score
    res["answers_changed_on_vs_off"] = {
        s: sum(answer("merged-on", k) != answer("merged-off", k) for k in keys if set_of[k] == s) for s in SETS
    }
    res["r_orig_merged_on_ge_floor"] = score["merged-on"]["r_orig"]["same_wine"] >= R_ORIG_FLOOR

    # 4. офлайн = сервис
    res["offline_eq_service"] = {
        "merged-on": offline_equality(recs["merged-on"], REPO / "configs" / "resolve" / "s2so400m-vlm35-lw-pool.json", P),
        "merged-off": offline_equality(recs["merged-off"], REPO / "configs" / "resolve" / "s2so400m-vlm35-goal.json", P),
    }  # fmt: skip
    res["metas"] = {
        v: json.loads((out_dir / f"{v}.meta.json").read_text(encoding="utf-8")) for v in VARIANTS
    }
    return res


def archive(commit: str, dst: Path) -> None:
    if (dst / "app").is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(
        ["git", "-C", str(REPO), "archive", commit, "app", "configs"],
        capture_output=True,
        check=True,
    ).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(dst, filter="data")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dump", "compare", "all"))
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--out-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "dump":
        dump(args.variant, args.out_dir, args.limit)
        return 0
    if args.cmd == "all":
        for commit in (FUND_COMMIT, AFTER_COMMIT):
            archive(commit, args.out_dir / f"src_{commit}")
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        procs = []
        for v in args.variants:
            cmd = [sys.executable, str(Path(__file__).resolve()), "dump", "--variant", v,
                   "--out-dir", str(args.out_dir)]  # fmt: skip
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            log = (args.out_dir / f"{v}.log").open("w", encoding="utf-8")
            procs.append(
                (v, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO))
            )
        for v, p in procs:
            code = p.wait()
            print(f"{v}: код {code}", flush=True)
            assert code == 0, v
    res = compare(args.out_dir)
    res["commits"] = {
        "fund": FUND_COMMIT,
        "after": AFTER_COMMIT,
        "merged": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    out = args.out_dir / "ship_gate.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    brief = {k: v for k, v in res.items() if k not in ("metas", "files_sha1")}
    brief["merged_on_vs_fund_on"] = {
        k: v for k, v in res["merged_on_vs_fund_on"].items() if k != "frames"
    }
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("→", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
