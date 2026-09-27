"""Критерий 2 PREREG_final (§6.3): R-оригиналы (65 WebP организатора) — кандидаты против базы.

Не kr-test: 65 кадров R набора v2 на исходных файлах организатора (`runs/field25/iters/runs/r_orig`,
PREREG `acc_plan/PREREG_R_originals.md`, база окна C — живой сервис @1d5b4e9: «то же вино» 62/65,
строго 58/65). Тот же CPU-реплей, что итоговый просмотр (`FundReplay.scan`): векторы запросов —
`acc_plan/retrieval/qemb_r_orig.npz` (CPU, путь `qemb.py`), чтения — кэш стенда (ключи дописало окно C).

- База — реплей без флага; сверяется с ответами дампа окна C (`r_orig/iter20/predictions.jsonl`)
  механически: это проверка, что CPU-векторы и кэш чтений повторяют живой прогон.
- Кандидаты — сервис с флагом `SVS_CANDIDATE=<имя>`, `SVS_CV_ADAPTER` — карта PREREG_final §2 (sha1
  сверяются, как в `final_look.py`).
- Метка — `catalog_v2/meta.jsonl` снимка по `query_id` («то же вино» = slug ∪ acceptable, строго = slug).

Критерий 2 записан до этого прогона: «то же вино» кандидата на 65 R-оригиналах не ниже базы. Ничего
по этому прогону не подбирается. BLAS — один поток (как `final_look.py`).

    PYTHONPATH=. python research/2026-09-26_fund/final/r_orig_check.py
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
REPO = FUND_DIR.parents[1]
sys.path.insert(0, str(FUND_DIR))

import protocol as P

RUN = P.ITERS / "runs" / "r_orig"
QEMB = P.QEMB / "qemb_r_orig.npz"
ADAPTER_FILE = P.FUND / "final" / "cv-adapter-lw.npz"
#: = PREREG_final.md, §2 (те же, что в final_look.py)
ADAPTER_FILE_SHA1 = "206627c7a95a759a781db21b8ee5c70d9e0c758b"
ADAPTER_CONTENT_SHA1 = "5d25e5c66ecd0034a018a7a6e43f092aba60daf8"
INDEX_SHA1 = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5"
RESOLVE_SHA1 = {
    "adapter-lw": ("s2so400m-vlm35-goal.json", "4c087cc008f75373792a81d9f69904b4344af91c"),
    "adapter-lw-ranker": (
        "s2so400m-vlm35-lw-pool.json",
        "b82580e22d89aa3d5fd0e9a9678ed29e1fecb809",
    ),
}
NAMES = ["bottle", "label", "band", "full"]


def sha1_lf(path: Path) -> str:
    return hashlib.sha1(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    import replay as R

    meta_r = {m["query_id"]: m for m in P.jsonl(RUN / "meta.jsonl")}
    v2 = {m["query_id"]: m for m in P.jsonl(P.FROZEN_FD / "sets" / "catalog_v2" / "meta.jsonl")}
    dump = {r["query_id"]: r for r in P.jsonl(RUN / "iter20" / "predictions.jsonl")}
    z = np.load(QEMB)
    qids = [str(q) for q in z["qids"]]
    vecs = z["vectors"]
    assert [str(n) for n in z["names"]] == NAMES
    _, _, kr_names = R.load_qemb("krasnostop_v1")
    assert kr_names == NAMES, kr_names  # тот же порядок окон, что у векторов kr-test и пула
    assert len(qids) == 65 and set(qids) == set(meta_r) == set(dump)
    checked = P.assert_no_test(qids, [meta_r[q]["image"] for q in qids])
    for q in qids:  # метка R-оригинала = метка v2
        assert meta_r[q]["slug"] == v2[q]["slug"], q
        assert list(meta_r[q].get("acceptable") or []) == list(v2[q].get("acceptable") or []), q

    assert P.sha1_file(ADAPTER_FILE) == ADAPTER_FILE_SHA1
    base_rp = R.FundReplay()
    assert base_rp.svc.cv_adapter is None
    assert P.sha1_file(base_rp.settings.index_path) == INDEX_SHA1
    reps: dict[str, Any] = {}
    for name, (model_name, model_sha1) in RESOLVE_SHA1.items():
        model_path = REPO / "configs" / "resolve" / model_name
        assert sha1_lf(model_path) == model_sha1, model_name
        rp = R.FundReplay(extra_env={"SVS_CANDIDATE": name, "SVS_CV_ADAPTER": str(ADAPTER_FILE)})
        assert rp.svc.cv_adapter is not None and rp.svc.cv_adapter.sha1 == ADAPTER_CONTENT_SHA1
        assert Path(rp.settings.resolve_model).resolve() == model_path.resolve()
        reps[name] = rp

    t0 = time.perf_counter()
    rows: dict[str, dict[str, Any]] = {}
    for q, vec in zip(qids, vecs, strict=True):
        data = Path(meta_r[q]["image"]).read_bytes()
        b = base_rp.scan(data, vec)
        assert b["ok"], (q, b.get("error"))
        row = {
            "base": b["answer"],
            "dump": dump[q]["slug"],
            "base_correct": P.correct(b["answer"], v2[q]),
            "base_strict": P.strict(b["answer"], v2[q]),
        }
        for name, rp in reps.items():
            c = rp.scan(data, vec)
            assert c["ok"], (name, q, c.get("error"))
            row[name] = c["answer"]
            row[f"{name}_correct"] = P.correct(c["answer"], v2[q])
            row[f"{name}_strict"] = P.strict(c["answer"], v2[q])
        rows[q] = row

    base_same_wine = sum(r["base_correct"] for r in rows.values())
    out: dict[str, Any] = {
        "frames": len(qids),
        "assert_no_test_checked": checked,
        "base_replay_equals_live_dump": f"{sum(r['base'] == r['dump'] for r in rows.values())}/{len(qids)}",
        "dump_same_wine": sum(P.correct(r["dump"], v2[q]) for q, r in rows.items()),
        "base": {
            "same_wine": base_same_wine,
            "strict": sum(r["base_strict"] for r in rows.values()),
        },
    }
    for name in reps:
        sw = sum(r[f"{name}_correct"] for r in rows.values())
        fixes = sum(r[f"{name}_correct"] and not r["base_correct"] for r in rows.values())
        breaks = sum(r["base_correct"] and not r[f"{name}_correct"] for r in rows.values())
        out[name] = {
            "same_wine": sw,
            "strict": sum(r[f"{name}_strict"] for r in rows.values()),
            "fixes": fixes,
            "breaks": breaks,
            "answers_changed": sum(r[name] != r["base"] for r in rows.values()),
            "sign_test_p": round(P.sign_test(fixes, breaks), 4),
            "criterion_2_not_below_base": sw >= base_same_wine,
        }
    out["reads"] = {
        "base": [base_rp.hits, base_rp.misses],
        **{n: [rp.hits, rp.misses] for n, rp in reps.items()},
    }
    out["blas_threads"] = os.environ.get("OMP_NUM_THREADS")
    out["rows"] = rows
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    dst = P.FUND / "final" / "r_orig_check.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=1))
    print("→", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
