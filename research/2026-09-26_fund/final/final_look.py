"""Итоговая оценка кандидата PREREG_final на kr-test (EVAL_PROTOCOL.md §5) — один запуск, один просмотр.

Не запускать без решения команды. Кандидаты и их файлы — `../PREREG_final.md` (sha1 ниже
сверяются до открытия). Скрипт отказывает, если журнал уже видел `final:<кандидат>` или два разных
кандидата, если в `app/`, `configs/` или `research/2026-09-26_fund/` есть незакоммиченные правки
(оценивается только закоммиченный код), и если сервис с флагом загрузил не те файлы.

Ответы кандидата считает **сервис с флагом** (`SVS_CANDIDATE=<кандидат>`, `SVS_CV_ADAPTER` — карта из
PREREG_final) тем же CPU-реплеем, что база (`FundReplay.scan`: векторы `qemb_krasnostop_v1.npz`, кэш
чтений). В том же проходе тот же реплей с выключенным флагом повторяет базу — проверка, что стенд не
уехал (ответы сравниваются с `baseline_kr_test.json` механически, кадры глазами не разбираются).
kr-test выдаёт только `protocol.test_frames("final:<кандидат>")` — строка «открыт» в журнале раньше
первого кадра; строка «итог» — в конце.

Порядок — PREREG_final §6.2: `adapter-lw-ranker` скрипт откроет, только если в журнале уже есть итог
`final:adapter-lw` с «критерий 1 — да».

Приёмка — §5 без изменений: (1) «то же вино» > 277 и точный двусторонний тест знаков по сменившимся
кадрам p < 0,05; (2) R-оригиналы (65) не ниже базы — прогон параллельного трека, здесь «не проверен»;
(3) v2 ≥ 314 вне фолда — из доказательств PREREG_final (не пересчитывается).

BLAS — один поток (до импорта numpy): проекция картой (матрица 1152 × 1152) в float32 зависит от
порядка сложения, а он — от числа потоков; на точных ничьих (две карточки одной фотосессии со счётом,
равным до 1e-7) это решает ответ. Один поток даёт один и тот же ответ при каждом запуске.

    PYTHONPATH=. python research/2026-09-26_fund/final/final_look.py --candidate adapter-lw --i-am-the-final-look
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FUND_DIR = HERE.parent
REPO = FUND_DIR.parents[1]
sys.path.insert(0, str(FUND_DIR))

import protocol as P

ADAPTER_FILE = P.FUND / "final" / "cv-adapter-lw.npz"
#: = PREREG_final.md, §2
ADAPTER_FILE_SHA1 = "206627c7a95a759a781db21b8ee5c70d9e0c758b"
ADAPTER_CONTENT_SHA1 = "5d25e5c66ecd0034a018a7a6e43f092aba60daf8"
INDEX_SHA1 = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5"
#: Модели resolve — sha1 содержимого с концами строк LF (= блоб git и файл на Linux): на Windows
#: git с autocrlf отдаёт рабочую копию JSON с CRLF.
RESOLVE_SHA1 = {
    "adapter-lw": ("s2so400m-vlm35-goal.json", "4c087cc008f75373792a81d9f69904b4344af91c"),
    "adapter-lw-ranker": (
        "s2so400m-vlm35-lw-pool.json",
        "b82580e22d89aa3d5fd0e9a9678ed29e1fecb809",
    ),
}
#: Критерий 3 (v2 ≥ 314 вне фолда) — доказательства PREREG_final, §4 (не пересчитываются здесь).
V2_OOF = {"adapter-lw": 316, "adapter-lw-ranker": 323}


def sha1_lf(path: Path) -> str:
    return hashlib.sha1(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def block(rows: dict[str, dict[str, Any]], qs: list[str]) -> dict[str, Any]:
    ok = [bool(rows[q]["correct"]) for q in qs]
    st = [bool(rows[q]["strict"]) for q in qs]
    grp = [str(rows[q]["group"]) for q in qs]
    by_g: dict[str, list[bool]] = defaultdict(list)
    for o, g in zip(ok, grp, strict=True):
        by_g[g].append(o)
    return {
        "frames": len(qs),
        "same_wine": sum(ok),
        "same_wine_micro": round(100 * sum(ok) / len(qs), 2),
        "same_wine_ci95_boot_by_wine": P.boot_ci_by_group(ok, grp, seed=0, n_boot=10000),
        "same_wine_macro_by_wine": round(
            100 * sum(sum(v) / len(v) for v in by_g.values()) / len(by_g), 2
        ),
        "strict": sum(st),
        "strict_micro": round(100 * sum(st) / len(qs), 2),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=sorted(RESOLVE_SHA1))
    look = ap.add_mutually_exclusive_group(required=True)
    look.add_argument("--i-am-the-final-look", action="store_true")
    look.add_argument(
        "--check-only",
        action="store_true",
        help="только проверки до открытия (файлы, код, журнал, сервис с флагом); kr-test не открывается",
    )
    args = ap.parse_args()
    name = args.candidate

    # ---- до открытия: файлы, код, журнал
    assert P.sha1_file(ADAPTER_FILE) == ADAPTER_FILE_SHA1, "карта не та, что в PREREG_final.md"
    model_name, model_sha1 = RESOLVE_SHA1[name]
    model_path = REPO / "configs" / "resolve" / model_name
    assert sha1_lf(model_path) == model_sha1, f"{model_name} не тот, что в PREREG_final.md"
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "status",
            "--porcelain",
            "--",
            "app",
            "configs",
            "research/2026-09-26_fund",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"незакоммиченные правки в коде кандидата — оценивается только коммит:\n{dirty}"
        )
    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    ledger = P.LEDGER.read_text(encoding="utf-8")
    opened = set(re.findall(r"\| открыт \| (final:[^ |]+)", ledger))
    if f"final:{name}" in opened:
        raise SystemExit(f"kr-test уже открывался для {name}: второго просмотра нет (§5)")
    if len(opened) >= 2:
        raise SystemExit(f"у трека уже два кандидата на kr-test: {sorted(opened)} (§3, §5)")
    if name == "adapter-lw-ranker":
        # PREREG_final §6.2: фиксированная последовательность — кандидат 2 только после того, как
        # кандидат 1 прошёл критерий 1 на kr-test
        gate = [
            line
            for line in ledger.splitlines()
            if "| итог | final:adapter-lw |" in line and "критерий 1 — да" in line
        ]
        if not gate and not args.check_only:
            raise SystemExit(
                "кандидат 2 открывается только после кандидата 1 с критерием 1 «да» (PREREG_final §6.2)"
            )

    import replay as R

    env = {"SVS_CANDIDATE": name, "SVS_CV_ADAPTER": str(ADAPTER_FILE)}
    rp = R.FundReplay(extra_env=env)
    adapter = rp.svc.cv_adapter
    assert adapter is not None and adapter.sha1 == ADAPTER_CONTENT_SHA1
    assert Path(rp.settings.resolve_model).resolve() == model_path.resolve(), (
        rp.settings.resolve_model
    )
    assert P.sha1_file(rp.settings.index_path) == INDEX_SHA1
    base_rp = R.FundReplay()  # флаг выключен — повтор базы в том же проходе
    assert base_rp.svc.cv_adapter is None
    if args.check_only:
        print(
            json.dumps(
                {
                    "candidate": name,
                    "ready": True,
                    "code_commit": commit,
                    "adapter_content_sha1": adapter.sha1,
                    "resolve_model": str(rp.settings.resolve_model),
                    "opened_before": sorted(opened),
                    "gate_for_candidate_2": (bool(gate) if name == "adapter-lw-ranker" else None),
                    "blas_threads": os.environ.get("OMP_NUM_THREADS"),
                    "kr_test": "не открывался (--check-only)",
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        return 0
    base = json.loads((P.PROTOCOL / "baseline_kr_test.json").read_text(encoding="utf-8"))
    meta = {
        m["query_id"]: m for m in P.jsonl(P.FROZEN_FD / "sets" / "krasnostop_v1" / "meta.jsonl")
    }
    all_q, vecs, _ = R.load_qemb("krasnostop_v1")
    vec = dict(zip(all_q, vecs, strict=True))

    t0 = time.perf_counter()
    qids = P.test_frames(f"final:{name}")  # строка «открыт» в журнале — здесь
    rows: dict[str, dict[str, Any]] = {}
    base_same = 0
    for q in qids:
        data = R.image_path(meta[q]).read_bytes()
        row = rp.scan(data, vec[q])
        assert row["ok"], (q, row.get("error"))
        b = base_rp.scan(data, vec[q])
        base_same += b["answer"] == base["rows"][q]["answer"]
        rows[q] = {
            "answer": row["answer"],
            "correct": P.correct(row["answer"], meta[q]),
            "strict": P.strict(row["answer"], meta[q]),
            "cv_top1": row["cv"][0][0],
            "group": base["rows"][q]["group"],
        }
    fixes = sum(rows[q]["correct"] and not base["rows"][q]["correct"] for q in qids)
    breaks = sum(base["rows"][q]["correct"] and not rows[q]["correct"] for q in qids)
    p = P.sign_test(fixes, breaks)
    a = block(rows, qids)
    crit1 = a["same_wine"] > base["all"]["same_wine"] and p < 0.05
    out = {
        "candidate": name,
        "code_commit": commit,
        "files_sha1": {
            "cv-adapter-lw.npz": ADAPTER_FILE_SHA1,
            model_name: model_sha1,
            "index": INDEX_SHA1,
        },
        "all": a,
        "baseline_same_wine": base["all"]["same_wine"],
        "fixes": fixes,
        "breaks": breaks,
        "sign_test_p": round(p, 4),
        "criterion_1_kr_test": crit1,
        "criterion_2_r_originals": "не проверен (прогон параллельного трека с тем же флагом)",
        "criterion_3_v2_oof": {"v2_oof": V2_OOF[name], "passes": V2_OOF[name] >= 314},
        "seen": block(rows, base["seen_qids"]),
        "unseen": block(rows, base["unseen_qids"]),
        "cv_top1_same_wine": sum(P.correct(rows[q]["cv_top1"], meta[q]) for q in qids),
        "baseline_replayed_same_answers": f"{base_same}/{len(qids)}",
        "reads": {"hits": rp.hits, "misses": rp.misses},
        "blas_threads": {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    dst = P.FUND / "final" / f"final_look_{name}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    line = (
        f"«то же вино» {a['same_wine']}/{a['frames']} = {a['same_wine_micro']} % "
        f"[{a['same_wine_ci95_boot_by_wine'][0]}–{a['same_wine_ci95_boot_by_wine'][1]}], починки {fixes} / "
        f"поломки {breaks}, p = {out['sign_test_p']}; критерий 1 — {'да' if crit1 else 'нет'}; "
        f"база повторена {base_same}/{len(qids)}; код {commit[:7]}"
    )
    with P.LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {datetime.now().strftime('%d.%m %H:%M:%S')} | итог | final:{name} | final_look.py | {line} |\n"
        )
    print(line, "→", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
