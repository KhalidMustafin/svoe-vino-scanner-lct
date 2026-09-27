"""Итоговая оценка кандидата adapter-lw на kr-test (EVAL_PROTOCOL.md §5) — один запуск, один просмотр.

Не запускать без решения команды: у трека не больше двух кандидатов на kr-test. Скрипт
сверяет sha1 матрицы с `PREREG.md`, отказывается, если журнал уже видел `final:adapter-lw`,
открывает kr-test только через `protocol.test_frames` (строка «открыт» — до первого кадра) и
считает ответы кандидата тем же реплеем, что база (`baseline_kr_test.py`), с одной подменой —
поиск по LW-проекции (`build_candidate.make_cv_fn`). Сравнение — с ответами базы по кадрам
(`protocol/baseline_kr_test.json`, только для теста знаков), итог — строкой в журнал.

    PYTHONPATH=. python research/2026-09-26_fund/adapter/final_kr_test.py --i-am-the-final-look
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import build_candidate as BC  # noqa: E402
import protocol as P  # noqa: E402

NAME = "adapter-lw"
CANDIDATE_SHA1 = "ae96acfb7c02b9074b2adcf6c697993157f67abf"  # = PREREG.md


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--i-am-the-final-look", action="store_true", required=True)
    ap.parse_args()
    # 26.09: заменён единым `final/final_look.py` (PREREG_final.md) — ответы кандидата считает сервис с
    # флагом SVS_CANDIDATE, а не подмена поиска стенда; имя просмотра то же (`final:adapter-lw`).
    raise SystemExit("заменён research/2026-09-26_fund/final/final_look.py --candidate adapter-lw (PREREG_final.md)")
    import replay as R

    path = BC.CAND / "lw_candidate.npz"
    assert P.sha1_file(path) == CANDIDATE_SHA1, "матрица кандидата не та, что в PREREG.md"
    ledger = P.LEDGER.read_text(encoding="utf-8")
    if f"final:{NAME}" in ledger:
        raise SystemExit(f"kr-test уже открывался для {NAME}: второго просмотра нет (§5)")
    import re

    seen = set(re.findall(r"\| открыт \| (final:[^ |]+)", ledger))
    if len(seen) >= 2:
        raise SystemExit(f"у трека уже два кандидата на kr-test: {sorted(seen)} (§3, §5)")
    rp = R.FundReplay()
    lw = BC.load_candidate(path)
    rp.cv_fn = BC.make_cv_fn(rp, lw)
    base = json.loads((P.PROTOCOL / "baseline_kr_test.json").read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    qids = P.test_frames(f"final:{NAME}")  # строка «открыт» в журнале — здесь
    meta = {m["query_id"]: m for m in P.jsonl(P.FROZEN_FD / "sets" / "krasnostop_v1" / "meta.jsonl")}
    all_q, vecs, _ = R.load_qemb("krasnostop_v1")
    vec = dict(zip(all_q, vecs, strict=True))
    rows = {}
    for q in qids:
        row = rp.scan(R.image_path(meta[q]).read_bytes(), vec[q])
        assert row["ok"], (q, row.get("error"))
        rows[q] = {"answer": row["answer"], "correct": P.correct(row["answer"], meta[q]),
                   "strict": P.strict(row["answer"], meta[q]), "cv_top1": row["cv"][0][0],
                   "group": base["rows"][q]["group"]}
    ok = [rows[q]["correct"] for q in qids]
    fixes = sum(rows[q]["correct"] and not base["rows"][q]["correct"] for q in qids)
    breaks = sum(base["rows"][q]["correct"] and not rows[q]["correct"] for q in qids)
    p = P.sign_test(fixes, breaks)
    n_ok = sum(ok)
    accepted_kr = n_ok > base["all"]["same_wine"] and p < 0.05
    out = {
        "candidate": NAME,
        "candidate_sha1": CANDIDATE_SHA1,
        "same_wine": n_ok,
        "frames": len(qids),
        "same_wine_micro": round(100 * n_ok / len(qids), 2),
        "ci95_boot_by_wine": P.boot_ci_by_group(ok, [rows[q]["group"] for q in qids]),
        "strict": sum(rows[q]["strict"] for q in qids),
        "baseline_same_wine": base["all"]["same_wine"],
        "fixes": fixes,
        "breaks": breaks,
        "sign_test_p": round(p, 4),
        "criterion_1_kr_test": accepted_kr,
        "seen": sum(rows[q]["correct"] for q in base["seen_qids"]),
        "unseen": sum(rows[q]["correct"] for q in base["unseen_qids"]),
        "reads": {"hits": rp.hits, "misses": rp.misses},
        "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    dst = BC.CAND / "final_kr_test.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    line = (f"«то же вино» {n_ok}/{len(qids)} = {out['same_wine_micro']} % "
            f"[{out['ci95_boot_by_wine'][0]}–{out['ci95_boot_by_wine'][1]}], починки {fixes} / поломки {breaks}, "
            f"p = {out['sign_test_p']}; критерий 1 — {'да' if accepted_kr else 'нет'}")
    stamp = time.strftime("%d.%m %H:%M:%S")
    with P.LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(f"| {stamp} | итог | final:{NAME} | final_kr_test.py | {line} |\n")
    print(line, "→", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
