"""База на kr-test (EVAL_PROTOCOL.md §4): ответы продукта @after-search на снимке данных fund.

Единственный, кроме итоговой оценки кандидата, скрипт, который открывает kr-test. Кандидатов в
нём нет: это ответы сервиса как есть (ранкер `-goal`, H5, Э1 P1, Э2, данные Э4, разбор Э6) —
реплей `replay.FundReplay` по CPU-векторам и кэшу чтений. Сверка — с ответами итоговой сборки
`goal26/final_kr.json` (`e3_measure.py measure`, 25.09): должны совпасть кадр в кадр.

Пишет `protocol/baseline_kr_test.json` (ответы по кадрам — для парного теста знаков у
кандидатов) и строку «итог» журнала.

    PYTHONPATH=. python research/2026-09-26_fund/baseline_kr_test.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P
import replay as R

PRODUCT_SHA1 = {
    "visual-s2so400m.npz": "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5",
    "gt_tokens.jsonl": "899d5db3747fd315b7c8bfedf121f6ae79799061",
    "lexicon.json": "c2c7befd13b3f999b76c3469df31cc6a0732da5c",
}
GOAL26 = Path(
    r"<tmp>\goal26\final_kr.json"
)
PAIRS_CV = (
    P.ROOT / "svoe-vino-scanner" / "runs" / "goal" / "iter20" / "cvall-pairs" / "predictions.jsonl"
)


def seen_wines() -> set[str]:
    """wine_id, у которых есть кадры в пуле обучения вне kr: v2 (353) и студийные пары (358)."""
    wines = P.wine_of()
    out: set[str] = set()
    for m in P.jsonl(P.FROZEN_FD / "sets" / "catalog_v2" / "meta.jsonl"):
        out |= {wines[s] for s in [m["slug"], *(m.get("acceptable") or [])] if s in wines}
    for r in P.jsonl(PAIRS_CV):
        if r["slug"] in wines:
            out.add(wines[r["slug"]])
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    t0 = time.perf_counter()
    qids = P.test_frames("baseline")
    meta = {
        m["query_id"]: m for m in P.jsonl(P.FROZEN_FD / "sets" / "krasnostop_v1" / "meta.jsonl")
    }
    split = P.load_split()
    group_of = {q: k for k, g in split["groups"].items() for q in g["primary"]}
    assert all(split["groups"][group_of[q]]["half"] == "test" for q in qids)

    rp = R.FundReplay()
    files = {
        p.name: P.sha1_file(p)
        for p in (rp.settings.index_path, rp.settings.attrs_path, rp.settings.lexicon_path)
    }
    assert files == PRODUCT_SHA1, files
    all_q, vecs, names = R.load_qemb("krasnostop_v1")
    vec = dict(zip(all_q, vecs, strict=True))
    rows: dict[str, dict[str, object]] = {}
    for q in qids:
        row = rp.scan(R.image_path(meta[q]).read_bytes(), vec[q])
        assert row["ok"], (q, row.get("error"))
        rows[q] = {
            "answer": row["answer"],
            "cv_top1": row["cv"][0][0],
            "correct": P.correct(row["answer"], meta[q]),
            "strict": P.strict(row["answer"], meta[q]),
            "cv_correct": P.correct(row["cv"][0][0], meta[q]),
            "in_top20": any(P.correct(c[0], meta[q]) for c in row["cv"]),
            "group": group_of[q],
        }
    # сверка с итоговой сборкой 25.09 (тот же код, те же данные)
    ref = json.loads(GOAL26.read_text(encoding="utf-8"))["rows"]
    differ = sorted(q for q in qids if ref[q]["off"] != rows[q]["answer"])

    seen = seen_wines()
    wines = P.wine_of()

    def is_seen(q: str) -> bool:
        m = meta[q]
        return any(wines.get(s) in seen for s in [m["slug"], *(m.get("acceptable") or [])])

    def block(qs: list[str]) -> dict[str, object]:
        ok = [bool(rows[q]["correct"]) for q in qs]
        st = [bool(rows[q]["strict"]) for q in qs]
        grp = [str(rows[q]["group"]) for q in qs]
        by_g: dict[str, list[bool]] = defaultdict(list)
        for o, g in zip(ok, grp, strict=True):
            by_g[g].append(o)
        return {
            "frames": len(qs),
            "groups": len(by_g),
            "same_wine": sum(ok),
            "same_wine_micro": round(100 * sum(ok) / len(qs), 2),
            "same_wine_ci95_boot_by_wine": P.boot_ci_by_group(ok, grp, seed=0, n_boot=10000),
            "same_wine_macro_by_wine": round(
                100 * sum(sum(v) / len(v) for v in by_g.values()) / len(by_g), 2
            ),
            "strict": sum(st),
            "strict_micro": round(100 * sum(st) / len(qs), 2),
            "strict_ci95_boot_by_wine": P.boot_ci_by_group(st, grp, seed=0, n_boot=10000),
            "cv_top1_same_wine": sum(bool(rows[q]["cv_correct"]) for q in qs),
            "ceiling_top20": sum(bool(rows[q]["in_top20"]) for q in qs),
        }

    seen_q = sorted(q for q in qids if is_seen(q))
    unseen_q = sorted(q for q in qids if not is_seen(q))
    out = {
        "what": "база kr-test: ответы продукта after-search @1d5b4e9 на снимке fund (EVAL_PROTOCOL.md §3)",
        "code": "svs-fund @"
        + __import__("subprocess")
        .run(
            ["git", "-C", str(R.REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        .stdout.strip(),
        "files_sha1": files,
        "split_sha1": P.sha1_file(P.SPLIT),
        "reads": {"hits": rp.hits, "misses": rp.misses},
        "same_as_goal26_final_kr": len(qids) - len(differ),
        "differ_from_goal26": differ,
        "all": block(qids),
        "seen_wines_subset": block(seen_q),
        "unseen_wines_subset": block(unseen_q),
        "seen_qids": seen_q,
        "unseen_qids": unseen_q,
        "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    dst = P.PROTOCOL / "baseline_kr_test.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    a = out["all"]
    line = (
        f"«то же вино» {a['same_wine']}/{a['frames']} = {a['same_wine_micro']} % "
        f"[{a['same_wine_ci95_boot_by_wine'][0]}–{a['same_wine_ci95_boot_by_wine'][1]}], "
        f"строго {a['strict']} = {a['strict_micro']} %; совпало с goal26 {out['same_as_goal26_final_kr']}/{len(qids)}"
    )
    from datetime import datetime

    with P.LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {datetime.now().strftime('%d.%m %H:%M:%S')} | итог | baseline | {Path(__file__).name} | {line} |\n"
        )
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("rows", "seen_qids", "unseen_qids")},
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
