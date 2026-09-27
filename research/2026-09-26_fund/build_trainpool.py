"""Таблицы пула обучения и разработки (EVAL_PROTOCOL.md §3) — только CPU, без kr-test.

Каждый кадр пула проходит `replay.FundReplay.scan` (сервис @after-search на снимке данных):
выдача CV top-20 по CPU-векторам запроса, чтение (кэш стенда или записанное чтение студии),
разбор, 34 признака ранкера `-goal` для каждого кандидата, счёт ранкера и ответ продукта.
Затем `resolve_row` повторяет слой выбора по записи без картинки — ответ обязан совпасть.

Наборы:
- `catalog_v2` (353) — полевые кадры каталога, векторы `qemb_catalog_v2.npz`;
- `kr_dev` (308 основных) и `kr_dev_sp` (156 same_packshot вин kr-dev) — `qemb_krasnostop_v1.npz`,
  строки kr-test отбрасываются при загрузке, страж проверяет все id и картинки;
- `ooc_v2` (409) — вина вне каталога, отрицательные примеры; векторы `qemb_ooc_v2.npz`;
- `pairs`, `pairs_phone` (358 + 358) — студийные пары и их «телефонные» копии; векторы
  `fund/protocol/qemb_pairs.npz` (`qemb_pairs.py`), чтения — записанные прогоны qwen3.5:4b
  (`runs/ocr-pairs-vlm35`, `runs/ocr-pairsphone-vlm35`, кэш `runs/cache-pairs-ocr`).

    PYTHONPATH=. python research/2026-09-26_fund/build_trainpool.py --sets catalog_v2 kr_dev ooc_v2
    PYTHONPATH=. python research/2026-09-26_fund/build_trainpool.py --sets pairs
    PYTHONPATH=. python research/2026-09-26_fund/build_trainpool.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as P
import replay as R
from app.reading.contracts import Reading

PARTS = P.PROTOCOL / "trainpool_parts"
RUNS = P.ROOT / "svoe-vino-scanner" / "runs"
PAIRS_JSON = P.ROOT / "Code" / "data" / "raw" / "pairs" / "benchmark.json"
PAIRS_IMG = P.ROOT / "Code" / "data" / "raw" / "pairs" / "roskachestvo"
PHONE_IMG = RUNS / "pairs-phone-images"
GOAL26 = Path(
    r"<tmp>\goal26"
)
TOP_K = 20


# ------------------------------------------------------------------ кадры наборов
def field_items(set_name: str, qset: str, keep: set[str] | None = None) -> list[dict[str, Any]]:
    meta = P.jsonl(P.FROZEN_FD / "sets" / set_name / "meta.jsonl")
    if keep is not None:
        meta = [m for m in meta if m["query_id"] in keep]
    qids, vecs, names = R.load_qemb(set_name)
    pos = {q: i for i, q in enumerate(qids)}
    out = []
    for m in meta:
        in_cat = bool(m.get("in_catalog", True)) and m.get("slug") not in (None, "__none__")
        out.append(
            {
                "id": m["query_id"],
                "set": qset,
                "source": m.get("dataset_source") or "",
                "photo": m.get("photo"),
                "image": str(R.image_path(m)),
                "slug": m["slug"] if in_cat else None,
                "acceptable": list(m.get("acceptable") or []) if in_cat else [],
                "in_catalog": in_cat,
                "wine_source": m.get("source"),
                "same_packshot": bool(m.get("same_packshot")),
                "_vec": np.asarray(vecs[pos[m["query_id"]]]),
                "_names": names,
            }
        )
    return out


def pair_readings(run: str) -> dict[str, Reading]:
    out = {}
    for r in P.jsonl(RUNS / run / "predictions.jsonl"):
        (rd,) = r["readers"]
        h = hashlib.sha1(rd["reader"].encode()).hexdigest()
        rec = json.loads(
            (RUNS / "cache-pairs-ocr" / h[:2] / f"{h}.json").read_text(encoding="utf-8")
        )
        assert rec["key"] == rd["reader"]
        out[r["query_id"]] = Reading.model_validate(rec["reading"])
    return out


def pair_items() -> list[dict[str, Any]]:
    z = np.load(P.PROTOCOL / "qemb_pairs.npz")
    pos = {str(q): i for i, q in enumerate(z["qids"])}
    names = [str(n) for n in z["names"]]
    recs = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    old_split = {
        r["query_id"]: r["meta"]["split"]
        for r in P.jsonl(RUNS / "goal" / "iter20" / "cvall-pairs" / "predictions.jsonl")
    }
    reads = {
        "pairs": pair_readings("ocr-pairs-vlm35"),
        "pairs_phone": pair_readings("ocr-pairsphone-vlm35"),
    }
    out = []
    for qset, img_dir, ext in (("pairs", PAIRS_IMG, "webp"), ("pairs_phone", PHONE_IMG, "jpg")):
        for r in recs:
            wid = r["wine_id"]
            qid = f"{qset}:{wid}"
            out.append(
                {
                    "id": qid,
                    "set": qset,
                    "source": "roskachestvo",
                    "photo": wid,
                    "image": str(img_dir / f"{wid}.{ext}"),
                    "slug": r["portal_slug"],
                    "acceptable": [],
                    "in_catalog": True,
                    "wine_source": wid,
                    "same_packshot": False,
                    "goal_split": old_split[wid],  # dev — в обучении -goal, test — нет
                    "_vec": np.asarray(z["vectors"][pos[qid]]),
                    "_names": names,
                    "_reading": reads[qset][wid],
                }
            )
    return out


def items_of(qset: str) -> list[dict[str, Any]]:
    split = P.load_split()
    if qset == "catalog_v2":
        return field_items("catalog_v2", "catalog_v2")
    if qset == "kr_dev":
        return field_items("krasnostop_v1", "kr_dev", set(split["kr_dev"]))
    if qset == "kr_dev_sp":
        return field_items("krasnostop_v1", "kr_dev_sp", set(split["kr_dev_same_packshot"]))
    if qset == "ooc_v2":
        return field_items("ooc_v2", "ooc_v2")
    if qset == "pairs":
        return pair_items()
    raise SystemExit(f"неизвестный набор {qset}")


# ------------------------------------------------------------------ прогон
def label(row: dict[str, Any]) -> dict[str, Any]:
    return {"slug": row["slug"], "acceptable": row["acceptable"]}


def run_set(rp: R.FundReplay, qset: str) -> None:
    items = items_of(qset)
    checked = P.assert_no_test([it["id"] for it in items], [it["image"] for it in items])
    names = list(rp.svc.model.feature_names)
    rows, feats, fslugs, vecs = [], [], [], []
    failed: list[str] = []
    t0 = time.perf_counter()
    for it in items:
        rp.forced_reading = it.get("_reading")
        row = rp.scan(Path(it["image"]).read_bytes(), it["_vec"])
        rp.forced_reading = None
        if not row["ok"]:
            # промах кэша чтений: у вина каталога — ошибка стенда, у вина вне каталога — пропуск
            assert not it["in_catalog"], (it["id"], row.get("error"))
            failed.append(it["id"])
            continue
        replayed = rp.resolve_row(row)
        f = np.full((TOP_K, len(names)), np.nan, dtype=np.float32)
        fs = [""] * TOP_K
        if row.get("features"):
            m = np.asarray(row["features"], dtype=np.float32)
            f[: len(m)] = m
            fs[: len(m)] = row["feat_slugs"]
        lab = label(it)
        cv_slugs = [c[0] for c in row["cv"]]
        rec = {k: v for k, v in it.items() if not k.startswith("_")}
        rec.update(
            answer=row["answer"],
            outcome=row["outcome"],
            p_answer=row.get("p_answer"),
            cv=row["cv"],
            margin=row["margin"],
            read=row["read"],
            vlm_status=row["vlm_status"],
            vlm_lines=row["vlm_lines"],
            resolve=row.get("resolve"),
            rank_slugs=row.get("rank_slugs"),
            rank_scores=row.get("rank_scores"),
            p_top1=row.get("p_top1"),
            replay_equal=replayed == row["answer"],
            correct=P.correct(row["answer"], lab) if it["in_catalog"] else None,
            strict=P.strict(row["answer"], lab) if it["in_catalog"] else None,
            cv_correct=P.correct(cv_slugs[0], lab) if it["in_catalog"] else None,
            true_rank_cv=next((i + 1 for i, s in enumerate(cv_slugs) if P.correct(s, lab)), None)
            if it["in_catalog"]
            else None,
        )
        rows.append(rec)
        feats.append(f)
        fslugs.append(fs)
        vecs.append(it["_vec"].astype(np.float16))
    PARTS.mkdir(parents=True, exist_ok=True)
    sets = sorted({r["set"] for r in rows})
    for s in sets:
        idx = [i for i, r in enumerate(rows) if r["set"] == s]
        with (PARTS / f"{s}.jsonl").open("w", encoding="utf-8") as fh:
            for i in idx:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        np.savez_compressed(
            PARTS / f"{s}.npz",
            ids=np.array([rows[i]["id"] for i in idx]),
            features=np.stack([feats[i] for i in idx]),
            feat_slugs=np.array([fslugs[i] for i in idx]),
            feature_names=np.array(names),
            qemb=np.stack([vecs[i] for i in idx]),
            qemb_names=np.array(items[0]["_names"]),
        )
        rs = [rows[i] for i in idx]
        inc = [r for r in rs if r["in_catalog"]]
        print(
            json.dumps(
                {
                    "set": s,
                    "frames": len(rs),
                    "guard_checked": checked,
                    "replay_equal": sum(r["replay_equal"] for r in rs),
                    "same_wine": sum(bool(r["correct"]) for r in inc),
                    "strict": sum(bool(r["strict"]) for r in inc),
                    "cv_top1": sum(bool(r["cv_correct"]) for r in inc),
                    "in_top20": sum(r["true_rank_cv"] is not None for r in inc),
                    "vlm_status": dict(Counter(r["vlm_status"] for r in rs)),
                    "reads": {"hits": rp.hits, "misses": rp.misses},
                    "skipped_cache_miss": failed,
                    "wall_s": round(time.perf_counter() - t0, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


# ------------------------------------------------------------------ сборка
def merge() -> None:
    order = ["catalog_v2", "kr_dev", "kr_dev_sp", "ooc_v2", "pairs", "pairs_phone"]
    rows: list[dict[str, Any]] = []
    npz: dict[str, list[np.ndarray]] = {"features": [], "feat_slugs": [], "qemb": []}
    names = qnames = None
    for s in order:
        path = PARTS / f"{s}.jsonl"
        if not path.exists():
            print("нет части", s)
            continue
        part = P.jsonl(path)
        z = np.load(PARTS / f"{s}.npz")
        assert [str(x) for x in z["ids"]] == [r["id"] for r in part]
        names = names or [str(x) for x in z["feature_names"]]
        qnames = qnames or [str(x) for x in z["qemb_names"]]
        assert names == [str(x) for x in z["feature_names"]] and qnames == [
            str(x) for x in z["qemb_names"]
        ]
        rows += part
        for k in npz:
            npz[k].append(z[k])
    P.assert_no_test([r["id"] for r in rows], [r["image"] for r in rows])
    # Группы вин по всему пулу: slug ∪ acceptable ∪ wine_id; вне каталога — своя группа на вино.
    dsu = P.DSU()
    for r in rows:
        if r["in_catalog"]:
            P.group_key(dsu, r)
    test_slugs = set(P.load_split()["test_slugs"])
    for i, r in enumerate(rows):
        r["row"] = i
        if r["in_catalog"]:
            r["group"] = dsu.find(P.label_nodes(r)[0])
            r["kr_test_wine"] = bool({r["slug"], *r["acceptable"]} & test_slugs)
        else:
            r["group"] = f"ooc:{r['wine_source']}"
            r["kr_test_wine"] = False
    with (P.PROTOCOL / "trainpool.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(
        P.PROTOCOL / "trainpool.npz",
        ids=np.array([r["id"] for r in rows]),
        features=np.concatenate(npz["features"]),
        feat_slugs=np.concatenate(npz["feat_slugs"]),
        feature_names=np.array(names),
        qemb=np.concatenate(npz["qemb"]),
        qemb_names=np.array(qnames),
    )
    by = Counter(r["set"] for r in rows)
    summary = {}
    for s in order:
        rs = [r for r in rows if r["set"] == s]
        if not rs:
            continue
        inc = [r for r in rs if r["in_catalog"]]
        summary[s] = {
            "frames": by[s],
            "groups": len({r["group"] for r in rs}),
            "same_wine": sum(bool(r["correct"]) for r in inc),
            "strict": sum(bool(r["strict"]) for r in inc),
            "cv_top1": sum(bool(r["cv_correct"]) for r in inc),
            "in_top20": sum(r["true_rank_cv"] is not None for r in inc),
            "replay_equal": sum(r["replay_equal"] for r in rs),
            "kr_test_wine_frames": sum(r["kr_test_wine"] for r in rs),
        }
    summary["_total"] = {"frames": len(rows), "groups": len({r["group"] for r in rows})}
    (P.PROTOCOL / "trainpool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def check_goal26() -> None:
    """Сверка ответов v2 и kr-dev с итоговой сборкой 25.09 (тот же код, те же данные)."""
    for s, f in (
        ("catalog_v2", "final_v2.json"),
        ("kr_dev", "final_kr.json"),
        ("kr_dev_sp", "final_kr.json"),
    ):
        path = PARTS / f"{s}.jsonl"
        if not path.exists():
            continue
        ref = json.loads((GOAL26 / f).read_text(encoding="utf-8"))["rows"]
        rs = P.jsonl(path)
        same = sum(ref[r["id"]]["off"] == r["answer"] for r in rs)
        print(f"{s}: совпало с goal26 {same}/{len(rs)}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.sets:
        rp = R.FundReplay()
        for s in args.sets:
            run_set(rp, s)
        check_goal26()
    if args.merge:
        merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
