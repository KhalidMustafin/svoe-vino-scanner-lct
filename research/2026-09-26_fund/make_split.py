"""Разбиение krasnostop_v1 на kr-dev и kr-test по группам вин (seed 20260926). Только CPU, без моделей.

Группа — связная компонента «того же вина» по всем 903 кадрам: slug, acceptable и их wine_id
(`wine_groups_final.json` снимка). Кадры одной группы, в том числе same_packshot, всегда в одной
половине. Порядок групп — перемешивание с seed, затем по убыванию числа основных кадров; каждая
группа идёт в половину, где меньше штраф дисбаланса: |Δ кадров| + Σ по винодельням |Δ| + Σ по
цветам |Δ| (всё в основных кадрах), при равенстве — монетка того же генератора. Ответы и ошибки
сканера не читаются: разбиение от них не зависит.

    PYTHONPATH=. python research/2026-09-26_fund/make_split.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P

KR_MATCH = P.ROOT / "krastostop" / "match"
# Поля строк сопоставления, где стоят slug: метка, допустимые и кандидаты (без «grapes»).
LABEL_KEYS = {
    "slug",
    "slug_csv",
    "acceptable",
    "acceptable_csv",
    "acceptable_equiv",
    "csv_slugs",
    "matched_card",
    "candidates",
    "best",
}


def catalog_slugs_in(obj: Any, known: set[str]) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, str):
        if obj in known:
            out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out |= catalog_slugs_in(v, known)
    elif isinstance(obj, list):
        for v in obj:
            out |= catalog_slugs_in(v, known)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    meta_path = P.FROZEN_FD / "sets" / "krasnostop_v1" / "meta.jsonl"
    groups_path = P.FROZEN_FD / "catalog" / "wine_groups_final.json"
    meta = P.jsonl(meta_path)
    wines = P.wine_of()
    attrs = P.wine_attrs()

    dsu = P.DSU()
    for m in meta:
        P.group_key(dsu, m)
    comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in meta:
        comp[dsu.find(P.label_nodes(m)[0])].append(m)

    def comp_info(ms: list[dict[str, Any]]) -> dict[str, Any]:
        wids = sorted({wines[s] for m in ms for s in [m["slug"], *(m.get("acceptable") or [])]})
        a = attrs[wines[ms[0]["slug"]]]
        return {
            "wine_ids": wids,
            "slugs": sorted({s for m in ms for s in [m["slug"], *(m.get("acceptable") or [])]}),
            "winery": a.get("winery") or "?",
            "colour": a.get("color") or "?",
            "sugar": a.get("sugar") or "?",
            "primary": sorted(m["query_id"] for m in ms if not m.get("same_packshot")),
            "same_packshot": sorted(m["query_id"] for m in ms if m.get("same_packshot")),
        }

    info = {k: comp_info(v) for k, v in comp.items()}
    rng = np.random.default_rng(P.SPLIT_SEED)
    keys = sorted(info)
    perm = [keys[i] for i in rng.permutation(len(keys))]
    order = sorted(perm, key=lambda k: -len(info[k]["primary"]))  # sorted стабилен

    frames = {"dev": 0, "test": 0}
    by_w: dict[str, Counter[str]] = {"dev": Counter(), "test": Counter()}
    by_c: dict[str, Counter[str]] = {"dev": Counter(), "test": Counter()}
    half_of: dict[str, str] = {}
    for k in order:
        c = info[k]
        n = len(c["primary"])
        costs = {}
        for h in ("dev", "test"):
            o = "test" if h == "dev" else "dev"
            f = abs(frames[h] + n - frames[o])
            w = abs(by_w[h][c["winery"]] + n - by_w[o][c["winery"]])
            col = abs(by_c[h][c["colour"]] + n - by_c[o][c["colour"]])
            costs[h] = f + w + col
        if costs["dev"] == costs["test"]:
            h = "dev" if rng.random() < 0.5 else "test"
        else:
            h = min(costs, key=costs.__getitem__)
        half_of[k] = h
        frames[h] += n
        by_w[h][c["winery"]] += n
        by_c[h][c["colour"]] += n

    def collect(h: str, field: str) -> list[str]:
        return sorted(q for k, c in info.items() if half_of[k] == h for q in c[field])

    kr_dev, kr_test = collect("dev", "primary"), collect("test", "primary")
    dev_sp, test_sp = collect("dev", "same_packshot"), collect("test", "same_packshot")
    test_slugs = sorted({s for k, c in info.items() if half_of[k] == "test" for s in c["slugs"]})
    dev_slugs = sorted({s for k, c in info.items() if half_of[k] == "dev" for s in c["slugs"]})
    assert not set(test_slugs) & set(dev_slugs)
    assert not set(kr_dev) & set(kr_test)

    by_q = {m["query_id"]: m for m in meta}
    allowed = sorted(P.norm_path(by_q[q]["image"]) for q in kr_dev + dev_sp)
    forbidden = {P.norm_path(by_q[q]["image"]) for q in kr_test + test_sp}
    # Прочие картинки krasnostop, где хоть где-то (метка, acceptable, кандидаты) стоит slug kr-test.
    # Кадры набора решает meta (в ней исправленные метки); здесь — картинки вне набора.
    known = set(wines)
    tset = set(test_slugs)
    in_set = {P.norm_path(m["image"]) for m in meta}
    extra = Counter()
    for name in ("matches.jsonl", "uncertain.jsonl", "none.jsonl", "auto.jsonl"):
        for row in P.jsonl(KR_MATCH / name):
            if not row.get("image") or P.norm_path(row["image"]) in in_set:
                continue
            fields = {k: v for k, v in row.items() if k in LABEL_KEYS}
            if catalog_slugs_in(fields, known) & tset:
                forbidden.add(P.norm_path(row["image"]))
                extra[name] += 1
    assert not forbidden & set(allowed)

    def mix(qs: list[str]) -> dict[str, Any]:
        cs = [info[dsu.find(P.label_nodes(by_q[q])[0])] for q in qs]
        return {
            "frames": len(qs),
            "groups": len({dsu.find(P.label_nodes(by_q[q])[0]) for q in qs}),
            "colour": dict(Counter(c["colour"] for c in cs).most_common()),
            "sugar": dict(Counter(c["sugar"] for c in cs).most_common()),
            "packaging": sum(bool(by_q[q].get("packaging")) for q in qs),
            "not_075": sum(by_q[q].get("volume_l") != 0.75 for q in qs),
            "match_verified": sum(by_q[q].get("match_method") == "verified" for q in qs),
            "wineries": len({c["winery"] for c in cs}),
        }

    w_dev = {info[k]["winery"] for k in info if half_of[k] == "dev" and info[k]["primary"]}
    w_test = {info[k]["winery"] for k in info if half_of[k] == "test" and info[k]["primary"]}
    top_w = Counter(info[k]["winery"] for k in info for _ in info[k]["primary"]).most_common(15)
    summary = {
        "kr_dev": mix(kr_dev),
        "kr_test": mix(kr_test),
        "same_packshot": {"dev": len(dev_sp), "test": len(test_sp)},
        "wineries_both_halves": len(w_dev & w_test),
        "wineries_only_dev": len(w_dev - w_test),
        "wineries_only_test": len(w_test - w_dev),
        "top15_wineries_primary": {
            w: {"dev": by_w["dev"][w], "test": by_w["test"][w]} for w, _ in top_w
        },
        "groups_total": len(info),
        "groups_merged_by_acceptable": sum(len(c["wine_ids"]) > 1 for c in info.values()),
        "forbidden_kr_images": len(forbidden),
        "forbidden_extra_from_match_files": dict(extra),
    }
    out = {
        "what": "разбиение krasnostop_v1 на kr-dev / kr-test по группам вин (EVAL_PROTOCOL.md)",
        "created": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "seed": P.SPLIT_SEED,
        "method": "компоненты slug ∪ acceptable ∪ wine_id; жадный баланс кадры + винодельня + цвет",
        "inputs": {
            "meta": str(meta_path),
            "meta_sha1": P.sha1_file(meta_path),
            "wine_groups": str(groups_path),
            "wine_groups_sha1": P.sha1_file(groups_path),
        },
        "summary": summary,
        "kr_dev": kr_dev,
        "kr_test": kr_test,
        "kr_dev_same_packshot": dev_sp,
        "kr_test_same_packshot": test_sp,
        "test_slugs": test_slugs,
        "allowed_kr_images": allowed,
        "forbidden_kr_images": sorted(forbidden),
        "groups": {
            k: {
                "half": half_of[k],
                **{
                    f: info[k][f]
                    for f in ("wine_ids", "winery", "colour", "primary", "same_packshot")
                },
            }
            for k in sorted(info)
        },
    }
    text = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
    P.SPLIT.write_text(text, encoding="utf-8")
    P.PROTOCOL.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P.SPLIT, P.PROTOCOL / "split.json")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("sha1", P.sha1_file(P.SPLIT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
