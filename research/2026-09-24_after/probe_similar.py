"""Приёмка утра Д1 (внутренний план команды вне репозитория, §3): похожие фактами на всём пуле.

Для каждого вина пула — `/similar` в режимах reco и plain тем же кодом, что у сервиса
(`app.recommend.facts`), и плитки так, как их отдаёт API. Критерии:

    ≥3 похожих у ≥99 % вин
    в тройке 0 вин той же винодельни и той же группы wine_id
    общий сорт у первого вина ≥90 %
    отличие фактом в тройке ≥90 % (регион, крепость ≥0,5°, набор сортов, сахар — как в зонде
    плана (вне репозитория); цвет в /similar всегда тот же)
    нет «%» и «расхождений почти нет» ни в одной строке ответа
    все строки проходят content_filter
    p95 < 20 мс на CPU (подбор + плитки, без HTTP)

Без портала (решение 24.09): пул — канонические позиции выгрузки организатора с фото выгрузки
(1 980 вин; фото чужой бутылки, `WRONG_PHOTOS`, не показывается, но вино в пуле), факты — только
выгрузка (сахар из названия и slug — известен у 1 727 из 2 103, крепость из slug без года урожая —
у 1 567). Отдельно считается, как живут якоря без сахара. До решения, на
пуле с порталом (1 996 вин, сахар живого портала у всех): ≥3 у 100 %, общий сорт у первого
94,8 %, отличие фактом 100 %, p95 1,9 мс (коммит 1dfc161, `probe_similar.json`).

Только CPU и файлы справочника; сеть и модели не нужны.

    .venv/Scripts/python.exe research/2026-09-24_after/probe_similar.py [--data DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.recommend.catalog import RecoCatalog  # noqa: E402
from app.recommend.content_filter import check  # noqa: E402
from app.recommend.facts import Facts, differs, plain, similar, tile  # noqa: E402

BANNED = ("%", "расхождений почти нет")


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.environ.get("SVS_DATA_DIR") or str(REPO / "data"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out", default=str(HERE / "probe_similar.json"))
    args = parser.parse_args()
    data = Path(args.data)

    t0 = time.perf_counter()
    catalog = RecoCatalog.load(
        data / "gt" / "gt_tokens.jsonl",
        wines_path=data / "catalog" / "wines.jsonl",
        groups_path=data / "catalog" / "wine_groups.json",
    )
    load_s = time.perf_counter() - t0
    pool = catalog.pool
    limit = args.limit
    print(f"справочник: {catalog.stats()} за {load_s:.2f} с")

    n = len(pool)
    full = 0
    same_winery = same_group = 0
    first_shared = 0
    anchors_with_grapes = first_shared_known = 0
    triple_differs = 0
    first_differs = 0
    banned_hits: Counter[str] = Counter()
    filter_hits: list[tuple[str, str, list[str]]] = []
    plain_full = 0
    plain_bad: list[str] = []
    times_ms: list[float] = []
    reason_kinds: Counter[str] = Counter()
    triples: Counter[tuple[str, ...]] = Counter()
    short: list[str] = []
    examples: dict[str, dict] = {}
    checked_strings = 0
    no_sugar = no_sugar_full = 0
    for anchor in pool:
        started = time.perf_counter()
        facts = Facts.of_wine(anchor, catalog.alcohol(anchor.slug))
        reco = similar(catalog, facts, limit)
        body = {
            "slug": anchor.slug,
            "order": "reco",
            "limit": limit,
            "notice": "Применяются рекомендательные технологии",
            "category_label": anchor.style_label,
            "items": [
                tile(p.wine, p.reasons, photo_url=f"/v1/wines/{p.wine.slug}/photo")
                for p in reco.picks
            ],
            "notes": [note.as_dict() for note in reco.notes],
        }
        times_ms.append((time.perf_counter() - started) * 1000)
        wines = reco.wines
        if len(wines) >= limit:
            full += 1
        else:
            short.append(anchor.slug)
        if anchor.sugar is None:
            no_sugar += 1
            no_sugar_full += len(wines) >= limit
        same_winery += sum(w.winery_norm == anchor.winery_norm for w in wines)
        same_group += sum(w.wine_id == anchor.wine_id for w in wines)
        if wines:
            shared = bool(set(wines[0].grapes) & set(anchor.grapes))
            first_shared += shared
            if anchor.grapes:
                anchors_with_grapes += 1
                first_shared_known += shared
            first_differs += differs(facts, wines[0], catalog.alcohol(wines[0].slug))
        if any(p.differs for p in reco.picks):
            triple_differs += 1
        triples[tuple(sorted(w.slug for w in wines))] += 1
        for p in reco.picks:
            for reason in p.reasons:
                reason_kinds[reason.split(" — ")[0].split(",")[0].split(" ")[0]] += 1

        flat = plain(catalog, facts, limit)
        plain_body = {
            "items": [tile(p.wine, (), photo_url=None) for p in flat.picks],
            "notes": [note.as_dict() for note in flat.notes],
        }
        if len(flat.picks) >= limit:
            plain_full += 1
        names = [p.wine.title.casefold() for p in flat.picks]
        if names != sorted(names) or any(
            p.wine.winery_norm == anchor.winery_norm or p.wine.wine_id == anchor.wine_id
            for p in flat.picks
        ):
            plain_bad.append(anchor.slug)

        for text in [*strings(body), *strings(plain_body)]:
            checked_strings += 1
            for banned in BANNED:
                if banned in text:
                    banned_hits[banned] += 1
            if body["notice"] == text:
                continue
            verdict = check(text)
            if not verdict.clean:
                filter_hits.append((anchor.slug, text, verdict.violations))
        if anchor.slug in (
            "shato-pino-shiraz-krasnoe-suhoe-14",
            "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16",
            "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135",
            "cru-lermont-risling",
        ):
            examples[anchor.slug] = body

    times = sorted(times_ms)
    p95 = times[int(0.95 * (len(times) - 1))]
    result = {
        "pool": n,
        "ge3": {"n": full, "share": round(full / n, 4), "need": 0.99},
        "anchors_without_sugar": {
            "n": no_sugar,
            "ge3": no_sugar_full,
            "share": round(no_sugar_full / max(1, no_sugar), 4),
        },
        "catalog": catalog.stats(),
        "same_winery_in_top": same_winery,
        "same_group_in_top": same_group,
        "first_shares_grape": {"n": first_shared, "share": round(first_shared / n, 4), "need": 0.9},
        "first_shares_grape_known": {
            "n": first_shared_known,
            "of": anchors_with_grapes,
            "share": round(first_shared_known / max(1, anchors_with_grapes), 4),
        },
        "triple_differs": {"n": triple_differs, "share": round(triple_differs / n, 4), "need": 0.9},
        "first_differs": {"n": first_differs, "share": round(first_differs / n, 4)},
        "banned": dict(banned_hits),
        "content_filter_violations": len(filter_hits),
        "strings_checked": checked_strings,
        "ms": {
            "p50": round(statistics.median(times), 3),
            "p95": round(p95, 3),
            "max": round(times[-1], 3),
            "need_p95": 20,
        },
        "distinct_triples": len(triples),
        "most_common_triple": triples.most_common(1)[0][1],
        "short": short[:20],
        "plain": {"ge3": plain_full, "bad_order_or_own_winery": len(plain_bad)},
        "reason_heads": dict(reason_kinds.most_common()),
    }
    verdict = {
        "ge3": full / n >= 0.99,
        "no_own_winery_or_group": same_winery == 0 and same_group == 0,
        "first_shares_grape": first_shared / n >= 0.9,
        "triple_differs": triple_differs / n >= 0.9,
        "no_percent_or_degenerate": not banned_hits,
        "content_filter": not filter_hits,
        "p95_under_20ms": p95 < 20,
    }
    result["pass"] = verdict
    Path(args.out).write_text(
        json.dumps({**result, "filter_hits": filter_hits[:20], "examples": examples},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    for slug, body in examples.items():
        print(f"\n{slug}:")
        for item in body["items"]:
            print(f"  {item['winery']} — {item['name']}: {' · '.join(item['reasons'])}")
    print("\nприёмка:", "пройдена" if all(verdict.values()) else "НЕ пройдена", verdict)
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
