"""Слой «после поиска» на настоящих данных, без моделей и без HTTP-порта (только CPU).

Что проверяется:

1. Карточка `/v1/wines/{slug}` для всех 2 176 позиций справочника и разметки: нет путей машины
   и `photo_path`, сколько карточек с фото, блюдами, температурой, крепостью, описанием портала.
2. `/similar` через тот же код, что у маршрута (`AfterSearch.similar`, с проверкой фото на
   диске): время на вино, строки проходят content_filter, нет «%».
3. `after` на записанных ответах поля: 353 кадра v2 в каталоге (`runs/.../v2prod/iter20`).
   Экран по правилу Д1, сошлась ли прочитанная винодельня с винодельней правильного вина,
   как часто «игристое» ставится верно (по разметке кадра `labels_v2.jsonl` и по справочнику).
4. «Не тупик» (`by-label`) на примерах из плана: Alveus Оранж Брют (Фанагория), Barolo.

    .venv/Scripts/python.exe research/2026-09-24_after/probe_api.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.api.after_layer import AfterSearch, ByLabelBody, after_block  # noqa: E402
from app.api.cards import CatalogCards, read_records  # noqa: E402
from app.api.config import ServiceSettings  # noqa: E402
from app.api.service import Confidence, ScanResult, TopItem  # noqa: E402
from app.recommend.content_filter import check  # noqa: E402
from app.resolve.attrs import CatalogAttrs  # noqa: E402

ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|/Users/|/home/")
FIELD = REPO.parent / "field_dataset"
PREDICTIONS = REPO.parent / "svoe-vino-scanner" / "runs/field25/iters/runs/v2prod/iter20"


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
    settings = ServiceSettings.from_env(os.environ)
    t0 = time.perf_counter()
    records = read_records(settings.attrs_path)
    attrs = CatalogAttrs.from_records(records)
    cards = CatalogCards.build(records, csv_path=settings.catalog_csv, photo_map=settings.photo_map)
    after = AfterSearch.load(settings, cards, attrs)
    print(f"сборка слоя: {time.perf_counter() - t0:.2f} с; {after.stats()}")
    out: dict = {"stats": after.stats()}

    # 1. карточки
    slugs = sorted(set(cards.by_slug) | set(after.catalog.by_slug))
    cover = Counter()
    leaks = []
    t0 = time.perf_counter()
    for slug in slugs:
        card = after.card(slug)
        cover["cards"] += 1
        cover["photo_url"] += card["photo_url"] is not None
        cover["portal_url"] += card["portal_url"] is not None
        cover["dishes"] += bool(card["dishes"])
        cover["dish_icons_all"] += bool(card["dishes"]) and all(d["icon_url"] for d in card["dishes"])
        cover["temperature"] += card["temperature"] is not None
        cover["alcohol"] += card["alcohol"] is not None
        cover[f"alcohol_src:{card['alcohol_src']}"] += 1
        cover[f"description_src:{card['description_src']}"] += 1
        cover["gradient"] += card["category_gradient"] is not None
        if "photo_path" in card or any(ABS_PATH.search(s) for s in strings(card)):
            leaks.append(slug)
    cards_s = time.perf_counter() - t0
    published = sum(1 for s in slugs if after.card(s)["published"])
    out["cards"] = {**cover, "published": published, "leaks": leaks[:10], "n_leaks": len(leaks),
                    "seconds": round(cards_s, 2)}  # fmt: skip
    print("карточки:", json.dumps(out["cards"], ensure_ascii=False))

    # 2. похожие через слой (с фото на диске)
    times = []
    bad_strings = []
    percent = 0
    photos_missing = 0
    for wine in after.catalog.pool:
        started = time.perf_counter()
        body = after.similar(wine.slug, limit=3, order="reco")
        times.append((time.perf_counter() - started) * 1000)
        text = json.dumps(body, ensure_ascii=False)
        percent += "%" in text
        photos_missing += sum(item["photo_url"] is None for item in body["items"])
        bad_strings.extend(s for s in strings(body) if not check(s).clean)
    times.sort()
    out["similar"] = {
        "anchors": len(times),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(times[int(0.95 * (len(times) - 1))], 3),
        "max_ms": round(times[-1], 3),
        "with_percent": percent,
        "content_filter_bad": len(bad_strings),
        "tiles_without_photo": photos_missing,
    }
    print("похожие:", json.dumps(out["similar"], ensure_ascii=False))

    # 3. after на записанных ответах поля
    labels = {}
    for line in (FIELD / "labels_v2.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = row
    states = Counter()
    winery = Counter()
    sparkling = Counter()
    catalog_sparkling = Counter()
    examples = []
    rows = [json.loads(line) for line in (PREDICTIONS / "predictions.jsonl").open(encoding="utf-8")]
    for row in rows:
        p1 = row.get("p_top1")
        result = ScanResult(
            slug=row["slug"],
            confidence=Confidence(top1=p1),
            top5=[TopItem(slug=s, score=p) for s, p in (row.get("resolve") or [])[:5]],
            outcome=row["outcome"],
            error=row.get("error"),
            evidence={"vlm": row.get("vlm") or {}},
        )
        block = after_block(result, after)
        states[block["state"]] += 1
        label = labels.get(row["query_id"].split("-")[0]) or {}
        truth = after.catalog.get(label.get("gt_slug") or "")
        read = block["read"]
        if read["winery"] is None:
            winery["not_read"] += 1
        elif not block["winery_in_catalog"]:
            winery["read_not_in_catalog"] += 1
        elif truth is not None and truth.slug in block["winery_slugs"]:
            winery["matches_truth_winery"] += 1
        elif truth is not None and truth.winery == read["winery"]:
            winery["matches_truth_winery_not_in_pool"] += 1
        else:
            winery["other_winery"] += 1
        # игристость сверяется с разметкой кадра (человек читал этикетку): у справочника флаг
        # игристого бывает не выставлен у полусладких игристых Инкермана и Золотой Балки
        seen = (label.get("reading") or {}).get("sparkling")
        sparkling[(seen, read["sparkling"])] += 1
        catalog_sparkling[(truth.sparkling if truth else None, read["sparkling"])] += 1
        if len(examples) < 8 and block["read_label"]:
            examples.append((row["query_id"], block["state"], block["read_label"]))
    out["field_after"] = {
        "frames": len(rows),
        "states": dict(states),
        "winery": dict(winery),
        "sparkling_label_vs_read": {f"{k[0]}→{k[1]}": v for k, v in sparkling.items()},
        "sparkling_catalog_vs_read": {
            f"{k[0]}→{k[1]}": v for k, v in catalog_sparkling.items()
        },
        "examples": examples,
    }
    print("after на поле:", json.dumps(out["field_after"], ensure_ascii=False, indent=1))

    # 4. «Не тупик»
    cases = {
        "alveus_orange_brut": {"winery": "Фанагория", "color": "Оранжевое", "sugar": "brut",
                               "grapes": [], "abv": 12, "sparkling": True},
        "barolo": {"winery": "Barolo", "color": "Красное", "sugar": "suhoe", "grapes": ["nebbiolo"],
                   "abv": 14, "sparkling": None},
        "only_winery": {"winery": "Абрау-Дюрсо"},
    }  # fmt: skip
    out["by_label"] = {}
    for name, payload in cases.items():
        body = after.by_label(ByLabelBody(**payload), limit=3, order="reco")
        short = {
            "winery": body["winery"],
            "same_winery": [(t["name"], t["reasons"]) for t in body["same_winery"][:3]],
            "similar": [(t["winery"], t["name"], t["reasons"]) for t in body["similar"]],
            "notes": body["notes"],
            "percent": "%" in json.dumps(body, ensure_ascii=False),
        }
        out["by_label"][name] = short
        print(f"\nby-label {name}:", json.dumps(short, ensure_ascii=False, indent=1))

    (HERE / "probe_api.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
