"""Приёмка «Сомелье у полки» (внутренний план команды вне репозитория, Д2 п. 10) на живом справочнике.

200 якорей из пула × 4 чипа блюда (meat, fish, cheese, none) × 3 направления (fresher, softer,
sweeter) = 2 400 комбинаций. Ответ собирается тем же кодом, что у маршрута
`GET /v1/wines/{slug}/shelf` (`AfterSearch.shelf`), вместе с плитками и фильтром строк.

Приёмка:

    0 исключений
    p95 < 50 мс на CPU (подбор + плитки, без HTTP)
    ≥3 вин в ≥90 % комбинаций, где направление можно оценить (после расширения пула); доля
    по всем комбинациям пишется рядом
    честная фраза в 100 % случаев, где вин меньше трёх или разницы нет; «разницы нет»
    проверяется отдельно: ни одно вино пула без своей винодельни и группы не сдвинуто в
    нужную сторону — тогда нота обязана быть `no_difference`, а если у якоря нет ни одного
    признака для направления (`Shelf.evaluable`: «послаще» без сахара и т. п.) — `not_evaluable`

«Направление не оценить» (перепроверка 24.09): у 376 вин выгрузки сахара нет ни в названии, ни в slug,
и «послаще» им считать не от чего. Такая комбинация — не провал подбора и не «разницы нет», а
«не оцениваемо»: она пишется отдельно (`not_evaluable`) и в долю ≥3 по оцениваемым не входит.
    все строки проходят content_filter, стоп-слов договора нет, знака «%» нет
    «помягче» никогда не «послаще»: 0 вин выдачи с известным сахаром выше потолка —
    max(полусухое, сахар якоря), у якоря без сахара полусухое

Записывается как есть:

    сколько различных троек даёт якорь на трёх направлениях (при одном блюде)
    сколько вин общих у «помягче» и «послаще»
    в какой доле выдач направление подтверждено фактом каталога, а не только осью «по сорту»
    сколько вин «помягче» с неизвестным сахаром (потолок их не проверяет) и сколько того же
    сахара, что у якоря слаще полусухого (их теперь пускает потолок)

Отдельно, без приёмки: каждое вино выдачи сверяется с направлением по сырым полям (сахар в
ступенях, крепость, оси профиля) — независимо от кода `shelf.py`.

`--sweep` — ещё и сплошной прогон: все вина справочника (и вне пула) × 4 блюда × 4 направления
(`reco`) и × 4 блюда (`plain`). Там те же проверки строк, честной фразы и потолка «помягче».

Потолок зонд считает сам, по сырым полям (`softer_ceiling`), а не через `move_of`.

Без портала (решение 24.09): чип блюда — правила сочетаний (`SVS_SOMM_DIR`, пары `yes` к
блюду группы чипа), пул — канонические позиции выгрузки с фото (1 980 вин), сахар — из названия
и slug (у 376 позиций неизвестен). Источник данных сомелье пишется в отчёт (`somm`): без сборки
движков сервис берёт 7 вин заглушки договора, и чип блюда почти всегда пуст. Отдельно, без
приёмки: каждое вино выдачи с выбранным блюдом сверяется с сырыми парами — у него есть пара
`yes` с блюдом этой группы. До решения, на блюдах портала и пуле 1 996 вин: ≥3 у 96,8 %
(коммит 1dfc161, `probe_shelf.json`).

Только CPU и файлы справочника; сеть и модели не нужны.

    SVS_DATA_DIR=... SVS_SOMM_DIR=... .venv/Scripts/python.exe \
        research/2026-09-24_after/probe_shelf.py --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.api.after_layer import AfterSearch  # noqa: E402
from app.api.cards import CatalogCards, read_records  # noqa: E402
from app.api.config import ServiceSettings  # noqa: E402
from app.recommend.catalog import SUGAR_STEPS  # noqa: E402
from app.recommend.content_filter import check  # noqa: E402
from app.recommend.facts import Facts, excluded  # noqa: E402
from app.recommend.shelf import SOFTER_MAX_STEP  # noqa: E402
from app.resolve.attrs import CatalogAttrs  # noqa: E402

FOODS = ("meat", "fish", "cheese", "none")
WANTS = ("fresher", "softer", "sweeter")
#: Стоп-слова договора (`tests/unit/test_after_fixtures.py`) и 38-ФЗ.
STOP = re.compile(
    r"купи|цен[аы]|₽|руб\.|лучш|вино недели|рейтинг|publicrating|скидк", re.IGNORECASE
)
EXAMPLES = (
    ("massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16", "fish", "fresher"),
    ("massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16", "none", "sweeter"),
    ("massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16", "none", "softer"),
    ("soyuz-vino-konfessa-sandzhoveze-polusladkoe-krasnoe-11", "none", "softer"),
    ("shato-pino-shiraz-krasnoe-suhoe-14", "meat", "fresher"),
    ("shato-pino-shiraz-krasnoe-suhoe-14", "meat", "softer"),
    ("shato-pino-shiraz-krasnoe-suhoe-14", "meat", "sweeter"),
    ("shato-pino-shiraz-krasnoe-suhoe-14", "fish", "softer"),
)


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from strings(value)


def softer_ceiling(anchor) -> int:
    """Потолок сахара «помягче» по правилу: полусухое или сахар якоря, если он слаще."""
    ours = SUGAR_STEPS.get(anchor.sugar or "")
    return SOFTER_MAX_STEP if ours is None else max(SOFTER_MAX_STEP, ours)


class SofterSugar:
    """Сахар вин «помягче» по сырым полям: выше потолка, неизвестный, на 2+ ступени выше.

    `above_semidry_same_as_anchor` — вина выше полусухого того же сахара, что якорь: их пускает
    потолок по сахару якоря (только «по сорту»). `drier` — суше якоря (должно быть 0).
    """

    def __init__(self) -> None:
        self.items = 0
        self.unknown = 0
        self.above: list[tuple[str, str, str]] = []
        self.drier: list[tuple[str, str, str]] = []
        self.same_above_semidry: Counter[str] = Counter()
        self.two_steps: Counter[str] = Counter()

    def add(self, anchor, wine, food: str, source: str | None) -> None:
        ours, theirs = SUGAR_STEPS.get(anchor.sugar or ""), SUGAR_STEPS.get(wine.sugar or "")
        self.items += 1
        if theirs is None:
            self.unknown += 1
            return
        if theirs > softer_ceiling(anchor):
            self.above.append((anchor.slug, food, wine.slug))
        elif ours is not None and theirs < ours:
            self.drier.append((anchor.slug, food, wine.slug))
        elif theirs > SOFTER_MAX_STEP:
            self.same_above_semidry[f"{wine.sugar}:{source}"] += 1
        elif ours is not None and theirs >= ours + 2:
            self.two_steps[str(source)] += 1

    def summary(self) -> dict:
        return {
            "items": self.items,
            "above_ceiling_known_sugar": len(self.above),
            "drier_than_anchor": len(self.drier),
            "above_semidry_same_as_anchor": dict(self.same_above_semidry),
            "unknown_sugar": self.unknown,
            "two_steps_up_within_cap": dict(self.two_steps),
            "above_examples": self.above[:10],
        }


def food_holds(after: AfterSearch, slug: str, food: str) -> bool:
    """Независимая сверка чипа блюда по сырым парам: есть пара `yes` с блюдом группы `food`."""
    pairs = after.somm.pairs.get(slug)
    if pairs is None:
        return False
    return any(
        pair.verdict == "yes" and (dish := after.somm.dishes.get(dish_id)) and dish.food == food
        for dish_id, pair in pairs.dishes.items()
    )


def direction_holds(after: AfterSearch, anchor, wine, want: str, source: str) -> bool:
    """Независимая сверка: сдвиг вина в сторону `want` по сырым полям, по заявленному источнику."""
    catalog = after.catalog
    profiles = after.sommelier.profiles
    ours, theirs = SUGAR_STEPS.get(anchor.sugar or ""), SUGAR_STEPS.get(wine.sugar or "")
    a_abv, w_abv = catalog.alcohol(anchor.slug).value, catalog.alcohol(wine.slug).value
    if source == "catalog":
        if ours is None or theirs is None:
            sugar_move = None
        else:
            sugar_move = theirs - ours
        if want == "sweeter":
            return sugar_move is not None and sugar_move > 0
        if want == "softer":
            return sugar_move == 1
        return (sugar_move is not None and sugar_move < 0) or (
            a_abv is not None and w_abv is not None and a_abv - w_abv >= 0.5
        )
    axis = "acidity" if want == "fresher" or anchor.color in ("Белое", "Розовое") else "tannin"
    sign = 1 if want == "fresher" else -1
    delta = sign * (profiles[wine.slug].axis(axis) - profiles[anchor.slug].axis(axis))
    return delta >= 0.3 - 1e-9


def sweep(after: AfterSearch) -> dict:
    """Все вина справочника × все чипы: исключения, честная фраза, строки, потолок «помягче»."""
    catalog = after.catalog
    pool = {wine.slug for wine in catalog.pool}
    anchors = sorted(catalog, key=lambda wine: wine.slug)
    combos = [("reco", food, want) for food in FOODS for want in (*WANTS, "none")]
    combos += [("plain", food, "none") for food in FOODS]
    times: list[float] = []
    errors = short_no_note = bad_strings = percent = calls = 0
    softer_sugar = SofterSugar()
    softer_by_anchor: Counter[str] = Counter()
    for anchor in anchors:
        for order, food, want in combos:
            calls += 1
            started = time.perf_counter()
            try:
                body = after.shelf(anchor.slug, food=food, want=want, order=order)
            except Exception:  # noqa: BLE001 — зонд считает исключения
                errors += 1
                continue
            times.append((time.perf_counter() - started) * 1000)
            codes = {note["code"] for note in body["notes"]}
            short_no_note += len(body["items"]) < 3 and not (
                codes & {"fewer_than_three", "no_difference", "not_evaluable"}
            )
            percent += "%" in json.dumps(body, ensure_ascii=False)
            bad_strings += sum(
                not check(text).clean or bool(STOP.search(text)) for text in strings(body)
            )
            if order == "reco" and want == "softer":
                for item in body["items"]:
                    softer_sugar.add(anchor, catalog.get(item["slug"]), food, item["want_source"])
                    softer_by_anchor["pool" if anchor.slug in pool else "not_in_pool"] += 1
    times.sort()
    summary = softer_sugar.summary()
    return {
        "anchors": len(anchors),
        "anchors_in_pool": len(pool),
        "calls": calls,
        "exceptions": errors,
        "ms": {
            "p50": round(statistics.median(times), 3),
            "p95": round(times[int(0.95 * (len(times) - 1))], 3),
            "max": round(times[-1], 3),
        },
        "short_without_note": short_no_note,
        "bad_strings": bad_strings,
        "percent_in_body": percent,
        "softer_sugar": {**summary, "items_by_anchor": dict(softer_by_anchor)},
        "pass": {
            "no_exceptions": errors == 0,
            "honest_when_short": short_no_note == 0,
            "content_filter": bad_strings == 0 and percent == 0,
            "softer_not_above_ceiling": summary["above_ceiling_known_sugar"] == 0
            and summary["drier_than_anchor"] == 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sweep", action="store_true", help="сплошной прогон по всем винам")
    parser.add_argument("--out", default=str(HERE / "probe_shelf.json"))
    args = parser.parse_args()

    t0 = time.perf_counter()
    settings = ServiceSettings.from_env(os.environ)
    records = read_records(settings.attrs_path)
    cards = CatalogCards.build(records)
    after = AfterSearch.load(settings, cards, CatalogAttrs.from_records(records))
    load_s = time.perf_counter() - t0
    catalog = after.catalog
    print(f"справочник: {catalog.stats()} за {load_s:.2f} с (с профилями стиля)")
    pool = sorted(wine.slug for wine in catalog.pool)
    anchors = random.Random(args.seed).sample(pool, args.anchors)

    times_ms: list[float] = []
    errors: list[tuple[str, str, str, str]] = []
    total = full = 0
    short_honest = short_total = 0
    no_diff_expected = no_diff_honest = 0
    no_diff_false = 0  # нота «разницы нет», хотя сдвиг в каталоге есть
    not_eval_expected = not_eval_honest = 0  # направление не оценить: нота `not_evaluable`
    evaluable_total = evaluable_full = 0
    not_eval_by_want: Counter[str] = Counter()
    note_codes: Counter[str] = Counter()
    pool_kind: Counter[str] = Counter()
    by_want: dict[str, Counter[str]] = {want: Counter() for want in WANTS}
    item_sources: Counter[str] = Counter()
    answer_sources: Counter[str] = Counter()
    direction_bad: list[tuple[str, str, str, str]] = []
    food_bad: list[tuple[str, str, str]] = []
    anchors_no_sugar = sum(catalog.get(slug).sugar is None for slug in anchors)
    filter_hits: list[tuple[str, str]] = []
    stop_hits: list[tuple[str, str]] = []
    percent_hits = 0
    strings_checked = 0
    distinct: Counter[int] = Counter()
    distinct_nonempty: Counter[int] = Counter()
    shared: dict[str, Counter[int]] = {}
    same_winery = same_group = 0
    softer_sugar = SofterSugar()
    reason_heads: Counter[str] = Counter()
    exists_cache: dict[tuple[str, str], bool] = {}

    for slug in anchors:
        anchor = catalog.get(slug)
        facts = Facts.of_wine(anchor, catalog.alcohol(slug))
        for food in FOODS:
            triples = []
            for want in WANTS:
                total += 1
                started = time.perf_counter()
                try:
                    body = after.shelf(slug, food=food, want=want, order="reco")
                except Exception:  # noqa: BLE001 — зонд считает исключения
                    errors.append((slug, food, want, traceback.format_exc(limit=3)))
                    continue
                times_ms.append((time.perf_counter() - started) * 1000)
                items = body["items"]
                codes = [note["code"] for note in body["notes"]]
                note_codes.update(codes)
                pool_kind[body["pool"]] += 1
                by_want[want]["total"] += 1
                evaluable = after.sommelier.evaluable(want, anchor)
                if evaluable:
                    evaluable_total += 1
                    by_want[want]["evaluable"] += 1
                else:
                    not_eval_by_want[want] += 1
                if len(items) >= 3:
                    full += 1
                    by_want[want]["ge3"] += 1
                    evaluable_full += evaluable
                else:
                    short_total += 1
                    if {"fewer_than_three", "no_difference", "not_evaluable"} & set(codes):
                        short_honest += 1
                key = (slug, want)
                if key not in exists_cache:
                    exists_cache[key] = any(
                        not excluded(wine, facts)
                        and after.sommelier.move(want, anchor, wine) is not None
                        for wine in catalog.pool
                    )
                for code in codes:
                    by_want[want][f"note:{code}"] += 1
                    by_want[want][f"note:{code}:{food}"] += 1
                if not evaluable:
                    not_eval_expected += 1
                    not_eval_honest += codes == ["not_evaluable"] and not items
                elif not exists_cache[key]:
                    no_diff_expected += 1
                    by_want[want][f"no_difference_sugar:{anchor.sugar}"] += 1
                    no_diff_honest += "no_difference" in codes and not items
                elif "no_difference" in codes or "not_evaluable" in codes:
                    no_diff_false += 1
                sources = [item["want_source"] for item in items]
                item_sources.update(str(source) for source in sources)
                if items:
                    if all(source == "catalog" for source in sources):
                        answer_sources["all_catalog"] += 1
                        by_want[want]["all_catalog"] += 1
                    elif any(source == "catalog" for source in sources):
                        answer_sources["mixed"] += 1
                    else:
                        answer_sources["grape_only"] += 1
                for item in items:
                    wine = catalog.get(item["slug"])
                    same_winery += wine.winery_norm == anchor.winery_norm
                    same_group += wine.wine_id == anchor.wine_id
                    if not direction_holds(after, anchor, wine, want, item["want_source"]):
                        direction_bad.append((slug, food, want, item["slug"]))
                    if food != "none" and not food_holds(after, item["slug"], food):
                        food_bad.append((slug, food, item["slug"]))
                    if want == "softer":
                        softer_sugar.add(anchor, wine, food, item["want_source"])
                    for reason in item["reasons"]:
                        reason_heads[reason.split(":")[0].split(" — ")[0]] += 1
                dumped = json.dumps(body, ensure_ascii=False)
                percent_hits += "%" in dumped
                for text in strings(body):
                    strings_checked += 1
                    if not check(text).clean:
                        filter_hits.append((slug, text))
                    if STOP.search(text):
                        stop_hits.append((slug, text))
                triples.append(tuple(sorted(item["slug"] for item in items)))
            distinct[len(set(triples))] += 1
            distinct_nonempty[len({t for t in triples if t})] += 1
            # Сколько вин общих у двух направлений одного якоря и блюда (обе выдачи непустые).
            for i, j in ((0, 1), (0, 2), (1, 2)):
                if len(triples) == 3 and triples[i] and triples[j]:
                    pair = f"{WANTS[i]}/{WANTS[j]}"
                    shared.setdefault(pair, Counter())[len(set(triples[i]) & set(triples[j]))] += 1

    # Обычная сортировка — те же якоря и блюда, без направления.
    plain_ms: list[float] = []
    plain_bad = 0
    for slug in anchors:
        for food in FOODS:
            started = time.perf_counter()
            body = after.shelf(slug, food=food, want="none", order="plain")
            plain_ms.append((time.perf_counter() - started) * 1000)
            names = [item["name"].casefold() for item in body["items"]]
            plain_bad += (
                body["notice"] is not None
                or names != sorted(names)
                or any(item["reasons"] or item["want_source"] for item in body["items"])
                or (len(body["items"]) < 3 and not body["notes"])
            )

    times = sorted(times_ms)
    p95 = times[int(0.95 * (len(times) - 1))] if times else float("inf")
    n_items = sum(item_sources.values())
    answered = sum(answer_sources.values())
    soft_sweet = shared.get("softer/sweeter", Counter())
    soft_sweet_pairs = sum(soft_sweet.values())
    result = {
        "anchors": len(anchors),
        "anchors_without_sugar": anchors_no_sugar,
        "somm": after.somm.stats()["pairs.json"],
        "catalog": catalog.stats(),
        "combos": total,
        "exceptions": len(errors),
        "ms": {
            "p50": round(statistics.median(times), 3),
            "p95": round(p95, 3),
            "max": round(times[-1], 3),
            "need_p95": 50,
        },
        "ge3": {
            "n": evaluable_full,
            "of_evaluable": evaluable_total,
            "share": round(evaluable_full / max(1, evaluable_total), 4),
            "need": 0.9,
            "all_combos": {"n": full, "of": total, "share": round(full / total, 4)},
        },
        "not_evaluable": {
            "combos": not_eval_expected,
            "by_want": dict(sorted(not_eval_by_want.items())),
            "honest_note": not_eval_honest,
        },
        "ge3_by_want": {
            want: round(c["ge3"] / c["total"], 4) for want, c in by_want.items() if c["total"]
        },
        "ge3_by_want_evaluable": {
            want: round(c["ge3"] / c["evaluable"], 4)
            for want, c in by_want.items()
            if c["evaluable"]
        },
        "short": {"n": short_total, "with_honest_note": short_honest},
        "notes_by_want": {
            want: {k: v for k, v in sorted(c.items()) if ":" in k} for want, c in by_want.items()
        },
        "no_difference": {
            "expected": no_diff_expected,
            "honest": no_diff_honest,
            "false_note": no_diff_false,
        },
        "notes": dict(note_codes.most_common()),
        "pool": dict(pool_kind),
        "same_winery_or_group_in_items": same_winery + same_group,
        "direction_check_failed": len(direction_bad),
        "food_check_failed": len(food_bad),
        "softer_sugar": softer_sugar.summary(),
        "content_filter_violations": len(filter_hits),
        "stop_words": len(stop_hits),
        "percent_in_body": percent_hits,
        "strings_checked": strings_checked,
        "as_is": {
            "distinct_triples_per_anchor_food": dict(sorted(distinct.items())),
            "distinct_nonempty_triples_per_anchor_food": dict(sorted(distinct_nonempty.items())),
            "shared_wines_between_directions": {
                pair: dict(sorted(counts.items())) for pair, counts in shared.items()
            },
            "softer_sweeter_share_2_or_3": {
                "n": soft_sweet[2] + soft_sweet[3],
                "of": soft_sweet_pairs,
                "share": round((soft_sweet[2] + soft_sweet[3]) / max(1, soft_sweet_pairs), 4),
            },
            "mean_distinct": round(
                sum(k * v for k, v in distinct.items()) / max(1, sum(distinct.values())), 3
            ),
            "item_want_source": {k: round(v / n_items, 4) for k, v in item_sources.items()},
            "answers_all_catalog": round(answer_sources["all_catalog"] / max(1, answered), 4),
            "answers": dict(answer_sources),
            "all_catalog_by_want": {
                want: round(c["all_catalog"] / max(1, c["total"]), 4) for want, c in by_want.items()
            },
        },
        "plain": {
            "combos": len(plain_ms),
            "bad": plain_bad,
            "p95_ms": round(sorted(plain_ms)[int(0.95 * (len(plain_ms) - 1))], 3),
        },
        "reason_heads": dict(reason_heads.most_common(20)),
    }
    verdict = {
        "no_exceptions": not errors,
        "p95_under_50ms": p95 < 50,
        "ge3_share_90": evaluable_full / max(1, evaluable_total) >= 0.9,
        "honest_when_short": short_honest == short_total,
        "honest_when_no_difference": no_diff_honest == no_diff_expected and no_diff_false == 0,
        "honest_when_not_evaluable": not_eval_honest == not_eval_expected,
        "food_from_pairing_rules": not food_bad,
        "content_filter": not filter_hits and not stop_hits,
        "no_percent": percent_hits == 0,
        "softer_not_above_ceiling": not softer_sugar.above and not softer_sugar.drier,
    }
    if args.sweep:
        result["sweep"] = sweep(after)
        verdict["sweep_clean"] = all(result["sweep"]["pass"].values())
    result["pass"] = verdict
    examples = {
        f"{slug} {food} {want}": after.shelf(slug, food=food, want=want, order="reco")
        for slug, food, want in EXAMPLES
    }
    Path(args.out).write_text(
        json.dumps(
            {
                **result,
                "errors": errors[:10],
                "direction_bad": direction_bad[:20],
                "food_bad": food_bad[:20],
                "filter_hits": filter_hits[:20],
                "examples": examples,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=1))
    for name, body in examples.items():
        print(f"\n{name}: pool={body['pool']} notes={[n['text'] for n in body['notes']]}")
        for item in body["items"]:
            print(f"  {item['winery']} — {item['name']} [{item['want_source']}]:"
                  f" {' · '.join(item['reasons'])}")
    print("\nприёмка:", "пройдена" if all(verdict.values()) else "НЕ пройдена", verdict)
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
