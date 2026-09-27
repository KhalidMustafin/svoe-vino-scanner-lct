"""Калибровка подсказки «похоже, этой позиции нет в каталоге» (план, Д2, п. 6) — только CPU.

Ответы сервиса берутся из записанных прогонов поля: каталог — 353 кадра, вне каталога — 409.
Для каждого кадра собирается тот же `ScanResult`, что отдал бы сервис (top-5 после H5 из
`whatif.json`, `evidence.abstain` пересчитан `abstain_check` из записанного чтения и CV), и
считается тем же кодом, что у `/v1/scan`: `after_block`, `AfterSearch.not_found_conditions`,
`not_found_reasons`. Меняется только порог счёта CV.

Правило (план, Д2, п. 6):
- `check` — p_top1 < 0,5 или ambiguous;
- `not_found` — винодельня прочитана однозначно, названия у неё не условные, каждая её позиция
  спорит с этикеткой по сахару, цвету или сортам (`winery_no_match`), или винодельня названа
  словом хозяйства и её в каталоге нет (`winery_not_in_catalog`); и счёт CV лучшей серии
  (zmax) ниже порога.

Порог — наибольший, при котором ложных `not_found` на кадрах каталога не больше 7 (≤ 2 %).
Полнота на кадрах вне каталога пишется раздельно: винодельня кадра есть в каталоге, её нет,
человек её не назвал (разметка `labels_v2.jsonl`, поле `reading.winery`). Полнота ниже 50 % —
автоэкрана нет (`NOT_FOUND_VISUAL_MAX = None`), остаются `check` и кнопка.

Подсказка Д3 `after.suggest_not_found` (договор, §2) — одно правило «счёт CV лучшей серии ниже
порога» (`after_layer.visual_low`), без винодельни; экрана `not_found` она не открывает, а
`found` делает `check`. Порог — наибольший, при котором флаг стоит не больше чем на 7 из 353
кадров каталога. Кроме полноты пишется цена для человека (сколько кадров каталога уходят из
`found` в `check` и сколько из них сервис угадал), самопроверка тем же `after_block` и оценка
вне выборки: 5 фолдов по винам (группа `wine_id` разметки), 20 перемешиваний.

Прогон можно повторить на других ответах (дорожка h6 меняет ответы):

    .venv/Scripts/python.exe research/2026-09-24_after/calibrate_hint.py \\
        [--catalog-run DIR] [--ooc-run DIR] [--data-dir DIR] [--labels FILE] [--out FILE]

`DIR` — папка итерации прогона с `predictions.jsonl` (и `whatif.json`, если есть).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.api.after_layer import (  # noqa: E402
    NOT_FOUND_VISUAL_MAX,
    SUGGEST_NOT_FOUND_VISUAL_MAX,
    VISUAL_LOW,
    AfterSearch,
    after_block,
    not_found_reasons,
    read_of_evidence,
    scan_state,
    visual_low,
)
from app.api.cards import CatalogCards, read_records  # noqa: E402
from app.api.config import ServiceSettings  # noqa: E402
from app.api.service import Confidence, ScanResult, TopItem  # noqa: E402
from app.features.contracts import VisualResult  # noqa: E402
from app.reading.contracts import LabelFields  # noqa: E402
from app.resolve.attrs import CatalogAttrs, norm_key  # noqa: E402
from app.resolve.rerank import DEFAULT_CONFIG, abstain_check  # noqa: E402

RUNS = REPO.parent / "svoe-vino-scanner" / "runs" / "field25" / "iters" / "runs"
FIELD = REPO.parent / "field_dataset"
#: Ложных `not_found` на 353 кадрах каталога — не больше 2 % (план, §4).
MAX_FALSE = 7
#: Та же доля для оценки вне выборки: на четырёх фолдах из пяти — не больше 2 % их кадров.
MAX_FALSE_SHARE = 0.02
#: Оценка вне выборки: фолды по винам и число перемешиваний.
FOLDS = 5
SHUFFLES = 20
#: Полнота, ниже которой автоэкрана нет (план, Д2, п. 6).
MIN_RECALL = 0.5
#: Порог «проверьте» сервиса (`service.AMBIGUOUS_P_TOP1`).
AMBIGUOUS_P_TOP1 = 0.5
#: Порядок H5 в `whatif.json`: при p_top1 < 0,5 — ключ `no_cluster+filter`.
H5_KEY = "no_cluster+filter"

# ------------------------------------------------------------------ правда о винодельне
#: Правки автоматического сопоставления «винодельня по разметке → каталог»: транслитерация,
#: которую не ловит норма, и «Кубань»/«Гай-Кодзор», которые в тексте разметки — регион, а не
#: винодельня. Ключ — подстрока `reading.winery`, значение — написание винодельни каталога
#: или `None`, если её в каталоге нет.
TRUTH_FIXES: dict[str, str | None] = {
    "Мильстрим": "MILLSTREAM",
    "Шато де Талю": "Chateau de Talu",
    "Семейная винодельня Литавщуков": "Литавщук. Litavshchuk vineyards & winery",
    "Strapi lists it under Раевское": "Раевское",  # линейка GLOW
    "Chateau Alvisa": "Шато АЛВИСА",
    "бренд винодельни Шумринка": None,  # Шумринки в каталоге нет, «Гай-Кодзор» — село
    "Шумринка (Shumrinka)": None,
    "производитель линейки — винодельня «Юбилейная»": None,
}
#: Человек не назвал ни винодельню, ни бренд.
NOT_NAMED = re.compile(
    r"^\s*$|^(not read|unknown|не читается|не указан[аы]? на (видимой|лицевой))", re.IGNORECASE
)
_SPLIT = re.compile(r"[()\[\];,/«»—–]|\s-\s|\bбренд\w*\b|\bbrand\b|\bлинейк\w*|\bТМ\b", re.I)
_REGIONS = frozenset({"кубань", "крым", "анапа", "россия", "грузия", "италия"})


def winery_truth(text: str, search: AfterSearch) -> tuple[str, set[str]]:
    """`yes` / `no` / `unknown` и ключи виноделен каталога, которые назвал человек."""
    for part, name in TRUTH_FIXES.items():
        if part in text:
            key = search.specific_key(name) if name else None
            return ("yes", {key}) if key else ("no", set())
    if NOT_NAMED.search(text) and "бренд" not in text and "brand" not in text:
        return "unknown", set()
    keys = set()
    for part in [text, *(p.strip(" .:\"'") for p in _SPLIT.split(text))]:
        if part and norm_key(part) not in _REGIONS and (key := search.specific_key(part)):
            keys.add(key)
    return ("yes" if keys else "no"), keys


# ------------------------------------------------------------------ записанные ответы
def load_rows(run: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (run / "predictions.jsonl").open(encoding="utf-8")]
    failed = run / "predictions_vlmfail.jsonl"  # ответ сервиса при сбое чтения
    if failed.is_file():
        rows += [json.loads(line) for line in failed.open(encoding="utf-8") if line.strip()]
    return rows


def load_whatif(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "whatif.json"
    if not path.is_file():
        return {}
    return {item["query_id"]: item for item in json.loads(path.read_text(encoding="utf-8"))}


def scan_result(row: Mapping[str, Any], whatif: Mapping[str, Any], attrs: CatalogAttrs, top_k: int):
    """Ответ сервиса по записи прогона: top-5 после H5, `evidence` — как пишет сервис.

    H5 (`rerank_ambiguous` сервиса) — без бонуса соседу и без спорящих по сахару и цвету: в
    `whatif.json` это ключ `no_cluster+filter`, а не `filter` (там бонус соседу остаётся).
    """
    ranked = [slug for slug, _ in row.get("resolve") or []]
    probs = {slug: float(p) for slug, p in row.get("resolve") or []}
    p1 = row.get("p_top1")
    h5 = whatif.get(H5_KEY)
    if p1 is not None and p1 < AMBIGUOUS_P_TOP1 and h5:
        ranked = list(h5) + [s for s in ranked if s not in h5]
    top5 = ranked[:5]
    visual = VisualResult.model_validate(row["visual"]) if row.get("visual") else None
    fields_raw = (row.get("text_read") or {}).get("fields")
    fields = LabelFields.model_validate(fields_raw) if fields_raw else None
    evidence: dict[str, Any] = {"vlm": row.get("vlm") or {}}
    if visual is not None:
        cfg = DEFAULT_CONFIG.model_copy(update={"abstain": "off", "top_k": top_k})
        evidence["abstain"] = abstain_check(visual, fields, attrs, cfg=cfg)
        evidence["cv"] = {
            "top5": [{"slug": c.slug, "score": c.score} for c in visual.candidates[:5]]
        }
    slug = top5[0] if top5 else row.get("slug")
    return ScanResult(
        slug=slug,
        confidence=Confidence(top1=probs.get(slug, p1) if p1 is not None else None),
        top5=[TopItem(slug=s, score=probs.get(s, 0.0)) for s in top5],
        outcome=row.get("outcome") or "error",
        error=row.get("error"),
        evidence=evidence,
    )


def frames(
    run: Path, search: AfterSearch, labels: Mapping[str, Any], top_k: int, *, in_catalog: bool
) -> list[dict[str, Any]]:
    whatif = load_whatif(run)
    out = []
    # Экран и флаг — правилами ниже, при любом пороге; `after` здесь — экраны Д1.
    search.not_found_max = None
    search.suggest_max = None
    for row in load_rows(run):
        qid = row["query_id"]
        result = scan_result(row, whatif.get(qid) or {}, search.attrs, top_k)
        block = after_block(result, search)
        label = read_of_evidence(result.evidence)
        prefer = [key for item in result.top5 if (key := search.winery_key(item.slug))]
        match = search.resolve_winery(
            label.wineries, prefer=dict.fromkeys(prefer), house=label.house
        )
        conditions = search.not_found_conditions(result, label, match)
        mark = labels.get(qid.split("-")[0]) or {}
        human = str((mark.get("reading") or {}).get("winery") or "")
        gt = mark.get("gt_slug") or ""
        if in_catalog:
            truth = {search.winery_key(gt)} if gt else set()
            group = "yes"
        else:
            group, truth = winery_truth(human, search)
        state, _ = scan_state(result)
        # «Верно» — как у организатора: ответ в группе того же вина (gt ∪ acceptable v2).
        correct = (
            in_catalog
            and result.slug is not None
            and (result.slug == gt or result.slug in set(mark.get("acceptable") or []))
        )
        out.append(
            {
                "qid": qid,
                "result": result,
                "correct": correct,
                "wine": (mark.get("v2") or {}).get("wine_id") or gt or qid,
                "state": state,
                "after": block,
                "match_keys": set(match.keys),
                "in_catalog": match.in_catalog,
                "conditions": conditions,
                "winery_confident": bool((result.evidence.get("abstain") or {}).get(
                    "winery_confident"
                )),
                "abstain": result.evidence.get("abstain") or {},
                "truth_keys": truth,
                "group": group,
                "human": human,
            }
        )
    return out


# ------------------------------------------------------------------ правила и порог
Rule = Callable[[dict[str, Any], float], bool]


def main_rule(frame: dict[str, Any], visual_max: float) -> bool:
    """Правило сервиса: `not_found_reasons` с порогом."""
    return frame["state"] != "error" and bool(not_found_reasons(frame["conditions"], visual_max))


def rule_catalog_only(frame: dict[str, Any], visual_max: float) -> bool:
    """Только ветка «винодельня каталога, ни одна позиция не сходится»."""
    return main_rule(frame, visual_max) and frame["conditions"]["winery"] == "catalog"


def rule_check_only(frame: dict[str, Any], visual_max: float) -> bool:
    """Правило сервиса, но только поверх экрана `check`."""
    return frame["state"] == "check" and main_rule(frame, visual_max)


def rule_confident(frame: dict[str, Any], visual_max: float) -> bool:
    """Правило сервиса плюс `abstain.winery_confident` (уверенность чтения ≥ 0,75)."""
    if frame["conditions"]["winery"] == "catalog" and not frame["winery_confident"]:
        return False
    return main_rule(frame, visual_max)


def rule_s10(frame: dict[str, Any], visual_max: float) -> bool:
    """Для сравнения: условия S10 из `evidence.abstain` (совпадение текста, а не спор фактов)."""
    ab = frame["abstain"]
    return (
        bool(ab.get("winery_confident"))
        and bool(ab.get("no_position_matches"))
        and not ab.get("conditional_names")
        and (frame["conditions"]["visual"] or math.inf) < visual_max
    )


def rule_visual(frame: dict[str, Any], visual_max: float) -> bool:
    """Один счёт CV, без винодельни: правило подсказки `suggest_not_found` (`visual_low`)."""
    return frame["state"] != "error" and visual_low(frame["conditions"]["visual"], visual_max)


RULES: dict[str, tuple[Rule, str]] = {
    "main": (main_rule, "правило плана: обе ветки, поверх любого экрана"),
    "catalog_branch": (rule_catalog_only, "только «винодельня каталога, позиции не сходятся»"),
    "check_only": (rule_check_only, "правило плана, только поверх `check`"),
    "winery_confident": (rule_confident, "правило плана + уверенность чтения винодельни ≥ 0,75"),
    "s10_text": (rule_s10, "для сравнения: условия S10 (текст не совпал) + порог"),
    "visual_only": (rule_visual, "только счёт CV, без винодельни (`suggest_not_found`)"),
}


def pick_threshold(catalog: list[dict[str, Any]], rule: Rule, max_false: int = MAX_FALSE) -> float:
    """Наибольший порог, при котором ложных на кадрах каталога не больше `max_false`.

    Срабатывание строгое (`счёт < порога`), поэтому порог — (max_false+1)-й снизу счёт среди
    кадров каталога, где правило сработало бы без порога; таких меньше — порога не нужно.
    """
    scores = sorted(
        f["conditions"]["visual"]
        for f in catalog
        if f["conditions"]["visual"] is not None and rule(f, math.inf)
    )
    return scores[max_false] if len(scores) > max_false else math.inf


def evaluate(
    catalog: list[dict[str, Any]], ooc: list[dict[str, Any]], rule: Rule, visual_max: float
) -> dict[str, Any]:
    fired_cat = [f["qid"] for f in catalog if rule(f, visual_max)]
    groups: dict[str, Counter] = {}
    for f in ooc:
        c = groups.setdefault(f["group"], Counter())
        c["frames"] += 1
        c["fired"] += rule(f, visual_max)
    total = sum(c["frames"] for c in groups.values())
    fired = sum(c["fired"] for c in groups.values())
    return {
        "visual_max": None if math.isinf(visual_max) else round(visual_max, 4),
        "false_on_catalog": len(fired_cat),
        "false_examples": fired_cat[:10],
        "recall": round(fired / total, 4) if total else None,
        "fired_ooc": fired,
        "ooc_frames": total,
        "by_winery_group": {
            g: {
                "frames": c["frames"],
                "fired": c["fired"],
                "recall": round(c["fired"] / c["frames"], 4) if c["frames"] else None,
            }
            for g, c in sorted(groups.items())
        },
    }


# ------------------------------------------------------------------ винодельня: правки (a), (b)
def winery_stats(items: Iterable[dict[str, Any]], *, in_catalog: bool) -> dict[str, int]:
    """Что `after` говорит о винодельне кадра против разметки."""
    out: Counter = Counter()
    for f in items:
        state = f["in_catalog"]
        if state is None:
            out["not_read_or_ambiguous"] += 1
        elif state is False:
            wrong = in_catalog or f["group"] == "yes"
            out["not_in_catalog_" + ("WRONG" if wrong else "ok")] += 1
        elif f["group"] != "yes":
            out[f"catalog_but_truth_{f['group']}"] += 1
        else:
            out["catalog_correct" if f["match_keys"] & f["truth_keys"] else "catalog_other"] += 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ подсказка suggest_not_found
def screen_shift(items: Iterable[dict[str, Any]], visual_max: float) -> dict[str, int]:
    """Что флаг делает с экранами: `found` → `check`, и сколько из них сервис угадал."""
    out: Counter = Counter()
    for f in items:
        if not rule_visual(f, visual_max):
            continue
        out["flagged"] += 1
        if f["state"] == "found":
            out["found_to_check"] += 1
            out["found_to_check_correct"] += f["correct"]
        else:
            out[f"reason_added_to_{f['state']}"] += 1
        out["flagged_correct"] += f["correct"]
    return dict(sorted(out.items()))


def service_agrees(items: Iterable[dict[str, Any]], search: AfterSearch, visual_max: float) -> int:
    """Самопроверка: `after_block` сервиса с этим порогом даёт тот же флаг, экран и причину.

    Возвращает число расхождений (должно быть 0). Порог сервиса на время проверки меняется
    и возвращается обратно.
    """
    saved = search.suggest_max
    search.suggest_max = None if math.isinf(visual_max) else visual_max
    bad = 0
    try:
        for f in items:
            block = after_block(f["result"], search)
            flag = rule_visual(f, visual_max)
            state = "check" if flag and f["state"] == "found" else f["state"]
            reasons_ok = (VISUAL_LOW in block["reasons"]) == flag
            bad += block["suggest_not_found"] != flag or block["state"] != state or not reasons_ok
    finally:
        search.suggest_max = saved
    return bad


def out_of_fold(
    catalog: list[dict[str, Any]], ooc: list[dict[str, Any]], rule: Rule
) -> dict[str, Any]:
    """Оценка вне выборки: порог подбирается на 4 фолдах по винам и проверяется на пятом.

    На фолдах обучения — не больше `MAX_FALSE_SHARE` ложных от их кадров. Ложные складываются
    по пяти фолдам (каждый кадр каталога — вне обучения ровно раз); полнота — среднее по пяти
    порогам на всех кадрах вне каталога (в подбор они не входят). Разброс — по перемешиваниям.
    """
    wines = sorted({f["wine"] for f in catalog})
    false_counts: list[int] = []
    recalls: list[float] = []
    thresholds: list[float] = []
    for seed in range(SHUFFLES):
        order = list(wines)
        random.Random(seed).shuffle(order)
        fold_of = {wine: i % FOLDS for i, wine in enumerate(order)}
        false = 0
        fold_recalls = []
        for k in range(FOLDS):
            train = [f for f in catalog if fold_of[f["wine"]] != k]
            test = [f for f in catalog if fold_of[f["wine"]] == k]
            limit = pick_threshold(train, rule, int(MAX_FALSE_SHARE * len(train)))
            thresholds.append(limit)
            false += sum(rule(f, limit) for f in test)
            fold_recalls.append(sum(rule(f, limit) for f in ooc) / len(ooc))
        false_counts.append(false)
        recalls.append(statistics.mean(fold_recalls))
    finite = [t for t in thresholds if not math.isinf(t)]
    return {
        "wines": len(wines),
        "folds": FOLDS,
        "shuffles": SHUFFLES,
        "false_median": statistics.median(false_counts),
        "false_min": min(false_counts),
        "false_max": max(false_counts),
        "false_share_median": round(statistics.median(false_counts) / len(catalog), 4),
        "recall_median": round(statistics.median(recalls), 4),
        "recall_min": round(min(recalls), 4),
        "recall_max": round(max(recalls), 4),
        "threshold_min": round(min(finite), 4) if finite else None,
        "threshold_max": round(max(finite), 4) if finite else None,
    }


def suggest_report(
    catalog: list[dict[str, Any]], ooc: list[dict[str, Any]], search: AfterSearch
) -> dict[str, Any]:
    """Порог `SUGGEST_NOT_FOUND_VISUAL_MAX` по правилу, цена для человека и оценка вне выборки."""
    threshold = pick_threshold(catalog, rule_visual)
    picked = evaluate(catalog, ooc, rule_visual, threshold)
    service = SUGGEST_NOT_FOUND_VISUAL_MAX
    at_service = evaluate(catalog, ooc, rule_visual, service) if service is not None else None
    return {
        "rule": "suggest_not_found = счёт CV лучшей серии (zmax) < порога; "
        "порог — наибольший, при котором флаг не больше чем на "
        f"{MAX_FALSE} из {len(catalog)} кадров каталога",
        "picked": picked,
        "service_visual_max": service,
        "service_matches_picked": service is not None
        and not math.isinf(threshold)
        and round(threshold, 4) == service,
        "at_service": at_service,
        "catalog_screens": screen_shift(catalog, threshold),
        "ooc_screens": screen_shift(ooc, threshold),
        "catalog_correct_total": sum(f["correct"] for f in catalog),
        "catalog_found_total": sum(f["state"] == "found" for f in catalog),
        "ooc_found_total": sum(f["state"] == "found" for f in ooc),
        "service_disagreements": service_agrees([*catalog, *ooc], search, threshold),
        "out_of_fold": out_of_fold(catalog, ooc, rule_visual),
    }


def print_suggest(report: dict[str, Any], n_catalog: int, n_ooc: int) -> None:
    picked = report["picked"]
    cat, ooc, oof = report["catalog_screens"], report["ooc_screens"], report["out_of_fold"]
    print()
    print("подсказка suggest_not_found:", report["rule"])
    print(
        f"  порог {picked['visual_max']}, в сервисе {report['service_visual_max']} "
        f"({'совпадает' if report['service_matches_picked'] else 'НЕ совпадает'})"
    )
    print(
        f"  флаг на каталоге: {picked['false_on_catalog']} из {n_catalog} "
        f"({', '.join(picked['false_examples'])})"
    )
    groups = picked["by_winery_group"]
    print(
        f"  полнота вне каталога: {picked['fired_ooc']}/{n_ooc} ({100 * picked['recall']:.1f} %); "
        + ", ".join(
            f"{name} {g['fired']}/{g['frames']} ({100 * (g['recall'] or 0):.1f} %)"
            for name, g in groups.items()
        )
    )
    print(
        f"  каталог: found → check {cat.get('found_to_check', 0)} из "
        f"{report['catalog_found_total']} found, из них сервис угадал "
        f"{cat.get('found_to_check_correct', 0)}; причина добавлена к check "
        f"{cat.get('reason_added_to_check', 0)}; всего с флагом угадано "
        f"{cat.get('flagged_correct', 0)} из {cat.get('flagged', 0)}"
    )
    print(
        f"  вне каталога: found → check {ooc.get('found_to_check', 0)} из "
        f"{report['ooc_found_total']} found (все ответы там неверны)"
    )
    print(f"  самопроверка after_block: расхождений {report['service_disagreements']}")
    print(
        f"  вне выборки ({oof['folds']} фолдов по {oof['wines']} винам, {oof['shuffles']} "
        f"перемешиваний): ложных {oof['false_median']} (от {oof['false_min']} до "
        f"{oof['false_max']}) из {n_catalog}, полнота {100 * oof['recall_median']:.1f} % "
        f"(от {100 * oof['recall_min']:.1f} до {100 * oof['recall_max']:.1f}), "
        f"порог от {oof['threshold_min']} до {oof['threshold_max']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog-run", type=Path, default=RUNS / "pfix_final" / "iter20")
    parser.add_argument("--ooc-run", type=Path, default=RUNS / "ooc_v2prod" / "iter20")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=FIELD / "labels_v2.jsonl")
    parser.add_argument("--out", type=Path, default=HERE / "calibrate_hint.json")
    args = parser.parse_args()

    env = dict(os.environ)
    if args.data_dir is not None:
        env["SVS_DATA_DIR"] = str(args.data_dir)
    elif "SVS_DATA_DIR" not in env and not (REPO / "data" / "gt" / "gt_tokens.jsonl").is_file():
        env["SVS_DATA_DIR"] = str(REPO.parent / "svoe-vino-scanner" / "data")
    settings = ServiceSettings.from_env(env)
    records = read_records(settings.attrs_path)
    attrs = CatalogAttrs.from_records(records)
    search = AfterSearch.load(settings, CatalogCards.build(records), attrs)
    labels = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["id"]] = row

    catalog = frames(args.catalog_run, search, labels, settings.top_k, in_catalog=True)
    ooc = frames(args.ooc_run, search, labels, settings.top_k, in_catalog=False)
    out: dict[str, Any] = {
        "catalog_run": args.catalog_run.as_posix().split("/runs/")[-1],
        "ooc_run": args.ooc_run.as_posix().split("/runs/")[-1],
        "frames": {"catalog": len(catalog), "ooc": len(ooc)},
        "ooc_winery_groups": dict(Counter(f["group"] for f in ooc)),
        "states_d1": {
            "catalog": dict(Counter(f["after"]["state"] for f in catalog)),
            "ooc": dict(Counter(f["after"]["state"] for f in ooc)),
        },
        "winery": {
            "catalog": winery_stats(catalog, in_catalog=True),
            "ooc": winery_stats(ooc, in_catalog=False),
        },
        "service_visual_max": NOT_FOUND_VISUAL_MAX,
        "rules": {},
    }
    # Сколько кадров проходит каждое условие без порога — где теряется полнота.
    funnel: dict[str, Counter] = {}
    for name, items in (("catalog", catalog), ("ooc", ooc)):
        c = funnel.setdefault(name, Counter())
        for f in items:
            cond = f["conditions"]
            c[f"winery={cond['winery']}"] += 1
            if cond["winery"] == "catalog":
                c["catalog: no_position_agrees"] += cond["no_position_agrees"]
                c["catalog: no_position_agrees & !conditional"] += (
                    cond["no_position_agrees"] and not cond["conditional_names"]
                )
    out["funnel"] = {k: dict(sorted(v.items())) for k, v in funnel.items()}

    print(f"кадров: каталог {len(catalog)}, вне каталога {len(ooc)}")
    print("винодельня кадров вне каталога по разметке:", out["ooc_winery_groups"])
    print("экраны Д1:", json.dumps(out["states_d1"], ensure_ascii=False))
    print("винодельня в after:", json.dumps(out["winery"], ensure_ascii=False))
    print("воронка условий без порога:", json.dumps(out["funnel"], ensure_ascii=False))
    print()
    print(
        f"| правило | порог zmax | ложных из {len(catalog)} | полнота на {len(ooc)} "
        "| есть в каталоге | нет | не названа |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, (rule, title) in RULES.items():
        threshold = pick_threshold(catalog, rule)
        picked = evaluate(catalog, ooc, rule, threshold)
        ceiling = evaluate(catalog, ooc, rule, math.inf)
        out["rules"][name] = {"title": title, "picked": picked, "no_threshold": ceiling}
        groups = picked["by_winery_group"]

        def cell(group: str, groups: dict[str, Any] = groups) -> str:
            g = groups.get(group) or {"fired": 0, "frames": 0, "recall": 0}
            return f"{g['fired']}/{g['frames']} ({100 * (g['recall'] or 0):.1f} %)"

        limit = "нет" if picked["visual_max"] is None else f"{picked['visual_max']:.4f}"
        print(
            f"| {title} | {limit} | {picked['false_on_catalog']} | "
            f"{picked['fired_ooc']}/{picked['ooc_frames']} ({100 * picked['recall']:.1f} %) | "
            f"{cell('yes')} | {cell('no')} | {cell('unknown')} |"
        )
    main_picked = out["rules"]["main"]["picked"]
    verdict = main_picked["false_on_catalog"] <= MAX_FALSE and main_picked["recall"] >= MIN_RECALL
    out["verdict"] = {
        "auto_screen": verdict,
        "reason": (
            f"полнота {100 * main_picked['recall']:.1f} % при {main_picked['false_on_catalog']} "
            f"ложных из {len(catalog)}: "
            + ("автоэкран включается" if verdict else "ниже 50 % — автоэкрана нет, check + кнопка")
        ),
    }
    print()
    print("решение:", out["verdict"]["reason"])
    out["suggest"] = suggest_report(catalog, ooc, search)
    print_suggest(out["suggest"], len(catalog), len(ooc))
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записано:", args.out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
