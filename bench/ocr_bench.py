"""Стенд замеров OCR: читатели, кроп и словарь фиксированы, кадры строго по очереди.

    python -m bench.ocr_bench --dataset public --reader vlm:qwen3-vl:4b-instruct \\
        --crop full --crop-px 1024 --out runs/vlm4b-full-1024/
    python -m bench.ocr_bench --dataset synth --reader vlm --reader easyocr \\
        --crop label --out runs/vlm-easyocr-label/
    python -m bench.ocr_bench --dataset synth --dry-run

Кадр идёт тем же путём, что в сервисе: `decode_image` → рамка цели → `read_label`
(кроп, читатели по очереди, токены, словарь, поля). Параметры здесь не подбираются:
прогон фиксирует читателей, кроп и размер и пишет `predictions.jsonl` и `metrics.json`.
Чтения идут через кэш, поэтому повторный прогон с теми же параметрами не занимает GPU.

`text_top5` в предсказаниях — только диагностика: slug из попаданий словаря, взвешенные
idf. Вино выбирает resolve поверх кандидатов CV, а не этот список.

Коды выхода: 0 — готово, 1 — набор не прошёл проверку, 2 — ошибка аргументов или
набора, 3 — конвейер, словарь или читатель не готовы.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import numpy as np

from app.config import Settings, get_settings
from app.detect.bottles import TargetSelection
from app.reading.contracts import Box, CropName, LabelFields, LexHit, Reader, Reading
from app.reading.lexicon.build import Lexicon, default_lexicon_path, idf
from app.reading.pipeline import DEFAULT_BUDGET_MS, label_hits, read_label
from app.reading.text.layout import token_segments
from app.reading.warmup import synthetic_label, warm_readers
from bench.datasets import (
    SEALED_METRICS_FILE,
    SEALED_NOTE,
    SEALED_SPLIT,
    SEALED_WARNING,
    BenchItem,
    DatasetError,
    Split,
    check_images,
    load_manifest,
    load_public,
    load_synth,
    seals_test,
    summarize,
    twin_clusters,
)
from bench.metrics import (
    NONE_SLUG,
    dumps,
    expected_label_tokens,
    field_metrics,
    latency_summary,
    load_gt_tokens,
    ooc_reject_rate,
    print_json,
    text_topk,
    token_cer_table,
)

READER_KINDS = ("vlm", "easyocr", "rapidocr")
DATASETS = ("public", "synth", "manifest")
TARGETS = ("annotation", "detector")
TEXT_TOP_K = 5
TEXT_TOP_NOTE = (
    "диагностика: slug из попаданий словаря, взвешенные idf; вино выбирает resolve поверх CV"
)
FILLED_FIELDS = ("winery", "cuvee", "grape", "sugar", "year", "serial", "abv", "color", "unmatched")
#: Оговорки к `field_accuracy`: что метрика не проверяет по построению эталона.
FIELD_ACCURACY_NOTES = {
    "cuvee_or_grape": (
        "часть «сорт» цикличная: эталон и парсер берут одну таблицу GRAPE_SYNONYMS, ошибка "
        "таблицы не видна, пока часть эталона не проверена глазами"
    ),
}

DecodeFn = Callable[[bytes], np.ndarray]
TargetFn = Callable[[np.ndarray, BenchItem], TargetSelection | None]

# Рамка бутылки из разметки (как в synth_annotations.jsonl) — оракул вместо детектора.
TARGET_BOX_KEY = "target_bbox_visible"


class BenchSetupError(RuntimeError):
    """Конвейер чтения не собрался: нет модуля, зависимости или читатель недоступен."""


@dataclass(frozen=True)
class Pipeline:
    decode: DecodeFn
    readers: tuple[Reader, ...]
    lexicon: Lexicon | None = None
    target: TargetFn | None = None  # None — рамки нет, кроп bottle/label/band уйдёт в full


# ------------------------------------------------------------------ сборка конвейера
def parse_reader_spec(spec: str) -> tuple[str, str | None]:
    """`vlm:qwen3-vl:4b-instruct` → ("vlm", "qwen3-vl:4b-instruct"); `easyocr` → ("easyocr", None).

    Голый `vlm` берёт модель из настроек; фактическая версия читателя пишется в metrics.json.
    """
    kind, _, arg = spec.partition(":")
    if kind not in READER_KINDS:
        raise ValueError(f"читатель {spec!r}: ожидается один из {', '.join(READER_KINDS)}")
    return kind, arg or None


def import_attr(module: str, attr: str) -> Any:
    """Ленивый импорт с понятной ошибкой: модуля ещё нет или не хватает зависимости."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module == missing or module.startswith(missing + ".")):
            raise BenchSetupError(
                f"модуль {module} ещё не написан: стенду нужен {module}.{attr}"
            ) from exc
        raise BenchSetupError(f"{module}: не установлена зависимость {missing!r}") from exc
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise BenchSetupError(f"в модуле {module} нет {attr}") from exc


def has_target_box(item: BenchItem) -> bool:
    raw = item.annotations.get(TARGET_BOX_KEY)
    return isinstance(raw, list | tuple) and len(raw) == 4


def annotation_box(item: BenchItem, width: int, height: int) -> Box | None:
    """Рамка бутылки из разметки: [x0, y0, x1, y1] в пикселях декодированного кадра → доли."""
    if not has_target_box(item):
        return None
    x0, y0, x1, y1 = (float(v) for v in item.annotations[TARGET_BOX_KEY])
    fx0, fx1 = (min(1.0, max(0.0, v / width)) for v in (x0, x1))
    fy0, fy1 = (min(1.0, max(0.0, v / height)) for v in (y0, y1))
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return Box(x0=fx0, y0=fy0, x1=fx1, y1=fy1)


def annotation_target(image: np.ndarray, item: BenchItem) -> TargetSelection | None:
    """Цель-оракул: рамка из разметки считается надёжной. Меряет читателя, а не детектор.

    Кроп label и band `read_label` строит из рамки бутылки так же, как в сервисе.
    """
    height, width = image.shape[:2]
    box = annotation_box(item, width, height)
    if box is None:
        return None
    return TargetSelection(target=box, method="detector", score=1.0, confident=True)


def detector_target_fn(settings: Settings) -> TargetFn:
    """Цель от детектора бутылок CV: модель поднимается сразу, иначе стенд не готов."""
    detector = import_attr("app.detect.bottles", "BottleDetector")(settings.device)
    select_target = import_attr("app.detect.bottles", "select_target")
    if not detector.available():
        raise BenchSetupError("детектор бутылок недоступен: нет весов или torch")

    def target(image: np.ndarray, item: BenchItem) -> TargetSelection | None:
        return select_target(detector.detect(image))

    return target


def _cached_reader_class() -> Callable[..., Reader]:
    errors = []
    for module in ("app.reading.readers.cache", "app.reading.readers.base"):
        try:
            return import_attr(module, "CachedReader")
        except BenchSetupError as exc:
            errors.append(str(exc))
    raise BenchSetupError("не найден CachedReader: " + "; ".join(errors))


def _cache_stats(reader: Reader) -> dict[str, int] | None:
    stats = getattr(reader, "stats", None)
    return stats.as_dict() if hasattr(stats, "as_dict") else None


def resolve_pipeline(
    reader_specs: Sequence[str] | str,
    settings: Settings,
    cache_dir: Path,
    *,
    lexicon: Lexicon | None = None,
    target: str = "annotation",
    crop: CropName = "full",
) -> Pipeline:
    """Модули конвейера импортируются лениво: --dry-run работает и без них.

    Точки входа: `app.normalize.decode.decode_image(bytes)`,
    `app.reading.readers.base.build_reader(spec, settings)`,
    `app.reading.readers.cache.CachedReader(reader, cache_dir)` и для `--target detector`
    `app.detect.bottles.BottleDetector`.
    """
    specs = [reader_specs] if isinstance(reader_specs, str) else list(reader_specs)
    decode_image = import_attr("app.normalize.decode", "decode_image")
    build_reader = import_attr("app.reading.readers.base", "build_reader")
    cached_reader = _cached_reader_class()
    readers = []
    for spec in specs:
        reader = cached_reader(build_reader(spec, settings), cache_dir)
        if not isinstance(reader, Reader):
            raise BenchSetupError(f"{type(reader).__name__} не реализует протокол Reader")
        readers.append(reader)
    target_fn: TargetFn | None = None
    if crop != "full":
        target_fn = annotation_target if target == "annotation" else detector_target_fn(settings)
    return Pipeline(decode=decode_image, readers=tuple(readers), lexicon=lexicon, target=target_fn)


# ------------------------------------------------------------------ диагностика
def text_top_slugs(
    hits: Sequence[tuple[Sequence[int], LexHit]], n_slugs: int, *, k: int = TEXT_TOP_K
) -> list[tuple[str, float]]:
    """Диагностика, не ответ: slug по сумме idf попаданий, делённой на (1 + цена).

    Одна сущность (поле, каноническая форма) голосует один раз — лучшим попаданием, как бы
    часто её ни повторили чтения.
    """
    best: dict[tuple[str, str], tuple[float, frozenset[str]]] = {}
    for _, hit in hits:
        if not hit.slugs:
            continue
        weight = idf(len(hit.slugs), max(n_slugs, len(hit.slugs))) / (1.0 + hit.cost)
        key = (hit.field, hit.canonical)
        if key not in best or weight > best[key][0]:
            best[key] = (weight, hit.slugs)
    scores: dict[str, float] = defaultdict(float)
    for weight, slugs in best.values():
        for slug in slugs:
            scores[slug] += weight
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(slug, round(score, 4)) for slug, score in ranked[:k]]


def flat_fields(fields: LabelFields) -> dict[str, Any]:
    """Поля в плоском виде `bench.metrics.field_metrics`: лучшее значение или список."""

    def first(values: list[Any]) -> Any:
        return values[0].value if values else None

    return {
        "winery": first(fields.producer),
        "cuvee": first(fields.cuvee),
        "grape": first(fields.grapes),
        "sugar": [str(e.value) for e in fields.sugar],
        "year": fields.vintage.value if fields.vintage else None,
        "serial": [e.value for e in fields.serial],
        "abv": fields.abv.value if fields.abv else None,
        "color": str(fields.color.value) if fields.color else None,
        "unmatched": [e.value for e in fields.unmatched],
        "is_wine_label": fields.is_wine_label,
    }


def overall_status(readings: Sequence[Reading]) -> str:
    """Статус кадра: ok, если хоть одно чтение ok, иначе статус первого чтения."""
    if not readings:
        return "error"
    if any(r.status == "ok" for r in readings):
        return "ok"
    return readings[0].status


# ------------------------------------------------------------------ прогон
def read_item(
    item: BenchItem, pipeline: Pipeline, *, crop: CropName, crop_px: int, budget_ms: int
) -> dict[str, Any]:
    """Один кадр: декодирование → цель → `read_label`. Сбой кадра записывается, прогон идёт."""
    record: dict[str, Any] = {
        "query_id": item.query_id,
        "image_path": item.image_path.as_posix(),
        "split": item.split,
        "slug": item.slug,
        "reader": None,
        "readers": [],
        "status": "error",
        "raw_text": "",
        "raw": None,
        "lines": [],
        "latency_ms": None,
        "prompt_tokens": None,
        "timings_ms": {},
        "crop_used": None,
        "degraded": [],
        "fields": {},
        "label_fields": None,
        "text_top5": [],
        "text_top5_scores": [],
    }
    stage = "decode"
    try:
        t0 = time.perf_counter()
        image = pipeline.decode(item.image_path.read_bytes())
        t1 = time.perf_counter()
        stage = "target"
        target = pipeline.target(image, item) if crop != "full" and pipeline.target else None
        t2 = time.perf_counter()
        stage = "read"
        result = read_label(
            image,
            readers=pipeline.readers,
            lexicon=pipeline.lexicon,
            target=target,
            crop=crop,
            crop_px=crop_px,
            budget_ms=budget_ms,
        )
        t3 = time.perf_counter()
        stage = "diagnostic"
        top: list[tuple[str, float]] = []
        if pipeline.lexicon is not None:
            segments = token_segments(result.tokens, result.readings)
            hits = label_hits(result.tokens, pipeline.lexicon, segments)
            top = text_top_slugs(hits, pipeline.lexicon.n_slugs)
    except Exception as exc:  # noqa: BLE001 — сбой одного кадра не должен ронять весь прогон
        record["error"] = f"{stage}: {type(exc).__name__}: {exc}"
        return record

    readings = result.readings
    prompt_tokens = [r.prompt_tokens for r in readings if r.prompt_tokens is not None]
    record.update(
        reader=readings[0].key if readings else None,
        readers=[
            {"reader": r.key, "status": r.status, "elapsed_ms": r.elapsed_ms} for r in readings
        ],
        status=overall_status(readings),
        raw_text="\n".join(r.text for r in readings if r.text),
        raw=readings[0].raw if len(readings) == 1 else [r.raw for r in readings],
        lines=[
            {**line.model_dump(mode="json"), "reading": n}
            for n, r in enumerate(readings)
            for line in r.lines
        ],
        # Время из чтений: у попадания в кэш — исходное время модели.
        latency_ms=sum(r.elapsed_ms for r in readings) if readings else None,
        prompt_tokens=sum(prompt_tokens) if prompt_tokens else None,
        timings_ms={
            "decode": round((t1 - t0) * 1000),
            "target": round((t2 - t1) * 1000),
            **{k: v for k, v in result.timings_ms.items() if k != "total"},
            "read_label": round((t3 - t2) * 1000),
        },
        crop_used=readings[0].crop if readings else None,
        degraded=result.degraded,
        fields=flat_fields(result.fields),
        label_fields=result.fields.model_dump(mode="json"),
        text_top5=[slug for slug, _ in top],
        text_top5_scores=[[slug, score] for slug, score in top],
    )
    return record


def _filled(value: Any) -> bool:
    return value is not None and value != [] and value != ""


def compute_run_metrics(
    items: Sequence[BenchItem],
    predictions: Sequence[dict[str, Any]],
    gt_tokens: dict[str, dict[str, Any]],
    *,
    text_top: bool = False,
) -> dict[str, Any]:
    """CER ключевых токенов, заполненность и точность полей, статусы, задержка.

    Кадры со статусом `error` (все чтения сорвались или сбой стенда) в CER и поля не входят.
    `text_top=True` добавляет диагностику текстового top-k — это не критерий решения.
    """
    by_qid = {p["query_id"]: p for p in predictions}
    sources: Counter[str] = Counter()
    samples: list[tuple[list[tuple[str, str]], str]] = []
    per_split: dict[str, list[tuple[list[tuple[str, str]], str]]] = {}
    rows: list[tuple[str, str]] = []
    rows_by_split: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        pred = by_qid.get(item.query_id)
        if pred is None or pred.get("status") == "error":
            continue
        rows.append((item.query_id, item.slug or NONE_SLUG))
        rows_by_split.setdefault(item.split, []).append(rows[-1])
        record = gt_tokens.get(item.slug) if item.slug else None
        source, tokens = expected_label_tokens(item.annotations, record)
        sources[source] += 1
        if tokens:
            sample = (tokens, str(pred.get("raw_text") or ""))
            samples.append(sample)
            per_split.setdefault(item.split, []).append(sample)
    statuses = Counter(str(p.get("status")) for p in predictions)
    degraded = Counter(flag for p in predictions for flag in p.get("degraded") or [])
    latencies = [p["latency_ms"] for p in predictions if p.get("latency_ms") is not None]
    warm = [p["latency_ms"] for p in predictions[1:] if p.get("latency_ms") is not None]
    prompt_tokens = [p["prompt_tokens"] for p in predictions if p.get("prompt_tokens") is not None]
    read = [p for p in predictions if p.get("status") != "error"]
    total = max(1, len(predictions))
    metrics: dict[str, Any] = {
        "items": len(items),
        "predictions": len(predictions),
        "splits": dict(sorted(Counter(item.split for item in items).items())),
        "statuses": dict(sorted(statuses.items())),
        "status_share": {k: round(v / total, 3) for k, v in sorted(statuses.items())},
        "degraded": dict(sorted(degraded.items())),
        "crop_used": dict(sorted(Counter(str(p.get("crop_used")) for p in predictions).items())),
        "token_sources": dict(sorted(sources.items())),
        "key_token_cer": token_cer_table(samples),
        "key_token_cer_by_split": (
            {s: token_cer_table(v) for s, v in sorted(per_split.items())}
            if len(per_split) > 1
            else {}
        ),
        # Доля прочитанных кадров, где поле не пустое: без эталона видно, что модуль что-то даёт.
        "fields_filled": {
            name: round(
                sum(_filled((p.get("fields") or {}).get(name)) for p in read) / len(read), 3
            )
            for name in FILLED_FIELDS
        }
        if read
        else {},
        "field_accuracy": field_metrics(rows, by_qid, gt_tokens) if gt_tokens else {},
        "field_accuracy_by_split": (
            {s: field_metrics(r, by_qid, gt_tokens) for s, r in sorted(rows_by_split.items())}
            if gt_tokens and len(rows_by_split) > 1
            else {}
        ),
        "field_accuracy_notes": FIELD_ACCURACY_NOTES if gt_tokens else {},
        "latency_ms": latency_summary(latencies),
        "latency_ms_warm": latency_summary(warm),  # без первого кадра: холодная загрузка модели
        "prompt_tokens": latency_summary(prompt_tokens),
    }
    if text_top:
        metrics["diagnostic_text_only"] = {
            "note": TEXT_TOP_NOTE,
            **text_topk(rows, by_qid),
            "ooc_reject_rate": ooc_reject_rate(rows, by_qid),
        }
    return metrics


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj) + "\n", encoding="utf-8")


def run_bench(
    items: Sequence[BenchItem],
    pipeline: Pipeline,
    *,
    crop: CropName,
    crop_px: int,
    budget_ms: int,
    out_dir: Path,
    gt_tokens: dict[str, dict[str, Any]],
    run_info: dict[str, Any] | None = None,
    warmup: bool = False,
    seal_test: bool = False,
) -> dict[str, Any]:
    """Прогон набора: predictions.jsonl пишется построчно, в конце — metrics.json.

    `warmup=True` — до цикла по кадрам каждый читатель читает синтетический кадр размера
    `crop_px` с большим бюджетом и мимо кэша. Время прогрева лежит в `run.warmup` и не
    входит ни в задержку кадров, ни в `run.wall_s`. `seal_test` — метрики `metrics.json` без
    кадров test, а метрики всего набора вместе с test — в конверт `SEALED_METRICS_FILE`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    warm: dict[str, dict[str, Any]] | None = None
    if warmup and pipeline.readers:
        warm = warm_readers(pipeline.readers, image=synthetic_label(crop_px))
    started = time.perf_counter()
    predictions = []
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:  # строго по одному кадру: карту занимает одна модель
            record = read_item(item, pipeline, crop=crop, crop_px=crop_px, budget_ms=budget_ms)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            predictions.append(record)
    sealed = [item for item in items if seal_test and item.split == SEALED_SPLIT]
    open_items = [item for item in items if not (seal_test and item.split == SEALED_SPLIT)]
    open_ids = {item.query_id for item in open_items}
    metrics = compute_run_metrics(
        open_items,
        [p for p in predictions if p["query_id"] in open_ids],
        gt_tokens,
        text_top=pipeline.lexicon is not None,
    )
    metrics["run"] = {
        **(run_info or {}),
        "reader_id": getattr(pipeline.readers[0], "id", None) if pipeline.readers else None,
        "readers": [
            {"id": getattr(r, "id", None), "version": getattr(r, "version", None)}
            for r in pipeline.readers
        ],
        "cache": [_cache_stats(r) for r in pipeline.readers],
        "lexicon_entries": len(pipeline.lexicon) if pipeline.lexicon is not None else None,
        "crop": crop,
        "crop_px": crop_px,
        "budget_ms": budget_ms,
        # Бюджет больше сервисного: статусы timeout и budget_exceeded не такие, как в сервисе.
        "sla_budget": budget_ms <= DEFAULT_BUDGET_MS,
        # По читателям: {"status", "elapsed_ms"}; None — прогрева не было.
        "warmup": warm,
        "wall_s": round(time.perf_counter() - started, 3),
    }
    if sealed:
        envelope = compute_run_metrics(
            items, predictions, gt_tokens, text_top=pipeline.lexicon is not None
        )
        envelope["warning"] = SEALED_WARNING
        envelope["run"] = metrics["run"]
        write_json(out_dir / SEALED_METRICS_FILE, envelope)
        metrics["sealed_test"] = {
            "file": SEALED_METRICS_FILE,
            "items": len(sealed),
            "note": SEALED_NOTE,
        }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def dry_run_report(
    items: Sequence[BenchItem],
    gt_tokens: dict[str, dict[str, Any]],
    *,
    crop: CropName = "full",
    target: str = "annotation",
    lexicon_path: Path | None = None,
    clusters: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Проверка набора без читателя: файлы, эталон, согласованность split, рамки, словарь."""
    missing = check_images(items)
    summary = summarize(items, clusters)
    boxes: dict[str, Any] | None = None
    if crop != "full" and target == "annotation":
        without_box = sum(not has_target_box(item) for item in items)
        boxes = {"source": "annotation", "key": TARGET_BOX_KEY, "missing": without_box}
    elif crop != "full":
        boxes = {"source": "detector"}
    sources = Counter(
        expected_label_tokens(item.annotations, gt_tokens.get(item.slug) if item.slug else None)[0]
        for item in items
    )
    return {
        "dry_run": True,
        "ok": bool(items)
        and not missing
        and not summary["split_conflicts"]
        and not summary["cluster_conflicts"]
        and not (boxes and boxes.get("missing")),
        **summary,
        "token_sources": dict(sorted(sources.items())),
        "gt_tokens_loaded": len(gt_tokens),
        "missing_images": missing,
        "crop": crop,
        "crop_boxes": boxes,
        "lexicon": (
            None
            if lexicon_path is None
            else {"path": str(lexicon_path), "exists": lexicon_path.is_file()}
        ),
    }


# ------------------------------------------------------------------ CLI
def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.ocr_bench", description="Замер чтения этикетки на наборе."
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--synth-dir", type=Path, help="по умолчанию data/raw/synth")
    parser.add_argument("--manifest", type=Path, help="manifest: query_id<TAB>image_path")
    parser.add_argument("--gt", type=Path, help="manifest: query_id<TAB>slug|__none__[…]")
    parser.add_argument("--ann", type=Path, help="manifest: JSONL разметки")
    parser.add_argument("--images-dir", type=Path, help="manifest: папка кадров")
    parser.add_argument(
        "--split",
        choices=("all", *get_args(Split)),
        help="по умолчанию прогон идёт по dev (public — smoke), --dry-run — по всему набору; "
        "test — только явно, один раз для отчёта; при all метрики test запечатаны в "
        f"{SEALED_METRICS_FILE}",
    )
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="с --split all: метрики test в metrics.json и в консоли вместе с остальными. Не для "
        "прогонов, которые идут во вход обучения",
    )
    parser.add_argument("--limit", type=int, help="первые N кадров (проверка стенда)")
    parser.add_argument(
        "--reader",
        action="append",
        help="vlm:<модель> | easyocr | rapidocr; можно повторить — читатели идут по очереди",
    )
    parser.add_argument("--crop", choices=get_args(CropName), default="full")
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="annotation",
        help="рамка цели для bottle/label/band: разметка кадра (оракул) или детектор CV",
    )
    parser.add_argument("--crop-px", type=int, default=1024)
    parser.add_argument(
        "--budget-ms",
        type=int,
        default=DEFAULT_BUDGET_MS,
        help="бюджет чтения кадра; больше сервисного — прогон без SLA (run.sla_budget=false)",
    )
    parser.add_argument(
        "--gt-tokens", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl"
    )
    parser.add_argument("--lexicon", type=Path, default=default_lexicon_path(settings))
    parser.add_argument("--no-lexicon", action="store_true", help="без словаря: только правила")
    parser.add_argument("--cache-dir", type=Path, help="по умолчанию <cache_dir>/readings")
    parser.add_argument("--out", type=Path, help="папка прогона, например runs/<имя>/")
    parser.add_argument("--overwrite", action="store_true", help="перезаписать прогон в --out")
    parser.add_argument("--dry-run", action="store_true", help="проверить набор без читателя")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="прогреть читателей синтетическим кадром до замера (по умолчанию да); "
        "время прогрева — в run.warmup, в задержку кадров не входит",
    )
    return parser


def load_items(
    args: argparse.Namespace, settings: Settings, clusters: dict[str, str] | None = None
) -> list[BenchItem]:
    if args.dataset == "public":
        return load_public(settings)
    if args.dataset == "synth":
        return load_synth(args.synth_dir or settings.data_dir / "raw" / "synth", clusters=clusters)
    if args.manifest is None or args.gt is None:
        raise DatasetError("--dataset manifest требует --manifest и --gt")
    return load_manifest(
        args.manifest, args.gt, args.ann, images_dir=args.images_dir, clusters=clusters
    )


def default_split(dataset: str, *, dry_run: bool) -> str:
    """Без `--split`: прогон — по dev (test смотрят один раз и явно), проверка набора — весь."""
    return "all" if dry_run or dataset == "public" else "dev"


def select_items(
    items: Sequence[BenchItem], *, split: str = "all", limit: int | None = None
) -> list[BenchItem]:
    chosen = [item for item in items if split == "all" or item.split == split]
    return chosen[:limit] if limit else chosen


def load_bench_lexicon(path: Path | None) -> Lexicon | None:
    """Словарь, если он собран; нет файла — None и предупреждение, битый файл — ошибка."""
    if path is None:
        return None
    if not path.is_file():
        print(
            f"словаря нет: {path} — винодельня, кюве, сорт и text_top5 будут пустыми "
            "(scripts/build_lexicon.py)",
            file=sys.stderr,
        )
        return None
    try:
        return Lexicon.load(path)
    except (OSError, ValueError, KeyError) as exc:
        raise BenchSetupError(f"словарь {path} не читается: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    gt_tokens = load_gt_tokens(args.gt_tokens) if args.gt_tokens.is_file() else {}
    clusters = twin_clusters(gt_tokens)
    split = args.split or default_split(args.dataset, dry_run=args.dry_run)
    try:
        items = select_items(load_items(args, settings, clusters), split=split, limit=args.limit)
    except DatasetError as exc:
        print(f"ошибка набора: {exc}", file=sys.stderr)
        return 2
    lexicon_path = None if args.no_lexicon else args.lexicon

    if args.dry_run:
        report = {
            "dataset": args.dataset,
            "split": split,
            **dry_run_report(
                items,
                gt_tokens,
                crop=args.crop,
                target=args.target,
                lexicon_path=lexicon_path,
                clusters=clusters,
            ),
        }
        if args.out is not None:
            write_json(args.out / "dataset_check.json", report)
        print_json(report)
        return 0 if report["ok"] else 1

    specs = args.reader or []
    if not specs or args.out is None:
        parser.error("без --dry-run нужны --reader и --out")
    for spec in specs:
        try:
            parse_reader_spec(spec)
        except ValueError as exc:
            parser.error(str(exc))
    if (args.out / "predictions.jsonl").exists() and not args.overwrite:
        print(f"{args.out} уже содержит прогон: другое имя или --overwrite", file=sys.stderr)
        return 2
    missing = check_images(items)
    if missing or not items:
        print("набор не готов:\n" + "\n".join(missing or ["нет кадров"]), file=sys.stderr)
        return 1
    try:
        lexicon = load_bench_lexicon(lexicon_path)
        cache_dir = args.cache_dir or settings.cache_dir / "readings"
        pipeline = resolve_pipeline(
            specs, settings, cache_dir, lexicon=lexicon, target=args.target, crop=args.crop
        )
        for spec, reader in zip(specs, pipeline.readers, strict=True):
            if not reader.available():
                raise BenchSetupError(f"читатель {spec} недоступен")
    except (BenchSetupError, ValueError) as exc:
        print(f"стенд не готов: {exc}", file=sys.stderr)
        return 3

    seal = seals_test(split, unseal=args.unseal_test)
    sealed = sum(item.split == SEALED_SPLIT for item in items) if seal else 0
    if sealed:
        print(
            f"--split all: метрики {sealed} кадров test запечатаны в {SEALED_METRICS_FILE}; "
            "metrics.json и вывод ниже — без них",
            file=sys.stderr,
        )
    run_info = {
        "dataset": args.dataset,
        "reader_specs": specs,
        "split": split,
        "test_sealed": bool(sealed),
        "limit": args.limit,
        "target": None if args.crop == "full" else args.target,
        "lexicon": (
            None
            if lexicon is None
            else {"path": str(lexicon_path), "meta": lexicon.meta, "entries": len(lexicon)}
        ),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    metrics = run_bench(
        items,
        pipeline,
        crop=args.crop,
        crop_px=args.crop_px,
        budget_ms=args.budget_ms,
        out_dir=args.out,
        gt_tokens=gt_tokens,
        run_info=run_info,
        warmup=args.warmup,
        seal_test=seal,
    )
    keys = ("items", "statuses", "degraded", "latency_ms", "key_token_cer", "fields_filled")
    print_json(
        {
            **{
                k: metrics[k]
                for k in (*keys, "field_accuracy", "diagnostic_text_only", "sealed_test")
                if k in metrics
            },
            "warmup": metrics["run"]["warmup"],
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
