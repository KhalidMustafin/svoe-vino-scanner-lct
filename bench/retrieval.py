"""Стенд поиска по фотографии: кадр → виды → индекс → кандидаты, предсказания и метрики.

    python -m bench.retrieval --index data/index/visual.npz --queries pairs \\
        --top-k 20 --out runs/pairs-siglip/
    python -m bench.retrieval --index data/index/visual.npz --queries pairs_phone \\
        --target detector --out runs/pairs-phone-detector/
    python -m bench.retrieval --index data/index/visual.npz --queries rwl_src --limit 10 \\
        --out runs/rwl-src-visual/
    python -m bench.retrieval --index data/index/visual.npz --queries pairs \\
        --with-ocr runs/vlm-label/predictions.jsonl --out runs/pairs-cv-ocr/

Кадр идёт тем же путём, что в сервисе: `decode_image` → (порча «как с телефона») → рамка
цели → `from_query` → `VisualIndex.search`. Стенд ничего не подбирает и не вводит порогов:
он фиксирует параметры прогона, пишет `predictions.jsonl` и считает `metrics.json`.

С `--with-ocr <predictions.jsonl>` к прогону добавляется текст: для тех же `query_id`
берутся поля этикетки из прогона `bench.ocr_bench`, кандидаты CV переранжирует
`app.resolve.rerank`, и метрики считаются дважды — «только CV» (`cv_top5`) и «CV+OCR»
(`final_top5`), с разницей в пунктах (`cv_vs_cv_ocr` в `metrics.json`). Разница считает
resolve целиком: и серии двойников, и признаки текста. Без флага стенд работает как
раньше — один визуальный канал, никакого resolve.

Что меряем и почему именно это:

    по карточке      top-1/3/5/10 — строгая цифра ТЗ: тот самый slug;
    по макету        то же, но попаданием считается любой член визуальной группы
                     (`visual_group`) — тех позиций, про которые в разметке прямо сказано
                     «image search cannot separate them». Разница с карточкой — цена
                     вопроса «какой год», который снимку не решить и который остаётся
                     тексту этикетки;
    по линейке       попаданием считается любой член `cluster_B`. Эта строка ЗАВЫШЕНА и
                     названа честно: `cluster_B` — не двойники, а «та же винодельня +
                     похожее имя или похожее фото», и внутри него меняются сорт, цвет и
                     категория (level_B: grape 208, name 275, category 125 из 318
                     кластеров). Приводить её как «тот же макет, другой год» нельзя;
    credited_slugs   сколько позиций каталога зачтено правильным ответом на запрос —
                     знаменатель поблажки, без которого две строки выше не перепроверить;
    recall@k         верный slug вообще попал в выдачу — потолок для resolve: чего нет в
                     списке, того текст уже не спасёт;
    margin           отрыв top-1 от top-2 отдельно у верных и у неверных ответов. Признак
                     уверенности канала: если у неверных отрыв такой же, различить их
                     нечем, и калибровать в resolve будет нечего;
    группы           top-1 внутри визуальных групп каталога — самая тяжёлая часть: там
                     этикетки почти одинаковые;
    ничьи            запросы, где у top-1 и top-2 счёт совпал до бита: семь файлов каталога
                     стоят у двух slug сразу, и снимку их не различить в принципе;
    неопубликованные доля ответов на карточку, которой на портале нет (`published=false`);
    p50/p95          время на кадр по шагам.

Разбиение dev/test (`bench.datasets.split_for`, ключ — `cluster_B`) проставляется каждому
запросу; без `--split` прогон идёт по dev, как в `bench.ocr_bench`, а test смотрят один
раз и явно. Разрез по половинам попадает в метрики при любом выборе. `--split all` пишет
предсказания по всему набору (вход `bench.train_resolve`), но метрики test запечатаны:
`metrics.json` и консоль — без кадров test, сами они — в конверте
`metrics.test-sealed.json` (`--unseal-test` снимает печать явно).

Коды выхода: 0 — готово, 1 — набор не прошёл проверку, 2 — ошибка аргументов или набора,
3 — индекс, модель или детектор не готовы.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

import numpy as np

from app.config import Settings, get_settings
from app.detect.bottles import BottleDetector, TargetSelection, select_target
from app.features.contracts import Aggregation, Candidate, Embedder
from app.features.embedder import DEFAULT_BATCH_SIZE, ModelNotAvailable, SiglipEmbedder
from app.features.index import IndexMismatch, VisualIndex
from app.features.views import BACKGROUND, from_query
from app.normalize.decode import decode_image
from app.reading.contracts import Color, Evidence, LabelFields, SugarClass
from app.resolve.attrs import CatalogAttrs
from app.resolve.rerank import (
    DEFAULT_CONFIG,
    WEIGHTS_NOTE,
    AbstainMode,
    RerankConfig,
    rerank,
)
from bench.datasets import (
    SEALED_METRICS_FILE,
    SEALED_NOTE,
    SEALED_SPLIT,
    SEALED_WARNING,
    DatasetError,
    seals_test,
    split_for,
)
from bench.metrics import (
    NONE_SLUG,
    dumps,
    latency_summary,
    load_gt_tokens,
    ooc_reject_rate,
    percentile,
    print_json,
    rate,
)
from bench.phone_shot import phone_shot_rgb
from bench.queries import QUERY_SETS, QueryItem, load_queries, missing_images, notes_for, summarize

TARGETS = ("detector", "none")
SPLITS = ("dev", "test", "all")
DEFAULT_TOP_K = 20
#: Места, на которых считаются попадания. Больше `--top-k` брать нечего.
TOP_AT = (1, 3, 5, 10)
#: Столбцы сравнения «CV» и «CV+OCR» в предсказаниях. Оба — списки slug, не длиннее пяти.
CV_KEY = "cv_top5"
FINAL_KEY = "final_top5"
#: Глубина сравнения: в обоих столбцах лежит по пять slug.
COMPARE_TOP_K = 5
CV_OCR_NOTE = (
    "разница считает resolve целиком: и серии двойников, и признаки текста. Веса и пороги "
    "resolve не подобраны на данных"
)
#: Почему строка по линейке завышена: пул resolve наполняется тем же ключом.
LINE_NOTE = (
    "попаданием считается любой член cluster_B, а cluster_B — не двойники, а «та же "
    "винодельня + похожее имя или похожее фото»: внутри меняются сорт, цвет и категория. "
    "Строка завышена; при series_pool=on пул resolve наполняется тем же ключом, и для "
    "колонки «CV+OCR» она почти тавтологична — читать by_card и by_visual_group"
)
#: Почему ничьи — потолок by_card, а не ошибка модели.
TIE_NOTE = (
    "счёт top-1 и top-2 совпал до бита: в slug_photo_map семь файлов стоят у двух slug "
    "сразу, их векторы одинаковы, и порядок решает стабильная сортировка. Различить их "
    "снимку нечем — это задача текста в resolve"
)
#: Счёт считается совпавшим, если разница меньше этого: float16 в индексе, float32 в счёте.
TIE_EPS = 1e-9

TargetFn = Callable[[np.ndarray], TargetSelection | None]
GtTokens = Mapping[str, Mapping[str, Any]]


class BenchSetupError(RuntimeError):
    """Стенд не собрался: нет индекса, весов модели или детектора."""


def _ms(seconds: float) -> float:
    """Миллисекунды с долями: поиск занимает 1,4 мс, и в целых числах это был бы ноль."""
    return round(seconds * 1000, 3)


@dataclass(frozen=True)
class Pipeline:
    """Всё, что нужно одному кадру: индекс, модель и способ найти на кадре бутылку.

    `attrs` не `None` — к кадру добавляется resolve: кандидаты CV переранжируются полями
    этикетки. Без признаков каталога стенд остаётся чисто визуальным.
    """

    index: VisualIndex
    embedder: Embedder
    target: TargetFn | None = None  # None — рамку не ищем, кадр режется окнами
    top_k: int = DEFAULT_TOP_K
    per_slug: Aggregation = "max"
    attrs: CatalogAttrs | None = None
    cfg: RerankConfig = DEFAULT_CONFIG


def detector_target_fn(device: str | None = None) -> TargetFn:
    """Цель от детектора бутылок. Весов нет — стенд не готов, а не тихо ищет по окнам."""
    detector = BottleDetector(device)
    if not detector.available():
        raise BenchSetupError("детектор бутылок недоступен: нет весов torchvision или torch")

    def target(image: np.ndarray) -> TargetSelection | None:
        return select_target(detector.detect(image))

    return target


# ------------------------------------------------------------------ поля из прогона OCR
def _evidence[T](value: T) -> Evidence[T]:
    return Evidence(value=value, sources=[], support=1, conf=None)


def fields_from_flat(flat: Mapping[str, Any]) -> LabelFields:
    """Плоские поля предсказания (`flat_fields` стенда OCR) → `LabelFields`.

    Запасной путь на случай старых прогонов: у плоской записи нет ни ключей чтений, ни
    уверенности, поэтому правило отказа по ней не срабатывает (`conf=None` не дотянет до
    `winery_conf_min`). Полные поля лежат в `label_fields`, и берутся они.
    """

    def listed(value: Any) -> list[Any]:
        if value is None or value == "":
            return []
        return [value] if isinstance(value, str | int | float) else list(value)

    color = flat.get("color")
    return LabelFields(
        producer=[_evidence(str(v)) for v in listed(flat.get("winery"))],
        cuvee=[_evidence(str(v)) for v in listed(flat.get("cuvee"))],
        grapes=[_evidence(str(v)) for v in listed(flat.get("grape"))],
        sugar=[
            _evidence(SugarClass(v))
            for v in listed(flat.get("sugar"))
            if v in SugarClass._value2member_map_
        ],
        vintage=_evidence(int(flat["year"])) if flat.get("year") else None,
        serial=[_evidence(str(v)) for v in listed(flat.get("serial"))],
        abv=_evidence(float(flat["abv"])) if flat.get("abv") is not None else None,
        color=_evidence(Color(color)) if color in Color._value2member_map_ else None,
        is_wine_label=bool(flat.get("is_wine_label", True)),
    )


def fields_of_record(record: Mapping[str, Any]) -> LabelFields | None:
    """Поля одного предсказания стенда OCR: полные, иначе плоские, иначе ничего."""
    full = record.get("label_fields")
    if isinstance(full, Mapping):
        return LabelFields.model_validate(full)
    flat = record.get("fields")
    if isinstance(flat, Mapping) and flat:
        return fields_from_flat(flat)
    return None


def load_read_fields(path: Path) -> dict[str, LabelFields]:
    """`predictions.jsonl` прогона `bench.ocr_bench` → поля этикетки по `query_id`.

    Кадры со сбоем чтения пропускаются: у них полей нет, а подставить пустые — значит
    сказать resolve «этикетка прочитана, и на ней ничего».
    """
    if not path.is_file():
        raise DatasetError(f"нет прогона OCR: {path}")
    out: dict[str, LabelFields] = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: не JSON: {exc}") from exc
            qid = record.get("query_id")
            if not qid or record.get("status") == "error":
                continue
            fields = fields_of_record(record)
            if fields is not None:
                out[str(qid)] = fields
    return out


# ------------------------------------------------------------------ группы каталога
def line_key(slug: str | None, gt_tokens: GtTokens) -> str | None:
    """Линейка винодельни: кластер `cluster_B`, а вне кластеров — сам slug.

    Это НЕ группа двойников. `cluster_B` собран как «level_A (та же винодельня + похожее
    имя) + почти одинаковое фото внутри винодельни», и внутри одного кластера лежат вина
    разного сорта, цвета и категории. Вне кластера вино равно только себе.
    """
    if not slug:
        return None
    record = gt_tokens.get(slug)
    cluster = record.get("cluster_B") if record else None
    return f"B{cluster}" if cluster is not None else f"slug:{slug}"


def group_key(slug: str | None, gt_tokens: GtTokens) -> str | None:
    """Визуальная группа каталога (`visual_group`); вне групп — `None`.

    Разметка строит её как «RMS < 12 или одинаковый md5 фото» и подписывает прямо: proxy
    for «image search cannot separate them». Только по ней честно считать «другой год той
    же этикетки»: `visual_mates` записи — ровно эта группа без самого slug.
    """
    record = gt_tokens.get(slug) if slug else None
    group = record.get("visual_group") if record else None
    if not group:
        return None
    return "-".join(str(part) for part in group) if isinstance(group, list) else str(group)


def twin_key(slug: str | None, gt_tokens: GtTokens) -> str | None:
    """Ключ «та же бутылка на снимке»: визуальная группа, а вне групп — сам slug."""
    if not slug:
        return None
    return group_key(slug, gt_tokens) or f"slug:{slug}"


def catalog_groups(gt_tokens: GtTokens) -> int:
    """Сколько визуальных групп в каталоге: знаменатель для строки про группы."""
    return len({key for slug in gt_tokens if (key := group_key(slug, gt_tokens))})


def key_sizes(gt_tokens: GtTokens, key_fn: Callable[[str | None, GtTokens], str | None]) -> Counter:
    """Размер каждой группы каталога по ключу: сколько slug зачтётся одним попаданием."""
    counts: Counter = Counter()
    for slug in gt_tokens:
        key = key_fn(slug, gt_tokens)
        if key is not None:
            counts[key] += 1
    return counts


def first_rank(slugs: Sequence[str], match: Callable[[str], bool]) -> int | None:
    """Место первого подошедшего кандидата (с единицы); не нашёлся — `None`."""
    for rank, slug in enumerate(slugs, start=1):
        if match(slug):
            return rank
    return None


# ------------------------------------------------------------------ один кадр
def candidate_row(candidate: Candidate) -> dict[str, Any]:
    return {
        "slug": candidate.slug,
        "score": round(candidate.score, 6),
        "view": candidate.view,
        "rank": candidate.rank,
    }


def search_item(
    item: QueryItem, pipeline: Pipeline, fields: LabelFields | None = None
) -> dict[str, Any]:
    """Один запрос: декодирование → порча → рамка → виды → поиск → (resolve).

    Сбой кадра записывается в предсказание и не валит прогон: на 358 запросах один битый
    файл не должен стоить всего замера. Resolve добавляется, только если стенд собран с
    признаками каталога (`--with-ocr`): без них колонки «CV+OCR» нет.
    """
    record: dict[str, Any] = {
        "query_id": item.query_id,
        "image_path": item.image_path.as_posix(),
        "slug": item.slug,
        "meta": item.meta,
        "status": "error",
        "error": None,
        "top": [],
        "cv_top5": [],
        "margin": 0.0,
        "hit_rank": None,
        "views": [],
        "target": None,
        "timings_ms": {},
    }
    if pipeline.attrs is not None:
        record.update(
            read_fields=fields is not None,
            final_top5=[],
            slug_pred=None,
            outcome=None,
            resolve_score=None,
            resolve_margin=None,
            series_margin=None,
            evidence={},
        )
    stage = "read"
    started = time.perf_counter()
    try:
        data = item.image_path.read_bytes()
        stage = "decode"
        # Прозрачность запроса сводится тем же цветом, что и у эталона: иначе PNG с альфой
        # ушёл бы в модель на белом фоне против серого у packshot.
        image = decode_image(data, background=BACKGROUND)
        decoded = time.perf_counter()
        stage = "phone_shot"
        seed = item.phone_seed
        if seed is not None:
            image = phone_shot_rgb(image, seed)
        spoiled = time.perf_counter()
        stage = "target"
        target = pipeline.target(image) if pipeline.target is not None else None
        located = time.perf_counter()
        stage = "views"
        views = from_query(image, target)
        cropped = time.perf_counter()
        stage = "search"
        result = pipeline.index.search(
            views, pipeline.embedder, top_k=pipeline.top_k, per_slug=pipeline.per_slug
        )
        searched = time.perf_counter()
        stage = "resolve"
        resolved = None
        if pipeline.attrs is not None:
            resolved = rerank(result, fields, pipeline.attrs, cfg=pipeline.cfg)
        resolve_ms = _ms(time.perf_counter() - searched)
    except Exception as exc:  # noqa: BLE001 — сбой одного кадра прогон не останавливает
        record["error"] = f"{stage}: {type(exc).__name__}: {exc}"
        record["timings_ms"] = {"total": _ms(time.perf_counter() - started)}
        return record

    slugs = [c.slug for c in result.candidates]
    if resolved is not None:
        record.update(
            # Отказ — пустой список: так `ooc_reject_rate` видит отказ, а не ответ.
            final_top5=[] if resolved.slug is None else [s for s, _ in resolved.top5],
            slug_pred=resolved.slug,
            outcome=resolved.outcome,
            resolve_score=resolved.score,
            resolve_margin=resolved.margin,
            series_margin=resolved.series_margin,
            evidence=resolved.evidence,
        )
    record.update(
        status="ok",
        top=[candidate_row(c) for c in result.candidates],
        cv_top5=slugs[:5],  # ключ `bench.metrics.cv_vs_cv_ocr`: сравнение «CV» и «CV+OCR»
        margin=round(result.margin, 6),
        hit_rank=first_rank(slugs, lambda s: s == item.slug) if item.slug else None,
        views=sorted(views),
        target=(
            None
            if target is None
            else {
                "method": target.method,
                "score": round(target.score, 4),
                "confident": target.confident,
            }
        ),
        timings_ms={
            "decode": _ms(decoded - started),
            "phone_shot": _ms(spoiled - decoded),
            "target": _ms(located - spoiled),
            "views": _ms(cropped - located),
            **result.timings_ms,
            **({"resolve": resolve_ms} if resolved is not None else {}),
            "total": _ms(time.perf_counter() - started),
        },
    )
    return record


# ------------------------------------------------------------------ метрики
def share(num: int, den: int) -> float | None:
    """Доля; мерить нечего — `None`, а не 0,0.

    `bench.metrics.rate` делит на `max(1, den)`, и на пустом наборе печатал бы «канал не
    угадал ничего» там, где запросов просто не было.
    """
    return None if den == 0 else rate(num, den)


def rank_table(ranks: Sequence[int | None], *, top_k: int) -> dict[str, float | int | None]:
    """Доли попаданий на местах `TOP_AT` и доля запросов, где верный ответ вообще в выдаче."""
    n = len(ranks)
    table: dict[str, float | int | None] = {
        f"top{k}": share(sum(r is not None and r <= k for r in ranks), n) for k in TOP_AT
    }
    table[f"recall_at_{top_k}"] = share(sum(r is not None for r in ranks), n)
    table["n"] = n
    return table


def margin_table(values: Sequence[float]) -> dict[str, float | int | None]:
    """Медиана и 10-й перцентиль отрыва: ближний ранг, без интерполяции."""
    return {
        "median": percentile(values, 0.5),
        "p10": percentile(values, 0.1),
        "n": len(values),
    }


def prediction_slugs(prediction: Mapping[str, Any], key: str = "top") -> list[str]:
    """Кандидаты предсказания: полная выдача CV (`top`) или столбец сравнения из slug."""
    values = prediction.get(key) or []
    if key == "top":
        return [str(candidate["slug"]) for candidate in values]
    return [str(slug) for slug in values]


def top_slug(prediction: Mapping[str, Any], key: str = "top") -> str | None:
    slugs = prediction_slugs(prediction, key)
    return slugs[0] if slugs else None


#: Столбцы попаданий: имя метрики → ключ каталога, по которому засчитывается ответ.
RANK_COLUMNS: dict[str, Callable[[str | None, GtTokens], str | None]] = {
    "by_visual_group": twin_key,
    "by_winery_line": line_key,
}


def rank_columns(
    items: Sequence[QueryItem],
    by_qid: Mapping[str, Mapping[str, Any]],
    gt_tokens: GtTokens,
    *,
    key: str = "top",
) -> dict[str, list[int | None]]:
    """Места верной карточки, верного макета и верной линейки по каждому запросу.

    Три строки вместо одной «по серии»: попадание в `cluster_B` и попадание в визуальную
    группу — разные поблажки, и мерить их одним числом значит выдавать вторую за первую.
    """
    columns: dict[str, list[int | None]] = {name: [] for name in ("by_card", *RANK_COLUMNS)}
    for item in items:
        if not item.slug:
            continue
        prediction = by_qid.get(item.query_id) or {}
        slugs = prediction_slugs(prediction, key)
        columns["by_card"].append(first_rank(slugs, lambda s, t=item.slug: s == t))
        for name, key_fn in RANK_COLUMNS.items():
            want = key_fn(item.slug, gt_tokens)
            columns[name].append(
                first_rank(slugs, lambda s, f=key_fn, w=want: f(s, gt_tokens) == w)
            )
    return columns


def credited_slugs(
    items: Sequence[QueryItem],
    gt_tokens: GtTokens,
    key_fn: Callable[[str | None, GtTokens], str | None],
) -> dict[str, Any]:
    """Сколько позиций каталога зачтено одному запросу правильным ответом.

    Без этой строки поблажку нельзя перепроверить: «по линейке 90 %» при пуле в три slug
    и при пуле в один — разные цифры.
    """
    sizes = key_sizes(gt_tokens, key_fn)
    pools = [sizes.get(key_fn(item.slug, gt_tokens) or "", 1) or 1 for item in items if item.slug]
    if not pools:
        return {"median": None, "max": None, "mean": None, "n": 0}
    return {
        "median": percentile(pools, 0.5),
        "max": max(pools),
        "mean": round(sum(pools) / len(pools), 3),
        "n": len(pools),
    }


def margins_by_correctness(
    items: Sequence[QueryItem], by_qid: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, float | int | None]]:
    """Отрыв top-1 от top-2 у верных и у неверных ответов — врозь.

    Одинаковые распределения означали бы, что отрыв не отличает верный ответ от чужого и
    калибровать уверенность в resolve нечем.
    """
    correct: list[float] = []
    wrong: list[float] = []
    for item in items:
        if not item.slug:
            continue
        prediction = by_qid.get(item.query_id) or {}
        if prediction.get("status") != "ok":
            continue
        margin = float(prediction.get("margin") or 0.0)
        (correct if top_slug(prediction) == item.slug else wrong).append(margin)
    return {"correct": margin_table(correct), "wrong": margin_table(wrong)}


def visual_group_metrics(
    items: Sequence[QueryItem], by_qid: Mapping[str, Mapping[str, Any]], gt_tokens: GtTokens
) -> dict[str, Any]:
    """Запросы, цель которых лежит в визуальной группе каталога: там макет общий.

    `answered_by_mate` — первым ответом стал двойник, а не сама карточка: ровно тот спор,
    который снимку не решить.
    """
    queries = correct = by_mate = 0
    groups: set[str] = set()
    for item in items:
        key = group_key(item.slug, gt_tokens)
        if key is None:
            continue
        queries += 1
        groups.add(key)
        first = top_slug(by_qid.get(item.query_id) or {})
        correct += int(first == item.slug)
        by_mate += int(
            first is not None and first != item.slug and group_key(first, gt_tokens) == key
        )
    return {
        "queries": queries,
        "groups_touched": len(groups),
        "groups_in_catalog": catalog_groups(gt_tokens),
        "top1": share(correct, queries),
        "answered_by_mate": share(by_mate, queries),
    }


def tie_rate(items: Sequence[QueryItem], by_qid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Запросы, где счёт top-1 и top-2 совпал: потолок `by_card`, а не ошибка модели."""
    queries = ties = 0
    for item in items:
        prediction = by_qid.get(item.query_id) or {}
        top = prediction.get("top") or []
        if prediction.get("status") != "ok" or len(top) < 2:
            continue
        queries += 1
        ties += int(abs(float(top[0]["score"]) - float(top[1]["score"])) <= TIE_EPS)
    return {"queries": queries, "ties": ties, "rate": share(ties, queries), "note": TIE_NOTE}


def unpublished_answers(
    items: Sequence[QueryItem], by_qid: Mapping[str, Mapping[str, Any]], gt_tokens: GtTokens
) -> dict[str, Any] | None:
    """Доля ответов на карточку, которой на портале нет (`published=false`).

    Неопубликованные позиции остаются в индексе — как эталоны и как отвлекающие, — но
    ответ на такую карточку это 404 у пользователя, и цифру надо знать заранее.
    """
    unpublished = {slug for slug, record in gt_tokens.items() if record.get("published") is False}
    if not unpublished:
        return None
    answered = hits = 0
    for item in items:
        prediction = by_qid.get(item.query_id) or {}
        if prediction.get("status") != "ok":
            continue
        first = top_slug(prediction)
        if first is None:
            continue
        answered += 1
        hits += int(first in unpublished)
    return {
        "in_catalog": len(unpublished),
        "answered": answered,
        "top1_unpublished": hits,
        "rate": share(hits, answered),
    }


def out_of_catalog_scores(
    items: Sequence[QueryItem], by_qid: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Диагностика кадров вне каталога: с каким счётом и отрывом канал предлагает чужое.

    Это не порог отказа — порогов стенд не вводит. Это распределение, по которому потом
    будет видно, есть ли вообще чему отказывать.
    """
    scores: list[float] = []
    margins: list[float] = []
    for item in items:
        if item.slug is not None:
            continue
        prediction = by_qid.get(item.query_id) or {}
        top = prediction.get("top") or []
        if prediction.get("status") != "ok" or not top:
            continue
        scores.append(float(top[0]["score"]))
        margins.append(float(prediction.get("margin") or 0.0))
    if not scores:
        return None
    return {
        "queries": len(scores),
        "top1_score": {"median": percentile(scores, 0.5), "p10": percentile(scores, 0.1)},
        "margin": {"median": percentile(margins, 0.5), "p10": percentile(margins, 0.1)},
    }


# ------------------------------------------------------------------ CV против CV+OCR
def column_table(ranks: Sequence[int | None], *, depth: int = COMPARE_TOP_K) -> dict[str, Any]:
    """Попадания на местах не глубже `depth`: столбцы сравнения длиной в пять slug."""
    table: dict[str, Any] = {
        f"top{k}": share(sum(r is not None and r <= k for r in ranks), len(ranks))
        for k in TOP_AT
        if k <= depth
    }
    table["n"] = len(ranks)
    return table


def column_metrics(
    items: Sequence[QueryItem],
    by_qid: Mapping[str, Mapping[str, Any]],
    gt_tokens: GtTokens,
    *,
    key: str,
) -> dict[str, Any]:
    """Одна колонка сравнения: карточка, макет, линейка и доля отказов вне каталога."""
    columns = rank_columns(items, by_qid, gt_tokens, key=key)
    # Отказ — пустой список кандидатов, но только у кадра, который дошёл до конца: сбой
    # разжатия тоже даёт пустую выдачу, и считать его осознанным отказом нельзя.
    rows = [
        (item.query_id, item.slug or NONE_SLUG)
        for item in items
        if (by_qid.get(item.query_id) or {}).get("status") == "ok"
    ]
    return {
        **{name: column_table(ranks) for name, ranks in columns.items()},
        "ooc_reject_rate": ooc_reject_rate(rows, by_qid, key=key),
        "ooc_scored": sum(1 for _, slug in rows if slug == NONE_SLUG),
    }


def delta_pp(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Разница колонок в пунктах: сколько дал resolve поверх сырого CV."""
    out: dict[str, Any] = {}
    for name, value in after.items():
        base = before.get(name)
        if isinstance(value, Mapping) and isinstance(base, Mapping):
            out[name] = delta_pp(base, value)
        elif name != "n" and isinstance(value, float) and isinstance(base, float):
            out[name] = round((value - base) * 100, 1)
    return out


def cv_vs_cv_ocr(
    items: Sequence[QueryItem],
    by_qid: Mapping[str, Mapping[str, Any]],
    gt_tokens: GtTokens,
    *,
    source: str | None = None,
    series_pool: bool | None = None,
) -> dict[str, Any]:
    """«Только CV» против «CV+OCR»: обе колонки и разница между ними в пунктах.

    Считается по одинаковой глубине (пять slug), чтобы разница была разницей ответа, а не
    длины списка. Кадры со сбоем в колонках остаются: пустая выдача — тоже ответ стенда.

    `with_read_fields` — сколько кадров ЭТОГО набора получили поля этикетки. Если ноль,
    прирост даёт один добор серий, и текст тут ни при чём; `series_pool` печатается рядом
    именно поэтому.
    """
    cv = column_metrics(items, by_qid, gt_tokens, key=CV_KEY)
    final = column_metrics(items, by_qid, gt_tokens, key=FINAL_KEY)
    outcomes = Counter(
        str(p.get("outcome")) for p in by_qid.values() if p.get("outcome") is not None
    )
    with_fields = sum(bool((by_qid.get(item.query_id) or {}).get("read_fields")) for item in items)
    return {
        "with_ocr": source,
        "queries": len(items),
        "with_read_fields": with_fields,
        "read_coverage": share(with_fields, len(items)),
        "series_pool": series_pool,
        "cv_only": cv,
        "cv_ocr": final,
        "delta_pp": delta_pp(cv, final),
        "outcomes": dict(sorted(outcomes.items())),
        "resolve_margin": margin_table(
            [float(p["resolve_margin"]) for p in by_qid.values() if p.get("resolve_margin")]
        ),
        "note": CV_OCR_NOTE,
        "by_winery_line_note": LINE_NOTE,
    }


def stage_latencies(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """p50 и p95 по шагам кадра: видно, что дороже — разжатие, модель или поиск."""
    stages: dict[str, list[float]] = {}
    for prediction in predictions:
        for stage, value in (prediction.get("timings_ms") or {}).items():
            if value is not None:
                stages.setdefault(stage, []).append(float(value))
    return {stage: latency_summary(values) for stage, values in sorted(stages.items())}


def split_of(item: QueryItem) -> str:
    """Половина разбиения, в которую попал запрос; не размечен — `unsplit`."""
    return str(item.meta.get("split") or "unsplit")


def by_split(
    items: Sequence[QueryItem],
    by_qid: Mapping[str, Mapping[str, Any]],
    gt_tokens: GtTokens,
    *,
    top_k: int,
) -> dict[str, Any]:
    """`by_card` отдельно по dev и test: подбирать пороги можно только на dev.

    Разрез считается всегда, даже когда прогон шёл по обеим половинам: иначе цифру,
    подобранную на dev, нечем было бы отделить от честной.
    """
    groups: dict[str, list[QueryItem]] = {}
    for item in items:
        groups.setdefault(split_of(item), []).append(item)
    return {
        name: {
            "queries": len(part),
            "by_card": rank_table(rank_columns(part, by_qid, gt_tokens)["by_card"], top_k=top_k),
        }
        for name, part in sorted(groups.items())
    }


def compute_metrics(
    items: Sequence[QueryItem],
    predictions: Sequence[Mapping[str, Any]],
    gt_tokens: GtTokens,
    *,
    top_k: int,
    with_ocr: str | None = None,
    series_pool: bool | None = None,
) -> dict[str, Any]:
    """Метрики прогона. Ничего не подбирается и не оптимизируется — только счёт.

    `cv_vs_cv_ocr` появляется, когда в предсказаниях есть колонка resolve: обе цифры —
    «только CV» и «CV+OCR» — и разница между ними лежат в одном блоке.
    """
    by_qid = {str(p["query_id"]): p for p in predictions}
    columns = rank_columns(items, by_qid, gt_tokens)
    failed = [p for p in predictions if p.get("status") != "ok"]
    resolved = any(FINAL_KEY in p for p in predictions)
    pools = {
        name: credited_slugs(items, gt_tokens, key_fn) for name, key_fn in RANK_COLUMNS.items()
    }
    return {
        **summarize(items),
        "predictions": len(predictions),
        "failed": len(failed),
        "errors": [f"{p['query_id']}: {p.get('error')}" for p in failed[:10]],
        "gt_tokens_loaded": len(gt_tokens),
        "by_card": rank_table(columns["by_card"], top_k=top_k),
        "by_visual_group": {
            **rank_table(columns["by_visual_group"], top_k=top_k),
            "credited_slugs": pools["by_visual_group"],
        },
        "by_winery_line": {
            **rank_table(columns["by_winery_line"], top_k=top_k),
            "credited_slugs": pools["by_winery_line"],
            "note": LINE_NOTE,
        },
        "by_split": by_split(items, by_qid, gt_tokens, top_k=top_k),
        "margin": margins_by_correctness(items, by_qid),
        "visual_groups": visual_group_metrics(items, by_qid, gt_tokens),
        "ties": tie_rate(items, by_qid),
        "unpublished": unpublished_answers(items, by_qid, gt_tokens),
        "out_of_catalog_scores": out_of_catalog_scores(items, by_qid),
        "cv_vs_cv_ocr": (
            cv_vs_cv_ocr(items, by_qid, gt_tokens, source=with_ocr, series_pool=series_pool)
            if resolved
            else None
        ),
        "latency_ms": stage_latencies(predictions),
    }


# ------------------------------------------------------------------ прогон
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj) + "\n", encoding="utf-8")


def run_retrieval(
    items: Sequence[QueryItem],
    pipeline: Pipeline,
    *,
    out_dir: Path,
    gt_tokens: GtTokens,
    run_info: dict[str, Any] | None = None,
    reads: Mapping[str, LabelFields] | None = None,
    with_ocr: str | None = None,
    seal_test: bool = False,
) -> dict[str, Any]:
    """Прогон набора: `predictions.jsonl` пишется построчно, в конце — `metrics.json`.

    `reads` — поля этикетки по `query_id` из прогона OCR; кадра нет в словаре — resolve
    получает пустые поля и опирается на один CV. `seal_test` — метрики `metrics.json` без
    кадров test, а метрики всего набора вместе с test — в конверт `SEALED_METRICS_FILE`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:  # строго по одному кадру: карту занимает одна модель
            record = search_item(item, pipeline, (reads or {}).get(item.query_id))
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            predictions.append(record)
    series_pool = None if pipeline.attrs is None else pipeline.cfg.series_pool
    sealed = [item for item in items if seal_test and split_of(item) == SEALED_SPLIT]
    open_items = [item for item in items if not (seal_test and split_of(item) == SEALED_SPLIT)]
    open_ids = {item.query_id for item in open_items}
    metrics = compute_metrics(
        open_items,
        [p for p in predictions if str(p["query_id"]) in open_ids],
        gt_tokens,
        top_k=pipeline.top_k,
        with_ocr=with_ocr,
        series_pool=series_pool,
    )
    metrics["run"] = {
        **(run_info or {}),
        "model": pipeline.index.meta.model,
        "index": pipeline.index.meta.model_dump(mode="json"),
        "top_k": pipeline.top_k,
        "per_slug": pipeline.per_slug,
        "resolve": (
            None
            if pipeline.attrs is None
            else {
                "attrs": pipeline.attrs.stats(),
                "cfg": pipeline.cfg.model_dump(mode="json"),
                # Нота лежит рядом с числами: паспорт прогона читают чаще, чем ноты внутри
                # блока сравнения, а ни одно из этих чисел на данных не подбиралось.
                "weights_note": WEIGHTS_NOTE,
            }
        ),
        "wall_s": round(time.perf_counter() - started, 3),
    }
    if sealed:
        envelope = compute_metrics(
            items,
            predictions,
            gt_tokens,
            top_k=pipeline.top_k,
            with_ocr=with_ocr,
            series_pool=series_pool,
        )
        envelope["warning"] = SEALED_WARNING
        envelope["run"] = metrics["run"]
        write_json(out_dir / SEALED_METRICS_FILE, envelope)
        metrics["sealed_test"] = {
            "file": SEALED_METRICS_FILE,
            "queries": len(sealed),
            "note": SEALED_NOTE,
        }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


# ------------------------------------------------------------------ набор: split и выборка
def assign_splits(items: Sequence[QueryItem], gt_tokens: GtTokens) -> list[QueryItem]:
    """Проставить каждому запросу dev/test, если набор не принёс своего разбиения.

    Ключ — `cluster_B` цели (`bench.datasets.split_for`): двойники одной винодельни не
    должны разъехаться по половинам, иначе правило, подобранное на одном, «проверялось» бы
    на почти том же другом. У `pairs` своего split нет вовсе — он назначается здесь.
    """
    out: list[QueryItem] = []
    for item in items:
        if item.meta.get("split"):
            out.append(item)
            continue
        record = gt_tokens.get(item.slug) if item.slug else None
        cluster = record.get("cluster_B") if record else None
        split = split_for(
            item.slug,
            item.query_id,
            cluster=None if cluster is None else str(cluster),
        )
        out.append(item.model_copy(update={"meta": {**item.meta, "split": split}}))
    return out


def default_split(queries: str) -> str:
    """Без `--split`: прогон идёт по dev, как у `bench.ocr_bench`.

    Test смотрят один раз и явно — иначе итоговая цифра окажется той самой, на которой
    подбирали пороги. `public` — три кадра организатора, дымовой тест: там делить нечего.
    """
    return "all" if queries == "public" else "dev"


def pick_split(items: Sequence[QueryItem], split: str) -> list[QueryItem]:
    """Половина набора; `all` — весь набор как есть."""
    if split == "all":
        return list(items)
    return [item for item in items if split_of(item) == split]


def take_limit(items: Sequence[QueryItem], limit: int, seed: int | None) -> list[QueryItem]:
    """Первые N запросов или случайные N с фиксированным сидом.

    Префикс `benchmark.json` отсортирован по `wine_id`, то есть первые записи — несколько
    виноделен подряд. Это проверка стенда, а не замер; `--sample-seed` даёт выборку,
    которую хотя бы можно назвать выборкой.
    """
    if seed is None:
        return list(items[:limit])
    picked = random.Random(seed).sample(range(len(items)), k=min(limit, len(items)))
    return [items[i] for i in sorted(picked)]


def split_shares(items: Sequence[QueryItem]) -> dict[str, int]:
    return dict(sorted(Counter(split_of(item) for item in items).items()))


# ------------------------------------------------------------------ CLI
def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.retrieval",
        description="Замер поиска вина по фотографии среди эталонов каталога.",
    )
    parser.add_argument("--index", type=Path, default=settings.data_dir / "index" / "visual.npz")
    parser.add_argument("--queries", choices=QUERY_SETS, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", type=Path, required=True, help="папка прогона, runs/<имя>/")
    parser.add_argument("--limit", type=int, help="первые N запросов (проверка стенда)")
    parser.add_argument(
        "--sample-seed",
        type=int,
        help="брать --limit запросов случайно с этим сидом, а не первыми по порядку файла",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        help="по умолчанию прогон идёт по dev (public — по всему набору), как в "
        "bench.ocr_bench: test смотрят один раз и явно, для отчёта. Разрез по половинам "
        "попадает в метрики при любом выборе; при all метрики test запечатаны в "
        f"{SEALED_METRICS_FILE}",
    )
    parser.add_argument(
        "--unseal-test",
        action="store_true",
        help="с --split all: метрики test в metrics.json и в консоли вместе с остальными. Не для "
        "прогонов, которые идут во вход обучения",
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="none",
        help="как искать бутылку на кадре: детектор CV или окна по долям сторон",
    )
    parser.add_argument(
        "--per-slug",
        choices=get_args(Aggregation),
        default="max",
        help="свести векторы slug: max, mean или zmax — max по косинусам, выровненным по парам "
        "«окно × вид» (app.features.index.align_pairs; так считает сервис)",
    )
    parser.add_argument(
        "--device", default=None, help=f"по умолчанию SVS_DEVICE ({settings.device})"
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gt-tokens", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl"
    )
    parser.add_argument(
        "--with-ocr",
        type=Path,
        help="predictions.jsonl прогона bench.ocr_bench: поля этикетки для тех же query_id; "
        "включает resolve и колонку «CV+OCR» в метриках",
    )
    parser.add_argument(
        "--abstain",
        choices=get_args(AbstainMode),
        default=DEFAULT_CONFIG.abstain,
        help="правило отказа S10: off — не отказываться, ooc_only — отказ по трём условиям",
    )
    parser.add_argument(
        "--no-series-pool",
        action="store_true",
        help="не добирать в пул resolve членов серии, которых CV не поднял в top-K",
    )
    parser.add_argument("--pairs-json", type=Path, help="pairs: свой benchmark.json")
    parser.add_argument("--pairs-images", type=Path, help="pairs: своя папка снимков")
    parser.add_argument("--rwl-dir", type=Path, help="rwl: папка манифестов")
    parser.add_argument("--rwl-images", type=Path, help="rwl: папка кадров")
    parser.add_argument("--overwrite", action="store_true", help="перезаписать прогон в --out")
    return parser


def build_pipeline(
    args: argparse.Namespace,
    settings: Settings,
    embedder: Embedder | None = None,
    attrs: CatalogAttrs | None = None,
) -> Pipeline:
    """Индекс, модель, рамка цели и — при `--with-ocr` — признаки каталога для resolve.

    Любая неготовность — `BenchSetupError`.
    """
    if not args.index.is_file():
        raise BenchSetupError(f"нет индекса эталонов: {args.index} (scripts/build_index.py)")
    try:
        index = VisualIndex.load(args.index)
    except (IndexMismatch, OSError, ValueError) as exc:
        raise BenchSetupError(f"индекс {args.index} не читается: {exc}") from exc
    if not len(index):
        raise BenchSetupError(f"индекс {args.index} пуст")
    if embedder is None:
        # Модель берётся из паспорта индекса: чужой моделью его всё равно не искать.
        embedder = SiglipEmbedder(
            index.meta.model, args.device, args.dtype, batch_size=args.batch_size
        )
    try:
        index.check_model(getattr(embedder, "model_name", ""))
    except IndexMismatch as exc:
        raise BenchSetupError(str(exc)) from exc
    try:
        dim = int(embedder.dim)
    except ModelNotAvailable as exc:
        raise BenchSetupError(f"модель недоступна: {exc}") from exc
    if dim != index.dim:
        raise BenchSetupError(f"модель даёт вектор {dim}, а в индексе {index.dim}")
    target = (
        detector_target_fn(args.device or settings.device) if args.target == "detector" else None
    )
    cfg = DEFAULT_CONFIG.model_copy(
        update={
            "top_k": args.top_k,
            "abstain": getattr(args, "abstain", DEFAULT_CONFIG.abstain),
            "series_pool": not getattr(args, "no_series_pool", False),
        }
    )
    return Pipeline(
        index=index,
        embedder=embedder,
        target=target,
        top_k=args.top_k,
        per_slug=args.per_slug,
        attrs=attrs,
        cfg=cfg,
    )


def main(argv: Sequence[str] | None = None, embedder: Embedder | None = None) -> int:
    settings = get_settings()
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        parser.error("--top-k должен быть больше нуля")
    gt_tokens = load_gt_tokens(args.gt_tokens) if args.gt_tokens.is_file() else {}
    if not gt_tokens:
        print(f"предупреждение: нет эталонных токенов {args.gt_tokens}", file=sys.stderr)
    reads: dict[str, LabelFields] | None = None
    attrs: CatalogAttrs | None = None
    if args.with_ocr is not None:
        if not gt_tokens:
            print(
                f"--with-ocr требует разметки каталога: нет {args.gt_tokens} "
                "(scripts/build_gt_tokens.py)",
                file=sys.stderr,
            )
            return 3
        # Признаки карточек берутся из тех же записей, что и метрики: второго файла не нужно.
        attrs = CatalogAttrs.from_records(gt_tokens.values())
    try:
        if args.with_ocr is not None:
            reads = load_read_fields(args.with_ocr)
        items = load_queries(
            args.queries,
            settings=settings,
            pairs_json=args.pairs_json,
            pairs_images=args.pairs_images,
            rwl_dir=args.rwl_dir,
            rwl_images=args.rwl_images,
        )
    except DatasetError as exc:
        print(f"ошибка набора: {exc}", file=sys.stderr)
        return 2
    split = args.split or default_split(args.queries)
    items = assign_splits(items, gt_tokens)
    whole = len(items)
    items = pick_split(items, split)
    if not items:
        print(f"в наборе {args.queries} нет запросов половины {split}", file=sys.stderr)
        return 1
    if split != "all":
        print(
            f"--split {split}: {len(items)} запросов из {whole}"
            + ("" if args.split else " (по умолчанию; весь набор — --split all)"),
            file=sys.stderr,
        )
    if args.limit:
        items = take_limit(items, args.limit, args.sample_seed)
        if args.sample_seed is None:
            print(
                f"--limit {args.limit} без --sample-seed: взят префикс набора "
                "(проверка стенда, не замер)",
                file=sys.stderr,
            )
    if reads is not None:
        covered = sum(item.query_id in reads for item in items)
        if not covered:
            print(
                f"--with-ocr {args.with_ocr}: ни один query_id набора {args.queries} не "
                f"встретился в прогоне OCR ({len(reads)} записей). Колонка «CV+OCR» тогда "
                "мерила бы не текст, а добор серий",
                file=sys.stderr,
            )
            return 2
        if covered < len(items):
            print(
                f"--with-ocr: поля этикетки есть у {covered} кадров из {len(items)}",
                file=sys.stderr,
            )
    if (args.out / "predictions.jsonl").exists() and not args.overwrite:
        print(f"{args.out} уже содержит прогон: другое имя или --overwrite", file=sys.stderr)
        return 2
    seal = seals_test(split, unseal=args.unseal_test)
    sealed = sum(split_of(item) == SEALED_SPLIT for item in items) if seal else 0
    if sealed:
        print(
            f"--split all: метрики {sealed} запросов test запечатаны в {SEALED_METRICS_FILE}; "
            "metrics.json и вывод ниже — без них",
            file=sys.stderr,
        )
    missing = missing_images(items)
    if missing or not items:
        print("набор не готов:\n" + "\n".join(missing[:20] or ["нет кадров"]), file=sys.stderr)
        return 1
    try:
        pipeline = build_pipeline(args, settings, embedder, attrs)
    except BenchSetupError as exc:
        print(f"стенд не готов: {exc}", file=sys.stderr)
        return 3

    run_info = {
        "queries_set": args.queries,
        "notes": notes_for(args.queries),
        "index_path": str(args.index),
        "target": args.target,
        "with_ocr": None if args.with_ocr is None else str(args.with_ocr),
        "with_ocr_fields": None if reads is None else len(reads),
        "with_ocr_covered": (
            None if reads is None else sum(item.query_id in reads for item in items)
        ),
        "split": split,
        "test_sealed": bool(sealed),
        "split_shares": split_shares(items),
        "split_of_set": whole,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "limit_note": (
            None
            if not args.limit
            else (
                "случайная выборка с сидом"
                if args.sample_seed is not None
                else "префикс набора по порядку файла: проверка стенда, не замер"
            )
        ),
        "device": args.device or settings.device,
        "dtype": args.dtype,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    metrics = run_retrieval(
        items,
        pipeline,
        out_dir=args.out,
        gt_tokens=gt_tokens,
        run_info=run_info,
        reads=reads,
        with_ocr=None if args.with_ocr is None else str(args.with_ocr),
        seal_test=seal,
    )
    keys = (
        "queries",
        "failed",
        "by_card",
        "by_visual_group",
        "by_winery_line",
        "by_split",
        "margin",
        "visual_groups",
        "ties",
        "unpublished",
        "out_of_catalog_scores",
        "cv_vs_cv_ocr",
        "sealed_test",
    )
    print_json(
        {k: metrics[k] for k in keys if k in metrics} | {"latency_ms": metrics["latency_ms"]}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
