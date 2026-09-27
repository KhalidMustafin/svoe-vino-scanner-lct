"""Обучаемый resolve: групповая K-fold на dev, выбор L2, отчёт out-of-fold и модель в JSON.

    python -m bench.train_resolve \\
        --cv runs/cvall-s2so400m-pairs/predictions.jsonl \\
        --cv runs/cvall-s2so400m-pairs_phone/predictions.jsonl \\
        --ocr vlm35:pairs=runs/ocr-pairs-vlm35/predictions.jsonl \\
        --ocr vlm35:pairs_phone=runs/ocr-pairsphone-vlm35/predictions.jsonl \\
        --ocr rapid:pairs=runs/ocr-pairs-rapid/predictions.jsonl \\
        --ocr rapid:pairs_phone=runs/ocr-pairsphone-rapid/predictions.jsonl \\
        --split dev --folds 5 --top-k 20 --out runs/resolve-s2so400m/

Вход — готовые прогоны: кандидаты CV (`bench.retrieval`) и поля этикетки (`bench.ocr_bench`).
Прогон OCR привязан к набору явно (`--ocr читатель:набор=путь`): у `pairs` и `pairs_phone`
одинаковые `query_id` и эталоны, и сопоставление идёт по паре «набор + query_id». Набор записи
CV — `meta.set`. Перепутанные файлы ловит сверка кадра: запрос порченого набора
(`meta.phone_seed` у записи CV) должен быть прочитан с порченого кадра, а не с исходного
студийного (`image_path` записи OCR ≠ `image_path` CV), запрос исходного набора — с того же
кадра. Половина записи OCR (`split`) сверяется с половиной запроса.

Половина задаётся тем же ключом, что у стенда поиска: `meta.split` записи, а без него —
`bench.datasets.split_for` по `cluster_B` эталона. У `pairs` и `pairs_phone` объявленная
половина сверяется с хэшем. Строка другой половины декодируется как JSON (иначе не узнать
`slug`), но из неё берутся только `query_id`, `slug` и `meta`: ни кандидатов, ни полей
этикетки, ни признаков.

Что делается на dev (и только на нём):

1. Признаки пар — `app.resolve.features`, кандидаты — top-K CV. Запрос без верного slug в
   top-K в обучение не входит, но в отчёте остаётся (промах) и считается в «потолок K».
   Знаки весов признаков с очевидным направлением держатся при обучении
   (`features.feature_sign`); `--free-signs` снимает ограничение — только для сравнения.
2. Фолды — групповые по `cluster_B` эталона (вне кластеров — по slug): кадры одной линейки
   и один и тот же кадр из `pairs` и `pairs_phone` всегда в одном фолде.
3. Отчёт out-of-fold — ВЛОЖЕННЫЙ: для каждого внешнего фолда L2 (и, с `--select-groups`,
   группы признаков) выбираются внутренней групповой K-fold на остальных фолдах, только по
   средней точности top-1; температура softmax — по внутренним out-of-fold счётам. Поэтому
   цифры «обученный» не подсмотрены выбором гиперпараметров.
4. Модель для сервиса — на всём dev с гиперпараметрами, выбранными обычной K-fold на всём
   dev (таблица сетки в `metrics.json`), температура — по её out-of-fold счётам.

Колонки отчёта для каждого варианта читателя и набора: «только CV», «ручной rerank»
(`app.resolve.rerank` с настройками `bench.retrieval --with-ocr`) и «обученный»; «исправил /
сломал» относительно CV; веса модели и контрасты групп; надёжностная таблица вероятности
top-1 по децилям.

Ничьи CV: у визуальных двойников счёт top-1 и top-2 бывает равен до бита. Тогда CV-top1 —
первый по порядку файла, ручной rerank при равном счёте берёт меньший slug, и «исправил /
сломал» на таком запросе — жребий. Такие запросы помечены (`cv_tie_k`), посчитаны отдельно
(«из них в ничьих») и дают CV честную долю 1/k (`cv_only.top1_tie_fair`).

Варианты: имя читателя; `both` — признаки всех читателей с префиксами (ручной rerank
получает поля, слитые по порядку `--ocr`: списки подряд, одиночные поля — от первого);
`none` — обучаемый реранк только по CV, чтобы отделить вклад текста от вклада обучения.

`--final-test` (по умолчанию выключен): после всего, что выше, читается test, и модели,
обученные на всём dev, оцениваются на нём ОДИН раз с теми же колонками. В `metrics.json` —
`"test_used": true` и предупреждение. Там же `cross_split_twins`: пары «запрос dev — запрос
test» с одним `query_id` без года («…-2018» и «…-2020»), чьи эталоны лежат в разных
`cluster_B` и потому разошлись по половинам. Это почти тот же продукт по обе стороны
разбиения — известная утечка, её надо назвать рядом с цифрой test. Без флага строки test не
читаются дальше `query_id`, `slug` и `meta`.

Коды выхода: 0 — готово; 2 — ошибка аргументов или данных; 3 — нет разметки каталога.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from app.config import get_settings
from app.features.contracts import Candidate, VisualResult
from app.reading.contracts import LabelFields
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import (
    AGREEMENT_FEATURES,
    DEFAULT_GROUPS,
    FEATURE_VERSION,
    REQUIRED_GROUPS,
    TEXT_GROUPS,
    CvQuery,
    FeatureOptions,
    TextRead,
    check_groups,
    cv_query,
    feature_names,
    feature_signs,
    group_of,
    query_features,
)
from app.resolve.learned import (
    LOSSES,
    LogisticRanker,
    QueryBlocks,
    fit_temperature,
    top1_probability,
)
from app.resolve.rerank import DEFAULT_CONFIG, rerank
from bench.datasets import DatasetError, split_for
from bench.metrics import dumps, load_gt_tokens
from bench.retrieval import fields_of_record

TRAIN_SPLIT = "dev"
TEST_SPLIT = "test"
DEFAULT_FOLDS = 5
DEFAULT_TOP_K = 20
#: Шкала L2 — средний лосс на пару (pointwise) или на запрос (listwise): у pointwise разумные
#: значения на порядки меньше, поэтому сетка широкая.
DEFAULT_L2: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BOTH = "both"
NONE = "none"
#: Наборы, у которых половина — хэш `cluster_B`: объявленная в записи сверяется с ним.
HASH_SPLIT_SETS = frozenset({"pairs", "pairs_phone"})
COLUMNS = ("cv_only", "manual", "learned")
CALIBRATION_BINS = 10
_READER_RE = re.compile(r"[A-Za-z0-9_-]+")
#: Год в конце `query_id` пары: «…-bryut-rozovoe-2018».
_YEAR_SUFFIX_RE = re.compile(r"-(?:19|20)\d\d$")

TEST_WARNING = (
    "ИСПОЛЬЗОВАН TEST: модели обучены на всём dev и оценены на test один раз. После этого "
    "менять признаки, сетку L2 или пороги по этим цифрам нельзя — иначе test станет dev"
)
NESTED_NOTE = (
    "out-of-fold вложенный: L2 (и группы признаков) каждого внешнего фолда выбраны внутренней "
    "групповой K-fold на остальных фолдах, температура — по внутренним out-of-fold счётам"
)
GRID_NOTE = (
    "сетка на всём dev обычной групповой K-fold: по ней выбраны гиперпараметры модели для "
    "сервиса. Точность выбранной строки завышена выбором — честная цифра в oof"
)
MANUAL_NOTE = (
    "ручной rerank — app.resolve.rerank с DEFAULT_CONFIG и top_k запуска (как bench.retrieval "
    "--with-ocr): серии добираются в пул, отказ выключен; у варианта both поля слиты"
)
TIES_NOTE = (
    "ничья CV-top1 — счёт top-1 и top-2 равен до бита: CV-top1 там — порядок файла, ручной "
    "rerank берёт меньший slug, «исправил/сломал» на таком запросе — жребий. top1_tie_fair "
    "засчитывает CV 1/k, если эталон среди k равных"
)
TWINS_NOTE = (
    "одна основа query_id без года по обе стороны разбиения, эталоны в разных cluster_B: почти "
    "тот же продукт в dev и test. Разбиение не меняется (оно общее со стендом поиска), пара "
    "называется рядом с цифрой test"
)


# ------------------------------------------------------------------ входные данные
@dataclass(frozen=True, slots=True)
class OcrSpec:
    reader: str
    set_name: str
    path: Path


def parse_ocr_spec(text: str) -> OcrSpec:
    """`читатель:набор=путь` → `OcrSpec`."""
    left, sep, path = text.partition("=")
    reader, colon, set_name = left.partition(":")
    if not sep or not colon or not path or not set_name:
        raise ValueError(f"--ocr {text!r}: ожидается читатель:набор=путь")
    if not _READER_RE.fullmatch(reader) or reader in (BOTH, NONE):
        raise ValueError(f"--ocr {text!r}: имя читателя — латиница, цифры, _ и -, не both/none")
    return OcrSpec(reader=reader, set_name=set_name, path=Path(path))


def parse_losses(text: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    unknown = [value for value in values if value not in LOSSES]
    if not values or unknown:
        raise ValueError(f"--loss {text!r}: ожидается одно или оба из {', '.join(LOSSES)}")
    return values


def parse_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"--l2 {text!r}: ожидается список чисел через запятую") from exc
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError(f"--l2 {text!r}: нужны неотрицательные конечные числа")
    return tuple(sorted(set(values)))


@dataclass
class Query:
    """Один запрос: кандидаты CV, чтения этикетки и эталон — только для цели и фолда."""

    set_name: str
    query_id: str
    slug: str | None
    split: str
    group: str
    cv: CvQuery
    reads: dict[str, TextRead] = field(default_factory=dict)
    ocr_status: dict[str, str] = field(default_factory=dict)
    image_path: str | None = None  # кадр, который искал CV (для порченых — исходный)
    derived: bool = False  # кадр испорчен на лету (`meta.phone_seed`): OCR читал копию

    @property
    def key(self) -> str:
        return f"{self.set_name}:{self.query_id}"


def group_key(slug: str | None, query_id: str, gt_tokens: Mapping[str, Mapping[str, Any]]) -> str:
    """Ключ фолда — тот же, что у разбиения: `cluster_B` эталона, иначе slug, иначе запрос."""
    record = gt_tokens.get(slug) if slug else None
    cluster = record.get("cluster_B") if record else None
    if cluster is not None:
        return f"cluster:{cluster}"
    return f"slug:{slug}" if slug else f"query:{query_id}"


def record_split(
    set_name: str,
    query_id: str,
    slug: str | None,
    meta: Mapping[str, Any],
    gt_tokens: Mapping[str, Mapping[str, Any]],
) -> str:
    """Половина запроса по правилу `bench.retrieval.assign_splits`."""
    record = gt_tokens.get(slug) if slug else None
    cluster = record.get("cluster_B") if record else None
    hashed = split_for(slug, query_id, cluster=None if cluster is None else str(cluster))
    declared = meta.get("split")
    if not declared:
        return hashed
    if set_name in HASH_SPLIT_SETS and declared != hashed:
        raise DatasetError(
            f"{set_name}:{query_id}: в записи split={declared}, а по cluster_B выходит {hashed} "
            "— разметка каталога не та, что у прогона CV"
        )
    return str(declared)


def _json_lines(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise DatasetError(f"нет файла: {path}")
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: не JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetError(f"{path}:{number}: ожидается объект")
            yield number, record


def load_cv(
    path: Path,
    gt_tokens: Mapping[str, Mapping[str, Any]],
    *,
    splits: frozenset[str],
    top_k: int,
) -> tuple[list[Query], dict[str, Any]]:
    """Запросы прогона CV нужных половин. Из строк других половин — только query_id, slug, meta."""
    queries: list[Query] = []
    other = 0
    seen: set[str] = set()
    for number, record in _json_lines(path):
        query_id = str(record.get("query_id") or "")
        meta = record.get("meta") or {}
        if not query_id or not isinstance(meta, Mapping) or not meta.get("set"):
            raise DatasetError(f"{path}:{number}: нужны query_id и meta.set")
        set_name = str(meta["set"])
        slug = str(record["slug"]) if record.get("slug") else None
        split = record_split(set_name, query_id, slug, meta, gt_tokens)
        if split not in splits:
            other += 1
            continue
        # Отсюда запись нужной половины читается целиком.
        key = f"{set_name}:{query_id}"
        if key in seen:
            raise DatasetError(f"{path}:{number}: повтор запроса {key}")
        seen.add(key)
        try:
            cv = cv_query(record, top_k=top_k) if record.get("status") == "ok" else CvQuery((), 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"{path}:{number}: кандидаты CV не читаются: {exc}") from exc
        image = record.get("image_path")
        queries.append(
            Query(
                set_name=set_name,
                query_id=query_id,
                slug=slug,
                split=split,
                group=group_key(slug, query_id, gt_tokens),
                cv=cv,
                image_path=str(image) if image else None,
                derived=meta.get("phone_seed") is not None,
            )
        )
    info = {
        "path": str(path),
        "sha1": file_sha1(path),
        "queries": len(queries),
        "sets": dict(sorted(Counter(q.set_name for q in queries).items())),
        "rows_other_splits": other,
    }
    return queries, info


def same_image(a: str, b: str) -> bool:
    """Один и тот же файл кадра: путь совпал, либо совпали имя файла и имя его папки.

    Второе правило — для переезда набора (абсолютный путь в одном прогоне, другой корень в
    другом): `roskachestvo/X.webp` остаётся собой, а `pairs-phone-images/X.jpg` — нет.
    """
    pa, pb = Path(a), Path(b)
    if os.path.normcase(os.path.abspath(pa)) == os.path.normcase(os.path.abspath(pb)):
        return True
    return pa.name.casefold() == pb.name.casefold() and (
        pa.parent.name.casefold() == pb.parent.name.casefold()
    )


def check_ocr_frame(spec: OcrSpec, number: int, record: Mapping[str, Any], query: Query) -> None:
    """Запись OCR — о том же кадре того же набора и той же половины, что запрос CV.

    У `pairs` и `pairs_phone` одинаковы и `query_id`, и эталон, поэтому набор выдаёт только
    кадр: порченый запрос CV ищет по исходному снимку (порча на лету, `meta.phone_seed`), а OCR
    читает сохранённую порченую копию. Значит, у порченого набора путь кадра OCR обязан
    отличаться от пути CV, у исходного — совпадать. Путей нет (синтетика) — сверки нет.
    """
    declared = record.get("split")
    if declared and query.split and str(declared) != query.split:
        raise DatasetError(
            f"{spec.path}:{number}: у {query.query_id} в прогоне OCR split={declared}, а у "
            f"запроса {query.split} — прогон OCR не от этого набора или другой разметки?"
        )
    ocr_meta = record.get("meta") if isinstance(record.get("meta"), Mapping) else {}
    image = record.get("image_path")
    if ocr_meta.get("phone_seed") is not None:
        ocr_derived: bool | None = True
    elif image and query.image_path:
        ocr_derived = not same_image(str(image), query.image_path)
    else:
        ocr_derived = None
    if ocr_derived is None or ocr_derived == query.derived:
        return
    if query.derived:
        problem = f"прочитан исходный кадр {image}, а запрос набора {spec.set_name} — порченый"
    else:
        problem = f"прочитан кадр {image}, а CV искал по {query.image_path}"
    raise DatasetError(
        f"{spec.path}:{number}: {query.query_id}: {problem} — перепутаны прогоны OCR наборов?"
    )


def reader_keys_of(record: Mapping[str, Any]) -> list[str]:
    """Читатели записи OCR без кадра: `id@версия|params_hash` из полей `readers[].reader`/`reader`.

    Полная строка читателя — `vlm@qwen3.5:4b|f3a017317f04|<sha1 кадра>|full|1024`: первые два
    звена называют модель и её параметры (промпт, размер, `num_predict`…), остальные — кадр.
    Сервис сверяет свой читатель с этим списком из `meta.reader_keys` модели.
    """
    names: list[str] = []
    readers = record.get("readers")
    if isinstance(readers, list):
        names.extend(
            str(item["reader"])
            for item in readers
            if isinstance(item, Mapping) and item.get("reader")
        )
    if not names and record.get("reader"):
        names.append(str(record["reader"]))
    return ["|".join(name.split("|")[:2]) for name in names]


def load_ocr(spec: OcrSpec, wanted: Mapping[str, Query]) -> tuple[dict[str, Any], int]:
    """Чтения одного прогона OCR для запросов `wanted` (по `query_id` внутри набора).

    Из строк чужих запросов (другая половина, другой набор) берётся только `query_id`.
    Сбой чтения (`status=error`) — чтение без полей, как у `bench.retrieval.load_read_fields`.
    Запись сверяется с запросом: эталон, половина и кадр (`check_ocr_frame`). Читатели
    сопоставленных записей собираются в `reader_keys` — они попадут в `meta` модели.
    """
    matched = 0
    other = 0
    keys: set[str] = set()
    for number, record in _json_lines(spec.path):
        query_id = str(record.get("query_id") or "")
        query = wanted.get(query_id)
        if query is None:
            other += 1
            continue
        if spec.reader in query.reads:
            raise DatasetError(f"{spec.path}:{number}: повтор query_id {query_id}")
        if record.get("slug") and query.slug and record["slug"] != query.slug:
            raise DatasetError(
                f"{spec.path}:{number}: у {query_id} эталон {record['slug']}, а в прогоне CV "
                f"{query.slug} — прогон OCR не от набора {spec.set_name}?"
            )
        check_ocr_frame(spec, number, record, query)
        status = str(record.get("status") or "")
        fields: LabelFields | None = None
        raw: str | None = None
        if status != "error":
            try:
                fields = fields_of_record(record)
            except ValidationError as exc:
                raise DatasetError(
                    f"{spec.path}:{number}: поля этикетки не читаются: {exc}"
                ) from exc
            raw = record.get("raw_text") if isinstance(record.get("raw_text"), str) else None
        query.reads[spec.reader] = TextRead(fields=fields, raw_text=raw)
        query.ocr_status[spec.reader] = status or "unknown"
        keys.update(reader_keys_of(record))
        matched += 1
    info = {
        "reader": spec.reader,
        "set": spec.set_name,
        "path": str(spec.path),
        "sha1": file_sha1(spec.path),
        "matched": matched,
        "wanted": len(wanted),
        "rows_other_queries": other,
        "reader_keys": sorted(keys),
    }
    return info, matched


def reader_keys_by_name(ocr_info: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """`{имя читателя: [id@версия|params_hash, …]}` по всем его прогонам OCR — для `meta`."""
    out: dict[str, set[str]] = {}
    for info in ocr_info:
        out.setdefault(str(info["reader"]), set()).update(info.get("reader_keys") or ())
    return {reader: sorted(keys) for reader, keys in out.items() if keys}


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(
    cv_paths: Sequence[Path],
    ocr_specs: Sequence[OcrSpec],
    gt_tokens: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    top_k: int,
) -> tuple[list[Query], dict[str, Any]]:
    """Все запросы одной половины: CV из всех `--cv`, чтения из всех `--ocr`."""
    queries: list[Query] = []
    cv_info = []
    for path in cv_paths:
        part, info = load_cv(path, gt_tokens, splits=frozenset({split}), top_k=top_k)
        queries.extend(part)
        cv_info.append(info)
    keys = Counter(q.key for q in queries)
    repeated = [key for key, count in keys.items() if count > 1]
    if repeated:
        raise DatasetError(f"запрос в двух прогонах CV: {', '.join(sorted(repeated)[:5])}")
    by_set: dict[str, dict[str, Query]] = {}
    for query in queries:
        by_set.setdefault(query.set_name, {})[query.query_id] = query
    ocr_info = []
    for spec in ocr_specs:
        wanted = by_set.get(spec.set_name)
        if wanted is None:
            raise DatasetError(
                f"--ocr {spec.reader}:{spec.set_name}: набора {spec.set_name} нет в прогонах CV "
                f"(есть: {', '.join(sorted(by_set))})"
            )
        info, _ = load_ocr(spec, wanted)
        ocr_info.append(info)
    queries.sort(key=lambda q: (q.set_name, q.query_id))
    return queries, {"cv": cv_info, "ocr": ocr_info}


# ------------------------------------------------------------------ варианты
def reader_order(specs: Sequence[OcrSpec]) -> list[str]:
    return list(dict.fromkeys(spec.reader for spec in specs))


def variant_readers(variant: str, readers: Sequence[str]) -> tuple[str, ...]:
    if variant == NONE:
        return ()
    if variant == BOTH:
        return tuple(readers)
    return (variant,)


def check_variants(
    variants: Sequence[str], specs: Sequence[OcrSpec], sets: Iterable[str]
) -> list[str]:
    """Каждому читателю варианта нужен прогон OCR для каждого набора CV."""
    readers = reader_order(specs)
    have = {(spec.reader, spec.set_name) for spec in specs}
    out = []
    for variant in variants:
        if variant == BOTH and len(readers) < 2:
            raise ValueError("вариант both требует двух и более читателей в --ocr")
        if variant not in (BOTH, NONE) and variant not in readers:
            raise ValueError(f"вариант {variant}: нет --ocr {variant}:<набор>=<путь>")
        for reader in variant_readers(variant, readers):
            for set_name in sets:
                if (reader, set_name) not in have:
                    raise ValueError(f"вариант {variant}: нет --ocr {reader}:{set_name}=<путь>")
        out.append(variant)
    return list(dict.fromkeys(out))


def default_variants(specs: Sequence[OcrSpec]) -> list[str]:
    readers = reader_order(specs)
    return [*readers, *([BOTH] if len(readers) > 1 else []), NONE]


def merge_fields(values: Sequence[LabelFields | None]) -> LabelFields | None:
    """Поля нескольких читателей для ручного rerank: списки подряд, одиночные — от первого."""
    present = [value for value in values if value is not None]
    if len(present) <= 1:
        return present[0] if present else None

    def listed(name: str) -> list[Any]:
        seen: set[str] = set()
        out = []
        for fields in present:
            for evidence in getattr(fields, name):
                key = str(evidence.value)
                if key not in seen:
                    seen.add(key)
                    out.append(evidence)
        return out

    def first(name: str) -> Any:
        return next((getattr(f, name) for f in present if getattr(f, name) is not None), None)

    return LabelFields(
        producer=listed("producer"),
        cuvee=listed("cuvee"),
        grapes=listed("grapes"),
        sugar=listed("sugar"),
        vintage=first("vintage"),
        serial=listed("serial"),
        abv=first("abv"),
        color=first("color"),
        unmatched=listed("unmatched"),
        is_wine_label=any(f.is_wine_label for f in present),
    )


def manual_top5(query: Query, readers: Sequence[str], attrs: CatalogAttrs, top_k: int) -> list[str]:
    """Ответ ручного rerank — с той же настройкой, что `bench.retrieval --with-ocr`."""
    if not query.cv.candidates:
        return []
    visual = VisualResult(
        candidates=[
            Candidate(slug=c.slug, score=min(1.0, max(0.0, c.score)), view="full", rank=c.rank)
            for c in query.cv.candidates
        ],
        margin=max(0.0, query.cv.margin),
    )
    fields = merge_fields([query.reads[r].fields for r in readers if r in query.reads])
    result = rerank(visual, fields, attrs, cfg=DEFAULT_CONFIG.model_copy(update={"top_k": top_k}))
    return [slug for slug, _ in result.top5]


# ------------------------------------------------------------------ матрица признаков
@dataclass
class Design:
    """Признаки всех пар варианта: строки запроса идут подряд, кандидаты — в порядке CV."""

    names: list[str]
    X: np.ndarray
    y: np.ndarray
    row_query: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    slugs: list[tuple[str, ...]]
    positive: np.ndarray  # индекс строки верного кандидата внутри запроса, −1 — нет в top-K
    groups: list[str]

    def rows_of(self, query_idx: Sequence[int]) -> np.ndarray:
        parts = [np.arange(self.starts[q], self.ends[q]) for q in query_idx]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def columns(self, names: Sequence[str]) -> np.ndarray:
        index = {name: i for i, name in enumerate(self.names)}
        return np.asarray([index[name] for name in names], dtype=np.int64)


def build_design(
    queries: Sequence[Query],
    readers: Sequence[str],
    attrs: CatalogAttrs,
    names: Sequence[str],
    *,
    top_k: int,
    options: FeatureOptions,
) -> Design:
    blocks: list[np.ndarray] = []
    targets: list[float] = []
    row_query: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    slugs: list[tuple[str, ...]] = []
    positive: list[int] = []
    for qi, query in enumerate(queries):
        reads = {reader: query.reads.get(reader, TextRead()) for reader in readers}
        features = query_features(query.cv, reads, attrs, top_k=top_k, options=options)
        starts.append(len(targets))
        if features.rows:
            blocks.append(features.matrix(names))
            hits = [slug == query.slug for slug in features.slugs]
            targets.extend(float(hit) for hit in hits)
            row_query.extend([qi] * len(hits))
            positive.append(hits.index(True) if any(hits) else -1)
        else:
            positive.append(-1)
        ends.append(len(targets))
        slugs.append(features.slugs)
    X = np.vstack(blocks) if blocks else np.zeros((0, len(names)))
    return Design(
        names=list(names),
        X=X,
        y=np.asarray(targets),
        row_query=np.asarray(row_query, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        ends=np.asarray(ends, dtype=np.int64),
        slugs=slugs,
        positive=np.asarray(positive, dtype=np.int64),
        groups=[query.group for query in queries],
    )


# ------------------------------------------------------------------ фолды и подбор
def group_folds(groups: Sequence[str], n_folds: int) -> np.ndarray:
    """Групповые фолды: группа целиком в одном фолде, фолды выровнены по числу запросов.

    Порядок не зависит от порядка строк: группы идут по убыванию размера, при равенстве —
    по sha256 имени; каждая кладётся в самый лёгкий фолд.
    """
    if n_folds < 2:
        raise ValueError("фолдов должно быть не меньше двух")
    counts = Counter(groups)
    if len(counts) < n_folds:
        raise ValueError(f"групп {len(counts)} меньше, чем фолдов {n_folds}")
    order = sorted(
        counts, key=lambda g: (-counts[g], hashlib.sha256(g.encode("utf-8")).hexdigest())
    )
    load = [0] * n_folds
    fold_of: dict[str, int] = {}
    for group in order:
        fold = min(range(n_folds), key=lambda i: (load[i], i))
        fold_of[group] = fold
        load[fold] += counts[group]
    return np.asarray([fold_of[group] for group in groups], dtype=np.int64)


def fit_on(
    design: Design,
    query_idx: Sequence[int],
    names: Sequence[str],
    l2: float,
    loss: str,
    *,
    signed: bool = True,
) -> LogisticRanker | None:
    """Модель на запросах `query_idx`, у которых верный кандидат есть в top-K.

    `signed` — держать знаки весов (`features.feature_sign`); без него веса свободны.
    """
    usable = [q for q in query_idx if design.positive[q] >= 0]
    if not usable:
        return None
    rows = design.rows_of(usable)
    cols = design.columns(names)
    signs = feature_signs(names) if signed else None
    return LogisticRanker(names, l2=l2, loss=loss, signs=signs).fit(  # type: ignore[arg-type]
        design.X[np.ix_(rows, cols)], design.y[rows], design.row_query[rows]
    )


def score_on(
    model: LogisticRanker | None, design: Design, query_idx: Sequence[int]
) -> dict[int, np.ndarray]:
    """Счёт кандидатов каждого запроса; модели нет — счёт CV (порядок выдачи)."""
    out: dict[int, np.ndarray] = {}
    if model is None:
        for q in query_idx:
            n = int(design.ends[q] - design.starts[q])
            out[q] = -np.arange(n, dtype=np.float64)
        return out
    rows = design.rows_of(query_idx)
    scores = (
        model.decision_function(design.X[np.ix_(rows, design.columns(model.feature_names))])
        if len(rows)
        else np.zeros(0)
    )
    offset = 0
    for q in query_idx:
        n = int(design.ends[q] - design.starts[q])
        out[q] = scores[offset : offset + n]
        offset += n
    return out


def order_of(scores: np.ndarray) -> list[int]:
    """Порядок кандидатов по счёту; при равенстве — ранг CV."""
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def is_correct(design: Design, q: int, scores: np.ndarray) -> bool:
    positive = int(design.positive[q])
    return len(scores) > 0 and positive >= 0 and order_of(scores)[0] == positive


def true_nll(design: Design, q: int, scores: np.ndarray) -> float | None:
    if design.positive[q] < 0 or not len(scores):
        return None
    z = scores - scores.max()
    return float(np.log(np.exp(z).sum()) - z[design.positive[q]])


@dataclass
class CvOutcome:
    """Результат одной настройки на групповой K-fold."""

    loss: str
    l2: float
    names: tuple[str, ...]
    groups: tuple[str, ...]
    fold_top1: list[float]
    nll: float
    scores: dict[int, np.ndarray]

    @property
    def mean_top1(self) -> float:
        return float(np.mean(self.fold_top1))

    @property
    def rank_key(self) -> tuple[float, float, float, str]:
        # Точность top-1; при равенстве — меньший NLL верного (softmax счётов), затем более
        # сильная регуляризация, затем имя функции потерь — чтобы выбор был однозначным.
        return (round(self.mean_top1, 12), -round(self.nll, 12), self.l2, self.loss)


def cross_validate(
    design: Design,
    query_idx: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
    groups: tuple[str, ...],
    readers: Sequence[str],
    l2: float,
    loss: str,
    *,
    signed: bool = True,
) -> CvOutcome:
    names = tuple(feature_names(readers, groups))
    scores: dict[int, np.ndarray] = {}
    fold_top1: list[float] = []
    for fold in range(n_folds):
        test = [int(q) for q in query_idx[folds == fold]]
        train = [int(q) for q in query_idx[folds != fold]]
        part = score_on(fit_on(design, train, names, l2, loss, signed=signed), design, test)
        scores.update(part)
        fold_top1.append(float(np.mean([is_correct(design, q, part[q]) for q in test])))
    nlls = [v for q in query_idx if (v := true_nll(design, int(q), scores[int(q)])) is not None]
    return CvOutcome(
        loss=loss,
        l2=l2,
        names=names,
        groups=groups,
        fold_top1=fold_top1,
        nll=float(np.mean(nlls)) if nlls else math.inf,
        scores=scores,
    )


@dataclass
class Selection:
    best: CvOutcome
    grid: list[CvOutcome]
    steps: list[dict[str, Any]]


def select_config(
    design: Design,
    query_idx: np.ndarray,
    *,
    n_folds: int,
    grid: Sequence[float],
    losses: Sequence[str],
    groups: tuple[str, ...],
    readers: Sequence[str],
    search_groups: bool,
    signed: bool = True,
) -> Selection:
    """Функция потерь и L2 по сетке, (опционально) обратное исключение групп — по K-fold top-1."""
    folds = group_folds([design.groups[q] for q in query_idx], n_folds)

    def evaluate(chosen: tuple[str, ...]) -> tuple[CvOutcome, list[CvOutcome]]:
        table = [
            cross_validate(
                design, query_idx, folds, n_folds, chosen, readers, l2, loss, signed=signed
            )
            for loss in losses
            for l2 in grid
        ]
        return max(table, key=lambda outcome: outcome.rank_key), table

    best, table = evaluate(groups)
    steps: list[dict[str, Any]] = []
    while search_groups:
        trials = []
        for group in best.groups:
            if group in REQUIRED_GROUPS:
                continue
            trial, _ = evaluate(tuple(g for g in best.groups if g != group))
            trials.append((trial.rank_key, group, trial))
        if not trials:
            break
        _key, removed, trial = max(trials, key=lambda item: (item[0], item[1]))
        if trial.mean_top1 <= best.mean_top1 + 1e-12:
            break
        steps.append(
            {
                "removed": removed,
                "mean_top1": round(trial.mean_top1, 4),
                "loss": trial.loss,
                "l2": trial.l2,
            }
        )
        best = trial
    return Selection(best=best, grid=table, steps=steps)


def temperature_from(design: Design, scores: Mapping[int, np.ndarray]) -> float:
    """Температура по out-of-fold счётам: запросы без кандидатов пропускаются."""
    queries = [q for q in sorted(scores) if len(scores[q])]
    if not queries:
        return 1.0
    flat = np.concatenate([scores[q] for q in queries])
    ids = np.concatenate([[q] * len(scores[q]) for q in queries])
    correct = np.asarray([is_correct(design, q, scores[q]) for q in queries], dtype=np.float64)
    return fit_temperature(flat, QueryBlocks.from_ids(ids), correct)


def p_top1(scores: np.ndarray, temperature: float) -> float | None:
    if not len(scores):
        return None
    ids = np.zeros(len(scores), dtype=np.int64)
    return float(top1_probability(scores, QueryBlocks.from_ids(ids), temperature)[0])


# ------------------------------------------------------------------ оценка
@dataclass
class QueryEval:
    query: Query
    cv_top5: list[str]
    manual_top5: list[str]
    learned_top5: list[str]
    in_top_k: bool
    p_top1: float | None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cv_tie(self) -> int:
        """Сколько кандидатов делят счёт CV-top1 (1 — ничьей нет)."""
        return self.query.cv.top1_tie

    @property
    def tie_slugs(self) -> tuple[str, ...]:
        return self.query.cv.slugs[: self.cv_tie]

    def column(self, name: str) -> list[str]:
        return {"cv_only": self.cv_top5, "manual": self.manual_top5, "learned": self.learned_top5}[
            name
        ]


def share(num: int, den: int) -> float | None:
    """Доля; шесть знаков — чтобы 123/149 печаталось 82,6, а не 82,5 из-за 0,8255."""
    return None if den == 0 else round(num / den, 6)


def column_stats(evals: Sequence[QueryEval], column: str) -> dict[str, Any]:
    hits1 = sum(bool(e.column(column)) and e.column(column)[0] == e.query.slug for e in evals)
    hits5 = sum(e.query.slug in e.column(column)[:5] for e in evals)
    out: dict[str, Any] = {
        "top1": share(hits1, len(evals)),
        "top5": share(hits5, len(evals)),
        "n": len(evals),
    }
    if column == "cv_only":
        # Ничья CV-top1 — жребий: честная доля 1/k, если эталон среди k равных.
        fair = sum(
            (1.0 / e.cv_tie if e.query.slug in e.tie_slugs else 0.0)
            if e.cv_tie > 1
            else float(bool(e.cv_top5) and e.cv_top5[0] == e.query.slug)
            for e in evals
        )
        out["top1_tie_fair"] = None if not evals else round(fair / len(evals), 6)
    return out


def fix_break(evals: Sequence[QueryEval], column: str) -> dict[str, int]:
    """Сколько ответов CV исправлено и сколько сломано колонкой; из них — в ничьих CV."""
    counts = Counter()
    for e in evals:
        cv_right = bool(e.cv_top5) and e.cv_top5[0] == e.query.slug
        top = e.column(column)
        right = bool(top) and top[0] == e.query.slug
        outcome = {(False, True): "fixed", (True, False): "broke"}.get((cv_right, right), "same")
        counts[outcome] += 1
        if e.cv_tie > 1:
            counts[f"{outcome}_in_ties"] += 1
    return {
        "fixed": counts["fixed"],
        "broke": counts["broke"],
        "net": counts["fixed"] - counts["broke"],
        "fixed_in_ties": counts["fixed_in_ties"],
        "broke_in_ties": counts["broke_in_ties"],
    }


def calibration(evals: Sequence[QueryEval], bins: int = CALIBRATION_BINS) -> dict[str, Any]:
    """Надёжностная таблица вероятности top-1: децили по предсказанной вероятности."""
    rows = sorted(
        (
            (e.p_top1, bool(e.learned_top5) and e.learned_top5[0] == e.query.slug, e.query.key)
            for e in evals
            if e.p_top1 is not None
        ),
        key=lambda item: (item[0], item[2]),
    )
    if not rows:
        return {"n": 0, "bins": []}
    p = np.asarray([row[0] for row in rows])
    hit = np.asarray([row[1] for row in rows], dtype=np.float64)
    table = []
    for i, index in enumerate(np.array_split(np.arange(len(rows)), min(bins, len(rows)))):
        table.append(
            {
                "bin": i + 1,
                "n": len(index),
                "p_min": round(float(p[index].min()), 4),
                "p_max": round(float(p[index].max()), 4),
                "p_mean": round(float(p[index].mean()), 4),
                "accuracy": round(float(hit[index].mean()), 4),
            }
        )
    ece = sum(abs(row["p_mean"] - row["accuracy"]) * row["n"] for row in table) / len(rows)
    return {
        "n": len(rows),
        "mean_p": round(float(p.mean()), 4),
        "accuracy": round(float(hit.mean()), 4),
        "ece": round(float(ece), 4),
        "brier": round(float(np.mean((p - hit) ** 2)), 4),
        "bins": table,
    }


def summarize(evals: Sequence[QueryEval]) -> dict[str, Any]:
    by_set: dict[str, list[QueryEval]] = {}
    for e in evals:
        by_set.setdefault(e.query.set_name, []).append(e)

    def block(part: Sequence[QueryEval]) -> dict[str, Any]:
        return {
            "n": len(part),
            "ceiling_k": share(sum(e.in_top_k for e in part), len(part)),
            "cv_ties": sum(e.cv_tie > 1 for e in part),
            **{column: column_stats(part, column) for column in COLUMNS},
            "vs_cv": {column: fix_break(part, column) for column in ("manual", "learned")},
            "calibration": calibration(part),
        }

    return {
        "by_set": {name: block(part) for name, part in sorted(by_set.items())},
        "all": block(evals),
    }


def contributions(
    model: LogisticRanker, design: Design, q: int, a: int, b: int, top: int = 3
) -> list[list[Any]]:
    """Какие группы признаков подняли кандидата a над b: Σ w·(z_a − z_b) по группе.

    Суммируется по группам, а не по признакам: `cv_rank_inv` и `cv_is_top1` почти одно и то
    же, и поодиночке их вклады велики и взаимно гасятся — объяснение из них было бы ложным.
    """
    assert model.coef_ is not None and model.scale_ is not None
    cols = design.columns(model.feature_names)
    xa = design.X[design.starts[q] + a, cols]
    xb = design.X[design.starts[q] + b, cols]
    delta = model.coef_ * (xa - xb) / model.scale_
    by_group: dict[str, float] = {}
    for name, value in zip(model.feature_names, delta, strict=True):
        prefix = name.rsplit(".", 1)[0] + "." if "." in name else ""
        key = prefix + group_of(name)
        by_group[key] = by_group.get(key, 0.0) + float(value)
    ranked = sorted(by_group.items(), key=lambda item: (-abs(item[1]), item[0]))[:top]
    return [[key, round(value, 3)] for key, value in ranked if value != 0]


def evaluate_query(
    design: Design,
    q: int,
    query: Query,
    manual: list[str],
    scores: np.ndarray,
    temperature: float,
    model: LogisticRanker | None,
) -> QueryEval:
    slugs = design.slugs[q]
    order = order_of(scores)
    learned = [slugs[i] for i in order[:5]]
    extra: dict[str, Any] = {}
    if model is not None and order and order[0] != 0:
        extra["why_not_cv_top1"] = contributions(model, design, q, order[0], 0)
    return QueryEval(
        query=query,
        cv_top5=list(slugs[:5]),
        manual_top5=manual,
        learned_top5=learned,
        in_top_k=bool(design.positive[q] >= 0),
        p_top1=p_top1(scores, temperature),
        extra=extra,
    )


# ------------------------------------------------------------------ вариант целиком
@dataclass
class VariantRun:
    variant: str
    readers: tuple[str, ...]
    report: dict[str, Any]
    evals: list[QueryEval]
    model: LogisticRanker | None
    names: list[str]


def run_variant(
    variant: str,
    queries: Sequence[Query],
    all_readers: Sequence[str],
    attrs: CatalogAttrs,
    *,
    groups: tuple[str, ...],
    options: FeatureOptions,
    grid: Sequence[float],
    losses: Sequence[str],
    n_folds: int,
    top_k: int,
    search_groups: bool,
    out_dir: Path,
    train_info: Mapping[str, Any],
    signed: bool = True,
) -> VariantRun:
    readers = variant_readers(variant, all_readers)
    active = tuple(g for g in groups if readers or g not in TEXT_GROUPS)
    names = feature_names(readers, active)
    design = build_design(queries, readers, attrs, names, top_k=top_k, options=options)
    manual = [manual_top5(query, readers, attrs, top_k) for query in queries]
    all_idx = np.arange(len(queries), dtype=np.int64)

    # 1. Вложенный out-of-fold: выбор внутри каждого внешнего фолда.
    outer = group_folds(design.groups, n_folds)
    evals: list[QueryEval | None] = [None] * len(queries)
    outer_report = []
    for fold in range(n_folds):
        train_idx = all_idx[outer != fold]
        test_idx = [int(q) for q in all_idx[outer == fold]]
        selection = select_config(
            design,
            train_idx,
            n_folds=n_folds,
            grid=grid,
            losses=losses,
            groups=active,
            readers=readers,
            search_groups=search_groups,
            signed=signed,
        )
        best = selection.best
        model = fit_on(
            design, [int(q) for q in train_idx], best.names, best.l2, best.loss, signed=signed
        )
        temperature = temperature_from(design, best.scores)
        scores = score_on(model, design, test_idx)
        for q in test_idx:
            e = evaluate_query(design, q, queries[q], manual[q], scores[q], temperature, model)
            e.extra.update(fold=fold, loss=best.loss, l2=best.l2)
            evals[q] = e
        outer_report.append(
            {
                "fold": fold,
                "queries": len(test_idx),
                "train_queries": len(train_idx),
                "loss": best.loss,
                "l2": best.l2,
                "inner_mean_top1": round(best.mean_top1, 4),
                "groups": list(best.groups),
                "temperature": round(temperature, 4),
            }
        )
    done = [e for e in evals if e is not None]

    # 2. Модель для сервиса: выбор обычной K-fold на всём dev, обучение на всём dev.
    selection = select_config(
        design,
        all_idx,
        n_folds=n_folds,
        grid=grid,
        losses=losses,
        groups=active,
        readers=readers,
        search_groups=search_groups,
        signed=signed,
    )
    best = selection.best
    final = fit_on(design, [int(q) for q in all_idx], best.names, best.l2, best.loss, signed=signed)
    model_path = out_dir / "models" / f"{variant}.json"
    model_report: dict[str, Any] = {"path": None}
    if final is not None:
        final.temperature_ = temperature_from(design, best.scores)
        final.meta = {
            **train_info,
            "variant": variant,
            "readers": list(readers),
            "groups": list(best.groups),
            "top_k": top_k,
            "signed": signed,
            "selection": "групповая K-fold по cluster_B на всём dev, критерий — средняя top-1",
            "cv_mean_top1": round(best.mean_top1, 4),
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        final.save(model_path)
        weights = final.weights()
        assert final.scale_ is not None
        raw = dict(zip(final.feature_names, (final.coef_ / final.scale_).tolist(), strict=True))
        model_report = {
            "path": str(model_path),
            "l2": final.l2,
            "loss": final.loss,
            "temperature": round(final.temperature_, 4),
            "intercept": round(final.intercept_, 4),
            "fit_info": final.fit_info,
            "weights": [
                {"name": name, "weight": round(w, 4), "raw_weight": round(raw[name], 4)}
                for name, w in sorted(weights.items(), key=lambda item: (-abs(item[1]), item[0]))
            ],
            "contrasts": group_contrasts(raw),
        }

    report = {
        "variant": variant,
        "readers": list(readers),
        "features": len(names),
        "groups": list(active),
        "oof": {**summarize(done), "outer_folds": outer_report, "note": NESTED_NOTE},
        "selection": {
            "loss": best.loss,
            "l2": best.l2,
            "groups": list(best.groups),
            "group_steps": selection.steps,
            "grid": [
                {
                    "loss": outcome.loss,
                    "l2": outcome.l2,
                    "mean_top1": round(outcome.mean_top1, 4),
                    "fold_top1": [round(v, 4) for v in outcome.fold_top1],
                    "nll": round(outcome.nll, 4),
                }
                for outcome in selection.grid
            ],
            "note": GRID_NOTE,
        },
        "model": model_report,
        "manual_note": MANUAL_NOTE,
        "ties_note": TIES_NOTE,
    }
    return VariantRun(variant, readers, report, done, final, names)


def group_contrasts(raw: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Сдвиг счёта от «совпало» и «противоречит» против «не прочитано» — в сырых единицах.

    Один столбец группы сам по себе не сравним с другой группой: у винодельни совпадение
    разложено на `winery_match` и `winery_match_conf` (корреляция 0,97), и L2 делит вес между
    ними. Контраст складывает их при уверенности 1: это и есть вклад группы в решение.
    """
    out: dict[str, dict[str, float]] = {}
    prefixes = sorted({name.rsplit(".", 1)[0] + "." if "." in name else "" for name in raw})
    for prefix in prefixes:
        for group in AGREEMENT_FEATURES:
            match = raw.get(f"{prefix}{group}_match")
            conflict = raw.get(f"{prefix}{group}_conflict")
            if match is None or conflict is None:
                continue
            match += raw.get(f"{prefix}{group}_match_conf", 0.0)
            conflict += raw.get(f"{prefix}{group}_conflict_conf", 0.0)
            out[f"{prefix}{group}"] = {"match": round(match, 4), "conflict": round(conflict, 4)}
    return out


def cross_split_twins(dev: Sequence[Query], test: Sequence[Query]) -> list[dict[str, Any]]:
    """Пары «dev — test» одного набора с одной основой `query_id` без года и разными группами.

    Берутся только `query_id`, набор и группа — ни кандидатов, ни полей test.
    """
    by_stem: dict[tuple[str, str], list[Query]] = {}
    for query in dev:
        by_stem.setdefault((query.set_name, _YEAR_SUFFIX_RE.sub("", query.query_id)), []).append(
            query
        )
    out = []
    for query in test:
        stem = _YEAR_SUFFIX_RE.sub("", query.query_id)
        if stem == query.query_id:
            continue
        for mate in by_stem.get((query.set_name, stem), []):
            if mate.query_id != query.query_id and mate.group != query.group:
                out.append(
                    {
                        "set": query.set_name,
                        "dev": mate.query_id,
                        "test": query.query_id,
                        "dev_group": mate.group,
                        "test_group": query.group,
                    }
                )
    return sorted(out, key=lambda row: (row["set"], row["test"], row["dev"]))


def evaluate_final(
    run: VariantRun,
    queries: Sequence[Query],
    attrs: CatalogAttrs,
    *,
    top_k: int,
    options: FeatureOptions,
) -> list[QueryEval]:
    """Одна оценка модели, обученной на всём dev, на запросах другой половины."""
    design = build_design(queries, run.readers, attrs, run.names, top_k=top_k, options=options)
    idx = list(range(len(queries)))
    scores = score_on(run.model, design, idx)
    temperature = run.model.temperature_ if run.model is not None else 1.0
    return [
        evaluate_query(
            design,
            q,
            queries[q],
            manual_top5(queries[q], run.readers, attrs, top_k),
            scores[q],
            temperature,
            run.model,
        )
        for q in idx
    ]


def prediction_rows(variant: str, evals: Sequence[QueryEval]) -> Iterable[dict[str, Any]]:
    for e in evals:
        slug = e.query.slug
        yield {
            "variant": variant,
            "set": e.query.set_name,
            "query_id": e.query.query_id,
            "split": e.query.split,
            "slug": slug,
            "in_top_k": e.in_top_k,
            "cv_top1": e.cv_top5[0] if e.cv_top5 else None,
            "manual_top1": e.manual_top5[0] if e.manual_top5 else None,
            "learned_top1": e.learned_top5[0] if e.learned_top5 else None,
            "learned_top5": e.learned_top5,
            "p_top1": None if e.p_top1 is None else round(e.p_top1, 4),
            "cv_correct": bool(e.cv_top5) and e.cv_top5[0] == slug,
            "manual_correct": bool(e.manual_top5) and e.manual_top5[0] == slug,
            "learned_correct": bool(e.learned_top5) and e.learned_top5[0] == slug,
            "cv_tie_k": e.cv_tie,
            "ocr_status": e.query.ocr_status,
            **e.extra,
        }


# ------------------------------------------------------------------ таблицы
def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def fix_break_cell(counts: Mapping[str, int]) -> str:
    """«+9/−6», а если часть пришлась на ничьи CV — «+9/−6 (н. +0/−1)»."""
    cell = f"+{counts['fixed']}/−{counts['broke']}"
    tied_fixed, tied_broke = counts.get("fixed_in_ties", 0), counts.get("broke_in_ties", 0)
    if tied_fixed or tied_broke:
        cell += f" (н. +{tied_fixed}/−{tied_broke})"
    return cell


def results_table(reports: Mapping[str, Mapping[str, Any]], section: str) -> str:
    lines = [
        (
            "| вариант | набор | n | потолок K | ничьи CV | CV top1 | ручной top1 | обуч. top1 | "
            "CV top5 | ручной top5 | обуч. top5 | ручной испр/слом | обуч. испр/слом |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, report in reports.items():
        block = report[section]
        for set_name, part in [*block["by_set"].items(), ("все", block["all"])]:
            vs = part["vs_cv"]
            cv = part["cv_only"]
            ties = part.get("cv_ties", 0)
            cv_top1 = pct(cv["top1"])
            if ties and cv.get("top1_tie_fair") != cv["top1"]:
                cv_top1 += f" ({pct(cv.get('top1_tie_fair'))})"
            lines.append(
                f"| {variant} | {set_name} | {part['n']} | {pct(part['ceiling_k'])} | {ties} | "
                f"{cv_top1} | {pct(part['manual']['top1'])} | "
                f"{pct(part['learned']['top1'])} | {pct(cv['top5'])} | "
                f"{pct(part['manual']['top5'])} | {pct(part['learned']['top5'])} | "
                f"{fix_break_cell(vs['manual'])} | {fix_break_cell(vs['learned'])} |"
            )
    return "\n".join(lines)


def grid_table(reports: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [row for r in reports.values() for row in r["selection"]["grid"]]
    grid = sorted({row["l2"] for row in rows})
    losses = sorted({row["loss"] for row in rows})
    lines = [
        "| вариант | потери | "
        + " | ".join(f"L2={v:g}" for v in grid)
        + " | выбрано (вся dev) | выбрано во внешних фолдах |",
        "|---|---|" + "---|" * (len(grid) + 2),
    ]
    for variant, report in reports.items():
        chosen = f"{report['selection']['loss']} {report['selection']['l2']:g}"
        outer = ", ".join(f"{r['loss'][0]}{r['l2']:g}" for r in report["oof"]["outer_folds"])
        for loss in losses:
            cells = {
                row["l2"]: row["mean_top1"]
                for row in report["selection"]["grid"]
                if row["loss"] == loss
            }
            if not cells:
                continue
            lines.append(
                f"| {variant} | {loss} | "
                + " | ".join(pct(cells.get(v)) for v in grid)
                + f" | {chosen} | {outer} |"
            )
    return "\n".join(lines)


def weights_table(reports: Mapping[str, Mapping[str, Any]], top: int = 12) -> str:
    blocks = []
    for variant, report in reports.items():
        model = report["model"]
        weights = model.get("weights") or []
        lines = [
            f"**{variant}** ({model.get('loss')}, L2={model.get('l2')}, "
            + f"T={model.get('temperature')})",
            "",
            "| признак | вес (станд.) | вес (сырой) |",
            "|---|---|---|",
        ]
        lines += [
            f"| {w['name']} | {w['weight']:+.3f} | {w['raw_weight']:+.3f} |" for w in weights[:top]
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def contrast_table(reports: Mapping[str, Mapping[str, Any]]) -> str:
    """Контрасты групп согласия моделей на всём dev: «совпало» / «противоречит» vs «не прочитано»."""
    variants = [(v, r["model"].get("contrasts") or {}) for v, r in reports.items()]
    variants = [(v, c) for v, c in variants if c]
    if not variants:
        return "(у вариантов без текста контрастов нет)"
    keys = list(dict.fromkeys(key for _, contrasts in variants for key in contrasts))
    lines = [
        "| признак | " + " | ".join(v for v, _ in variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for key in keys:
        cells = []
        for _, contrasts in variants:
            row = contrasts.get(key)
            cells.append("" if row is None else f"{row['match']:+.2f} / {row['conflict']:+.2f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def calibration_table(reports: Mapping[str, Mapping[str, Any]], section: str) -> str:
    lines = [
        "| вариант | ECE | Brier | "
        + " | ".join(f"d{i}" for i in range(1, CALIBRATION_BINS + 1))
        + " |",
        "|---|---|---|" + "---|" * CALIBRATION_BINS,
    ]
    for variant, report in reports.items():
        cal = report[section]["all"]["calibration"]
        cells = [f"{b['p_mean']:.2f}→{b['accuracy']:.2f}" for b in cal.get("bins", [])]
        cells += [""] * (CALIBRATION_BINS - len(cells))
        lines.append(
            f"| {variant} | {cal.get('ece')} | {cal.get('brier')} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m bench.train_resolve",
        description="Обучение resolve (логистическая регрессия) на dev с групповой K-fold.",
    )
    parser.add_argument(
        "--cv",
        type=Path,
        action="append",
        required=True,
        help="predictions.jsonl прогона bench.retrieval; можно повторять (pairs и pairs_phone)",
    )
    parser.add_argument(
        "--ocr",
        action="append",
        default=[],
        metavar="ЧИТАТЕЛЬ:НАБОР=ПУТЬ",
        help="predictions.jsonl прогона bench.ocr_bench для набора CV (meta.set); можно повторять",
    )
    parser.add_argument(
        "--variants",
        help="через запятую: имена читателей, both, none; по умолчанию — все читатели, both "
        "(если читателей двое и больше) и none",
    )
    parser.add_argument(
        "--split",
        choices=(TRAIN_SPLIT,),
        default=TRAIN_SPLIT,
        help="половина для обучения и выбора — только dev",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--l2", default=",".join(f"{v:g}" for v in DEFAULT_L2), help="сетка L2 через запятую"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--loss",
        default=",".join(LOSSES),
        help="функции потерь через запятую (pointwise, listwise): выбираются вместе с L2 по "
        "K-fold top-1; по умолчанию обе",
    )
    parser.add_argument(
        "--groups",
        help="группы признаков через запятую; по умолчанию " + ",".join(DEFAULT_GROUPS),
    )
    parser.add_argument(
        "--published",
        action="store_true",
        help="включить слабый признак unpublished (карточка не опубликована на портале)",
    )
    parser.add_argument(
        "--select-groups",
        action="store_true",
        help="обратное исключение групп признаков по K-fold top-1 (внутри каждого фолда)",
    )
    parser.add_argument(
        "--free-signs",
        action="store_true",
        help="не держать знаки весов (согласие ≥ 0, противоречие ≤ 0) — только для сравнения",
    )
    parser.add_argument(
        "--gt-tokens", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl"
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="ОДНА оценка на test моделей, обученных на всём dev. По умолчанию выключено: без "
        "флага test не читается",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="папка прогона, runs/resolve-<имя>/"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = [parse_ocr_spec(text) for text in args.ocr]
        grid = parse_floats(args.l2)
        losses = parse_losses(args.loss)
        if args.top_k <= 0:
            raise ValueError("--top-k должен быть больше нуля")
        if args.folds < 2:
            raise ValueError("--folds должен быть не меньше двух")
        groups = check_groups(args.groups.split(",") if args.groups else DEFAULT_GROUPS)
        if args.published:
            groups = check_groups([*groups, "published"])
        missing = REQUIRED_GROUPS - set(groups)
        if missing:
            raise ValueError(f"без групп {', '.join(sorted(missing))} модель не видит CV")
    except ValueError as exc:
        print(f"ошибка аргументов: {exc}", file=sys.stderr)
        return 2
    if len({(s.reader, s.set_name) for s in specs}) != len(specs):
        print("ошибка аргументов: --ocr повторяет пару читатель:набор", file=sys.stderr)
        return 2
    if (args.out / "metrics.json").exists() and not args.overwrite:
        print(f"{args.out} уже содержит прогон: другое имя или --overwrite", file=sys.stderr)
        return 2
    if not args.gt_tokens.is_file():
        print(
            f"нет разметки каталога {args.gt_tokens} (scripts/build_gt_tokens.py)", file=sys.stderr
        )
        return 3
    gt_tokens = load_gt_tokens(args.gt_tokens)
    attrs = CatalogAttrs.from_records(gt_tokens.values())
    options = FeatureOptions(
        published=args.published,
        unpublished=frozenset(slug for slug, r in gt_tokens.items() if r.get("published") is False),
    )

    started = time.perf_counter()
    try:
        queries, inputs = load_split(args.cv, specs, gt_tokens, split=TRAIN_SPLIT, top_k=args.top_k)
        if not queries:
            raise DatasetError("в прогонах CV нет запросов dev")
        sets = sorted({q.set_name for q in queries})
        variants = check_variants(
            args.variants.split(",") if args.variants else default_variants(specs), specs, sets
        )
    except (DatasetError, ValueError) as exc:
        print(f"ошибка данных: {exc}", file=sys.stderr)
        return 2
    readers = reader_order(specs)
    for info in inputs["ocr"]:
        if info["matched"] < info["wanted"]:
            print(
                f"--ocr {info['reader']}:{info['set']}: чтения есть у {info['matched']} запросов "
                f"из {info['wanted']} — остальные идут без текста",
                file=sys.stderr,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    train_info = {
        "split": TRAIN_SPLIT,
        "sets": sets,
        "queries": len(queries),
        "cv_files": [info["path"] for info in inputs["cv"]],
        "cv_sha1": [info["sha1"] for info in inputs["cv"]],
        "ocr_files": [f"{i['reader']}:{i['set']}={i['path']}" for i in inputs["ocr"]],
        # Какой моделью и с какими параметрами читали: сервис сверяет с этим свой читатель.
        "reader_keys": reader_keys_by_name(inputs["ocr"]),
        "gt_tokens_sha1": file_sha1(args.gt_tokens),
        "feature_version": FEATURE_VERSION,
        "signed": not args.free_signs,
    }
    try:
        groups_count = len({q.group for q in queries})
        if groups_count < args.folds:
            raise ValueError(f"групп cluster_B на dev {groups_count} меньше, чем фолдов")
    except ValueError as exc:
        print(f"ошибка данных: {exc}", file=sys.stderr)
        return 2

    runs: dict[str, VariantRun] = {}
    for variant in variants:
        print(f"вариант {variant}…", file=sys.stderr)
        runs[variant] = run_variant(
            variant,
            queries,
            readers,
            attrs,
            groups=groups,
            options=options,
            grid=grid,
            n_folds=args.folds,
            top_k=args.top_k,
            losses=losses,
            search_groups=args.select_groups,
            out_dir=args.out,
            train_info=train_info,
            signed=not args.free_signs,
        )
    reports = {variant: run.report for variant, run in runs.items()}
    with (args.out / "oof_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for variant, run in runs.items():
            for row in prediction_rows(variant, run.evals):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics: dict[str, Any] = {
        "test_used": False,
        "run": {
            **train_info,
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "folds": args.folds,
            "l2_grid": list(grid),
            "top_k": args.top_k,
            "losses": list(losses),
            "groups": list(groups),
            "select_groups": args.select_groups,
            "published_feature": args.published,
            "variants": variants,
            "inputs": inputs,
            "dev_groups": groups_count,
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "variants": reports,
    }
    tables = [
        "## dev, out-of-fold (вложенный выбор L2)",
        results_table(reports, "oof"),
        "## Сетка L2: средняя top-1 по фолдам на всём dev",
        grid_table(reports),
        "## Калибровка вероятности top-1 (dev out-of-fold, децили: средняя p → точность)",
        calibration_table(reports, "oof"),
        "## Веса моделей, обученных на всём dev",
        weights_table(reports),
        (
            "## Контрасты групп согласия (модели на всём dev; сырые единицы счёта, «совпало / "
            "противоречит» против «не прочитано», у винодельни — при уверенности 1)"
        ),
        contrast_table(reports),
        "Ничьи CV: " + TIES_NOTE + ". В скобках у CV top1 — top1_tie_fair, «н.» — из них в ничьих.",
    ]

    if args.final_test:
        try:
            test_queries, test_inputs = load_split(
                args.cv, specs, gt_tokens, split=TEST_SPLIT, top_k=args.top_k
            )
            if not test_queries:
                raise DatasetError("в прогонах CV нет запросов test (прогоны только по dev?)")
        except DatasetError as exc:
            print(f"ошибка данных test: {exc}", file=sys.stderr)
            return 2
        test_reports: dict[str, Any] = {}
        with (args.out / "test_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for variant, run in runs.items():
                evals = evaluate_final(run, test_queries, attrs, top_k=args.top_k, options=options)
                test_reports[variant] = {"test": summarize(evals)}
                for row in prediction_rows(variant, evals):
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics["test_used"] = True
        metrics["test_warning"] = TEST_WARNING
        twins = cross_split_twins(queries, test_queries)
        metrics["test"] = {
            "inputs": test_inputs,
            "queries": len(test_queries),
            "cross_split_twins": twins,
            "cross_split_twins_note": TWINS_NOTE,
            "variants": {variant: report["test"] for variant, report in test_reports.items()},
        }
        tables += [
            "## TEST — одна оценка моделей, обученных на всём dev",
            TEST_WARNING,
            results_table(test_reports, "test"),
            calibration_table(test_reports, "test"),
            "Известные двойники по обе стороны разбиения ("
            + TWINS_NOTE
            + "): "
            + (
                "; ".join(f"{t['set']}: {t['test']} (test) ~ {t['dev']} (dev)" for t in twins)
                or "нет"
            ),
        ]
        print(TEST_WARNING, file=sys.stderr)

    metrics["run"]["wall_s"] = round(time.perf_counter() - started, 3)
    (args.out / "metrics.json").write_text(dumps(metrics) + "\n", encoding="utf-8")
    text = "\n\n".join(tables) + "\n"
    (args.out / "tables.md").write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "backslashreplace").decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
