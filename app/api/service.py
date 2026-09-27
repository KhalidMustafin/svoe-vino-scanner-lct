"""Сервис сканера: кадр → поиск по картинке → чтение этикетки → обученный resolve → slug.

Схема та же, что у итогового замера на test (so400m + qwen3.5:4b + обученный resolve поверх
top-20, 85,2 % студия / 74,6 % «телефон» у `s2so400m-vlm35.json`): индекс, CV, VLM, кроп и
бюджет ниже не меняются, иначе цифры не переносятся. Агрегация CV, модель resolve и разбор полей
этикетки — уже не те, что в замере. С 26.09 по умолчанию (`SVS_CANDIDATE=adapter-lw-ranker`,
`app/api/config.py`) векторы SigLIP проходят адаптер поиска LW (`app/features/adapter.py`, карта
собрана под индекс `ccd3a01f`), а модель resolve — `s2so400m-vlm35-lw-pool.json`, обученная на его
выдаче (принят по замороженному тесту kr-test, `research/2026-09-26_fund/`). `SVS_CANDIDATE=off` —
прежний путь: поиск без адаптера и `s2so400m-vlm35-goal.json` (iter 20 цикла улучшений, счёты CV
`per_slug="zmax"`, признаки `resolve-features/3`). Модель замера с нынешним кодом не загружается:

    decode_on_backgrounds   один разбор байтов; серый фон эталонов — визуальному каналу
                            (`bench.retrieval`), белый — чтению (`bench.ocr_bench`)
    from_query(image, None) виды `bench.retrieval --target none`: кадр целиком и окна
    VisualIndex.search      top-20, `per_slug="zmax"` (пары «окно × вид» выровнены),
                            SigLIP float32, батч 16; с адаптером запрос и строки индекса —
                            в его пространстве
    read_label              OllamaVlmReader без дискового кэша, `crop="full"`, 1024 px,
                            бюджет min(SVS_VLM_TIMEOUT_MS, остаток общего бюджета)
    rank_query              чтение VLM под именем читателя модели (`vlm35`), как в обучении

Деградация вместо исключений:

    VLM упал, не успел или  ответ — CV top-1 и top-5 в порядке CV, вероятности нет, слой выбора
    не прочитал ни строки   не вызывается (Э2, `reader_failure`); флаг чтения в `degraded`
                            (`vlm_timeout`, `vlm_unavailable`, …) остаётся
    resolve упал            ответ — CV top-1, флаг `resolve_error`, вероятности нет
    декодирование или CV    slug null, `outcome="error"` и текст ошибки

Модели на видеокарте (SigLIP и VLM в Ollama) вызываются под одним замком: скрипт
организатора шлёт кадры по одному, но интерфейс может прислать два скана сразу, а две
GPU-задачи на 10 ГБ — это CUDA OOM и «@@@@». Ожидание замка тратит бюджет запроса.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from app.api.after_layer import AfterSearch, after_block
from app.api.cards import CatalogCards, read_records
from app.api.config import (
    CV_DTYPE,
    CV_PER_SLUG,
    MEASURED_READER_NAME,
    MEASURED_VLM_READER,
    POST_READ_RESERVE_MS,
    SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER,
    VLM_CROP,
    VLM_CROP_PX,
    ServiceSettings,
)
from app.api.lineage import GtLineage, LineageError, default_gt_lineage
from app.features.adapter import AdapterError, LinearAdapter, file_sha1
from app.features.contracts import Embedder, VisualResult
from app.features.embedder import DEFAULT_BATCH_SIZE, SiglipEmbedder
from app.features.index import IndexMismatch, VisualIndex
from app.features.views import BACKGROUND, from_query
from app.normalize.decode import (
    MAX_PIXELS,
    WHITE,
    DecodeError,
    decode_on_backgrounds,
    format_support,
    heif_available,
    sniff_format,
)
from app.reading.contracts import CropName, LabelFields, Reader, Reading, ReadResult
from app.reading.lexicon.build import Lexicon
from app.reading.pipeline import read_label
from app.reading.readers.base import CACHEABLE_STATUSES, long_side, make_reading
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.reading.warmup import DEFAULT_WARMUP_BUDGET_MS, synthetic_label, warm_readers
from app.recommend.catalog import RecoCatalog
from app.resolve.ambiguous import block_bonus_flip, rerank_ambiguous
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import FEATURE_VERSION, TextRead, readers_of
from app.resolve.learned import LogisticRanker, ModelFormatError, rank_query_detailed
from app.resolve.rerank import DEFAULT_CONFIG, abstain_check

logger = logging.getLogger(__name__)

Outcome = Literal["matched", "ambiguous", "out_of_catalog", "error"]
ServiceState = Literal["starting", "ready", "degraded"]

#: Ответ помечается неуверенным, если калиброванная вероятность верности top-1 ниже
#: половины: «скорее нет, чем да». Граница не подобрана на данных — это подсказка интерфейсу,
#: slug отдаётся всё равно.
AMBIGUOUS_P_TOP1 = 0.5
#: Меньше этого VLM не успеет даже принять кадр: чтение пропускается сразу.
VLM_MIN_BUDGET_MS = 300
#: Батч SigLIP — как у `bench.retrieval` по умолчанию.
CV_BATCH_SIZE = DEFAULT_BATCH_SIZE
#: Прогревочный кадр — синтетическая этикетка, как у стенда OCR.
WARM_FRAME_PX = VLM_CROP_PX
#: Статусы чтения, при которых модель ответила: прогрев VLM удался.
ANSWERED_STATUSES: frozenset[str] = frozenset(CACHEABLE_STATUSES)
#: Статус `evidence.vlm` у модели выбора без признаков текста: читателя нет, сбоя чтения тоже.
NO_READER_STATUS = "not_needed"
#: Сколько раз прогон прогревочного кадра с боевыми таймаутами повторяется, пока он не пройдёт
#: без флагов `degraded`: один случайный всплеск задержки не должен объявить сервис degraded.
WARM_SCAN_ATTEMPTS = 2


class StartupError(RuntimeError):
    """Сервис не может стартовать: нет файла или сборка не та, на которой мерили."""


class GpuBusy(TimeoutError):
    """Видеокарту дольше остатка бюджета держит другой запрос."""


# ------------------------------------------------------------------ ответ
class TopItem(BaseModel):
    slug: str
    score: float


class Confidence(BaseModel):
    """Калиброванная вероятность, что верен top-1, и что верный slug среди top-5.

    Температуру подбирали только для top-1 (`LogisticRanker.calibrate`): `top5` — сумма тех же
    вероятностей, её калибровку никто не проверял. `None` — вероятности нет (ответ CV без
    resolve или ошибка).
    """

    top1: float | None = None
    top5: float | None = None


class ScanResult(BaseModel):
    """Итог одного кадра. `slug` на верхнем уровне — его читает скрипт организатора."""

    slug: str | None = None
    confidence: Confidence = Field(default_factory=Confidence)
    margin: float = 0.0
    top5: list[TopItem] = Field(default_factory=list)
    outcome: Outcome = "error"
    degraded: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    def predict_body(self) -> dict[str, Any]:
        """Тело `/v1/eval/predict`: без доказательств.

        Новые поля ответа сюда не попадают только потому, что их нет в модели: `candidates` и
        `after` живут в `scan_body`. Любое поле, добавленное в `ScanResult`, ушло бы скрипту
        организатора (тест `test_predict_unchanged.py`).
        """
        return self.model_dump(mode="json", exclude={"evidence"})

    def scan_body(
        self,
        card: Mapping[str, Any] | None,
        *,
        candidates: Sequence[Mapping[str, Any]] | None = None,
        after: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Тело `/v1/scan`: всё, плюс карточка, кандидаты и подсказка экрана (`after`).

        Без `after` подсказка считается по одному ответу, без сверки винодельни с каталогом —
        так отвечает `/v1/scan`, пока сервис не собран.
        """
        return {
            **self.model_dump(mode="json"),
            "card": dict(card) if card else None,
            "candidates": [dict(item) for item in candidates or ()],
            "after": dict(after) if after is not None else after_block(self),
        }


def failure(message: str, *, timings_ms: Mapping[str, float] | None = None) -> ScanResult:
    """Ответ без slug, когда до сканера дело не дошло (пустой запрос, не тот файл)."""
    return ScanResult(slug=None, outcome="error", error=message, timings_ms=dict(timings_ms or {}))


# ------------------------------------------------------------------ чтение → признаки
def reading_status(readings: Sequence[Reading]) -> str:
    """Статус кадра, как `bench.ocr_bench.overall_status`: ok, если хоть одно чтение ok."""
    if not readings:
        return "error"
    if any(reading.status == "ok" for reading in readings):
        return "ok"
    return readings[0].status


def text_read_of(result: ReadResult) -> TextRead:
    """`ReadResult` → чтение для resolve ровно так, как его собирал `bench.train_resolve`.

    Там запись прогона OCR со статусом `error` давала пустое чтение, а остальные — поля
    `label_fields` и `raw_text`, склеенный из текстов чтений через перевод строки.
    """
    if reading_status(result.readings) == "error":
        return TextRead()
    raw = "\n".join(reading.text for reading in result.readings if reading.text)
    return TextRead(fields=result.fields, raw_text=raw)


def reader_failure(vlm: Mapping[str, Any] | None) -> str | None:
    """Почему кадр отвечает CV top-1 без слоя выбора (Э2); `None` — путь прежний.

    Вход — `evidence.vlm`, как его пишет `_read`. Сбой — любой статус, кроме `ok`: у читателя
    `timeout`, `unavailable`, `error`, `empty`, `loop`, `garbage`, у сервиса `disabled`,
    `skipped` и `error` исключения. Сбоем считается и `ok` без единой строки (`no_lines`).
    Модель без признаков текста (`not_needed`) и скан без записи о чтении идут прежним путём.

    Без текста обученный слой выбора с правилом спорного кадра хуже голого CV top-1: на
    записанных прогонах с пустым чтением v2 +23/−10, krasnostop +15/−1
    (`research/2026-09-25_acc/PREREG_E2_reader_fallback.md`).
    """
    if not vlm:
        return None
    status = str(vlm.get("status") or "unknown")
    if status == NO_READER_STATUS:
        return None
    if status != "ok":
        return status
    if not vlm.get("lines"):
        return "no_lines"
    return None


def cv_answer(visual: VisualResult) -> dict[str, Any]:
    """Ответ по одной картинке: CV top-1, top-5 в порядке CV со счётами CV, вероятности нет."""
    return {
        "slug": visual.candidates[0].slug,
        "confidence": Confidence(),
        "margin": _r(visual.margin),
        "top5": [TopItem(slug=c.slug, score=_r(c.score)) for c in visual.candidates[:5]],
        "outcome": "matched",
    }


def fields_summary(fields: LabelFields | None) -> dict[str, Any]:
    """Прочитанные поля этикетки в плоском виде — для доказательств ответа."""
    if fields is None:
        return {}

    def values(items: Sequence[Any]) -> list[str]:
        return [str(item.value) for item in items]

    return {
        "winery": values(fields.producer),
        "cuvee": values(fields.cuvee),
        "grapes": values(fields.grapes),
        "sugar": values(fields.sugar),
        "vintage": fields.vintage.value if fields.vintage else None,
        "serial": values(fields.serial),
        "abv": fields.abv.value if fields.abv else None,
        "color": str(fields.color.value) if fields.color else None,
        "unmatched": values(fields.unmatched)[:10],
    }


def softmax(scores: Sequence[float], temperature: float) -> np.ndarray:
    """Вероятности кандидатов запроса — та же формула, что `learned.top1_probability`."""
    z = np.asarray(scores, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


# ------------------------------------------------------------------ замок видеокарты
class LockedReader:
    """Читатель, который зовёт модель только под замком видеокарты.

    Ожидание замка вычитается из бюджета чтения. Не дождался — чтение со статусом
    `timeout`, как у исчерпанного бюджета: resolve получит пустой текст.
    """

    def __init__(
        self, reader: Reader, lock: threading.Lock, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.reader = reader
        self.lock = lock
        self.clock = clock
        self.id = str(reader.id)
        self.version = str(reader.version)
        self.params_hash = str(getattr(reader, "params_hash", ""))

    def available(self) -> bool:
        return self.reader.available()

    def crop_px_for(self, image: np.ndarray) -> int:
        custom = getattr(self.reader, "crop_px_for", None)
        return int(custom(image)) if callable(custom) else long_side(image)

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        started = self.clock()
        if not self.lock.acquire(timeout=max(0.0, budget_ms / 1000)):
            return make_reading(
                self,
                image,
                params=self.params_hash,
                crop=crop,
                status="timeout",
                elapsed_ms=max(0, round((self.clock() - started) * 1000)),
                raw=json.dumps({"error": "gpu busy: замок видеокарты не освободился"}),
            )
        try:
            waited_ms = round((self.clock() - started) * 1000)
            return self.reader.read(image, crop=crop, budget_ms=max(0, budget_ms - waited_ms))
        finally:
            self.lock.release()

    def warm(self, image: np.ndarray, *, budget_ms: int) -> Reading:
        with self.lock:
            warm = getattr(self.reader, "warm", None)
            if callable(warm):
                return warm(image, budget_ms=budget_ms)
            return self.reader.read(image, crop="full", budget_ms=budget_ms)


# ------------------------------------------------------------------ загрузка
def load_index(path: Path, cv_model: str) -> VisualIndex:
    """Индекс эталонов. Нет файла или он собран другой моделью — `StartupError`."""
    if not path.is_file():
        raise StartupError(f"нет индекса эталонов {path}: соберите scripts/build_index.py")
    try:
        index = VisualIndex.load(path, model=cv_model)
    except IndexMismatch as exc:
        raise StartupError(
            f"модель CV не совпадает с моделью индекса: {exc}. SVS_CV_MODEL должен быть моделью, "
            "которой собран SVS_INDEX_PATH: векторы разных моделей несравнимы"
        ) from exc
    except (OSError, ValueError) as exc:
        raise StartupError(f"индекс {path} не читается: {exc}") from exc
    if not len(index):
        raise StartupError(f"индекс {path} пуст")
    return index


def load_cv_adapter(path: Path, index_path: Path) -> LinearAdapter:
    """Адаптер поиска (`SVS_CANDIDATE`, по умолчанию `adapter-lw-ranker`). Нет файла, чужой формат
    или карта собрана для другого индекса — `StartupError`, а не тихий переход на поиск без карты:
    карта зависит от векторов индекса (центр, поворот, пары). Причина отказа говорит, как
    запустить прежний путь без адаптера."""
    fallback = "прежний путь без адаптера — SVS_CANDIDATE=off"
    try:
        adapter = LinearAdapter.load(path)
    except AdapterError as exc:
        raise StartupError(
            f"адаптер поиска {path}: {exc} (карта кладётся в index/ пачки данных "
            f"deploy/pack_data.sh; {fallback})"
        ) from exc
    index_sha1 = file_sha1(index_path)
    if adapter.index_sha1 != index_sha1:
        raise StartupError(
            f"адаптер поиска {path} собран для индекса {str(adapter.index_sha1)[:8]}, а загружен "
            f"{index_sha1[:8]} ({index_path}): карту надо пересобрать под этот индекс — это новый "
            f"артефакт, его приёмка — на новых отложенных данных; {fallback}"
        )
    return adapter


def load_resolve_model(path: Path) -> LogisticRanker:
    """Модель слоя выбора. Нет файла, другой формат или версия признаков — `StartupError`."""
    if not path.is_file():
        raise StartupError(f"нет модели resolve {path} (configs/resolve/, bench.train_resolve)")
    try:
        model = LogisticRanker.load(path)
    except (ModelFormatError, OSError, ValueError, KeyError, TypeError) as exc:
        raise StartupError(f"модель resolve {path} не загружается: {exc}") from exc
    if not model.fitted:
        raise StartupError(f"модель resolve {path} не обучена")
    return model


def load_lexicon(path: Path) -> Lexicon:
    """Словарь каталога: без него поля этикетки не те, на которых учился resolve."""
    if not path.is_file():
        raise StartupError(f"нет словаря каталога {path}: scripts/build_lexicon.py")
    try:
        return Lexicon.load(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StartupError(f"словарь {path} не читается: {exc}") from exc


def load_catalog(path: Path) -> tuple[list[dict[str, Any]], CatalogAttrs]:
    """Записи `gt_tokens.jsonl` и таблица признаков каталога из них."""
    if not path.is_file():
        raise StartupError(f"нет признаков каталога {path}: scripts/build_gt_tokens.py")
    try:
        records = read_records(path)
        attrs = CatalogAttrs.from_records(
            records,
            meta={
                "source": path.name,
                "records": len(records),
                "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
            },
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StartupError(f"признаки каталога {path} не читаются: {exc}") from exc
    if not len(attrs):
        raise StartupError(f"признаки каталога {path} пусты")
    return records, attrs


def reader_identity(reader: Any) -> str:
    """Строка читателя без кадра: `id@версия|params_hash`, как начало поля `reader` прогонов OCR."""
    return f"{reader.id}@{reader.version}|{getattr(reader, 'params_hash', '')}"


def trained_reader_keys(model: LogisticRanker, reader_key: str | None) -> list[str] | None:
    """Какие читатели дали чтения, на которых училась модель под ключом `reader_key`.

    Новые модели хранят их в `meta.reader_keys` (`bench.train_resolve`). У модели без этой
    записи ответ известен только для ключа замера `vlm35` — `MEASURED_VLM_READER`; для прочих
    ключей — `None`, сверить не с чем.
    """
    if reader_key is None:
        return None
    keys = model.meta.get("reader_keys")
    if isinstance(keys, Mapping):
        listed = keys.get(reader_key)
        if isinstance(listed, Sequence) and not isinstance(listed, str) and listed:
            return [str(item) for item in listed]
    if reader_key == MEASURED_READER_NAME:
        return [MEASURED_VLM_READER]
    return None


def provenance(
    model: LogisticRanker,
    attrs: CatalogAttrs,
    lexicon: Lexicon | None,
    *,
    reader_key: str | None = None,
    vlm_reader: str | None = None,
    live_cards: bool = False,
    lineage: GtLineage | None = None,
    cv_adapter: LinearAdapter | None = None,
) -> dict[str, Any]:
    """Из той ли разметки каталога, тем ли читателем и тем ли поиском собраны признаки и модель.

    Все трое хранят sha1 `gt_tokens.jsonl`: модель — в `meta.gt_tokens_sha1`, словарь — в
    `meta.source_sha1`, таблица признаков — хэш прочитанного файла. Словарь должен быть собран
    из загруженного gt, а сам gt — быть тем, на котором обучена модель, или его объявленной
    производной (`lineage`, по умолчанию `configs/resolve/gt_lineage.json`, `app/api/lineage.py`):
    тогда `derivation` — метки шагов цепочки (`gt_fixes@2d7050bf`), иначе `null`. Шаг живых
    карточек действует только при `live_cards` (`SVS_LIVE_CARDS=1`). Читатель этикетки сервиса
    (`vlm_reader`, `id@модель|params_hash`) сверяется с тем, чьи чтения видела модель
    (`trained_reader_keys`): другая VLM или другой промпт — другие признаки текста. Адаптер поиска
    сервиса (`cv_adapter`, sha1 содержимого) сверяется с тем, на чьей выдаче CV училась модель
    (`meta.cv_adapter_sha1`; нет записи — модель училась на поиске без адаптера).
    Что не сошлось — в `mismatch` (`gt_tokens`, `lexicon`, `vlm_reader`, `cv_adapter`).
    Расхождение разметки или читателя не роняет старт (каталог могли честно пересобрать, модель
    VLM — сменить ради опыта), но цифры замера к такой сборке не относятся. Модель, обученная на
    адаптере, без своего адаптера не стартует вовсе (`ScannerService._check`).
    """
    hashes = {
        "gt_tokens_sha1": attrs.meta.get("sha1"),
        "model_gt_tokens_sha1": model.meta.get("gt_tokens_sha1"),
        "lexicon_source_sha1": lexicon.meta.get("source_sha1") if lexicon is not None else None,
    }
    gt, lexicon_gt = hashes["gt_tokens_sha1"], hashes["lexicon_source_sha1"]
    trained_gt = hashes["model_gt_tokens_sha1"]
    mismatch: list[str] = []
    if gt and lexicon_gt and gt != lexicon_gt:
        mismatch.append("lexicon")
    # Разметка каталога сервиса: файл признаков, а если его хэш не известен — источник словаря.
    catalog_gt = gt or lexicon_gt
    derivation: str | None = None
    if trained_gt and catalog_gt and catalog_gt != trained_gt:
        steps = (lineage if lineage is not None else default_gt_lineage()).derive(
            trained_gt, catalog_gt, live_cards=live_cards
        )
        if steps is None:
            mismatch.append("gt_tokens")
        else:
            derivation = "+".join(step.label for step in steps)
    trained = trained_reader_keys(model, reader_key)
    if not (vlm_reader is None or trained is None or vlm_reader in trained):
        mismatch.append("vlm_reader")
    # Адаптер поиска: модель, обученная на его выдаче, хранит sha1 его содержимого; модель без
    # записи училась на поиске без адаптера. Карта сверена с индексом при загрузке (`load_cv_adapter`).
    trained_adapter = model.meta.get("cv_adapter_sha1") or None
    adapter_sha1 = cv_adapter.sha1 if cv_adapter is not None else None
    if trained_adapter != adapter_sha1:
        mismatch.append("cv_adapter")
    return {
        **hashes,
        "derivation": derivation,
        "vlm_reader": vlm_reader,
        "vlm_reader_trained": trained,
        "model_cv_adapter_sha1": trained_adapter,
        "cv_adapter_sha1": adapter_sha1,
        "cv_adapter_index_sha1": cv_adapter.index_sha1 if cv_adapter is not None else None,
        "mismatch": mismatch,
        "consistent": not mismatch,
    }


class ScanStats:
    """Счётчики сканов с прогрева: сколько ответов пришло без чтения этикетки и почему.

    Прогрев проверяет цепочку на одном синтетическом кадре, а настоящие этикетки длиннее: VLM
    может успевать на прогреве и не успевать на кадрах скрипта. Счётчики в `/v1/health`
    показывают это после прогона (`run_eval.sh` печатает разницу до и после).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.total = 0
            self.errors = 0
            self.null_slug = 0
            self.text_read = 0
            self.max_total_ms = 0.0
            self.degraded: dict[str, int] = {}

    def add(self, result: ScanResult) -> None:
        with self._lock:
            self.total += 1
            self.errors += result.outcome == "error"
            self.null_slug += result.slug is None
            text = result.evidence.get("resolve", {}).get("text_read")
            self.text_read += bool(text)
            self.max_total_ms = max(self.max_total_ms, float(result.timings_ms.get("total", 0.0)))
            for flag in result.degraded:
                self.degraded[flag] = self.degraded.get(flag, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self.total,
                "errors": self.errors,
                "null_slug": self.null_slug,
                "text_read": self.text_read,
                "max_total_ms": round(self.max_total_ms, 1),
                "degraded": dict(sorted(self.degraded.items())),
            }


def _jpeg(image: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# ------------------------------------------------------------------ один запрос
@dataclass
class _Run:
    """Часы, флаги и доказательства одного скана."""

    clock: Callable[[], float]
    started: float
    deadline: float
    timings: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def ms_since(self, t0: float) -> float:
        return round((self.clock() - t0) * 1000, 1)

    def left_ms(self) -> float:
        return (self.deadline - self.clock()) * 1000

    def flag(self, *names: str) -> None:
        for name in names:
            if name not in self.degraded:
                self.degraded.append(name)

    def finish(self, **fields: Any) -> ScanResult:
        if self.clock() > self.deadline:
            self.flag("over_budget")
        self.timings["total"] = self.ms_since(self.started)
        return ScanResult(
            **fields,
            degraded=list(self.degraded),
            timings_ms=dict(self.timings),
            evidence=dict(self.evidence),
        )

    def error(self, message: str) -> ScanResult:
        return self.finish(slug=None, outcome="error", error=message)


# ------------------------------------------------------------------ сервис
class ScannerService:
    """Все компоненты в памяти и `scan(bytes)`, который не бросает исключений."""

    def __init__(
        self,
        settings: ServiceSettings,
        *,
        index: VisualIndex,
        embedder: Embedder,
        lexicon: Lexicon | None,
        attrs: CatalogAttrs,
        model: LogisticRanker,
        vlm: Reader | None,
        cards: CatalogCards | None = None,
        after: AfterSearch | None = None,
        model_path: Path | None = None,
        clock: Callable[[], float] = time.perf_counter,
        lineage: GtLineage | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.embedder = embedder
        self.lexicon = lexicon
        self.attrs = attrs
        self.model = model
        self.cards = cards or CatalogCards([])
        #: Слой «после поиска»: карточка для показа, фото, похожие. Не задан — без справочника
        #: рекомендаций и снимка портала (карточки из разметки, похожих нет).
        self.after = after or AfterSearch(
            RecoCatalog([]), self.cards, attrs, photo_dir=settings.photo_dir
        )
        self.model_path = model_path
        self._clock = clock
        readers = readers_of(model.feature_names)
        self._check(readers)
        #: Под каким именем модель ждёт чтение этикетки (`vlm35`); `None` — модель без текста.
        self.reader_key: str | None = readers[0] if readers else None
        self.gpu_lock = threading.Lock()
        self.vlm: LockedReader | None = (
            LockedReader(vlm, self.gpu_lock, clock)
            if vlm is not None and self.reader_key is not None
            else None
        )
        self.state: ServiceState = "starting"
        self.degraded_reasons: list[str] = []
        self.warmed_at: str | None = None
        self.warm_report: dict[str, Any] = {}
        self.stats = ScanStats()
        # Пороги на шкале счёта CV: у адаптера она своя (верх кадра каталога 0,92 → 0,76), и
        # его файл несёт пороги S10 и подсказки «нет в каталоге», пересчитанные тем же правилом.
        # Порог подсказки перекалиброванной карты — из `SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER`.
        adapter = getattr(index, "adapter", None)
        scale: dict[str, float] = {}
        if adapter is not None:
            floor = adapter.threshold("abstain_visual_floor")
            if floor is not None:
                scale["visual_floor"] = floor
            suggest = SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER.get(
                adapter.sha1, adapter.threshold("suggest_not_found_visual_max")
            )
            if suggest is not None:
                self.after.suggest_max = suggest
        self._abstain_cfg = DEFAULT_CONFIG.model_copy(
            update={"abstain": "ooc_only", "top_k": settings.top_k, **scale}
        )
        #: То же правило без права отказа: условия пишутся в evidence, slug не трогается.
        self._abstain_cfg_off = DEFAULT_CONFIG.model_copy(
            update={"abstain": "off", "top_k": settings.top_k, **scale}
        )
        try:
            self.provenance = provenance(
                model,
                attrs,
                lexicon,
                reader_key=self.reader_key,
                vlm_reader=reader_identity(self.vlm) if self.vlm is not None else None,
                live_cards=settings.live_cards,
                lineage=lineage,
                cv_adapter=adapter,
            )
        except LineageError as exc:
            raise StartupError(f"объявленные производные разметки каталога: {exc}") from exc
        if not self.provenance["consistent"]:
            logger.warning(
                "Разметка каталога, читатель этикетки или адаптер поиска не те, на которых обучен "
                "resolve: %s — "
                "цифры замера к этой сборке не относятся",
                json.dumps(self.provenance, ensure_ascii=False),
            )
        elif self.provenance["derivation"]:
            logger.info(
                "Разметка каталога %s — объявленная производная разметки обучения resolve %s: %s "
                "(configs/resolve/gt_lineage.json)",
                str(self.provenance["gt_tokens_sha1"])[:8],
                str(self.provenance["model_gt_tokens_sha1"])[:8],
                self.provenance["derivation"],
            )

    def _check(self, readers: Sequence[str]) -> None:
        """Сборка должна быть той, на которой мерили: иначе отказ старта, а не тихий промах."""
        settings, index = self.settings, self.index
        if index.meta.model != settings.cv_model:
            raise StartupError(
                f"индекс {settings.index_path} собран моделью {index.meta.model!r}, а SVS_CV_MODEL "
                f"— {settings.cv_model!r}: векторы разных моделей несравнимы"
            )
        embedder_model = getattr(self.embedder, "model_name", None)
        if embedder_model != index.meta.model:
            raise StartupError(
                f"эмбеддер считает моделью {embedder_model!r}, а индекс собран {index.meta.model!r}"
            )
        if not len(index):
            raise StartupError("индекс пуст")
        if not self.model.fitted:
            raise StartupError("модель resolve не обучена")
        model_k = self.model.meta.get("top_k")
        if model_k is not None and int(model_k) != settings.top_k:
            raise StartupError(
                f"модель resolve обучена на top-{model_k} CV, а SVS_TOP_K={settings.top_k}: "
                "признаки ранга и выдачи считались бы не так, как при обучении"
            )
        if len(readers) > 1:
            raise StartupError(
                f"модель resolve ждёт чтения читателей {', '.join(readers)}, а сервис читает "
                "этикетку одной моделью зрения"
            )
        adapter = getattr(index, "adapter", None)
        if settings.cv_adapter is not None and adapter is None:
            raise StartupError(
                f"SVS_CANDIDATE={settings.candidate} включает адаптер поиска, а индекс загружен без него"
            )
        trained_adapter = self.model.meta.get("cv_adapter_sha1")
        if trained_adapter and (adapter is None or adapter.sha1 != trained_adapter):
            have = adapter.sha1[:8] if adapter is not None else "нет"
            raise StartupError(
                f"модель resolve обучена на выдаче адаптера поиска {str(trained_adapter)[:8]}, а в "
                f"сервисе адаптер {have}: признаки CV считались бы не так, как при обучении"
            )

    # -------------------------------------------------------------- сборка
    @classmethod
    def load(
        cls, settings: ServiceSettings, *, clock: Callable[[], float] = time.perf_counter
    ) -> ScannerService:
        """Компоненты с диска. Веса SigLIP поднимаются лениво — в `warm()`, а не здесь."""
        index = load_index(settings.index_path, settings.cv_model)
        if settings.cv_adapter is not None:
            index = index.with_adapter(load_cv_adapter(settings.cv_adapter, settings.index_path))
        model = load_resolve_model(settings.resolve_model)
        records, attrs = load_catalog(settings.attrs_path)
        lexicon = load_lexicon(settings.lexicon_path)
        cards = CatalogCards.build(
            records, csv_path=settings.catalog_csv, photo_map=settings.photo_map
        )
        after = AfterSearch.load(settings, cards, attrs)
        missing = [slug for slug in index.slug_order if slug not in attrs]
        if missing:
            logger.warning(
                "%d slug индекса нет в признаках каталога (первый — %s): resolve увидит их без "
                "винодельни и сорта",
                len(missing),
                missing[0],
            )
        embedder = SiglipEmbedder(
            index.meta.model, settings.device, CV_DTYPE, batch_size=CV_BATCH_SIZE
        )
        vlm = (
            OllamaVlmReader(
                settings.vlm_model,
                settings.ollama_url,
                timeout_s=settings.vlm_timeout_ms / 1000,
                keep_alive=settings.vlm_keep_alive,
            )
            if settings.vlm_enabled
            else None
        )
        return cls(
            settings,
            index=index,
            embedder=embedder,
            lexicon=lexicon,
            attrs=attrs,
            model=model,
            vlm=vlm,
            cards=cards,
            after=after,
            model_path=settings.resolve_model,
            clock=clock,
        )

    # -------------------------------------------------------------- прогрев
    def warm(self) -> dict[str, Any]:
        """Поднять веса SigLIP, загрузить VLM в память Ollama и прогнать скан целиком.

        Без прогрева первый кадр упирается в бюджет: холодная загрузка VLM — 3–5 с, а таймаут
        чтения — 2,5 с. Поэтому VLM греется с бюджетом 60 с, а уже потом синтетический кадр
        проходит всю цепочку с боевыми таймаутами (до `WARM_SCAN_ATTEMPTS` раз, пока не
        пройдёт без флагов). Исключений нет: итог — в `health()`.

        `ready` — прогревочный скан прошёл чисто: картинка и текст укладываются в бюджет, SigLIP
        на том устройстве, что просили. `degraded` — сервис отвечает, но хуже замера: причины в
        `degraded_reasons` (VLM не ответил или не успевает в 2,5 с, CV на CPU вместо видеокарты,
        бюджет съеден до чтения, сбой CV — тогда slug null).
        """
        report: dict[str, Any] = {}
        frame = synthetic_label(WARM_FRAME_PX)
        started = self._clock()
        try:
            with self.gpu_lock:
                self.index.search(
                    from_query(frame, None),
                    self.embedder,
                    top_k=self.settings.top_k,
                    per_slug=CV_PER_SLUG,
                )
            report["cv"] = {
                "status": "ok",
                "elapsed_ms": self._ms_since(started),
                "device": self.cv_device,
            }
        except Exception as exc:  # неготовность CV показывается в health
            logger.exception("Прогрев CV не удался")
            report["cv"] = {
                "status": "error",
                "elapsed_ms": self._ms_since(started),
                "error": f"{type(exc).__name__}: {exc}",
            }
        if self.vlm is not None:
            warm = warm_readers([self.vlm], image=frame, budget_ms=DEFAULT_WARMUP_BUDGET_MS)
            report["vlm"] = next(iter(warm.values()))
        else:
            report["vlm"] = {"status": "disabled" if self.reader_key else "not_needed"}
        if report["cv"]["status"] == "ok":
            data = _jpeg(frame)
            for attempt in range(1, WARM_SCAN_ATTEMPTS + 1):
                result = self.scan(data)
                report["scan"] = {
                    "attempt": attempt,
                    "outcome": result.outcome,
                    "degraded": result.degraded,
                    "timings_ms": result.timings_ms,
                }
                if result.outcome != "error" and not result.degraded:
                    break
        reasons = self._degraded_reasons(report)
        self.warm_report = report
        self.warmed_at = _now()
        self.degraded_reasons = reasons
        self.state = "degraded" if reasons else "ready"
        self.stats.reset()  # прогревочные сканы в счётчики прогона не идут
        logger.log(
            logging.WARNING if reasons else logging.INFO,
            "Прогрев: %s%s — %s",
            self.state,
            f" ({', '.join(reasons)})" if reasons else "",
            json.dumps(report, ensure_ascii=False),
        )
        return report

    def _degraded_reasons(self, report: Mapping[str, Any]) -> list[str]:
        """Почему сервис после прогрева хуже замера. Пусто — `ready`."""
        reasons: list[str] = []
        if report["cv"]["status"] != "ok":
            reasons.append("cv_warm_error")
        vlm_status = str(report["vlm"]["status"])
        if vlm_status not in ANSWERED_STATUSES and vlm_status != "not_needed":
            reasons.append(f"vlm_warm_{vlm_status}")
        fallback = self.cv_device_fallback
        if fallback:
            reasons.append(fallback)
        if not heif_available():
            # HEIC — формат камеры iPhone по умолчанию: без pillow-heif такой кадр не разжимается.
            reasons.append("heic_unavailable")
        scan = report.get("scan")
        if scan is not None:
            if scan["outcome"] == "error":
                reasons.append("warm_scan_error")
            reasons.extend(f"warm_scan:{flag}" for flag in scan["degraded"])
        return list(dict.fromkeys(reasons))

    @property
    def cv_device(self) -> str | None:
        """Где на деле считает SigLIP: `None`, пока веса не подняты (или у эмбеддера без устройства)."""
        device = getattr(self.embedder, "device", None)
        return str(device) if device else None

    @property
    def cv_device_fallback(self) -> str | None:
        """`cv_device_fallback:cuda->cpu`, если SigLIP ушёл не на то устройство, что просили.

        `SiglipEmbedder` без CUDA молча переходит на CPU (только строка в журнале), а CV на CPU
        бюджетом не ограничен: so400m по видам — секунды, и кадр уходит за `curl --max-time 10`.
        """
        actual = self.cv_device
        requested = self.settings.device
        if actual is None or actual == requested:
            return None
        if requested == "cuda" and actual.startswith("cuda"):
            return None
        return f"cv_device_fallback:{requested}->{actual}"

    def _ms_since(self, t0: float) -> float:
        return round((self._clock() - t0) * 1000, 1)

    # -------------------------------------------------------------- справка
    @property
    def model_name(self) -> str:
        return self.model_path.name if self.model_path else "in-memory"

    @property
    def cv_adapter(self) -> LinearAdapter | None:
        """Адаптер поиска (`SVS_CANDIDATE`, по умолчанию есть); `None` — путь `off` без адаптера."""
        return getattr(self.index, "adapter", None)

    def health(self) -> dict[str, Any]:
        meta = self.index.meta
        return {
            "status": self.state,
            "degraded_reasons": list(self.degraded_reasons),
            "warnings": self.settings.warnings(),
            "model": {
                "cv": meta.model,
                "cv_device": self.cv_device,
                "resolve": self.model_name,
                "resolve_reader": self.reader_key,
                "resolve_trained_at": self.model.meta.get("trained_at"),
                "feature_version": FEATURE_VERSION,
                "vlm": self.settings.vlm_model,
                "cv_adapter": self.cv_adapter.public() if self.cv_adapter is not None else None,
                # Действующий порог подсказки `suggest_not_found` (у карты из словаря config —
                # не тот, что в её мете `cv_adapter.suggest_not_found_visual_max`).
                "suggest_not_found_visual_max": self.after.suggest_max,
            },
            "index": {
                "path": str(self.settings.index_path),
                "model": meta.model,
                "dim": meta.dim,
                "n_slugs": meta.n_slugs,
                "n_vectors": meta.n_vectors,
                # Строк нормировки zmax: меньше n_vectors — индекс дополнен (живые карточки).
                "n_base": getattr(self.index, "n_base", meta.n_vectors),
                "built_at": meta.built_at,
            },
            "vlm": {
                "model": self.settings.vlm_model,
                "url": self.settings.ollama_url,
                "enabled": self.vlm is not None,
                "warm": self.warm_report.get("vlm"),
            },
            "warmed_at": self.warmed_at,
            "warm": self.warm_report,
            "settings": {
                "budget_ms": self.settings.budget_ms,
                "vlm_timeout_ms": self.settings.vlm_timeout_ms,
                "abstain": self.settings.abstain,
                "top_k": self.settings.top_k,
                "device": self.settings.device,
                "live_cards": self.settings.live_cards,
                "candidate": self.settings.candidate,
                "vlm_crop": VLM_CROP,
                "vlm_crop_px": VLM_CROP_PX,
            },
            "catalog": {"slugs": len(self.attrs), "cards": self.cards.sources},
            "after": self.after.stats(),
            "provenance": self.provenance,
            "formats": format_support(),
            "scans": self.stats.as_dict(),
        }

    def card(self, slug: str | None) -> dict[str, Any] | None:
        """Карточка для показа (договор «после поиска», §3): без путей машины, с `photo_url`."""
        return self.after.card(slug)

    def photo_file(self, slug: str, *, extra_dirs: Sequence[Path] = ()) -> Path | None:
        """Файл эталонного фото позиции или `None`. Путь остаётся внутри сервиса."""
        return self.after.photo_file(slug, extra_dirs=extra_dirs)

    def scan_body(self, result: ScanResult) -> dict[str, Any]:
        """Тело `/v1/scan`: ответ, карточка, кандидаты и подсказка экрана.

        `/v1/eval/predict` этим не пользуется: у него `result.predict_body()`, и новые поля
        туда не попадают. Сбой слоя «после поиска» ответ не прячет: скан уходит с карточкой из
        разметки, без кандидатов и с подсказкой по одному ответу.
        """
        try:
            card = self.card(result.slug)
            candidates = self.after.candidates(result)
            after = self.after.after(result)
        except Exception:
            logger.exception("Слой «после поиска» упал: /v1/scan отвечает без него")
            base = self.cards.get(result.slug) if result.slug else None
            card = {k: v for k, v in base.to_dict().items() if k != "photo_path"} if base else None
            candidates, after = [], None
        return result.scan_body(card, candidates=candidates, after=after)

    # -------------------------------------------------------------- скан
    def scan(self, data: bytes) -> ScanResult:
        """Один кадр → ответ. Любой сбой — поле ответа, а не исключение."""
        started = self._clock()
        run = _Run(self._clock, started, started + self.settings.budget_ms / 1000)
        try:
            result = self._scan(data, run)
        except Exception as exc:  # последняя страховка: наружу ничего не летит
            logger.exception("Скан упал")
            result = run.error(f"internal: {type(exc).__name__}: {exc}")
        self.stats.add(result)
        logger.info(
            "скан: slug=%s outcome=%s p=%s total=%.0f мс degraded=%s%s",
            result.slug,
            result.outcome,
            result.confidence.top1,
            result.timings_ms.get("total", 0.0),
            ",".join(result.degraded) or "-",
            f" error={result.error}" if result.error else "",
        )
        return result

    def _scan(self, data: bytes, run: _Run) -> ScanResult:
        t0 = self._clock()
        try:
            image_cv, image_ocr = decode_on_backgrounds(
                data, (BACKGROUND, WHITE), max_pixels=MAX_PIXELS
            )
        except DecodeError as exc:
            run.timings["decode"] = run.ms_since(t0)
            return run.error(f"decode: {exc}")
        run.timings["decode"] = run.ms_since(t0)
        height, width = image_cv.shape[:2]
        run.evidence["image"] = {
            "format": sniff_format(data),
            "width": width,
            "height": height,
            "bytes": len(data),
        }

        try:
            visual = self._search(image_cv, run)
        except Exception as exc:  # без CV ответа нет, но и исключения тоже
            logger.exception("Поиск по картинке упал")
            return run.error(f"cv: {type(exc).__name__}: {exc}")

        reads, fields = self._read(image_ocr, run)
        answer = self._resolve(visual, reads, fields, run)
        self._abstain(visual, fields, answer, run)
        return run.finish(**answer)

    def _search(self, image: np.ndarray, run: _Run) -> VisualResult:
        t0 = self._clock()
        views = from_query(image, None)
        run.timings["views"] = run.ms_since(t0)
        t1 = self._clock()
        if not self.gpu_lock.acquire(timeout=max(0.0, run.left_ms() / 1000)):
            raise GpuBusy("видеокарту держит другой запрос дольше остатка бюджета")
        try:
            run.timings["gpu_wait"] = run.ms_since(t1)
            t2 = self._clock()
            visual = self.index.search(
                views, self.embedder, top_k=self.settings.top_k, per_slug=CV_PER_SLUG
            )
        finally:
            self.gpu_lock.release()
        run.timings["cv"] = run.ms_since(t2)
        if not visual.candidates:
            raise ValueError("индекс не дал ни одного кандидата")
        run.evidence["cv"] = {
            "model": visual.model,
            "margin": _r(visual.margin, 6),
            "top5": [
                {"slug": c.slug, "score": _r(c.score, 6), "view": c.view, "rank": c.rank}
                for c in visual.candidates[:5]
            ],
            "timings_ms": visual.timings_ms,
        }
        return visual

    def _read(self, image: np.ndarray, run: _Run) -> tuple[dict[str, TextRead], LabelFields | None]:
        """Чтение этикетки под именем читателя модели. Сбой — пустое чтение и флаг."""
        key = self.reader_key
        if key is None:
            run.evidence["vlm"] = {"status": NO_READER_STATUS}
            return {}, None
        if self.vlm is None:
            run.flag("vlm_disabled")
            run.evidence["vlm"] = {"status": "disabled"}
            return {key: TextRead()}, None
        budget = int(min(self.settings.vlm_timeout_ms, run.left_ms() - POST_READ_RESERVE_MS))
        if budget < VLM_MIN_BUDGET_MS:
            run.flag("vlm_skipped_budget")
            run.evidence["vlm"] = {"status": "skipped", "budget_ms": max(0, budget)}
            return {key: TextRead()}, None
        if budget < self.settings.vlm_timeout_ms:
            run.flag("vlm_budget_cut")
        t0 = self._clock()
        try:
            result = read_label(
                image,
                readers=[self.vlm],
                lexicon=self.lexicon,
                target=None,
                crop=VLM_CROP,
                crop_px=VLM_CROP_PX,
                budget_ms=budget,
                clock=self._clock,
            )
        except Exception as exc:  # без текста resolve всё равно ответит
            logger.exception("Чтение этикетки упало")
            run.timings["vlm"] = run.ms_since(t0)
            run.flag("vlm_exception")
            run.evidence["vlm"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            return {key: TextRead()}, None
        run.timings["vlm"] = run.ms_since(t0)
        run.flag(*result.degraded)
        read = text_read_of(result)
        run.evidence["vlm"] = {
            "model": self.settings.vlm_model,
            "status": reading_status(result.readings),
            "budget_ms": budget,
            "elapsed_ms": sum(reading.elapsed_ms for reading in result.readings),
            "lines": [line.text for reading in result.readings for line in reading.lines],
            "fields": fields_summary(read.fields),
            "timings_ms": result.timings_ms,
        }
        return {key: read}, read.fields

    def _resolve(
        self,
        visual: VisualResult,
        reads: Mapping[str, TextRead],
        fields: LabelFields | None,
        run: _Run,
    ) -> dict[str, Any]:
        t0 = self._clock()
        top = visual.candidates[0]
        failure = reader_failure(run.evidence.get("vlm"))
        if failure is not None:
            # Сбой читателя (Э2): без текста слой выбора и правило спорного кадра хуже CV top-1.
            run.timings["resolve"] = run.ms_since(t0)
            run.evidence["resolve"] = {
                "fallback": f"reader_{failure}",
                "cv_top1": top.slug,
                "changed_cv_top1": False,
                "text_read": False,
            }
            return cv_answer(visual)
        try:
            ranking, features = rank_query_detailed(self.model, visual, reads, self.attrs)
            if not ranking.slugs:
                raise ValueError("resolve не вернул ни одного кандидата")
            probs = softmax(ranking.scores, self.model.temperature_)
            if not np.isfinite(probs).all():
                raise ValueError("вероятности resolve не числа")
        except Exception as exc:  # сбой слоя выбора: отвечает CV
            logger.exception("Resolve упал, ответ — CV top-1")
            run.timings["resolve"] = run.ms_since(t0)
            run.flag("resolve_error")
            run.evidence["resolve"] = {"error": f"{type(exc).__name__}: {exc}"}
            return cv_answer(visual)
        p1 = float(probs[0])
        ranked = list(ranking.slugs)
        ambiguous = p1 < AMBIGUOUS_P_TOP1
        flip_blocked = False
        if ambiguous:
            # Спорный кадр: бонус соседу по кластеру снимается, спорящие по сахару и цвету выбывают.
            ranked = list(rerank_ambiguous(self.model, features, fields, self.attrs))
        else:
            # Уверенный кадр: бонус соседу не переворачивает ответ к карточке, спорящей с этикеткой.
            guarded = list(block_bonus_flip(self.model, features, ranked, fields, self.attrs))
            flip_blocked = guarded[0] != ranked[0]
            ranked = guarded
        by_slug = dict(zip(ranking.slugs, (float(p) for p in probs), strict=False))
        top5 = ranked[:5]
        shown = [by_slug.get(s, 0.0) for s in top5]
        run.timings["resolve"] = run.ms_since(t0)
        slug = ranked[0]
        p_answer = by_slug.get(slug, p1)
        p_next = shown[1] if len(shown) > 1 else 0.0
        text = next(iter(reads.values()), None)
        run.evidence["resolve"] = {
            "model": self.model_name,
            "reader": self.reader_key,
            "temperature": self.model.temperature_,
            "p_top1": _r(p1),
            "scores_top5": [_r(score) for score in ranking.scores[:5]],
            "cv_top1": top.slug,
            "changed_cv_top1": slug != top.slug,
            "text_read": bool(text and text.raw_text),
            "ambiguous_rerank": ambiguous,
            "changed_by_rerank": ambiguous and slug != ranking.slugs[0],
            "bonus_flip_blocked": flip_blocked,
        }
        return {
            "slug": slug,
            "confidence": Confidence(top1=_r(p_answer), top5=_r(min(1.0, float(sum(shown))))),
            "margin": _r(p_answer - p_next),
            "top5": [TopItem(slug=s, score=_r(p)) for s, p in zip(top5, shown, strict=False)],
            "outcome": "matched" if p1 >= AMBIGUOUS_P_TOP1 else "ambiguous",
        }

    def _abstain(
        self,
        visual: VisualResult,
        fields: LabelFields | None,
        answer: dict[str, Any],
        run: _Run,
    ) -> None:
        """Правило S10: винодельня прочитана уверенно, ни одна её позиция не сходится, CV слаб.

        Условия правила пишутся в `evidence.abstain` всегда: из них страница после поиска
        считает подсказку «похоже, этой позиции нет в каталоге» (план, Д2). Менять ответ правило
        может только при `SVS_ABSTAIN=ooc_only`. При `off` проверка идёт с конфигом без права
        отказа (`fired` всегда false), а её сбой не ставит флага в `degraded`: `degraded`
        уходит в `/v1/eval/predict`, и тело predict при `off` не должно зависеть от этого слоя.
        """
        enforce = self.settings.abstain == "ooc_only"
        cfg = self._abstain_cfg if enforce else self._abstain_cfg_off
        try:
            check = abstain_check(visual, fields, self.attrs, cfg=cfg)
        except Exception as exc:  # сбой правила отказа не отменяет ответ
            logger.exception("Правило отказа упало")
            if enforce:
                run.flag("abstain_error")
            run.evidence["abstain"] = {"error": f"{type(exc).__name__}: {exc}"}
            return
        run.evidence["abstain"] = check
        if enforce and check.get("fired"):
            answer["slug"] = None
            answer["outcome"] = "out_of_catalog"
