"""Чтение этикетки одного кадра: кроп → читатели → токены → словарь → поля.

OCR не выбирает вино: на выходе признаки этикетки с доказательствами (`ReadResult`),
а решение принимает resolve поверх кандидатов CV. Текстового top-1 здесь нет.

Слияние чтений v1 — простое голосование без весов: токены всех чтений идут одним списком,
одно чтение — один голос за значение поля, при равенстве голосов между чтениями побеждает
чтение, стоящее раньше. `support` полей словаря — число разных чтений, где есть тот же
фрагмент: слова сравниваются по скелету с одной правкой, числа, годы, крепость и доли —
только точно. Весов читателей, выравнивания строк и триграмм нет: развивать слияние — только
после полевых замеров.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from app.reading.contracts import (
    Box,
    CropName,
    Evidence,
    LabelFields,
    LexField,
    LexHit,
    Reader,
    Reading,
    ReadResult,
    ReadStatus,
    TokenSpan,
    image_sha1,
)
from app.reading.crops import box_pixels, center_band, crop_full, label_box, pad_box
from app.reading.fields import extract_fields
from app.reading.lexicon.correct import (
    SHORT_WORD,
    LookupResult,
    bounded_levenshtein,
    search,
    unmatched,
)
from app.reading.readers.base import crop_px_for, ensure_rgb
from app.reading.text.layout import token_segments
from app.reading.text.tokenize import tokenize

if TYPE_CHECKING:
    from app.detect.bottles import TargetSelection
    from app.reading.lexicon.build import Lexicon

DEFAULT_BUDGET_MS = 2500
DEFAULT_CROP_PX = 1024
#: Отступ вокруг рамки бутылки: детектор режет впритык, края этикетки теряются.
BOTTLE_PAD_X = 0.08
BOTTLE_PAD_Y = 0.04
#: Этикетка — нижние 70 % рамки бутылки: у горлышка букв обычно нет.
LABEL_TOP = 0.30
#: Полоса `band` — середина этикетки, где цилиндр не сжимает буквы.
BAND_WIDTH_SHARE = 0.6
BAND_UPSCALE = 1.5
#: Статусы чтения, при которых кадр прочитан хуже, чем мог бы.
DEGRADED_STATUSES: frozenset[ReadStatus] = frozenset(
    {"timeout", "unavailable", "error", "loop", "garbage"}
)
#: Сколько правок скелета слова ещё считается тем же фрагментом в другом чтении.
SUPPORT_EDITS = 1

FULL_FRAME = Box(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
_LEXICAL_FIELDS: tuple[tuple[str, LexField], ...] = (
    ("producer", "producer"),
    ("cuvee", "cuvee"),
    ("grapes", "grape"),
)

Hits = list[tuple[tuple[int, ...], LexHit]]


def _ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


# ---------------------------------------------------------------------- кроп
@dataclass(frozen=True, slots=True)
class CropChoice:
    """Что реально читается: кадр, режим кропа и где вырез лежит в исходном кадре."""

    image: np.ndarray
    crop: CropName
    region: Box
    requested: CropName

    @property
    def fallback(self) -> bool:
        return self.crop != self.requested


def select_crop(
    image: np.ndarray,
    crop: CropName,
    *,
    target: TargetSelection | None,
    crop_px: int = DEFAULT_CROP_PX,
) -> CropChoice:
    """Кроп по рамке цели — только при `target.confident`, иначе весь кадр.

    Ошибочная рамка необратимо отрезает цель, а весь кадр 1024 px с промптом про центр
    читается приемлемо.
    """
    ensure_rgb(image)
    if crop_px <= 0:
        raise ValueError(f"crop_px должен быть > 0, получено {crop_px}")
    if crop == "full" or target is None or not target.confident:
        return CropChoice(crop_full(image, crop_px), "full", FULL_FRAME, crop)

    height, width = image.shape[:2]
    if crop == "bottle":
        box = pad_box(target.target, BOTTLE_PAD_X, BOTTLE_PAD_Y)
    else:
        box = pad_box(label_box(target.target, LABEL_TOP, 1.0), BOTTLE_PAD_X, 0.0)
    x0, y0, x1, y1 = box_pixels(box, width, height)
    region = np.ascontiguousarray(image[y0:y1, x0:x1])
    if crop == "band":
        margin = (1.0 - BAND_WIDTH_SHARE) / 2
        bx0, _, bx1, _ = box_pixels(Box(x0=margin, y0=0.0, x1=1.0 - margin, y1=1.0), x1 - x0, 1)
        region = center_band(region, BAND_WIDTH_SHARE, BAND_UPSCALE)
        x0, x1 = x0 + bx0, x0 + bx1
    frame = Box(x0=x0 / width, y0=y0 / height, x1=x1 / width, y1=y1 / height)
    return CropChoice(crop_full(region, crop_px), crop, frame, crop)


def _to_frame(box: Box, region: Box) -> Box | None:
    """Рамка в долях выреза → в долях исходного кадра."""
    w, h = region.x1 - region.x0, region.y1 - region.y0
    try:
        return Box(
            x0=min(1.0, region.x0 + box.x0 * w),
            y0=min(1.0, region.y0 + box.y0 * h),
            x1=min(1.0, region.x0 + box.x1 * w),
            y1=min(1.0, region.y0 + box.y1 * h),
        )
    except ValueError:
        return None


def _reading_in_frame(reading: Reading, region: Box) -> Reading:
    """Читатели дают рамки в долях переданного выреза, контракт `Box` — в долях кадра."""
    if region == FULL_FRAME or not any(line.box is not None for line in reading.lines):
        return reading
    lines = [
        line if line.box is None else line.model_copy(update={"box": _to_frame(line.box, region)})
        for line in reading.lines
    ]
    return reading.model_copy(update={"lines": lines})


# ---------------------------------------------------------------------- читатели
def _failed_reading(
    reader: Reader, choice: CropChoice, status: ReadStatus, elapsed_ms: int, error: str
) -> Reading:
    return Reading(
        reader=str(getattr(reader, "id", type(reader).__name__)),
        version=str(getattr(reader, "version", "")),
        params_hash=str(getattr(reader, "params_hash", "")),
        image_sha1=image_sha1(choice.image),
        crop=choice.crop,
        crop_px=crop_px_for(reader, choice.image),
        raw=json.dumps({"error": error}, ensure_ascii=False),
        status=status,
        elapsed_ms=elapsed_ms,
    )


def _run_readers(
    choice: CropChoice,
    readers: Sequence[Reader],
    *,
    deadline: float,
    clock: Callable[[], float],
    timings: dict[str, int],
) -> list[Reading]:
    """Строго по очереди: следующий читатель получает только остаток бюджета.

    Исчерпанный бюджет — чтение со статусом timeout без вызова читателя. Порядок списка
    задаёт вызывающий: OCR-читатели бюджетом не прерываются.
    """
    readings: list[Reading] = []
    seen: dict[str, int] = {}
    for reader in readers:
        name = str(getattr(reader, "id", type(reader).__name__))
        seen[name] = seen.get(name, 0) + 1
        label = name if seen[name] == 1 else f"{name}#{seen[name]}"
        started = clock()
        remaining = _ms(deadline - started) if deadline > started else 0
        if remaining <= 0:
            reading = _failed_reading(reader, choice, "timeout", 0, "budget exhausted")
        else:
            try:
                reading = reader.read(choice.image, crop=choice.crop, budget_ms=remaining)
            except Exception as exc:  # noqa: BLE001 — сбой одного читателя не роняет скан
                error = f"{type(exc).__name__}: {exc}"
                reading = _failed_reading(reader, choice, "error", _ms(clock() - started), error)
        timings[f"read:{label}"] = _ms(clock() - started)
        readings.append(_reading_in_frame(reading, choice.region))
    return readings


# ---------------------------------------------------------------------- словарь
def _is_bare_short_number(tokens: Sequence[TokenSpan], ids: Sequence[int]) -> bool:
    return all(
        tokens[i].kind == "number" and tokens[i].norm.isdigit() and len(tokens[i].norm) <= 2
        for i in ids
    )


def label_search(
    tokens: Sequence[TokenSpan], lexicon: Lexicon | None, segments: Sequence[int] | None = None
) -> LookupResult:
    """Попадания токенов в словарь каталога, без голых чисел из одной-двух цифр.

    Одиночное «4» из «сахара не более 4 г/дм³» совпадает с номерной серией у нескольких
    slug: без якоря («№», «cuvée») такое число — шум, а не признак. `parts` — слова
    каталожных фраз без собственного попадания: они не «вне словаря». `segments` — отрезки
    текста по рамкам строк (`token_segments`): фраза не склеивается с разных мест кадра.
    """
    if lexicon is None or not tokens:
        return LookupResult([], frozenset())
    found = search(tokens, lexicon, segments=segments)
    hits = [(ids, hit) for ids, hit in found.hits if not _is_bare_short_number(tokens, ids)]
    return LookupResult(hits, found.parts)


def label_hits(
    tokens: Sequence[TokenSpan], lexicon: Lexicon | None, segments: Sequence[int] | None = None
) -> Hits:
    """Попадания словаря для полей и диагностики — `label_search(...).hits`."""
    return label_search(tokens, lexicon, segments).hits


# ---------------------------------------------------------------------- support
def _compact(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _same_fragment(a: Sequence[TokenSpan], b: Sequence[TokenSpan]) -> bool:
    """Слова — скелет не дальше `SUPPORT_EDITS` правок (короткие — точно), прочее — точно."""
    if all(t.kind == "word" for t in a) and all(t.kind == "word" for t in b):
        left = " ".join(t.skeleton or t.norm for t in a)
        right = " ".join(t.skeleton or t.norm for t in b)
        if min(_compact(left), _compact(right)) <= SHORT_WORD:
            return left == right
        return bounded_levenshtein(left, right, SUPPORT_EDITS) <= SUPPORT_EDITS
    return [(t.kind, t.norm) for t in a] == [(t.kind, t.norm) for t in b]


@dataclass(slots=True)
class _Support:
    """Какие чтения содержат тот же фрагмент, что и окно токенов одного чтения."""

    tokens: Sequence[TokenSpan]
    sequences: dict[str, list[int]] = field(default_factory=dict)
    memo: dict[tuple[tuple[int, ...], str], tuple[int, ...] | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for i, token in enumerate(self.tokens):
            self.sequences.setdefault(token.reading, []).append(i)

    def windows_for(self, ids: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
        """Пары (ключ чтения, индексы окна): своё окно и первое похожее в каждом другом чтении."""
        span = [self.tokens[i] for i in ids]
        own = span[0].reading
        found = [(own, tuple(ids))]
        for key, seq in self.sequences.items():
            if key == own:
                continue
            memo_key = (tuple(ids), key)
            if memo_key not in self.memo:
                n = len(span)
                self.memo[memo_key] = next(
                    (
                        tuple(seq[start : start + n])
                        for start in range(len(seq) - n + 1)
                        if _same_fragment(span, [self.tokens[i] for i in seq[start : start + n]])
                    ),
                    None,
                )
            window = self.memo[memo_key]
            if window is not None:
                found.append((key, window))
        return found


def _order(keys: Sequence[str], rank: dict[str, int]) -> list[str]:
    return sorted(dict.fromkeys(keys), key=lambda key: rank.get(key, len(rank)))


def _fuse_lexical(
    fields: LabelFields, hits: Hits, support: _Support, rank: dict[str, int]
) -> tuple[LabelFields, Hits]:
    """`support` полей словаря: чтения с тем же фрагментом, даже если там он не попал в словарь.

    Вторым значением — окна других чтений, подтвердившие попадание: они не «вне словаря».
    """
    update: dict[str, list[Evidence[str]]] = {}
    confirmed: Hits = []
    for attr, lex_field in _LEXICAL_FIELDS:
        fused: list[Evidence[str]] = []
        for evidence in getattr(fields, attr):
            keys = list(evidence.sources)
            for ids, hit in hits:
                if hit.field == lex_field and hit.canonical == evidence.value:
                    for key, window in support.windows_for(ids):
                        keys.append(key)
                        if window != ids:
                            confirmed.append((window, hit))
            sources = _order(keys, rank)
            fused.append(evidence.model_copy(update={"sources": sources, "support": len(sources)}))
        update[attr] = fused
    return fields.model_copy(update=update), confirmed


def _unmatched_evidence(
    tokens: Sequence[TokenSpan], hits: Hits, rank: dict[str, int], known: frozenset[int]
) -> list[Evidence[str]]:
    """Сильные слова вне словаря, сгруппированные по скелету через все чтения."""
    groups: list[tuple[TokenSpan, list[str], list[float], int]] = []
    for position, token in enumerate(unmatched(tokens, hits, known=known)):
        for first, keys, confs, _ in groups:
            if _same_fragment([first], [token]):
                keys.append(token.reading)
                if token.conf is not None:
                    confs.append(token.conf)
                break
        else:
            confs = [token.conf] if token.conf is not None else []
            groups.append((token, [token.reading], confs, position))
    evidence = [
        (
            Evidence[str](
                value=first.norm,
                sources=_order(keys, rank),
                support=len(set(keys)),
                conf=max(confs) if confs else None,
            ),
            position,
        )
        for first, keys, confs, position in groups
    ]
    evidence.sort(key=lambda item: (-item[0].support, item[1]))
    return [item for item, _ in evidence]


# ---------------------------------------------------------------------- модуль
def read_label(
    image: np.ndarray,
    *,
    readers: Sequence[Reader],
    lexicon: Lexicon | None,
    target: TargetSelection | None = None,
    crop: CropName = "full",
    crop_px: int = DEFAULT_CROP_PX,
    budget_ms: int = DEFAULT_BUDGET_MS,
    year_now: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ReadResult:
    """Признаки этикетки кадра `image` (RGB uint8, HxWx3, исходный кадр).

    - `crop` bottle, label и band режутся по `target` только при `target.confident`,
      иначе читается весь кадр и в `degraded` пишется `crop_fallback_full`.
    - Бюджет `budget_ms` отсчитывается от входа в функцию и делится между читателями
      строго по очереди; разбор текста после чтения бюджетом не ограничен.
    - Без словаря (`lexicon=None`) винодельня, кюве и сорт не заполняются, `unmatched`
      пуст, а в `degraded` пишется `lexicon_missing`.
    - `year_now` и `clock` — для детерминированных тестов.

    Рамки строк в `ReadResult.readings` пересчитаны в доли исходного кадра.
    """
    started = clock()
    deadline = started + max(0, budget_ms) / 1000
    timings: dict[str, int] = {}
    degraded: list[str] = []

    choice = select_crop(image, crop, target=target, crop_px=crop_px)
    timings["crop"] = _ms(clock() - started)
    if choice.fallback:
        degraded.append("crop_fallback_full")

    t0 = clock()
    readings = _run_readers(choice, readers, deadline=deadline, clock=clock, timings=timings)
    timings["read"] = _ms(clock() - t0)
    if not readers:
        degraded.append("no_readers")
    for reading in readings:
        if reading.status in DEGRADED_STATUSES:
            degraded.append(f"{reading.reader}_{reading.status}")
    if readings and clock() > deadline:
        degraded.append("budget_exceeded")

    t0 = clock()
    tokens = [token for reading in readings for token in tokenize(reading, lexicon=lexicon)]
    timings["tokenize"] = _ms(clock() - t0)

    t0 = clock()
    found = label_search(tokens, lexicon, token_segments(tokens, readings))
    hits = found.hits
    timings["lexicon"] = _ms(clock() - t0)
    if lexicon is None:
        degraded.append("lexicon_missing")

    t0 = clock()
    fields = extract_fields(
        tokens, [(list(ids), hit) for ids, hit in hits], readings=readings, year_now=year_now
    )
    rank = {reading.key: i for i, reading in enumerate(readings)}
    confirmed: Hits = []
    if len({token.reading for token in tokens}) > 1:
        fields, confirmed = _fuse_lexical(fields, hits, _Support(tokens), rank)
    timings["fields"] = _ms(clock() - t0)

    t0 = clock()
    if lexicon is not None:
        strong = _unmatched_evidence(tokens, hits + confirmed, rank, found.parts)
        fields = fields.model_copy(update={"unmatched": strong})
    timings["unmatched"] = _ms(clock() - t0)
    timings["total"] = _ms(clock() - started)

    return ReadResult(
        readings=readings,
        tokens=tokens,
        fields=fields,
        timings_ms=timings,
        degraded=list(dict.fromkeys(degraded)),
    )
