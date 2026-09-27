"""Ворота видеокарты для голоса сомелье: скан всегда первый.

Замок видеокарты сервиса (`ScannerService.gpu_lock`) — обычный `threading.Lock`: поиск CV и
чтение этикетки берут его с таймаутом, как и до сомелье. Ворота этот замок не подменяют и его
семантику не трогают. Они добавляют голосу три условия входа и одно правило выхода:

- голос входит только в тишину: ни одного скана в работе (`scan_started` / `scan_finished`
  вокруг `run_scan`) и прошло `quiet_s` секунд после последнего `/v1/eval/predict` — скрипт
  организатора шлёт кадры подряд, и пауза между ними ещё не тишина;
- замок голос берёт без ожидания (`acquire(blocking=False)`): занят прогревом или другим
  голосом — голоса нет, остаётся шаблон;
- первый же скан обрывает голос: `scan_started` ставит слоту `cancelled` и зовёт колбэки
  `on_cancel`. Колбэк клиента закрывает сокет к Ollama, генерация останавливается, и поток
  голоса отдаёт замок. Скан ждёт замок столько, сколько идёт этот обрыв, а не весь ответ модели.

Одновременно активен не больше одного слота. Методы ворот исключений не бросают: хуки стоят в
пути predict, а тело predict от них не зависит.

Порядок замков один на весь модуль — мьютекс ворот, потом мьютекс слота: `scan_started` и
`preempt` обрывают слот под мьютексом ворот, `_release` берёт их в том же порядке. Мьютекс ворот
реентерабельный: колбэк обрыва вправе сразу отдать слот, не попадая в самоблокировку.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Self

logger = logging.getLogger(__name__)

#: Тихое окно после каждого `/v1/eval/predict`, секунд. Из окружения его читает
#: `SommSettings.quiet_s` (договор, §6.7): переменная окружения — одна на все настройки сомелье.
DEFAULT_QUIET_S = 15.0
#: Причины отказа во входе — коды `reason` договора (§3.3).
REFUSALS: tuple[str, ...] = ("not_ready", "busy", "quiet", "locked")
#: Кто обрывает слот: скан, закрытый клиентом поток, сторож времени голоса.
CANCEL_REASONS: tuple[str, ...] = ("scan", "client", "timeout")


def _call_quietly(callback: Callable[[], None]) -> None:
    """Колбэк обрыва не должен уронить скан, который его вызвал."""
    try:
        callback()
    except Exception:  # колбэк чужой, а скан, который его позвал, важнее
        logger.warning("колбэк обрыва голоса упал", exc_info=True)


class VoiceSlot:
    """Право голоса на одну генерацию: держит замок видеокарты до `release()`.

    `cancelled` ставится один раз — сканом, закрытым клиентом потоком или сторожем времени;
    `reason` говорит, кем. Колбэки `on_cancel` зовутся ровно один раз: при обрыве или сразу при
    регистрации, если слот уже оборван.
    """

    def __init__(self, gate: SommGate, lock: threading.Lock, entered_at: float) -> None:
        self.cancelled = threading.Event()
        self.reason: str | None = None
        self.entered_at = entered_at
        self.cancelled_at: float | None = None
        self._gate = gate
        self._lock = lock
        self._mutex = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def on_cancel(self, callback: Callable[[], None]) -> None:
        """Зарегистрировать обрыв (закрыть сокет к Ollama); слот уже оборван — позвать сразу."""
        with self._mutex:
            if not self.cancelled.is_set():
                self._callbacks.append(callback)
                return
        _call_quietly(callback)

    def cancel(self, reason: str) -> bool:
        """Оборвать слот. Уже оборван или отдан — ничего; `True` — оборван этим вызовом."""
        with self._mutex:
            if self.cancelled.is_set() or self._released:
                return False
            self.reason = reason
            self.cancelled_at = self._gate.now()
            self.cancelled.set()
            callbacks, self._callbacks = self._callbacks, []
        self._gate._note_cancel(reason)
        for callback in callbacks:
            _call_quietly(callback)
        return True

    def release(self) -> None:
        """Отдать замок видеокарты. Повторный вызов ничего не делает."""
        self._gate._release(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class SommGate:
    """Скан всегда первый: голос входит только в тишину и уходит по первому скану."""

    def __init__(
        self, *, quiet_s: float = DEFAULT_QUIET_S, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.quiet_s = max(0.0, float(quiet_s))
        self._clock = clock
        self._mutex = threading.RLock()
        self._gpu: threading.Lock | None = None
        self._scans = 0
        self._quiet_until = -math.inf
        self._slot: VoiceSlot | None = None
        self._counts = {"scans": 0, "predicts": 0, "entered": 0, "released": 0}
        self._refused = dict.fromkeys(REFUSALS, 0)
        self._cancelled = dict.fromkeys(CANCEL_REASONS, 0)
        self._hold_ms = {"last": None, "max": None}
        #: От обрыва до отдачи замка — столько скан ждёт голос. Цель договора — ≤ 300 мс p95.
        self._release_ms = {"last": None, "max": None}

    def now(self) -> float:
        return self._clock()

    @property
    def bound(self) -> bool:
        return self._gpu is not None

    @property
    def active_scans(self) -> int:
        return self._scans

    # -------------------------------------------------------------- связь с сервисом
    def bind(self, gpu_lock: threading.Lock) -> None:
        """Замок видеокарты собранного сервиса. До `bind` голос не входит (`not_ready`)."""
        with self._mutex:
            self._gpu = gpu_lock

    # -------------------------------------------------------------- хуки скана
    def scan_started(self) -> None:
        """+1 скан; активный голос обрывается сразу, до чтения кадра."""
        try:
            with self._mutex:
                self._scans += 1
                self._counts["scans"] += 1
                self._cancel_active("scan")
        except Exception:  # хук в пути predict: наружу ничего
            logger.warning("ворота сомелье: scan_started упал", exc_info=True)

    def scan_finished(self) -> None:
        """−1 скан; ниже нуля не уходит: лишний вызов не даст следующему скану счёт «0»."""
        try:
            with self._mutex:
                self._scans = max(0, self._scans - 1)
        except Exception:  # хук в пути predict: наружу ничего
            logger.warning("ворота сомелье: scan_finished упал", exc_info=True)

    def predict_done(self) -> None:
        """Ответ predict ушёл: тихое окно продлевается до `now + quiet_s`."""
        try:
            with self._mutex:
                self._counts["predicts"] += 1
                self._quiet_until = max(self._quiet_until, self._clock() + self.quiet_s)
        except Exception:  # хук в пути predict: наружу ничего
            logger.warning("ворота сомелье: predict_done упал", exc_info=True)

    # -------------------------------------------------------------- голос
    def try_enter(self) -> tuple[VoiceSlot | None, str | None]:
        """Слот голоса или причина отказа. Никогда не ждёт."""
        try:
            with self._mutex:
                reason = self._refusal()
                if reason is None:
                    gpu = self._gpu
                    assert gpu is not None  # проверено в _refusal
                    if gpu.acquire(blocking=False):
                        slot = VoiceSlot(self, gpu, self._clock())
                        self._slot = slot
                        self._counts["entered"] += 1
                        return slot, None
                    reason = "locked"
                self._refused[reason] += 1
                return None, reason
        except Exception:  # без голоса ответ всё равно будет
            logger.warning("ворота сомелье: try_enter упал", exc_info=True)
            return None, "locked"

    def preempt(self, reason: str = "scan") -> None:
        """Оборвать активный слот (скан, закрытый клиентом поток, сторож времени)."""
        try:
            with self._mutex:
                self._cancel_active(reason)
        except Exception:  # обрыв не должен уронить того, кто обрывает
            logger.warning("ворота сомелье: preempt упал", exc_info=True)

    def _refusal(self) -> str | None:
        if self._gpu is None:
            return "not_ready"
        if self._scans > 0:
            return "busy"
        if self._clock() < self._quiet_until:
            return "quiet"
        if self._slot is not None:
            return "locked"
        return None

    def _cancel_active(self, reason: str) -> None:
        slot = self._slot
        if slot is not None:
            slot.cancel(reason)

    def _note_cancel(self, reason: str) -> None:
        with self._mutex:
            self._cancelled[reason] = self._cancelled.get(reason, 0) + 1

    def _release(self, slot: VoiceSlot) -> None:
        try:
            with self._mutex, slot._mutex:
                if slot._released:
                    return
                slot._released = True
                if self._slot is slot:
                    self._slot = None
                now = self._clock()
                self._counts["released"] += 1
                _track(self._hold_ms, (now - slot.entered_at) * 1000)
                if slot.cancelled_at is not None:
                    _track(self._release_ms, (now - slot.cancelled_at) * 1000)
                slot._lock.release()
        except Exception:  # замок уже отдан или чужой: в журнал, не наружу
            logger.warning("ворота сомелье: слот не отдался", exc_info=True)

    # -------------------------------------------------------------- справка
    def stats(self) -> dict[str, Any]:
        """Для `/v1/health.somm.gate`: состояние, счётчики и время отдачи замка."""
        with self._mutex:
            now = self._clock()
            return {
                "bound": self._gpu is not None,
                "quiet_s": self.quiet_s,
                "active_scans": self._scans,
                "quiet_left_s": round(max(0.0, self._quiet_until - now), 2),
                "voice_active": self._slot is not None,
                **self._counts,
                "refused": dict(self._refused),
                "cancelled": dict(self._cancelled),
                "hold_ms": dict(self._hold_ms),
                "release_after_cancel_ms": dict(self._release_ms),
            }


def _track(box: dict[str, float | None], value_ms: float) -> None:
    value = round(value_ms, 1)
    box["last"] = value
    box["max"] = value if box["max"] is None else max(box["max"], value)
