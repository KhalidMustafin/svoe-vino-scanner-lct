"""Живой голос сомелье: пересказ готового пакета фактов, только когда видеокарта свободна.

Модель `qwen3.5:4b` ничего не решает: вердикт, блюда, вина и числа уже посчитаны правилами, и
шаблон ответа (`verdict_template`) уже показан. Голос лишь пересказывает его живым языком —
если ворота впустили, модель успела за `timeout_s` и текст прошёл все проверки. Любой отказ,
обрыв или брак — шаблон с меткой «Текст и подбор — алгоритм»; прошедший текст — метка «Текст —
ИИ, подбор — алгоритм» (договор, §1 и §6.3).

Порядок `speak` (договор, §6.3): намерение не пересказывается → `plain` при `order=plain`,
иначе `not_voiced`; голос выключен (`SVS_SOMM_LIVE=0`) → `off`; клиента нет (VLM выключен или
сервис не собран) → `not_ready`; ответ уже был → кэш без этапов; Ollama только что не
ответила → `unavailable` без вызова (`UNAVAILABLE_BACKOFF_S`); ворота не впустили → их причина;
иначе этап `voice`, вызов модели в потоке, этап `verify`, проверки. Имя хоста Ollama
разрешается до входа в ворота (`OllamaText.resolve`): под замком видеокарты — только соединение
и поток, которые скан обрывает сразу.

**Что видит модель** (§6.4) — только пакет: намерение нашей формулировкой, строки фактов и
готовый ответ. Вопрос гостя, описание из выгрузки, дескрипторы приоров, числа профиля и
контекст разговора в сообщения не попадают — их нет в `FactsPackage.voice_input`. Вход не
длиннее `MAX_INPUT_CHARS` (≈ 700 токенов, 2 100 символов русского текста): лишние строки фактов
срезаются с конца, а строки вин пакет ставит последними.

**Кэш** — LRU в памяти процесса по хэшу сообщений и тегу модели; хранит только прошедшие
проверки тексты, на диск ничего не пишет.

**Слот ворот** отдаётся в потоке вызова, как только поток к Ollama закрыт, и ещё раз (без
последствий) в `finally`: скан, оборвавший голос, получает замок сразу после обрыва сокета.

Текст модели, вопрос и контекст в журнал не пишутся: только коды причин и проверок.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.sommelier import guards
from app.sommelier.entity_lock import EntityLock
from app.sommelier.gate import SommGate, VoiceSlot
from app.sommelier.guards import FactsLike
from app.sommelier.ollama_text import ChatResult, OllamaText

logger = logging.getLogger(__name__)

LABEL_GENERATED = "Текст — ИИ, подбор — алгоритм"
LABEL_TEMPLATE = "Текст и подбор — алгоритм"

#: Причины шаблона вместо живого текста — коды `reason` договора (§3.3).
REASONS: tuple[str, ...] = (
    "not_voiced",
    "plain",
    "off",
    "not_ready",
    "busy",
    "quiet",
    "locked",
    "preempted",
    "timeout",
    "unavailable",
    "guard",
    "error",
    "cancelled",
)

#: Промпт «только факты» — `sommelier.py:61-77` «Лозы», переписанный под пакет фактов. Без
#: цифр: цифровой замок сверяет ответ с пакетом, и число из промпта было бы в ответе чужим.
SYSTEM_PROMPT = """Ты — сомелье справочного сервиса «Своё Вино». /no_think

Тебе дают намерение, список фактов и готовый ответ, собранный правилами сочетаний и подачи.
Перескажи готовый ответ спокойным разговорным русским языком, двумя-тремя короткими фразами.

Нерушимые правила:
- Называй только вина, винодельни, сорта, регионы, блюда и числа из фактов. Ничего не добавляй.
- Не описывай вкус, аромат и цвет вина, если их нет в фактах: никаких «нот» и «оттенков».
- Не хвали и не оценивай: без превосходных степеней, восторгов и слов «лучший», «идеальный».
- Не призывай пить, пробовать или покупать; не упоминай цены, магазины, скидки и проценты.
- Не говори о пользе и вреде алкоголя и не давай медицинских советов.
- Не перечисляй подборку целиком: карточки вин гость видит под текстом.
- Отвечай только текстом ответа, без вступлений, пометок, извинений и списков."""

#: Потолок входа модели: ≈ 700 токенов на системное и пользовательское сообщения вместе.
#: Русский текст у токенизатора Qwen — около 3 символов на токен, а не 4, как у английского:
#: 700 × 3 = 2 100 символов. Прежние 2 800 были бы ≈ 900 токенов.
#: Проверить `prompt_eval_count` в замере `scripts/somm_probe.py` на видеокарте.
MAX_INPUT_CHARS = 2100
#: Сколько символов этого потолка остаётся строкам фактов, намерению и готовому ответу.
USER_BUDGET_CHARS = MAX_INPUT_CHARS - len(SYSTEM_PROMPT)
#: Как часто голос спрашивает, не закрыл ли клиент поток.
POLL_S = 0.05
#: Запас сторожа сверх `timeout_s`: клиент Ollama обрывает себя сам, это страховка.
WATCHDOG_SLACK_S = 1.0
#: Ollama не отвечает — столько секунд голос её не зовёт: иначе каждый вопрос держал бы замок
#: видеокарты на время отказа соединения (на Windows — около 2 с повторов SYN).
UNAVAILABLE_BACKOFF_S = 30.0


@dataclass(frozen=True, slots=True)
class VoiceResult:
    text: str  # что показать: проверенный текст модели или verdict_template
    generated: bool
    label: str  # одна из двух меток §1
    reason: str | None  # None — живой текст прошёл; иначе код §3.3
    guard: str | None  # код проверки при reason == "guard"


def user_message(intent_label: str, lines: Sequence[str], verdict_template: str) -> str:
    facts = "\n".join(f"- {line}" for line in lines)
    return (
        f"Намерение: {intent_label}\nФакты:\n{facts}\nГотовый ответ: {verdict_template}\n/no_think"
    )


def build_messages(facts: FactsLike) -> list[dict[str, str]]:
    """Системное и пользовательское сообщения; строки фактов срезаются с конца до потолка входа."""
    lines = [str(line) for line in facts.voice_input if str(line).strip()]
    user = user_message(facts.intent_label, lines, facts.verdict_template)
    while len(SYSTEM_PROMPT) + len(user) > MAX_INPUT_CHARS and len(lines) > 1:
        lines.pop()
        user = user_message(facts.intent_label, lines, facts.verdict_template)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def cache_key(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    blob = json.dumps({"model": model, "messages": list(messages)}, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Voice:
    """Живой пересказ пакета фактов через ворота видеокарты; `speak` не бросает."""

    def __init__(
        self,
        gate: SommGate,
        client: OllamaText | None,
        lock: EntityLock,
        *,
        live: bool,
        timeout_s: float = 4.0,
        cache_size: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gate = gate
        self.client = client
        self.lock = lock
        self.live = live
        self.timeout_s = timeout_s
        self.cache_size = max(0, cache_size)
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._mutex = threading.Lock()
        self._calls = 0
        self._generated = 0
        self._cache_hits = 0
        self._reasons: Counter[str] = Counter()
        self._guards: Counter[str] = Counter()
        self._last: dict[str, Any] = {}
        self._clock = clock
        self._down_until = float("-inf")

    # -------------------------------------------------------------- ответ
    async def speak(
        self,
        facts: FactsLike,
        *,
        on_stage: Callable[[str], Awaitable[None]] | None = None,
        disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> VoiceResult:
        """Проверенный живой текст или шаблон с причиной. Исключений нет (кроме отмены задачи)."""
        with self._mutex:
            self._calls += 1
        try:
            result = await self._speak(facts, on_stage, disconnected)
        except asyncio.CancelledError:
            self._count(template(facts, "cancelled"))
            raise
        except Exception as exc:  # noqa: BLE001 — голос не роняет ответ
            logger.warning("голос сомелье упал (%s) — шаблон", type(exc).__name__)
            result = template(facts, "error")
        self._count(result)
        return result

    async def _speak(
        self,
        facts: FactsLike,
        on_stage: Callable[[str], Awaitable[None]] | None,
        disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> VoiceResult:
        if not facts.voiced:
            return template(
                facts, "plain" if facts.public.get("order") == "plain" else "not_voiced"
            )
        if not self.live:
            return template(facts, "off")
        client = self.client
        if client is None:
            return template(facts, "not_ready")
        messages = build_messages(facts)
        key = cache_key(client.model, messages)
        cached = self._cached(key)
        if cached is not None:
            return VoiceResult(cached, True, LABEL_GENERATED, None, None)
        if self._clock() < self._down_until:
            return template(facts, "unavailable")
        resolve = getattr(client, "resolve", None)
        if callable(resolve) and not await asyncio.to_thread(resolve):
            self._down_until = self._clock() + UNAVAILABLE_BACKOFF_S
            return template(facts, "unavailable")
        slot, refusal = self.gate.try_enter()
        if slot is None:
            return template(facts, refusal or "locked")
        try:
            if not await _stage(on_stage, "voice"):
                slot.cancel("client")
                return template(facts, "cancelled")
            chat = await self._call(client, messages, slot, disconnected)
            reason = _chat_reason(chat, slot)
            if reason == "unavailable":
                self._down_until = self._clock() + UNAVAILABLE_BACKOFF_S
            if reason is not None:
                return template(facts, reason)
            if chat.content.strip() and not await _stage(on_stage, "verify"):
                return template(facts, "cancelled")
            code = guards.check(
                chat.content,
                facts,
                self.lock,
                done_reason=chat.done_reason,
                thinking_only=chat.status == "thinking_only",
            )
            if code is not None:
                return template(facts, "guard", guard=code)
            text = guards.clean(chat.content)
            self._remember(key, text)
            return VoiceResult(text, True, LABEL_GENERATED, None, None)
        finally:
            slot.release()

    async def _call(
        self,
        client: OllamaText,
        messages: list[dict[str, str]],
        slot: VoiceSlot,
        disconnected: Callable[[], Awaitable[bool]] | None,
    ) -> ChatResult:
        """Вызов модели в потоке; клиент ушёл — слот обрывается, сокет закрывается сразу."""

        def run() -> ChatResult:
            try:
                return client.chat(
                    messages,
                    timeout_s=self.timeout_s,
                    cancel=slot.cancelled,
                    on_abort=slot.on_cancel,
                )
            finally:
                slot.release()  # замок видеокарты — скану, как только поток к Ollama закрыт

        started = time.perf_counter()
        task = asyncio.ensure_future(asyncio.to_thread(run))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=POLL_S)
                if done:
                    break
                if disconnected is not None and await _gone(disconnected):
                    slot.cancel("client")
                if time.perf_counter() - started > self.timeout_s + WATCHDOG_SLACK_S:
                    slot.cancel("timeout")
        except asyncio.CancelledError:
            slot.cancel("client")
            raise
        result = task.result()
        with self._mutex:
            self._last = {
                "status": result.status,
                "elapsed_ms": result.elapsed_ms,
                "first_token_ms": result.first_token_ms,
                "eval_count": result.eval_count,
                "prompt_eval_count": result.prompt_eval_count,
                "load_ms": result.load_ms,
                "error": result.error,
            }
        return result

    # -------------------------------------------------------------- кэш и счёт
    def _cached(self, key: str) -> str | None:
        with self._mutex:
            text = self._cache.get(key)
            if text is not None:
                self._cache.move_to_end(key)
                self._cache_hits += 1
            return text

    def _remember(self, key: str, text: str) -> None:
        if not self.cache_size:
            return
        with self._mutex:
            self._cache[key] = text
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def _count(self, result: VoiceResult) -> None:
        with self._mutex:
            if result.generated:
                self._generated += 1
            else:
                self._reasons[result.reason or "error"] += 1
            if result.guard:
                self._guards[result.guard] += 1

    def stats(self) -> dict[str, Any]:
        """Для `/v1/health.somm.voice`: вызовы, прошедшие проверки, откаты по причинам."""
        with self._mutex:
            return {
                "live": self.live,
                "model": self.client.model if self.client is not None else None,
                "timeout_s": self.timeout_s,
                "calls": self._calls,
                "generated": self._generated,
                "cache_hits": self._cache_hits,
                "cache_size": len(self._cache),
                "reasons": dict(sorted(self._reasons.items())),
                "guards": dict(sorted(self._guards.items())),
                "last_call": dict(self._last),
                "unavailable_left_s": round(max(0.0, self._down_until - self._clock()), 1),
            }


def template(facts: FactsLike, reason: str, *, guard: str | None = None) -> VoiceResult:
    """Шаблон пакета с причиной отката."""
    return VoiceResult(facts.verdict_template, False, LABEL_TEMPLATE, reason, guard)


def _chat_reason(chat: ChatResult, slot: VoiceSlot) -> str | None:
    """Причина отката по итогу вызова; `None` — текст есть, дальше проверки."""
    if chat.status in {"ok", "thinking_only"}:
        return None
    if chat.status == "cancelled":
        return {"scan": "preempted", "client": "cancelled", "timeout": "timeout"}.get(
            slot.reason or "", "cancelled"
        )
    if chat.status in {"timeout", "unavailable", "error"}:
        return chat.status
    return "error"


async def _stage(on_stage: Callable[[str], Awaitable[None]] | None, stage: str) -> bool:
    """Отдать этап странице; поток закрыт — `False`."""
    if on_stage is None:
        return True
    try:
        await on_stage(stage)
    except Exception:  # noqa: BLE001 — клиент ушёл посреди ответа
        return False
    return True


async def _gone(disconnected: Callable[[], Awaitable[bool]]) -> bool:
    """Закрыл ли клиент поток; упавшая проверка — тоже «закрыл»: слать ответ некуда."""
    try:
        return bool(await disconnected())
    except Exception:  # noqa: BLE001
        return True
