"""Маршруты «Сомелье». Договор — `docs/api-sommelier.md`, логика — `app/sommelier/`.

    GET  /v1/wines/{slug}/sommelier   блоки карточки: заметка-шаблон, профиль по 8 осям,
                                      подача, три блюда с чипами правил, чипы входа в лист
    POST /v1/sommelier/ask            вопрос листа (чип или текст) → поток NDJSON:
                                      stage* (facts stage* text | error) done

Ответ считается на CPU из готовых данных (`app/sommelier/answers.py`), модель только
пересказывает готовый пакет фактов, и то лишь когда ворота видеокарты пускают голос. Событие
`facts` уходит сразу после расчёта — шаблон виден на странице, пока голос ещё говорит.

**Вопрос гостя не пишется никуда**: ни в журнал, ни в метрики, ни в исключения — ни текстом,
ни отрывком, ни длиной (договор, §1 «Журнал»). На один вопрос — одна строка INFO с кодами:
намерение, чип, причина голоса, проверка, время. Ошибка сборки пишется типом исключения и
кадрами стека без сообщения: сообщение исключения могло бы унести слово вопроса.

Выключатели (`SommSettings`, `app/sommelier/settings.py`): `SVS_SOMM_LIVE=0` — живого голоса
нет (шаблон с `reason: "off"`), `SVS_SOMM_INPUT=0` — вопрос текстом отвечает
`422 {"detail": "вопрос текстом выключен"}`, чипы работают, `SVS_SOMM_SAFETY=1` — поверх
барьера правил смысловой слой (`app/sommelier/safety.py`, по умолчанию выключен). Маршрут
вопроса считается в потоке: смысловому слою может понадобиться дождаться загрузки модели.

Маршруты отдельно от логики, как у слоя «после поиска»: `app/sommelier` не знает FastAPI.
Старт сервиса (`create_app`, lifespan) зовёт `SommRuntime.start`: данные сомелье — те же, что
уже прочитал слой «после поиска» (`AfterSearch.somm`, один `SVS_SOMM_DIR`), а голос —
`Voice` с воротами видеокарты `app.state.somm_gate`, клиентом `OllamaText.mirroring` читателя
этикетки сервиса и замком сущностей по `vocab.json`. Без старта (маршруты в чужом приложении)
данные читаются из `SVS_SOMM_DIR` при первом вопросе, а ворота без замка видеокарты голос не
впускают (`not_ready`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import traceback
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from app.recommend.content_filter import check
from app.recommend.somm_data import SommData, load_somm_data
from app.sommelier import templates as t
from app.sommelier.answers import FactsPackage, Sommelier, WineSource
from app.sommelier.entity_lock import EntityLock
from app.sommelier.gate import SommGate
from app.sommelier.ollama_text import OllamaText
from app.sommelier.router import (
    CHIP_ARGS,
    GUIDED_FOOD,
    GUIDED_WANT,
    REQUIRED_ARGS,
    ContextError,
    Router,
    context_of,
)
from app.sommelier.safety import SafetyLayer
from app.sommelier.settings import SommSettings
from app.sommelier.text import MAX_QUESTION, clean_question
from app.sommelier.voice import Voice

__all__ = ["SommRuntime", "SommSettings", "register_somm_routes", "somm_health", "somm_warning"]

logger = logging.getLogger(__name__)

Order = Literal["reco", "plain"]
NDJSON = "application/x-ndjson; charset=utf-8"
STREAM_HEADERS = {"Cache-Control": "no-store", "X-Accel-Buffering": "no"}


class AskBody(BaseModel):
    """Тело `POST /v1/sommelier/ask` (договор, §3.1). Ровно одно из `question` и `chip`."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    question: str | None = None
    chip: str | None = None
    args: dict[str, str] | None = None
    context: dict[str, Any] | None = None


def invalid(field: str, message: str) -> RequestValidationError:
    """422 в стандартном теле FastAPI — без эха присланного значения."""
    return RequestValidationError(
        [{"type": "value_error", "loc": ("body", field), "msg": message, "input": None}]
    )


#: Пути сомелье: у них 422 не повторяет присланное (договор, §1 «Журнал» и «Ошибки»).
SOMM_PATH = re.compile(r"^/v1/(?:sommelier/|wines/[^/]+/sommelier$)")
#: Поля ошибки pydantic, в которых может оказаться присланное значение: `input` — само значение,
#: `ctx` — его части и границы, `url` — ссылка на справку pydantic.
ECHO_KEYS = frozenset({"input", "ctx", "url"})
#: Что из `loc` можно вернуть: место в запросе и имена полей тела и строки запроса сомелье. Всё
#: остальное в `loc` прислал гость — лишний ключ тела (`extra_forbidden`), ключ `args` или
#: `context`, — и оно тоже не повторяется (повторная проверка серверной части).
SAFE_LOC = frozenset(
    {"body", "query", "path", "slug", "question", "chip", "args", "context", "order"}
)


def _safe_loc(loc: Any) -> list[str | int]:
    """`loc` без присланных ключей: `["body", "<ключ гостя>"]` → `["body"]`."""
    return [part for part in loc if isinstance(part, int) or part in SAFE_LOC]


def validation_handler(previous: Any) -> Any:
    """Обработчик 422: на путях сомелье — стандартное тело FastAPI без эха, иначе — `previous`.

    Стандартный 422 FastAPI кладёт в `detail[].input` присланное значение: вопрос-список или
    вопрос-число вернулся бы гостю и прокси в теле ответа. У сомелье остаются `type`, `loc` (без
    присланных ключей) и `msg`; остальные маршруты сервиса (и тело predict) отвечают как раньше.
    """

    async def handler(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, RequestValidationError) or not SOMM_PATH.match(request.url.path):
            return await previous(request, exc)
        detail = []
        for error in exc.errors():
            item = {key: value for key, value in dict(error).items() if key not in ECHO_KEYS}
            if "loc" in item:
                item["loc"] = _safe_loc(item["loc"])
            detail.append(item)
        return JSONResponse({"detail": jsonable_encoder(detail)}, status_code=422)

    return handler


# ------------------------------------------------------------------ состояние сомелье
class SommRuntime:
    """Настройки, данные, сомелье и голос одного приложения; всё собирается один раз.

    `data` и `voice` заданы — берутся как есть (тесты, зонды). Иначе их ставит `start` при
    старте сервиса, а без старта они собираются при первом вопросе.
    """

    def __init__(
        self,
        settings: SommSettings,
        *,
        data: SommData | None = None,
        voice: Any | None = None,
        safety: SafetyLayer | None = None,
    ) -> None:
        self.settings = settings
        self._data = data
        self._voice = voice
        #: Смысловой слой барьера; выключен (`SVS_SOMM_SAFETY=0`) — отказы только по правилам.
        self.safety = safety if safety is not None else SafetyLayer(enabled=settings.safety)
        self._lock = threading.Lock()
        self._sommelier: tuple[int, Sommelier] | None = None
        self._router: Router | None = None
        self.asked = 0

    def data(self) -> SommData:
        """Данные §7: `SVS_SOMM_DIR`, недостающие файлы — из заглушек договора."""
        if self._data is None:
            with self._lock:
                if self._data is None:
                    self._data = load_somm_data(self.settings.data_dir)
        return self._data

    def start(self, app: FastAPI, service: Any) -> None:
        """Старт сервиса: данные и голос готовы до первого вопроса.

        Данные — те же, что прочитал слой «после поиска» для «Сомелье у полки», если каталог
        один (`ServiceSettings.somm` — это и есть `self.settings`): 20 МБ пар не читаются
        дважды. Голос собирается сразу, чтобы `/v1/health.somm.voice` был виден с первой минуты.
        Смысловой слой (если включён) грузит модель в фоне: старт сервиса её не ждёт.
        """
        self.safety.start()
        if self._data is None:
            shared = _shared_data(service, self.settings)
            if shared is not None:
                with self._lock:
                    if self._data is None:
                        self._data = shared
        self.voice(app, service)

    def router(self) -> Router:
        if self._router is None:
            safety = self.safety if self.safety.enabled else None
            self._router = Router(self.data(), safety=safety)
        return self._router

    def sommelier(self, after: Any) -> Sommelier:
        """Сомелье поверх слоя «после поиска» сервиса; пересобирается, если слой другой."""
        cached = self._sommelier
        if cached is None or cached[0] != id(after):
            sommelier = Sommelier(
                self.data(),
                WineSource.of_after(after),
                live=bool(self.settings.live),
                input_enabled=bool(self.settings.input),
            )
            self._sommelier = cached = (id(after), sommelier)
        return cached[1]

    def voice(self, app: FastAPI, service: Any) -> Any:
        """Голос: заданный явно или собранный один раз (`build_voice`)."""
        if self._voice is None:
            data = self.data()
            with self._lock:
                if self._voice is None:
                    self._voice = build_voice(app, service, self.settings, data)
        return self._voice

    def stats(self, app: FastAPI) -> dict[str, Any]:
        """Блок `somm` для `/v1/health` (договор, §1): выключатели, данные, ворота, голос.

        `status` — `starting` (данные ещё не прочитаны), `degraded` (хоть один файл `data/somm`
        взят из заглушек или его нет: причины — в `degraded_reasons`) или `ok`.
        """
        data, voice = self._data, self._voice
        gate = getattr(app.state, "somm_gate", None)
        voice_stats = getattr(voice, "stats", None)
        reasons = data.degraded_reasons() if data is not None else []
        if data is None:
            status = "starting"
        else:
            status = "degraded" if reasons else "ok"
        return {
            "status": status,
            "degraded_reasons": reasons,
            "live": bool(self.settings.live),
            "input": bool(self.settings.input),
            "data": data.stats() if data is not None else None,
            "gate": gate.stats() if gate is not None else None,
            "voice": voice_stats() if callable(voice_stats) else None,
            "safety": dict(self.safety.stats()),
            "asked": self.asked,
        }


def _shared_data(service: Any, settings: SommSettings) -> SommData | None:
    """Данные сомелье слоя «после поиска» сервиса, если они из того же каталога."""
    after = getattr(service, "after", None)
    data = getattr(after, "somm", None)
    service_dir = getattr(getattr(service, "settings", None), "somm_dir", None)
    if not isinstance(data, SommData) or service_dir is None:
        return None
    try:
        same = Path(service_dir).resolve() == Path(settings.data_dir).resolve()
    except OSError:
        return None
    return data if same else None


def build_voice(app: FastAPI, service: Any, settings: SommSettings, data: SommData) -> Voice:
    """Живой голос (§6.3): ворота видеокарты приложения, клиент Ollama, замок сущностей.

    Опции клиента зеркалят читателя этикетки сервиса (`OllamaText.mirroring`): иначе Ollama
    перезагрузила бы модель, и скан заплатил бы 3–4 с. VLM выключен (или читатель не Ollama) —
    клиента нет, голос отвечает `not_ready`. Ворот у приложения нет (маршруты вне
    `create_app`) — они создаются без замка видеокарты и голос тоже не впускают.
    """
    gate = getattr(app.state, "somm_gate", None)
    if gate is None:
        gate = app.state.somm_gate = SommGate(quiet_s=settings.quiet_s)
    client: OllamaText | None = None
    locked = getattr(service, "vlm", None)
    if locked is not None:
        try:
            client = OllamaText.mirroring(locked)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Сомелье: клиент голоса не собран (%s) — шаблон", type(exc).__name__)
    return Voice(
        gate,
        client,
        EntityLock(data.vocab),
        live=settings.live,
        timeout_s=settings.timeout_s,
    )


def runtime_of(app: FastAPI) -> SommRuntime:
    return app.state.somm


def somm_health(app: FastAPI) -> dict[str, Any] | None:
    """Блок `somm` для `/v1/health`: выключатели, данные, ворота и голос; маршрутов нет — `None`."""
    runtime = getattr(app.state, "somm", None)
    return runtime.stats(app) if runtime is not None else None


def somm_warning(block: dict[str, Any] | None) -> str | None:
    """Предупреждение для общего списка `warnings` `/v1/health`, если данные сомелье — заглушки.

    Общий список читает проверка готовности `scripts/run_eval.sh`: без предупреждения сервис на
    заглушках (семь вин вместо 2 103) выглядел бы готовым (финальная проверка 24.09).
    """
    if not block or block.get("status") != "degraded":
        return None
    files = ", ".join(reason.removeprefix("data:") for reason in block["degraded_reasons"])
    return (
        f"Сомелье на неполных данных ({files}): соберите data/somm (scripts/build_somm.py) или "
        "задайте SVS_SOMM_DIR"
    )


# ------------------------------------------------------------------ маршруты
def register_somm_routes(
    app: FastAPI,
    *,
    settings: SommSettings | None = None,
    data: SommData | None = None,
    voice: Any | None = None,
    preload: bool = True,
) -> SommRuntime:
    """Маршруты сомелье; сервис — `app.state.service` (как у слоя «после поиска»).

    `settings` не заданы — из окружения (`SVS_SOMM_*`); неверный флаг — отказ старта.
    `preload` — прочитать данные в фоне сразу, чтобы первый вопрос не ждал загрузки; сервис
    (`create_app`) его выключает и отдаёт данные слоя «после поиска» в `SommRuntime.start`.
    """
    runtime = SommRuntime(settings or SommSettings.from_env(), data=data, voice=voice)
    app.state.somm = runtime
    previous = app.exception_handlers.get(
        RequestValidationError, request_validation_exception_handler
    )
    app.add_exception_handler(RequestValidationError, validation_handler(previous))
    if preload and data is None:
        threading.Thread(target=runtime.data, name="somm-data", daemon=True).start()

    def service(request: Request) -> Any:
        return getattr(request.app.state, "service", None)

    @app.get("/v1/wines/{slug}/sommelier")
    async def wine_sommelier(slug: str, request: Request, order: Order = "reco") -> JSONResponse:
        """Заметка, профиль, подача, блюда и чипы карточки (договор, §2). Без модели."""
        svc = service(request)
        if svc is None:
            return JSONResponse({"detail": "сервис ещё не собран"}, status_code=503)
        body = runtime_of(request.app).sommelier(svc.after).card(slug, order)
        if body is None:
            return JSONResponse({"detail": f"нет карточки {slug!r}"}, status_code=404)
        return JSONResponse(body)

    @app.post("/v1/sommelier/ask")
    async def sommelier_ask(payload: AskBody, request: Request, order: Order = "reco") -> Response:
        """Вопрос листа → поток NDJSON (договор, §3). Вопрос разбирают правила, не модель."""
        started = time.perf_counter()
        state = runtime_of(request.app)
        if (payload.question is None) == (payload.chip is None):
            raise invalid("question", "нужно ровно одно из question и chip")
        question: str | None = None
        if payload.question is not None:
            if not state.settings.input:
                return JSONResponse({"detail": t.INPUT_OFF}, status_code=422)
            question = clean_question(payload.question)
            if not question:
                raise invalid("question", "пустой вопрос")
            if len(question) > MAX_QUESTION:
                raise invalid("question", f"вопрос длиннее {MAX_QUESTION} символов")
        svc = service(request)
        if svc is None:
            log_ask("-", "not_ready", payload.chip, started)
            return stream_response(not_ready_events(started))
        sommelier = state.sommelier(svc.after)
        anchor = sommelier.anchor(payload.slug)
        if anchor is None:
            # slug в журнал не идёт: он не из каталога, а прислан как есть.
            log_ask("-", "no_card", payload.chip, started)
            return JSONResponse({"detail": f"нет карточки {payload.slug!r}"}, status_code=404)
        data = state.data()
        args = payload.args or {}
        if payload.chip is not None:
            check_chip(payload.chip, args, data)
        try:
            context = context_of(payload.context, data.dishes)
        except ContextError as exc:
            raise invalid("context", str(exc)) from None
        router = state.router()

        def route() -> Any:
            if payload.chip is not None:
                return router.chip(payload.chip, args, context)
            return router.text(question or "", context, grapes=anchor.grapes)

        voice = state.voice(request.app, svc)
        state.asked += 1
        events = ask_events(
            request,
            started=started,
            route=route,
            answer=lambda r: sommelier.answer(anchor, r, order, context),
            voice=voice,
            slug=anchor.slug,
            name=anchor.name,
            chip=payload.chip,
        )
        return stream_response(events)

    return runtime


def check_chip(chip_id: str, args: Mapping[str, str], data: Any) -> None:
    """Чип и его аргументы по таблице §4.2: неизвестный ключ или значение — 422."""
    if chip_id not in CHIP_ARGS:
        raise invalid("chip", "неизвестный чип")
    unknown = set(args) - CHIP_ARGS[chip_id]
    if unknown:
        raise invalid("args", "лишние аргументы чипа")
    missing = REQUIRED_ARGS.get(chip_id, frozenset()) - set(args)
    if missing:
        raise invalid("args", "не хватает аргументов чипа")
    if "dish" in args and args["dish"] not in data.dishes:
        raise invalid("args", "неизвестное блюдо")
    if "grape" in args and args["grape"] not in data.grapes:
        raise invalid("args", "неизвестный сорт")
    if "topic" in args and args["topic"] not in data.topics:
        raise invalid("args", "неизвестная тема")
    if "food" in args and args["food"] not in GUIDED_FOOD:
        raise invalid("args", "неизвестная группа блюд")
    if "want" in args and args["want"] not in GUIDED_WANT:
        raise invalid("args", "неизвестное направление")


# ------------------------------------------------------------------ поток
def stream_response(events: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(events, media_type=NDJSON, headers=STREAM_HEADERS)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _line(event: Mapping[str, Any], started: float) -> bytes:
    body = {**event, "t_ms": _ms(started)}
    return (json.dumps(body, ensure_ascii=False) + "\n").encode("utf-8")


def log_ask(slug: str, voice: str, chip: str | None, started: float) -> None:
    """Строка журнала вопроса, на который ответ не собирался (сервис не готов, нет карточки).

    Формат — тот же, что у `ask_events` (договор, §1 «Журнал»): одна строка на вопрос, без
    вопроса, контекста и присланного slug.
    """
    logger.info(
        "somm ask slug=%s intent=- input=%s chip=%s voice=%s guard=- facts_ms=-1 total_ms=%d",
        slug,
        "chip" if chip else "text",
        chip if chip in CHIP_ARGS else ("-" if chip is None else "?"),
        voice,
        _ms(started),
    )


async def not_ready_events(started: float) -> AsyncIterator[bytes]:
    yield _line({"type": "error", "code": "not_ready", "text": t.ERROR_TEXTS["not_ready"]}, started)
    yield _line({"type": "done"}, started)


def _frames(exc: BaseException) -> str:
    """Кадры стека без сообщения исключения: в сообщении могло оказаться слово вопроса."""
    return "".join(traceback.format_tb(exc.__traceback__))


def text_event(package: FactsPackage, result: Any) -> dict[str, Any]:
    """Событие `text`: проверенный текст модели или шаблон с причиной (договор, §3.3)."""
    generated = bool(getattr(result, "generated", False))
    text = str(getattr(result, "text", "") or "")
    reason = getattr(result, "reason", None)
    guard = getattr(result, "guard", None)
    if generated and not check(text).clean:
        # Последняя сетка: проверки голоса пропустили нарушение — показываем шаблон.
        generated, reason, guard = False, "guard", "legal"
    if generated:
        return {
            "type": "text",
            "text": text,
            "generated": True,
            "label": t.LABEL_AI,
            "reason": None,
            "guard": None,
        }
    return {
        "type": "text",
        "text": package.verdict_template,
        "generated": False,
        "label": t.LABEL_ALGO,
        "reason": reason or "error",
        "guard": guard if reason == "guard" else None,
    }


async def ask_events(
    request: Request,
    *,
    started: float,
    route: Any,
    answer: Any,
    voice: Any,
    slug: str,
    name: str,
    chip: str | None,
) -> AsyncIterator[bytes]:
    """Этапы сборки → `facts` → этапы голоса → `text` → `done`; ошибка сборки → `error`."""
    record = {
        "intent": "-",
        "input": "chip" if chip else "text",
        "chip": chip or "-",
        "voice": "-",
        "guard": "-",
        "facts_ms": -1,
    }
    try:
        try:
            # В потоке: смысловой слой барьера считает на CPU и может ждать загрузки модели.
            chosen = await asyncio.to_thread(route)
            record["intent"] = chosen.intent
            builder = answer(chosen)
            while True:
                stage_id = next(builder)
                yield _line(t.stage(stage_id, name), started)
        except StopIteration as stop:
            package: FactsPackage = stop.value
        except Exception as exc:  # noqa: BLE001 — ответ не собран: событие error
            logger.error("somm ask: ответ не собран (%s)\n%s", type(exc).__name__, _frames(exc))
            record["voice"] = "error"
            yield _line(
                {"type": "error", "code": "internal", "text": t.ERROR_TEXTS["internal"]}, started
            )
            yield _line({"type": "done"}, started)
            return
        record["facts_ms"] = _ms(started)
        yield _line({"type": "facts", **package.public}, started)

        queue: asyncio.Queue[str] = asyncio.Queue()

        async def on_stage(stage_id: str) -> None:
            await queue.put(stage_id)

        task = asyncio.ensure_future(
            voice.speak(package, on_stage=on_stage, disconnected=request.is_disconnected)
        )
        getter: asyncio.Future[str] | None = None
        try:
            while True:
                getter = asyncio.ensure_future(queue.get())
                done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter in done:
                    yield _line(t.stage(getter.result(), name), started)
                    continue
                getter.cancel()
                if getter.done() and not getter.cancelled():
                    yield _line(t.stage(getter.result(), name), started)
                break
            while not queue.empty():
                yield _line(t.stage(queue.get_nowait(), name), started)
            try:
                result = task.result()
            except Exception as exc:  # noqa: BLE001 — голос не бросает, но договор важнее
                logger.error("somm ask: голос упал (%s)\n%s", type(exc).__name__, _frames(exc))
                result = None
        finally:
            # Клиент ушёл посреди ожидания: и голос, и ожидание этапа отменяются сразу, а не
            # остаются висеть до сборки мусора.
            if getter is not None and not getter.done():
                getter.cancel()
            if not task.done():
                task.cancel()
                record["voice"] = "cancelled"
        event = text_event(package, result)
        record["voice"] = "ok" if event["generated"] else str(event["reason"])
        record["guard"] = event["guard"] or "-"
        yield _line(event, started)
        yield _line({"type": "done"}, started)
    finally:
        logger.info(
            "somm ask slug=%s intent=%s input=%s chip=%s voice=%s guard=%s facts_ms=%d total_ms=%d",
            slug,
            record["intent"],
            record["input"],
            record["chip"],
            record["voice"],
            record["guard"],
            record["facts_ms"],
            _ms(started),
        )
