"""HTTP-сервис сканера: FastAPI поверх `ScannerService`.

    POST /v1/eval/predict   multipart `image` → ВСЕГДА HTTP 200 и JSON со `slug` наверху
                            (строка или null). Контракт `participant_test.sh`: засчитывается
                            только 200/201 и непустая строка `.slug`, всё остальное — null.
    POST /v1/scan           то же, плюс `evidence` (кандидаты CV, чтение, resolve), `card`,
                            `candidates` (top-5 без счётов) и `after` (какой экран открыть)
    GET  /v1/health         starting | ready | degraded, модели, индекс, прогрев

Слой «после поиска» (`app.api.after`, договор `docs/api-after-search.md`): карточка
`/v1/wines/{slug}`, фото, похожие из других виноделен, «Не тупик», «Сомелье у полки». Страница
продукта (`app.api.page`) — на `/`, её файлы оформления — под `/static`.

Голос сомелье (`app.sommelier`, договор `docs/api-sommelier.md`) делит с чтением этикетки
видеокарту и одну модель Ollama, поэтому скан всегда первый: ворота `SommGate` считают сканы
вокруг `run_scan` и держат тихое окно после каждого `/v1/eval/predict`, а голос берёт замок
видеокарты только без ожидания и уходит по первому скану. Это два хука в пути predict; замок
видеокарты и тело predict они не меняют. Маршруты сомелье — `app.api.somm`, их настройки
(`SVS_SOMM_*`) — `ServiceSettings.somm`, а блок `somm` в `/v1/health` — выключатели, данные,
ворота и голос.

Поверх — полевой контур (`app.api.field`): страница съёмки на `/field`, приём кадра с записью
в архив и отметка «верно / не то». Он вынесен в отдельный модуль, потому что боевой контракт
менять нельзя: `/v1/eval/predict` и `/v1/scan` на диск ничего не пишут и про архив не знают.
Контур включается только флагом `SVS_FIELD=1`: без него маршрутов записи кадров нет вовсе, а
при нём старые ссылки стенда `/?k=…` перенаправляются на `/field?k=…`.

Прогрев идёт в lifespan: uvicorn не принимает запросы, пока он не кончился, поэтому первый
кадр скрипта уже попадает на тёплые модели. Сервис — один процесс и один воркер: модели в
памяти и замок видеокарты общие.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.after import register_after_routes
from app.api.config import ServiceSettings
from app.api.field import FieldSettings, register_field_routes
from app.api.page import register_page_routes
from app.api.service import ScannerService, ScanResult, failure
from app.api.somm import register_somm_routes, somm_health, somm_warning
from app.sommelier.gate import SommGate
from app.sommelier.settings import SommSettings

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
#: Поле multipart с кадром — как в скрипте организатора.
IMAGE_FIELD = "image"
#: Запас на заголовки multipart поверх предела файла при проверке Content-Length.
MULTIPART_SLACK = 64 * 1024


#: Маршруты полевого контура, которые принимают кадр: на время приёма голос уходит, как у скана.
FIELD_SCAN_PATHS = frozenset({"/v1/field/scan", "/v1/field/submit"})


class BodyTooLarge(Exception):
    """Тело запроса без Content-Length (chunked) переросло предел — разбор обрывается."""


class LimitedBody:
    """`receive` запроса, который считает байты тела и обрывает чтение за пределом.

    Content-Length отсекает большой запрос до разбора, но у `Transfer-Encoding: chunked` его
    нет: без счёта Starlette разобрал бы весь гигабайт во временный файл и только потом
    получил бы `too_large`. Здесь чтение обрывается, как только тело переросло `limit`.
    """

    def __init__(self, receive: Receive, limit: int) -> None:
        self._receive = receive
        self.limit = limit
        self.seen = 0
        self.exceeded = False

    async def __call__(self) -> Message:
        message = await self._receive()
        if message["type"] == "http.request":
            self.seen += len(message.get("body", b""))
            if self.seen > self.limit:
                self.exceeded = True
                raise BodyTooLarge(f"тело больше {self.limit} байт")
        return message


class FieldScanGate:
    """ASGI-обёртка полевого контура: приём кадра считается сканом для ворот сомелье.

    Полевые маршруты (`app.api.field`) зовут `svc.scan` сами, мимо `run_scan`. Обёртка трогает
    только `POST` из `FIELD_SCAN_PATHS` и только при `SVS_FIELD=1`; остальные запросы проходят
    насквозь. Тихого окна у поля нет: его держит только predict скрипта организатора.
    """

    def __init__(self, app: ASGIApp, *, gate: SommGate) -> None:
        self.app = app
        self.gate = gate

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in FIELD_SCAN_PATHS
        ):
            await self.app(scope, receive, send)
            return
        self.gate.scan_started()
        try:
            await self.app(scope, receive, send)
        finally:
            self.gate.scan_finished()


def somm_settings_of(
    settings: ServiceSettings | None, service: ScannerService | None
) -> SommSettings:
    """Настройки сомелье приложения — из тех же настроек, по которым собран сервис.

    Так каталог данных сомелье у маршрутов и у «Сомелье у полки» слоя «после поиска» один и тот
    же, а переменные `SVS_SOMM_*` читаются в одном месте (`ServiceSettings.from_env`).
    """
    for candidate in (settings, getattr(service, "settings", None)):
        if isinstance(candidate, ServiceSettings):
            return candidate.somm
    return ServiceSettings.from_env().somm


async def read_upload(request: Request, max_bytes: int) -> tuple[bytes | None, str | None]:
    """Байты файла из поля `image` или текст ошибки. Исключений нет.

    Предел тела — файл плюс `MULTIPART_SLACK` на заголовки частей: по Content-Length до
    разбора, а без него (chunked) — счётом байт по ходу разбора.
    """
    limit = max_bytes + MULTIPART_SLACK
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        return None, f"too_large: запрос {int(length)} байт, предел файла {max_bytes}"
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        return None, f"bad_request: ожидается multipart/form-data с полем {IMAGE_FIELD!r}"
    body = LimitedBody(request.receive, limit)
    try:
        form = await Request(request.scope, body).form(max_files=4, max_fields=16)
    except Exception as exc:  # noqa: BLE001 — битый multipart — ответ с ошибкой, а не 400
        if body.exceeded:
            return None, f"too_large: запрос больше {limit} байт, предел файла {max_bytes}"
        return None, f"bad_request: multipart не разобрался: {exc}"
    try:
        item = form.get(IMAGE_FIELD)
        if not isinstance(item, UploadFile):
            return None, f"no_image: нет файла в поле {IMAGE_FIELD!r}"
        data = await item.read(max_bytes + 1)
    except Exception as exc:  # noqa: BLE001
        return None, f"bad_request: файл не читается: {exc}"
    finally:
        await form.close()
    if len(data) > max_bytes:
        return None, f"too_large: файл больше {max_bytes} байт"
    return data, None


def create_app(
    settings: ServiceSettings | None = None,
    service: ScannerService | None = None,
    *,
    warm: bool = True,
    field: FieldSettings | None = None,
) -> FastAPI:
    """Приложение. `service` не задан — собирается из `settings` (или окружения) при старте.

    Отказ сборки (`StartupError`) роняет старт: uvicorn не поднимет порт с чужой моделью.
    `warm=False` — без прогрева (тесты): `/v1/health` тогда отвечает `starting`.
    `field` не задан — полевой контур берёт настройки из окружения (`SVS_FIELD`,
    `SVS_FIELD_*`), а каталог архива — рядом с индексом, в `<data>/field`. Маршруты контура
    регистрируются только при `field.enabled` (`SVS_FIELD=1`).

    Ворота сомелье (`app.state.somm_gate`) создаются здесь, чтобы хуки скана работали и до сборки
    сервиса; замок видеокарты они получают в lifespan, когда сервис собран и прогрет. Тихое
    окно — `SVS_SOMM_QUIET_S` (15 с по договору). Там же сомелье получает данные слоя «после
    поиска» и собирает голос (`SommRuntime.start`).
    """
    somm = somm_settings_of(settings, service)
    gate = SommGate(quiet_s=somm.quiet_s)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc = service
        if svc is None:
            svc = await run_in_threadpool(
                ScannerService.load, settings or ServiceSettings.from_env()
            )
        app.state.service = svc
        if warm:
            await run_in_threadpool(svc.warm)
        # После прогрева: голос не входит, пока сервис не собран, и не спорит с прогревом.
        gate.bind(svc.gpu_lock)
        await run_in_threadpool(somm_runtime.start, app, svc)
        yield

    app = FastAPI(
        title="Сканер вин «Своё Вино»",
        version=API_VERSION,
        summary="Фото бутылки → slug карточки каталога",
        lifespan=lifespan,
    )
    app.state.somm_gate = gate

    def current(request: Request) -> ScannerService | None:
        return getattr(request.app.state, "service", None)

    async def run_scan(request: Request) -> tuple[ScannerService | None, ScanResult]:
        # Скан первым делом обрывает голос сомелье: тот уходит, пока грузится кадр.
        gate.scan_started()
        try:
            svc = current(request)
            if svc is None:
                return None, failure("not_ready: сервис ещё не собран")
            data, error = await read_upload(request, svc.settings.max_upload_bytes)
            if error is not None or data is None:
                result = failure(error or "no_image")
                svc.stats.add(result)
                return svc, result
            try:
                return svc, await run_in_threadpool(svc.scan, data)
            except Exception as exc:  # scan не бросает, но контракт важнее
                logger.exception("Скан упал вне сервиса")
                return svc, failure(f"internal: {type(exc).__name__}: {exc}")
        finally:
            gate.scan_finished()

    @app.post("/v1/eval/predict")
    async def predict(request: Request) -> JSONResponse:
        """Кадр → top-1 slug. Всегда 200: скрипт организатора пишет null на любой ответ без slug."""
        try:
            _, result = await run_scan(request)
            body = result.predict_body()
        except Exception as exc:
            logger.exception("predict упал")
            body = failure(f"internal: {type(exc).__name__}: {exc}").predict_body()
        finally:
            # Скрипт организатора шлёт кадры подряд: голос ждёт тихое окно после каждого.
            gate.predict_done()
        return JSONResponse(body, status_code=200)

    @app.post("/v1/scan")
    async def scan(request: Request) -> JSONResponse:
        """Кадр → ответ с доказательствами, карточкой, кандидатами и подсказкой экрана. Всегда 200.

        Карточка (`GET /v1/wines/{slug}`) и остальные маршруты слоя «после поиска» — в
        `app.api.after`.
        """
        try:
            svc, result = await run_scan(request)
            body = svc.scan_body(result) if svc is not None else result.scan_body(None)
        except Exception as exc:
            logger.exception("scan упал")
            body = failure(f"internal: {type(exc).__name__}: {exc}").scan_body(None)
        return JSONResponse(body, status_code=200)

    @app.get("/v1/health")
    async def health(request: Request) -> JSONResponse:
        svc = current(request)
        body: dict[str, Any] = (
            svc.health()
            if svc is not None
            else {
                "status": "starting",
                "model": None,
                "index": None,
                "vlm": None,
                "warmed_at": None,
            }
        )
        body["somm"] = somm_health(request.app)
        # Сомелье на заглушках — предупреждение в общий список: его видит run_eval.sh.
        warning = somm_warning(body["somm"])
        if warning is not None and svc is not None:
            body.setdefault("warnings", []).append(warning)
        return JSONResponse(body, status_code=200)

    # Слой «после поиска»: карточка, фото, похожие, «Не тупик», «Сомелье у полки» — для страницы.
    register_after_routes(app)
    # «Сомелье»: карточка и лист (docs/api-sommelier.md). Данные и голос — в lifespan.
    somm_runtime = register_somm_routes(app, settings=somm, preload=False)

    # По умолчанию полевой контур выключен (`SVS_FIELD`): `/v1/field/*` пишет кадры на диск, а
    # продукт кадров не хранит (152-ФЗ). На полевом VPS флаг включается в env.
    data_dir = (settings or ServiceSettings.from_env()).index_path.parent.parent
    field_settings = field or FieldSettings.from_env(os.environ, data_dir)

    # Страница продукта на `/` открыта всегда. При включённом контуре старые ссылки стенда
    # `/?k=…` уходят на `/field?k=…`.
    register_page_routes(app, field_redirect=field_settings.enabled)

    # Полевой контур регистрируется последним: боевые маршруты выше не зависят от него, и если
    # страница или архив недоступны, сервис для скрипта организатора всё равно поднимется.
    if field_settings.enabled:
        register_field_routes(app, field_settings, read_upload=read_upload)
        app.add_middleware(FieldScanGate, gate=gate)
    else:
        logger.info("полевой контур выключен: SVS_FIELD не задан, /v1/field/* не регистрируются")
    return app
