"""Полевой контур: страница съёмки, архив кадров и отметки «верно / не то».

Зачем отдельно от `/v1/scan`. Скрипт организатора и интерфейс жюри работают с одним кадром и
ничего не помнят. Полевой прогон — противоположность: его ценность не в ответе на кадр, а в
том, что кадр и ответ **сохранились** и потом лягут в замер (главный рычаг точности —
больше настоящих фото новых вин). Поэтому здесь к той же цепочке
добавлены три вещи, которых в боевом контракте быть не должно:

* кадр пишется на диск как есть, байт в байт, вместе с полным ответом сервиса;
* человек в магазине сразу отмечает, то вино нашлось или нет, — разметка появляется на месте,
  а не восстанавливается по памяти через неделю;
* страница живёт на том же адресе, что API, поэтому CORS не нужен.

Архив (`SVS_FIELD_DIR`, по умолчанию `<data>/field`):

    2026-09-20/143712-a1b2c3d4.jpg    кадр как прислан (расширение — по сигнатуре байтов)
    2026-09-20/143712-a1b2c3d4.json   ответ сервиса целиком + карточка + мета
    index.jsonl                       по строке на скан
    verdicts.jsonl                    по строке на отметку человека

`index.jsonl` и `verdicts.jsonl` только дописываются: отметку можно поменять, старая строка
остаётся. Разбор полевого прогона берёт последнюю запись по `id`.

Доступ. Сервис авторизации не имеет, а полевой стенд смотрит в интернет. `SVS_FIELD_TOKEN`
закрывает страницу и полевые маршруты общим ключом: без него `/v1/field/*` отдаёт 401, а
ссылка для съёмки выглядит как `http://адрес/field?k=КЛЮЧ` (прежняя `http://адрес/?k=КЛЮЧ`
перенаправляется туда же: на `/` теперь страница продукта). Пустой токен — открыто (локальная
отладка). Это защита от случайного прохожего и сканера портов, а не от целенаправленной атаки:
по http ключ идёт открытым текстом.

Включение. Контур пишет кадры людей на диск, а продукт этого делать не должен (152-ФЗ):
по умолчанию маршруты контура не регистрируются вовсе. Их включает `SVS_FIELD=1` — на полевом
VPS он стоит в `deploy/svs-field.env.example`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from app.api.after_layer import image_type
from app.normalize.decode import sniff_format

logger = logging.getLogger(__name__)

#: Страница съёмки: один файл без сборки и без внешних адресов — VPS в интернет не ходит.
PAGE_PATH = Path(__file__).with_name("static") / "field.html"

#: Расширение файла кадра по сигнатуре. Незнакомая сигнатура — `.bin`: кадр всё равно сохраняем,
#: даже если сервис его не разобрал, иначе потеряется именно тот случай, ради которого стенд и
#: поднимали.
_EXTENSIONS = {
    "jpeg": ".jpg",
    "png": ".png",
    "webp": ".webp",
    "heic": ".heic",
    "avif": ".avif",
    "gif": ".gif",
    "bmp": ".bmp",
    "tiff": ".tiff",
}

#: Что человек может сказать про ответ. `unknown` — «сам не знаю», тоже полезная разметка:
#: такой кадр пойдёт на разбор глазами, а не в замер.
VERDICTS = frozenset({"ok", "wrong", "unknown"})

#: Сколько кадров ждут очереди. Пачка из магазина — обычный случай: снять двадцать бутылок
#: быстрее, чем разобрать одну, поэтому предел считается от пачки, а не от одного человека.
#: Кадры ждут очереди в памяти (до 25 МБ каждый), отсюда и потолок.
MAX_PENDING = 24
#: Сколько разборов помнить. Страница опрашивает свой и забывает; память не резиновая.
KEEP_JOBS = 50


def _stamp() -> tuple[str, str, str]:
    """Дата, время суток и отметка времени скана — в UTC, как всё остальное время сервиса.

    Каталог дня поэтому переключается в 03:00 по Москве, а не в полночь: вечерние кадры
    лягут в папку следующего дня. Так сделано осознанно — иначе отметка в архиве и счётчик
    «снято сегодня» на странице (он считает по UTC) разошлись бы на три часа.
    """
    now = datetime.now(UTC)
    date = now.strftime("%Y-%m-%d")
    clock = now.strftime("%H%M%S")
    return date, clock, now.replace(microsecond=0).isoformat()


#: Значения `SVS_FIELD`, которые включают контур.
FIELD_ON = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class FieldSettings:
    """Настройки полевого контура: включён ли, каталог архива, ключ, откуда брать фото каталога."""

    directory: Path
    token: str = ""
    photo_dir: Path | None = None
    #: `SVS_FIELD=1`. Выключен — маршруты контура не регистрируются, кадры не пишутся.
    enabled: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str], data_dir: Path) -> FieldSettings:
        raw_dir = (env.get("SVS_FIELD_DIR") or "").strip()
        raw_photos = (env.get("SVS_FIELD_PHOTO_DIR") or "").strip()
        return cls(
            directory=Path(raw_dir).expanduser().resolve() if raw_dir else data_dir / "field",
            token=(env.get("SVS_FIELD_TOKEN") or "").strip(),
            photo_dir=Path(raw_photos).expanduser().resolve() if raw_photos else None,
            enabled=(env.get("SVS_FIELD") or "").strip().lower() in FIELD_ON,
        )


class FieldArchive:
    """Кадры и ответы на диске. Пишет под замком: сканы идут по одному, но отметки — нет."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()
        # Каталог создаётся при первой записи, а не при сборке приложения: тесты поднимают
        # маршруты десятками раз, и мусорить каталогами на каждый вызов create_app незачем.

    @property
    def index_path(self) -> Path:
        return self.directory / "index.jsonl"

    @property
    def verdicts_path(self) -> Path:
        return self.directory / "verdicts.jsonl"

    def _append(self, path: Path, row: Mapping[str, Any]) -> None:
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save(
        self,
        data: bytes,
        body: Mapping[str, Any],
        *,
        note: str = "",
        client: str = "",
    ) -> str:
        """Кадр и ответ на диск. Возвращает идентификатор, по которому придёт отметка."""
        date, clock, at = _stamp()
        scan_id = f"{clock}-{uuid.uuid4().hex[:8]}"
        day = self.directory / date
        day.mkdir(parents=True, exist_ok=True)
        suffix = _EXTENSIONS.get(sniff_format(data) or "", ".bin")
        frame = day / f"{scan_id}{suffix}"
        frame.write_bytes(data)
        record = {
            "id": scan_id,
            "at": at,
            "date": date,
            "file": f"{date}/{frame.name}",
            "bytes": len(data),
            "note": note,
            "client": client,
            "response": dict(body),
        }
        (day / f"{scan_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        confidence = body.get("confidence") or {}
        self._append(
            self.index_path,
            {
                "id": scan_id,
                "at": at,
                "file": record["file"],
                "bytes": len(data),
                "slug": body.get("slug"),
                "outcome": body.get("outcome"),
                "p_top1": confidence.get("top1"),
                "degraded": body.get("degraded") or [],
                "total_ms": (body.get("timings_ms") or {}).get("total"),
                "top5": [item.get("slug") for item in (body.get("top5") or [])],
                "client": client,
            },
        )
        return scan_id

    def verdict(self, scan_id: str, verdict: str, *, slug: str = "", note: str = "") -> None:
        """Отметка человека. Строка дописывается, прежние не трогаются."""
        _, _, at = _stamp()
        self._append(
            self.verdicts_path,
            {"id": scan_id, "at": at, "verdict": verdict, "correct_slug": slug, "note": note},
        )

    def stats(self) -> dict[str, Any]:
        """Счётчики за всё время архива: сколько снято, сколько размечено, сколько «не то»."""
        scans = 0
        with_slug = 0
        days: dict[str, int] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scans += 1
                with_slug += bool(row.get("slug"))
                day = str(row.get("at", ""))[:10]
                days[day] = days.get(day, 0) + 1
        marks: dict[str, str] = {}
        if self.verdicts_path.exists():
            for line in self.verdicts_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                marks[str(row.get("id"))] = str(row.get("verdict"))
        counts = {name: sum(v == name for v in marks.values()) for name in sorted(VERDICTS)}
        marked = len(marks)
        return {
            "scans": scans,
            "with_slug": with_slug,
            "marked": marked,
            "verdicts": counts,
            "accuracy": round(counts["ok"] / marked, 3) if marked else None,
            "by_day": dict(sorted(days.items())),
            "directory": str(self.directory),
        }


@dataclass
class FieldJob:
    """Один разбор в очереди: что с ним сейчас и что получилось."""

    id: str
    submitted: float
    status: str = "queued"  # queued | running | done | error
    started: float | None = None
    finished: float | None = None
    body: dict[str, Any] | None = None
    error: str | None = None
    who: str = ""

    def public(self, *, ahead: int = 0) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "job_id": self.id,
            "status": self.status,
            "queued_ahead": ahead,
            "elapsed_ms": round(((self.finished or now) - self.submitted) * 1000),
            "error": self.error,
            **(self.body or {}),
        }


class FieldQueue:
    """Очередь разборов: кадр принимается сразу, ответ забирают опросом.

    Зачем не синхронно. Полевой стенд стоит за быстрым туннелем Cloudflare, а тот рвёт запрос
    на сотой секунде (524). Кадр на четырёх ядрах идёт 40–90 с, и это ещё без очереди: сервис
    однопоточный, второй снимающий ждёт первого. Синхронный ответ в такие рамки не влезает —
    человек получил бы ошибку на уже посчитанный кадр. Поэтому отправка и результат разведены:
    POST возвращает номер сразу, страница опрашивает его короткими запросами.

    Разбирает один поток: модели и замок всё равно общие, параллелить нечего.
    """

    def __init__(self, archive: FieldArchive, *, max_pending: int = MAX_PENDING) -> None:
        self.archive = archive
        self.max_pending = max_pending
        self._queue: queue.Queue[tuple[str, Any, bytes, str, str]] = queue.Queue()
        self._jobs: OrderedDict[str, FieldJob] = OrderedDict()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="field-queue", daemon=True)
            self._worker.start()

    @property
    def pending(self) -> int:
        with self._lock:
            return sum(job.status in {"queued", "running"} for job in self._jobs.values())

    def submit(
        self, service: Any, data: bytes, *, who: str = "", note: str = ""
    ) -> FieldJob | None:
        """Кадр в очередь. `None` — очередь переполнена, пусть снимет ещё раз попозже."""
        if self.pending >= self.max_pending:
            return None
        job = FieldJob(id=uuid.uuid4().hex[:12], submitted=time.monotonic(), who=who)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > KEEP_JOBS:
                self._jobs.popitem(last=False)
        self._queue.put((job.id, service, data, who, note))
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> tuple[FieldJob, int] | None:
        """Разбор и сколько кадров стоит перед ним в очереди."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status != "queued":
                return job, 0
            ahead = 0
            for other in self._jobs.values():
                if other.id == job.id:
                    break
                if other.status in {"queued", "running"}:
                    ahead += 1
            return job, ahead

    def _run(self) -> None:
        while True:
            job_id, service, data, who, note = self._queue.get()
            job = self.get(job_id)
            if job is None:  # вытеснен из памяти, пока стоял в очереди
                continue
            item = job[0]
            item.status = "running"
            item.started = time.monotonic()
            try:
                result = service.scan(data)
                body = service.scan_body(result)
                try:
                    scan_id = self.archive.save(data, body, note=note, client=who)
                except OSError:
                    logger.exception("Кадр не сохранился")
                    scan_id = ""
                item.body = {**body, "field_id": scan_id, "archived": bool(scan_id)}
                item.status = "done"
            except Exception as exc:  # наружу ничего не летит: это фоновый поток
                logger.exception("Разбор кадра упал")
                item.error = f"internal: {type(exc).__name__}: {exc}"
                item.status = "error"
            finally:
                item.finished = time.monotonic()


def register_field_routes(
    app: FastAPI,
    settings: FieldSettings,
    *,
    read_upload: Any,
) -> FieldArchive:
    """Полевые маршруты поверх собранного приложения.

    `read_upload` передаётся из `app.api.main`, чтобы предел размера и разбор multipart были
    ровно те же, что у боевых маршрутов: полевой кадр не должен приниматься по другим правилам.
    """
    archive = FieldArchive(settings.directory)
    jobs = FieldQueue(archive)

    def allowed(request: Request) -> bool:
        if not settings.token:
            return True
        given = request.headers.get("x-field-key") or request.query_params.get("k") or ""
        return given == settings.token

    def denied() -> JSONResponse:
        return JSONResponse({"error": "нужен ключ: добавьте ?k=… к адресу"}, status_code=401)

    @app.get("/field", include_in_schema=False)
    async def page(request: Request) -> Response:
        """Страница съёмки. Ключ проверяется здесь же: без него страница не открывается.

        На `/` теперь страница продукта (`app.api.page`); старые ссылки `/?k=…` она
        перенаправляет сюда, пока контур включён.
        """
        if not allowed(request):
            return HTMLResponse("<h1>Нужен ключ</h1><p>Откройте ссылку целиком, с «?k=…».</p>", 401)
        if not PAGE_PATH.exists():
            return HTMLResponse("<h1>Страница не собрана</h1>", status_code=500)
        # Без кэша: страницу правят по ходу полевого прогона, а браузер отдаёт старую копию
        # молча — человек жмёт на кнопки, которых в ней ещё нет, и не понимает, почему
        # поведение не то, что описано.
        return HTMLResponse(
            PAGE_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    @app.post("/v1/field/scan")
    async def field_scan(request: Request) -> JSONResponse:
        """Кадр из поля: та же цепочка, что `/v1/scan`, плюс запись кадра и ответа на диск."""
        if not allowed(request):
            return denied()
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            return JSONResponse({"error": "сервис ещё не собран"}, status_code=503)
        data, error = await read_upload(request, svc.settings.max_upload_bytes)
        if error is not None or data is None:
            return JSONResponse({"error": error or "no_image"}, status_code=400)
        result = await run_in_threadpool(svc.scan, data)
        body = svc.scan_body(result)
        client = request.query_params.get("who", "")[:40]
        note = request.query_params.get("note", "")[:200]
        try:
            scan_id = await run_in_threadpool(archive.save, data, body, note=note, client=client)
        except OSError:
            logger.exception("Кадр не сохранился")
            scan_id = ""
        return JSONResponse({**body, "field_id": scan_id, "archived": bool(scan_id)})

    @app.post("/v1/field/submit")
    async def field_submit(request: Request) -> JSONResponse:
        """Кадр в очередь: ответ приходит сразу, результат забирают по `/v1/field/job/{id}`.

        Этим путём ходит страница. Через быстрый туннель Cloudflare синхронный разбор не
        проходит: туннель рвёт запрос на сотой секунде, а кадр на процессоре идёт дольше.
        """
        if not allowed(request):
            return denied()
        svc = getattr(request.app.state, "service", None)
        if svc is None:
            return JSONResponse({"error": "сервис ещё не собран"}, status_code=503)
        data, error = await read_upload(request, svc.settings.max_upload_bytes)
        if error is not None or data is None:
            return JSONResponse({"error": error or "no_image"}, status_code=400)
        job = jobs.submit(
            svc,
            data,
            who=request.query_params.get("who", "")[:40],
            note=request.query_params.get("note", "")[:200],
        )
        if job is None:
            return JSONResponse(
                {"error": f"очередь занята: ждут {jobs.pending} кадров, попробуйте через минуту"},
                status_code=429,
            )
        found = jobs.get(job.id)
        return JSONResponse(job.public(ahead=found[1] if found else 0), status_code=202)

    @app.get("/v1/field/job/{job_id}")
    async def field_job(job_id: str, request: Request) -> JSONResponse:
        """Как там кадр: `queued` (и сколько перед ним), `running`, `done` со всем ответом, `error`."""
        if not allowed(request):
            return denied()
        found = jobs.get(job_id)
        if found is None:
            return JSONResponse({"error": "такого разбора нет"}, status_code=404)
        job, ahead = found
        return JSONResponse(job.public(ahead=ahead))

    @app.post("/v1/field/verdict")
    async def field_verdict(request: Request) -> JSONResponse:
        """Отметка «верно / не то / не знаю» к сохранённому скану."""
        if not allowed(request):
            return denied()
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 — тело от телефона, любое
            return JSONResponse({"error": "ожидается JSON"}, status_code=400)
        scan_id = str(payload.get("id") or "").strip()
        verdict = str(payload.get("verdict") or "").strip()
        if not scan_id or verdict not in VERDICTS:
            return JSONResponse(
                {"error": f"нужны id и verdict из {sorted(VERDICTS)}"}, status_code=400
            )
        await run_in_threadpool(
            archive.verdict,
            scan_id,
            verdict,
            slug=str(payload.get("correct_slug") or "").strip()[:200],
            note=str(payload.get("note") or "").strip()[:500],
        )
        return JSONResponse({"ok": True})

    @app.get("/v1/field/stats")
    async def field_stats(request: Request) -> JSONResponse:
        """Счётчики прогона: сколько кадров снято и как они размечены."""
        if not allowed(request):
            return denied()
        return JSONResponse(await run_in_threadpool(archive.stats))

    @app.get("/v1/field/photo/{slug}", include_in_schema=False)
    async def field_photo(slug: str, request: Request) -> Response:
        """Эталонное фото каталога: человек сверяет его с бутылкой в руке.

        Источники те же, что у `/v1/wines/{slug}/photo` (`ScannerService.photo_file`): лёгкая
        пересжатая копия `<SVS_FIELD_PHOTO_DIR>/<slug>.webp` (её кладёт
        `scripts/make_photo_pack.py` — весь каталог умещается в десяток мегабайт), тот же
        каталог по имени файла из выгрузки, путь из `slug_photo_map.csv` и путь справочника.
        Путь из выгрузки ведёт на машину разработки, поэтому на сервере он почти всегда мимо,
        а локально работает без всякой подготовки.
        """
        if not allowed(request):
            return denied()
        svc = getattr(request.app.state, "service", None)
        if svc is None or svc.card(slug) is None:
            return JSONResponse({"error": "нет карточки"}, status_code=404)
        extra = [settings.photo_dir] if settings.photo_dir else []
        path = svc.photo_file(slug, extra_dirs=extra)
        if path is not None:
            return FileResponse(
                path,
                media_type=image_type(path),
                headers={"Cache-Control": "public, max-age=86400"},
            )
        return JSONResponse({"error": "фото каталога нет на этой машине"}, status_code=404)

    logger.info(
        "полевой контур: архив %s, ключ %s",
        settings.directory,
        "задан" if settings.token else "не задан (открыто)",
    )
    return archive
