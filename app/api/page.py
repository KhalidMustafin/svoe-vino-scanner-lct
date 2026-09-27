"""Страница продукта: `/` и её файлы под `/static`.

Страница `static/index.html` — файл без сборки: фото бутылки → карточка каталога с сомелье,
«Не то вино?», похожие из других виноделен и «Моего вина здесь нет». Лист сомелье живёт рядом, в
`static/somm.js` и `static/somm.css` (договор `docs/api-sommelier.md`, §8), и тоже без сборки.
Страница ходит только в `/v1/scan`, справочные маршруты слоя «после поиска» (`app.api.after`) и
маршруты сомелье и ничего не пишет на диск, поэтому открыта всегда, без флага и ключа.

Полевой стенд (`app.api.field`) живёт на `/field` и только при `SVS_FIELD=1`. Раньше он стоял
на `/`, и в ходу остались ссылки вида `/?k=КЛЮЧ`: при включённом контуре такой адрес
перенаправляется на `/field?k=КЛЮЧ`, а `/` без ключа — страница продукта.

Под `/static` отдаются только файлы оформления (шрифт, картинки) и лист сомелье (`.js`, `.css`):
HTML оттуда не отдаётся, иначе страница полевого стенда открывалась бы по `/static/field.html` в
обход флага и ключа. Тип файла задан таблицей, а не `mimetypes`: у Python 3.12 своего `.woff2`
нет, он берёт тип из системы (`/etc/mime.types`, реестр Windows), и там, где системы не хватает,
шрифт уходил бы как `application/octet-stream` (на Windows-машине разработки 24.09 так и было).
На Windows тот же реестр отдаёт `.js` как `text/plain`, и браузер с `nosniff` скрипт не запустил
бы — поэтому у скрипта и стилей тип тоже свой, с кодировкой.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

STATIC_DIR = Path(__file__).with_name("static")
PAGE_PATH = STATIC_DIR / "index.html"
#: Что можно отдать из `/static` — шрифт, картинки оформления, скрипт и стили листа сомелье —
#: и с каким типом.
ASSET_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
}
ASSET_SUFFIXES = frozenset(ASSET_TYPES)
#: Параметр ключа полевого стенда в старых ссылках `/?k=…`.
FIELD_KEY = "k"


class AssetFiles(StaticFiles):
    """`StaticFiles` без HTML и прочего: файл с чужим расширением — 404, как несуществующий."""

    def lookup_path(self, path: str) -> tuple[str, os.stat_result | None]:
        if Path(path).suffix.lower() not in ASSET_SUFFIXES:
            return "", None
        return super().lookup_path(path)

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        media = ASSET_TYPES.get(Path(full_path).suffix.lower())
        # У ответа 304 типа нет вовсе: заголовок ставится только тому, у кого он есть.
        if media and "content-type" in response.headers:
            response.headers["content-type"] = media
        return response


def register_page_routes(app: FastAPI, *, field_redirect: bool = False) -> None:
    """Страница продукта на `/`, файлы оформления и лист сомелье на `/static`.

    `field_redirect` — полевой контур включён: `/?k=…` уводит на `/field?k=…` со всеми
    параметрами. Перенаправление временное (307): постоянное браузер запомнил бы, и после
    выключения контура старая ссылка всё равно вела бы на `/field`.
    """

    @app.get("/", include_in_schema=False)
    async def page(request: Request) -> Response:
        """Страница продукта; старая ссылка стенда `/?k=…` — на `/field`, если контур включён."""
        if field_redirect and FIELD_KEY in request.query_params:
            return RedirectResponse(f"/field?{request.url.query}", status_code=307)
        if not PAGE_PATH.exists():
            return HTMLResponse("<h1>Страница не собрана</h1>", status_code=500)
        # Без кэша: страницу правят до показа, а браузер молча отдал бы старую копию.
        return HTMLResponse(
            PAGE_PATH.read_text(encoding="utf-8"), headers={"Cache-Control": "no-cache"}
        )

    app.mount("/static", AssetFiles(directory=STATIC_DIR, check_dir=False), name="static")
