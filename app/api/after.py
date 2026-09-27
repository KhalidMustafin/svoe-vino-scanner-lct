"""Маршруты слоя «после поиска». Логика — в `app.api.after_layer`, договор — `docs/api-after-search.md`.

    GET  /v1/wines/{slug}           карточка: только выгрузка организатора (сахар и крепость по
                                    правилам договора, описание как есть), `photo_url` вместо
                                    пути к файлу
    GET  /v1/wines/{slug}/photo     фото выгрузки организатора, открыто, с кэшем на сутки
    GET  /v1/wines/{slug}/similar   похожие из других виноделен, `limit` 1–12, `order=reco|plain`
    POST /v1/similar/by-label       «Не тупик»: вина прочитанной винодельни и близкие по стилю,
                                    без отвергнутых вин и их групп (`exclude`, до 20 slug)
    GET  /v1/wines/{slug}/shelf     «Сомелье у полки»: чипы `food` и `want` → три вина или
                                    честная фраза, `order=reco|plain`

Маршрута иконок блюд портала (`/v1/icons/dishes/…`) больше нет (решение 24.09): блюда правил
сочетаний страница рисует своими глифами по категории (договор сомелье, §7.2).

Маршруты отдельно от логики, потому что `ScannerService` собирает слой и без FastAPI (стенды и
замеры импортируют сервис, а FastAPI — необязательная зависимость `api`).
"""

from __future__ import annotations

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from app.api.after_layer import CACHE_CONTROL, ByLabelBody, Order, image_type
from app.recommend.facts import DEFAULT_LIMIT, MAX_LIMIT
from app.recommend.shelf import Food, Want


def register_after_routes(app: FastAPI) -> None:
    """Маршруты слоя поверх собранного приложения; сервис — `app.state.service`.

    Регистрируются до полевого контура (`app.api.main`): продуктовой странице они нужны
    всегда, а полевой контур по умолчанию выключен (`SVS_FIELD`).
    """

    def service(request: Request):
        return getattr(request.app.state, "service", None)

    def missing(slug: str) -> JSONResponse:
        return JSONResponse({"detail": f"нет карточки {slug!r}"}, status_code=404)

    @app.get("/v1/wines/{slug}")
    async def wine(slug: str, request: Request) -> JSONResponse:
        """Карточка позиции каталога (договор, §3): 404 — нет такой."""
        svc = service(request)
        card = svc.card(slug) if svc is not None else None
        if card is None:
            return missing(slug)
        return JSONResponse(card)

    @app.get("/v1/wines/{slug}/photo", include_in_schema=False)
    async def wine_photo(slug: str, request: Request) -> Response:
        """Фото выгрузки организатора: открыто без ключа, с кэшем на сутки."""
        svc = service(request)
        path = svc.photo_file(slug) if svc is not None else None
        if path is None:
            return JSONResponse({"detail": "фото каталога нет на этой машине"}, status_code=404)
        return FileResponse(
            path, media_type=image_type(path), headers={"Cache-Control": CACHE_CONTROL}
        )

    @app.get("/v1/wines/{slug}/similar")
    async def wine_similar(
        slug: str,
        request: Request,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        order: Order = "reco",
    ) -> JSONResponse:
        """Похожие из других виноделен, объяснённые фактами каталога (договор, §5)."""
        svc = service(request)
        body = svc.after.similar(slug, limit=limit, order=order) if svc is not None else None
        if body is None:
            return missing(slug)
        return JSONResponse(body)

    @app.post("/v1/similar/by-label")
    async def similar_by_label(
        payload: ByLabelBody,
        request: Request,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        order: Order = "reco",
    ) -> JSONResponse:
        """«Не тупик»: вина винодельни и близкие по стилю, без `exclude` (договор, §6)."""
        svc = service(request)
        if svc is None:
            return JSONResponse({"detail": "сервис ещё не собран"}, status_code=503)
        return JSONResponse(svc.after.by_label(payload, limit=limit, order=order))

    @app.get("/v1/wines/{slug}/shelf")
    async def wine_shelf(
        slug: str,
        request: Request,
        food: Food = "none",
        want: Want = "none",
        order: Order = "reco",
    ) -> JSONResponse:
        """«Сомелье у полки» (договор, §7): на входе только значения чипов, иначе — 422."""
        svc = service(request)
        body = svc.after.shelf(slug, food=food, want=want, order=order) if svc is not None else None
        if body is None:
            return missing(slug)
        return JSONResponse(body)
