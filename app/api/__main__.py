"""Точка входа: `python -m app.api` — сборка, прогрев и uvicorn на SVS_HOST:SVS_PORT.

Коды выхода: 0 — сервис остановлен штатно; 2 — настройки не разбираются; 3 — сборка не та
(нет индекса, модели, словаря или модель CV не совпала с индексом).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace

from app.api.config import ServiceSettings, SettingsError
from app.api.main import create_app
from app.api.service import ScannerService, StartupError

logger = logging.getLogger("app.api")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.api",
        description="Сервис сканера вин: POST /v1/eval/predict, /v1/scan; GET /v1/health.",
    )
    parser.add_argument("--host", help="по умолчанию SVS_HOST (0.0.0.0)")
    parser.add_argument("--port", type=int, help="по умолчанию SVS_PORT (8080)")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = ServiceSettings.from_env()
        overrides = {k: v for k, v in (("host", args.host), ("port", args.port)) if v is not None}
        if overrides:
            settings = replace(settings, **overrides)
    except SettingsError as exc:
        logger.error("Настройки не разбираются: %s", exc)
        return 2
    for warning in settings.warnings():
        logger.warning("%s", warning)
    try:
        service = ScannerService.load(settings)
    except StartupError as exc:
        logger.error("Сервис не стартует: %s", exc)
        return 3
    import uvicorn

    uvicorn.run(
        create_app(settings, service),
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=args.log_level,
        # Кадры идут по одному; держать соединение скрипту не нужно, а зависшему клиенту — тем более.
        timeout_keep_alive=5,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
