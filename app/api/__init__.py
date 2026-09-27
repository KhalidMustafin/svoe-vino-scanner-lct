"""HTTP-сервис сканера: настройки, сборка компонентов, скан и FastAPI-приложение.

Модуль `main` (FastAPI) импортируется отдельно: для него нужен extra `api`.
"""

from app.api.config import ServiceSettings, SettingsError
from app.api.service import ScannerService, ScanResult, StartupError

__all__ = ["ScanResult", "ScannerService", "ServiceSettings", "SettingsError", "StartupError"]
