"""Настройки из переменных окружения с префиксом `SVS_`."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    # Распакованный датасет организатора: strapi_output0709.csv, eval/, uploads Strapi.
    dataset_dir: Path = field(
        default_factory=lambda: _path("SVS_DATASET_DIR", REPO_ROOT / "data" / "raw" / "dataset")
    )
    # Производные данные: словарь, эталонные токены, манифесты синтетики.
    data_dir: Path = field(default_factory=lambda: _path("SVS_DATA_DIR", REPO_ROOT / "data"))
    cache_dir: Path = field(
        default_factory=lambda: _path("SVS_CACHE_DIR", REPO_ROOT / "data" / "cache")
    )
    ollama_url: str = field(
        default_factory=lambda: os.environ.get("SVS_OLLAMA_URL", "http://127.0.0.1:11434")
    )
    vlm_model: str = field(
        default_factory=lambda: os.environ.get("SVS_VLM_MODEL", "qwen3-vl:4b-instruct")
    )
    device: str = field(default_factory=lambda: os.environ.get("SVS_DEVICE", "cuda"))


def get_settings() -> Settings:
    return Settings()
