"""Читатели этикетки: модель зрения через Ollama, EasyOCR, RapidOCR и кэш чтений."""

from app.reading.readers.base import build_reader
from app.reading.readers.cache import CachedReader, CacheStats

__all__ = ["CacheStats", "CachedReader", "build_reader"]
