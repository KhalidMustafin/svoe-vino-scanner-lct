"""Кэш чтений на диске: один JSON на ключ `Reading.key`, атомарная запись."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from app.reading.contracts import CropName, Reader, Reading, image_sha1
from app.reading.readers.base import crop_px_for, is_cacheable

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    skipped: int = 0  # статус не кэшируется (timeout/unavailable/error)
    corrupt: int = 0  # битый файл — считается промахом

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def reading_key(
    reader_id: str,
    version: str,
    params: str,
    image_hash: str,
    crop: CropName,
    crop_px: int,
) -> str:
    """Тот же формат, что `Reading.key`, но до чтения."""
    return f"{reader_id}@{version}|{params}|{image_hash}|{crop}|{crop_px}"


class CachedReader:
    """Обёртка над читателем: попадание отдаёт сохранённое чтение, сбои среды не кэшируются."""

    def __init__(self, reader: Reader, cache_dir: str | os.PathLike[str]) -> None:
        self.reader = reader
        self.cache_dir = Path(cache_dir)
        self.stats = CacheStats()

    @property
    def id(self) -> str:
        return self.reader.id

    @property
    def version(self) -> str:
        return self.reader.version

    def available(self) -> bool:
        return self.reader.available()

    def key_for(self, image: np.ndarray, *, crop: CropName) -> str:
        return reading_key(
            self.reader.id,
            self.reader.version,
            self.reader.params_hash,  # type: ignore[attr-defined]
            image_sha1(image),
            crop,
            crop_px_for(self.reader, image),
        )

    def _path(self, key: str) -> Path:
        name = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.cache_dir / name[:2] / f"{name}.json"

    def get(self, key: str) -> Reading | None:
        path = self._path(key)
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            self.stats.corrupt += 1
            logger.warning("Битый файл кэша %s: %s", path, exc)
            return None
        try:
            if blob.get("key") != key:
                raise ValueError("ключ в файле не совпадает")
            return Reading.model_validate(blob["reading"])
        except (KeyError, TypeError, ValueError) as exc:
            self.stats.corrupt += 1
            logger.warning("Битый файл кэша %s: %s", path, exc)
            return None

    def put(self, reading: Reading, key: str | None = None) -> bool:
        """Сохранить чтение. False — статус не кэшируется."""
        if not is_cacheable(reading.status):
            self.stats.skipped += 1
            return False
        key = key or reading.key
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"key": key, "reading": reading.model_dump(mode="json")}, ensure_ascii=False
        )
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.stats.writes += 1
        return True

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        key = self.key_for(image, crop=crop)
        cached = self.get(key)
        if cached is not None:
            self.stats.hits += 1
            return cached
        self.stats.misses += 1
        reading = self.reader.read(image, crop=crop, budget_ms=budget_ms)
        if reading.key != key:
            logger.warning("Ключ чтения %s не совпал с ожидаемым %s", reading.key, key)
        self.put(reading, key=key)
        return reading
