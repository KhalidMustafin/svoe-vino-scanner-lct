import json
from pathlib import Path

import numpy as np
import pytest

from app.reading.contracts import Reader, TextLine
from app.reading.readers.base import make_reading
from app.reading.readers.cache import CachedReader
from app.reading.readers.ollama_vlm import OllamaVlmReader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ollama"


class FakeReader:
    id = "fake"
    version = "1"
    params_hash = "p0"

    def __init__(self, status="ok", crop_px=None):
        self.status = status
        self.crop_px = crop_px
        self.calls = 0

    def available(self):
        return True

    def crop_px_for(self, image):
        return self.crop_px or max(image.shape[:2])

    def read(self, image, *, crop, budget_ms):
        self.calls += 1
        lines = [TextLine(id=0, text="АБРАУ")] if self.status == "ok" else []
        return make_reading(
            self,
            image,
            params=self.params_hash,
            crop=crop,
            status=self.status,
            elapsed_ms=3,
            lines=lines,
        )


@pytest.fixture
def image():
    return np.full((64, 32, 3), 200, dtype=np.uint8)


def test_cached_reader_is_a_reader(tmp_path):
    assert isinstance(CachedReader(FakeReader(), tmp_path), Reader)


def test_hit_returns_saved_reading_without_calling(tmp_path, image):
    inner = FakeReader()
    cached = CachedReader(inner, tmp_path)
    first = cached.read(image, crop="label", budget_ms=100)
    second = cached.read(image, crop="label", budget_ms=100)
    assert inner.calls == 1
    assert second == first
    assert cached.stats.as_dict() == {
        "hits": 1,
        "misses": 1,
        "writes": 1,
        "skipped": 0,
        "corrupt": 0,
    }


def test_get_by_key(tmp_path, image):
    cached = CachedReader(FakeReader(), tmp_path)
    reading = cached.read(image, crop="full", budget_ms=100)
    assert cached.get(reading.key) == reading
    assert cached.get(reading.key + "x") is None


def test_file_named_by_sha1_and_no_tmp_left(tmp_path, image):
    cached = CachedReader(FakeReader(), tmp_path)
    reading = cached.read(image, crop="full", budget_ms=100)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1
    assert len(files[0].stem) == 40 and not files[0].name.startswith(".tmp")
    assert json.loads(files[0].read_text(encoding="utf-8"))["key"] == reading.key


@pytest.mark.parametrize("status", ["timeout", "unavailable", "error"])
def test_transient_statuses_not_cached(tmp_path, image, status):
    inner = FakeReader(status=status)
    cached = CachedReader(inner, tmp_path)
    cached.read(image, crop="full", budget_ms=100)
    cached.read(image, crop="full", budget_ms=100)
    assert inner.calls == 2
    assert cached.stats.skipped == 2 and cached.stats.writes == 0


@pytest.mark.parametrize("status", ["loop", "garbage", "empty"])
def test_model_outcomes_cached(tmp_path, image, status):
    inner = FakeReader(status=status)
    cached = CachedReader(inner, tmp_path)
    cached.read(image, crop="full", budget_ms=100)
    assert cached.read(image, crop="full", budget_ms=100).status == status
    assert inner.calls == 1


def test_crop_px_change_changes_key(tmp_path, image):
    inner = FakeReader(crop_px=768)
    cached = CachedReader(inner, tmp_path)
    key_768 = cached.key_for(image, crop="label")
    cached.read(image, crop="label", budget_ms=100)
    inner.crop_px = 1024
    assert cached.key_for(image, crop="label") != key_768
    cached.read(image, crop="label", budget_ms=100)
    assert inner.calls == 2


def test_crop_name_and_params_change_key(tmp_path, image):
    inner = FakeReader()
    cached = CachedReader(inner, tmp_path)
    key = cached.key_for(image, crop="label")
    assert cached.key_for(image, crop="band") != key
    inner.params_hash = "p1"
    assert cached.key_for(image, crop="label") != key


def test_corrupt_file_is_a_miss(tmp_path, image):
    inner = FakeReader()
    cached = CachedReader(inner, tmp_path)
    reading = cached.read(image, crop="full", budget_ms=100)
    path = next(p for p in tmp_path.rglob("*.json"))
    path.write_text("{не json", encoding="utf-8")
    assert cached.get(reading.key) is None
    assert cached.stats.corrupt == 1
    cached.read(image, crop="full", budget_ms=100)
    assert inner.calls == 2


def test_vlm_reader_through_cache(tmp_path):
    response = json.loads((FIXTURES / "ok.json").read_text(encoding="utf-8"))
    calls = []

    def transport(payload, *, timeout_s):
        calls.append(payload)
        return response

    vlm = OllamaVlmReader("qwen2.5vl:3b", transport=transport, long_side=512)
    cached = CachedReader(vlm, tmp_path)
    image = np.full((900, 400, 3), 128, dtype=np.uint8)
    first = cached.read(image, crop="bottle", budget_ms=5000)
    assert first.crop_px == 512
    assert cached.key_for(image, crop="bottle") == first.key
    assert cached.read(image, crop="bottle", budget_ms=5000) == first
    assert len(calls) == 1


def test_thinking_only_vlm_answer_is_not_cached(tmp_path):
    response = json.loads((FIXTURES / "thinking_only.json").read_text(encoding="utf-8"))
    calls = []

    def transport(payload, *, timeout_s):
        calls.append(payload)
        return response

    cached = CachedReader(OllamaVlmReader("qwen3-vl:8b", transport=transport), tmp_path)
    image = np.full((90, 40, 3), 128, dtype=np.uint8)
    for _ in range(2):
        assert cached.read(image, crop="full", budget_ms=5000).status == "error"
    assert len(calls) == 2
    assert cached.stats.writes == 0 and cached.stats.skipped == 2
