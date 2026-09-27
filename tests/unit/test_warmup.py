import numpy as np
import pytest

from app.reading.contracts import Reading, TextLine, image_sha1
from app.reading.readers.cache import CachedReader, CacheStats
from app.reading.warmup import DEFAULT_WARMUP_BUDGET_MS, synthetic_label, warm_readers


class FakeClock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class FakeReader:
    version = "1"
    params_hash = "p"

    def __init__(self, id, *, status="ok", seconds=0.0, clock=None, error=None):
        self.id = id
        self.status = status
        self.seconds = seconds
        self.clock = clock
        self.error = error
        self.calls = []

    def available(self):
        return True

    def read(self, image, *, crop, budget_ms):
        self.calls.append({"shape": image.shape, "crop": crop, "budget_ms": budget_ms})
        if self.clock is not None:
            self.clock.now += self.seconds
        if self.error is not None:
            raise self.error
        return Reading(
            reader=self.id,
            version=self.version,
            params_hash=self.params_hash,
            image_sha1=image_sha1(image),
            crop=crop,
            crop_px=max(image.shape[:2]),
            lines=[TextLine(id=0, text="WARMUP")] if self.status == "ok" else [],
            status=self.status,
            elapsed_ms=round(self.seconds * 1000),
        )


def test_statuses_and_time_per_reader_without_raising():
    clock = FakeClock()
    readers = [
        FakeReader("vlm", seconds=3.7, clock=clock),
        FakeReader("easyocr", status="timeout", seconds=2.0, clock=clock),
        FakeReader("rapidocr", seconds=0.5, clock=clock, error=RuntimeError("CUDA out of memory")),
    ]
    report = warm_readers(readers, clock=clock)
    assert report == {
        "vlm": {"status": "ok", "elapsed_ms": 3700},
        "easyocr": {"status": "timeout", "elapsed_ms": 2000},
        "rapidocr": {
            "status": "error",
            "elapsed_ms": 500,
            "error": "RuntimeError: CUDA out of memory",
        },
    }
    for reader in readers:
        (call,) = reader.calls
        assert call == {
            "shape": (1024, 1024, 3),
            "crop": "full",
            "budget_ms": DEFAULT_WARMUP_BUDGET_MS,
        }


def test_given_frame_and_budget_are_passed():
    reader = FakeReader("ocr")
    frame = np.zeros((20, 10, 3), np.uint8)
    assert warm_readers([reader], image=frame, budget_ms=5000)["ocr"]["status"] == "ok"
    assert reader.calls == [{"shape": (20, 10, 3), "crop": "full", "budget_ms": 5000}]


def test_default_frame_is_white_with_dark_text_lines():
    frame = synthetic_label()
    assert frame.shape == (1024, 1024, 3) and frame.dtype == np.uint8
    assert (frame == 255).all(axis=2).mean() > 0.9
    assert frame.min() < 100
    dark_rows = np.flatnonzero((frame < 100).any(axis=(1, 2)))
    assert np.count_nonzero(np.diff(dark_rows) > 1) >= 3  # несколько отдельных строк
    assert synthetic_label(512).shape == (512, 512, 3)
    with pytest.raises(ValueError):
        synthetic_label(0)


def test_warmup_bypasses_reading_cache(tmp_path):
    inner = FakeReader("vlm")
    cached = CachedReader(inner, tmp_path)
    frame = np.full((64, 32, 3), 200, np.uint8)
    assert warm_readers([cached], image=frame)["vlm"]["status"] == "ok"
    assert len(inner.calls) == 1
    assert cached.stats.as_dict() == CacheStats().as_dict()
    assert list(tmp_path.rglob("*.json")) == []
    cached.read(frame, crop="full", budget_ms=1000)  # тот же кадр в прогоне — промах кэша
    assert (cached.stats.misses, cached.stats.hits, len(inner.calls)) == (1, 0, 2)


def test_reader_warm_method_is_preferred():
    class Warmable(FakeReader):
        def warm(self, image, *, budget_ms):
            self.warmed = (image.shape, budget_ms)
            return super().read(image, crop="full", budget_ms=budget_ms)

    reader = Warmable("vlm")
    report = warm_readers([reader], image=np.zeros((8, 8, 3), np.uint8), budget_ms=30_000)
    assert report["vlm"]["status"] == "ok" and reader.warmed == ((8, 8, 3), 30_000)


def test_duplicate_reader_ids_are_numbered():
    report = warm_readers([FakeReader("ocr"), FakeReader("ocr", status="empty")])
    assert [(key, row["status"]) for key, row in report.items()] == [
        ("ocr", "ok"),
        ("ocr#2", "empty"),
    ]
