import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.reading.readers.easyocr_reader import EasyOcrReader
from app.reading.readers.rapidocr_reader import RapidOcrConfigError, RapidOcrReader

GPU_TESTS = os.environ.get("SVS_RUN_GPU_TESTS") == "1"


def frame(h: int = 200, w: int = 100) -> np.ndarray:
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = 220  # красный канал: серый зависит от порядка каналов
    image[..., 2] = 20
    return image


def quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --- EasyOCR через поддельный движок ---


class FakeEasyEngine:
    def __init__(self, results):
        self.results = results
        self.detect_args = None
        self.recognize_args = None

    def detect(self, img, **kw):
        self.detect_args = (img, kw)
        return [[[0, 10, 0, 10]]], [[]]

    def recognize(self, grey, horizontal, free, **kw):
        self.recognize_args = (grey, horizontal, free, kw)
        return self.results


def easy_with(results) -> tuple[EasyOcrReader, FakeEasyEngine]:
    reader = EasyOcrReader(gpu=False)
    engine = FakeEasyEngine(results)
    reader._engine = engine
    return reader, engine


def test_easyocr_keeps_boxes_conf_and_reading_order():
    results = [
        (quad(50, 150, 90, 170), "2021", 0.61),  # ниже
        (quad(55, 20, 95, 40), "ДОЛИНА", 0.88),  # правее в верхнем ряду
        (quad(5, 22, 45, 42), "СОЛНЕЧНАЯ", 0.93),
        (quad(0, 0, 10, 10), "   ", 0.2),
    ]
    reader, _ = easy_with(results)
    reading = reader.read(frame(), crop="label", budget_ms=1000)
    assert reading.status == "ok"
    assert [line.text for line in reading.lines] == ["СОЛНЕЧНАЯ", "ДОЛИНА", "2021"]
    assert [line.id for line in reading.lines] == [0, 1, 2]
    first = reading.lines[0]
    assert first.conf == pytest.approx(0.93)
    assert first.box.x0 == pytest.approx(0.05) and first.box.y1 == pytest.approx(0.21)
    assert first.angle == pytest.approx(0.0)
    assert reading.crop_px == 200


def test_easyocr_grey_is_rgb_luminance_and_detector_gets_rgb():
    reader, engine = easy_with([])
    image = frame()
    reading = reader.read(image, crop="full", budget_ms=1000)
    assert reading.status == "empty"
    assert engine.detect_args[0] is image
    assert np.array_equal(engine.recognize_args[0], cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    assert engine.recognize_args[3]["detail"] == 1
    assert engine.detect_args[1]["canvas_size"] == 2560


def test_easyocr_unavailable_when_engine_cannot_be_built(monkeypatch):
    reader = EasyOcrReader(gpu=False)

    def boom():
        raise ImportError("No module named 'easyocr'")

    monkeypatch.setattr(reader, "_ensure_engine", boom)
    reading = reader.read(frame(), crop="full", budget_ms=1000)
    assert reading.status == "unavailable" and "easyocr" in reading.raw


def test_easyocr_engine_error_status():
    class Broken(FakeEasyEngine):
        def detect(self, img, **kw):
            raise RuntimeError("CUDA out of memory")

    reader = EasyOcrReader(gpu=False)
    reader._engine = Broken([])
    assert reader.read(frame(), crop="full", budget_ms=1000).status == "error"


def test_easyocr_params_hash_changes_with_thresholds():
    assert EasyOcrReader(low_text=0.3).params_hash != EasyOcrReader().params_hash
    assert EasyOcrReader(gpu=False).params_hash == EasyOcrReader(gpu=True).params_hash


def test_easyocr_available_matches_import():
    pytest.importorskip("easyocr")
    assert EasyOcrReader().available()


@pytest.mark.skipif(not GPU_TESTS, reason="нужны веса EasyOCR и видеокарта: SVS_RUN_GPU_TESTS=1")
def test_easyocr_real_engine_smoke():
    pytest.importorskip("easyocr")
    image = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(image, "BRUT 2021", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    reading = EasyOcrReader(gpu=True).read(image, crop="full", budget_ms=10_000)
    assert reading.status in ("ok", "empty")


# --- RapidOCR через поддельный движок ---


class FakeRapidEngine:
    def __init__(self, output):
        self.output = output
        self.seen = None

    def __call__(self, img):
        self.seen = img
        return self.output


def test_rapidocr_keeps_boxes_conf_and_passes_bgr():
    output = SimpleNamespace(
        boxes=np.array([quad(10, 120, 90, 140), quad(10, 20, 90, 40)], dtype=np.float32),
        txts=("сухое", "САПЕРАВИ"),
        scores=(0.77, 0.95),
    )
    reader = RapidOcrReader()
    engine = FakeRapidEngine(output)
    reader._engine = engine
    image = frame()
    reading = reader.read(image, crop="label", budget_ms=1000)
    assert [line.text for line in reading.lines] == ["САПЕРАВИ", "сухое"]
    assert reading.lines[0].conf == pytest.approx(0.95)
    assert reading.lines[0].box.x1 == pytest.approx(0.9)
    assert np.array_equal(engine.seen, image[:, :, ::-1])


def test_rapidocr_empty_output():
    reader = RapidOcrReader()
    reader._engine = FakeRapidEngine(SimpleNamespace(boxes=None, txts=None, scores=None))
    assert reader.read(frame(), crop="full", budget_ms=1000).status == "empty"


def test_rapidocr_pins_ppocrv5_explicitly():
    params = RapidOcrReader().engine_params()
    assert params["Det.ocr_version"] == "PP-OCRv5" and params["Rec.ocr_version"] == "PP-OCRv5"
    assert params["Det.model_type"] == "mobile" and params["Rec.model_type"] == "mobile"
    assert params["Rec.lang_type"] == "eslav"
    assert params["EngineConfig.onnxruntime.use_cuda"] is False
    assert RapidOcrReader(rec_model_type="server").engine_params()["Rec.model_type"] == "server"
    assert RapidOcrReader(rec_model_type="server").params_hash != RapidOcrReader().params_hash


def test_rapidocr_check_config_rejects_silent_v6():
    params = RapidOcrReader().engine_params()
    ok = {
        "Det": {
            "engine_type": "onnxruntime",
            "lang_type": "ch",
            "model_type": "mobile",
            "ocr_version": "PP-OCRv5",
        },
        "Rec": {
            "engine_type": "onnxruntime",
            "lang_type": "eslav",
            "model_type": "mobile",
            "ocr_version": "PP-OCRv5",
        },
    }
    RapidOcrReader.check_config(ok, params)
    bad = {"Det": dict(ok["Det"]), "Rec": dict(ok["Rec"], ocr_version="PP-OCRv6")}
    with pytest.raises(RapidOcrConfigError, match="Rec.ocr_version"):
        RapidOcrReader.check_config(bad, params)
    as_attrs = SimpleNamespace(
        Det=SimpleNamespace(**ok["Det"]),
        Rec=SimpleNamespace(**dict(ok["Rec"], model_type=SimpleNamespace(value="small"))),
    )
    with pytest.raises(RapidOcrConfigError, match="Rec.model_type"):
        RapidOcrReader.check_config(as_attrs, params)


def test_rapidocr_config_error_is_raised_not_swallowed(monkeypatch):
    reader = RapidOcrReader()

    def wrong():
        raise RapidOcrConfigError("Rec.ocr_version: запрошено 'PP-OCRv5', в движке 'PP-OCRv6'")

    monkeypatch.setattr(reader, "_ensure_engine", wrong)
    with pytest.raises(RapidOcrConfigError):
        reader.read(frame(), crop="full", budget_ms=1000)


def test_rapidocr_params_become_enums_with_real_typings():
    typings = pytest.importorskip("rapidocr.utils.typings")
    params = RapidOcrReader.to_enums(RapidOcrReader().engine_params(), typings)
    # update_batch в rapidocr 3.9.2 требует Enum: строка даёт TypeError.
    assert params["Det.ocr_version"] is typings.OCRVersion.PPOCRV5
    assert params["Rec.ocr_version"] is typings.OCRVersion.PPOCRV5
    assert params["Det.model_type"] is typings.ModelType.MOBILE
    assert params["Rec.model_type"] is typings.ModelType.MOBILE
    assert params["Rec.lang_type"] is typings.LangRec.ESLAV
    assert params["Det.lang_type"] is typings.LangDet.CH
    assert params["Det.engine_type"] is typings.EngineType.ONNXRUNTIME
    assert params["Global.use_cls"] is False


FAKE_TYPINGS = SimpleNamespace(
    EngineType=str, ModelType=str, OCRVersion=str, LangDet=str, LangRec=str
)


def v5_cfg(**rec):
    det = {
        "engine_type": "onnxruntime",
        "lang_type": "ch",
        "model_type": "mobile",
        "ocr_version": "PP-OCRv5",
    }
    rec_cfg = {
        "engine_type": "onnxruntime",
        "lang_type": "eslav",
        "model_type": "mobile",
        "ocr_version": "PP-OCRv5",
    } | rec
    return {"Det": det, "Rec": rec_cfg}


def with_engine_class(monkeypatch, reader, engine_cls):
    monkeypatch.setattr(reader, "_load_rapidocr", lambda: (engine_cls, FAKE_TYPINGS))


def test_rapidocr_unsupported_combo_is_config_error_not_unavailable(monkeypatch):
    class Refuses:
        def __init__(self, params):
            raise ValueError("Unsupported Rec.lang_type='eslav' for PP-OCRv5 server model.")

    reader = RapidOcrReader(rec_model_type="server")
    with_engine_class(monkeypatch, reader, Refuses)
    with pytest.raises(RapidOcrConfigError, match="lang_type"):
        reader.read(frame(), crop="full", budget_ms=1000)


def test_rapidocr_engine_silently_on_v6_is_rejected(monkeypatch):
    class SilentV6:
        def __init__(self, params):
            self.params = params
            self.cfg = v5_cfg(ocr_version="PP-OCRv6", model_type="small")

    reader = RapidOcrReader()
    with_engine_class(monkeypatch, reader, SilentV6)
    with pytest.raises(RapidOcrConfigError, match="Rec"):
        reader.read(frame(), crop="full", budget_ms=1000)
    assert reader._engine is None


def test_rapidocr_engine_built_with_v5_is_accepted(monkeypatch):
    output = SimpleNamespace(
        boxes=np.array([quad(10, 20, 90, 40)], dtype=np.float32), txts=("БРЮТ",), scores=(0.9,)
    )

    class GoodV5:
        def __init__(self, params):
            self.params = params
            self.cfg = v5_cfg()

        def __call__(self, img):
            return output

    reader = RapidOcrReader()
    with_engine_class(monkeypatch, reader, GoodV5)
    reading = reader.read(frame(), crop="full", budget_ms=1000)
    assert reading.status == "ok" and reading.text == "БРЮТ"
    assert reader._engine.params["Rec.ocr_version"] == "PP-OCRv5"


def test_rapidocr_missing_package_is_unavailable(monkeypatch):
    reader = RapidOcrReader()

    def missing():
        raise ImportError("No module named 'rapidocr'")

    monkeypatch.setattr(reader, "_load_rapidocr", missing)
    reading = reader.read(frame(), crop="full", budget_ms=1000)
    assert reading.status == "unavailable" and "rapidocr" in reading.raw


@pytest.mark.skipif(not GPU_TESTS, reason="создание движка грузит модели: SVS_RUN_GPU_TESTS=1")
def test_rapidocr_real_engine_is_v5():
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    reader = RapidOcrReader()
    engine = reader._ensure_engine()
    assert str(getattr(engine.cfg.Rec.ocr_version, "value", engine.cfg.Rec.ocr_version)) == (
        "PP-OCRv5"
    )
