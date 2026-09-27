import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.reading.contracts import Box, TextLine
from app.reading.readers.base import (
    box_from_quad,
    build_reader,
    encode_jpeg_b64,
    ensure_rgb,
    is_cacheable,
    resize_long_side,
    sort_reading_order,
)
from app.reading.readers.easyocr_reader import EasyOcrReader
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.reading.readers.rapidocr_reader import RapidOcrReader


def test_resize_long_side_shrinks_keeping_aspect():
    image = np.zeros((2000, 1000, 3), dtype=np.uint8)
    out = resize_long_side(image, 1024)
    assert out.shape == (1024, 512, 3)


def test_resize_long_side_never_enlarges():
    image = np.zeros((300, 200, 3), dtype=np.uint8)
    assert resize_long_side(image, 1024) is image


def test_encode_jpeg_b64_round_trip():
    image = np.full((40, 60, 3), 90, dtype=np.uint8)
    decoded = Image.open(BytesIO(base64.b64decode(encode_jpeg_b64(image, 92))))
    assert decoded.format == "JPEG" and decoded.size == (60, 40)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_ensure_rgb_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ensure_rgb(bad)


def test_box_from_quad_normalizes_and_clamps():
    box = box_from_quad([[-5, 10], [120, 10], [120, 30], [-5, 30]], width=100, height=200)
    assert box == Box(x0=0.0, y0=0.05, x1=1.0, y1=0.15)
    assert box_from_quad([[5, 5], [5, 5], [5, 5], [5, 5]], 100, 100) is None


def test_sort_reading_order_rows_then_columns():
    def line(text, x0, y0, x1, y1):
        return TextLine(id=9, text=text, box=Box(x0=x0, y0=y0, x1=x1, y1=y1))

    lines = [
        line("низ", 0.1, 0.80, 0.5, 0.85),
        line("право", 0.6, 0.11, 0.9, 0.16),
        line("лево", 0.1, 0.10, 0.5, 0.15),
        TextLine(id=9, text="без рамки"),
    ]
    ordered = sort_reading_order(lines)
    assert [item.text for item in ordered] == ["лево", "право", "низ", "без рамки"]
    assert [item.id for item in ordered] == [0, 1, 2, 3]


def test_cacheable_statuses():
    assert all(is_cacheable(s) for s in ("ok", "empty", "loop", "garbage"))
    assert not any(is_cacheable(s) for s in ("timeout", "unavailable", "error"))


def test_build_reader_registry():
    settings = Settings(device="cpu", vlm_model="qwen3-vl:4b-instruct", ollama_url="http://h:1")
    vlm = build_reader("vlm:qwen2.5vl:3b", settings)
    assert isinstance(vlm, OllamaVlmReader) and vlm.model == "qwen2.5vl:3b"
    assert vlm.url == "http://h:1"
    assert build_reader("vlm", settings).model == "qwen3-vl:4b-instruct"
    easy = build_reader("easyocr", settings)
    assert isinstance(easy, EasyOcrReader) and easy.gpu is False
    assert build_reader("rapidocr", settings).rec_model_type == "mobile"
    server = build_reader("rapidocr:server", settings)
    assert isinstance(server, RapidOcrReader) and server.rec_model_type == "server"
    with pytest.raises(ValueError):
        build_reader("tesseract", settings)
