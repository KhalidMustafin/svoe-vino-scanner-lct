import numpy as np
import pytest
from pydantic import ValidationError

from app.reading.contracts import (
    Box,
    Evidence,
    LabelFields,
    Reader,
    Reading,
    ReadResult,
    SugarClass,
    TextLine,
    image_sha1,
    params_hash,
)


def test_box_rejects_inverted_coordinates():
    with pytest.raises(ValidationError):
        Box(x0=0.5, y0=0.1, x1=0.4, y1=0.9)


def test_box_center_and_area():
    box = Box(x0=0.2, y0=0.0, x1=0.6, y1=0.5)
    assert box.center == pytest.approx((0.4, 0.25))
    assert box.area == pytest.approx(0.2)


def test_reading_key_changes_with_crop_size():
    base = {
        "reader": "vlm",
        "version": "qwen3-vl:4b-instruct",
        "params_hash": "abc",
        "image_sha1": "f00",
        "crop": "label",
        "elapsed_ms": 10,
    }
    assert Reading(crop_px=768, **base).key != Reading(crop_px=1024, **base).key


def test_reading_text_joins_lines():
    reading = Reading(
        reader="easyocr",
        version="1.7.2",
        params_hash="p",
        image_sha1="s",
        crop="full",
        crop_px=1024,
        elapsed_ms=5,
        lines=[TextLine(id=0, text="МАССАНДРА"), TextLine(id=1, text="2023")],
    )
    assert reading.text == "МАССАНДРА\n2023"


def test_label_fields_generic_evidence_validates_enum():
    fields = LabelFields(sugar=[Evidence[SugarClass](value="brut", sources=["k"])])
    assert fields.sugar[0].value is SugarClass.BRUT
    with pytest.raises(ValidationError):
        LabelFields(sugar=[Evidence[SugarClass](value="сухое")])


def test_read_result_round_trips_json():
    result = ReadResult(fields=LabelFields(vintage=Evidence[int](value=2023, support=2)))
    restored = ReadResult.model_validate_json(result.model_dump_json())
    assert restored.fields.vintage is not None and restored.fields.vintage.value == 2023


def test_image_sha1_depends_on_pixels_and_shape():
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 1
    assert image_sha1(a) != image_sha1(b)
    assert image_sha1(a) != image_sha1(a.reshape(6, 4, 3))


def test_params_hash_ignores_key_order():
    assert params_hash({"a": 1, "b": "x"}) == params_hash({"b": "x", "a": 1})


def test_reader_protocol_is_structural():
    class Dummy:
        id = "dummy"
        version = "0"

        def available(self) -> bool:
            return True

        def read(self, image, *, crop, budget_ms):
            raise NotImplementedError

    assert isinstance(Dummy(), Reader)
