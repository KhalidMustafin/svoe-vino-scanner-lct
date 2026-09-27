"""Проверки без весов и без сети: letterbox, нормировка и отказ вместо скачивания."""

import numpy as np
import pytest

from app.features.embedder import (
    LETTERBOX_SIDE,
    ModelNotAvailable,
    SiglipEmbedder,
    letterbox,
    unit_rows,
)

# Модели с таким именем нет ни в кэше, ни на диске: подъём обязан кончиться отказом,
# а не походом в сеть — `local_files_only=True`.
ABSENT_MODEL = "svoe-vino/no-such-model-in-cache"


def _image(height: int, width: int, value: int = 64) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_letterbox_makes_a_square_with_white_margins():
    out = letterbox(_image(400, 200), side=100)
    assert out.shape == (100, 100, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (255, 255, 255)  # поля белые, как фон packshot
    assert tuple(out[50, 50]) == (64, 64, 64)


def test_letterbox_keeps_proportions():
    out = letterbox(_image(400, 200), side=100)
    filled = np.flatnonzero((out != 255).any(axis=(0, 2)))
    assert filled.size == 50  # 200/400 стороны — половина квадрата
    assert filled[0] == 25 and filled[-1] == 74  # и она по центру


def test_letterbox_upscales_small_frame():
    assert letterbox(_image(10, 30), side=64).shape == (64, 64, 3)


def test_letterbox_fill_is_a_parameter():
    assert tuple(letterbox(_image(40, 20), side=32, fill=128)[0, 0]) == (128, 128, 128)


def test_letterbox_rejects_rgba_and_zero_side():
    with pytest.raises(ValueError, match="RGB uint8"):
        letterbox(np.zeros((10, 10, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="side"):
        letterbox(_image(10, 10), side=0)


def test_default_letterbox_side_is_bigger_than_model_input():
    assert LETTERBOX_SIDE == 448  # процессор уменьшит сам, но верх бутылки не срежет


def test_unit_rows_normalizes_and_keeps_zeros():
    out = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(out[0], [0.6, 0.8]) and np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_unit_rows_rejects_a_single_vector():
    with pytest.raises(ValueError, match="матрица"):
        unit_rows(np.array([1.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_unit_rows_refuses_nan_and_inf(bad: float):
    """Нефинитный вектор давал косинус NaN, а NaN в ранжировании не проигрывал никому.

    Кадр получал «ответ» — первый slug индекса со счётом 0,0 и отрывом 0,0, со статусом
    «ok». Теперь это ошибка, и такой кадр уходит в сбойные.
    """
    with pytest.raises(ValueError, match="NaN или Inf"):
        unit_rows(np.array([[1.0, 0.0], [bad, 0.0]], dtype=np.float32))
    assert np.allclose(unit_rows(np.array([[3.0, 4.0]], dtype=np.float32)), [[0.6, 0.8]])


def test_embedder_checks_its_arguments():
    with pytest.raises(ValueError, match="batch_size"):
        SiglipEmbedder(device="cpu", batch_size=0)
    with pytest.raises(ValueError, match="dtype"):
        SiglipEmbedder(device="cpu", dtype="bfloat16")


def test_missing_weights_do_not_start_a_download():
    """Весов нет — понятная ошибка и «канал недоступен», а не скачивание 400 МБ."""
    embedder = SiglipEmbedder(ABSENT_MODEL, device="cpu")
    assert embedder.available() is False
    with pytest.raises(ModelNotAvailable, match=ABSENT_MODEL):
        _ = embedder.dim
    with pytest.raises(ModelNotAvailable):
        embedder.embed([_image(32, 32)])


def test_embed_rejects_empty_list():
    with pytest.raises(ValueError, match="пустой список"):
        SiglipEmbedder(ABSENT_MODEL, device="cpu").embed([])


def test_model_name_is_the_key_of_the_index():
    assert SiglipEmbedder(ABSENT_MODEL, device="cpu").model_name == ABSENT_MODEL
