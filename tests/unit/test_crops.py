import numpy as np
import pytest

from app.reading.contracts import Box
from app.reading.crops import (
    box_pixels,
    center_band,
    crop_box,
    crop_full,
    label_band,
    label_box,
    pad_box,
)


def _grid(height: int = 100, width: int = 200) -> np.ndarray:
    """Кадр, в пикселях которого записаны их координаты: канал 0 — x, канал 1 — y."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    arr[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    arr[..., 2] = 7
    return arr


def test_box_pixels_maps_shares_without_float_drift():
    assert box_pixels(Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9), 200, 100) == (50, 10, 150, 90)
    assert box_pixels(Box(x0=0.2, y0=0.0, x1=0.8, y1=1.0), 200, 100) == (40, 0, 160, 100)


def test_box_pixels_never_empty():
    x0, y0, x1, y1 = box_pixels(Box(x0=0.999, y0=0.999, x1=1.0, y1=1.0), 10, 10)
    assert x1 - x0 >= 1 and y1 - y0 >= 1


def test_crop_box_without_padding_is_exact():
    out = crop_box(_grid(), Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9), pad_x=0.0, pad_y=0.0)
    assert out.shape == (80, 100, 3)
    assert tuple(out[0, 0, :2]) == (50, 10)
    assert tuple(out[-1, -1, :2]) == (149, 89)


def test_crop_box_padding_is_relative_to_box_size():
    out = crop_box(_grid(), Box(x0=0.25, y0=0.1, x1=0.75, y1=0.9))
    # ширина рамки 0.5 → отступ 0.04 (8 px); высота 0.8 → 0.032 (3.2 px)
    assert tuple(out[0, 0, :2]) == (42, 6)
    assert out.shape == (88, 116, 3)


def test_pad_box_clips_to_frame():
    padded = pad_box(Box(x0=0.0, y0=0.02, x1=0.1, y1=1.0), 0.5, 0.5)
    assert padded.x0 == 0.0 and padded.y0 == 0.0 and padded.y1 == 1.0
    assert padded.x1 == pytest.approx(0.15)


def test_crop_box_returns_independent_copy():
    image = _grid()
    out = crop_box(image, Box(x0=0.0, y0=0.0, x1=0.5, y1=0.5), pad_x=0.0, pad_y=0.0)
    out[:] = 0
    assert image[0, 0, 2] == 7


def test_crop_full_caps_long_side_and_does_not_upscale_by_default():
    assert crop_full(np.zeros((500, 1000, 3), np.uint8), 400).shape == (200, 400, 3)
    small = _grid(50, 80)
    out = crop_full(small, 1024)
    assert out.shape == small.shape and out is not small
    assert crop_full(small, 160, upscale=True).shape == (100, 160, 3)


def test_label_box_uses_bottle_height_shares():
    band = label_box(Box(x0=0.2, y0=0.1, x1=0.6, y1=0.9), top=0.3, bottom=1.0)
    assert (band.x0, band.x1) == (0.2, 0.6)
    assert band.y0 == pytest.approx(0.34) and band.y1 == pytest.approx(0.9)


def test_label_band_cuts_below_the_neck():
    out = label_band(_grid(), Box(x0=0.25, y0=0.0, x1=0.75, y1=1.0), top=0.3, pad_x=0.0)
    assert out.shape == (70, 100, 3)
    assert tuple(out[0, 0, :2]) == (50, 30)


def test_label_band_rejects_inverted_shares():
    with pytest.raises(ValueError):
        label_band(_grid(), Box(x0=0.2, y0=0.0, x1=0.8, y1=1.0), top=0.8, bottom=0.5)


def test_center_band_takes_middle_and_upscales():
    image = _grid()
    exact = center_band(image, width_share=0.6, upscale=1.0)
    assert exact.shape == (100, 120, 3)
    assert exact[0, 0, 0] == 40 and exact[0, -1, 0] == 159
    big = center_band(image)
    assert big.shape == (150, 180, 3)
    assert abs(int(big[75, 90, 0]) - 100) <= 2


@pytest.mark.parametrize(("share", "upscale"), [(0.0, 1.5), (1.2, 1.5), (0.6, 0.0)])
def test_center_band_rejects_bad_arguments(share, upscale):
    with pytest.raises(ValueError):
        center_band(_grid(), width_share=share, upscale=upscale)


def test_no_mask_helper_until_k1():
    # Заливка вне рамки необратимо стирает цель при ошибке детектора:
    # помощника нет, пока замер K1 не покажет пользу маски при надёжной рамке.
    from app.reading import crops

    assert not hasattr(crops, "mask_outside")
