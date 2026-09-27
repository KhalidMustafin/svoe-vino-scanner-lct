import numpy as np
import pytest

from app.detect import TargetSelection
from app.features.views import (
    BACKGROUND,
    QUERY_WINDOWS,
    WINDOWS_NOTE,
    alpha_bounds,
    flatten_alpha,
    from_bottle,
    from_packshot,
    from_query,
    order_views,
)
from app.reading.contracts import Box


def _packshot(
    height: int = 200, width: int = 100, pad: int = 20, color: tuple[int, int, int] = (200, 30, 40)
) -> np.ndarray:
    """Бутылка на прозрачном фоне: непрозрачный прямоугольник с полями `pad`."""
    image = np.zeros((height, width, 4), dtype=np.uint8)
    image[pad : height - pad, pad : width - pad, :3] = color
    image[pad : height - pad, pad : width - pad, 3] = 255
    return image


def _frame(height: int = 200, width: int = 300) -> np.ndarray:
    """Кадр, в пикселях которого записаны их координаты: канал 0 — x, канал 1 — y."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    return image


def test_alpha_bounds_finds_opaque_rectangle():
    assert alpha_bounds(_packshot()) == (20, 20, 80, 180)


def test_alpha_bounds_ignores_soft_edges_below_threshold():
    image = _packshot()
    image[0, 0, 3] = 4  # мягкий край WebP: почти прозрачный пиксель — не бутылка
    assert alpha_bounds(image) == (20, 20, 80, 180)
    assert alpha_bounds(image, alpha_min=2)[0] == 0


def test_alpha_bounds_without_alpha_is_whole_frame():
    assert alpha_bounds(_frame(50, 70)) == (0, 0, 70, 50)


def test_alpha_bounds_of_empty_file_is_whole_frame():
    assert alpha_bounds(np.zeros((30, 40, 4), dtype=np.uint8)) == (0, 0, 40, 30)


def test_flatten_alpha_fills_transparent_with_background():
    out = flatten_alpha(_packshot(), (128, 128, 128))
    assert out.shape == (200, 100, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (128, 128, 128)
    assert tuple(out[100, 50]) == (200, 30, 40)


def test_flatten_alpha_keeps_rgb_and_rejects_float():
    rgb = _frame(10, 10)
    assert np.array_equal(flatten_alpha(rgb), rgb)
    with pytest.raises(ValueError, match="uint8"):
        flatten_alpha(rgb.astype(np.float32))


def test_from_packshot_gives_three_views_cropped_by_alpha():
    views = from_packshot(_packshot())
    assert set(views) == {"bottle", "label", "band"}
    assert views["bottle"].shape == (160, 60, 3)  # поля обрезаны: 200-2*20, 100-2*20
    assert all(v.dtype == np.uint8 and v.ndim == 3 for v in views.values())


def test_from_packshot_label_is_middle_of_the_bottle():
    views = from_packshot(_packshot())
    bottle_height = views["bottle"].shape[0]  # 160
    # 30–85 % высоты бутылки: у горлышка и у донышка этикетки нет
    assert views["label"].shape[0] == round(0.85 * bottle_height) - round(0.30 * bottle_height)
    assert views["label"].shape[1] == views["bottle"].shape[1]


def test_from_packshot_band_is_narrower_and_upscaled():
    views = from_packshot(_packshot())
    label, band = views["label"], views["band"]
    assert band.shape[1] == pytest.approx(0.6 * label.shape[1] * 1.5, abs=2)
    assert band.shape[0] == pytest.approx(label.shape[0] * 1.5, abs=2)


def test_from_packshot_background_is_a_parameter():
    image = _packshot()
    image[20:40, 20:30, 3] = 0  # прозрачный вырез внутри рамки: там и видно фон
    assert tuple(from_packshot(image, background=(7, 8, 9))["bottle"][0, 0]) == (7, 8, 9)
    assert tuple(from_packshot(image)["bottle"][0, 0]) == BACKGROUND


def test_views_are_capped_by_max_side():
    """Огромный packshot режется уже уменьшенным: перед моделью всё равно квадрат 448."""
    views = from_packshot(_packshot(height=4000, width=1000, pad=0), max_side=400)
    assert views["bottle"].shape[:2] == (400, 100)
    assert max(views["band"].shape[:2]) < 400 * 1.5 + 2


def test_max_side_does_not_upscale_small_frames():
    views = from_packshot(_packshot(height=200, width=100, pad=20), max_side=896)
    assert views["bottle"].shape[:2] == (160, 60)


def test_query_windows_are_cut_from_the_shrunken_frame():
    views = from_query(_frame(2000, 1000), max_side=500)
    assert views["full"].shape[:2] == (500, 250)
    assert views["bottle"].shape[:2] == (500, 150)  # 0.2–0.8 ширины


def test_from_bottle_rejects_rgba():
    with pytest.raises(ValueError, match="RGB uint8"):
        from_bottle(_packshot())


def test_from_query_uses_confident_box():
    target = TargetSelection(
        target=Box(x0=0.4, y0=0.1, x1=0.6, y1=0.9),
        method="detector",
        score=0.9,
        confident=True,
    )
    views = from_query(_frame(), target)
    assert set(views) == {"bottle", "label", "band"}
    # рамка 0.4–0.6 кадра 300×200 плюс отступ 0.08 ширины и 0.04 высоты рамки
    assert views["bottle"].shape == (174, 70, 3)
    assert int(views["bottle"][0, 0, 0]) == 115  # x левого края кропа
    assert int(views["bottle"][0, 0, 1]) == 13  # y верхнего края кропа


def test_from_query_without_target_returns_windows():
    views = from_query(_frame())
    assert set(views) == set(QUERY_WINDOWS)
    assert "full" in views
    assert views["full"].shape == (200, 300, 3)
    assert views["bottle"].shape == (200, 180, 3)  # 0.2–0.8 ширины


def test_from_query_with_unsure_box_falls_back_to_windows():
    target = TargetSelection(
        target=Box(x0=0.4, y0=0.1, x1=0.6, y1=0.9),
        method="detector",
        score=0.4,
        confident=False,
    )
    assert set(from_query(_frame(), target)) == set(QUERY_WINDOWS)


def test_from_query_rejects_rgba():
    with pytest.raises(ValueError, match="RGB uint8"):
        from_query(_packshot())


def test_query_windows_are_inside_the_frame():
    for name, box in QUERY_WINDOWS.items():
        assert 0.0 <= box.x0 < box.x1 <= 1.0, name
        assert 0.0 <= box.y0 < box.y1 <= 1.0, name


def test_order_views_is_canonical_and_rejects_strangers():
    views = {"band": np.zeros(1), "full": np.zeros(1), "bottle": np.zeros(1)}
    assert order_views(views) == ["bottle", "band", "full"]
    with pytest.raises(ValueError, match="неизвестные виды"):
        order_views({"neck": np.zeros(1)})


def test_background_default_is_grey():
    assert BACKGROUND == (128, 128, 128)


def test_views_never_share_pixels_with_the_input():
    """Кадр меньше MAX_SIDE не уменьшался, и вид оказывался самим входным массивом.

    Запись в такой вид молча меняла исходный снимок у вызывающего — соглашение
    `app.reading.crops` («на выходе всегда новый массив») ломалось.
    """
    bottle = np.zeros((10, 10, 3), dtype=np.uint8)
    views = from_bottle(bottle)
    assert views["bottle"] is not bottle and views["bottle"].base is not bottle
    views["bottle"][0, 0] = (1, 2, 3)
    assert bottle[0, 0].tolist() == [0, 0, 0]

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    windows = from_query(image)
    for name, view in windows.items():
        assert view is not image and view.base is not image, name
        view[0, 0] = (7, 7, 7)
    assert image[0, 0].tolist() == [0, 0, 0]


def test_windows_note_says_the_geometry_is_in_sample():
    # Окна подобраны на тех же 358 парах, на которых стенд потом меряет точность.
    assert "358" in WINDOWS_NOTE and "in-sample" in WINDOWS_NOTE
