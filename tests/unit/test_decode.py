import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, features

from app.normalize import DecodeError, decode_image, resize_long_side, sniff_format


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def _quadrants() -> np.ndarray:
    """Лежащий кадр 200×120 с четырьмя цветными четвертями — ориентацию видно сразу."""
    arr = np.zeros((120, 200, 3), dtype=np.uint8)
    arr[:60, :100] = (220, 30, 30)
    arr[:60, 100:] = (30, 220, 30)
    arr[60:, :100] = (30, 30, 220)
    arr[60:, 100:] = (240, 240, 240)
    return arr


def _quadrant_means(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    points = [
        (h // 4, w // 4),
        (h // 4, 3 * w // 4),
        (3 * h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
    ]
    return np.array([arr[y - 5 : y + 5, x - 5 : x + 5].mean(axis=(0, 1)) for y, x in points])


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _png_header_only(width: int, height: int) -> bytes:
    """PNG, у которого заголовок обещает огромный кадр, а данных почти нет."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 16))
        + _png_chunk(b"IEND", b"")
    )


def test_png_decodes_losslessly_into_contiguous_rgb():
    arr = _quadrants()
    out = decode_image(_encode(Image.fromarray(arr), "PNG"))
    assert out.dtype == np.uint8 and out.shape == (120, 200, 3)
    assert out.flags.c_contiguous and out.flags.writeable
    np.testing.assert_array_equal(out, arr)


def test_jpeg_keeps_rgb_channel_order():
    out = decode_image(_encode(Image.fromarray(_quadrants()), "JPEG", quality=95))
    np.testing.assert_allclose(_quadrant_means(out), _quadrant_means(_quadrants()), atol=12)


def test_webp_under_jpg_extension_is_detected_by_signature(tmp_path):
    arr = _quadrants()
    path = tmp_path / "bottle.jpg"
    path.write_bytes(_encode(Image.fromarray(arr), "WEBP", lossless=True))
    data = path.read_bytes()
    assert sniff_format(data) == "webp"
    np.testing.assert_array_equal(decode_image(data), arr)


@pytest.mark.parametrize(("orientation", "turns"), [(3, 2), (6, -1), (8, 1)])
def test_exif_orientation_is_applied(orientation, turns):
    arr = _quadrants()
    exif = Image.Exif()
    exif[0x0112] = orientation
    data = _encode(Image.fromarray(arr), "JPEG", quality=95, exif=exif.tobytes())
    out = decode_image(data)
    expected = np.rot90(arr, k=turns)  # k > 0 — против часовой стрелки
    assert out.shape == expected.shape
    np.testing.assert_allclose(_quadrant_means(out), _quadrant_means(expected), atol=12)


def test_sixteen_bit_png_is_stretched_not_clipped():
    gradient = np.tile(np.linspace(0, 65535, 200, dtype=np.uint16), (80, 1))
    image = Image.fromarray(gradient)
    assert image.mode.startswith("I;16")
    out = decode_image(_encode(image, "PNG"))
    assert out.shape == (80, 200, 3)
    assert out.min() < 30 and out.max() > 225
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert np.all(np.diff(out[0, :, 0].astype(int)) >= 0)


def test_rgba_transparency_becomes_white():
    image = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    image.paste((200, 20, 20, 255), (0, 0, 30, 40))
    out = decode_image(_encode(image, "PNG"))
    assert tuple(out[20, 45]) == (255, 255, 255)
    assert tuple(out[20, 10]) == (200, 20, 20)


def test_transparency_can_be_flattened_onto_the_catalog_background():
    """Визуальный канал сводит альфу запроса тем же серым, что и альфу packshot каталога.

    Раньше эталон уходил в модель на сером фоне, а PNG-запрос с альфой — на белом, и фон
    сам становился признаком, которого у второй стороны нет.
    """
    from app.features.views import BACKGROUND

    image = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    image.paste((200, 20, 20, 255), (0, 0, 30, 40))
    data = _encode(image, "PNG")
    assert tuple(decode_image(data, background=BACKGROUND)[20, 45]) == BACKGROUND
    assert tuple(decode_image(data)[20, 45]) == (255, 255, 255)  # для чтения букв — белый
    assert tuple(decode_image(data, background=BACKGROUND)[20, 10]) == (200, 20, 20)


def test_palette_transparency_becomes_white():
    indices = np.zeros((20, 20), dtype=np.uint8)
    indices[:, 10:] = 1
    image = Image.fromarray(indices, mode="P")
    image.putpalette([0, 0, 0, 10, 120, 200] + [0] * 762)
    out = decode_image(_encode(image, "PNG", transparency=0))
    assert tuple(out[5, 2]) == (255, 255, 255)
    assert tuple(out[5, 15]) == (10, 120, 200)


@pytest.mark.parametrize(
    ("mode", "color"), [("L", 90), ("CMYK", (0, 255, 255, 0)), ("1", 1), ("LA", (90, 255))]
)
def test_other_modes_become_three_channels(mode, color):
    fmt = "JPEG" if mode == "CMYK" else "PNG"
    out = decode_image(_encode(Image.new(mode, (32, 24), color), fmt))
    assert out.shape == (24, 32, 3) and out.dtype == np.uint8


def test_cmyk_jpeg_colors_survive():
    out = decode_image(_encode(Image.new("CMYK", (32, 32), (0, 255, 255, 0)), "JPEG", quality=95))
    np.testing.assert_allclose(out[16, 16], (255, 0, 0), atol=40)


def _truncated_jpeg() -> bytes:
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    data = _encode(Image.fromarray(noise), "JPEG", quality=95)
    return data[: len(data) // 2]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"not an image at all",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,
        b"RIFF\x10\x00\x00\x00WEBPVP8 garbage",
        _truncated_jpeg(),
    ],
)
def test_broken_bytes_raise_decode_error(payload):
    with pytest.raises(DecodeError):
        decode_image(payload)


@pytest.mark.parametrize("side", [12_000, 30_000])
def test_huge_frame_is_refused_before_decoding(side):
    with pytest.raises(DecodeError):
        decode_image(_png_header_only(side, side))


def test_max_pixels_limit_is_inclusive():
    data = _encode(Image.new("RGB", (100, 100), (1, 2, 3)), "PNG")
    assert decode_image(data, max_pixels=10_000).shape == (100, 100, 3)
    with pytest.raises(DecodeError, match="предел"):
        decode_image(data, max_pixels=9_999)


@pytest.mark.parametrize(
    ("max_pixels", "shape"),
    [
        (10_000, (60, 100, 3)),  # 200×120 = 24 000 > 10 000 → вдвое: 100×60 = 6 000
        (5_000, (30, 50, 3)),  # вдвое мало (6 000) → вчетверо
        (400, (15, 25, 3)),  # вчетверо мало (1 500) → в восемь раз
    ],
)
def test_jpeg_over_the_limit_is_decoded_reduced_not_refused(max_pixels, shape):
    """Снимок телефона на 108 Мп — JPEG: libjpeg уменьшает его прямо при разжатии.

    Раньше кадр больше предела отвергался, и скрипт организатора записывал null, хотя моделям
    нужно не больше 1024 px. Уменьшение — наименьшее из 2, 4, 8, при котором кадр влезает.
    """
    data = _encode(Image.fromarray(_quadrants()), "JPEG", quality=95)
    out = decode_image(data, max_pixels=max_pixels)
    assert out.shape == shape
    h, w = out.shape[:2]
    centres = [out[y, x] for y in (h // 4, 3 * h // 4) for x in (w // 4, 3 * w // 4)]
    np.testing.assert_allclose(centres, _quadrant_means(_quadrants()), atol=16)


def test_reduced_jpeg_keeps_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6
    data = _encode(Image.fromarray(_quadrants()), "JPEG", quality=95, exif=exif.tobytes())
    out = decode_image(data, max_pixels=10_000)
    assert out.shape == (100, 60, 3)
    expected = np.rot90(_quadrants(), k=-1)
    np.testing.assert_allclose(_quadrant_means(out), _quadrant_means(expected), atol=16)


def test_jpeg_beyond_eightfold_and_other_formats_over_the_limit_are_refused():
    jpeg = _encode(Image.fromarray(_quadrants()), "JPEG", quality=95)
    with pytest.raises(DecodeError, match="предел"):
        decode_image(jpeg, max_pixels=300)  # даже в восемь раз — 25×15 = 375
    png = _encode(Image.fromarray(_quadrants()), "PNG")
    with pytest.raises(DecodeError, match="только у JPEG"):
        decode_image(png, max_pixels=10_000)


def test_real_108_megapixel_jpeg_decodes_under_the_default_limit():
    """12 000×9 000 (108 Мп, 1,7 МБ) при пределе 60 Мп разжимается вдвое меньшим: 6 000×4 500."""
    data = _encode(Image.new("RGB", (12_000, 9_000), (200, 10, 10)), "JPEG", quality=80)
    out = decode_image(data)
    assert out.shape == (4_500, 6_000, 3)
    np.testing.assert_allclose(out[2_000, 3_000], (200, 10, 10), atol=6)


def test_format_support_reports_heic_by_plugin(monkeypatch):
    from app.normalize import decode

    monkeypatch.setattr(decode, "heif_available", lambda: False)
    support = decode.format_support()
    assert support["heic"] is False and support["jpeg"] and support["png"]
    assert support["avif"] is bool(features.check("avif"))
    monkeypatch.setattr(decode, "heif_available", lambda: True)
    assert decode.format_support()["heic"] is True and decode.format_support()["avif"] is True


def test_heic_without_plugin_gives_clear_error():
    data = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic" + b"\x00" * 64
    assert sniff_format(data) == "heic"
    try:
        import pillow_heif  # noqa: F401
    except ImportError:
        with pytest.raises(DecodeError, match="pillow-heif"):
            decode_image(data)
    else:
        with pytest.raises(DecodeError):
            decode_image(data)


def test_avif_roundtrip_when_supported():
    if not features.check("avif"):
        pytest.skip("Pillow собран без AVIF")
    data = _encode(Image.fromarray(_quadrants()), "AVIF", quality=90)
    assert sniff_format(data) == "avif"
    out = decode_image(data)
    np.testing.assert_allclose(_quadrant_means(out), _quadrant_means(_quadrants()), atol=20)


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "jpeg"),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8L", "webp"),
        (b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf", "avif"),
        (b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1heic", "heic"),
        (b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2", None),
        (b"<html>", None),
    ],
)
def test_sniff_format_by_signature(head, expected):
    assert sniff_format(head) == expected


def test_resize_long_side_keeps_aspect():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    assert resize_long_side(image, 200).shape == (150, 200, 3)
    assert resize_long_side(image, 800).shape == (600, 800, 3)


def test_resize_long_side_without_upscale_returns_same_frame():
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    assert resize_long_side(image, 80, upscale=False) is image
    with pytest.raises(ValueError):
        resize_long_side(image, 0)


def test_resize_down_averages_fine_stripes():
    """INTER_AREA при уменьшении: полосы в 1 px сливаются в серый, а не в муар."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, ::2] = 255
    out = resize_long_side(image, 25)
    assert 100 <= out.mean() <= 155 and out.std() < 10
