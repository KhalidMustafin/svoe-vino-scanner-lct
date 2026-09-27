"""Кадр из байтов: формат по сигнатуре, ориентация, 8 бит, без прозрачности.

Один раз на запрос и дальше только массив RGB uint8: между слоями кадр в JPEG не
перекодируется, иначе каждое звено добавляет свои артефакты к буквам этикетки.
"""

from __future__ import annotations

import importlib.util
import io
import warnings
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps

ImageFormat = Literal["jpeg", "png", "webp", "gif", "bmp", "tiff", "heic", "avif"]

# Кадр больше этого числа пикселей целиком не разжимается: 60 Мп RGB — уже 180 МБ памяти.
# JPEG больше предела разжимается сразу уменьшенным (`_draft_under`): так проходят снимки
# телефонов на 108 Мп. Остальные форматы больше предела — `DecodeError`: уменьшить их можно
# только после полного разжатия, а это та самая память. Выше порога Pillow против
# «декомпрессионной бомбы» (~179 Мп) не открывается ничего.
MAX_PIXELS = 60_000_000

#: Во сколько раз libjpeg умеет уменьшать JPEG прямо при разжатии (масштабирование DCT).
JPEG_DRAFT_FACTORS: tuple[int, ...] = (2, 4, 8)

#: Фон по умолчанию под прозрачностью: белый, как у фото каталога и как любит OCR.
WHITE: tuple[int, int, int] = (255, 255, 255)

# Имя формата → плагины Pillow, которым разрешено открыть байты.
_PIL_FORMATS: dict[ImageFormat, tuple[str, ...]] = {
    "jpeg": ("JPEG", "MPO"),
    "png": ("PNG",),
    "webp": ("WEBP",),
    "gif": ("GIF",),
    "bmp": ("BMP",),
    "tiff": ("TIFF",),
    "heic": ("HEIF",),
    "avif": ("AVIF", "HEIF"),
}
_HEIC_BRANDS = frozenset({b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs"})
_AVIF_BRANDS = frozenset({b"avif", b"avis"})
_MIF_BRANDS = frozenset({b"mif1", b"msf1"})
_HIGH_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N", "I", "F"})


class DecodeError(ValueError):
    """Байты не разобрались как поддерживаемое изображение."""


def sniff_format(data: bytes) -> ImageFormat | None:
    """Формат по сигнатуре байтов, а не по расширению: WebP часто приходит как .jpg."""
    head = bytes(data[:64])
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return _sniff_isobmff(bytes(data[:512]))


def _sniff_isobmff(head: bytes) -> ImageFormat | None:
    """HEIC и AVIF: коробка `ftyp` с основным и совместимыми брендами."""
    if len(head) < 16 or head[4:8] != b"ftyp":
        return None
    size = int.from_bytes(head[:4], "big")
    end = min(len(head), max(size, 16))
    brands = {head[8:12]} | {head[i : i + 4] for i in range(16, end - 3, 4)}
    if brands & _AVIF_BRANDS:
        return "avif"
    if brands & _HEIC_BRANDS or brands & _MIF_BRANDS:
        return "heic"
    return None


def heif_available() -> bool:
    """Установлен ли `pillow-heif`: без него HEIC (формат камеры iPhone) не разбирается."""
    return importlib.util.find_spec("pillow_heif") is not None


def format_support() -> dict[str, bool]:
    """Какие форматы кадра разбираются в этом окружении — для `/v1/health`.

    JPEG, PNG, GIF, BMP и TIFF Pillow читает всегда. WebP и AVIF — если собран с libwebp и
    libavif (колёса PyPI собраны). HEIC — только с `pillow-heif`, иначе `DecodeError`.
    """
    from PIL import features

    heif = heif_available()
    return {
        "jpeg": True,
        "png": True,
        "webp": bool(features.check("webp")),
        "heic": heif,
        "avif": bool(features.check("avif")) or heif,
    }


def _draft_under(image: Image.Image, max_pixels: int) -> Image.Image:
    """JPEG больше `max_pixels` — разжать сразу уменьшенным в 2, 4 или 8 раз.

    `Image.draft` настраивает libjpeg на масштабирование при разжатии: полный кадр в память не
    попадает, 108 Мп при уменьшении вдвое разжимаются как 27 Мп. Берётся наименьшее уменьшение,
    при котором кадр влезает в предел: моделям нужно не больше 1024 px (VLM) и 448 px (SigLIP),
    а окна видов режутся из того, что осталось. Кадр до предела не трогается — ровно то, что
    видел замер. Остальные форматы так не умеют: кадр возвращается как есть.
    """
    if image.format not in ("JPEG", "MPO"):
        return image
    width, height = image.size
    for factor in JPEG_DRAFT_FACTORS:
        reduced = -(-width // factor) * -(-height // factor)
        if reduced <= max_pixels:
            image.draft(image.mode, (width // factor, height // factor))
            break
    return image


def _ensure_plugin(fmt: ImageFormat) -> None:
    """HEIC Pillow сам не читает; AVIF читает, если собран с libavif."""
    if fmt not in ("heic", "avif"):
        return
    if fmt == "avif":
        from PIL import features

        if features.check("avif"):
            return
    try:
        import pillow_heif
    except ImportError as exc:
        raise DecodeError(
            f"{fmt.upper()}: для разбора нужен пакет pillow-heif, он не установлен"
        ) from exc
    pillow_heif.register_heif_opener()
    if fmt == "avif" and hasattr(pillow_heif, "register_avif_opener"):
        pillow_heif.register_avif_opener()


def _to_eight_bits(image: Image.Image) -> Image.Image:
    """16-битный PNG и TIFF с плавающей точкой — в обычные восемь бит.

    `convert("RGB")` у таких кадров обрезает всё выше 255, и снимок выходит сплошь
    белым или чёрным. Значения растягиваются на 0–255 целиком.
    """
    if image.mode not in _HIGH_BIT_MODES:
        return image
    values = np.asarray(image, dtype=np.float32)
    low, high = float(values.min()), float(values.max())
    if high > low:
        scaled = (values - low) * (255.0 / (high - low))
    else:
        scaled = np.zeros_like(values)
    return Image.fromarray(np.clip(scaled + 0.5, 0, 255).astype(np.uint8), mode="L")


def has_alpha(image: Image.Image) -> bool:
    """Есть ли у кадра прозрачность, которую придётся сводить на фон."""
    return image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        image.mode == "P" and "transparency" in image.info
    )


def flatten_onto(image: Image.Image, background: tuple[int, int, int] = WHITE) -> Image.Image:
    """Прозрачность — на заданный фон через альфа-маску."""
    if has_alpha(image):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, tuple(background))
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        return canvas
    return image if image.mode == "RGB" else image.convert("RGB")


def flatten_white(image: Image.Image) -> Image.Image:
    """Прозрачность — на белый фон через альфа-маску, как у фото каталога."""
    return flatten_onto(image, WHITE)


def decode_image(
    data: bytes, *, max_pixels: int = MAX_PIXELS, background: tuple[int, int, int] = WHITE
) -> np.ndarray:
    """Байты снимка → RGB uint8 HxWx3, повёрнутый по EXIF, без прозрачности.

    Размер проверяется по заголовку до разжатия пикселей. Любая неудача — `DecodeError`.

    `background` — чем заливается альфа. По умолчанию белый: так модуль чтения видит буквы
    на том же фоне, что и фото каталога. Визуальный канал передаёт сюда
    `app.features.views.BACKGROUND`, чтобы запрос и эталон сходились и по фону.
    """
    return decode_on_backgrounds(data, (background,), max_pixels=max_pixels)[0]


def decode_on_backgrounds(
    data: bytes,
    backgrounds: Sequence[tuple[int, int, int]],
    *,
    max_pixels: int = MAX_PIXELS,
) -> list[np.ndarray]:
    """Один разбор байтов — кадр на каждом из фонов `backgrounds`, в том же порядке.

    Сервису нужен один снимок дважды: визуальному каналу — на сером фоне эталонов, модулю
    чтения — на белом, как в замерах `bench.retrieval` и `bench.ocr_bench`. Разжимать 12 Мп
    дважды ради PNG с альфой незачем: сводится только прозрачность. У кадра без альфы (любой
    JPEG) фон ни на что не влияет, и все элементы списка — один и тот же массив: менять его на
    месте нельзя. Правила и ошибки — как у `decode_image`.
    """
    if not backgrounds:
        raise ValueError("decode_on_backgrounds: нужен хотя бы один фон")
    if not isinstance(data, bytes | bytearray | memoryview) or len(data) == 0:
        raise DecodeError("пустые данные вместо изображения")
    fmt = sniff_format(data)
    if fmt is None:
        raise DecodeError("неизвестная сигнатура: ожидается JPEG, PNG, WebP, HEIC или AVIF")
    _ensure_plugin(fmt)
    try:
        with warnings.catch_warnings():
            # Свою границу проверяем сами ниже; предупреждение Pillow здесь — шум.
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            source = Image.open(io.BytesIO(bytes(data)), formats=_PIL_FORMATS[fmt])
            width, height = source.size
            if width <= 0 or height <= 0:
                raise DecodeError(f"пустой кадр {width}x{height}")
            if width * height > max_pixels:
                source = _draft_under(source, max_pixels)
                if source.size[0] * source.size[1] > max_pixels:
                    raise DecodeError(
                        f"кадр {width}x{height} больше предела {max_pixels} пикселей "
                        "(уменьшение при разжатии есть только у JPEG, до 8 раз)"
                    )
            source.load()
            oriented = _to_eight_bits(ImageOps.exif_transpose(source))
            if has_alpha(oriented):
                flat: dict[tuple[int, ...], np.ndarray] = {}
                for background in backgrounds:
                    key = tuple(background)
                    if key not in flat:
                        flat[key] = np.array(flatten_onto(oriented, background), dtype=np.uint8)
                arrays = [flat[tuple(background)] for background in backgrounds]
            else:
                array = np.array(flatten_onto(oriented), dtype=np.uint8)
                arrays = [array] * len(backgrounds)
    except DecodeError:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, ValueError) as exc:
        raise DecodeError(f"{fmt}: изображение не разобралось: {exc}") from exc
    for array in arrays:
        if array.ndim != 3 or array.shape[2] != 3:
            raise DecodeError(f"неожиданная форма кадра после разбора: {array.shape}")
    # Массив из Pillow уже сплошной, и `ascontiguousarray` вернёт его же: общий массив
    # кадров без альфы остаётся общим.
    return [np.ascontiguousarray(array) for array in arrays]


def resize_long_side(image: np.ndarray, px: int, *, upscale: bool = True) -> np.ndarray:
    """Длинная сторона — `px`, пропорции сохраняются.

    Уменьшение — INTER_AREA (без муара на мелких буквах), увеличение — Ланцош.
    При `upscale=False` кадр меньше `px` возвращается как есть.
    """
    if px <= 0:
        raise ValueError(f"resize_long_side: px должен быть > 0, получено {px}")
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("resize_long_side: пустой кадр")
    long_side = max(height, width)
    if long_side == px or (long_side < px and not upscale):
        return image
    scale = px / long_side
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(image, size, interpolation=interpolation)
