"""Кадр: байты → RGB uint8, один раз на запрос."""

from app.normalize.decode import (
    MAX_PIXELS,
    WHITE,
    DecodeError,
    decode_image,
    decode_on_backgrounds,
    flatten_onto,
    flatten_white,
    format_support,
    has_alpha,
    resize_long_side,
    sniff_format,
)

__all__ = [
    "MAX_PIXELS",
    "WHITE",
    "DecodeError",
    "decode_image",
    "decode_on_backgrounds",
    "flatten_onto",
    "flatten_white",
    "format_support",
    "has_alpha",
    "resize_long_side",
    "sniff_format",
]
