"""Разбор кадра на два фона за один раз и карточки каталога из выгрузки организатора."""

from __future__ import annotations

import io

import numpy as np
import pytest
from api_env import API_RECORDS
from PIL import Image

from app.api.cards import CatalogCards, card_of, read_catalog_csv, sugar_label
from app.features.views import BACKGROUND
from app.normalize import WHITE, DecodeError, decode_image, decode_on_backgrounds


def _png(mode: str, color) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (20, 10), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_alpha_is_flattened_on_each_background_like_decode_image():
    data = _png("RGBA", (200, 0, 0, 0))  # полностью прозрачный кадр
    gray, white = decode_on_backgrounds(data, (BACKGROUND, WHITE))
    assert tuple(gray[0, 0]) == BACKGROUND and tuple(white[0, 0]) == WHITE
    assert np.array_equal(gray, decode_image(data, background=BACKGROUND))
    assert np.array_equal(white, decode_image(data))


def test_frame_without_alpha_is_decoded_once_and_shared():
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), (10, 120, 30)).save(buffer, format="JPEG")
    first, second = decode_on_backgrounds(buffer.getvalue(), (BACKGROUND, WHITE))
    assert first is second
    assert np.array_equal(first, decode_image(buffer.getvalue()))


def test_decode_on_backgrounds_errors_like_decode_image():
    with pytest.raises(DecodeError):
        decode_on_backgrounds(b"", (WHITE,))
    with pytest.raises(DecodeError):
        decode_on_backgrounds(_png("RGB", (1, 2, 3)), (WHITE,), max_pixels=100)
    with pytest.raises(ValueError):
        decode_on_backgrounds(_png("RGB", (1, 2, 3)), ())


def test_catalog_csv_takes_first_row_per_slug(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text(
        "Название вина,Категория,Цвет,Регион,Сорт винограда,Описание,Винодельня,Slug,Название фото\n"
        ' Кокур,Белое,Соломенный,Крым,"Кокур Белый, Алиготе",Первое.,Солнечная Долина,kokur,k.webp\n'
        " Кокур,Белое,Соломенный,Крым,Кокур Белый,Второе.,Солнечная Долина,kokur,k.webp\n",
        encoding="utf-8-sig",
    )
    rows = read_catalog_csv(path)
    assert rows["kokur"]["name"] == "Кокур" and rows["kokur"]["description"] == "Первое."
    record = {"slug": "kokur", "fields": {}}
    card = card_of(record, rows["kokur"], None)
    assert card.grapes == ["Кокур Белый", "Алиготе"]  # сортов в разметке нет — берутся из CSV
    assert card.name == "Кокур" and card.region == "Крым" and card.photo_path is None


def test_catalog_csv_keeps_description_as_is(tmp_path):
    """«Описание» выгрузки идёт в карточку как есть: абзацы остаются, края без пробелов."""
    path = tmp_path / "c.csv"
    path.write_text(
        "Название вина,Категория,Цвет,Регион,Сорт винограда,Описание,Винодельня,Slug,Название фото\n"
        ' Кокур,Белое,,Крым,Кокур,"  Первый абзац.\n\nВторой  абзац. ",Долина,kokur,k.webp\n',
        encoding="utf-8-sig",
        newline="\n",  # как в выгрузке: переносы строк — \n и на Windows
    )
    assert read_catalog_csv(path)["kokur"]["description"] == "Первый абзац.\n\nВторой  абзац."


def test_cards_build_without_optional_files(tmp_path):
    cards = CatalogCards.build(
        API_RECORDS, csv_path=tmp_path / "missing.csv", photo_map=tmp_path / "missing.csv"
    )
    assert len(cards) == len(API_RECORDS)
    assert cards.sources["csv"] is None and cards.sources["photo_map"] is None
    # класс сахара разметки у 2 013 позиций — из живого портала: карточка берёт правило
    # выгрузки, и у «Мускат» (alfa-muskat) сахара нет ни в названии, ни в slug
    assert cards.get("alfa-muskat").sugar == "" and cards.get("alfa-muskat").sugar_class is None
    record = {"slug": "x-beloe-suhoe-12", "name": "Икс", "fields": {"sugar": {"class": "brut"}}}
    card = card_of(record, None, None)
    assert (card.sugar, card.sugar_class) == ("сухое", "suhoe")
    assert not {"published", "live_category"} & set(card.to_dict())
    assert sugar_label("brut") == "брют" and sugar_label("nonsense") == ""
