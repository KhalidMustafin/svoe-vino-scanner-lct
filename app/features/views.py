"""Виды кадра: что именно показывают модели зрения. Только геометрия, без решений.

Эталон каталога — packshot: бутылка на прозрачном фоне, медиана 336×1080. Вектор такого
файла целиком кодирует в основном пустоту полей и силуэт, а не этикетку, и близнецы
каталога (одна этикетка, разный год) для него неразличимы. Поэтому и эталон, и запрос
режутся на одни и те же виды, а близость берётся лучшая по всем парам:

    bottle  вся бутылка — силуэт, цвет стекла, колпачок;
    label   средние 30–85 % высоты — контрэтикетки и горлышка в кадре нет;
    band    центральные 60 % ширины этикетки, увеличенные Ланцошем: середина цилиндра
            не сжата кривизной, и там читается бренд;
    full    весь кадр запроса, когда бутылку на нём не нашли.

Фон packshot заливается серым, а не белым: у кадра с телефона за бутылкой полка, и
белый фон эталона сам по себе становится признаком, которого у запроса нет. Значение
`BACKGROUND` — параметр, его ещё предстоит замерить на парах.

Тем же цветом сводится прозрачность запроса: `decode_image` по умолчанию заливает альфу
белым (так буквы читаются лучше), поэтому визуальный канал зовёт его как
`decode_image(data, background=BACKGROUND)` — иначе эталон уходил бы в модель на сером
фоне, а PNG-запрос с альфой на белом, и фон сам стал бы признаком.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.detect.bottles import TargetSelection
from app.features.contracts import VIEWS, ViewName
from app.normalize.decode import resize_long_side
from app.reading.contracts import Box
from app.reading.crops import box_pixels, center_band, crop_box

#: Полоса этикетки внутри бутылки: доли высоты от верха рамки.
LABEL_TOP = 0.30
LABEL_BOTTOM = 0.85

#: Лента внутри этикетки: доля ширины и увеличение Ланцошем.
BAND_WIDTH = 0.60
BAND_UPSCALE = 1.5

#: Длинная сторона кропа. Перед моделью вид всё равно ужмётся до 448 (`LETTERBOX_SIDE`),
#: а packshot каталога бывает и 1950×7371: три вида такого файла в полном размере — это
#: около 100 МБ на одну бутылку, и батч из 32 кадров в память уже не влезает.
MAX_SIDE = 896

#: Чем заливается прозрачный фон packshot.
BACKGROUND: tuple[int, int, int] = (128, 128, 128)

#: Пиксель прозрачнее этого — фон, а не бутылка. У WebP каталога края мягкие.
ALPHA_MIN = 8

#: Оговорка о происхождении окон: едет в `metrics.json` через `bench.queries.notes_for`.
WINDOWS_NOTE = (
    "окна запроса (QUERY_WINDOWS), LETTERBOX_SIDE=448 и границы label/band перенесены из "
    "«Лозы», где их выбрали по метрике на этих же 358 парах с phone_shot (Code/README.md: "
    "«Кадр целиком 28,2 % → + окна кадра 43,6 %»). Цифры pairs и pairs_phone для этой "
    "геометрии — in-sample, при защите приводить их как подобранные на том же наборе"
)

#: Окна кадра запроса, когда рамке бутылки верить нельзя. Перенесены из «Лозы»
#: (`backend/app/vision/embedder.py: QUERY_WINDOWS`), где они подняли top-1 на кадрах
#: «как с телефона» с 28 до 44 %. Там окон было пять; шестое имя виду не нужно, и окно
#: «без краёв» (0.1–0.9) выброшено как ближайший двойник кадра целиком.
#:
#: Важно: те 28 → 44 % измерены на том же наборе 358 пар, что служит здесь главным
#: замером, — см. `WINDOWS_NOTE`. Геометрия видов подобрана на нём, а не проверена по нему.
QUERY_WINDOWS: dict[ViewName, Box] = {
    "full": Box(x0=0.0, y0=0.0, x1=1.0, y1=1.0),
    "bottle": Box(x0=0.2, y0=0.0, x1=0.8, y1=1.0),  # бутылка стоит по центру
    "label": Box(x0=0.15, y0=0.3, x1=0.85, y1=0.95),
    "band": Box(x0=0.25, y0=0.2, x1=0.75, y1=0.8),
}


def flatten_alpha(image: np.ndarray, background: tuple[int, int, int] = BACKGROUND) -> np.ndarray:
    """RGBA, серый или RGB кадр → RGB uint8; прозрачность сводится на `background`."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"flatten_alpha: ожидается uint8, получено {array.dtype}")
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"flatten_alpha: ожидается HxW, HxWx3 или HxWx4, получено {array.shape}")
    if array.shape[2] == 3:
        return np.ascontiguousarray(array)
    alpha = array[..., 3:4].astype(np.float32) / 255.0
    fill = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    blended = array[..., :3].astype(np.float32) * alpha + fill * (1.0 - alpha)
    return np.ascontiguousarray(np.clip(blended + 0.5, 0, 255).astype(np.uint8))


def alpha_bounds(image: np.ndarray, alpha_min: int = ALPHA_MIN) -> tuple[int, int, int, int]:
    """Рамка непрозрачного в пикселях (x0, y0, x1, y1). Альфы нет — весь кадр."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 4:
        height, width = array.shape[:2]
        return 0, 0, width, height
    mask = array[..., 3] >= alpha_min
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:  # пустой файл: пусть решает кадр целиком
        height, width = array.shape[:2]
        return 0, 0, width, height
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def from_bottle(
    bottle: np.ndarray,
    *,
    label_top: float = LABEL_TOP,
    label_bottom: float = LABEL_BOTTOM,
    band_width: float = BAND_WIDTH,
    band_upscale: float = BAND_UPSCALE,
    max_side: int = MAX_SIDE,
) -> dict[ViewName, np.ndarray]:
    """Три вида по вырезанной бутылке: она сама, полоса этикетки и лента внутри неё.

    Каждый вид — новый массив, как в `app.reading.crops`: кадр меньше `max_side` не
    уменьшается, и без копии вид «bottle» был бы самим входным кадром — запись в него
    молча испортила бы исходный снимок у вызывающего.
    """
    if bottle.ndim != 3 or bottle.shape[2] != 3 or bottle.dtype != np.uint8:
        raise ValueError(f"from_bottle: ожидается RGB uint8 HxWx3, получено {bottle.shape}")
    resized = resize_long_side(bottle, max_side, upscale=False)
    frame = resized.copy() if resized is bottle else np.ascontiguousarray(resized)
    label = crop_box(
        frame,
        Box(x0=0.0, y0=label_top, x1=1.0, y1=label_bottom),
        pad_x=0.0,
        pad_y=0.0,
    )
    return {
        "bottle": frame,
        "label": label,
        "band": center_band(label, band_width, band_upscale),
    }


def from_packshot(
    image_rgba: np.ndarray,
    *,
    background: tuple[int, int, int] = BACKGROUND,
    alpha_min: int = ALPHA_MIN,
    max_side: int = MAX_SIDE,
    **view_params: float,
) -> dict[ViewName, np.ndarray]:
    """Виды эталона каталога: обрезка по альфе, фон — `background`, затем bottle/label/band.

    Прозрачные поля обрезаются до заливки: иначе вектор кодировал бы размер пустого
    холста, который у каждого файла свой. Остальные именованные параметры — у `from_bottle`.

    Кадр уменьшается до заливки фона, а не после: на packshot 1950×7371 сведение альфы
    в полном размере стоит 540 мс против 15 мс после уменьшения — почти вся сборка
    индекса. Платим за это разницей до 25/255 на считаных пикселях по краю бутылки
    (в среднем 0,03) — там, где альфа не 0 и не 255.
    """
    x0, y0, x1, y1 = alpha_bounds(image_rgba, alpha_min)
    cropped = np.asarray(image_rgba)[y0:y1, x0:x1]
    small = resize_long_side(cropped, max_side, upscale=False)
    return from_bottle(flatten_alpha(small, background), max_side=max_side, **view_params)


def from_query(
    image: np.ndarray,
    target: TargetSelection | None = None,
    *,
    windows: dict[ViewName, Box] | None = None,
    max_side: int = MAX_SIDE,
    **view_params: float,
) -> dict[ViewName, np.ndarray]:
    """Виды кадра запроса.

    Рамка надёжна — те же три вида, что у эталона, но по рамке: снимок у полки тогда
    приведён к виду packshot. Рамки нет или ей нельзя верить — кадр целиком и окна по
    долям сторон: одно из них накрывает бутылку плотно почти при любом кадрировании.
    Остальные именованные параметры — у `from_bottle`.

    Окна выбраны не здесь: см. `WINDOWS_NOTE` — их подобрали на том же наборе пар,
    на котором стенд потом меряет точность.

    Каждый вид — новый массив: окно «кадр целиком» иначе оставалось бы срезом входа.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"from_query: ожидается RGB uint8 HxWx3, получено {image.shape}")
    if target is not None and target.confident:
        return from_bottle(crop_box(image, target.target), max_side=max_side, **view_params)
    # Окна режутся от уже уменьшенного кадра: перед моделью всё равно квадрат 448.
    small = resize_long_side(image, max_side, upscale=False)
    height, width = small.shape[:2]
    views: dict[ViewName, np.ndarray] = {}
    for name, box in (windows or QUERY_WINDOWS).items():
        wx0, wy0, wx1, wy1 = box_pixels(box, width, height)
        views[name] = np.array(small[wy0:wy1, wx0:wx1], dtype=np.uint8, order="C", copy=True)
    return views


def order_views(views: dict[ViewName, np.ndarray]) -> list[ViewName]:
    """Имена видов в каноническом порядке: индекс и запрос обходятся одинаково."""
    known = [name for name in VIEWS if name in views]
    unknown = sorted(set(views) - set(VIEWS))
    if unknown:
        raise ValueError(f"неизвестные виды: {unknown}")
    return known
