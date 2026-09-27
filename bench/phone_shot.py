"""Кадр «как с телефона у полки»: перспектива, поворот, блик, расфокус, фон, сжатие.

Запросы честных пар — студийные снимки Роскачества: бутылка в центре, ровный свет, чистый
фон. У полки так не снимают. Порча приводит студийный кадр к тому, что реально приходит в
сканер, и показывает, сколько точности стоит каждое отличие: лишний фон вокруг бутылки,
наклон телефона, тёмный зал, блик лампы, промах автофокуса и пережатие мессенджера.

Функция перенесена из «Лозы» (`Code/scripts/eval_scanner_pairs.py: phone_shot`) без
изменения порядка и параметров искажений: цифры нового стенда сравнимы со старыми. Все
случайные величины берутся из `random.Random(seed)`, поэтому кадр повторяется по сиду.

Единственное отступление — фоновый шум. `Image.effect_noise` берёт числа у
неинициализированного `rand()` из C: два вызова подряд дают разный шум, и повторить кадр
по сиду невозможно. Здесь шум того же распределения (128 + sigma·N(0,1)) считает numpy по
тому же сиду; поток `random.Random` при этом не трогается, и все остальные величины
совпадают с «Лозой» до бита.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

#: Цвет полей, которые появляются после поворота и перспективы.
FILL = (128, 128, 128)

#: Доля шума в фоне вокруг бутылки: полка и соседи — не ровная заливка.
NOISE_SHARE = 0.3


def perspective_coeffs(
    src: list[tuple[float, float]], dst: list[tuple[float, float]]
) -> list[float]:
    """Восемь коэффициентов `Image.PERSPECTIVE`, переводящих `dst` в `src`.

    Порядок аргументов — как у `Image.transform`: коэффициенты отображают точки кадра-цели
    обратно в кадр-источник, поэтому «куда тянем» задаёт `dst`, а система решается по `src`.
    """
    matrix = []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.array(matrix, dtype=float)
    b = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(a, b).tolist()


def noise_image(size: tuple[int, int], sigma: float, seed: int) -> Image.Image:
    """Гауссов шум вокруг 128 — замена `Image.effect_noise`, повторяемая по сиду.

    Распределение то же; значения вне 0..255 обрезаются, а не заворачиваются, как при
    приведении к `UINT8` в C.
    """
    width, height = size
    values = np.random.default_rng(seed).normal(128.0, sigma, size=(height, width))
    return Image.fromarray(np.clip(values + 0.5, 0, 255).astype(np.uint8), mode="L")


def phone_shot(image: Image.Image, seed: int) -> Image.Image:
    """Кадр «как с телефона у полки»: всё, что портит снимок в магазине."""
    rng = random.Random(seed)
    w, h = image.size
    # Лишний фон вокруг бутылки: полка, соседние бутылки — серый шум.
    pad = int(max(w, h) * rng.uniform(0.05, 0.35))
    canvas = Image.new(
        "RGB", (w + 2 * pad, h + 2 * pad), tuple(rng.randint(60, 200) for _ in range(3))
    )
    noise = noise_image(canvas.size, rng.uniform(20, 60), seed).convert("RGB")
    canvas = Image.blend(canvas, noise, NOISE_SHARE)
    canvas.paste(image, (pad, pad))
    image = canvas
    # Поворот и перспектива: телефон редко держат ровно.
    image = image.rotate(rng.uniform(-12, 12), resample=Image.BICUBIC, expand=True, fillcolor=FILL)
    w, h = image.size
    dx, dy = w * rng.uniform(0, 0.12), h * rng.uniform(0, 0.08)
    coeffs = perspective_coeffs(
        [(0, 0), (w, 0), (w, h), (0, h)],
        [
            (rng.uniform(0, dx), rng.uniform(0, dy)),
            (w - rng.uniform(0, dx), rng.uniform(0, dy)),
            (w - rng.uniform(0, dx), h - rng.uniform(0, dy)),
            (rng.uniform(0, dx), h - rng.uniform(0, dy)),
        ],
    )
    image = image.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=FILL)
    # Кадрирование: этикетка не всегда в центре и не всегда целиком.
    cw, ch = int(w * rng.uniform(0.75, 1.0)), int(h * rng.uniform(0.7, 1.0))
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    image = image.crop((x0, y0, x0 + cw, y0 + ch))
    # Свет: тёмный зал, блик лампы, расфокус.
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.55, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.2))
    if rng.random() < 0.5:
        glare = Image.new("RGB", image.size, (255, 255, 240))
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        gx, gy = rng.randint(0, image.size[0]), rng.randint(0, image.size[1])
        r = int(min(image.size) * rng.uniform(0.08, 0.2))
        draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=int(rng.uniform(90, 180)))
        mask = mask.filter(ImageFilter.GaussianBlur(r / 2))
        image = Image.composite(glare, image, mask)
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.8)))
    # Сжатие мессенджера.
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(45, 85))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def phone_shot_rgb(image: np.ndarray, seed: int) -> np.ndarray:
    """То же для массива RGB uint8: между слоями кадр живёт массивом, а не картинкой PIL."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"phone_shot_rgb: ожидается RGB uint8 HxWx3, получено {image.shape}")
    spoiled = phone_shot(Image.fromarray(image), seed)
    return np.ascontiguousarray(np.array(spoiled, dtype=np.uint8))
