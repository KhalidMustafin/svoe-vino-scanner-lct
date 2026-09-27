"""Генератор синтетических «полевых» кадров для замеров OCR (перенос прототипа).

Только CPU: PIL, OpenCV, numpy. Из packshot «Своего Вина» с альфа-каналом собирается
кадр «как с телефона у полки» (по умолчанию 3024x4032):

  цель (slug) + 1–2 соседа (сначала двойники из визуальной группы level_B, затем
  кластер, затем та же винодельня) на процедурном фоне (стеллаж / стол / светлая
  полка), ценник с названием соседа (текстовая ловушка), цилиндрический поворот
  этикетки и «улыбка» строк, затенение, блик-полоса, конденсат, наклон 0–15°,
  перспектива, расфокус и смаз, баланс белого, виньетка, шум, резкость как в ISP
  телефона, JPEG/WebP.

Этикетка занимает 20–30 % площади кадра. Горлышко и дно могут быть срезаны кадром.

Выход (в --out):
  <id>.jpg|.webp           кадры
  synth_manifest.tsv       query_id<TAB>image_path   (формат participant_test.sh)
  synth_gt.tsv             query_id<TAB>slug<TAB>in_catalog
  synth_annotations.jsonl  рамки цели, этикетки и соседей, связи соседей, параметры
  synth_contact.jpg        контактный лист; crops/<id>_label.jpg — этикетка крупно

Ограничения: рамка этикетки — эвристика по цвету, не разметка; соседи — лицевые
packshot без контрэтикеток; фактура бумаги, тиснение и фольга не моделируются;
packshot обычно увеличивается в 1,5–4 раза, поэтому текст читается легче, чем на
полевом кадре. Синтетика — регрессия геометрии, а не оценка абсолютной точности.

    python -m bench.synth.make_field_synth --n 12 --out data/raw/synth \\
        --photo-stats photo_stats.jsonl --photo-map slug_photo_map.csv \\
        --gt-tokens gt_tokens.jsonl --clusters near_dup_clusters.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import get_settings

FRAME_SIZE = (3024, 4032)  # ширина, высота
THUMB_SIZE = (378, 504)


@dataclass(frozen=True)
class Fonts:
    regular: Path | None = Path(r"C:\Windows\Fonts\arial.ttf")
    bold: Path | None = Path(r"C:\Windows\Fonts\arialbd.ttf")


DEFAULT_FONTS = Fonts()


@dataclass(frozen=True)
class Catalog:
    stats: dict[str, dict[str, Any]]  # photo_stats.jsonl: рамка бутылки, альфа
    paths: dict[str, str]  # slug → путь к packshot
    gt: dict[str, dict[str, Any]]  # gt_tokens.jsonl
    members: dict[Any, list[str]]  # кластер level_B → slug


Background = Callable[[random.Random, np.random.Generator, int, int], tuple[np.ndarray, list[int]]]


# ------------------------------------------------------------------ каталог
def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_catalog(photo_stats: Path, photo_map: Path, gt_tokens: Path, clusters: Path) -> Catalog:
    stats = {r["slug"]: r for r in _read_jsonl(photo_stats)}
    with photo_map.open(encoding="utf-8", newline="") as fh:
        paths = {row["slug"]: row["path"] for row in csv.DictReader(fh)}
    gt = {r["slug"]: r for r in _read_jsonl(gt_tokens)}
    nd = json.loads(clusters.read_text(encoding="utf-8"))
    members = {
        cl["cluster_id"]: [m["slug"] for m in cl["members"]] for cl in nd["level_B"]["clusters"]
    }
    return Catalog(stats=stats, paths=paths, gt=gt, members=members)


def usable(st: dict[str, Any], min_h: int) -> bool:
    return (
        "bbox" in st
        and st.get("alpha_kind") == "alpha"
        and st.get("big_components") == 1
        and 0.12 <= st["bottle_aspect"] <= 0.45
        and st["bbox"][3] >= min_h
    )


# ------------------------------------------------------------------ бутылка
def load_bottle(path: str | Path, bbox: Sequence[int]) -> np.ndarray:
    with Image.open(path) as src:
        im = src.convert("RGBA")
    x, y, w, h = bbox
    pad = max(2, int(0.005 * h))
    im = im.crop(
        (max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))
    )
    arr = np.asarray(im).astype(np.float32)
    ys, xs = np.nonzero(arr[:, :, 3] > 8)
    return np.ascontiguousarray(arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])


def resize_rgba(arr: np.ndarray, scale: float) -> np.ndarray:
    h, w = arr.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pm = arr.copy()
    pm[:, :, :3] *= arr[:, :, 3:4] / 255.0  # премультипликация: без неё по контуру ореол фона
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    out = cv2.resize(pm, (nw, nh), interpolation=interp)
    al = np.clip(out[:, :, 3:4], 0, 255) / 255.0
    out[:, :, :3] = np.where(al > 1e-3, out[:, :, :3] / np.maximum(al, 1e-3), 0)
    out[:, :, 3] = np.clip(out[:, :, 3], 0, 255)
    return np.clip(out, 0, 255)


def row_geometry(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Центр и полуширина силуэта по строкам (сглажено) и маска непустых строк."""
    m = alpha > 127
    h, w = m.shape
    any_ = m.any(1)
    left = np.where(any_, m.argmax(1), w // 2).astype(np.float32)
    right = np.where(any_, w - 1 - m[:, ::-1].argmax(1), w // 2).astype(np.float32)
    k = max(3, (h // 150) | 1)
    left = cv2.blur(left.reshape(-1, 1), (1, k)).ravel()
    right = cv2.blur(right.reshape(-1, 1), (1, k)).ravel()
    return (left + right) / 2, np.maximum((right - left) / 2, 1.0), any_


def estimate_label_rows(arr: np.ndarray) -> tuple[float, float, str]:
    """Полоса этикетки: длинная серия строк, цвет центра которых далёк от стекла на плечах."""
    h, w = arr.shape[:2]
    rows = 400
    small = cv2.resize(arr, (max(8, int(w * rows / h)), rows), interpolation=cv2.INTER_AREA)
    c, r, _ = row_geometry(small[:, :, 3])

    def central(y: int) -> np.ndarray:
        x0, x1 = int(c[y] - 0.5 * r[y]), int(c[y] + 0.5 * r[y]) + 1
        return small[y, max(0, x0) : max(x0 + 1, x1), :3]

    shoulders = range(int(0.30 * rows), int(0.42 * rows))
    ref = np.median(np.concatenate([central(y) for y in shoulders]), axis=0)
    dist = np.array(
        [np.median(np.linalg.norm(central(y) - ref, axis=1)) for y in range(rows)], np.float32
    )
    dist = np.convolve(dist, np.ones(5) / 5, mode="same")
    lab = dist > 45
    best, cur, start = (0, 0), 0, 0
    for y in range(int(0.40 * rows), rows):
        if lab[y]:
            start = y if cur == 0 else start
            cur += 1
            if cur > best[1] - best[0]:
                best = (start, y + 1)
        else:
            cur = 0
    if best[1] - best[0] >= 0.10 * rows:
        return best[0] / rows, best[1] / rows, "color_run"
    return 0.55, 0.89, "fallback"


def cylinder_warp(
    arr: np.ndarray,
    phase: float,
    kappa: float,
    shade: float,
    glare: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Поворот цилиндра на phase (рад), «улыбка» строк kappa, затенение к краям, блик-полоса."""
    h, w = arr.shape[:2]
    c, r, any_ = row_geometry(arr[:, :, 3])
    X = np.arange(w, dtype=np.float32)[None, :]
    Y = np.arange(h, dtype=np.float32)[:, None]
    t = (X - c[:, None]) / r[:, None]
    inside = (np.abs(t) <= 1.0) & any_[:, None]
    th = np.arcsin(np.clip(t, -1, 1)).astype(np.float32)
    th2 = np.clip(th + phase, -1.55, 1.55)
    mapx = np.where(inside, c[:, None] + r[:, None] * np.sin(th2), X).astype(np.float32)
    mapy = np.where(inside, Y + kappa * r[:, None] * (1 - np.cos(th)), Y + 0 * X).astype(np.float32)
    out = cv2.remap(arr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[:, :, 3] = arr[:, :, 3]
    cos = np.clip(np.cos(th), 0, 1)
    f = np.where(inside, (1 - shade) + shade * np.sqrt(cos), 1.0).astype(np.float32)
    out[:, :, :3] *= f[:, :, None]
    if glare is not None:
        gpos, gw, gi = glare
        g = (np.exp(-(((np.sin(th) - math.sin(gpos)) / gw) ** 2)) * gi).astype(np.float32)
        g = np.where(inside, g, 0).astype(np.float32)
        out[:, :, :3] += (255 - out[:, :, :3]) * g[:, :, None]
    return out


def droplets(layer: np.ndarray, rng: random.Random, n: int, rmin: float, rmax: float) -> None:
    h, w = layer.shape[:2]
    dark = np.zeros((h, w), np.float32)
    spec = np.zeros((h, w), np.float32)
    for _ in range(n):
        x, y = rng.randrange(w), rng.randrange(h)
        if layer[y, x, 3] < 200:
            continue
        rr = rng.uniform(rmin, rmax)
        cv2.circle(dark, (x, y), max(1, int(rr)), 1.0, -1, lineType=cv2.LINE_AA)
        center = (int(x - 0.35 * rr), int(y - 0.35 * rr))
        cv2.circle(spec, center, max(1, int(0.25 * rr)), 1.0, -1, lineType=cv2.LINE_AA)
    dark = cv2.GaussianBlur(dark, (0, 0), 1.2)
    spec = cv2.GaussianBlur(spec, (0, 0), 0.8)
    layer[:, :, :3] *= (1 - 0.22 * dark)[:, :, None]
    layer[:, :, :3] += (255 - layer[:, :, :3]) * (0.55 * spec)[:, :, None]


# ------------------------------------------------------------------ фоны (на 1/4 разрешения)
def smooth_noise(nrng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    small = nrng.standard_normal((max(2, h // cell), max(2, w // cell))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def bokeh_bottles(
    img: np.ndarray, rng: random.Random, n: int, y0: float, y1: float, dark: bool = True
) -> None:
    h, w = img.shape[:2]
    for _ in range(n):
        bw = int(w * rng.uniform(0.06, 0.12))
        x = rng.randrange(-bw, w)
        top = int(rng.uniform(y0, y1) * h)
        if dark:
            col = [rng.uniform(10, 40)] * 3
        else:
            col = [rng.uniform(40, 160), rng.uniform(40, 140), rng.uniform(20, 90)]
        hl = int(x + bw * rng.uniform(0.2, 0.35))
        hl_col = [min(255, c + rng.uniform(40, 90)) for c in col]
        if h == 0:  # пустая полоса между полками: случайные числа всё равно выбраны
            continue
        cv2.rectangle(img, (x, top), (x + bw, h), col, -1)
        cv2.ellipse(img, (x + bw // 2, top), (bw // 2, bw // 3), 0, 180, 360, col, -1)
        cv2.line(img, (hl, top + bw // 3), (hl, h), hl_col, max(1, bw // 10))


def bg_shelf(
    rng: random.Random, nrng: np.random.Generator, height: int, width: int
) -> tuple[np.ndarray, list[int]]:
    h, w = height // 4, width // 4
    top = rng.uniform(12, 40)
    tint = np.array([1.0, rng.uniform(0.9, 1.05), rng.uniform(0.8, 1.1)], np.float32)
    grad = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (
        (top * (1 - grad) + top * rng.uniform(1.3, 2.3) * grad)
        * tint
        * np.ones((h, w, 3), np.float32)
    )
    bokeh_bottles(img, rng, rng.randint(4, 8), 0.0, 0.3, dark=True)
    for _ in range(rng.randint(2, 4)):  # стойки стеллажа
        x0, pw = rng.randrange(w), int(w * rng.uniform(0.04, 0.1))
        sub = img[:, x0 : x0 + pw]
        if sub.shape[1]:
            sub[:] = rng.uniform(25, 60) + 6 * smooth_noise(nrng, h, sub.shape[1], 6)[:, :, None]
    rails = []
    for _ in range(rng.randint(1, 2)):  # металлические направляющие полки
        yc, th = int(h * rng.uniform(0.80, 0.95)), max(3, int(h * rng.uniform(0.006, 0.012)))
        th = min(th, h - yc)
        prof = (np.sin(np.linspace(0, math.pi, th)) * rng.uniform(120, 200) + 40).astype(np.float32)
        img[yc : yc + th] = prof[:, None, None] * np.ones((1, w, 3), np.float32)
        rails.append(yc * 4)
    for _ in range(rng.randint(3, 8)):  # огни
        center = (rng.randrange(w), rng.randrange(h // 2))
        radius = int(w * rng.uniform(0.01, 0.04))
        color = (rng.uniform(150, 255), rng.uniform(120, 220), rng.uniform(60, 160))
        cv2.circle(img, center, radius, color, -1)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2, 6))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC), rails


def bg_table(
    rng: random.Random, nrng: np.random.Generator, height: int, width: int
) -> tuple[np.ndarray, list[int]]:
    h, w = height // 4, width // 4
    ty = int(h * rng.uniform(0.55, 0.72))
    wall = np.array([rng.uniform(150, 215)] * 3, np.float32) * np.array(
        [1.0, rng.uniform(0.95, 1.0), rng.uniform(0.85, 1.0)]
    )
    img = wall * np.ones((h, w, 3), np.float32)
    img[:ty] *= np.linspace(0.8, 1.0, ty, dtype=np.float32)[:, None, None]
    for _ in range(rng.randint(3, 7)):  # предметы на заднем плане
        x, y = rng.randrange(w), rng.randrange(max(1, ty))
        corner = (x + int(w * rng.uniform(0.05, 0.2)), y + int(h * rng.uniform(0.05, 0.2)))
        color = (rng.uniform(30, 250), rng.uniform(30, 250), rng.uniform(30, 250))
        cv2.rectangle(img, (x, y), corner, color, -1)
    th_ = h - ty
    n = smooth_noise(nrng, th_, w, 1)
    streak = cv2.resize(
        nrng.standard_normal((max(2, th_ // 3), max(2, w // 60))).astype(np.float32),
        (w, th_),
        interpolation=cv2.INTER_CUBIC,
    )
    yy = np.arange(th_, dtype=np.float32)[:, None]
    grain = np.sin(2 * math.pi * yy / rng.uniform(6, 14) + 2.5 * streak) * 0.5 + 0.5
    wood = np.array([rng.uniform(140, 190), rng.uniform(100, 140), rng.uniform(60, 95)], np.float32)
    img[ty:] = wood[None, None, :] * (0.78 + 0.22 * grain[:, :, None]) + 4 * n[:, :, None]
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(3, 8))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC), []


def bg_store(
    rng: random.Random, nrng: np.random.Generator, height: int, width: int
) -> tuple[np.ndarray, list[int]]:
    h, w = height // 4, width // 4
    img = np.array([rng.uniform(170, 225)] * 3, np.float32) * np.ones((h, w, 3), np.float32)
    rails = []
    ys = sorted(rng.uniform(0.2, 0.95) for _ in range(rng.randint(2, 3)))
    prev = 0
    for yv in ys:
        y = int(h * yv)
        bokeh_bottles(img[prev:y], rng, rng.randint(5, 10), 0.1, 0.5, dark=rng.random() < 0.5)
        cv2.rectangle(img, (0, y), (w, y + int(h * 0.02)), (rng.uniform(200, 250),) * 3, -1)
        cv2.rectangle(img, (0, y + int(h * 0.02)), (w, y + int(h * 0.028)), (60, 60, 60), -1)
        rails.append(y * 4)
        prev = y + int(h * 0.028)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(3, 7))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC), rails


BACKGROUNDS: dict[str, Background] = {
    "shelf_dark": bg_shelf,
    "table": bg_table,
    "store_light": bg_store,
}


# ------------------------------------------------------------------ композиция
def paste(
    canvas: np.ndarray,
    layer: np.ndarray,
    x: float,
    y: float,
    ids: np.ndarray | None = None,
    idval: int = 0,
) -> None:
    H, W = canvas.shape[:2]
    h, w = layer.shape[:2]
    x0, y0 = round(x), round(y)
    sx, sy, dx0, dy0 = max(0, -x0), max(0, -y0), max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + w), min(H, y0 + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    sub = layer[sy : sy + dy1 - dy0, sx : sx + dx1 - dx0]
    if canvas.ndim == 3:
        a = sub[:, :, 3:4] / 255.0
        canvas[dy0:dy1, dx0:dx1] = canvas[dy0:dy1, dx0:dx1] * (1 - a) + sub[:, :, :3] * a
        if ids is not None:
            ids[dy0:dy1, dx0:dx1][sub[:, :, 3] > 127] = idval
    else:  # маска uint8
        canvas[dy0:dy1, dx0:dx1] = np.maximum(canvas[dy0:dy1, dx0:dx1], sub)


def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path is not None and path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def price_tag(
    rng: random.Random, name: str, frame_size: tuple[int, int], fonts: Fonts
) -> tuple[np.ndarray, str]:
    fw, fh = frame_size
    tw, th = int(fw * rng.uniform(0.24, 0.34)), int(fh * rng.uniform(0.065, 0.09))
    styles = [
        ((250, 215, 40), (20, 20, 20)),
        ((215, 30, 35), (255, 255, 255)),
        ((245, 245, 240), (30, 30, 30)),
    ]
    style = rng.choice(styles)
    img = Image.new("RGBA", (tw, th), style[0] + (255,))
    d = ImageDraw.Draw(img)
    f_name = _font(fonts.regular, max(12, int(th * 0.17)))
    f_price = _font(fonts.bold, max(16, int(th * 0.42)))
    pad = int(th * 0.08)
    short = name if len(name) <= 34 else name[:33] + "…"
    d.text((pad, pad), short, font=f_name, fill=style[1])
    price = rng.randint(4, 45) * 100 - 10
    label = f"{price // 1000} {price % 1000:03d} ₽" if price >= 1000 else f"{price} ₽"
    d.text((pad, int(th * 0.40)), label, font=f_price, fill=style[1])
    return np.asarray(img).astype(np.float32), short


def bbox_of(mask: np.ndarray) -> list[int] | None:
    rows, cols = np.any(mask, 1), np.any(mask, 0)
    if not rows.any():
        return None
    y0, y1 = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), len(cols) - np.argmax(cols[::-1])
    return [int(x0), int(y0), int(x1), int(y1)]


def pick_neighbors(
    target: str,
    rng: random.Random,
    gt: dict[str, dict[str, Any]],
    members: dict[Any, list[str]],
    ok: set[str],
) -> list[tuple[str, str]]:
    """Соседи по приоритету: визуальный двойник → кластер → та же винодельня → любой."""
    rec = gt[target]
    pools = [
        ("visual_twin", [s for s in rec["visual_mates"] if s in ok]),
        ("cluster_B", [s for s in members.get(rec["cluster_B"], []) if s != target and s in ok]),
        (
            "same_winery",
            [s for s, g in gt.items() if g["winery"] == rec["winery"] and s != target and s in ok],
        ),
        ("random", [s for s in sorted(ok) if s != target]),  # sorted: порядок set не стабилен
    ]
    k = 2 if rng.random() < 0.6 else 1
    out: list[tuple[str, str]] = []
    used = {target}
    for rel, pool in pools:
        pool = [s for s in pool if s not in used]
        rng.shuffle(pool)
        for s in pool[: k - len(out)]:
            out.append((s, rel))
            used.add(s)
        if len(out) >= k:
            break
    return out


def make_frame(
    qid: str,
    target: str,
    seed: int,
    catalog: Catalog,
    ok: set[str],
    out_dir: Path,
    *,
    frame_size: tuple[int, int] = FRAME_SIZE,
    fonts: Fonts = DEFAULT_FONTS,
) -> tuple[dict[str, Any], Image.Image]:
    """Один кадр и его разметка. Порядок вызовов rng сохранён: тот же сид — тот же кадр."""
    t0 = time.time()
    gt = catalog.gt
    fw, fh = frame_size
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    cw, ch = int(fw * 1.35), int(fh * 1.35)
    fx0, fy0 = (cw - fw) / 2, (ch - fh) / 2
    scene = rng.choices(["shelf_dark", "table", "store_light"], weights=[0.5, 0.25, 0.25])[0]
    canvas, _rails = BACKGROUNDS[scene](rng, nrng, ch, cw)
    ids = np.zeros((ch, cw), np.uint8)
    full_t = np.zeros((ch, cw), np.uint8)
    label_m = np.zeros((ch, cw), np.uint8)

    # --- цель: масштаб так, чтобы этикетка заняла 20–30 % кадра
    arr = load_bottle(catalog.paths[target], catalog.stats[target]["bbox"])
    l0, l1, lmethod = estimate_label_rows(arr)
    h0 = arr.shape[0]
    _, r0, _ = row_geometry(arr[:, :, 3])
    lrow = int((l0 + l1) / 2 * h0)
    body_w0, label_h0 = 2 * float(r0[lrow]), (l1 - l0) * h0
    frac = rng.uniform(0.20, 0.30)
    scale = math.sqrt(frac * fw * fh / (body_w0 * label_h0))
    scale = min(scale, 0.8 * fh / label_h0, 0.72 * fw / body_w0)
    lay = resize_rgba(arr, scale)
    del arr
    hs = lay.shape[0]
    phase_t, kappa = rng.uniform(-0.18, 0.18), rng.uniform(-0.10, 0.10)
    glare = (
        (
            rng.uniform(-0.75, -0.25) * rng.choice([-1, 1]),
            rng.uniform(0.04, 0.12),
            rng.uniform(0.15, 0.55),
        )
        if rng.random() < 0.8
        else None
    )
    shade = rng.uniform(0.2, 0.45)
    lay = cylinder_warp(lay, phase_t, kappa, shade=shade, glare=glare)
    cond = (scene == "table" and rng.random() < 0.6) or rng.random() < 0.1
    if cond:
        droplets(lay, rng, int(lay.shape[0] * lay.shape[1] / 900), 2, 9)
    ct, rt, _ = row_geometry(lay[:, :, 3])
    lrow_s = int((l0 + l1) / 2 * hs)
    lx, ly = fw * (0.5 + rng.uniform(-0.10, 0.10)), fh * rng.uniform(0.46, 0.60)
    px, py = fx0 + lx - ct[lrow_s], fy0 + ly - lrow_s
    bottom_t, cx_t, r_t = py + hs, px + ct[lrow_s], rt[lrow_s]

    # --- соседи (рисуются раньше цели: они позади)
    neigh = pick_neighbors(target, rng, gt, catalog.members, ok)
    sides = ["left", "right"] if len(neigh) == 2 else [rng.choice(["left", "right"])]
    nrecs: list[dict[str, Any]] = []
    for (ns, rel), side in zip(neigh, sides, strict=False):
        a2 = load_bottle(catalog.paths[ns], catalog.stats[ns]["bbox"])
        depth = rng.uniform(0.55, 0.8) if scene == "table" else rng.uniform(0.93, 1.04)
        l2 = resize_rgba(a2, hs * depth / a2.shape[0])
        del a2
        phase2, kappa2, shade2 = (
            rng.uniform(-0.9, 0.9),
            rng.uniform(-0.08, 0.08),
            rng.uniform(0.3, 0.55),
        )
        glare2 = (
            (rng.uniform(-0.8, 0.8), rng.uniform(0.05, 0.12), rng.uniform(0.1, 0.4))
            if rng.random() < 0.6
            else None
        )
        l2 = cylinder_warp(l2, phase2, kappa2, shade=shade2, glare=glare2)
        l2[:, :, :3] *= rng.uniform(0.65, 0.95)
        sig = rng.uniform(4, 10) if scene == "table" else rng.uniform(0.5, 2.5)
        l2 = cv2.GaussianBlur(l2, (0, 0), sig)
        c2, r2, _ = row_geometry(l2[:, :, 3])
        row2 = int(0.72 * l2.shape[0])
        gap = rng.uniform(1.1, 1.8) if scene == "table" else rng.uniform(1.0, 1.3)
        cx2 = cx_t + (-1 if side == "left" else 1) * (r_t + r2[row2]) * gap
        if scene == "table":
            bottom2 = bottom_t - rng.uniform(0.05, 0.15) * hs
        else:
            bottom2 = bottom_t - rng.uniform(-0.04, 0.04) * hs
        idv = 2 + len(nrecs)
        paste(canvas, l2, cx2 - c2[row2], bottom2 - l2.shape[0], ids, idv)
        nrecs.append(
            {
                "slug": ns,
                "relation": rel,
                "side": side,
                "blur_sigma": round(sig, 2),
                "id": idv,
                "name": gt[ns]["name"],
            }
        )
        del l2

    paste(canvas, lay, px, py, ids, 1)
    paste(full_t, (lay[:, :, 3] > 127).astype(np.uint8) * 255, px, py)
    lm = np.zeros(lay.shape[:2], np.uint8)
    lm[int(l0 * hs) : int(l1 * hs)] = 255
    lm[lay[:, :, 3] < 128] = 0
    paste(label_m, lm, px, py)
    del lay, lm

    # --- ценник с названием соседа (текстовая ловушка), поверх бутылок
    distractors = []
    if scene != "table" and rng.random() < 0.75:
        if nrecs and rng.random() < 0.7:
            name = nrecs[0]["name"]
        else:
            name = gt[rng.choice(sorted(ok))]["name"]
        tag, short = price_tag(rng, name, frame_size, fonts)
        if rng.random() < 0.5:
            tx = fx0 + fw * rng.uniform(-0.08, 0.08)
        else:
            tx = fx0 + fw * rng.uniform(0.66, 0.8)
        ty = fy0 + fh * rng.uniform(0.80, 0.92)
        paste(canvas, tag, tx, ty, ids, 9)
        distractors.append({"type": "price_tag", "text": short})

    # --- камера: наклон 0–15°, перспектива, кадрирование
    ang = rng.uniform(0, 15) * rng.choice([-1, 1])
    jit = rng.uniform(0.0, 0.06)
    cxc, cyc = cw / 2, ch / 2
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    corners = []
    for ux, uy in [(-fw / 2, -fh / 2), (fw / 2, -fh / 2), (fw / 2, fh / 2), (-fw / 2, fh / 2)]:
        x = cxc + ux * ca - uy * sa + rng.uniform(-jit, jit) * fw
        y = cyc + ux * sa + uy * ca + rng.uniform(-jit, jit) * fh
        corners.append((x, y))
    dst = np.float32([(0, 0), (fw, 0), (fw, fh), (0, fh)])
    M = cv2.getPerspectiveTransform(np.float32(corners), dst)
    frame = cv2.warpPerspective(
        canvas, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    del canvas
    nearest = {"flags": cv2.INTER_NEAREST, "borderMode": cv2.BORDER_CONSTANT}
    ids_w = cv2.warpPerspective(ids, M, (fw, fh), **nearest)
    full_w = cv2.warpPerspective(full_t, M, (fw, fh), **nearest)
    lab_w = cv2.warpPerspective(label_m, M, (fw, fh), **nearest)

    # --- оптика и сенсор
    focus = rng.uniform(0.6, 3.0)
    frame = cv2.GaussianBlur(frame, (0, 0), focus)
    motion = None
    if rng.random() < 0.3:
        L = rng.randint(5, 17)
        k = np.zeros((L, L), np.float32)
        a = math.radians(rng.uniform(0, 180))
        p1 = (int(L / 2 - L / 2 * math.cos(a)), int(L / 2 - L / 2 * math.sin(a)))
        p2 = (int(L / 2 + L / 2 * math.cos(a)), int(L / 2 + L / 2 * math.sin(a)))
        cv2.line(k, p1, p2, 1.0, 1)
        frame = cv2.filter2D(frame, -1, k / k.sum())
        motion = L
    wb = rng.choice(["warm", "cool", "neutral"])
    base_gains = {"warm": (1.10, 1.0, 0.84), "cool": (0.88, 1.0, 1.10), "neutral": (1.0, 1.0, 1.0)}
    gains = np.array([g * rng.uniform(0.97, 1.03) for g in base_gains[wb]], np.float32)
    gamma = rng.uniform(0.85, 1.2)
    frame = 255.0 * np.power(np.clip(frame, 0, 255) / 255.0, gamma) * gains[None, None, :]
    yy = (np.arange(fh, dtype=np.float32)[:, None] / fh - 0.5) * 2
    xx = (np.arange(fw, dtype=np.float32)[None, :] / fw - 0.5) * 2
    vig = rng.uniform(0.0, 0.35)
    frame *= (1 - vig * (xx**2 + yy**2) / 2)[:, :, None]
    sigma = rng.uniform(1.5, 5.0)
    frame += nrng.normal(0, sigma, (fh, fw, 1)).astype(np.float32)
    frame += nrng.normal(0, sigma * 0.5, (fh, fw, 3)).astype(np.float32)
    sharpen = rng.random() < 0.5
    if sharpen:
        frame = frame + 0.6 * (frame - cv2.GaussianBlur(frame, (0, 0), 2.0))
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    im = Image.fromarray(frame)  # кадр уже RGB: packshot читался как RGB
    del frame
    fmt = rng.choice(["jpg", "webp"])
    q = rng.randint(72, 92) if fmt == "jpg" else rng.randint(70, 90)
    fname = f"{random.Random(seed * 7 + 1).getrandbits(32):08x}.{fmt}"
    buf = io.BytesIO()
    im.save(buf, format="JPEG" if fmt == "jpg" else "WEBP", quality=q)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / fname).write_bytes(buf.getvalue())

    # --- разметка
    vis_t = ids_w == 1
    lab = lab_w > 0
    lb = bbox_of(lab)
    full_box = bbox_of(full_w > 0)
    ann = {
        "query_id": qid,
        "image_path": fname,
        "target_slug": target,
        "in_catalog": True,
        "scene": scene,
        "target_name": gt[target]["name"],
        "target_winery": gt[target]["winery"],
        "target_bbox_visible": bbox_of(vis_t),
        "target_bbox_full": full_box,
        "label_bbox_approx": lb,
        "label_rows_method": lmethod,
        "label_area_share": round(float(lab.mean()), 3),
        "label_visible_share": round(float((lab & vis_t).sum() / max(1, lab.sum())), 3),
        "neck_cut": bool(full_box and full_box[1] <= 0),
        "neighbors": [
            dict(
                n,
                bbox_visible=bbox_of(ids_w == n["id"]),
                frame_share=round(float((ids_w == n["id"]).mean()), 3),
            )
            for n in nrecs
        ],
        "distractors": distractors,
        "params": {
            "seed": seed,
            "frame_size": [fw, fh],
            "label_frac_target": round(frac, 3),
            "source_scale": round(scale, 3),
            "phase_rad": round(phase_t, 3),
            "kappa": round(kappa, 3),
            "glare": glare,
            "condensation": cond,
            "tilt_deg": round(ang, 2),
            "persp_jitter": round(jit, 3),
            "focus_sigma": round(focus, 2),
            "motion_len": motion,
            "wb": wb,
            "gamma": round(gamma, 3),
            "vignette": round(vig, 3),
            "noise_sigma": round(sigma, 2),
            "sharpen": sharpen,
            "format": fmt,
            "quality": q,
            "bytes": len(buf.getvalue()),
        },
        "gen_ms": int((time.time() - t0) * 1000),
    }
    if lb:  # этикетка крупно для глазной проверки
        x0, y0, x1, y1 = lb
        pad = 60
        crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(fw, x1 + pad), min(fh, y1 + pad)))
        crop.thumbnail((1400, 1400))
        (out_dir / "crops").mkdir(exist_ok=True)
        crop.save(out_dir / "crops" / f"{fname.split('.')[0]}_label.jpg", quality=90)
    thumb = im.resize(THUMB_SIZE, Image.BILINEAR)
    return ann, thumb


# ------------------------------------------------------------------ набор
def pick_targets(
    gt: dict[str, dict[str, Any]],
    ok: set[str],
    *,
    n: int,
    seed: int,
    explicit: Sequence[str] = (),
) -> list[str]:
    """Цели: сначала до 7 двойников из разных виноделен, затем прочие винодельни."""
    targets = [t for t in explicit if t]
    if targets:
        return targets[:n]
    rng = random.Random(seed)
    twins = sorted(s for s in ok if any(m in ok for m in gt[s]["visual_mates"]))
    rng.shuffle(twins)
    seen: set[str] = set()
    for s in twins:
        if gt[s]["winery"] not in seen:
            targets.append(s)
            seen.add(gt[s]["winery"])
        if len(targets) >= min(7, n):
            break
    rest = sorted(ok - set(targets))
    rng.shuffle(rest)
    for s in rest:
        if len(targets) >= n:
            break
        if gt[s]["winery"] not in seen:
            targets.append(s)
            seen.add(gt[s]["winery"])
    return targets[:n]


def write_outputs(anns: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Манифест (формат participant_test.sh), эталон и разметка."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "synth_manifest.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("query_id\timage_path\n")
        for a in anns:
            f.write(f"{a['query_id']}\t{a['image_path']}\n")
    with (out_dir / "synth_gt.tsv").open("w", encoding="utf-8", newline="\n") as f:
        f.write("query_id\tslug\tin_catalog\n")
        for a in anns:
            f.write(f"{a['query_id']}\t{a['target_slug']}\t1\n")
    with (out_dir / "synth_annotations.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for a in anns:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")


def contact_sheet(thumbs: Sequence[Image.Image], qids: Sequence[str], path: Path) -> None:
    cols = 4
    tw, th = THUMB_SIZE
    rows = max(1, math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * tw, rows * th), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (thumb, qid) in enumerate(zip(thumbs, qids, strict=True)):
        sheet.paste(thumb, ((i % cols) * tw, (i // cols) * th))
        d.text(((i % cols) * tw + 8, (i // cols) * th + 8), qid, fill=(255, 255, 0))
    sheet.save(path, quality=88)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    catalog_dir = settings.data_dir / "catalog"
    parser = argparse.ArgumentParser(
        prog="python -m bench.synth.make_field_synth", description="Синтетические полевые кадры."
    )
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--out", type=Path, default=settings.data_dir / "raw" / "synth")
    parser.add_argument("--min-h", type=int, default=1200)
    parser.add_argument("--targets", default="", help="slug через запятую; иначе выбор по сиду")
    parser.add_argument("--photo-stats", type=Path, default=catalog_dir / "photo_stats.jsonl")
    parser.add_argument("--photo-map", type=Path, default=catalog_dir / "slug_photo_map.csv")
    parser.add_argument(
        "--gt-tokens", type=Path, default=settings.data_dir / "gt" / "gt_tokens.jsonl"
    )
    parser.add_argument("--clusters", type=Path, default=catalog_dir / "near_dup_clusters.json")
    parser.add_argument("--font-regular", type=Path, default=Fonts.regular)
    parser.add_argument("--font-bold", type=Path, default=Fonts.bold)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.photo_stats, args.photo_map, args.gt_tokens, args.clusters):
        if not path.is_file():
            print(f"нет входного файла: {path}", file=sys.stderr)
            return 2
    Image.MAX_IMAGE_PIXELS = None  # packshot бывают до 9 506 px в высоту
    catalog = load_catalog(args.photo_stats, args.photo_map, args.gt_tokens, args.clusters)
    ok = {s for s, st in catalog.stats.items() if usable(st, args.min_h) and s in catalog.gt}
    explicit = [t for t in args.targets.split(",") if t]
    targets = pick_targets(catalog.gt, ok, n=args.n, seed=args.seed, explicit=explicit)
    fonts = Fonts(regular=args.font_regular, bold=args.font_bold)
    anns, thumbs = [], []
    for i, target in enumerate(targets):
        qid = f"s-{i + 1:06d}"
        ann, thumb = make_frame(
            qid, target, args.seed * 1000 + i, catalog, ok, args.out, fonts=fonts
        )
        anns.append(ann)
        thumbs.append(thumb)
        print(
            qid,
            ann["image_path"],
            target,
            ann["scene"],
            ann["label_area_share"],
            ann["gen_ms"],
            "ms",
            flush=True,
        )
    write_outputs(anns, args.out)
    contact_sheet(thumbs, [a["query_id"] for a in anns], args.out / "synth_contact.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
