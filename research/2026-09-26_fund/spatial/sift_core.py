"""Пространственная проверка кандидатов (классическое распознавание экземпляра): SIFT + RANSAC.

Идея: SigLIP сравнивает кадр и эталон целиком и путает соседние вина винодельни (одна
этикетка, другой сорт). Локальные признаки SIFT на этикетке + геометрия (гомография,
RANSAC) проверяют, что *те же самые* буквы и рисунок стоят *в том же взаимном положении*:
число согласованных точек (inliers) — довод «это та самая этикетка», которого у эмбеддинга нет.

Параметры выбраны заранее из учебных значений, а не подбором по пулу:
- SIFT OpenCV по умолчанию (contrastThreshold 0.04, edgeThreshold 10, sigma 1.6), RootSIFT
  (Arandjelović & Zisserman 2012: L1-норма, корень) — сравнение по ядру Хеллингера;
- тест отношения Лоу 0.8 (Lowe 2004), направление «кадр → эталон»: у полки повторяются
  одинаковые бутылки, и в обратном направлении тест отношения их бы выбросил;
- один к одному по точке эталона (лучшая пара), RANSAC-гомография с порогом 8 px на кадре
  с длинной стороной ~1000 px (≈0,8 %), MAGSAC++ OpenCV; гомография без вырождения
  (выпуклый образ рамки inliers эталона без зеркала, площадь в разумных пределах), иначе
  inliers = 0.

Эталон — ИСХОДНОЕ фото каталога организатора (путь из `slug_photo_map.csv` снимка
`frozen_data`, у 9 правленых эталонов — копия `frozen_data/catalog/packshots_fixed`):
обрезка по альфе, заливка серым (как у индекса), ширина бутылки не больше 520 px, высота
не больше 1600 px (без увеличения). Кадр запроса — `decode_on_backgrounds` сервиса:
- `full` — кадр целиком, длинная сторона 1024;
- `label` — окно `QUERY_WINDOWS["label"]` сервиса (x 0.15–0.85, y 0.30–0.95) из кадра с
  длинной стороной 1600 — центральная этикетка крупнее.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.features.views import BACKGROUND, QUERY_WINDOWS, alpha_bounds, flatten_alpha
from app.normalize.decode import decode_on_backgrounds
from app.reading.crops import box_pixels

# ------------------------------------------------------------------ параметры (заданы заранее)
REF_MAX_W = 520
REF_MAX_H = 1600
REF_NFEAT = 1500
Q_FULL_SIDE = 1024
Q_LABEL_SIDE = 1600
Q_NFEAT = 2000
RATIO = 0.8
RANSAC_PX = 8.0
MIN_MATCHES = 6  # меньше — гомографию не ищем, inliers = 0
LABEL_Y0, LABEL_Y1 = 0.30, 0.85  # полоса этикетки эталона (как `views.LABEL_TOP/BOTTOM`)

QUERY_VIEWS = ("full", "label")


def _sift(nfeatures: int) -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=nfeatures)


def root_sift(desc: np.ndarray) -> np.ndarray:
    """SIFT → RootSIFT (единичная L2-норма): скалярное произведение = ядро Хеллингера."""
    d = np.asarray(desc, dtype=np.float32)
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d)


def _resize(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ эталон
def load_reference_gray(path: Path) -> np.ndarray:
    """Исходное фото каталога → серый uint8: обрезка по альфе, заливка серым, как у индекса."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P") or "transparency" in im.info:
            arr = np.asarray(im.convert("RGBA"))
        else:
            arr = np.asarray(im.convert("RGB"))
    x0, y0, x1, y1 = alpha_bounds(arr)
    arr = np.ascontiguousarray(arr[y0:y1, x0:x1])
    h, w = arr.shape[:2]
    scale = min(1.0, REF_MAX_W / w, REF_MAX_H / h)
    if scale < 1.0:
        arr = cv2.resize(arr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    rgb = flatten_alpha(arr, BACKGROUND)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


@dataclass
class RefFeatures:
    kp: np.ndarray  # (n, 4) float32: x, y, size, angle — в пикселях уменьшенного эталона
    desc: np.ndarray  # (n, 128) uint8 — сырой SIFT OpenCV (целые 0..255)
    w: int
    h: int


def extract_reference(path: Path) -> RefFeatures:
    gray = load_reference_gray(path)
    kps, desc = _sift(REF_NFEAT).detectAndCompute(gray, None)
    if desc is None or not kps:
        return RefFeatures(np.zeros((0, 4), np.float32), np.zeros((0, 128), np.uint8), gray.shape[1], gray.shape[0])
    kp = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kps], dtype=np.float32)
    return RefFeatures(kp, np.clip(np.rint(desc), 0, 255).astype(np.uint8), gray.shape[1], gray.shape[0])


def photo_map(frozen: Path) -> dict[str, list[Path]]:
    """slug → исходные фото эталона. Правленые эталоны — из копии `frozen_data`, не общего data."""
    fixed = frozen / "catalog" / "packshots_fixed"
    out: dict[str, list[Path]] = {}
    with (frozen / "catalog" / "slug_photo_map.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = Path(row["path"])
            if "packshots_fixed" in p.parts:
                p = fixed / p.name
            out.setdefault(row["slug"], []).append(p)
    for alt in fixed.glob("*__alt.*"):
        slug = alt.stem.split("__")[0]
        if slug in out:
            out[slug].append(alt)
    return out


class RefStore:
    """Признаки всех эталонов: сплошные массивы + смещения; `desc` отображается с диска."""

    def __init__(self, folder: Path, *, mmap: bool = True) -> None:
        meta = np.load(folder / "refs_meta.npz", allow_pickle=False)
        self.slugs = [str(s) for s in meta["slugs"]]  # по эталону (у slug может быть два)
        self.offsets = meta["offsets"].astype(np.int64)
        self.wh = meta["wh"].astype(np.int32)
        self.kp = np.load(folder / "refs_kp.npy", mmap_mode="r" if mmap else None)
        self.desc = np.load(folder / "refs_desc.npy", mmap_mode="r" if mmap else None)
        self.by_slug: dict[str, list[int]] = {}
        for i, s in enumerate(self.slugs):
            self.by_slug.setdefault(s, []).append(i)
        self._root: dict[int, np.ndarray] = {}

    def ref(self, i: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        d = self._root.get(i)
        if d is None:
            d = root_sift(self.desc[a:b])
            if len(self._root) > 4000:
                self._root.clear()
            self._root[i] = d
        return np.asarray(self.kp[a:b]), d, int(self.wh[i, 0]), int(self.wh[i, 1])


# ------------------------------------------------------------------ запрос
@dataclass
class QueryView:
    kp: np.ndarray  # (n, 2) float32 — x, y в пикселях вида
    desc: np.ndarray  # (n, 128) float32 RootSIFT
    w: int
    h: int


def query_gray(data: bytes) -> np.ndarray:
    rgb = decode_on_backgrounds(data, [BACKGROUND])[0]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def query_views(gray: np.ndarray) -> dict[str, np.ndarray]:
    h, w = gray.shape[:2]
    full = _resize(gray, Q_FULL_SIDE / max(h, w))
    big = _resize(gray, Q_LABEL_SIDE / max(h, w))
    bh, bw = big.shape[:2]
    x0, y0, x1, y1 = box_pixels(QUERY_WINDOWS["label"], bw, bh)
    return {"full": full, "label": np.ascontiguousarray(big[y0:y1, x0:x1])}


def extract_query(gray: np.ndarray, views: Sequence[str] = QUERY_VIEWS) -> dict[str, QueryView]:
    out: dict[str, QueryView] = {}
    sift = _sift(Q_NFEAT)
    for name, img in query_views(gray).items():
        if name not in views:
            continue
        kps, desc = sift.detectAndCompute(img, None)
        if desc is None or not kps:
            out[name] = QueryView(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), img.shape[1], img.shape[0])
            continue
        kp = np.array([k.pt for k in kps], dtype=np.float32)
        out[name] = QueryView(kp, root_sift(desc), img.shape[1], img.shape[0])
    return out


# ------------------------------------------------------------------ сопоставление
@dataclass
class Match:
    good: int = 0  # пар после теста отношения и «один к одному»
    inliers: int = 0  # согласованных с гомографией (0, если гомография вырождена)
    raw_inliers: int = 0  # до проверки вырождения
    area_ref: float = 0.0  # площадь выпуклой оболочки inliers на эталоне / площадь эталона
    area_label: float = 0.0  # доля полосы этикетки эталона, накрытая оболочкой inliers
    area_q: float = 0.0  # площадь оболочки inliers на кадре / площадь вида
    valid: bool = False
    ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32))))


def _homography_ok(H: np.ndarray, rs: np.ndarray, qw: int, qh: int) -> bool:
    """Гомография не вырождена там, где она измерена: образ рамки inliers эталона — выпуклый
    четырёхугольник без зеркала, площадью от 0,01 % до 4 площадей вида кадра.

    Проверяется рамка точек-inliers, а не весь эталон: гомография, подогнанная по этикетке
    на цилиндре бутылки, законно «разлетается» за пределами этикетки (горлышко, дно), и
    проверка всего эталона отбрасывала верные совпадения (правка 26.09 — 15 кадров v2 с
    ≥ 15 согласованными точками у верной карточки получали 0).
    """
    if H is None or not np.all(np.isfinite(H)) or len(rs) < 4:
        return False
    x0, y0 = rs.min(axis=0)
    x1, y1 = rs.max(axis=0)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32).reshape(-1, 1, 2)
    try:
        proj = cv2.perspectiveTransform(box, H).reshape(-1, 2)
    except cv2.error:
        return False
    if not np.all(np.isfinite(proj)):
        return False
    if not cv2.isContourConvex(proj.astype(np.float32)):
        return False
    area = abs(cv2.contourArea(proj.astype(np.float32)))
    if not (1e-4 * qw * qh <= area <= 4.0 * qw * qh):
        return False
    a = proj[1] - proj[0]
    b = proj[3] - proj[0]
    return float(a[0] * b[1] - a[1] * b[0]) > 0


def match(qv: QueryView, rkp: np.ndarray, rdesc: np.ndarray, rw: int, rh: int) -> Match:
    t0 = time.perf_counter()
    m = Match()
    if len(qv.desc) < 2 or len(rdesc) < 2:
        m.ms = (time.perf_counter() - t0) * 1e3
        return m
    sims = qv.desc @ rdesc.T  # (nq, nr); d² = 2 − 2·sim
    rows = np.arange(len(sims))
    best = sims.argmax(axis=1)
    sb = sims[rows, best].copy()
    sims[rows, best] = -2.0
    ss = sims.max(axis=1)
    d1 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sb))
    d2 = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * ss))
    keep = d1 < RATIO * d2
    qi = np.flatnonzero(keep)
    ri = best[keep]
    if len(qi):
        # один к одному по точке эталона: лучшая пара
        order = np.argsort(-sb[qi], kind="stable")
        _, first_idx = np.unique(ri[order], return_index=True)
        sel = order[first_idx]
        qi, ri = qi[sel], ri[sel]
    m.good = len(qi)
    if m.good >= MIN_MATCHES:
        src = rkp[ri, :2].astype(np.float32)
        dst = qv.kp[qi].astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=2000, confidence=0.999)
        if H is not None and mask is not None:
            inl = mask.ravel().astype(bool)
            m.raw_inliers = int(inl.sum())
            m.valid = _homography_ok(H, src[inl], qv.w, qv.h)
            if m.valid and m.raw_inliers >= 4:
                m.inliers = m.raw_inliers
                rs, qs = src[inl], dst[inl]
                m.area_ref = _hull_area(rs) / float(rw * rh)
                band = (rs[:, 1] >= LABEL_Y0 * rh) & (rs[:, 1] <= LABEL_Y1 * rh)
                m.area_label = _hull_area(rs[band]) / float(rw * rh * (LABEL_Y1 - LABEL_Y0))
                m.area_q = _hull_area(qs) / float(qv.w * qv.h)
    m.ms = (time.perf_counter() - t0) * 1e3
    return m


def match_slug(qviews: dict[str, QueryView], store: RefStore, slug: str) -> dict[str, Match]:
    """Лучшее сопоставление по эталонам slug для каждого вида кадра (по числу inliers)."""
    out: dict[str, Match] = {}
    for i in store.by_slug.get(slug, []):
        rkp, rdesc, rw, rh = store.ref(i)
        for name, qv in qviews.items():
            mm = match(qv, rkp, rdesc, rw, rh)
            cur = out.get(name)
            if cur is None or (mm.inliers, mm.good) > (cur.inliers, cur.good):
                if cur is not None:
                    mm.ms += cur.ms
                out[name] = mm
            else:
                cur.ms += mm.ms
    return out


def match_shortlist(qv: QueryView, store: RefStore, slugs: Sequence[str]) -> list[Match]:
    """Сопоставление с эталонами всего списка кандидатов сразу + различимость внутри списка.

    Для каждого кандидата — то же, что `match` (тест отношения внутри эталона, один к одному,
    RANSAC-гомография, проверка вырождения), плюс тест отношения Лоу *по списку* (как в исходной
    статье — против всей базы): точка кадра засчитывается кандидату как различимая, если её
    лучшая пара у этого кандидата ближе 0.8 от лучшей пары у любого *другого* кандидата
    списка. Общий шаблон винодельни (логотип, рамка, герб) одинаково похож на всех соседей и
    выпадает; остаются буквы сорта, рисунок, серия — то, чем соседи различаются.

    `extra`: `uniq_good` — различимых пар кандидата, `uniq_inl` — из них согласных с его
    гомографией (0, если гомографии нет). Один эталон у slug может быть не один: соперники —
    другие slug, а не другое фото того же slug.
    """
    t0 = time.perf_counter()
    uniq_slugs = list(dict.fromkeys(slugs))
    refs: list[tuple[int, int]] = []  # (номер slug в списке, номер эталона)
    for si, s in enumerate(uniq_slugs):
        for ri in store.by_slug.get(s, []):
            refs.append((si, ri))
    n_s = len(uniq_slugs)
    if not refs or len(qv.desc) < 2:
        return [Match() for _ in slugs]
    parts = [store.ref(ri) for _, ri in refs]
    D = np.concatenate([p[1] for p in parts])
    bounds = np.cumsum([0] + [len(p[1]) for p in parts])
    sims_all = qv.desc @ D.T  # (nq, M)
    nq = len(qv.desc)
    rows = np.arange(nq)
    # лучший по каждому slug (для теста различимости)
    best_by_slug = np.full((nq, n_s), -2.0, dtype=np.float32)
    per_ref: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for k, (si, _) in enumerate(refs):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if b - a < 2:
            per_ref.append((np.zeros(nq, np.int64), np.full(nq, -2.0, np.float32), np.full(nq, -2.0, np.float32)))
            continue
        blk = sims_all[:, a:b]
        best = blk.argmax(axis=1)
        sb = blk[rows, best]
        tmp = blk.copy()
        tmp[rows, best] = -2.0
        ss = tmp.max(axis=1)
        per_ref.append((best, sb, ss))
        np.maximum(best_by_slug[:, si], sb, out=best_by_slug[:, si])
    # лучший среди ДРУГИХ slug для каждого slug: максимум по строке без своего столбца
    if n_s > 1:
        top2 = np.partition(best_by_slug, n_s - 2, axis=1)[:, -2:]
        first, second = top2.max(axis=1), top2.min(axis=1)
    else:
        first = best_by_slug[:, 0]
        second = np.full(nq, -2.0, np.float32)

    def dist(s: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * s))

    out_by_slug: dict[int, Match] = {}
    for k, (si, ri) in enumerate(refs):
        best, sb, ss = per_ref[k]
        rkp, _, rw, rh = parts[k]
        m = Match()
        keep = dist(sb) < RATIO * dist(ss)
        other = np.where(best_by_slug[:, si] >= first, second, first)  # лучший у других slug
        uniq = keep & (dist(sb) < RATIO * dist(other))
        qi = np.flatnonzero(keep)
        rix = best[keep]
        if len(qi):
            order = np.argsort(-sb[qi], kind="stable")
            _, fi = np.unique(rix[order], return_index=True)
            sel = order[fi]
            qi, rix = qi[sel], rix[sel]
        m.good = len(qi)
        m.extra = {"uniq_good": int(uniq[qi].sum()) if len(qi) else 0, "uniq_inl": 0}
        if m.good >= MIN_MATCHES:
            src = rkp[rix, :2].astype(np.float32)
            dst = qv.kp[qi].astype(np.float32)
            H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, RANSAC_PX, maxIters=2000, confidence=0.999)
            if H is not None and mask is not None:
                inl = mask.ravel().astype(bool)
                m.raw_inliers = int(inl.sum())
                m.valid = _homography_ok(H, src[inl], qv.w, qv.h)
                if m.valid and m.raw_inliers >= 4:
                    m.inliers = m.raw_inliers
                    rs, qs = src[inl], dst[inl]
                    m.area_ref = _hull_area(rs) / float(rw * rh)
                    band = (rs[:, 1] >= LABEL_Y0 * rh) & (rs[:, 1] <= LABEL_Y1 * rh)
                    m.area_label = _hull_area(rs[band]) / float(rw * rh * (LABEL_Y1 - LABEL_Y0))
                    m.area_q = _hull_area(qs) / float(qv.w * qv.h)
                    m.extra["uniq_inl"] = int((uniq[qi] & inl).sum())
        cur = out_by_slug.get(si)
        if cur is None or (m.inliers, m.good) > (cur.inliers, cur.good):
            out_by_slug[si] = m
    ms = (time.perf_counter() - t0) * 1e3
    res = []
    for s in slugs:
        m = out_by_slug.get(uniq_slugs.index(s), Match())
        m.ms = ms / max(1, len(slugs))
        res.append(m)
    return res


def set_threads(n: int) -> None:
    cv2.setNumThreads(n)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
