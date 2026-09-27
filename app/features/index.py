"""Индекс эталонов каталога и поиск по нему.

В индексе не одна строка на вино, а по строке на каждый вид каждого эталона: у одного
slug их обычно три (bottle, label, band). Запрос приходит тоже несколькими видами, и
близость вина — лучшая по всем парам «вид запроса × вид эталона». Вино узнаётся, если
совпало хоть одно сочетание: на кадре с телефона этикетка видна крупно, на packshot —
целиком, и сравнивать их «кадр к кадру» бессмысленно.

Формат — `npz`: 2 103 slug × 3 вида × 768 float16 весят 9 МБ, плоский поиск по такой
матрице занимает миллисекунды, и ни ANN-библиотеки, ни сервера индекс не требует.

Индекс привязан к модели: векторы разных моделей несравнимы, поэтому чужая модель —
отказ, а не молчаливо неверная выдача.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import numpy as np

from app.features.adapter import LinearAdapter
from app.features.contracts import (
    VIEWS,
    Aggregation,
    Candidate,
    Embedder,
    IndexMeta,
    ViewName,
    VisualResult,
)
from app.features.embedder import unit_rows
from app.features.views import order_views

#: Сколько кадров уходит в модель за раз при сборке.
DEFAULT_BATCH_SIZE = 32

Entry = tuple[str, dict[ViewName, np.ndarray]]


class IndexMismatch(ValueError):
    """Индекс и модель (или файл) не сходятся: искать по ним нельзя."""


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _ms(seconds: float) -> float:
    """Миллисекунды с долями: поиск по индексу занимает 1,4 мс, и в целых он был бы нулём."""
    return round(seconds * 1000, 3)


#: Способы свести близости векторов одного slug к одной.
AGGREGATIONS: tuple[str, ...] = get_args(Aggregation)

#: σ пары меньше этого — выравнивать не по чему (один вектор вида, одинаковые векторы).
PAIR_SD_MIN = 1e-6


def view_groups(views: Sequence[str]) -> list[np.ndarray]:
    """Номера строк индекса по видам эталона: одна группа — один вид."""
    names = np.asarray(list(views))
    return [np.flatnonzero(names == name) for name in dict.fromkeys(views)]


def align_pairs(
    sims: np.ndarray, groups: Sequence[np.ndarray], base_rows: np.ndarray | None = None
) -> np.ndarray:
    """Косинусы пар «окно запроса × вид эталона», приведённые к общему распределению запроса.

    `sims` — (векторы индекса, окна запроса), `groups` — строки каждого вида (`view_groups`).
    У каждой пары (окно w, вид v) свой фон: полоса этикетки запроса похожа на любую этикетку
    каталога сильнее, чем кадр целиком — на любую полосу, и максимум по парам выбирал бы пару
    с высоким фоном, а не лучшее совпадение. Поэтому косинусы пары стандартизуются по всем
    векторам вида v (среднее и σ по каталогу для этого окна) и возвращаются в единицы косинуса
    средним M и σ s всей матрицы запроса: `M + s·(c − mu_wv) / sd_wv`. Внутри пары порядок
    не меняется — меняется только то, какая пара выигрывает. Уровень запроса сохраняется: кадр,
    непохожий ни на что, остаётся с низким счётом. Параметров нет. Пара с σ меньше
    `PAIR_SD_MIN` (вырожденный индекс) остаётся сырым косинусом.

    `base_rows` — маска строк (bool, по строке индекса), по которым считаются mu, sd пар и общие
    M, s; выравниваются все строки. Так дополнение индекса (живые карточки портала поверх CSV)
    не сдвигает счёт старых строк: их нормировка заморожена по базе. `None` — все строки, путь
    прежний. Вид, у которого в базе нет ни одной строки, выровнять не по чему — сырой косинус.
    """
    sims = np.asarray(sims, dtype=np.float32)
    out = sims.copy()
    ref = sims if base_rows is None else sims[base_rows]
    total_sd = float(ref.std())
    if total_sd < PAIR_SD_MIN:
        return out
    total_mean = float(ref.mean())
    for rows in groups:
        block = sims[rows]
        if base_rows is None:
            stats = block
        else:
            stats = sims[rows[base_rows[rows]]]
            if not len(stats):
                continue
        mean = stats.mean(axis=0, keepdims=True)
        sd = stats.std(axis=0, keepdims=True)
        ok = sd >= PAIR_SD_MIN
        aligned = total_mean + total_sd * (block - mean) / np.where(ok, sd, 1.0)
        out[rows] = np.where(ok, aligned, block)
    return out


def base_rows_mask(base_rows: object, n_rows: int) -> np.ndarray | None:
    """Маска базовых строк индекса или `None`, если база — весь индекс (путь без маски).

    Маска «все строки базовые» сводится к `None`: индекс без добавлений считает ровно как
    прежде, бит в бит. Длина не та, не bool или ни одной базовой строки — `IndexMismatch`.
    """
    if base_rows is None:
        return None
    mask = np.asarray(base_rows)
    if mask.dtype != np.bool_ or mask.shape != (n_rows,):
        raise IndexMismatch(
            f"base_rows: ожидается bool длины {n_rows}, получено {mask.dtype} {mask.shape}"
        )
    if not mask.any():
        raise IndexMismatch("base_rows: ни одной базовой строки — нормировать не по чему")
    return None if mask.all() else mask.copy()


Batch = tuple[list[str], list[ViewName], list[np.ndarray]]


def _batches(entries: Iterable[Entry], batch_size: int) -> Iterator[Batch]:
    """Кадры всех видов подряд, порциями по `batch_size`: модель любит батчи."""
    slugs: list[str] = []
    views: list[ViewName] = []
    images: list[np.ndarray] = []
    for slug, view_map in entries:
        if not view_map:
            raise ValueError(f"{slug}: ни одного вида")
        for name in order_views(view_map):
            slugs.append(slug)
            views.append(name)
            images.append(view_map[name])
            if len(images) == batch_size:
                yield slugs, views, images
                slugs, views, images = [], [], []
    if images:
        yield slugs, views, images


class VisualIndex:
    """Векторы эталонов, их slug и виды. Поиск — плоский косинус по всей матрице.

    `base_rows` — маска строк, по которым нормируется `per_slug="zmax"` (`align_pairs`):
    у индекса, дополненного карточками вне каталога CSV, это строки CSV. `None` — все строки.

    `adapter` — линейный адаптер поиска (`app/features/adapter.py`, путь `SVS_CANDIDATE`):
    `None` — поиск без адаптера (`SVS_CANDIDATE=off`), бит в бит как до 26.09. Подключается `with_adapter`: строки индекса
    проецируются один раз, запрос — в `_rank`.
    """

    adapter: LinearAdapter | None = None

    def __init__(
        self,
        slugs: list[str],
        views: list[ViewName],
        vectors: np.ndarray,
        meta: IndexMeta,
        base_rows: np.ndarray | None = None,
    ) -> None:
        if not (len(slugs) == len(views) == len(vectors)):
            raise ValueError(
                f"индекс не сходится: {len(slugs)} slug, {len(views)} видов, {len(vectors)} векторов"
            )
        self.slugs = list(slugs)
        self.views = list(views)
        self.vectors = unit_rows(vectors)
        self.meta = meta
        self.base_rows = base_rows_mask(base_rows, len(self.slugs))
        # Порядок slug — первого появления: он же порядок строк в агрегации.
        self.slug_order: list[str] = list(dict.fromkeys(self.slugs))
        position = {slug: i for i, slug in enumerate(self.slug_order)}
        self.slug_ids = np.array([position[s] for s in self.slugs], dtype=np.int64)
        # Строки по видам для `per_slug="zmax"`: считаются один раз, а не на каждый запрос.
        self.view_groups = view_groups(self.views)

    def __len__(self) -> int:
        return len(self.slugs)

    @property
    def n_base(self) -> int:
        """Сколько строк задают нормировку `zmax`: все, если индекс без добавлений."""
        return len(self.slugs) if self.base_rows is None else int(self.base_rows.sum())

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if len(self.vectors) else self.meta.dim

    @property
    def query_dim(self) -> int:
        """Размер вектора запроса до адаптера: его отдаёт эмбеддер."""
        return self.adapter.dim_in if self.adapter is not None else self.dim

    def with_adapter(self, adapter: LinearAdapter) -> VisualIndex:
        """Тот же индекс в пространстве адаптера: строки проецируются здесь, запрос — в `_rank`.

        Проекция — `adapter.apply` без повторной `unit_rows`: так выдача совпадает бит в бит с
        выдачей стенда, на которой оценивались кандидаты.
        """
        if self.adapter is not None:
            raise IndexMismatch("индекс уже в пространстве адаптера")
        if adapter.dim_in != self.dim:
            raise IndexMismatch(f"адаптер ждёт векторы {adapter.dim_in}, а в индексе {self.dim}")
        out = VisualIndex.__new__(VisualIndex)
        out.__dict__.update(self.__dict__)
        out.vectors = adapter.apply(self.vectors)
        out.adapter = adapter
        return out

    @property
    def n_slugs(self) -> int:
        return len(self.slug_order)

    @classmethod
    def build(
        cls,
        entries: Iterable[Entry],
        embedder: Embedder,
        *,
        source_sha1: str = "",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> VisualIndex:
        """Собрать индекс, считая векторы порциями. `entries` может быть генератором."""
        if batch_size <= 0:
            raise ValueError(f"build: batch_size должен быть > 0, получено {batch_size}")
        slugs: list[str] = []
        views: list[ViewName] = []
        chunks: list[np.ndarray] = []
        for batch_slugs, batch_views, images in _batches(entries, batch_size):
            vectors = np.asarray(embedder.embed(images), dtype=np.float32)
            if vectors.shape[0] != len(images):
                raise ValueError(
                    f"эмбеддер вернул {vectors.shape[0]} векторов на {len(images)} кадров"
                )
            slugs += batch_slugs
            views += batch_views
            chunks.append(vectors)
        dim = int(chunks[0].shape[1]) if chunks else int(embedder.dim)
        matrix = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, dim), np.float32)
        meta = IndexMeta(
            model=embedder.model_name,
            dim=dim,
            views=[name for name in VIEWS if name in set(views)],
            n_slugs=len(set(slugs)),
            n_vectors=len(slugs),
            built_at=_now(),
            source_sha1=source_sha1,
        )
        return cls(slugs, views, matrix, meta)

    def save(self, path: str | Path) -> Path:
        """Записать индекс в `npz`. Векторы — float16: разница с float32 ниже шума кадра."""
        if self.adapter is not None:
            raise IndexMismatch(
                "индекс в пространстве адаптера не сохраняется: адаптер — отдельный файл"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "slugs": np.array(self.slugs, dtype=np.str_),
            "views": np.array(self.views, dtype=np.str_),
            "vectors": self.vectors.astype(np.float16),
            "meta": np.array(self.meta.model_dump_json(), dtype=np.str_),
        }
        if self.base_rows is not None:
            arrays["base_rows"] = self.base_rows
        # Через файл, а не путь: `np.savez` сам дописывает «.npz» к имени без расширения,
        # и сохранённый файл оказался бы не там, где его потом ищут.
        with path.open("wb") as fh:
            np.savez(fh, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path, *, model: str | None = None) -> VisualIndex:
        """Прочитать индекс. `model` задан и не совпал с паспортом — `IndexMismatch`.

        Необязательное поле `base_rows` — маска строк нормировки `zmax` (см. `VisualIndex`).
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            missing = {"slugs", "views", "vectors", "meta"} - set(data.files)
            if missing:
                raise IndexMismatch(f"{path}: в файле нет полей {sorted(missing)}")
            meta = IndexMeta.model_validate_json(str(data["meta"].item()))
            slugs = [str(s) for s in data["slugs"].tolist()]
            views = [str(v) for v in data["views"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            base_rows = np.asarray(data["base_rows"]) if "base_rows" in data.files else None
        if model is not None and model != meta.model:
            raise IndexMismatch(
                f"{path}: индекс собран моделью {meta.model!r}, а считать просят {model!r}"
            )
        if vectors.ndim != 2 or (len(vectors) and vectors.shape[1] != meta.dim):
            raise IndexMismatch(f"{path}: векторы {vectors.shape} против dim {meta.dim} в паспорте")
        unknown = sorted(set(views) - set(VIEWS))
        if unknown:
            raise IndexMismatch(f"{path}: неизвестные виды {unknown}")
        try:
            return cls(slugs, views, vectors, meta, base_rows)  # type: ignore[arg-type]
        except IndexMismatch as exc:
            raise IndexMismatch(f"{path}: {exc}") from None

    def check_model(self, model: str) -> None:
        """Векторы разных моделей несравнимы: чужая модель — отказ, а не тихий мусор."""
        if model != self.meta.model:
            raise IndexMismatch(
                f"индекс собран моделью {self.meta.model!r}, а запрос считает {model!r}"
            )

    def search(
        self,
        query_views: dict[ViewName, np.ndarray],
        embedder: Embedder,
        *,
        top_k: int = 20,
        per_slug: Aggregation = "max",
    ) -> VisualResult:
        """Кандидаты по убыванию близости.

        Близость вина — максимум (или среднее при `per_slug="mean"`) косинусов его
        векторов, каждый из которых взят по лучшему виду запроса. `per_slug="zmax"` — тот же
        максимум, но по косинусам, выровненным по парам «окно × вид» (`align_pairs`).
        """
        if top_k <= 0:
            raise ValueError(f"search: top_k должен быть > 0, получено {top_k}")
        if per_slug not in AGGREGATIONS:
            raise ValueError(f"search: per_slug одно из {AGGREGATIONS}, получено {per_slug!r}")
        if not query_views:
            raise ValueError("search: ни одного вида запроса")
        self.check_model(getattr(embedder, "model_name", ""))
        started = time.perf_counter()
        names = order_views(query_views)
        queries = unit_rows(np.asarray(embedder.embed([query_views[n] for n in names])))
        embed_ms = _ms(time.perf_counter() - started)
        if queries.shape[1] != self.query_dim:
            raise IndexMismatch(
                f"вектор запроса {queries.shape[1]} против индекса {self.query_dim}"
            )
        matched = time.perf_counter()
        candidates, margin = self._rank(queries, top_k, per_slug)
        match_ms = _ms(time.perf_counter() - matched)
        return VisualResult(
            candidates=candidates,
            margin=margin,
            timings_ms={
                "embed": embed_ms,
                "match": match_ms,
                "total": _ms(time.perf_counter() - started),
            },
            model=self.meta.model,
        )

    def _rank(
        self, queries: np.ndarray, top_k: int, per_slug: Aggregation
    ) -> tuple[list[Candidate], float]:
        if not len(self.vectors):
            return [], 0.0
        if self.adapter is not None:
            queries = self.adapter.apply(queries)
        sims = self.vectors @ queries.T
        if per_slug == "zmax":
            sims = align_pairs(sims, self.view_groups, self.base_rows)
        # Лучший вид запроса для каждого вектора эталона.
        best = sims.max(axis=1)
        if per_slug == "mean":
            counts = np.bincount(self.slug_ids, minlength=self.n_slugs)
            sums = np.bincount(self.slug_ids, weights=best, minlength=self.n_slugs)
            scores = (sums / np.maximum(counts, 1)).astype(np.float32)
        else:
            scores = np.full(self.n_slugs, -np.inf, dtype=np.float32)
            np.maximum.at(scores, self.slug_ids, best)
        # Сырой косинус выше 1 — только округление float; выровненный бывает выше 1 по делу.
        ceiling = np.inf if per_slug == "zmax" else 1.0
        # Вид, давший лучшую близость: первый вектор каждого slug в общем порядке убывания.
        order = np.argsort(-best, kind="stable")
        _, first = np.unique(self.slug_ids[order], return_index=True)
        best_vector = order[first]
        order = np.argsort(-scores, kind="stable")
        candidates = [
            Candidate(
                slug=self.slug_order[int(slug_id)],
                score=float(min(ceiling, max(0.0, scores[slug_id]))),
                view=self.views[int(best_vector[slug_id])],
                rank=rank,
            )
            for rank, slug_id in enumerate(order[:top_k], start=1)
        ]
        # Отрыв считается по всему каталогу, а не по срезанной выдаче: при `top_k=1` длина
        # среза — единица, и главный признак уверенности молча вырождался бы в ноль.
        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
        return candidates, max(0.0, margin)
