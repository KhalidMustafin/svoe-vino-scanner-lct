"""Обучаемый resolve: L2-логистическая регрессия поверх признаков пар «запрос — кандидат».

Вход — матрица признаков пар (`app.resolve.features`), цель — 1 у верного slug запроса и 0 у
остальных кандидатов. Счёт кандидата — `w·z + b`, где `z` — признаки, стандартизованные
средними и σ обучающих пар. Ответ — argmax счёта внутри запроса (при равенстве — меньший
ранг CV, то есть более ранняя строка). Вероятность того, что ответ верен, — максимум
softmax счётов кандидатов запроса с температурой `T`.

Две функции потерь:

    pointwise  бинарная логистическая регрессия по парам — режим по умолчанию;
    listwise   softmax по кандидатам запроса (условный логит): тот же линейный счёт, но
               учится сразу «верный выше соседей».

Оптимизация — метод Ньютона с дроблением шага: признаков десятки, пар тысячи, и решение
сходится за 10–20 итераций без случайности. Свободный член не штрафуется.

Знаки весов (`signs`): у признака с очевидным направлением вес держится на своей стороне
нуля (+1 — вес ≥ 0, −1 — вес ≤ 0). Без этого на 300 запросах модель выучивала «противоречие
по сахару повышает счёт» — шум, который в поле сломал бы ответ. Ограничение — проекционный
метод Ньютона: переменные на границе, которых градиент тянет наружу, закрепляются, по
остальным делается шаг Ньютона, а шаг с дроблением проецируется на допустимую область.
Если дробление не нашло убывания, `fit_info.converged` — false с причиной.

Запрос, у которого верного slug нет среди кандидатов, в обучение не входит: учиться там
нечему, а для отчёта это отдельная метрика «потолок K».

Модель сохраняется в JSON: веса, имена признаков, средние и σ, температура, версия признаков
и мета — на каких данных и когда обучена. Загрузка проверяет формат и версию признаков.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np

from app.reading.contracts import LabelFields
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import (
    DEFAULT_OPTIONS,
    DEFAULT_TOP_K,
    FEATURE_VERSION,
    FeatureOptions,
    QueryCv,
    QueryFeatures,
    TextRead,
    query_features,
    readers_of,
)

Loss = Literal["pointwise", "listwise"]
LOSSES: tuple[Loss, ...] = get_args(Loss)
MODEL_FORMAT = "svs-resolve-logistic/1"

#: Пределы поиска температуры softmax: шире не нужно, счёт и так в единицах логита.
TEMPERATURE_BOUNDS = (0.02, 50.0)
_RIDGE_EPS = 1e-10


class ModelFormatError(ValueError):
    """Файл модели другого формата или другой версии признаков."""


# ------------------------------------------------------------------ запросы
@dataclass(frozen=True, slots=True)
class QueryBlocks:
    """Разбиение строк матрицы на запросы: у каждого запроса — сплошной отрезок строк."""

    starts: np.ndarray  # int64, начало отрезка
    ends: np.ndarray  # int64, конец отрезка (не включая)

    @classmethod
    def from_ids(cls, query_ids: Sequence[Any] | np.ndarray) -> QueryBlocks:
        ids = list(query_ids)
        starts: list[int] = []
        seen: set[Any] = set()
        for i, qid in enumerate(ids):
            if i == 0 or qid != ids[i - 1]:
                if qid in seen:
                    raise ValueError(f"строки запроса {qid!r} идут не подряд")
                seen.add(qid)
                starts.append(i)
        ends = [*starts[1:], len(ids)]
        return cls(np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.starts)

    def slices(self) -> list[slice]:
        return [slice(int(a), int(b)) for a, b in zip(self.starts, self.ends, strict=True)]


def query_softmax(scores: np.ndarray, blocks: QueryBlocks, temperature: float = 1.0) -> np.ndarray:
    """Softmax счётов внутри каждого запроса."""
    out = np.empty_like(scores, dtype=np.float64)
    for part in blocks.slices():
        z = scores[part] / temperature
        z = z - z.max()
        e = np.exp(z)
        out[part] = e / e.sum()
    return out


def query_argmax(scores: np.ndarray, blocks: QueryBlocks) -> np.ndarray:
    """Индекс строки-ответа каждого запроса; при равенстве — первая строка (лучший ранг CV)."""
    return np.asarray(
        [part.start + int(np.argmax(scores[part])) for part in blocks.slices()], dtype=np.int64
    )


def top1_probability(scores: np.ndarray, blocks: QueryBlocks, temperature: float) -> np.ndarray:
    """Вероятность, что ответ запроса верен: максимум softmax с температурой."""
    probs = query_softmax(scores, blocks, temperature)
    return np.asarray([float(probs[part].max()) for part in blocks.slices()], dtype=np.float64)


def fit_temperature(
    scores: np.ndarray,
    blocks: QueryBlocks,
    correct: np.ndarray,
    *,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    iterations: int = 80,
) -> float:
    """Температура, при которой вероятность top-1 лучше всего отвечает факту «top-1 верен».

    Минимизируется бинарный log-loss по запросам: `correct[q]` — верен ли ответ запроса q.
    Запросы без верного кандидата участвуют с `correct=0`: это ровно та ситуация, где
    уверенный ответ стоит дорого. Поиск — золотое сечение по log T, детерминированный.
    """
    correct = np.asarray(correct, dtype=np.float64)
    if len(correct) != len(blocks):
        raise ValueError("fit_temperature: correct должен быть по одному на запрос")
    if not len(blocks):
        return 1.0

    def loss(log_t: float) -> float:
        p = np.clip(top1_probability(scores, blocks, math.exp(log_t)), 1e-9, 1 - 1e-9)
        return float(-(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean())

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = loss(a), loss(b)
    for _ in range(iterations):
        if fa <= fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = loss(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = loss(b)
    return float(math.exp((lo + hi) / 2))


# ------------------------------------------------------------------ модель
class LogisticRanker:
    """L2-логистическая регрессия, ранжирующая кандидатов внутри запроса."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        l2: float = 1.0,
        loss: Loss = "pointwise",
        signs: Sequence[int] | None = None,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        if not feature_names:
            raise ValueError("LogisticRanker: нет признаков")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("LogisticRanker: имена признаков повторяются")
        if l2 < 0:
            raise ValueError("LogisticRanker: l2 не может быть отрицательным")
        if loss not in LOSSES:
            raise ValueError(f"LogisticRanker: неизвестная функция потерь {loss!r}")
        if signs is None:
            signs = [0] * len(feature_names)
        if len(signs) != len(feature_names) or any(sign not in (-1, 0, 1) for sign in signs):
            raise ValueError("LogisticRanker: signs — по одному −1, 0 или +1 на признак")
        self.feature_names: list[str] = list(feature_names)
        self.signs: list[int] = [int(sign) for sign in signs]
        self.l2 = float(l2)
        self.loss: Loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.temperature_: float = 1.0
        self.meta: dict[str, Any] = {}
        self.fit_info: dict[str, Any] = {}

    # -------------------------------------------------------------- обучение
    @property
    def fitted(self) -> bool:
        return self.coef_ is not None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.scale_ is not None
        return (X - self.mean_) / self.scale_

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        query_ids: Sequence[Any] | np.ndarray,
        *,
        meta: Mapping[str, Any] | None = None,
    ) -> LogisticRanker:
        """Обучить на парах. Запросы без верного кандидата отбрасываются; два верных — ошибка."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"fit: ожидается матрица (пары × {len(self.feature_names)}), пришла {X.shape}"
            )
        if len(y) != len(X) or len(query_ids) != len(X):
            raise ValueError("fit: длины X, y и query_ids не совпадают")
        if not np.isfinite(X).all():
            raise ValueError("fit: в признаках NaN или бесконечность")
        blocks = QueryBlocks.from_ids(query_ids)
        keep_rows: list[np.ndarray] = []
        kept_ids: list[int] = []
        dropped = 0
        for q, part in enumerate(blocks.slices()):
            positives = int(y[part].sum())
            if positives > 1:
                raise ValueError(f"fit: у запроса {q} два верных кандидата")
            if positives == 0:
                dropped += 1
                continue
            keep_rows.append(np.arange(part.start, part.stop))
            kept_ids.extend([q] * (part.stop - part.start))
        if not keep_rows:
            raise ValueError("fit: ни у одного запроса нет верного кандидата")
        rows = np.concatenate(keep_rows)
        X, y = X[rows], y[rows]
        blocks = QueryBlocks.from_ids(kept_ids)

        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        Z = self._standardize(X)
        # σ > 0, поэтому знак веса стандартизованного признака — это знак и сырого.
        signs = np.asarray(self.signs)
        lower = np.where(signs > 0, 0.0, -np.inf)
        upper = np.where(signs < 0, 0.0, np.inf)
        if self.loss == "pointwise":
            w, b, info = _newton_pointwise(Z, y, self.l2, self.max_iter, self.tol, lower, upper)
        else:
            w, info = _newton_listwise(Z, y, blocks, self.l2, self.max_iter, self.tol, lower, upper)
            b = 0.0
        self.coef_, self.intercept_ = w, float(b)
        self.temperature_ = 1.0
        self.fit_info = {
            **info,
            "queries": len(blocks),
            "pairs": len(X),
            "queries_without_positive": dropped,
        }
        self.meta = dict(meta or {})
        return self

    # -------------------------------------------------------------- предсказание
    def _check(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("модель не обучена")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"ожидается матрица (кандидаты × {len(self.feature_names)}), пришла {X.shape}"
            )
        return X

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Счёт кандидатов: `w·z + b`."""
        X = self._check(X)
        assert self.coef_ is not None
        return self._standardize(X) @ self.coef_ + self.intercept_

    def predict(
        self, X: np.ndarray, query_ids: Sequence[Any] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Индексы строк-ответов по запросам и вероятность верности каждого ответа."""
        scores = self.decision_function(X)
        blocks = QueryBlocks.from_ids(query_ids)
        return query_argmax(scores, blocks), top1_probability(scores, blocks, self.temperature_)

    def calibrate(self, scores: np.ndarray, query_ids: Sequence[Any], correct: np.ndarray) -> float:
        """Подобрать температуру по счётам, которых модель при обучении не видела."""
        self.temperature_ = fit_temperature(
            np.asarray(scores, dtype=np.float64), QueryBlocks.from_ids(query_ids), correct
        )
        return self.temperature_

    def weights(self) -> dict[str, float]:
        """Веса в единицах стандартизованных признаков (сравнимы между собой)."""
        if self.coef_ is None:
            raise RuntimeError("модель не обучена")
        return {name: float(w) for name, w in zip(self.feature_names, self.coef_, strict=True)}

    # -------------------------------------------------------------- JSON
    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("модель не обучена")
        assert self.coef_ is not None and self.mean_ is not None and self.scale_ is not None
        return {
            "format": MODEL_FORMAT,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(self.feature_names),
            "coef": [float(v) for v in self.coef_],
            "intercept": self.intercept_,
            "mean": [float(v) for v in self.mean_],
            "scale": [float(v) for v in self.scale_],
            "temperature": self.temperature_,
            "l2": self.l2,
            "loss": self.loss,
            "signs": list(self.signs),
            "fit_info": self.fit_info,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LogisticRanker:
        if data.get("format") != MODEL_FORMAT:
            raise ModelFormatError(
                f"формат модели {data.get('format')!r}, ожидается {MODEL_FORMAT}"
            )
        if data.get("feature_version") != FEATURE_VERSION:
            raise ModelFormatError(
                f"модель обучена на признаках {data.get('feature_version')!r}, "
                f"а код считает {FEATURE_VERSION}"
            )
        names = list(data["feature_names"])
        signs = data.get("signs")
        if signs is not None and len(signs) != len(names):
            raise ModelFormatError(f"модель: signs длины {len(signs)}, признаков {len(names)}")
        model = cls(names, l2=float(data["l2"]), loss=data["loss"], signs=signs)
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in ("coef", "mean", "scale")}
        for key, value in arrays.items():
            if value.shape != (len(names),):
                raise ModelFormatError(f"модель: {key} длины {value.shape}, признаков {len(names)}")
        model.coef_, model.mean_, model.scale_ = arrays["coef"], arrays["mean"], arrays["scale"]
        model.intercept_ = float(data["intercept"])
        model.temperature_ = float(data["temperature"])
        model.fit_info = dict(data.get("fit_info") or {})
        model.meta = dict(data.get("meta") or {})
        return model

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=1, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> LogisticRanker:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ один запрос
@dataclass(frozen=True, slots=True)
class LearnedRanking:
    """Кандидаты запроса по убыванию счёта модели и вероятность, что первый верен."""

    slugs: tuple[str, ...]
    scores: tuple[float, ...]
    p_top1: float | None

    @property
    def slug(self) -> str | None:
        return self.slugs[0] if self.slugs else None


def rank_query_detailed(
    model: LogisticRanker,
    query_cv: QueryCv,
    reads: Mapping[str, TextRead | LabelFields | None],
    attrs: CatalogAttrs,
    *,
    top_k: int | None = None,
    options: FeatureOptions = DEFAULT_OPTIONS,
) -> tuple[LearnedRanking, QueryFeatures]:
    """То же, что `rank_query`, но отдаёт и признаки: их пересчитывает разбор спорного кадра."""
    k = top_k if top_k is not None else int(model.meta.get("top_k") or DEFAULT_TOP_K)
    needed = {reader: reads.get(reader) for reader in readers_of(model.feature_names)}
    features = query_features(query_cv, needed, attrs, top_k=k, options=options)
    if not features.rows:
        return LearnedRanking((), (), None), features
    scores = model.decision_function(features.matrix(model.feature_names))
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    blocks = QueryBlocks.from_ids([0] * len(scores))
    ranking = LearnedRanking(
        slugs=tuple(features.slugs[i] for i in order),
        scores=tuple(float(scores[i]) for i in order),
        p_top1=float(top1_probability(scores, blocks, model.temperature_)[0]),
    )
    return ranking, features


def rank_query(
    model: LogisticRanker,
    query_cv: QueryCv,
    reads: Mapping[str, TextRead | LabelFields | None],
    attrs: CatalogAttrs,
    *,
    top_k: int | None = None,
    options: FeatureOptions = DEFAULT_OPTIONS,
) -> LearnedRanking:
    """Решение обученной модели для одного запроса — то, что вызывает сервис.

    `reads` — чтения под теми же именами читателей, на которых модель обучена (`vlm35`,
    `rapid`); `top_k` по умолчанию — из меты модели. Читателя модели нет в `reads` (сбой или
    таймаут OCR) — чтение пустое, как при обучении: решает CV. Чтения других читателей
    модели не нужны и не считаются. Модель с признаком `unpublished` требует
    `options.published=True`, иначе `KeyError`: молча подставлять ноль нельзя.
    """
    ranking, _ = rank_query_detailed(model, query_cv, reads, attrs, top_k=top_k, options=options)
    return ranking


# ------------------------------------------------------------------ Ньютон
def _sigmoid(s: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(s))  # без переполнения при любом знаке
    return np.where(s >= 0, 1 / (1 + e), e / (1 + e))


Derivatives = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def _newton_pointwise(
    Z: np.ndarray,
    y: np.ndarray,
    l2: float,
    max_iter: int,
    tol: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    n, d = Z.shape
    A = np.hstack([Z, np.ones((n, 1))])
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0

    def objective(t: np.ndarray) -> float:
        s = A @ t
        return float(np.mean(np.logaddexp(0.0, s) - y * s) + 0.5 * np.sum(reg * t * t))

    def derivatives(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = _sigmoid(A @ t)
        grad = A.T @ (p - y) / n + reg * t
        H = (A * (p * (1 - p))[:, None]).T @ A / n + np.diag(reg + _RIDGE_EPS)
        return grad, H

    # Свободный член без ограничений: к границам признаков добавляется (−∞, +∞).
    theta, info = _minimize(
        objective,
        derivatives,
        np.zeros(d + 1),
        np.append(lower, -np.inf),
        np.append(upper, np.inf),
        max_iter,
        tol,
    )
    return theta[:-1].copy(), float(theta[-1]), info


def _newton_listwise(
    Z: np.ndarray,
    y: np.ndarray,
    blocks: QueryBlocks,
    l2: float,
    max_iter: int,
    tol: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    d = Z.shape[1]
    q = len(blocks)
    starts = blocks.starts
    positive = np.flatnonzero(y > 0.5)

    def objective(t: np.ndarray) -> float:
        s = Z @ t
        lse = np.asarray([_logsumexp(s[part]) for part in blocks.slices()])
        return float((lse.sum() - s[positive].sum()) / q + 0.5 * l2 * t @ t)

    def derivatives(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = query_softmax(Z @ t, blocks)
        grad = Z.T @ (p - y) / q + l2 * t
        M = np.add.reduceat(Z * p[:, None], starts, axis=0)
        H = ((Z * p[:, None]).T @ Z - M.T @ M) / q + np.eye(d) * (l2 + _RIDGE_EPS)
        return grad, H

    return _minimize(objective, derivatives, np.zeros(d), lower, upper, max_iter, tol)


def _logsumexp(z: np.ndarray) -> float:
    top = float(z.max())
    return top + math.log(float(np.exp(z - top).sum()))


def _minimize(
    objective: Callable[[np.ndarray], float],
    derivatives: Derivatives,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Проекционный метод Ньютона для выпуклой цели с границами `lower ≤ θ ≤ upper`.

    Без конечных границ это обычный Ньютон с дроблением шага. Переменная на границе (или в
    ε от неё), которую градиент тянет наружу, в шаге Ньютона не участвует: её шаг — ровно до
    границы. Остальные идут по шагу Ньютона на своей подматрице гессиана. Точка старта
    должна быть допустимой.
    """
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("_minimize: точка старта вне границ")
    value = objective(theta)
    iterations = 0
    status = "max_iter"
    for iterations in range(1, max_iter + 1):
        grad, H = derivatives(theta)
        # ε-активное множество (Бертсекас): ε не больше невязки проекционного градиента,
        # чтобы у самого решения закреплялись только переменные на границе.
        residual = float(np.max(np.abs(theta - np.clip(theta - grad, lower, upper)), initial=0.0))
        eps = min(1e-6, residual)
        at_lower = (theta <= lower + eps) & (grad > 0)
        at_upper = (theta >= upper - eps) & (grad < 0)
        pinned = at_lower | at_upper
        free = ~pinned
        step = np.zeros_like(theta)
        if free.any():
            step[free] = np.linalg.solve(H[np.ix_(free, free)], grad[free])
        step[pinned] = theta[pinned] - np.where(at_lower, lower, upper)[pinned]
        decrement = float(grad @ step)
        if decrement / 2 < tol:
            status = "converged"
            break
        new_theta, new_value = _line_search(objective, theta, step, grad, value, lower, upper)
        if new_theta is None:
            status = "line_search_failed"
            break
        change = abs(value - new_value)
        theta, value = new_theta, new_value
        if change < tol:
            status = "converged"
            break
    return theta, _info(iterations, status, value, theta, lower, upper)


def _line_search(
    objective: Callable[[np.ndarray], float],
    theta: np.ndarray,
    step: np.ndarray,
    grad: np.ndarray,
    value: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Шаг с дроблением по проекционной дуге: цель должна убывать (условие Армихо).

    Убывания нет и после 40 дроблений — `(None, value)`: оптимизатор остановится и честно
    скажет, что не сошёлся, а не выдаст последнюю точку за решение.
    """
    t = 1.0
    for _ in range(40):
        candidate = np.clip(theta - t * step, lower, upper)
        new_value = objective(candidate)
        if new_value <= value - 1e-4 * float(grad @ (theta - candidate)):
            return candidate, new_value
        t /= 2
    return None, value


def _info(
    iterations: int,
    status: str,
    value: float,
    theta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    bounded = np.isfinite(lower) | np.isfinite(upper)
    at_bound = bounded & ((theta == lower) | (theta == upper))
    return {
        "iterations": iterations,
        "converged": status == "converged",
        "status": status,
        "objective": round(value, 8),
        "constrained": int(bounded.sum()),
        "at_bound": int(at_bound.sum()),
    }
