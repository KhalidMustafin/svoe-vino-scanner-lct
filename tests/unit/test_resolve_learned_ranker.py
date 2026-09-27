"""LogisticRanker: обучение на numpy, argmax внутри запроса, температура, JSON, детерминизм."""

import json

import numpy as np
import pytest

from app.resolve import learned
from app.resolve.features import FEATURE_VERSION
from app.resolve.learned import (
    LOSSES,
    LogisticRanker,
    ModelFormatError,
    QueryBlocks,
    _minimize,
    fit_temperature,
    top1_probability,
)

NAMES = ["signal", "noise", "constant"]


def ranking_data(queries: int = 120, size: int = 8, seed: int = 0, noise: float = 0.2):
    """Запросы по `size` кандидатов: верный — с наибольшим `signal + шум`."""
    rng = np.random.default_rng(seed)
    X = np.zeros((queries * size, len(NAMES)))
    X[:, 0] = rng.normal(size=queries * size)
    X[:, 1] = rng.normal(size=queries * size)
    X[:, 2] = 3.0
    y = np.zeros(queries * size)
    ids = np.repeat(np.arange(queries), size)
    for q in range(queries):
        part = slice(q * size, (q + 1) * size)
        y[q * size + int(np.argmax(X[part, 0] + noise * rng.normal(size=size)))] = 1.0
    return X, y, ids


@pytest.mark.parametrize("loss", LOSSES)
def test_fit_learns_signal_and_ranks_within_query(loss):
    X, y, ids = ranking_data()
    model = LogisticRanker(NAMES, l2=0.01, loss=loss).fit(X, y, ids)
    weights = model.weights()
    assert weights["signal"] > 5 * abs(weights["noise"])
    assert weights["constant"] == 0.0  # постоянный признак: σ=1, стандартизованный — ноль
    best, prob = model.predict(X, ids)
    assert np.mean(y[best] == 1.0) > 0.7
    assert np.all((prob > 0) & (prob <= 1)) and len(best) == 120
    assert model.fit_info["converged"]


def test_queries_without_positive_are_dropped_and_do_not_change_weights():
    X, y, ids = ranking_data(queries=60)
    extra_X = np.random.default_rng(5).normal(size=(8, 3))
    X2 = np.vstack([X, extra_X])
    y2 = np.concatenate([y, np.zeros(8)])
    ids2 = np.concatenate([ids, np.full(8, 999)])
    a = LogisticRanker(NAMES, l2=0.1, loss="listwise").fit(X, y, ids)
    b = LogisticRanker(NAMES, l2=0.1, loss="listwise").fit(X2, y2, ids2)
    assert b.fit_info["queries_without_positive"] == 1 and b.fit_info["queries"] == 60
    np.testing.assert_array_equal(a.coef_, b.coef_)


def test_invalid_training_input_is_rejected():
    X, y, ids = ranking_data(queries=4, size=3)
    two = y.copy()
    two[:3] = 1.0
    with pytest.raises(ValueError, match="два верных"):
        LogisticRanker(NAMES).fit(X, two, ids)
    with pytest.raises(ValueError, match="не подряд"):
        LogisticRanker(NAMES).fit(X, y, [0, 1, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4])
    with pytest.raises(ValueError, match="ни у одного"):
        LogisticRanker(NAMES).fit(X, np.zeros_like(y), ids)
    with pytest.raises(ValueError):
        LogisticRanker(["a", "a"])


def test_stronger_l2_shrinks_weights():
    X, y, ids = ranking_data()
    weak = LogisticRanker(NAMES, l2=0.001).fit(X, y, ids)
    strong = LogisticRanker(NAMES, l2=1.0).fit(X, y, ids)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


@pytest.mark.parametrize("loss", LOSSES)
def test_fit_is_deterministic(loss):
    X, y, ids = ranking_data(seed=3)
    a = LogisticRanker(NAMES, l2=0.05, loss=loss).fit(X, y, ids)
    b = LogisticRanker(NAMES, l2=0.05, loss=loss).fit(X.copy(), y.copy(), ids.copy())
    np.testing.assert_array_equal(a.coef_, b.coef_)
    assert a.intercept_ == b.intercept_
    np.testing.assert_array_equal(a.decision_function(X), b.decision_function(X))


def test_save_and_load_round_trip(tmp_path):
    X, y, ids = ranking_data(seed=7)
    model = LogisticRanker(NAMES, l2=0.02, loss="listwise").fit(
        X, y, ids, meta={"split": "dev", "sets": ["pairs"]}
    )
    model.temperature_ = 1.7
    path = tmp_path / "models" / "m.json"
    model.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["feature_names"] == NAMES and data["feature_version"] == FEATURE_VERSION
    assert set(data) >= {"coef", "mean", "scale", "temperature", "meta", "l2", "loss"}
    loaded = LogisticRanker.load(path)
    np.testing.assert_array_equal(loaded.decision_function(X), model.decision_function(X))
    assert loaded.temperature_ == 1.7 and loaded.meta == {"split": "dev", "sets": ["pairs"]}
    assert loaded.loss == "listwise" and loaded.feature_names == NAMES


def test_load_refuses_other_feature_version(tmp_path):
    X, y, ids = ranking_data(queries=20)
    data = LogisticRanker(NAMES).fit(X, y, ids).to_dict()
    with pytest.raises(ModelFormatError, match="признаках"):
        LogisticRanker.from_dict({**data, "feature_version": "resolve-features/0"})
    with pytest.raises(ModelFormatError):
        LogisticRanker.from_dict({**data, "format": "other"})
    with pytest.raises(ModelFormatError):
        LogisticRanker.from_dict({**data, "coef": [1.0]})


def test_temperature_recovers_how_overconfident_scores_are():
    rng = np.random.default_rng(1)
    queries, size, true_t = 2000, 5, 3.0
    scores = rng.normal(size=queries * size) * 4
    blocks = QueryBlocks.from_ids(np.repeat(np.arange(queries), size))
    # Верный кандидат выпадает по softmax(s / 3): счёты втрое самоувереннее правды.
    correct = np.zeros(queries)
    for q, part in enumerate(blocks.slices()):
        z = scores[part] / true_t
        p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        correct[q] = float(rng.choice(size, p=p) == int(np.argmax(scores[part])))
    t = fit_temperature(scores, blocks, correct)
    assert 2.4 < t < 3.6
    assert abs(float(top1_probability(scores, blocks, t).mean()) - correct.mean()) < 0.03
    assert fit_temperature(scores, blocks, correct) == t


# ------------------------------------------------------------------ знаки весов
def contrary_data(queries: int = 150, size: int = 6, seed: int = 4):
    """`signal` решает, а `wrong_way` по построению чуть мешает верному: свободный вес у него
    отрицательный — как «противоречие по сахару повышает счёт» на 300 запросах dev."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(queries * size, 3))
    y = np.zeros(queries * size)
    ids = np.repeat(np.arange(queries), size)
    for q in range(queries):
        part = slice(q * size, (q + 1) * size)
        y[
            q * size + int(np.argmax(X[part, 0] - 0.6 * X[part, 1] + 0.3 * rng.normal(size=size)))
        ] = 1
    return X, y, ids


@pytest.mark.parametrize("loss", LOSSES)
def test_signs_keep_weights_on_their_side_of_zero(loss):
    names = ["signal", "wrong_way", "free"]
    X, y, ids = contrary_data()
    free = LogisticRanker(names, l2=0.01, loss=loss).fit(X, y, ids)
    assert free.weights()["wrong_way"] < -0.5
    signed = LogisticRanker(names, l2=0.01, loss=loss, signs=[1, 1, 0]).fit(X, y, ids)
    w = signed.weights()
    assert w["wrong_way"] == 0.0 and w["signal"] > 0
    assert signed.fit_info["converged"] and signed.fit_info["at_bound"] == 1
    # Вес на границе — как будто признака нет: остальные веса те же, что без него.
    without = LogisticRanker(["signal", "free"], l2=0.01, loss=loss).fit(X[:, [0, 2]], y, ids)
    np.testing.assert_allclose(signed.coef_[[0, 2]], without.coef_, atol=1e-6)
    mirrored = LogisticRanker(names, l2=0.01, loss=loss, signs=[-1, -1, 0]).fit(X, y, ids)
    assert mirrored.weights()["signal"] == 0.0 and mirrored.weights()["wrong_way"] < 0


def test_signs_are_saved_and_checked(tmp_path):
    X, y, ids = contrary_data(queries=40)
    model = LogisticRanker(["a", "b", "c"], signs=[1, -1, 0]).fit(X, y, ids)
    model.save(tmp_path / "m.json")
    assert LogisticRanker.load(tmp_path / "m.json").signs == [1, -1, 0]
    with pytest.raises(ValueError):
        LogisticRanker(["a", "b"], signs=[1])
    with pytest.raises(ValueError):
        LogisticRanker(["a"], signs=[2])


def test_failed_line_search_is_not_reported_as_convergence():
    """Шаг, который цель не уменьшает ни при каком дроблении, — остановка с причиной."""

    def objective(t):
        return float(t @ t)

    def wrong_derivatives(t):  # градиент с неверным знаком: шаг Ньютона ведёт вверх
        return -2 * t, 2 * np.eye(len(t))

    inf = np.full(1, np.inf)
    theta, info = _minimize(objective, wrong_derivatives, np.ones(1), -inf, inf, 50, 1e-10)
    assert info["status"] == "line_search_failed" and info["converged"] is False
    np.testing.assert_array_equal(theta, np.ones(1))


def test_fit_reports_line_search_failure(monkeypatch):
    X, y, ids = ranking_data(queries=30)
    monkeypatch.setattr(learned, "_line_search", lambda *args, **kwargs: (None, args[4]))
    model = LogisticRanker(NAMES, l2=0.1).fit(X, y, ids)
    assert model.fit_info["converged"] is False
    assert model.fit_info["status"] == "line_search_failed"
