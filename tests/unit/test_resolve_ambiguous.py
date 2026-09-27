"""Разбор спорного кадра: снятый бонус соседу по кластеру и отсев по сахару и цвету.

Модель здесь обучается на игрушечных данных, где сосед по кластеру — сильный признак: так видно,
что правило действительно снимает его вклад, а не переставляет кандидатов случайно.
"""

import numpy as np
import pytest
from catalog import toy_attrs, wine_record

from app.reading.contracts import Color, Evidence, LabelFields, SugarClass
from app.resolve.ambiguous import (
    CLUSTER_FEATURE,
    block_bonus_flip,
    contradicts,
    rerank_ambiguous,
    without_bonus,
)
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import QueryFeatures
from app.resolve.learned import LogisticRanker

NAMES = ("cv_score", CLUSTER_FEATURE)


def evidence(value: object) -> Evidence:
    return Evidence(value=value, sources=("тест",), support=1, conf=1.0)


@pytest.fixture
def attrs() -> CatalogAttrs:
    return toy_attrs(
        [
            wine_record("suhoe-beloe", name="Сухое", sugar="suhoe", color="Белое"),
            wine_record(
                "polusladkoe-beloe", name="Полусладкое", sugar="polusladkoe", color="Белое"
            ),
            wine_record("suhoe-krasnoe", name="Красное", sugar="suhoe", color="Красное"),
            wine_record("bez-klassov", name="Без классов"),
        ]
    )


@pytest.fixture
def model() -> LogisticRanker:
    """Сосед по кластеру тянет ответ сильнее, чем косинус CV."""
    rng = np.random.default_rng(0)
    rows, labels, ids = [], [], []
    for query in range(80):
        cv = rng.random(3)
        mate = np.zeros(3)
        mate[int(rng.integers(0, 3))] = 1.0
        for i in range(3):
            rows.append([cv[i], mate[i]])
            labels.append(1.0 if mate[i] == 1.0 else 0.0)
            ids.append(query)
    return LogisticRanker(NAMES, l2=0.01, loss="listwise").fit(
        np.array(rows), np.array(labels), ids
    )


def features(pairs: list[tuple[str, float, float]]) -> QueryFeatures:
    return QueryFeatures(
        slugs=tuple(slug for slug, _, _ in pairs),
        rows=tuple({"cv_score": cv, CLUSTER_FEATURE: mate} for _, cv, mate in pairs),
    )


def test_cluster_mate_bonus_is_dropped(model, attrs):
    """Без отсева ответ меняется только потому, что снят бонус за соседство с CV-лидером."""
    query = features([("suhoe-beloe", 0.9, 0.0), ("suhoe-krasnoe", 0.4, 1.0)])
    by_model = model.decision_function(query.matrix(NAMES))
    assert query.slugs[int(np.argmax(by_model))] == "suhoe-krasnoe"  # бонус перевесил косинус
    assert rerank_ambiguous(model, query, None, attrs)[0] == "suhoe-beloe"


def test_read_sugar_and_color_drop_contradicting_cards(model, attrs):
    """Прочитано «полусладкое белое» — сухое и красное выбывают, даже если CV их любит."""
    fields = LabelFields(
        sugar=[evidence(SugarClass("polusladkoe"))], color=evidence(Color("Белое"))
    )
    query = features(
        [("suhoe-beloe", 0.9, 0.0), ("suhoe-krasnoe", 0.8, 0.0), ("polusladkoe-beloe", 0.1, 0.0)]
    )
    assert rerank_ambiguous(model, query, fields, attrs)[0] == "polusladkoe-beloe"


def test_empty_after_filter_keeps_order(model, attrs):
    """Отсев выкинул всех — ответ остаётся прежним: пустой ответ хуже спорного."""
    fields = LabelFields(sugar=[evidence(SugarClass("brut"))])
    query = features([("suhoe-beloe", 0.9, 0.0), ("polusladkoe-beloe", 0.5, 0.0)])
    assert rerank_ambiguous(model, query, fields, attrs) == ("suhoe-beloe", "polusladkoe-beloe")


def test_unknown_class_is_not_a_conflict(attrs):
    """Каталог не знает сахара карточки — это не спор, а незнание."""
    fields = LabelFields(sugar=[evidence(SugarClass("suhoe"))], color=evidence(Color("Белое")))
    assert contradicts("bez-klassov", fields, attrs) is False
    assert contradicts("polusladkoe-beloe", fields, attrs) is True
    assert contradicts("suhoe-krasnoe", fields, attrs) is True
    assert contradicts("suhoe-beloe", fields, attrs) is False
    assert contradicts("polusladkoe-beloe", None, attrs) is False


def test_orange_and_white_do_not_contradict():
    """Оранжевые вина маркируются «белое»: пара «Белое» / «Оранжевое» — не спор (Э4)."""
    attrs = toy_attrs(
        [
            wine_record("oranzh", name="Ркацители Оранж", color="Оранжевое"),
            wine_record("oranzh-v-kataloge-beloe", name="Мускат Оранж", color="Белое"),
            wine_record("krasnoe", name="Каберне", color="Красное"),
        ]
    )
    white = LabelFields(color=evidence(Color("Белое")))
    orange = LabelFields(color=evidence(Color("Оранжевое")))
    assert contradicts("oranzh", white, attrs) is False
    assert contradicts("oranzh-v-kataloge-beloe", orange, attrs) is False
    assert contradicts("krasnoe", orange, attrs) is True
    assert contradicts("oranzh", LabelFields(color=evidence(Color("Розовое"))), attrs) is True


# ------------------------------------------------------------------ P1: уверенный кадр
def white() -> LabelFields:
    return LabelFields(color=evidence(Color("Белое")))


def flipped(model) -> tuple[QueryFeatures, list[str]]:
    """Бонус соседу сделал лидером красное, без бонуса лидер — белое; порядок модели."""
    query = features(
        [("suhoe-beloe", 0.9, 0.0), ("suhoe-krasnoe", 0.4, 1.0), ("polusladkoe-beloe", 0.2, 0.0)]
    )
    ranked = [query.slugs[i] for i in np.argsort(-model.decision_function(query.matrix(NAMES)))]
    assert ranked[0] == "suhoe-krasnoe" and without_bonus(model, query)[0] == "suhoe-beloe"
    return query, ranked


def test_bonus_flip_against_read_color_is_blocked(model, attrs):
    """Прочитано «белое»: переворот бонусом к красному снят, остальные — в порядке модели."""
    query, ranked = flipped(model)
    guarded = block_bonus_flip(model, query, ranked, white(), attrs)
    assert guarded[0] == "suhoe-beloe"
    assert list(guarded[1:]) == [slug for slug in ranked if slug != "suhoe-beloe"]


def test_bonus_flip_stays_without_evidence_against_it(model, attrs):
    """Не прочитано ничего или прочитанное не спорит с лидером — ответ модели не трогается."""
    query, ranked = flipped(model)
    assert block_bonus_flip(model, query, ranked, None, attrs) == tuple(ranked)
    red = LabelFields(color=evidence(Color("Красное")))
    assert block_bonus_flip(model, query, ranked, red, attrs) == tuple(ranked)


def test_bonus_flip_stays_when_the_plain_leader_also_conflicts(model, attrs):
    """Лидер без бонуса тоже спорит с этикеткой — довода за него нет, ответ модели остаётся."""
    query, ranked = flipped(model)
    brut = LabelFields(sugar=[evidence(SugarClass("brut"))])
    assert block_bonus_flip(model, query, ranked, brut, attrs) == tuple(ranked)


def test_no_flip_no_change(model, attrs):
    """Лидер модели и без бонуса тот же — правилу нечего запрещать."""
    query = features([("suhoe-krasnoe", 0.9, 1.0), ("suhoe-beloe", 0.4, 0.0)])
    ranked = list(without_bonus(model, query))
    assert block_bonus_flip(model, query, ranked, white(), attrs) == tuple(ranked)
