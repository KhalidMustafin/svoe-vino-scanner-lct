"""Разбор спорного кадра: когда обученный слой не уверен, ему мешают две вещи.

Первая — признак `cluster_mate_of_top1` (вес +0,31, около +2 σ): сосед по визуальному кластеру
CV-лидера получает бонус просто за соседство, и на полевых кадрах именно он уводит ответ к вину той
же винодельни с другим сахаром или цветом. Вторая — вес сахара, обученный ровно в ноль
(ограничение знака обрезало его на студийном dev), из-за чего прочитанное «полусладкое» не мешает
выбрать брют.

Поэтому при низкой уверенности кандидаты пересчитываются без бонуса соседу, а те, чьи сахар и цвет
спорят с прочитанным, из выдачи выбывают. Если после отсева не осталось никого — порядок остаётся
прежним: пустой ответ хуже спорного. При высокой уверенности (`block_bonus_flip`, правило P1)
бонус соседу не может перевернуть ответ к карточке, которая спорит с прочитанным сахаром или цветом.

Замер на общем полевом датасете (`field_dataset`, вина каталога, макросреднее по винам):
оригиналы 84,8 → 90,5 % (чинит 7 кадров, ломает 0), реалистичные искажения 81,6 → 86,0 %.
Правило параметров не имеет: порог — существующая константа `AMBIGUOUS_P_TOP1` сервиса.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.reading.contracts import LabelFields
from app.resolve.attrs import CatalogAttrs, colors_compatible
from app.resolve.features import QueryFeatures
from app.resolve.learned import LogisticRanker

#: Бонус за соседство с CV-лидером по визуальному кластеру — его и снимаем.
CLUSTER_FEATURE = "cluster_mate_of_top1"


def _value(wine: object, field: str) -> str | None:
    return getattr(getattr(wine, field, None), "value", None)


def contradicts(slug: str, fields: LabelFields | None, attrs: CatalogAttrs) -> bool:
    """Спорит ли карточка с уверенно прочитанным сахаром или цветом.

    Спор — только когда обе стороны известны: у карточки нет класса — не спор, не прочитали — не спор.
    «Белое» и «Оранжевое» не спорят (`colors_compatible`): оранжевые вина маркируются «белое».
    """
    if fields is None:
        return False
    wine = attrs.get(slug)
    if wine is None:
        return False
    sugar = {str(item.value) for item in fields.sugar}
    if sugar:
        card = _value(wine, "sugar")
        if card and card not in sugar:
            return True
    if fields.color is not None:
        card = _value(wine, "color")
        read = str(fields.color.value)
        if card and card != read and not colors_compatible(card, read):
            return True
    return False


def without_bonus(model: LogisticRanker, features: QueryFeatures) -> tuple[str, ...]:
    """Кандидаты по счёту модели с обнулённым бонусом соседу; при равенстве — лучший ранг CV."""
    names: Sequence[str] = model.feature_names
    if not features.rows or model.coef_ is None or model.mean_ is None or model.scale_ is None:
        return features.slugs
    coef = np.asarray(model.coef_, dtype=np.float64).copy()
    if CLUSTER_FEATURE in names:
        coef[list(names).index(CLUSTER_FEATURE)] = 0.0
    z = (features.matrix(names) - model.mean_) / model.scale_
    scores = z @ coef + model.intercept_
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return tuple(features.slugs[i] for i in order)


def rerank_ambiguous(
    model: LogisticRanker,
    features: QueryFeatures,
    fields: LabelFields | None,
    attrs: CatalogAttrs,
) -> tuple[str, ...]:
    """Порядок кандидатов спорного кадра: без бонуса соседу и без спорящих по сахару и цвету."""
    ranked = without_bonus(model, features)
    kept = [slug for slug in ranked if not contradicts(slug, fields, attrs)]
    return tuple(kept or ranked)


def block_bonus_flip(
    model: LogisticRanker,
    features: QueryFeatures,
    ranked: Sequence[str],
    fields: LabelFields | None,
    attrs: CatalogAttrs,
) -> tuple[str, ...]:
    """Уверенный кадр: бонус соседу не уводит ответ к карточке, которая спорит с этикеткой.

    Правило P1 (`research/2026-09-25_acc/PREREG_E1_rules.md`); `ranked` — порядок модели. Если
    её лидер — не лидер без бонуса соседу, а прочитанные сахар или цвет спорят с лидером модели и
    не спорят с лидером без бонуса, ответом становится лидер без бонуса, остальные — в порядке
    модели. Иначе порядок модели не меняется. При уверенности модели бонус работает как монетка
    (на catalog_v2 11 верных ответов с ним против 12 без него), а спор с прочитанным — довод против
    переворота. Параметров нет: порог уверенности тот же, что у H5. Замер Э1 (25.09, стенд по
    записанным прогонам): catalog_v2 +6/−0, krasnostop (617) +1/−0, срез R не тронут.
    """
    if not ranked or fields is None:
        return tuple(ranked)
    leader = ranked[0]
    plain = without_bonus(model, features)
    if not plain or plain[0] == leader:
        return tuple(ranked)
    if contradicts(leader, fields, attrs) and not contradicts(plain[0], fields, attrs):
        return (plain[0], *(slug for slug in ranked if slug != plain[0]))
    return tuple(ranked)
