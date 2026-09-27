"""Решение: кандидаты CV, переранжированные признаками этикетки.

CV первичен и отвечает на вопрос «какая это бутылка», текст — на вопрос «какая из
почти одинаковых». Поэтому здесь нет второго поиска по тексту: пул строится из top-K
кандидатов CV и их серий, а поля этикетки только двигают кандидатов внутри пула.

Серия (`cluster_B`) — группа двойников, которых модель зрения не различает: у них общая
этикетка и разные год, объём или сахар. Косинусы внутри серии почти равны, поэтому счёт
серии берётся максимумом по её членам, и все члены серии — даже те, кого CV не поднял в
top-K, — входят в пул с этим счётом. Иначе верный двойник, оказавшийся тридцатым по
косинусу, не имел бы шанса подняться по году с этикетки.

Счёт кандидата — линейная комбинация с явными весами:

    score = visual_weight * счёт серии + Σ (match | conflict) по семи признакам

Никакого обучения: все веса и пороги в `RerankConfig` выставлены руками и не подобраны на
данных (`WEIGHTS_NOTE`). Логистическая регрессия по плану появится, когда полевых кадров
станет больше трёх сотен.

Признак даёт три исхода: «совпало», «противоречит» и «нет чтения». Третий не штрафует
никого: пустое поле этикетки значит «не прочитано», а пустое поле каталога — «в каталоге
не заполнено» (год известен у 116 позиций из 2 103). Штраф ставится только там, где обе
стороны сказали своё и сказали разное.

Отказ (`slug=None`, `outcome="out_of_catalog"`) — правило S10 плана, три условия сразу:
винодельня прочитана уверенно; ни одна её позиция не сходится с прочитанным текстом по
названию, кюве, сорту или серии; лучший счёт серии по CV ниже порога. У виноделен, где все
названия условные («Пино Нуар», «Белое сухое»), отказ запрещён: по такому названию нельзя
судить, что этикетка чужая. Режим `cfg.abstain="off"` выключает правило целиком.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.features.contracts import VisualResult
from app.reading.contracts import Evidence, LabelFields, SugarClass
from app.resolve.attrs import (
    CatalogAttrs,
    WineAttrs,
    abv_key,
    grape_key,
    norm_key,
    read_keys,
    serial_key,
    sugar_key,
    year_key,
)

#: Признаки согласия текста и карточки.
Feature = Literal["winery", "cuvee", "grape", "sugar", "year", "serial", "abv"]
FEATURES: tuple[Feature, ...] = get_args(Feature)

#: Исход признака: обе стороны сказали одно, сказали разное, или кто-то промолчал.
Agreement = Literal["match", "conflict", "unknown"]

#: Что решено: вино выбрано, выбрано неуверенно, ответа нет.
Outcome = Literal["matched", "ambiguous", "out_of_catalog"]

#: Режим отказа: правило S10 выключено или работает.
AbstainMode = Literal["off", "ooc_only"]

WEIGHTS_NOTE = (
    "веса и пороги не подобраны на данных: выставлены вручную по ТЗ и «Лозе», "
    "обучения нет; подбирать на dev-наборе полевых кадров"
)


class FeatureWeights(BaseModel):
    """Вклад признака в счёт: сколько прибавить за совпадение и отнять за противоречие."""

    model_config = ConfigDict(frozen=True)

    match: float = 0.0
    conflict: float = 0.0

    def delta(self, agreement: Agreement) -> float:
        return {"match": self.match, "conflict": self.conflict, "unknown": 0.0}[agreement]


class RerankConfig(BaseModel):
    """Все числа решения в одном месте. Ни одно из них не подобрано на данных.

    Веса заданы в шкале косинуса: совпавшая винодельня стоит 0,30 — больше, чем обычный
    разрыв между соседями в каталоге, но меньше, чем разрыв между разными бутылками.
    Противоречие винодельни (0,45) дороже любого совпадения: чужая винодельня — самый
    надёжный признак «не то вино».
    """

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=20, gt=0)  # сколько кандидатов CV берём в пул
    visual_weight: float = 1.0  # множитель счёта серии
    series_pool: bool = True  # добавлять в пул членов серии, которых CV не поднял

    winery: FeatureWeights = FeatureWeights(match=0.30, conflict=-0.45)
    cuvee: FeatureWeights = FeatureWeights(match=0.14, conflict=-0.04)
    grape: FeatureWeights = FeatureWeights(match=0.10, conflict=-0.12)
    sugar: FeatureWeights = FeatureWeights(match=0.08, conflict=-0.12)
    year: FeatureWeights = FeatureWeights(match=0.12, conflict=-0.20)
    serial: FeatureWeights = FeatureWeights(match=0.12, conflict=-0.06)
    abv: FeatureWeights = FeatureWeights(match=0.06, conflict=-0.08)

    abv_tolerance: float = 0.05  # 12,5 % против 12,5 % — то же вино
    ambiguous_margin: float = 0.03  # отрыв меньше — ответ помечается неуверенным

    abstain: AbstainMode = "off"
    winery_conf_min: float = 0.75  # уверенность чтения винодельни (1/(1+цена попадания))
    visual_floor: float = 0.75  # ниже этого счёта серии CV сам себе не верит

    def weights(self, feature: Feature) -> FeatureWeights:
        return getattr(self, feature)  # type: ignore[no-any-return]


DEFAULT_CONFIG = RerankConfig()


class ResolveResult(BaseModel):
    """Ответ слоя resolve.

    `slug=None` — ответа нет: либо отказ по правилу S10, либо CV не дал ни одного кандидата.
    `top5` остаётся заполненным и при отказе: это «может быть, вы искали» для карточки,
    а не ответ. `score` — счёт ранжирования, не вероятность: он складывается из косинуса и
    весов, поэтому бывает и больше единицы, и отрицательным.
    """

    slug: str | None = None
    score: float = 0.0
    margin: float = 0.0  # отрыв top-1 от top-2 после переранжирования
    series_margin: float = 0.0  # отрыв лучшей серии от второй по CV
    top5: list[tuple[str, float]] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    outcome: Outcome = "out_of_catalog"


# ------------------------------------------------------------------ прочитанное
@dataclass(frozen=True, slots=True)
class ReadKeys:
    """Поля этикетки в ключах сравнения — ровно то, что сопоставляется с карточкой.

    Множества пусты, когда поле не прочитано. `winery_slugs` — позиции каталога, которым
    прочитанная винодельня подходит: незнакомая каталогу винодельня не противоречит никому.
    """

    winery: frozenset[str]
    winery_conf: float
    winery_slugs: frozenset[str]
    cuvee: frozenset[str]
    grapes: frozenset[str]
    sugar: frozenset[SugarClass]
    year: int | None
    serial: frozenset[str]
    abv: float | None

    @property
    def text(self) -> frozenset[str]:
        """Слова, которыми этикетка отличает позицию: кюве, сорт, серия."""
        return self.cuvee | self.grapes | self.serial

    @property
    def empty(self) -> bool:
        return not (
            self.winery or self.text or self.sugar or self.year is not None or self.abv is not None
        )


def _values[T](evidences: Sequence[Evidence[T]]) -> list[T]:
    return [e.value for e in evidences]


def read_fields(fields: LabelFields | None, attrs: CatalogAttrs) -> ReadKeys:
    """`LabelFields` → ключи сравнения. `None` и пустые поля дают пустые ключи."""
    fields = fields or LabelFields()
    winery = read_keys(_values(fields.producer), norm_key)
    slugs: set[str] = set()
    for value in winery:
        slugs.update(attrs.slugs_of_winery(value))
    conf = max((e.conf or 0.0 for e in fields.producer), default=0.0)
    return ReadKeys(
        winery=winery,
        winery_conf=conf,
        winery_slugs=frozenset(slugs),
        cuvee=read_keys(_values(fields.cuvee), norm_key),
        grapes=read_keys(_values(fields.grapes), grape_key),
        sugar=frozenset(
            key for value in _values(fields.sugar) if (key := sugar_key(value)) is not None
        ),
        year=year_key(fields.vintage.value) if fields.vintage else None,
        serial=read_keys(_values(fields.serial), serial_key),
        abv=abv_key(fields.abv.value) if fields.abv else None,
    )


# ------------------------------------------------------------------ признаки согласия
def _sets(read: frozenset[Any], card: frozenset[Any]) -> Agreement:
    """Общее правило множеств: пересеклись — совпало, обе непусты и не пересеклись — спор."""
    if read & card:
        return "match"
    if read and card:
        return "conflict"
    return "unknown"


def agreement(
    feature: Feature, wine: WineAttrs | None, read: ReadKeys, *, abv_tolerance: float = 0.05
) -> Agreement:
    """Согласие одного признака. Карточки нет в каталоге — «нет чтения», а не спор."""
    if wine is None:
        return "unknown"
    if feature == "winery":
        # Спор — только когда прочитанная винодельня известна каталогу: позиция не её.
        if read.winery & wine.winery_keys:
            return "match"
        return "conflict" if read.winery_slugs and wine.slug not in read.winery_slugs else "unknown"
    if feature == "cuvee":
        return _sets(read.cuvee, wine.cuvee)
    if feature == "grape":
        return _sets(read.grapes, wine.grapes)
    if feature == "sugar":
        return _sets(read.sugar, frozenset({wine.sugar} if wine.sugar else set()))
    if feature == "year":
        # Мягко: у 1 987 позиций из 2 103 года в каталоге нет, и это не повод штрафовать.
        if read.year is None or wine.year is None:
            return "unknown"
        return "match" if read.year == wine.year else "conflict"
    if feature == "serial":
        return _sets(read.serial, wine.serial)
    if read.abv is None or not wine.abv:
        return "unknown"
    return "match" if _abv_match(read.abv, wine.abv, abv_tolerance) else "conflict"


def _abv_match(read: float, card: Sequence[float], tolerance: float) -> bool:
    return any(abs(read - value) <= tolerance for value in card)


def _card_value(feature: Feature, wine: WineAttrs | None) -> Any:
    if wine is None:
        return None
    values: dict[Feature, Any] = {
        "winery": wine.winery,
        "cuvee": sorted(wine.cuvee),
        "grape": sorted(wine.grapes),
        "sugar": str(wine.sugar) if wine.sugar else None,
        "year": wine.year,
        "serial": sorted(wine.serial),
        "abv": list(wine.abv),
    }
    return values[feature]


def _read_value(feature: Feature, read: ReadKeys) -> Any:
    values: dict[Feature, Any] = {
        "winery": sorted(read.winery),
        "cuvee": sorted(read.cuvee),
        "grape": sorted(read.grapes),
        "sugar": sorted(str(value) for value in read.sugar),
        "year": read.year,
        "serial": sorted(read.serial),
        "abv": read.abv,
    }
    return values[feature]


def score_wine(
    wine: WineAttrs | None, read: ReadKeys, visual: float, cfg: RerankConfig
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Счёт одной позиции и разбор по признакам: что совпало, что поспорило и почём."""
    total = cfg.visual_weight * visual
    evidence: dict[str, dict[str, Any]] = {}
    for feature in FEATURES:
        agree = agreement(feature, wine, read, abv_tolerance=cfg.abv_tolerance)
        delta = cfg.weights(feature).delta(agree)
        total += delta
        evidence[feature] = {
            "agree": agree,
            "read": _read_value(feature, read),
            "card": _card_value(feature, wine),
            "delta": round(delta, 4),
        }
    return total, evidence


# ------------------------------------------------------------------ пул и ранжирование
@dataclass(frozen=True, slots=True)
class _Row:
    """Позиция пула со счётом: всё, что нужно для сортировки и отчёта."""

    slug: str
    series: str
    score: float
    visual: float
    cv_score: float | None  # None — CV эту позицию не поднял, она пришла из серии
    year: int
    features: dict[str, dict[str, Any]]

    @property
    def sort_key(self) -> tuple[float, float, int, str]:
        """Детерминированный порядок: счёт, собственный косинус, свежий год, slug.

        Тай-брейк S11 («текст названия → последний год → шаблонный slug → свежий lastmod»)
        здесь только в части года: остального у слоя ещё нет.
        """
        return (-self.score, -(self.cv_score or 0.0), -self.year, self.slug)


def _series_scores(
    visual: VisualResult, attrs: CatalogAttrs, top_k: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Косинусы по slug и счёт серии — максимум по её членам среди кандидатов."""
    cv: dict[str, float] = {}
    series: dict[str, float] = {}
    for candidate in visual.candidates[:top_k]:
        if candidate.slug in cv:
            continue  # один slug дважды в выдаче: остаётся лучший ранг
        cv[candidate.slug] = candidate.score
        key = attrs.series_of(candidate.slug)
        series[key] = max(series.get(key, 0.0), candidate.score)
    return cv, series


def _pool(
    cv: dict[str, float], series: dict[str, float], attrs: CatalogAttrs, *, members: bool
) -> dict[str, str]:
    """Позиции для переранжирования: кандидаты CV и, если разрешено, все члены их серий."""
    pool: dict[str, str] = {}
    for key in series:
        for slug in attrs.members(key) if members else ():
            pool.setdefault(slug, key)
    for slug in cv:
        pool.setdefault(slug, attrs.series_of(slug))
    return pool


def _abstain_check(
    read: ReadKeys, attrs: CatalogAttrs, best_visual: float, cfg: RerankConfig
) -> dict[str, Any]:
    """Три условия отказа S10 по отдельности — чтобы в отчёте было видно, какое не сошлось."""
    confident = bool(read.winery) and read.winery_conf >= cfg.winery_conf_min
    slugs = sorted(read.winery_slugs)
    known = confident and bool(slugs)
    wines = [wine for slug in slugs if (wine := attrs.get(slug))]
    # Ноль пересечений — только если этикетка вообще дала слова: по одной винодельне
    # сказать «этого вина нет в каталоге» нельзя.
    no_match = bool(read.text) and known and not any(wine.text_keys & read.text for wine in wines)
    # Винодельня, у которой все названия условные: «Пино Нуар», «Белое сухое», одно слово.
    conditional = bool(wines) and all(wine.conditional_name for wine in wines)
    below_floor = best_visual < cfg.visual_floor
    fired = cfg.abstain == "ooc_only" and known and no_match and below_floor and not conditional
    return {
        "mode": cfg.abstain,
        "winery_confident": known,
        "winery_conf": round(read.winery_conf, 3),
        "winery_positions": len(slugs),
        "no_position_matches": no_match,
        "conditional_names": conditional,
        "best_visual": round(best_visual, 4),
        "below_floor": below_floor,
        "fired": fired,
    }


def abstain_check(
    visual: VisualResult,
    fields: LabelFields | None,
    attrs: CatalogAttrs,
    *,
    cfg: RerankConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Правило отказа S10 само по себе, без ручного ранжирования.

    Нужно сервису с обучаемым resolve: вино выбирает модель, а «этого вина нет в каталоге»
    решает то же правило, что и в `rerank`, с теми же порогами `cfg` и лучшим счётом серии по
    top-K CV. Сработать правило может только при `cfg.abstain="ooc_only"`: при `off`
    возвращаются те же три условия с `fired=False`.
    """
    _, series = _series_scores(visual, attrs, cfg.top_k)
    best_visual = max(series.values(), default=0.0)
    return _abstain_check(read_fields(fields, attrs), attrs, best_visual, cfg)


def _runner_up(row: _Row) -> dict[str, Any]:
    return {
        "slug": row.slug,
        "score": round(row.score, 4),
        "match": [name for name, item in row.features.items() if item["agree"] == "match"],
        "conflict": [name for name, item in row.features.items() if item["agree"] == "conflict"],
    }


def rerank(
    visual: VisualResult,
    fields: LabelFields | None,
    attrs: CatalogAttrs,
    *,
    cfg: RerankConfig = DEFAULT_CONFIG,
) -> ResolveResult:
    """Выбрать вино: кандидаты CV, поднятые и опущенные признаками этикетки.

    Функция чистая и детерминированная: одни и те же вход и `cfg` дают тот же ответ,
    включая порядок `top5`. Пустые поля этикетки — это просто ноль слагаемых: порядок
    тогда остаётся визуальным.
    """
    read = read_fields(fields, attrs)
    cv, series = _series_scores(visual, attrs, cfg.top_k)
    ranked_series = sorted(series.values(), reverse=True)
    best_visual = ranked_series[0] if ranked_series else 0.0
    series_margin = round(ranked_series[0] - ranked_series[1], 6) if len(ranked_series) > 1 else 0.0
    abstain = _abstain_check(read, attrs, best_visual, cfg)

    rows = []
    for slug, key in _pool(cv, series, attrs, members=cfg.series_pool).items():
        wine = attrs.get(slug)
        score, features = score_wine(wine, read, series[key], cfg)
        rows.append(
            _Row(
                slug=slug,
                series=key,
                score=score,
                visual=series[key],
                cv_score=cv.get(slug),
                year=(wine.year or 0) if wine else 0,
                features=features,
            )
        )
    rows.sort(key=lambda row: row.sort_key)

    top = visual.top
    evidence: dict[str, Any] = {
        "visual": {
            "top1": top.slug if top else None,
            "top1_score": round(top.score, 4) if top else None,
            "margin": round(visual.margin, 6),
            "candidates": len(visual.candidates),
            "model": visual.model,
        },
        "pool": len(rows),
        "series_count": len(series),
        "text_empty": read.empty,
        "abstain": abstain,
        "weights_note": WEIGHTS_NOTE,
    }
    if not rows:
        evidence["reason"] = "CV не дал ни одного кандидата"
        return ResolveResult(evidence=evidence, outcome="out_of_catalog")

    best = rows[0]
    margin = round(best.score - rows[1].score, 6) if len(rows) > 1 else 0.0
    evidence["series"] = {
        "id": best.series,
        "size": len(attrs.members(best.series)) or 1,
        "cv_score": round(best.visual, 4),
        "is_cv_best": best.visual >= best_visual,
        "from_series_pool": best.cv_score is None,
    }
    evidence["features"] = best.features
    evidence["runners_up"] = [_runner_up(row) for row in rows[1:4]]
    outcome: Outcome = "matched"
    if abstain["fired"]:
        outcome = "out_of_catalog"
    elif len(rows) > 1 and margin < cfg.ambiguous_margin:
        # Единственный кандидат неуверенным не считается: путать его не с кем.
        outcome = "ambiguous"
    return ResolveResult(
        slug=None if outcome == "out_of_catalog" else best.slug,
        score=round(best.score, 6),
        margin=margin,
        series_margin=series_margin,
        top5=[(row.slug, round(row.score, 4)) for row in rows[:5]],
        evidence=evidence,
        outcome=outcome,
    )
