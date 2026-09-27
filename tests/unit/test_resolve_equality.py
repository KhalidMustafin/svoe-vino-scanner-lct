"""Равенство «офлайн = сервис» (`bench.resolve_equality`) на записанных прогонах.

`ScannerService._resolve` на входах `pfix_final` (353 кадра catalog_v2) и `kr_holdout` (617
основных кадров krasnostop) должен дать ровно офлайн-ответы: H5 при p < 0,5 и P1 при p ≥ 0,5 —
и с моделью пути по умолчанию (`-lw-pool`), и с моделью пути `off` (`-goal`).
Нужны данные сервиса (`SVS_DATA_DIR`) и прогоны рядом с ними; без них тест пропускается.
Офлайн-сторона проверяется и сама: без P1 в сервисе расхождения обязаны появиться.
"""

from __future__ import annotations

import dataclasses

import pytest
from catalog import toy_attrs, wine_record

from app.api.config import CANDIDATE_RESOLVE_MODELS
from app.reading.contracts import Color, Evidence, LabelFields
from app.resolve.ambiguous import contradicts
from bench import resolve_equality

PATHS = resolve_equality.default_paths()
needs_runs = pytest.mark.skipif(
    PATHS is None or not PATHS.ready(),
    reason="нужны данные сервиса и прогоны: SVS_DATA_DIR (или SVS_RUNS_DIR, SVS_FIELD_DATASET_DIR)",
)


@needs_runs
@pytest.mark.parametrize("candidate", ["adapter-lw-ranker", "off"])
def test_service_resolve_equals_offline_on_recorded_runs(candidate):
    """Модель пути по умолчанию (`-lw-pool`) и прежнего пути (`-goal`): слой выбора сервиса =
    офлайн-ответ на тех же входах."""
    assert PATHS is not None
    model = CANDIDATE_RESOLVE_MODELS[candidate]
    assert (PATHS.model == model) is (candidate == "adapter-lw-ranker")
    report = resolve_equality.check(dataclasses.replace(PATHS, model=model))
    assert {run: (res["equal"], res["frames"]) for run, res in report.items()} == {
        "pfix_final": (353, 353),
        "kr_holdout": (617, 617),
    }


@needs_runs
def test_offline_side_catches_a_service_without_p1(monkeypatch):
    """Сервис без P1 расходится с офлайн-ответом: сверка не проходит сама собой.

    На модели `-goal` (путь `off`): у модели по умолчанию P1 на входах `pfix_final` не меняет ни
    одного ответа (ранкер на пуле выучил сахар и цвет сам, `PREREG_final.md`, §2.3), и сверка с
    ней отключённый P1 не видит — это записано вторым утверждением.
    """
    from app.api import service

    assert PATHS is not None
    monkeypatch.setattr(
        service, "block_bonus_flip", lambda model, features, ranked, fields, attrs: tuple(ranked)
    )
    goal = dataclasses.replace(PATHS, model=CANDIDATE_RESOLVE_MODELS["off"])
    report = resolve_equality.check(goal, runs=("pfix_final",))
    assert report["pfix_final"]["differ"]
    pool = resolve_equality.check(PATHS, runs=("pfix_final",))
    assert pool["pfix_final"]["differ"] == []


def test_paths_come_from_the_data_dir(tmp_path):
    data = tmp_path / "scanner" / "data"
    paths = resolve_equality.default_paths({"SVS_DATA_DIR": str(data)})
    assert paths is not None
    assert paths.runs == tmp_path / "scanner" / "runs" / "field25" / "iters" / "runs"
    assert paths.field == tmp_path / "field_dataset"
    assert paths.gt_tokens == data / "gt" / "gt_tokens.jsonl"
    assert not paths.ready()
    assert resolve_equality.default_paths({}) is None


def test_offline_conflict_knows_white_orange_on_its_own():
    """Офлайн-сторона держит совместимость Э4 своим кодом и совпадает с сервисом на всех парах."""
    colors = [color.value for color in Color]
    attrs = toy_attrs(
        [wine_record(f"w{i}", name=f"Вино {i}", color=c) for i, c in enumerate(colors)]
    )
    for i, card in enumerate(colors):
        for read in colors:
            fields = LabelFields(
                color=Evidence(value=Color(read), sources=("тест",), support=1, conf=1.0)
            )
            offline = resolve_equality._conflict(f"w{i}", fields, attrs)
            assert offline == contradicts(f"w{i}", fields, attrs)
            assert offline is (card != read and {card, read} != {"Белое", "Оранжевое"})
