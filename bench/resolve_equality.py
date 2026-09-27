"""Равенство «офлайн = сервис»: `ScannerService._resolve` на входах записанных прогонов.

Страховка от ловушки стенда, где slug в дампе — ответ до правил выбора: сервис должен
принимать ровно те решения, какие посчитаны офлайн. Офлайн-ответ здесь считается отдельно от
кода решений сервиса (`app.resolve.ambiguous`, `ScannerService._resolve`):

    счёт модели         numpy по весам модели: z = (x − mean) / scale, s = z·w + b;
                        порядок — по убыванию s, при равенстве — лучший ранг CV
    p лидера            softmax(s / T) у лидера
    без бонуса соседу   тот же счёт с нулевым весом `cluster_mate_of_top1`
    спор с этикеткой    прочитанный сахар (любой из прочитанных) или цвет ≠ классу карточки;
                        класса у карточки нет или ничего не прочитано — не спор; цвета
                        «Белое» и «Оранжевое» друг с другом не спорят (Э4)
    H5  (p < 0,5)       первый без спора в порядке без бонуса; спорят все — первый в нём
    P1  (p ≥ 0,5)       лидер модели ≠ лидеру без бонуса, лидер спорит, а лидер без бонуса
                        нет — ответ лидер без бонуса; иначе — лидер модели

Общие с сервисом только входы: признаки кандидатов (`query_features`), модель, разметка
каталога. Прогоны — стенд `iterbench.py` вне репозитория (`runs/field25/iters/runs/<прогон>/iter20`): 353 кадра
catalog_v2 (`pfix_final`) и 617 основных кадров krasnostop (`kr_holdout`, без `same_packshot`).

    python -m bench.resolve_equality [--runs DIR] [--field DIR] [--gt-tokens FILE] [--model FILE]

Модель по умолчанию — модель сервиса по умолчанию (`-lw-pool`); прежний путь —
`--model configs/resolve/s2so400m-vlm35-goal.json`. Выдача CV прогонов посчитана без адаптера
поиска: сверяется решение слоя выбора на одних и тех же входах, а не точность модели на них.

По умолчанию пути — от `SVS_DATA_DIR`: прогоны в `<data>/../runs/field25/iters/runs`, наборы в
`<data>/../../field_dataset` (`SVS_RUNS_DIR`, `SVS_FIELD_DATASET_DIR` их перекрывают). Код
выхода 1 — есть расхождения.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.api.config import DEFAULT_RESOLVE_MODEL, ServiceSettings
from app.api.service import ScannerService, _Run, load_catalog, load_resolve_model
from app.features.contracts import IndexMeta, VisualResult
from app.features.index import VisualIndex
from app.reading.contracts import LabelFields
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import TextRead, query_features, readers_of

#: Прогон → (набор field_dataset, брать ли кадры `same_packshot`).
RUNS: dict[str, tuple[str, bool]] = {
    "pfix_final": ("catalog_v2", True),
    "kr_holdout": ("krasnostop_v1", False),
}
CLUSTER_FEATURE = "cluster_mate_of_top1"
AMBIGUOUS_P_TOP1 = 0.5
#: Разные цвета, которые не спорят: оранжевые вина маркируются «белое» (Э4, 25.09).
WHITE_ORANGE = frozenset({"Белое", "Оранжевое"})


@dataclass(frozen=True)
class Paths:
    runs: Path
    field: Path
    gt_tokens: Path
    model: Path

    def ready(self) -> bool:
        return self.gt_tokens.is_file() and all(
            (self.runs / run / "iter20" / "predictions.jsonl").is_file()
            and (self.field / "sets" / name / "meta.jsonl").is_file()
            for run, (name, _) in RUNS.items()
        )


def default_paths(env: Mapping[str, str] | None = None) -> Paths | None:
    """Пути от окружения; без `SVS_DATA_DIR` и явных путей — None."""
    env = os.environ if env is None else env
    data = Path(env["SVS_DATA_DIR"]) if env.get("SVS_DATA_DIR") else None
    runs = env.get("SVS_RUNS_DIR") or (data and data.parent / "runs" / "field25" / "iters" / "runs")
    field = env.get("SVS_FIELD_DATASET_DIR") or (data and data.parent.parent / "field_dataset")
    if not (data and runs and field):
        return None
    return Paths(Path(runs), Path(field), data / "gt" / "gt_tokens.jsonl", DEFAULT_RESOLVE_MODEL)


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


# ------------------------------------------------------------------ офлайн
@dataclass(frozen=True)
class Weights:
    """Веса модели выбора как массивы — без `LogisticRanker`."""

    names: tuple[str, ...]
    coef: np.ndarray
    coef_plain: np.ndarray  # вес бонуса соседу обнулён
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    temperature: float
    top_k: int

    @classmethod
    def load(cls, path: Path) -> Weights:
        raw = json.loads(path.read_text(encoding="utf-8"))
        names = tuple(raw["feature_names"])
        coef = np.asarray(raw["coef"], dtype=np.float64)
        plain = coef.copy()
        if CLUSTER_FEATURE in names:
            plain[names.index(CLUSTER_FEATURE)] = 0.0
        return cls(
            names=names,
            coef=coef,
            coef_plain=plain,
            mean=np.asarray(raw["mean"], dtype=np.float64),
            scale=np.asarray(raw["scale"], dtype=np.float64),
            intercept=float(raw["intercept"]),
            temperature=float(raw["temperature"]),
            top_k=int((raw.get("meta") or {}).get("top_k") or 20),
        )


def _order(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def _conflict(slug: str, fields: LabelFields | None, attrs: CatalogAttrs) -> bool:
    if fields is None:
        return False
    wine = attrs.get(slug)
    if wine is None:
        return False
    read_sugar = {str(item.value) for item in fields.sugar}
    card_sugar = wine.sugar.value if wine.sugar is not None else None
    if read_sugar and card_sugar and card_sugar not in read_sugar:
        return True
    card_color = wine.color.value if wine.color is not None else None
    read_color = str(fields.color.value) if fields.color is not None else None
    if not (card_color and read_color) or card_color == read_color:
        return False
    return {card_color, read_color} != WHITE_ORANGE


def offline_answer(
    weights: Weights,
    visual: VisualResult,
    read: TextRead,
    attrs: CatalogAttrs,
    reader: str,
) -> str:
    fields = read.fields
    features = query_features(visual, {reader: read}, attrs, top_k=weights.top_k)
    z = (features.matrix(list(weights.names)) - weights.mean) / weights.scale
    scores = z @ weights.coef + weights.intercept
    order = _order(scores)
    ranked = [features.slugs[i] for i in order]
    shifted = scores[order] / weights.temperature
    shifted = shifted - shifted.max()
    p_top1 = float(np.exp(shifted[0]) / np.exp(shifted).sum())
    plain = [features.slugs[i] for i in _order(z @ weights.coef_plain + weights.intercept)]
    if p_top1 < AMBIGUOUS_P_TOP1:
        return next((slug for slug in plain if not _conflict(slug, fields, attrs)), plain[0])
    leader = ranked[0]
    if (
        leader != plain[0]
        and _conflict(leader, fields, attrs)
        and not _conflict(plain[0], fields, attrs)
    ):
        return plain[0]
    return leader


# ------------------------------------------------------------------ сервис
class _NoEmbedder:
    """Эмбеддер-заглушка: `_resolve` картинку не считает, вектор CV берётся из прогона."""

    dim = 1

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        raise RuntimeError("равенство resolve: CV не пересчитывается")


def resolve_only_service(model_path: Path, attrs: CatalogAttrs) -> ScannerService:
    """Сервис, у которого настоящие только модель выбора и разметка каталога.

    Поиска здесь нет — выдача CV берётся из прогона, поэтому сервис собирается путём `off`, а
    запись о карте адаптера (`meta.cv_adapter_sha1` у `-lw-pool`) снимается: сверяется слой выбора
    (веса, H5, P1) на готовых входах, а не то, какой поиск эти входы дал.
    """
    settings = ServiceSettings(resolve_model=model_path, candidate="off", cv_adapter=None)
    meta = IndexMeta(model=settings.cv_model, dim=1, views=["bottle"], n_slugs=1, n_vectors=1)
    index = VisualIndex(["__equality__"], ["bottle"], np.ones((1, 1), dtype=np.float32), meta)
    model = load_resolve_model(model_path)
    model.meta.pop("cv_adapter_sha1", None)
    return ScannerService(
        settings,
        index=index,
        embedder=_NoEmbedder(settings.cv_model),
        lexicon=None,
        attrs=attrs,
        model=model,
        vlm=None,
    )


# ------------------------------------------------------------------ сверка
def frames(paths: Paths, run: str) -> list[dict[str, Any]]:
    name, with_same = RUNS[run]
    meta = {
        m["query_id"]: m
        for m in _jsonl(paths.field / "sets" / name / "meta.jsonl")
        if with_same or not m.get("same_packshot")
    }
    return [
        rec
        for rec in _jsonl(paths.runs / run / "iter20" / "predictions.jsonl")
        if rec["query_id"] in meta
    ]


def read_of(rec: Mapping[str, Any]) -> TextRead:
    tr = rec.get("text_read") or {}
    fields = LabelFields.model_validate(tr["fields"]) if tr.get("fields") else None
    return TextRead(fields=fields, raw_text=tr.get("raw_text"))


def check(paths: Paths, *, runs: Sequence[str] = tuple(RUNS)) -> dict[str, Any]:
    """Сколько кадров каждого прогона сервис решает так же, как офлайн; расхождения — списком."""
    _, attrs = load_catalog(paths.gt_tokens)
    service = resolve_only_service(paths.model, attrs)
    weights = Weights.load(paths.model)
    reader = readers_of(list(weights.names))[0]
    assert service.reader_key == reader
    out: dict[str, Any] = {}
    for run in runs:
        rows = frames(paths, run)
        differ = []
        for rec in rows:
            visual = VisualResult.model_validate(rec["visual"])
            read = read_of(rec)
            now = time.perf_counter()
            clock = _Run(clock=time.perf_counter, started=now, deadline=now + 3600.0)
            got = service._resolve(visual, {reader: read}, read.fields, clock)["slug"]
            want = offline_answer(weights, visual, read, attrs, reader)
            if got != want:
                differ.append({"query_id": rec["query_id"], "service": got, "offline": want})
        out[run] = {"frames": len(rows), "equal": len(rows) - len(differ), "differ": differ}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--field", type=Path)
    ap.add_argument("--gt-tokens", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_RESOLVE_MODEL)
    args = ap.parse_args(argv)
    found = default_paths()
    runs = args.runs or (found.runs if found else None)
    field = args.field or (found.field if found else None)
    gt = args.gt_tokens or (found.gt_tokens if found else None)
    if not (runs and field and gt):
        ap.error("нужны --runs, --field и --gt-tokens или SVS_DATA_DIR")
    paths = Paths(runs, field, gt, args.model)
    if not paths.ready():
        ap.error(f"нет прогонов или наборов: {paths}")
    report = check(paths)
    for run, res in report.items():
        print(f"{run}: {res['equal']}/{res['frames']} кадров — сервис = офлайн")
        for row in res["differ"][:20]:
            print("   ", json.dumps(row, ensure_ascii=False))
    return 0 if all(not res["differ"] for res in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
