"""Синтетика для обучаемого resolve: каталог из десяти позиций с двойниками и кадры к нему.

Каталог — три винодельни; у первой две пары соседних позиций линейки, на которых ломался
ручной rerank («Мускат» — «Мускат Чёрный», «Южная Вертикаль» — «… Премиум»), и два года
одной этикетки в одной визуальной группе.

Кадры строятся так, чтобы правило «уверенная картинка важнее ошибочно прочитанной
винодельни» было выучиваемым и проверяемым:

    confident   верный slug — CV-top1 с отрывом 0,06–0,12; винодельня прочитана верно в 70 %
                случаев, иначе — чужая винодельня одного из кандидатов;
    ambiguous   верный slug и позиция чужой винодельни идут вплотную (0,80–0,82) в случайном
                порядке — картинка права в половине случаев; винодельня прочитана верно в 90 %.

Модуль не собирается pytest (имя не начинается с `test_`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from catalog import wine_record

from app.reading.contracts import Evidence, LabelFields
from app.resolve.attrs import CatalogAttrs

ALFA = {"winery": "Альфа Долина", "key_tokens": ("альфа", "долина"), "variants": ("alfa dolina",)}
BETA = {"winery": "Бета Холмы", "key_tokens": ("бета", "холмы"), "variants": ("beta kholmy",)}
GAMMA = {"winery": "Гамма Берег", "key_tokens": ("гамма", "берег"), "variants": ("gamma bereg",)}
WINERIES = {"alfa": ALFA, "beta": BETA, "gamma": GAMMA}

SYNTH_RECORDS: list[dict[str, Any]] = [
    wine_record(
        "alfa-muskat", **ALFA, name="Мускат", grapes=["muscat"], grape_values=["Мускат"],
        sugar="suhoe", color="Белое", cluster=1,
    ),
    wine_record(
        "alfa-muskat-chernyj", **ALFA, name="Мускат Черный", grapes=["muscat"],
        grape_values=["Мускат"], sugar="sladkoe", color="Красное", cluster=1,
    ),
    wine_record(
        "alfa-yuzhnaya", **ALFA, name="Южная Вертикаль Каберне Фран",
        cuvee=["южная", "вертикаль"], grapes=["cabernet_franc"], sugar="suhoe", color="Красное",
        cluster=2,
    ),
    wine_record(
        "alfa-yuzhnaya-premium", **ALFA, name="Южная Вертикаль Каберне Фран Премиум",
        cuvee=["южная", "вертикаль"], grapes=["cabernet_franc"], sugar="suhoe", color="Красное",
        serial_keywords=["премиум"], cluster=2,
    ),
    wine_record(
        "alfa-riesling-2023", **ALFA, name="Рислинг", grapes=["riesling"], year=2023,
        color="Белое", cluster=3, group=[3, 0], mates=["alfa-riesling-2024"],
    ),
    wine_record(
        "alfa-riesling-2024", **ALFA, name="Рислинг", grapes=["riesling"], year=2024,
        color="Белое", cluster=3, group=[3, 0], mates=["alfa-riesling-2023"],
    ),
    wine_record(
        "beta-merlot", **BETA, name="Мерло Терруар", cuvee=["терруар"], grapes=["merlot"],
        color="Красное", abv=[12.5],
    ),
    wine_record(
        "beta-shardone", **BETA, name="Шардоне Резерв", grapes=["chardonnay"],
        serial_keywords=["reserve"], color="Белое",
    ),
    wine_record(
        "gamma-saperavi", **GAMMA, name="Саперави", grapes=["saperavi"], color="Красное",
        abv=[13.5],
    ),
    wine_record(
        "gamma-rkatsiteli", **GAMMA, name="Ркацители Янтарь", cuvee=["янтарь"],
        grapes=["rkatsiteli"], color="Белое",
    ),
]  # fmt: skip
SYNTH_SLUGS: tuple[str, ...] = tuple(record["slug"] for record in SYNTH_RECORDS)
WINERY_OF: dict[str, str] = {record["slug"]: record["winery"] for record in SYNTH_RECORDS}


def synth_attrs() -> CatalogAttrs:
    return CatalogAttrs.from_records(SYNTH_RECORDS)


def write_gt(path: Path, records: Sequence[dict[str, Any]] = SYNTH_RECORDS) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps({**record, "published": True}, ensure_ascii=False) + "\n")
    return path


def cv_record(
    query_id: str,
    slug: str | None,
    ranked: Sequence[tuple[str, float]],
    *,
    set_name: str = "synthetic",
    split: str | None = "dev",
) -> dict[str, Any]:
    """Запись `predictions.jsonl` стенда поиска: кандидаты уже по убыванию счёта."""
    meta: dict[str, Any] = {"set": set_name}
    if split is not None:
        meta["split"] = split
    top = [
        {"slug": s, "score": round(score, 6), "view": "full", "rank": rank}
        for rank, (s, score) in enumerate(ranked, start=1)
    ]
    return {
        "query_id": query_id,
        "slug": slug,
        "meta": meta,
        "status": "ok",
        "top": top,
        "margin": round(ranked[0][1] - ranked[1][1], 6) if len(ranked) > 1 else 0.0,
        "hit_rank": next((c["rank"] for c in top if c["slug"] == slug), None),
    }


def label_fields(winery: str | None, grape: str | None = None) -> LabelFields:
    return LabelFields(
        producer=[Evidence[str](value=winery, conf=1.0)] if winery else [],
        grapes=[Evidence[str](value=grape, conf=1.0)] if grape else [],
    )


def ocr_record(query_id: str, slug: str | None, fields: LabelFields, raw: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "slug": slug,
        "status": "ok",
        "raw_text": raw,
        "label_fields": fields.model_dump(mode="json"),
        "fields": {},
    }


def synth_queries(
    n: int, *, seed: int, set_name: str = "synthetic", split: str | None = "dev", prefix: str = "q"
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Кадры: (запись CV, запись OCR, сценарий `confident`/`ambiguous` + `:misread`)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        target = SYNTH_SLUGS[int(rng.integers(len(SYNTH_SLUGS)))]
        others = [s for s in SYNTH_SLUGS if s != target]
        foreign = [s for s in others if WINERY_OF[s] != WINERY_OF[target]]
        scores = {s: float(rng.uniform(0.55, 0.70)) for s in others}
        confident = bool(rng.random() < 0.5)
        if confident:
            second = max(scores.values())
            scores[target] = second + float(rng.uniform(0.06, 0.12))
            misread = bool(rng.random() < 0.3)
        else:
            rival = foreign[int(rng.integers(len(foreign)))]
            scores[target] = float(rng.uniform(0.80, 0.82))
            scores[rival] = float(rng.uniform(0.80, 0.82))
            misread = bool(rng.random() < 0.1)
        if misread:
            winery = WINERY_OF[foreign[int(rng.integers(len(foreign)))]]
        else:
            winery = WINERY_OF[target]
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        qid = f"{prefix}{i:03d}"
        fields = label_fields(winery)
        kind = ("confident" if confident else "ambiguous") + (":misread" if misread else "")
        out.append(
            (
                cv_record(qid, target, ranked, set_name=set_name, split=split),
                ocr_record(qid, target, fields, raw=winery.upper()),
                kind,
            )
        )
    return out


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
