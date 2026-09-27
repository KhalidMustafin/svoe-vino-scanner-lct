"""Игрушечный каталог для тестов resolve: восемь позиций, три винодельни, две пары двойников.

Записи по форме те же, что в `gt_tokens.jsonl`, но маленькие: тесту нужно проверить
правила решения, а не разметку. Двойники подобраны так, чтобы их различал ровно один
признак этикетки:

    cluster 1 — «Алиготе Баррель» 2023 и 2024: различает год;
    cluster 2 — «Резерв Брют» и «Полусладкое»: различают серия и сахар.

`tretya-pino-nuar` — винодельня, у которой единственное название условное («Пино Нуар»):
на ней проверяется запрет отказа из правила S10.

Модуль не собирается pytest (имя не начинается с `test_`) и живёт рядом с `fakes.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.resolve.attrs import CatalogAttrs


def wine_record(
    slug: str,
    *,
    winery: str = "Тестовая Долина",
    key_tokens: Sequence[str] = ("тестовая", "долина"),
    variants: Sequence[str] = ("testovaya dolina",),
    brands: Sequence[str] = (),
    name: str = "",
    cuvee: Sequence[str] = (),
    grapes: Sequence[str] = (),
    grape_values: Sequence[str] = (),
    sugar: str | None = None,
    year: int | None = None,
    abv: Sequence[float] = (),
    serial_tokens: Sequence[str] = (),
    serial_keywords: Sequence[str] = (),
    color: str | None = None,
    cluster: int | None = None,
    group: Sequence[int] | None = None,
    mates: Sequence[str] = (),
    flags: Sequence[str] = (),
) -> dict[str, Any]:
    """Запись каталога в форме `gt_tokens.jsonl`; всё, что не задано, остаётся пустым."""
    return {
        "slug": slug,
        "name": name,
        "winery": winery,
        "category": color,
        "fields": {
            "winery": {
                "key_tokens": list(key_tokens),
                "variants": list(variants),
                "brands": list(brands),
            },
            "cuvee": {"tokens": list(cuvee), "variants": []},
            "grape": {"values": list(grape_values), "codes": list(grapes), "variants": []},
            "sugar": {"class": sugar},
            "year": {"value": year},
            "abv": {"value": list(abv)},
            "serial": {"tokens": list(serial_tokens), "keywords": list(serial_keywords)},
            "color": {"class": color},
        },
        "cluster_B": cluster,
        "visual_group": list(group) if group else None,
        "visual_mates": list(mates),
        "noise_flags": list(flags),
    }


OTHER = {
    "winery": "Другая Винодельня",
    "key_tokens": ("другая", "винодельня"),
    "variants": ("drugaya",),
}
THIRD = {"winery": "Третья Марка", "key_tokens": ("третья", "марка"), "variants": ("tretya",)}

TOY_RECORDS: list[dict[str, Any]] = [
    wine_record(
        "dolina-aligote-2023",
        name="Алиготе Баррель",
        cuvee=["баррель"],
        grapes=["aligote"],
        grape_values=["Алиготе"],
        sugar="suhoe",
        year=2023,
        abv=[12.5],
        color="Белое",
        cluster=1,
        group=[1, 0],
        mates=["dolina-aligote-2024"],
    ),
    wine_record(
        "dolina-aligote-2024",
        name="Алиготе Баррель",
        cuvee=["баррель"],
        grapes=["aligote"],
        grape_values=["Алиготе"],
        sugar="suhoe",
        year=2024,
        abv=[13.0],
        color="Белое",
        cluster=1,
        group=[1, 0],
        mates=["dolina-aligote-2023"],
    ),
    wine_record(
        "dolina-reserve-brut",
        name="Резерв Брют",
        grapes=["chardonnay"],
        grape_values=["Шардоне"],
        sugar="brut",
        abv=[12.0],
        serial_tokens=["XXIV"],
        serial_keywords=["reserve"],
        color="Белое",
        cluster=2,
        group=[2, 0],
        mates=["dolina-polusladkoe"],
    ),
    wine_record(
        "dolina-polusladkoe",
        name="Полусладкое",
        grapes=["chardonnay"],
        grape_values=["Шардоне"],
        sugar="polusladkoe",
        abv=[11.0],
        color="Белое",
        cluster=2,
        group=[2, 0],
        mates=["dolina-reserve-brut"],
        flags=["name_generic_only"],
    ),
    wine_record(
        "dolina-merlot",
        name="Мерло Терруар",
        cuvee=["терруар"],
        grapes=["merlot"],
        grape_values=["Мерло"],
        sugar="suhoe",
        abv=[13.5],
        color="Красное",
    ),
    wine_record(
        "drugaya-kaberne",
        **OTHER,
        name="Гранат",
        cuvee=["гранат"],
        grapes=["cabernet_sauvignon"],
        grape_values=["Каберне Совиньон"],
        sugar="suhoe",
        abv=[14.0],
        color="Красное",
    ),
    wine_record(
        "drugaya-rislling",
        **OTHER,
        name="Янтарь",
        cuvee=["янтарь"],
        grapes=["riesling"],
        grape_values=["Рислинг"],
        sugar="polusuhoe",
        abv=[11.5],
        color="Белое",
    ),
    wine_record(
        "tretya-pino-nuar",
        **THIRD,
        name="Пино Нуар",
        grapes=["pinot_noir"],
        grape_values=["Пино Нуар"],
        sugar="suhoe",
        abv=[13.0],
        color="Красное",
        flags=["name_grape_only"],
    ),
]

#: Все slug каталога в порядке записей.
TOY_SLUGS: tuple[str, ...] = tuple(record["slug"] for record in TOY_RECORDS)


def toy_attrs(records: Sequence[dict[str, Any]] | None = None) -> CatalogAttrs:
    """Таблица признаков игрушечного каталога (или своего набора записей)."""
    return CatalogAttrs.from_records(records if records is not None else TOY_RECORDS)
