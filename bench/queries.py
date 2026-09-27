"""Наборы запросов визуального стенда: какой кадр показываем индексу и какой slug ждём.

Эталоны индекса — packshot каталога, то есть фото карточек портала «Своё Вино». Значит,
меряться на них же бессмысленно: картинка найдёт сама себя. Поэтому основной набор —
честные пары «Лозы»: запрос берётся из другого источника.

    pairs        358 студийных снимков Роскачества; эталон — `portal_slug` каталога.
                 Фото портала и packshot каталога — одни и те же пиксели, поэтому
                 запросом служит снимок другого фотографа: другой свет, фон, ракурс,
                 иногда другой год на этикетке. Самопоиск исключён по построению.
    pairs_phone  те же пары, испорченные `phone_shot` (сид — номер записи в benchmark.json,
                 так что `--limit` не меняет ни одного кадра).
    public       три кадра организатора (`$SVS_DATASET_DIR/eval/queries`); в каталоге
                 только один из них — дымовой тест, не замер.
    rwl_src      10 исходных кадров russian_wine_labels_raw: 5 вин каталога, 5 вне его.
    rwl_aug      500 вариаций тех же кадров (по 50 на бутылку).

Наборы без каталожного slug (`slug=None`) — не мусор: на них видно, что канал отвечает,
когда вина в каталоге нет. Порогов отказа стенд не вводит, он только записывает счёт.

Своего разбиения dev/test у `pairs` нет: его проставляет стенд (`bench.retrieval.
assign_splits`) тем же ключом `cluster_B`, что и наборы OCR. Оговорки набора едут в
`metrics.json` — в том числе та, что геометрия видов выбрана на этих же 358 парах.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.config import REPO_ROOT, Settings, get_settings
from app.features.views import WINDOWS_NOTE
from bench.datasets import BenchItem, DatasetError, load_manifest, load_public

QuerySet = Literal["pairs", "pairs_phone", "public", "rwl_src", "rwl_aug"]
QUERY_SETS: tuple[QuerySet, ...] = get_args(QuerySet)

#: Репозиторий «Лозы» рядом с нашим: там собраны честные пары и снимки Роскачества.
LOZA_ROOT = REPO_ROOT.parent / "Code"
DEFAULT_PAIRS_JSON = LOZA_ROOT / "data" / "raw" / "pairs" / "benchmark.json"
DEFAULT_PAIRS_IMAGES = LOZA_ROOT / "data" / "raw" / "pairs" / "roskachestvo"

DEFAULT_RWL_DIR = REPO_ROOT / "runs" / "rwl"
DEFAULT_RWL_IMAGES = REPO_ROOT.parent / "russian_wine_labels_raw"

#: Оговорка к `pairs`, которая едет в metrics.json: почему замер честный.
PAIRS_NOTE = (
    "эталон каталога для portal_slug — то самое фото портала, что стоит в slug_photo_map; "
    "запрос поэтому взят из другого источника (студийный снимок Роскачества), и самопоиск "
    "исключён по построению набора"
)
#: Воронка разметки пар: как 580 привязок стали 358 замером и чем это смещает цифру.
PAIRS_FUNNEL_NOTE = (
    "воронка разметки: 580 привязок портала к Роскачеству → 82 отброшены правилами по "
    "цвету и игристости → 498 проверены глазами (Code/data/raw/pairs/manual_review.json) → "
    "358 с вердиктом «same» в замере. Вердикт выносился по двум фотографиям, поэтому верные "
    "пары, где два снимка выглядят слишком по-разному, могли уйти вместе с чужими: набор "
    "смещён в сторону лёгких запросов"
)
PHONE_NOTE = (
    "кадр портится phone_shot; сид — номер записи в benchmark.json, --limit его не сдвигает"
)


class QueryItem(BaseModel):
    """Один запрос стенда: кадр, ожидаемый slug и всё, что о нём известно.

    `slug=None` — вина нет в каталоге. `meta` уходит в предсказания как есть, поэтому там
    лежат только мелкие поля: имя набора, источник, сид порчи, split исходного набора.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    image_path: Path
    slug: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def phone_seed(self) -> int | None:
        """Сид порчи «как с телефона»; `None` — кадр идёт как есть."""
        seed = self.meta.get("phone_seed")
        return None if seed is None else int(seed)


def load_pairs(
    path: Path = DEFAULT_PAIRS_JSON,
    images_dir: Path = DEFAULT_PAIRS_IMAGES,
    *,
    phone: bool = False,
) -> list[QueryItem]:
    """Честные пары: `benchmark.json` со строками `{wine_id, portal_slug}`.

    Запрос — `{images_dir}/{wine_id}.webp`, эталон — `portal_slug` каталога. При
    `phone=True` каждому кадру приписывается сид порчи, равный номеру записи в файле.
    """
    if not path.is_file():
        raise DatasetError(f"нет файла пар: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{path}: не читается как JSON: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise DatasetError(f"{path}: ожидается непустой список записей")
    name = "pairs_phone" if phone else "pairs"
    items: list[QueryItem] = []
    seen: set[str] = set()
    for number, record in enumerate(records):
        if not isinstance(record, dict):
            raise DatasetError(f"{path}: запись {number} не объект")
        wine_id = str(record.get("wine_id") or "").strip()
        slug = str(record.get("portal_slug") or "").strip()
        if not wine_id or not slug:
            raise DatasetError(f"{path}: записи {number} не хватает wine_id или portal_slug")
        if wine_id in seen:
            raise DatasetError(f"{path}: повтор wine_id {wine_id}")
        seen.add(wine_id)
        meta: dict[str, Any] = {
            "set": name,
            "source": "roskachestvo",
            "wine_id": wine_id,
            "pair_index": number,
        }
        if phone:
            meta["phone_seed"] = number
        items.append(
            QueryItem(
                query_id=wine_id,
                image_path=images_dir / f"{wine_id}.webp",
                slug=slug,
                meta=meta,
            )
        )
    return items


def from_bench_items(items: Sequence[BenchItem], name: str) -> list[QueryItem]:
    """`BenchItem` стенда OCR → `QueryItem`: наборы у двух стендов общие, разметка не нужна."""
    return [
        QueryItem(
            query_id=item.query_id,
            image_path=item.image_path,
            slug=item.slug,
            meta={"set": name, "split": item.split, "in_catalog": item.in_catalog},
        )
        for item in items
    ]


def load_rwl(
    kind: Literal["src", "aug"],
    directory: Path = DEFAULT_RWL_DIR,
    images_dir: Path = DEFAULT_RWL_IMAGES,
) -> list[QueryItem]:
    """Манифесты набора russian_wine_labels_raw: `runs/rwl/{kind}_manifest.tsv` и эталон рядом."""
    items = load_manifest(
        directory / f"{kind}_manifest.tsv",
        directory / f"{kind}_gt.tsv",
        directory / f"{kind}_ann.jsonl",
        images_dir=images_dir,
    )
    return from_bench_items(items, f"rwl_{kind}")


def load_queries(
    name: str,
    *,
    settings: Settings | None = None,
    pairs_json: Path | None = None,
    pairs_images: Path | None = None,
    rwl_dir: Path | None = None,
    rwl_images: Path | None = None,
) -> list[QueryItem]:
    """Набор по имени. Неизвестное имя — `DatasetError`, а не тихий пустой список."""
    if name not in QUERY_SETS:
        raise DatasetError(f"неизвестный набор {name!r}: ожидается один из {', '.join(QUERY_SETS)}")
    if name in ("pairs", "pairs_phone"):
        return load_pairs(
            pairs_json or DEFAULT_PAIRS_JSON,
            pairs_images or DEFAULT_PAIRS_IMAGES,
            phone=name == "pairs_phone",
        )
    if name == "public":
        return from_bench_items(load_public(settings or get_settings()), name)
    kind: Literal["src", "aug"] = "src" if name == "rwl_src" else "aug"
    return load_rwl(kind, rwl_dir or DEFAULT_RWL_DIR, rwl_images or DEFAULT_RWL_IMAGES)


def notes_for(name: str) -> list[str]:
    """Оговорки набора для metrics.json: чего цифры не значат."""
    notes = []
    if name in ("pairs", "pairs_phone"):
        notes.append(PAIRS_NOTE)
        notes.append(PAIRS_FUNNEL_NOTE)
        # Геометрия видов выбрана по метрике на этом же наборе — цифры in-sample.
        notes.append(WINDOWS_NOTE)
    if name == "pairs_phone":
        notes.append(PHONE_NOTE)
    if name == "public":
        notes.append("три кадра организатора: дымовой тест, параметры по ним не подбираются")
    return notes


def summarize(items: Sequence[QueryItem]) -> dict[str, Any]:
    """Сводка набора: сколько кадров, сколько из каталога, сколько порченых."""
    in_catalog = sum(item.slug is not None for item in items)
    return {
        "queries": len(items),
        "in_catalog": in_catalog,
        "out_of_catalog": len(items) - in_catalog,
        "slugs": len({item.slug for item in items if item.slug}),
        "phone_shots": sum(item.phone_seed is not None for item in items),
    }


def missing_images(items: Sequence[QueryItem]) -> list[str]:
    """Кадры, которых нет на диске или которые пусты."""
    problems = []
    for item in items:
        if not item.image_path.is_file():
            problems.append(f"{item.query_id}: нет файла {item.image_path}")
        elif item.image_path.stat().st_size == 0:
            problems.append(f"{item.query_id}: пустой файл {item.image_path}")
    return problems
