"""Наборы для замеров OCR: публичные кадры, синтетика, полевой манифест.

Разбиение dev/test закреплено хэшем (40/60): все кадры одной бутылки попадают в один
split, а подбор параметров идёт только на dev. Ключ — группа двойников уровня B
(`cluster_B` из `gt_tokens.jsonl`), чтобы почти одинаковые этикетки не расходились по dev
и test; для slug вне групп — сам slug. Кадры без slug делятся по `bottle_id` из разметки,
иначе по query_id. Публичные кадры — split="smoke": дымовой тест, параметры по ним не
подбираются.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from bench.metrics import NONE_SLUG

Split = Literal["dev", "test", "smoke"]
DEV_SHARE = 0.4
MANIFEST_HEADER = ("query_id", "image_path")

#: Половина, которую смотрят один раз и явно (`--split test`). Прогон по всему набору
#: (`--split all`) нужен ради предсказаний по всем кадрам — это вход `bench.train_resolve
#: --final-test`. Его метрики test уходят в отдельный файл-«конверт», а `metrics.json` и
#: консоль считаются по остальным половинам: случайно открытый паспорт прогона test не покажет.
SEALED_SPLIT = "test"
SEALED_METRICS_FILE = "metrics.test-sealed.json"
SEALED_NOTE = (
    "метрики test запечатаны в конверт: здесь всё посчитано без кадров test. Test смотрят один "
    "раз и явно — прогоном --split test или открыв конверт для итогового отчёта"
)
SEALED_WARNING = (
    "КОНВЕРТ С TEST: метрики по всему набору вместе с test. Открывать один раз, для итогового "
    "отчёта; подбирать по этим цифрам ничего нельзя"
)


def seals_test(split: str, *, unseal: bool = False) -> bool:
    """Прячет ли прогон метрики test: да при `--split all`, если не снято явно (`--unseal-test`)."""
    return split == "all" and not unseal


_QUERY_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
_DRIVE_RE = re.compile(r"[A-Za-z]:")
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


class DatasetError(ValueError):
    """Набор не соответствует формату или разметка противоречит сама себе."""


class BenchItem(BaseModel):
    """Один кадр набора с эталоном и разметкой."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    image_path: Path
    slug: str | None  # None — вне каталога
    in_catalog: bool
    annotations: dict[str, Any] = Field(default_factory=dict)
    split: Split


# ------------------------------------------------------------------ разбиение
def twin_clusters(gt_tokens: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """slug → группа двойников уровня B (`cluster_B` в `gt_tokens.jsonl`)."""
    return {
        slug: str(record["cluster_B"])
        for slug, record in gt_tokens.items()
        if record.get("cluster_B") is not None
    }


def split_for(
    slug: str | None,
    query_id: str,
    *,
    group: str | None = None,
    cluster: str | None = None,
    dev_share: float = DEV_SHARE,
) -> Literal["dev", "test"]:
    """Детерминированный split: sha256 ключа, не зависит от процесса и порядка строк.

    `cluster` — группа двойников slug: правило, подобранное на одном двойнике в dev, иначе
    проверялось бы на почти том же тексте другого в test.
    """
    if cluster:
        key = f"cluster:{cluster}"
    elif slug:
        key = f"slug:{slug}"
    elif group:
        key = f"group:{group}"
    else:
        key = f"query:{query_id}"
    bucket = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    return "dev" if bucket < dev_share else "test"


# ------------------------------------------------------------------ чтение файлов
def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise DatasetError(f"нет файла ({what}): {path}")


def read_tsv(path: Path) -> list[dict[str, str]]:
    """TSV с заголовком; короткие строки дополняются пустыми ячейками."""
    _require_file(path, "TSV")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise DatasetError(f"{path}: пустой файл")
    header = [cell.strip() for cell in lines[0].split("\t")]
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) > len(header):
            raise DatasetError(f"{path}:{number}: ячеек больше, чем в заголовке")
        cells += [""] * (len(header) - len(cells))
        rows.append({name: cell.strip() for name, cell in zip(header, cells, strict=True)})
    return rows


def read_annotations(path: Path) -> dict[str, dict[str, Any]]:
    """JSONL разметки: объект на строку, ключ — query_id."""
    _require_file(path, "разметка")
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{number}: не JSON: {exc}") from exc
            qid = rec.get("query_id") if isinstance(rec, dict) else None
            if not qid:
                raise DatasetError(f"{path}:{number}: нет query_id")
            if qid in out:
                raise DatasetError(f"{path}:{number}: повтор query_id {qid}")
            out[qid] = rec
    return out


def _unsafe_path(image_path: str) -> bool:
    parts = re.split(r"[\\/]", image_path)
    return image_path.startswith(("/", "\\")) or bool(_DRIVE_RE.match(image_path)) or ".." in parts


def read_manifest(path: Path) -> list[tuple[str, str]]:
    """Манифест в формате participant_test.sh: `query_id<TAB>image_path`, id уникальны."""
    _require_file(path, "манифест")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines or tuple(lines[0].split("\t")) != MANIFEST_HEADER:
        raise DatasetError(f"{path}: заголовок должен быть query_id<TAB>image_path")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) != 2:
            raise DatasetError(f"{path}:{number}: ожидается две ячейки")
        qid, image = cells
        if not _QUERY_ID_RE.fullmatch(qid):
            raise DatasetError(f"{path}:{number}: недопустимый query_id {qid!r}")
        if qid in seen:
            raise DatasetError(f"{path}:{number}: повтор query_id {qid}")
        if not image or _unsafe_path(image):
            raise DatasetError(f"{path}:{number}: небезопасный image_path {image!r}")
        seen.add(qid)
        rows.append((qid, image))
    if not rows:
        raise DatasetError(f"{path}: нет ни одного кадра")
    return rows


def _read_gt(path: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    if rows and not {"query_id", "slug"} <= rows[0].keys():
        raise DatasetError(f"{path}: в заголовке нужны query_id и slug")
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["query_id"] in out:
            raise DatasetError(f"{path}: повтор query_id {row['query_id']}")
        out[row["query_id"]] = row
    return out


# ------------------------------------------------------------------ сборка
def _in_catalog(row: dict[str, str], slug: str | None, where: Path) -> bool:
    raw = row.get("in_catalog", "").lower()
    if not raw:
        return slug is not None
    if raw not in _TRUE | _FALSE:
        raise DatasetError(f"{where}: {row['query_id']}: in_catalog={raw!r}")
    flag = raw in _TRUE
    if flag != (slug is not None):
        raise DatasetError(f"{where}: {row['query_id']}: in_catalog={raw} противоречит slug")
    return flag


def _row_split(
    row: dict[str, str],
    slug: str | None,
    ann: dict[str, Any],
    where: Path,
    clusters: Mapping[str, str] | None = None,
) -> Split:
    explicit = row.get("split", "").lower()
    if explicit:
        if explicit not in get_args(Split):
            raise DatasetError(f"{where}: {row['query_id']}: split={explicit!r}")
        return explicit  # type: ignore[return-value]
    group = ann.get("bottle_id")
    cluster = clusters.get(slug) if clusters and slug else None
    return split_for(slug, row["query_id"], group=str(group) if group else None, cluster=cluster)


def _check_annotation(ann: dict[str, Any], slug: str | None, qid: str, where: Path | None) -> None:
    if "target_slug" in ann and (ann["target_slug"] or None) != slug:
        raise DatasetError(f"{where}: {qid}: target_slug={ann['target_slug']!r}, в эталоне {slug}")
    if "in_catalog" in ann and bool(ann["in_catalog"]) != (slug is not None):
        raise DatasetError(f"{where}: {qid}: in_catalog разметки противоречит эталону")


def load_manifest(
    manifest_tsv: Path,
    gt_tsv: Path,
    ann_jsonl: Path | None = None,
    *,
    images_dir: Path | None = None,
    split: Split | None = None,
    clusters: Mapping[str, str] | None = None,
) -> list[BenchItem]:
    """Полевой формат: манифест participant_test.sh + эталон `query_id, slug|__none__[, …]`.

    Необязательные колонки эталона: `in_catalog` (1/0) и `split` (dev/test — перекрывает хэш).
    Кадры ищутся в `images_dir`, по умолчанию рядом с манифестом. `clusters` — slug → группа
    двойников (`twin_clusters`): вся группа попадает в один split.
    """
    rows = read_manifest(manifest_tsv)
    gt = _read_gt(gt_tsv)
    annotations = read_annotations(ann_jsonl) if ann_jsonl is not None else {}
    missing = [qid for qid, _ in rows if qid not in gt]
    if missing:
        raise DatasetError(
            f"{gt_tsv}: нет эталона для {len(missing)} кадров, например {', '.join(missing[:5])}"
        )
    base = images_dir if images_dir is not None else manifest_tsv.parent
    items = []
    for qid, image in rows:
        row = gt[qid]
        slug = None if row["slug"] in ("", NONE_SLUG) else row["slug"]
        ann = annotations.get(qid, {})
        _check_annotation(ann, slug, qid, ann_jsonl)
        items.append(
            BenchItem(
                query_id=qid,
                image_path=base / image,
                slug=slug,
                in_catalog=_in_catalog(row, slug, gt_tsv),
                annotations=ann,
                split=split or _row_split(row, slug, ann, gt_tsv, clusters),
            )
        )
    return items


def load_synth(directory: Path, *, clusters: Mapping[str, str] | None = None) -> list[BenchItem]:
    """Синтетика `make_field_synth`: synth_manifest.tsv, synth_gt.tsv, synth_annotations.jsonl."""
    return load_manifest(
        directory / "synth_manifest.tsv",
        directory / "synth_gt.tsv",
        directory / "synth_annotations.jsonl",
        clusters=clusters,
    )


def load_public(settings: Settings | None = None, *, gt_dir: Path | None = None) -> list[BenchItem]:
    """Публичные кадры организатора (dataset_dir/eval) с нашей разметкой из data/gt.

    Только дымовой тест: split="smoke".
    """
    settings = settings or get_settings()
    eval_dir = settings.dataset_dir / "eval"
    gt_dir = gt_dir if gt_dir is not None else settings.data_dir / "gt"
    manifest = eval_dir / "queries.tsv"
    if not manifest.is_file():
        raise DatasetError(f"нет {manifest}: укажите SVS_DATASET_DIR (распакованный датасет)")
    return load_manifest(
        manifest,
        gt_dir / "public_gt.tsv",
        gt_dir / "public_annotations.jsonl",
        images_dir=eval_dir / "queries",
        split="smoke",
    )


# ------------------------------------------------------------------ проверки
def check_images(items: Sequence[BenchItem]) -> list[str]:
    """Кадры, которых нет на диске или которые пусты."""
    problems = []
    for item in items:
        if not item.image_path.is_file():
            problems.append(f"{item.query_id}: нет файла {item.image_path}")
        elif item.image_path.stat().st_size == 0:
            problems.append(f"{item.query_id}: пустой файл {item.image_path}")
    return problems


def summarize(
    items: Sequence[BenchItem], clusters: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Сводка набора.

    `split_conflicts` — slug, кадры которого разошлись по dev и test; `cluster_conflicts` —
    группы двойников, разошедшиеся по dev и test (например, из-за колонки `split`).
    """
    splits_by_slug: dict[str, set[str]] = defaultdict(set)
    splits_by_cluster: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item.slug and item.split != "smoke":
            splits_by_slug[item.slug].add(item.split)
            if clusters and item.slug in clusters:
                splits_by_cluster[clusters[item.slug]].add(item.split)
    in_catalog = sum(item.in_catalog for item in items)
    return {
        "items": len(items),
        "splits": dict(sorted(Counter(item.split for item in items).items())),
        "in_catalog": in_catalog,
        "out_of_catalog": len(items) - in_catalog,
        "slugs": len({item.slug for item in items if item.slug}),
        "annotated": sum(bool(item.annotations) for item in items),
        "split_conflicts": sorted(s for s, splits in splits_by_slug.items() if len(splits) > 1),
        "cluster_conflicts": sorted(
            c for c, splits in splits_by_cluster.items() if len(splits) > 1
        ),
    }
