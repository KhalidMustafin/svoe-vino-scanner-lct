"""Индекс эталонов каталога: packshot → виды → векторы → `data/index/visual.npz`.

Вход — `slug_photo_map.csv` (колонки `slug`, `path`): 2 103 packshot каталога «Своё Вино»,
PNG и WebP с альфой. Каждый файл режется на виды (`app.features.views.from_packshot`) и
считается башней зрения SigLIP; в индекс идёт строка на вид.

Боевой индекс `data/index/visual-s2so400m.npz` лежит в git; пересобирать его нужно под ту
модель, которой потом будут считаться запросы.

    python scripts/build_index.py --limit 30 --out data/index/probe.npz --device cpu
    python scripts/build_index.py --out data/index/visual.npz --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.config import get_settings
from app.features.contracts import Embedder, ViewName
from app.features.embedder import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    ModelNotAvailable,
    SiglipEmbedder,
)
from app.features.index import VisualIndex
from app.features.views import BACKGROUND, from_packshot

#: Карта «slug → фото» рядом с выгрузкой организатора (папка датасета — соседняя с репозиторием).
DEFAULT_PHOTO_MAP = Path("../Датасет и подробное задание/analysis/slug_photo_map.csv")
#: Виды эталона: у packshot нет «всего кадра» — бутылка и есть кадр.
PACKSHOT_VIEWS: tuple[ViewName, ...] = ("bottle", "label", "band")


def read_rgba(path: Path) -> np.ndarray:
    """Файл каталога → массив RGBA uint8: альфа нужна, чтобы обрезать пустые поля.

    Ориентация снимается по EXIF, как в `app.normalize.decode.decode_image`: эталон и
    запрос должны готовиться одинаково, иначе повёрнутый packshot попал бы в индекс лёжа.
    У всех 2 103 файлов каталога сейчас Orientation=1, но полагаться на это нельзя.
    """
    with Image.open(path) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source) or source
        return np.array(oriented.convert("RGBA"), dtype=np.uint8)


def read_photo_map(path: Path, limit: int | None = None) -> list[tuple[str, Path]]:
    """Строки `slug_photo_map.csv` как пары (slug, файл). `limit` — первые N строк."""
    rows: list[tuple[str, Path]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for field in ("slug", "path"):
            if field not in (reader.fieldnames or []):
                raise ValueError(f"{path}: нет колонки {field!r}")
        for row in reader:
            slug, photo = (row.get("slug") or "").strip(), (row.get("path") or "").strip()
            if not slug or not photo:
                continue
            rows.append((slug, Path(photo)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_entries(
    rows: Sequence[tuple[str, Path]],
    views: Sequence[ViewName],
    stats: dict[str, Any],
    *,
    background: tuple[int, int, int] = BACKGROUND,
) -> Iterator[tuple[str, dict[ViewName, np.ndarray]]]:
    """Виды каждого эталона. Битый или пропавший файл не валит сборку, а считается.

    Под `try` не только чтение, но и кропы: кадр нулевой высоты или с испорченной альфой
    разбирается уже после `read_rgba`, а сборка на 2 103 файлах не должна падать на одном.
    Эталон, от которого не осталось ни одного нужного вида, тоже считается пропущенным.
    """
    for slug, photo in rows:
        started = time.perf_counter()
        try:
            prepared = from_packshot(read_rgba(photo), background=background)
        except (OSError, ValueError) as exc:
            stats["skipped"].append(f"{slug}: {exc}")
            continue
        picked = {name: prepared[name] for name in views if name in prepared}
        if not picked:
            stats["skipped"].append(f"{slug}: ни одного из видов {list(views)}")
            continue
        stats["read_s"] += time.perf_counter() - started
        stats["read"] += 1
        yield slug, picked


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python scripts/build_index.py",
        description="Индекс визуальных эталонов каталога «Своё Вино».",
    )
    parser.add_argument("--photo-map", type=Path, default=DEFAULT_PHOTO_MAP)
    parser.add_argument("--out", type=Path, default=settings.data_dir / "index" / "visual.npz")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device", default=None, help=f"по умолчанию SVS_DEVICE ({settings.device})"
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(PACKSHOT_VIEWS),
        # Только виды packshot: «full» у эталона не бывает (бутылка и есть кадр), и с ним
        # `from_packshot` не дал бы ни одного вида — сборка валилась бы на первом файле.
        choices=list(PACKSHOT_VIEWS),
        help="какие виды эталона класть в индекс",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="первые N строк — для проверки")
    return parser


def main(argv: Sequence[str] | None = None, embedder: Embedder | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.photo_map.is_file():
        print(f"нет файла эталонов: {args.photo_map}", file=sys.stderr)
        return 2
    rows = read_photo_map(args.photo_map, args.limit)
    if not rows:
        print(f"в {args.photo_map} нет ни одной строки со slug и path", file=sys.stderr)
        return 2
    if embedder is None:
        embedder = SiglipEmbedder(args.model, args.device, args.dtype, batch_size=args.batch_size)
    stats: dict[str, Any] = {"read": 0, "read_s": 0.0, "skipped": []}
    started = time.perf_counter()
    try:
        index = VisualIndex.build(
            iter_entries(rows, args.views, stats),
            embedder,
            source_sha1=file_sha1(args.photo_map),
            batch_size=args.batch_size,
        )
    except ModelNotAvailable as exc:
        print(f"модель недоступна: {exc}", file=sys.stderr)
        return 3
    elapsed = time.perf_counter() - started
    if not len(index):
        print(
            f"ни одного эталона из {len(rows)}: пустой индекс не сохранён",
            file=sys.stderr,
        )
        for line in stats["skipped"][:10]:
            print(f"  пропущен {line}", file=sys.stderr)
        return 2
    path = index.save(args.out)

    print(
        f"эталонов: {index.n_slugs} slug, {len(index)} векторов, виды {', '.join(index.meta.views)}"
    )
    print(f"прочитано файлов: {stats['read']} из {len(rows)}, пропущено {len(stats['skipped'])}")
    for line in stats["skipped"][:10]:
        print(f"  пропущен {line}")
    per_vector = elapsed / len(index) if len(index) else 0.0
    print(
        f"время: чтение и кропы {stats['read_s']:.1f} с, всего {elapsed:.1f} с "
        f"({per_vector * 1000:.0f} мс на вектор)"
    )
    print(
        f"индекс: {path}, {path.stat().st_size / 2**20:.1f} МБ, "
        f"модель {index.meta.model}, dim {index.meta.dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
