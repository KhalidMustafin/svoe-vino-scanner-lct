"""Пересжатые фото выгрузки организатора для карточки и полевого стенда: 136 МБ → около 13 МБ.

Зачем. В карточке человек сверяет найденное вино с бутылкой в руке — без картинки отметка
«верно / не то» превращается в чтение slug. Возить на VPS всю выгрузку Strapi (2 096 файлов,
136 МБ) незачем: на телефоне картинка занимает сотню пикселей по ширине.

Файлы кладутся по slug (`<slug>.webp`), а не по имени из выгрузки: в CSV имя фото и имя файла
на диске не совпадают, а slug — ключ, по которому сервис и спрашивает картинку.

Источники — только фото выгрузки организатора (решение 24.09): карта `slug_photo_map.csv`
(2 103 slug выгрузки, пути в дампе Strapi организатора). Правка эталонов 23.09 направила 9 строк
карты на фото живого портала и Роскачества (`method = packshot_fix`, каталог
`packshots_fixed/`) — у таких строк берётся исходный путь из копии карты до правки
(`slug_photo_map.pre-packshot-fix.csv`), а без неё строка пропускается. Фото 73 живых карточек
портала (путь `photo` справочника `wines.jsonl`) в пачку не идут.

Сборка ничего не удаляет, поэтому собирается в пустой каталог: файл, которого она не писала
(фото живой карточки или правки эталонов от прошлой сборки), — ошибка и код выхода 1.

    python scripts/make_photo_pack.py                      # data/catalog/photos_small
    python scripts/make_photo_pack.py --side 800 --quality 82

Прозрачный фон заливается белым: без этого на светлой карточке у бутылки появляется тёмная
кайма (тот же приём, что `flatten_white` в цепочке чтения).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import REPO_ROOT

#: Строки карты, которые правка эталонов 23.09 направила не на фото выгрузки.
FOREIGN_METHODS = frozenset({"packshot_fix"})
FOREIGN_DIR = "packshots_fixed"
#: Суффикс копии карты, сохранённой перед правкой эталонов 23.09.
ORIGINAL_SUFFIX = ".pre-packshot-fix"


@dataclass(frozen=True)
class Sources:
    """Пары slug → фото выгрузки и что пришлось сделать со строками карты."""

    pairs: list[tuple[str, str]]
    missing: int = 0  # строк карты без пути
    restored: int = 0  # строк правки, у которых взят исходный путь выгрузки
    dropped: int = 0  # строк правки без исходного пути: фото нет


def original_map(photo_map: Path) -> Path:
    """Копия карты до правки эталонов рядом с картой."""
    return photo_map.with_name(photo_map.stem + ORIGINAL_SUFFIX + photo_map.suffix)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def foreign(row: dict[str, str]) -> bool:
    """Путь строки ведёт не к фото выгрузки: правка эталонов или каталог `packshots_fixed`."""
    path = (row.get("path") or "").strip()
    method = (row.get("method") or "").strip()
    return method in FOREIGN_METHODS or FOREIGN_DIR in Path(path).parts


def sources(photo_map: Path, original: Path | None = None) -> Sources:
    """Фото выгрузки по slug; строки правки — исходным путём из копии карты до правки."""
    before: dict[str, str] = {}
    original = original if original is not None else original_map(photo_map)
    if original.is_file():
        for row in _read(original):
            slug, path = (row.get("slug") or "").strip(), (row.get("path") or "").strip()
            if slug and path and not foreign(row):
                before[slug] = path
    pairs: dict[str, str] = {}
    missing = restored = dropped = 0
    for row in _read(photo_map):
        slug = (row.get("slug") or "").strip()
        path = (row.get("path") or "").strip()
        if not slug or not path:
            missing += 1
            continue
        if foreign(row):
            if slug in before:
                pairs[slug] = before[slug]
                restored += 1
            else:
                dropped += 1
            continue
        pairs[slug] = path
    return Sources(list(pairs.items()), missing, restored, dropped)


@dataclass(frozen=True)
class Built:
    """Что сделала сборка: slug записанных фото, сколько пропущено, байт на выходе."""

    written: list[str]
    skipped: int = 0
    size: int = 0


def stale_files(out_dir: Path, written: list[str]) -> list[Path]:
    """Файлы каталога, которых эта сборка не писала.

    Сборка только дописывает и перезаписывает `<slug>.webp`, но ничего не удаляет: собранная
    поверх старого каталога, она оставила бы в нём фото 73 живых карточек портала или строки
    правки эталонов без исходного фото выгрузки (`dropped`) — такие файлы в пачку идти не должны.
    """
    if not out_dir.is_dir():
        return []
    keep = {f"{slug}.webp" for slug in written}
    return sorted(path for path in out_dir.iterdir() if path.is_file() and path.name not in keep)


def build(pairs: list[tuple[str, str]], out_dir: Path, side: int, quality: int) -> Built:
    """Пересжимает фото выгрузки в `<out_dir>/<slug>.webp`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped = size = 0
    for slug, source in pairs:
        if not Path(source).is_file():
            skipped += 1
            continue
        target = out_dir / f"{slug}.webp"
        try:
            with Image.open(source) as image:
                image.load()
                if image.mode in ("RGBA", "LA", "P"):
                    image = image.convert("RGBA")
                    flat = Image.new("RGB", image.size, (255, 255, 255))
                    flat.paste(image, mask=image.split()[-1])
                    image = flat
                else:
                    image = image.convert("RGB")
                # Только уменьшение: половина эталонов и так уже 336 px по ширине, и
                # растягивание их до «стандарта» только добавило бы веса.
                scale = min(1.0, side / max(image.size))
                if scale < 1.0:
                    image = image.resize(
                        (
                            max(1, round(image.width * scale)),
                            max(1, round(image.height * scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                image.save(target, "WEBP", quality=quality, method=4)
        except (OSError, ValueError) as exc:
            print(f"пропущено {slug}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        written.append(slug)
        size += target.stat().st_size
    return Built(written, skipped, size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--photo-map", type=Path, default=REPO_ROOT / "data/catalog/slug_photo_map.csv"
    )
    parser.add_argument(
        "--original-map",
        type=Path,
        default=None,
        help="карта до правки эталонов; по умолчанию slug_photo_map.pre-packshot-fix.csv у карты",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data/catalog/photos_small")
    parser.add_argument("--side", type=int, default=480, help="длинная сторона, px")
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args(argv)

    if not args.photo_map.is_file():
        print(f"нет карты фото: {args.photo_map}", file=sys.stderr)
        return 2
    found = sources(args.photo_map, args.original_map)
    built = build(found.pairs, args.out, args.side, args.quality)
    print(
        f"{len(built.written)} фото выгрузки в {args.out} ({built.size / 1024 / 1024:.1f} МБ), "
        f"пропущено {built.skipped + found.missing}; строк правки эталонов: исходное фото у "
        f"{found.restored}, без фото {found.dropped}"
    )
    stale = stale_files(args.out, built.written)
    if stale:
        names = ", ".join(path.name for path in stale[:5])
        print(
            f"ОШИБКА: в {args.out} ещё {len(stale)} файлов, которых эта сборка не писала "
            f"(фото живых карточек портала или правки эталонов прошлой сборки): {names}. "
            "Уберите каталог (сначала копию) и соберите заново.",
            file=sys.stderr,
        )
        return 1
    return 0 if built.written else 1


if __name__ == "__main__":
    sys.exit(main())
