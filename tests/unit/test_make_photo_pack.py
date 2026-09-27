"""Лёгкие фото выгрузки (`scripts/make_photo_pack.py`) — на синтетических картинках.

Проверяется то, из-за чего карточка показывала бы не то фото или никакого: в пачку идут только
фото выгрузки организатора — строка правки эталонов 23.09 (`packshot_fix`, `packshots_fixed/`)
берёт исходный путь из копии карты до правки, фото живых карточек портала из справочника не
берутся, прозрачный фон заливается белым, картинка только уменьшается.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_photo_pack.py"
FIELDS = ["slug", "path", "method"]


@pytest.fixture(scope="module")
def pack() -> Any:
    spec = importlib.util.spec_from_file_location("make_photo_pack", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def image(path: Path, color: tuple[int, ...], size: tuple[int, int] = (100, 300)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "RGBA" if len(color) == 4 else "RGB"
    Image.new(mode, size, color).save(path)
    return str(path)


def write_map(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    """Карта после правки (две правленые строки, одна без файла) и копия карты до правки."""
    dump = tmp_path / "dump"
    fixed_dir = tmp_path / "packshots_fixed"
    organizer = image(dump / "fixed-organizer.png", (0, 0, 255))
    clear = image(dump / "clear.png", (0, 0, 0, 0), size=(960, 2400))
    write_map(
        tmp_path / "slug_photo_map.csv",
        [
            {"slug": "fixed", "path": image(fixed_dir / "fixed.png", (255, 0, 0)),
             "method": "packshot_fix"},
            {"slug": "lost", "path": image(fixed_dir / "lost.png", (255, 0, 0)),
             "method": "packshot_fix"},
            {"slug": "clear", "path": clear, "method": "sitemap"},
            {"slug": "gone", "path": str(dump / "gone.png"), "method": "sitemap"},
        ],
    )  # fmt: skip
    write_map(
        tmp_path / "slug_photo_map.pre-packshot-fix.csv",
        [
            {"slug": "fixed", "path": organizer, "method": "sitemap"},
            {"slug": "clear", "path": clear, "method": "sitemap"},
        ],
    )
    live = image(tmp_path / "live_photos" / "live.webp", (0, 255, 0))
    (tmp_path / "wines.jsonl").write_text(
        json.dumps({"slug": "live", "photo": live, "in_csv": False}) + "\n", encoding="utf-8"
    )
    return tmp_path


def pixel(path: Path) -> tuple[int, ...]:
    with Image.open(path) as im:
        return im.convert("RGB").getpixel((im.width // 2, im.height // 2))


def close(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(abs(x - y) <= 8 for x, y in zip(a, b, strict=True))


def test_only_organizer_photos(pack: Any, catalog: Path) -> None:
    found = pack.sources(catalog / "slug_photo_map.csv")
    by_slug = dict(found.pairs)
    assert by_slug["fixed"].endswith("fixed-organizer.png")  # исходное фото выгрузки
    assert "lost" not in by_slug  # правка без исходного пути — фото нет
    assert "live" not in by_slug  # живые карточки портала из справочника не берутся
    assert not any("packshots_fixed" in path for path in by_slug.values())
    assert (found.missing, found.restored, found.dropped) == (0, 1, 1)


def test_foreign_path_is_recognized_without_method(pack: Any, tmp_path: Path) -> None:
    fixed = image(tmp_path / "packshots_fixed" / "x.png", (1, 2, 3))
    write_map(tmp_path / "map.csv", [{"slug": "x", "path": fixed, "method": ""}])
    found = pack.sources(tmp_path / "map.csv", tmp_path / "no-original.csv")
    assert found.pairs == [] and found.dropped == 1


def test_build_flattens_downscales_and_skips_missing(pack: Any, catalog: Path) -> None:
    out = catalog / "photos_small"
    rc = pack.main(["--photo-map", str(catalog / "slug_photo_map.csv"), "--out", str(out)])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["clear.webp", "fixed.webp"]
    assert close(pixel(out / "fixed.webp"), (0, 0, 255))  # синее исходное, а не красная правка
    assert close(pixel(out / "clear.webp"), (255, 255, 255))
    with Image.open(out / "clear.webp") as im:
        assert max(im.size) == 480
    with Image.open(out / "fixed.webp") as im:
        assert im.size == (100, 300)


def test_build_over_old_folder_fails_on_stale_files(
    pack: Any, catalog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Поверх старого каталога: фото живой карточки и правки без исходного фото не остаются молча."""
    out = catalog / "photos_small"
    image(out / "live.webp", (0, 255, 0))  # живая карточка портала от прошлой сборки
    image(out / "lost.webp", (255, 0, 0))  # правка эталонов без исходного фото выгрузки
    args = ["--photo-map", str(catalog / "slug_photo_map.csv"), "--out", str(out)]
    assert pack.main(args) == 1
    err = capsys.readouterr().err
    assert "2 файлов" in err and "live.webp" in err and "lost.webp" in err
    for name in ("live.webp", "lost.webp"):
        (out / name).unlink()
    assert pack.main(args) == 0
