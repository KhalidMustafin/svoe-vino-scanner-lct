import csv
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from fakes import HashEmbedder
from PIL import Image

from app.features.index import VisualIndex

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture(scope="module")
def cli():
    """Скрипт как модуль: `scripts` не пакет, и так его грузят остальные тесты."""
    spec = importlib.util.spec_from_file_location("script_build_index", SCRIPTS / "build_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packshot(path: Path, color: tuple[int, int, int] = (200, 30, 40)) -> None:
    """Файл каталога: бутылка на прозрачном фоне, как packshot Strapi."""
    width, height = 60, 120
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bottle = Image.new("RGBA", (width // 2, height // 2), (*color, 255))
    image.paste(bottle, (width // 4, height // 4))
    image.save(path)


def _catalog(tmp_path: Path, n: int = 3) -> Path:
    """Каталог из `n` эталонов и `slug_photo_map.csv` рядом."""
    photos = tmp_path / "photos"
    photos.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        path = photos / f"wine-{i}.png"
        _packshot(path, color=(200, 30 + 40 * i, 40))
        rows.append({"slug": f"wine-{i}", "path": str(path), "bytes": "100"})
    photo_map = tmp_path / "slug_photo_map.csv"
    with photo_map.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["slug", "path", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    return photo_map


def test_read_rgba_keeps_transparency(cli, tmp_path):
    _packshot(tmp_path / "wine.png")
    image = cli.read_rgba(tmp_path / "wine.png")
    assert image.shape == (120, 60, 4) and image.dtype == np.uint8
    assert image[0, 0, 3] == 0 and image[60, 30, 3] == 255


def test_read_rgba_turns_the_frame_by_exif(cli, tmp_path):
    """Эталон и запрос готовятся одинаково: `decode_image` крутит по EXIF, и `read_rgba` тоже.

    Раньше повёрнутый packshot попадал в индекс лёжа, а тот же файл как запрос — стоя.
    """
    from app.normalize.decode import decode_image

    path = tmp_path / "turned.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Orientation: повернуть на 90°
    Image.new("RGB", (60, 120), (10, 20, 30)).save(path, format="JPEG", exif=exif)
    assert cli.read_rgba(path).shape[:2] == decode_image(path.read_bytes()).shape[:2] == (60, 120)


def test_read_photo_map_reads_slug_and_path(cli, tmp_path):
    rows = cli.read_photo_map(_catalog(tmp_path, 3))
    assert len(rows) == 3 and rows[0][0] == "wine-0"
    assert rows[0][1].name == "wine-0.png"


def test_read_photo_map_honours_limit(cli, tmp_path):
    assert len(cli.read_photo_map(_catalog(tmp_path, 5), limit=2)) == 2


def test_read_photo_map_skips_rows_without_slug_or_path(cli, tmp_path):
    photo_map = _catalog(tmp_path, 1)
    with photo_map.open("a", encoding="utf-8", newline="") as fh:
        fh.write(",nowhere.png,0\nno-photo,,0\n")
    assert len(cli.read_photo_map(photo_map)) == 1


def test_read_photo_map_requires_columns(cli, tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("wine,file\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="slug"):
        cli.read_photo_map(path)


def test_iter_entries_makes_views_and_counts_losses(cli, tmp_path):
    rows = cli.read_photo_map(_catalog(tmp_path, 2))
    rows.append(("missing", tmp_path / "photos" / "no-such.png"))
    stats = {"read": 0, "read_s": 0.0, "skipped": []}
    entries = list(cli.iter_entries(rows, ["bottle", "label"], stats))
    assert [slug for slug, _ in entries] == ["wine-0", "wine-1"]
    assert set(entries[0][1]) == {"bottle", "label"}  # только заказанные виды
    assert stats["read"] == 2 and len(stats["skipped"]) == 1
    assert stats["skipped"][0].startswith("missing:")


def test_iter_entries_survives_a_broken_frame_after_reading(cli, tmp_path, monkeypatch):
    """Под `try` не только чтение файла, но и кропы: кадр разбирается уже после `read_rgba`."""
    rows = cli.read_photo_map(_catalog(tmp_path, 2))
    original = cli.from_packshot

    def explode(image, **kwargs):
        if not explode.done:
            explode.done = True
            raise ValueError("кадр нулевой высоты")
        return original(image, **kwargs)

    explode.done = False
    monkeypatch.setattr(cli, "from_packshot", explode)
    stats = {"read": 0, "read_s": 0.0, "skipped": []}
    entries = list(cli.iter_entries(rows, ["bottle"], stats))
    assert [slug for slug, _ in entries] == ["wine-1"]
    assert stats["skipped"] == ["wine-0: кадр нулевой высоты"]


def test_iter_entries_skips_an_entry_without_a_single_view(cli, tmp_path):
    """Вид, которого у packshot не бывает, — пропуск с записью, а не ValueError в сборке."""
    rows = cli.read_photo_map(_catalog(tmp_path, 1))
    stats = {"read": 0, "read_s": 0.0, "skipped": []}
    assert list(cli.iter_entries(rows, ["full"], stats)) == []
    assert stats["read"] == 0 and "ни одного из видов" in stats["skipped"][0]


def test_cli_refuses_full_as_a_packshot_view(cli, tmp_path):
    """`--views full` разбирался argparse и валил сборку на первом же файле."""
    photo_map = _catalog(tmp_path, 1)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--photo-map", str(photo_map), "--views", "full"], embedder=HashEmbedder())
    assert exc.value.code == 2


def test_cli_does_not_save_an_empty_index(cli, tmp_path, capsys):
    photo_map = _catalog(tmp_path, 2)
    for photo in (tmp_path / "photos").glob("*.png"):
        photo.write_bytes(b"not an image")
    out = tmp_path / "visual.npz"
    code = cli.main(
        ["--photo-map", str(photo_map), "--out", str(out)], embedder=HashEmbedder(dim=8)
    )
    assert code == 2 and not out.exists()
    assert "ни одного эталона" in capsys.readouterr().err


def test_file_sha1_is_the_hash_of_the_source(cli, tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"slug,path\n")
    assert cli.file_sha1(path) == hashlib.sha1(b"slug,path\n").hexdigest()


def test_cli_builds_index_on_three_files(cli, tmp_path, capsys):
    photo_map = _catalog(tmp_path, 3)
    out = tmp_path / "index" / "visual.npz"
    code = cli.main(
        ["--photo-map", str(photo_map), "--out", str(out), "--limit", "3", "--batch-size", "4"],
        embedder=HashEmbedder(dim=8),
    )
    assert code == 0 and out.is_file()
    index = VisualIndex.load(out, model="fake/hash")
    assert index.n_slugs == 3 and len(index) == 9  # три вида на эталон
    assert index.meta.views == ["bottle", "label", "band"]
    assert index.meta.source_sha1 == cli.file_sha1(photo_map)
    printed = capsys.readouterr().out
    assert "3 slug" in printed and "9 векторов" in printed and "fake/hash" in printed


def test_cli_limit_cuts_the_catalog(cli, tmp_path):
    photo_map = _catalog(tmp_path, 5)
    out = tmp_path / "visual.npz"
    code = cli.main(
        ["--photo-map", str(photo_map), "--out", str(out), "--limit", "2", "--views", "bottle"],
        embedder=HashEmbedder(dim=8),
    )
    assert code == 0
    index = VisualIndex.load(out)
    assert index.n_slugs == 2 and len(index) == 2 and index.meta.views == ["bottle"]


def test_cli_index_is_searchable(cli, tmp_path):
    """Собранный индекс ищет: свой же эталон стоит первым."""
    photo_map = _catalog(tmp_path, 3)
    out = tmp_path / "visual.npz"
    embedder = HashEmbedder(dim=8)
    cli.main(["--photo-map", str(photo_map), "--out", str(out)], embedder=embedder)
    query = cli.from_packshot(cli.read_rgba(tmp_path / "photos" / "wine-1.png"))
    result = VisualIndex.load(out).search(query, embedder, top_k=3)
    assert result.candidates[0].slug == "wine-1"
    assert result.candidates[0].score == pytest.approx(1.0, abs=1e-3)
    assert result.margin > 0


def test_cli_without_photo_map_returns_two(cli, tmp_path, capsys):
    code = cli.main(["--photo-map", str(tmp_path / "нет.csv")], embedder=HashEmbedder())
    assert code == 2 and "нет файла эталонов" in capsys.readouterr().err


def test_cli_reports_empty_photo_map(cli, tmp_path, capsys):
    path = tmp_path / "empty.csv"
    path.write_text("slug,path\n", encoding="utf-8")
    code = cli.main(["--photo-map", str(path)], embedder=HashEmbedder())
    assert code == 2 and "ни одной строки" in capsys.readouterr().err
