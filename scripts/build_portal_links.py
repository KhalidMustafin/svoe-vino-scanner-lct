"""Какие позиции выгрузки опубликованы на портале — по карте сайта из дампа Strapi организатора.

Ссылка `portal_url` (`https://vino-svoe.ru/wines/<slug>`) — единственная внешняя ссылка ответа,
и вести она должна на живую страницу. Флаг «опубликовано» живого портала после решения 24.09 не
читается, но в дампе Strapi организатора (`prod-svoe-vino-strapi.part*.rar`) лежит его же карта
сайта вин: `strapi/uploads/wines_sitemap_<hash>.xml`, выгрузка 04.09.2026. В карту сайта попадают
только опубликованные страницы, так что это и есть статус публикации по данным организатора.

Замер 24.09: в карте 2 037 вин, все — slug выгрузки `strapi_output0709.csv`; 66 из 2 103 slug
выгрузки в ней нет (Союз-Вино, Золотая Балка, черновики). Сервис читает список этих 66
(`app/recommend/portal_links.json`, `catalog.portal_url_of`): у них `portal_url: null`.

Карту сайта можно дать готовым файлом (`--sitemap`) или архивом дампа (`--dump`): тогда она
извлекается 7-Zip в `--work` (другие распаковщики RAR5 не умеют). `--check` пересобирает список
в памяти и сверяет с файлом (код выхода 1 при расхождении).

    python scripts/build_portal_links.py --csv "<Датасет>/strapi_output0709.csv" \
        --dump "<Датасет>/prod-svoe-vino-strapi.part1.rar" --work <каталог>
    python scripts/build_portal_links.py --csv … --sitemap <wines_sitemap_….xml> --check
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "recommend" / "portal_links.json"
#: Путь карты сайта вин в дампе: имя файла с хэшем Strapi, поэтому по маске.
SITEMAP_MASK = r"prod-svoe-vino-strapi\prod-svoe-vino\strapi\uploads\wines_sitemap_*.xml"
_LOC = re.compile(r"<loc>https://vino-svoe\.ru/wines/([^<]+)</loc>")
_LASTMOD = re.compile(r"<lastmod>([^<]+)</lastmod>")
SEVEN_ZIP = ("7z", r"C:\Program Files\7-Zip\7z.exe")


def csv_slugs(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sorted({(row.get("Slug") or "").strip() for row in csv.DictReader(handle)} - {""})


def sitemap_slugs(text: str) -> tuple[set[str], str | None]:
    """Slug вин карты сайта и самая поздняя дата правки в ней."""
    dates = _LASTMOD.findall(text)
    return set(_LOC.findall(text)), max(dates) if dates else None


def extract_sitemap(dump: Path, work: Path) -> Path:
    """Карта сайта вин из многотомного RAR5 — 7-Zip, только этот файл."""
    exe = next((p for p in (shutil.which(SEVEN_ZIP[0]), SEVEN_ZIP[1]) if p and Path(p).is_file()),
               None)  # fmt: skip
    if exe is None:
        raise SystemExit("нет 7-Zip: дайте карту сайта готовым файлом (--sitemap)")
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "e", str(dump), f"-o{work}", SITEMAP_MASK, "-y"], check=True,
                   capture_output=True)  # fmt: skip
    found = sorted(work.glob("wines_sitemap_*.xml"))
    if not found:
        raise SystemExit(f"в {dump} нет карты сайта вин")
    return found[-1]


def build(slugs: list[str], sitemap: Path) -> dict[str, Any]:
    listed, lastmod = sitemap_slugs(sitemap.read_text(encoding="utf-8"))
    ours = set(slugs)
    return {
        "_comment": (
            "Slug выгрузки strapi_output0709.csv, которых нет в карте сайта вин из дампа Strapi "
            "организатора: страница не опубликована, portal_url — null. Собирает "
            "scripts/build_portal_links.py."
        ),
        "source": f"prod-svoe-vino-strapi.part*.rar: uploads/{sitemap.name}",
        "sitemap_lastmod": lastmod,
        "catalog": len(ours),
        "published": len(ours & listed),
        "sitemap_not_in_catalog": len(listed - ours),
        "unlisted": sorted(ours - listed),
    }


def dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True, help="strapi_output0709.csv")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sitemap", type=Path, help="wines_sitemap_….xml из дампа")
    source.add_argument("--dump", type=Path, help="prod-svoe-vino-strapi.part1.rar")
    parser.add_argument("--work", type=Path, help="куда извлечь карту сайта из --dump")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true", help="сверить с --out, ничего не писать")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if args.dump is not None:
        if args.work is None:
            parser.error("--dump требует --work")
        sitemap = extract_sitemap(args.dump, args.work)
    else:
        sitemap = args.sitemap
    text = dump_json(build(csv_slugs(args.csv), sitemap))
    data = json.loads(text)
    print(f"выгрузка {data['catalog']}, в карте сайта {data['published']}, "
          f"нет в карте {len(data['unlisted'])}, лишних в карте {data['sitemap_not_in_catalog']}")  # fmt: skip
    if args.check:
        same = args.out.is_file() and args.out.read_text(encoding="utf-8") == text
        print("совпадает" if same else f"РАСХОДИТСЯ с {args.out}")
        return 0 if same else 1
    args.out.write_text(text, encoding="utf-8")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
