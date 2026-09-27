"""Сборка закрытого словаря каталога: `gt_tokens.jsonl` → `data/index/lexicon.json`.

python scripts/build_lexicon.py
python scripts/build_lexicon.py --gt-tokens data/gt/gt_tokens.jsonl --out data/index/lexicon.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from app.config import get_settings
from app.reading.lexicon.build import (
    Lexicon,
    build_from_gt,
    default_gt_tokens_path,
    default_lexicon_path,
)


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python scripts/build_lexicon.py", description="Словарь каталога из gt_tokens.jsonl."
    )
    parser.add_argument("--gt-tokens", type=Path, default=default_gt_tokens_path(settings))
    parser.add_argument("--out", type=Path, default=default_lexicon_path(settings))
    parser.add_argument(
        "--no-builtin-terms",
        action="store_true",
        help="без общих терминов сахара, цвета и серий — только то, что есть в каталоге",
    )
    parser.add_argument("--top", type=int, default=5, help="сколько частых норм печатать")
    return parser


def print_json(obj: object) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(obj, ensure_ascii=True, indent=1))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.gt_tokens.is_file():
        print(f"нет эталонных токенов: {args.gt_tokens}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    lexicon = build_from_gt(args.gt_tokens, builtin_terms=not args.no_builtin_terms)
    built_ms = int((time.perf_counter() - started) * 1000)
    path = lexicon.save(args.out)
    started = time.perf_counter()
    reloaded = Lexicon.load(path)
    load_ms = int((time.perf_counter() - started) * 1000)
    if reloaded.entries != lexicon.entries:
        print("словарь после загрузки отличается от собранного", file=sys.stderr)
        return 1
    stats = lexicon.stats(top=args.top)
    stats.update(
        out=str(path),
        size_bytes=path.stat().st_size,
        build_ms=built_ms,
        load_ms=load_ms,
        meta=lexicon.meta,
    )
    print_json(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
