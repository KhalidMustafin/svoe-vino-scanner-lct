"""Комплект «живые карточки» (`SVS_LIVE_CARDS=1`): индекс, gt и словарь каталога CSV + портала.

Э3 плана точности 25.09 (`research/2026-09-25_acc/PREREG_E3.md`). В комплект идут карточки
живого портала, чьего вина нет в выгрузке CSV: 73 slug портала без двух дублей вин CSV
(`chardonnay-2024` ↔ `one-barrel-uan-barrel`, `merlo-2` ↔ `merlo-litavshhuk` по
`wine_groups_final.json`) = 71 карточка.

- Индекс `index/visual-s2so400m-live71.npz`: все строки базового индекса как есть (те же
  float16, тот же порядок), за ними — виды 71 карточки из индекса стенда, где они уже посчитаны
  (`runs/field25/iters/cache/index/07b02d3881dc2055.npz`, строки с slug не из базы). Маска
  `base_rows` (строки базы) замораживает нормировку `zmax` по CSV: счёт карточек CSV не меняется.
- gt `gt/gt_tokens-live71.jsonl`: строки базового `gt_tokens.jsonl` байт в байт + записи 71
  карточки из `--new-cards` (`new_cards_gt_tokens.jsonl`, в репозиторий не входит; как в файле).
- Словарь `index/lexicon-live71.json`: `scripts/build_lexicon.py` по живому gt.

    python scripts/build_live_set.py --extra-index <.../07b02d3881dc2055.npz> \
        --wine-groups <field_dataset/catalog/wine_groups_final.json> [--data-dir data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.api.config import (
    DEFAULT_ATTRS_NAME,
    DEFAULT_INDEX_NAME,
    LIVE_ATTRS_NAME,
    LIVE_INDEX_NAME,
    LIVE_LEXICON_NAME,
)
from app.config import REPO_ROOT, get_settings
from app.features.contracts import IndexMeta
from app.features.index import VisualIndex


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_live_set.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--extra-index", type=Path, required=True, help="индекс с видами карточек")
    parser.add_argument("--wine-groups", type=Path, required=True, help="wine_groups_final.json")
    # Карточки живого портала в репозиторий не входят: файл передаётся явно.
    parser.add_argument("--new-cards", type=Path, required=True, help="gt карточек портала")
    return parser


def csv_duplicates(new_slugs: Sequence[str], base_slugs: set[str], groups: dict) -> set[str]:
    """Карточки портала, чьё вино уже есть в CSV: группа вина содержит slug базового gt."""
    dup: set[str] = set()
    for group in groups.values():
        members = set(group.get("members") or [])
        if members & base_slugs:
            dup |= members & set(new_slugs)
    return dup


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = build_parser().parse_args(argv)
    data = args.data_dir
    base_index_path = data / "index" / DEFAULT_INDEX_NAME
    base_gt_path = data / "gt" / DEFAULT_ATTRS_NAME
    out_index = data / "index" / LIVE_INDEX_NAME
    out_gt = data / "gt" / LIVE_ATTRS_NAME
    out_lexicon = data / "index" / LIVE_LEXICON_NAME

    # ---- карточки: 73 живых минус дубли вин CSV
    base_gt_bytes = base_gt_path.read_bytes()
    if not base_gt_bytes.endswith(b"\n"):
        raise SystemExit(f"{base_gt_path}: нет перевода строки в конце")
    base_slugs = {json.loads(line)["slug"] for line in base_gt_bytes.decode("utf-8").splitlines()}
    card_lines = [
        line for line in args.new_cards.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    cards = [(json.loads(line)["slug"], line) for line in card_lines]
    new_slugs = [slug for slug, _ in cards]
    if len(set(new_slugs)) != len(new_slugs) or set(new_slugs) & base_slugs:
        raise SystemExit("карточки портала повторяются или уже есть в gt CSV")
    groups = json.loads(args.wine_groups.read_text(encoding="utf-8"))
    dup = csv_duplicates(new_slugs, base_slugs, groups)
    kept = [(slug, line) for slug, line in cards if slug not in dup]
    print(f"карточек портала {len(cards)}, дубли вин CSV {sorted(dup)}, в комплект {len(kept)}")

    # ---- индекс: база как есть + виды карточек из индекса стенда
    with (
        np.load(base_index_path, allow_pickle=False) as base,
        np.load(args.extra_index, allow_pickle=False) as extra,
    ):
        base_meta = IndexMeta.model_validate_json(str(base["meta"].item()))
        extra_meta = IndexMeta.model_validate_json(str(extra["meta"].item()))
        if "base_rows" in base.files:
            raise SystemExit(f"{base_index_path}: базовый индекс уже с маской")
        if (extra_meta.model, extra_meta.dim) != (base_meta.model, base_meta.dim):
            raise SystemExit(f"{args.extra_index}: другая модель или размерность")
        b_slugs, b_views, b_vec = base["slugs"], base["views"], base["vectors"]
        e_slugs = [str(s) for s in extra["slugs"].tolist()]
        base_index_slugs = {str(s) for s in b_slugs.tolist()}
        rows = [i for i, s in enumerate(e_slugs) if s not in base_index_slugs]
        added = {e_slugs[i] for i in rows}
        want = {slug for slug, _ in kept}
        if added != want:
            raise SystemExit(
                f"виды в {args.extra_index.name} не те: лишние {sorted(added - want)[:5]}, "
                f"нет {sorted(want - added)[:5]}"
            )
        if b_vec.dtype != extra["vectors"].dtype:
            raise SystemExit("векторы базы и дополнения разной точности")
        slugs = np.concatenate([b_slugs, extra["slugs"][rows]])
        views = np.concatenate([b_views, extra["views"][rows]])
        vectors = np.concatenate([b_vec, extra["vectors"][rows]])
    base_rows = np.zeros(len(slugs), dtype=bool)
    base_rows[: len(b_slugs)] = True
    meta = base_meta.model_copy(
        update={
            "n_slugs": len(set(slugs.tolist())),
            "n_vectors": len(slugs),
            "built_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
    )
    out_index.parent.mkdir(parents=True, exist_ok=True)
    with out_index.open("wb") as fh:
        np.savez(
            fh,
            slugs=slugs,
            views=views,
            vectors=vectors,
            meta=np.array(meta.model_dump_json(), dtype=np.str_),
            base_rows=base_rows,
        )
    check = VisualIndex.load(out_index, model=base_meta.model)
    print(
        f"индекс: {len(b_slugs)} строк базы + {len(rows)} строк карточек = {len(check)}, "
        f"slug {check.n_slugs}, базовых строк {check.n_base}"
    )

    # ---- gt: база байт в байт + записи карточек
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    with out_gt.open("wb") as fh:
        fh.write(base_gt_bytes)
        for _, line in kept:
            fh.write(line.encode("utf-8") + b"\n")

    # ---- словарь тем же сборщиком, что и базовый
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_lexicon

    rc = build_lexicon.main(["--gt-tokens", str(out_gt), "--out", str(out_lexicon), "--top", "0"])
    if rc not in (0, None):
        raise SystemExit(f"build_lexicon: код {rc}")

    for path in (out_index, out_gt, out_lexicon):
        print(f"{path}  sha1 {sha1(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
