"""Признаки других треков фундаментального трека для ранкера: таблицы «кадр × кандидат CV».

Таблица — `.npz` с `ids` (кадры в порядке строк `trainpool.npz`), массивом (N, 20, F) в порядке
кандидатов CV (как `features`), именами признаков и `feat_slugs` для сверки. Нет значения
(кандидата или кадра) — NaN, в ранкер идёт 0 и отдельный признак не заводится: у пар пула
значения есть у всех кандидатов top-20 (проверяет `attach`).

Имена таблиц (`--extra`):
- `sift` — все признаки `spatial/sift_features.npz` (пространственная проверка SIFT);
- `sift:<имя>` — один признак оттуда (например, `sift:sift_inl_log` — основной, объявлен
  треком spatial до замера).

Знак веса: у всех признаков сходства SIFT вес ≥ 0 (больше согласованных точек — не довод
против кандидата); `sift_inl_gap` и `sift_inl_rel` — тоже сходство.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

import common as C

#: Трек spatial переписывает свою таблицу по ходу работы, поэтому ранкер читает только снимок
#: (`snapshot`, sha1 — в `extra_tables/SHA1SUMS.txt` и в мете OOF), а не живой файл.
SNAP = C.OUT / "extra_tables"
SOURCES = {"sift": C.P.FUND / "spatial" / "sift_features.npz"}
TABLES = {
    "sift": (SNAP / "sift_features.npz", "sift", "sift_names"),
}


def snapshot(base: str = "sift") -> str:
    """Скопировать живую таблицу трека в снимок ранкера; вернуть sha1 снимка."""
    import hashlib
    import shutil
    from datetime import datetime

    SNAP.mkdir(parents=True, exist_ok=True)
    src = SOURCES[base]
    dst = TABLES[base][0]
    shutil.copy2(src, dst)
    sha = hashlib.sha1(dst.read_bytes()).hexdigest()
    stamp = datetime.fromtimestamp(src.stat().st_mtime).strftime("%d.%m %H:%M:%S")
    with (SNAP / "SHA1SUMS.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"{sha} *{dst.name} (из {src}, изменён {stamp})\n")
    return sha


def table_sha1(specs: Sequence[str]) -> dict[str, str]:
    import hashlib

    return {spec: hashlib.sha1(TABLES[spec.partition(":")[0]][0].read_bytes()).hexdigest() for spec in specs}


def resolve_names(specs: Sequence[str]) -> list[tuple[str, Path, str, str, list[str] | None]]:
    out = []
    for spec in specs:
        base, _, one = spec.partition(":")
        path, arr, names = TABLES[base]
        out.append((base, path, arr, names, [one] if one else None))
    return out


def extra_names(specs: Sequence[str]) -> list[str]:
    names: list[str] = []
    for base, path, arr, nkey, only in resolve_names(specs):
        z = np.load(path)
        all_names = [str(n) for n in z[nkey]]
        names += only or all_names
    return names


def attach(frames: Sequence[C.Frame], specs: Sequence[str]) -> list[str]:
    """Дописывает `frame.extra[имя]` (по кандидатам кадра) и знаки в `C.EXTRA_SIGNS`."""
    added: list[str] = []
    for base, path, arr, nkey, only in resolve_names(specs):
        z = np.load(path)
        ids = [str(x) for x in z["ids"]]
        all_names = [str(n) for n in z[nkey]]
        use = only or all_names
        cols = [all_names.index(n) for n in use]
        data = z[arr]
        fs = z["feat_slugs"] if "feat_slugs" in z.files else None
        pos = {q: i for i, q in enumerate(ids)}
        missing = 0
        for f in frames:
            i = pos.get(f.id)
            n = len(f.slugs)
            if i is None:
                missing += 1
                vals = np.zeros((n, len(use)))
            else:
                if fs is not None:
                    assert tuple(str(s) for s in fs[i][:n]) == f.slugs, (base, f.id)
                vals = np.asarray(data[i, :n][:, cols], dtype=np.float64)
                if n and not np.isfinite(vals).all():
                    missing += 1
                    vals = np.nan_to_num(vals, nan=0.0)
            for j, name in enumerate(use):
                f.extra[name] = vals[:, j]
        for name in use:
            C.EXTRA_SIGNS[name] = 1
        added += use
        print(f"[extras] {base}: {len(use)} признаков, кадров без значений {missing}", flush=True)
    return added


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for b in sys.argv[1:] or ["sift"]:
        print(b, snapshot(b), extra_names([b]))
