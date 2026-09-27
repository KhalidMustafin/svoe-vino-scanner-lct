"""Снимки кода для стенда Э4 (PREREG_E4_card_attrs.md, §7; дополнение PREREG_E4_addendum.md).

- `at7b`   — `git archive` after-search @7b76247 как есть: ворота 0 (= дампы);
- `base`   — `git archive` слияния с Э1 (@e9f92f6), совместимость «Белое / Оранжевое» выключена:
             `COMPATIBLE_COLORS` в `app/resolve/attrs.py` заменена пустым множеством. Пустое
             множество — ровно поведение без совместимости (`colors_compatible` всегда ложь);
             код 12 строк таблицы по одной;
- `compat` — `git archive` слияния как есть: единица «совместимость».

    python e4_snapshots.py [имя …]
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SNAP_ROOT = Path(
    r"<корень>\svoe-vino-scanner\runs\field25\iters\snap"
)
AT7B = "7b76247"
#: Слияние acc-e1 в acc-e4: база замера «после Э1» и код Э4.
MERGED = "e9f92f6"
ATTRS = "app/resolve/attrs.py"
COMPAT_RE = re.compile(r"^(COMPATIBLE_COLORS: frozenset\[frozenset\[Color\]\] = ).+$", re.MULTILINE)
#: Снимок → (ревизия, совместимость включена).
VARIANTS: dict[str, tuple[str, bool]] = {
    "at7b": (AT7B, False),
    "base": (MERGED, False),
    "compat": (MERGED, True),
}


def build(name: str) -> Path:
    rev, compat = VARIANTS[name]
    target = SNAP_ROOT / f"e4_{name}"
    if target.exists():
        raise SystemExit(f"снимок {target} уже есть")
    target.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", rev, "app", "scripts", "configs"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    if rev == MERGED and not compat:
        path = target / ATTRS
        source = path.read_bytes().decode("utf-8")
        new, count = COMPAT_RE.subn(lambda m: m.group(1) + "frozenset()", source)
        if count != 1:
            raise SystemExit(f"{ATTRS}: найдено {count} определений COMPATIBLE_COLORS")
        path.write_bytes(new.encode("utf-8"))
    return target


if __name__ == "__main__":
    for snap in sys.argv[1:] or list(VARIANTS):
        print(build(snap))
