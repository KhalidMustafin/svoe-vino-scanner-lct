"""Снимки кода для стенда Э1: база, P1, каждый пункт R1 отдельно, пакеты (PREREG, §4).

Снимок = `git archive` базы (@7b76247) + четыре файла из коммита кода Э1 в нужном виде:
- сервис и `ambiguous.py` — из коммита кода только у снимков с P1, иначе из базы;
- `taxonomy.py` и `fields.py` — из коммита кода; таблицы пунктов R1, которых в снимке нет,
  заменяются пустыми. Пустые таблицы — это ровно поведение базы: снимок `off` (все пять пустые)
  обязан повторить базу, это проверка самого способа.

    python e1_snapshots.py <коммит кода> [имя …]
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
BASE = "7b76247"
TAXONOMY = "app/reading/taxonomy.py"
FIELDS = "app/reading/fields.py"
P1_FILES = ("app/api/service.py", "app/resolve/ambiguous.py")
#: Пункт R1 → (имя таблицы, пустое значение).
TABLES = {
    "a": ("COLOR_TYPOS", "{}"),
    "b": ("SUGAR_VARIANTS", "{}"),
    "c": ("SUGAR_COMBINED", "{}"),
    "g": ("COLOR_BLOCKERS", "()"),
    "d": ("REGION_WORDS", "frozenset()"),
}
#: Снимок → (P1 в сервисе, пункты R1).
VARIANTS: dict[str, tuple[bool, str]] = {
    "off": (False, ""),
    "p1": (True, ""),
    "r1a": (False, "a"),
    "r1b": (False, "b"),
    "r1c": (False, "c"),
    "r1g": (False, "g"),
    "r1d": (False, "d"),
    "r1": (False, "abcgd"),
    "pkg": (True, "abcgd"),
}


def git_show(rev: str, path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=REPO, check=True, capture_output=True
    ).stdout
    return out.decode("utf-8").replace("\r\n", "\n")


def empty_table(source: str, name: str, empty: str) -> str:
    """Определение таблицы `name` (до закрывающей скобки в первой колонке) → пустое значение."""
    pattern = re.compile(rf"^({name}: [^\n=]+ = ).*?^[)}}][^\n]*$", re.MULTILINE | re.DOTALL)
    new, count = pattern.subn(lambda m: m.group(1) + empty, source)
    if count != 1:
        raise SystemExit(f"таблица {name}: найдено {count} определений")
    return new


def build(code_rev: str, name: str) -> Path:
    p1, items = VARIANTS[name]
    target = SNAP_ROOT / f"e1_{name}"
    if target.exists():
        raise SystemExit(f"снимок {target} уже есть")
    target.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", BASE, "app", "scripts", "configs"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    taxonomy = git_show(code_rev, TAXONOMY)
    for item, (table, empty) in TABLES.items():
        if item not in items:
            taxonomy = empty_table(taxonomy, table, empty)
    (target / TAXONOMY).write_text(taxonomy, encoding="utf-8")
    (target / FIELDS).write_text(git_show(code_rev, FIELDS), encoding="utf-8")
    for path in P1_FILES:
        (target / path).write_text(git_show(code_rev if p1 else BASE, path), encoding="utf-8")
    return target


if __name__ == "__main__":
    rev, *names = sys.argv[1:]
    for snap in names or list(VARIANTS):
        print(build(rev, snap))
