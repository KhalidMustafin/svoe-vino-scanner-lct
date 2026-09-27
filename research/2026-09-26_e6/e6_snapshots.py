"""Снимки кода для стенда Э6 (PREREG.md, §3): каждый пункт отдельно и пакет.

Снимок = `git archive` коммита кода Э6 (`app`, `scripts`, `configs`) → `snap/e6_<имя>`, где
таблицы пунктов, которых в снимке нет, заменены пустыми `frozenset()`. Пустая таблица — ровно
поведение базы: снимок `off` (все пять пустые) обязан повторить `base` (@e266d2a), это проверка
самого способа. Замена — по дереву разбора (`ast`): строки определения таблицы целиком.

    python e6_snapshots.py <коммит кода> <имя>[:<пункты через запятую>] …
    python e6_snapshots.py 1a2b3c4 off g v s r sh pkg:g,s

Имя без списка пунктов: `off` — ни одного, `g`/`v`/`s`/`r`/`sh` — один этот пункт.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SNAP_ROOT = Path(
    r"<корень>\svoe-vino-scanner\runs\field25\iters\snap"
)
FIELDS = "app/reading/fields.py"
TAXONOMY = "app/reading/taxonomy.py"
#: Пункт → (файл, таблица).
TABLES: dict[str, tuple[str, str]] = {
    "g": (FIELDS, "NESTED_LEX_FIELDS"),
    "v": (FIELDS, "LONGER_CLAIM_FIELDS"),
    "s": (TAXONOMY, "NAME_STYLE_WORDS"),
    "r": (TAXONOMY, "REGION_WORDS_EXTRA"),
    "sh": (TAXONOMY, "NAME_WINE_WORDS"),
}


def empty_table(source: str, name: str) -> str:
    """Определение `name: тип = …` (любой длины) → `name: тип = frozenset()`."""
    found = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    ]
    if len(found) != 1:
        raise SystemExit(f"таблица {name}: найдено {len(found)} определений")
    node = found[0]
    lines = source.split("\n")
    annotation = ast.get_source_segment(source, node.annotation)
    start, end = node.lineno - 1, node.end_lineno or node.lineno
    lines[start:end] = [f"{name}: {annotation} = frozenset()"]
    return "\n".join(lines)


def build(code_rev: str, spec: str) -> Path:
    name, _, listed = spec.partition(":")
    if listed:
        items = set(listed.split(","))
    elif name in TABLES:
        items = {name}
    elif name == "off":
        items = set()
    else:
        raise SystemExit(f"снимок {spec}: неизвестное имя без списка пунктов")
    unknown = items - set(TABLES)
    if unknown:
        raise SystemExit(f"снимок {spec}: неизвестные пункты {sorted(unknown)}")
    target = SNAP_ROOT / f"e6_{name}"
    if target.exists():
        raise SystemExit(f"снимок {target} уже есть")
    target.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", code_rev, "app", "scripts", "configs"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    for item, (path, table) in TABLES.items():
        if item in items:
            continue
        file = target / path
        source = file.read_bytes().decode("utf-8").replace("\r\n", "\n")
        file.write_bytes(empty_table(source, table).encode("utf-8"))
    (target / "E6_SNAPSHOT.txt").write_text(
        f"код {code_rev}; пункты {sorted(items)}\n", encoding="utf-8"
    )
    return target


if __name__ == "__main__":
    rev, *specs = sys.argv[1:]
    for snap_spec in specs:
        print(build(rev, snap_spec))
