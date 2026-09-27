"""Эталоны `gt_tokens` для единиц Э4 (PREREG_E4_card_attrs.md, §7; дополнение, §2).

Строки правок берутся из замороженной таблицы — `data/gt/gt_fixes.tsv` в коммите 17016d0, а не
из рабочего дерева. Сборка — `scripts/build_gt_tokens.py` снимка `e4_base` (код слияния с Э1).

    python e4_gt.py nofix              — без правок (`--no-fixes`), обязан дать sha1 bf6a8823…
    python e4_gt.py row01 … row12      — таблица из одной строки N
    python e4_gt.py pkg 1 2 5 …        — таблица из перечисленных строк (прошедшие единицы)

Выход: `runs/field25/iters/runs/e4/gt/<имя>/` — `gt_fixes.tsv` (что применено), `gt_tokens.jsonl`,
`gt_summary.json`, `gt_checks.txt`, `build_info.json` (sha1 и отличия записей от `nofix`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path(r"<корень>")
SCANNER = ROOT / "svoe-vino-scanner"
PYTHON = SCANNER / ".venv" / "Scripts" / "python.exe"
SNAP = SCANNER / "runs" / "field25" / "iters" / "snap" / "e4_base"
GT_ROOT = SCANNER / "runs" / "field25" / "iters" / "runs" / "e4" / "gt"
DATASET = ROOT / "Датасет и подробное задание" / "unpacked" / "Датасет"
#: Входы сборки не из папки данных сервиса: кластеры, живые вина, карта виноделен.
ANALYSIS = ROOT / "Датасет и подробное задание" / "analysis"
FROZEN = "17016d0"
BASE_SHA1 = "bf6a8823a45004b1e13e586e5a92e05b8beb07b0"


def frozen_rows() -> tuple[str, list[str]]:
    text = subprocess.run(
        ["git", "show", f"{FROZEN}:data/gt/gt_fixes.tsv"], cwd=REPO, check=True, capture_output=True
    ).stdout.decode("utf-8")
    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    return lines[0], lines[1:]


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def records(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {json.loads(line)["slug"]: line for line in fh if line.strip()}


def build(name: str, rows: list[int] | None) -> dict[str, object]:
    out = GT_ROOT / name
    if (out / "gt_tokens.jsonl").exists():
        raise SystemExit(f"эталон {out} уже собран")
    out.mkdir(parents=True, exist_ok=True)
    args = [
        str(PYTHON), "scripts/build_gt_tokens.py", "--out-dir", str(out),
        "--csv", str(DATASET / "strapi_output0709.csv"),
        "--clusters", str(ANALYSIS / "near_dup_clusters.json"),
        "--live", str(ANALYSIS / "plan_live_wines.json"),
        "--winery-map", str(ANALYSIS / "strapi_winery_map.txt"),
    ]  # fmt: skip
    header, table = frozen_rows()
    if rows is None:
        args.append("--no-fixes")
    else:
        chosen = [table[i - 1] for i in rows]
        (out / "gt_fixes.tsv").write_bytes(("\n".join([header, *chosen]) + "\n").encode("utf-8"))
        args += ["--fixes", str(out / "gt_fixes.tsv")]
    env = {
        **os.environ,
        "PYTHONPATH": str(SNAP),
        "SVS_DATA_DIR": str(SCANNER / "data"),
        "SVS_DATASET_DIR": str(DATASET),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    done = subprocess.run(
        args, cwd=SNAP, env=env, capture_output=True, text=True, encoding="utf-8", check=False
    )
    if done.returncode != 0:
        raise SystemExit(f"сборка {name}: {done.returncode}\n{done.stdout}\n{done.stderr}")
    gt = out / "gt_tokens.jsonl"
    info: dict[str, object] = {"name": name, "rows": rows, "sha1": sha1(gt), "stdout": done.stdout}
    nofix = GT_ROOT / "nofix" / "gt_tokens.jsonl"
    if rows is not None and nofix.is_file():
        old, new = records(nofix), records(gt)
        info["records"] = len(new)
        info["changed"] = sorted(s for s in new if old.get(s) != new[s])
        info["missing"] = sorted(set(old) - set(new))
    (out / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), "utf-8")
    return info


def main(argv: list[str]) -> int:
    name, *rest = argv
    if name == "nofix":
        info = build(name, None)
        ok = info["sha1"] == BASE_SHA1
        print(name, info["sha1"], "= bf6a8823" if ok else "≠ bf6a8823 — ОШИБКА")
        return 0 if ok else 1
    if name.startswith("row"):
        rows = [int(name[3:])]
    elif name == "pkg":
        rows = sorted({int(x) for x in rest})
    else:
        raise SystemExit(f"неизвестный эталон {name}")
    info = build(name, rows)
    print(name, rows, info["sha1"], "изменены", info.get("changed"), "пропали", info.get("missing"))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main(sys.argv[1:]))
