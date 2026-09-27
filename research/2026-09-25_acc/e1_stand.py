"""Стенд Э1: `iterbench.py --no-gpu` со снимка кода, ключ кэша CV — как у дампов (PREREG, §4).

Ключ кэша CV стенда хэширует `service.py` и `config.py` снимка. У `after-search` они другие, чем у
`snap/iter20`, которым записаны `pfix_final`, `kr_holdout` и `ooc_v2prod`, хотя `app/features`,
`app/normalize` и `scripts/build_index.py` совпадают (без учёта CRLF), а при попадании в кэш
`_search` вовсе не вызывается. Поэтому хэш кода CV здесь подменяется хэшем `snap/iter20` — и
только он: всё остальное делает сам `iterbench.py`.

Сети нет: `SVS_OLLAMA_URL` указывает на закрытый порт, промах кэша чтений в `--no-gpu` — сбой кадра.

    cd <снимок>; python e1_stand.py <имя прогона> <набор: v2|kr|ooc>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"<корень>")
SCANNER = ROOT / "svoe-vino-scanner"
ITERS = SCANNER / "runs" / "field25" / "iters"
FD = ROOT / "field_dataset"
#: `cv_hash` из `run_info.json` дампов `pfix_final`, `kr_holdout`, `ooc_v2prod` (код `snap/iter20`).
DUMP_CV_HASH = "579f3799c4248cea4947d50d7b1a89536e2ef1ed"
DUMP_CV_KEY = "ae2c4db886d1b1e9"
SETS = {"v2": "catalog_v2", "kr": "krasnostop_v1", "ooc": "ooc_v2"}
MODEL = SCANNER / "runs" / "goal" / "iter20" / "resolve" / "models" / "vlm35.json"


def main() -> int:
    run_name, set_key = sys.argv[1], sys.argv[2]
    run_dir = ITERS / "runs" / "e1" / f"{run_name}_{set_key}"
    os.environ["SVS_MAIN_DATA_DIR"] = str(SCANNER / "data")
    os.environ["SVS_OLLAMA_URL"] = "http://127.0.0.1:9"  # закрытый порт: Ollama стенд не трогает
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ITERS))
    import iterbench

    original = iterbench.code_hash

    def code_hash(globs: list[str]) -> str:
        return DUMP_CV_HASH if "app/api/service.py" in globs else original(globs)

    iterbench.code_hash = code_hash
    sys.argv = [
        "iterbench.py",
        "--run-dir", str(run_dir),
        "--model", str(MODEL),
        "--base-index", str(SCANNER / "data" / "index" / "visual-s2so400m.npz"),
        "--manifest", str(FD / "sets" / SETS[set_key] / "manifest.tsv"),
        "--images-root", str(FD),
        "--cache-dir", str(ITERS / "cache"),
        "--gt-ext", str(SCANNER / "data" / "gt" / "gt_tokens.jsonl"),
        "--photo-map-ext", str(SCANNER / "data" / "catalog" / "slug_photo_map.csv"),
        "--no-gpu",
    ]  # fmt: skip
    code = iterbench.main()
    info = (run_dir / "run_info.json").read_text(encoding="utf-8")
    if DUMP_CV_KEY not in info:
        print(f"ключ CV не {DUMP_CV_KEY}: стенд читал не тот кэш", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
