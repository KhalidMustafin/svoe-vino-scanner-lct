#!/usr/bin/env bash
# Цепочка стенда Э6: для каждого снимка snap/e6_<имя> — v2, kr, ooc по очереди, эталон 899d5db3
# (своя копия в runs/e6/gt/base), затем решение кадров kr кодом снимка (`e6_redecide.py`:
# «как есть» = стенд и «без водяного знака»).
#
#     bash e6_chain.sh base i1 i2 …
set -u
PY="<корень>/svoe-vino-scanner/.venv/Scripts/python.exe"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCANNER="<корень>/svoe-vino-scanner"
SNAPS="$SCANNER/runs/field25/iters/snap"
L="$SCANNER/runs/field25/iters/runs/e6"
GT="$L/gt/base/gt_tokens.jsonl"
mkdir -p "$L"
for name in "$@"; do
  for s in v2 kr ooc; do
    (cd "$SNAPS/e6_$name" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/e6_stand.py" "$name" "$s" "$GT" > "$L/${name}_$s.log" 2>&1)
    echo "$name $s стенд $?"
  done
  (cd "$SNAPS/e6_$name" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/e6_redecide.py" "$L/${name}_kr" "$GT" >> "$L/${name}_kr.log" 2>&1)
  echo "$name kr решение $?"
done
