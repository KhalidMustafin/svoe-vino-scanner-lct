#!/usr/bin/env bash
# Цепочка стенда Э4: прогон = <имя> <снимок> <эталон>; для каждого — v2 и kr по очереди, затем
# решение кадров кодом снимка (`e4_redecide.py`; у ворот 0 не нужно — там код до Э1).
#
#     bash e4_chain.sh "base base nofix" "row01 base row01" …
# Эталон: имя папки в runs/e4/gt или «data» — общий data/gt/gt_tokens.jsonl (bf6a8823).
set -u
PY="<корень>/svoe-vino-scanner/.venv/Scripts/python.exe"
HERE="$(cd "$(dirname "$0")" && pwd)"
SCANNER="<корень>/svoe-vino-scanner"
SNAPS="$SCANNER/runs/field25/iters/snap"
L="$SCANNER/runs/field25/iters/runs/e4"
mkdir -p "$L"
for spec in "$@"; do
  read -r name snap gtname <<< "$spec"
  if [ "$gtname" = "data" ]; then
    gt="$SCANNER/data/gt/gt_tokens.jsonl"
  else
    gt="$L/gt/$gtname/gt_tokens.jsonl"
  fi
  for s in v2 kr; do
    (cd "$SNAPS/e4_$snap" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/e4_stand.py" "$name" "$s" "$gt" > "$L/${name}_$s.log" 2>&1)
    echo "$name $s стенд $?"
    if [ "$name" != "gate0" ]; then
      (cd "$SNAPS/e4_$snap" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/e4_redecide.py" "$L/${name}_$s" "$gt" >> "$L/${name}_$s.log" 2>&1)
      echo "$name $s решение $?"
    fi
  done
done
