#!/usr/bin/env bash
# Стенд базы Э5: снимок acc-freeze (snap/e5_base) и эталон 899d5db3 — v2 и kr по очереди, затем
# решение кадров кодом снимка (`e4_redecide.py`: «как есть» и «без водяного знака»).
#
#     bash e5_chain.sh
set -u
PY="<корень>/svoe-vino-scanner/.venv/Scripts/python.exe"
HERE="$(cd "$(dirname "$0")" && pwd)"
ACC="$HERE/../2026-09-25_acc"
SCANNER="<корень>/svoe-vino-scanner"
SNAP="$SCANNER/runs/field25/iters/snap/e5_base"
L="$SCANNER/runs/field25/iters/runs/e5"
GT="$SCANNER/runs/field25/iters/runs/e4/gt/pkg/gt_tokens.jsonl"
mkdir -p "$L"
for s in v2 kr; do
  (cd "$SNAP" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$HERE/e5_stand.py" base "$s" "$GT" > "$L/base_$s.log" 2>&1)
  echo "base $s стенд $?"
  (cd "$SNAP" && PYTHONDONTWRITEBYTECODE=1 "$PY" "$ACC/e4_redecide.py" "$L/base_$s" "$GT" >> "$L/base_$s.log" 2>&1)
  echo "base $s решение $?"
done
