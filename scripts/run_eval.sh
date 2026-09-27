#!/usr/bin/env bash
# Прогон оригинального participant_test.sh организатора против запущенного сервиса:
# проверка окружения и /v1/health → скрипт организатора → проверка формата → bench.judge.
#
#   scripts/run_eval.sh --images-dir <датасет>/eval/queries --manifest <датасет>/eval/queries.tsv \
#       --gt data/gt/public_gt.tsv --out runs/eval-public
#
# Скрипт организатора не меняется. Окружение Windows (Git Bash) приводится к тому, что он
# ждёт на Linux:
#   - ~/bin в начале PATH (там jq.exe);
#   - jq.exe пишет CRLF, и slug приезжает с хвостом \r. Если так, jq подменяется обёрткой
#     `jq -b` во временном каталоге (двоичный режим, только LF) — поведение jq то же;
#   - curl из mingw не открывает пути вида /c/..., поэтому каталог кадров передаётся как C:/...
# После прогона в predictions.jsonl не должно быть ни одного \r — иначе ошибка.
#
# Коды выхода: 0 — готово; 1 — сбой прогона или формата; 2 — ошибка аргументов или окружения.

set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
images_dir=""
manifest=""
gt=""
out=""
endpoint="http://127.0.0.1:8080/v1/eval/predict"
script=""
allow_degraded=0
python_bin=""

usage() {
  cat <<'EOF'
Usage: scripts/run_eval.sh --images-dir DIR --manifest FILE --gt FILE --out DIR
                           [--endpoint URL] [--script participant_test.sh]
                           [--python PATH] [--allow-degraded]

  --images-dir      каталог кадров (как у participant_test.sh)
  --manifest        queries.tsv: query_id<TAB>image_path
  --gt              эталон query_id<TAB>slug|__none__ (публичные кадры: data/gt/public_gt.tsv)
  --out             каталог прогона; predictions.jsonl и judge.json пишутся туда (не должны существовать)
  --endpoint        по умолчанию http://127.0.0.1:8080/v1/eval/predict
  --script          по умолчанию $SVS_DATASET_DIR/eval/participant_test.sh
  --python          интерпретатор для bench.judge (по умолчанию .venv проекта)
  --allow-degraded  прогонять и тогда, когда сервис не на цепочке замера: /v1/health не ready,
                    прогревочный скан с флагами, предупреждения настроек или provenance
                    не сходится (например, без VLM или на CPU) — не для отчёта

После прогона пишется service_stats.json: сколько ответов сервис дал за прогон, сколько без
текста этикетки и с какими флагами degraded (разница счётчиков /v1/health до и после).
EOF
}

# die "сообщение" [код выхода, по умолчанию 1]
die() {
  printf 'ОШИБКА: %s\n' "$1" >&2
  exit "${2:-1}"
}

bad_args() {
  printf 'ОШИБКА: %s\n\n' "$*" >&2
  usage >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --images-dir) [ "$#" -ge 2 ] || bad_args "--images-dir без значения"; images_dir="$2"; shift 2 ;;
    --manifest) [ "$#" -ge 2 ] || bad_args "--manifest без значения"; manifest="$2"; shift 2 ;;
    --gt) [ "$#" -ge 2 ] || bad_args "--gt без значения"; gt="$2"; shift 2 ;;
    --out) [ "$#" -ge 2 ] || bad_args "--out без значения"; out="$2"; shift 2 ;;
    --endpoint) [ "$#" -ge 2 ] || bad_args "--endpoint без значения"; endpoint="$2"; shift 2 ;;
    --script) [ "$#" -ge 2 ] || bad_args "--script без значения"; script="$2"; shift 2 ;;
    --python) [ "$#" -ge 2 ] || bad_args "--python без значения"; python_bin="$2"; shift 2 ;;
    --allow-degraded) allow_degraded=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) bad_args "неизвестный аргумент: $1" ;;
  esac
done

[ -n "$images_dir" ] || bad_args "нужен --images-dir"
[ -n "$manifest" ] || bad_args "нужен --manifest"
[ -n "$gt" ] || bad_args "нужен --gt"
[ -n "$out" ] || bad_args "нужен --out"

# Абсолютный путь: дальше интерпретатор запускается из корня проекта.
absolute() {
  case "$1" in
    /*|[A-Za-z]:/*|[A-Za-z]:\\*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}

# Путь для программ Windows (curl из mingw): C:/... вместо /c/...
native_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

images_dir=$(absolute "$images_dir")
manifest=$(absolute "$manifest")
gt=$(absolute "$gt")
out=$(absolute "$out")

[ -d "$images_dir" ] || die "нет каталога кадров: $images_dir" 2
[ -f "$manifest" ] || die "нет манифеста: $manifest" 2
[ -f "$gt" ] || die "нет эталона: $gt" 2

if [ -z "$script" ]; then
  dataset_dir=${SVS_DATASET_DIR:-"$repo_root/data/raw/dataset"}
  script="$dataset_dir/eval/participant_test.sh"
fi
script=$(absolute "$script")
[ -f "$script" ] || die "нет скрипта организатора: $script (задайте --script или SVS_DATASET_DIR)" 2

# ------------------------------------------------------------------ окружение
export PATH="$HOME/bin:$PATH"
for command_name in curl jq awk mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || die "нет команды $command_name (jq.exe ожидается в ~/bin)" 2
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  die "нет sha256sum или shasum" 2
fi

probe=$(mktemp -d) || die "mktemp -d не сработал" 2
case "$probe" in
  /tmp/*|/private/tmp/*|/var/tmp/*|/var/folders/*|/private/var/folders/*) ;;
  *) rm -rf -- "$probe"; die "mktemp -d даёт $probe, а скрипт организатора ждёт каталог в /tmp" 2 ;;
esac
shim_dir="$probe"
cleanup() {
  rm -rf -- "$shim_dir"
}
trap cleanup EXIT HUP INT TERM

has_cr() {
  case "$1" in
    *$'\r'*) return 0 ;;
    *) return 1 ;;
  esac
}

jq_probe=$(printf '{"slug":"x"}' | jq -er '.slug'; printf '.')
if has_cr "$jq_probe"; then
  real_jq=$(command -v jq)
  printf '#!/usr/bin/env bash\nexec "%s" -b "$@"\n' "$real_jq" > "$shim_dir/jq"
  chmod +x "$shim_dir/jq"
  export PATH="$shim_dir:$PATH"
  jq_probe=$(printf '{"slug":"x"}' | jq -er '.slug'; printf '.')
  has_cr "$jq_probe" && die "jq пишет \\r даже с -b: прогон дал бы slug с хвостом \\r" 2
  printf 'jq: %s пишет CRLF — подменён обёрткой «jq -b» (%s)\n' "$real_jq" "$shim_dir/jq"
fi

# ------------------------------------------------------------------ сервис
base_url=${endpoint%/v1/eval/predict}
health=$(curl --silent --show-error --connect-timeout 5 --max-time 10 "$base_url/v1/health") ||
  die "сервис не отвечает на $base_url/v1/health — запустите python -m app.api" 2
health_field() {
  printf '%s' "$health" | jq -r "$1" | tr -d '\r'
}

status=$(health_field '.status // "unknown"')
printf 'health: %s\n' "$status"
# Отчётный прогон — только на цепочке замера. Кроме status проверяется и то, что сервис мог
# бы не учесть в статусе (старая версия): флаги прогревочного скана, предупреждения настроек
# (другая VLM, урезанный бюджет) и сверка сборки с моделью resolve.
problems=()
[ "$status" = "ready" ] || problems+=("status=$status (ждём ready)")
reasons=$(health_field '(.degraded_reasons // []) | join(", ")')
[ -z "$reasons" ] || problems+=("причины degraded: $reasons")
warm_flags=$(health_field '(.warm.scan.degraded // []) | join(", ")')
[ -z "$warm_flags" ] || problems+=("прогревочный скан с флагами: $warm_flags")
settings_warnings=$(health_field '(.warnings // []) | join(" | ")')
[ -z "$settings_warnings" ] || problems+=("предупреждения настроек: $settings_warnings")
consistent=$(health_field 'if .provenance.consistent == false then "false" else "true" end')
[ "$consistent" = "true" ] ||
  problems+=("сборка не та, на которой обучен resolve: $(health_field '.provenance | tojson')")
if [ "${#problems[@]}" -gt 0 ]; then
  for problem in "${problems[@]}"; do
    printf '  - %s\n' "$problem" >&2
  done
  if [ "$allow_degraded" -eq 1 ]; then
    printf 'ВНИМАНИЕ: сервис не на цепочке замера (выше) — цифры прогона не для отчёта\n'
  else
    die "сервис не готов к отчётному прогону (выше); --allow-degraded — прогон не для отчёта" 2
  fi
fi
scans_before=$(health_field '.scans // {} | tojson')

# ------------------------------------------------------------------ прогон
mkdir -p "$out" || die "не создать $out" 2
predictions="$out/predictions.jsonl"
[ ! -e "$predictions" ] || die "уже есть $predictions — выберите другой --out" 2

started=$(date +%s)
bash "$script" \
  --images-dir "$(native_path "$images_dir")" \
  --manifest "$manifest" \
  --endpoint "$endpoint" \
  --output "$predictions"
rc=$?
[ "$rc" -eq 0 ] || die "participant_test.sh завершился с кодом $rc"
[ -f "$predictions" ] || die "скрипт не записал $predictions"
printf 'прогон: %s с, строк: %s\n' "$(( $(date +%s) - started ))" "$(wc -l < "$predictions" | tr -d ' ')"

# ------------------------------------------------------------------ счётчики сервиса
# Прогрев проверяет цепочку на одном синтетическом кадре; настоящие этикетки длиннее, и VLM
# может не успевать уже на кадрах скрипта. Разница счётчиков /v1/health до и после прогона
# показывает, сколько ответов ушло без чтения этикетки и почему.
if health=$(curl --silent --show-error --connect-timeout 5 --max-time 10 "$base_url/v1/health"); then
  scans_after=$(health_field '.scans // {} | tojson')
  stats=$(jq -cn --argjson a "$scans_before" --argjson b "$scans_after" '
    def n(x): x // 0;
    {scans: (n($b.total) - n($a.total)),
     errors: (n($b.errors) - n($a.errors)),
     null_slug: (n($b.null_slug) - n($a.null_slug)),
     text_read: (n($b.text_read) - n($a.text_read)),
     degraded: (($b.degraded // {}) | to_entries
                | map({key, value: (.value - n($a.degraded[.key]))})
                | map(select(.value > 0)) | from_entries)}' | tr -d '\r')
  [ -n "$stats" ] || stats='{}'
  printf '%s\n' "$stats" > "$out/service_stats.json"
  printf 'сервис: %s\n' "$stats"
  scans=$(printf '%s' "$stats" | jq -r '.scans // 0' | tr -d '\r')
  no_text=$(printf '%s' "$stats" | jq -r '(.scans // 0) - (.text_read // 0)' | tr -d '\r')
  if [ "${no_text:-0}" -gt 0 ]; then
    printf 'ВНИМАНИЕ: %s ответов из %s — без прочитанного текста этикетки (флаги выше)\n' \
      "$no_text" "$scans"
  fi
else
  printf 'ВНИМАНИЕ: /v1/health после прогона не ответил — счётчики сервиса не сняты\n'
fi

# ------------------------------------------------------------------ формат
if grep -q $'\r' "$predictions"; then
  die "в $predictions есть символ \\r — строки или slug испорчены CRLF"
fi
if grep -qF '\r' "$predictions"; then
  die "в значениях $predictions есть экранированный \\r — slug приехал с хвостом CRLF"
fi

# ------------------------------------------------------------------ судья
if [ -z "$python_bin" ]; then
  for candidate in "$repo_root/.venv/Scripts/python.exe" "$repo_root/.venv/bin/python"; do
    if [ -x "$candidate" ]; then
      python_bin="$candidate"
      break
    fi
  done
fi
[ -n "$python_bin" ] || python_bin=python

cd "$repo_root" || die "нет каталога проекта $repo_root" 2
PYTHONIOENCODING=utf-8 "$python_bin" -m bench.judge \
  --pred "$(native_path "$predictions")" \
  --gt "$(native_path "$gt")" \
  --out "$(native_path "$out/judge.json")"
rc=$?
[ "$rc" -eq 0 ] || die "bench.judge завершился с кодом $rc"
printf 'готово: %s, %s\n' "$predictions" "$out/judge.json"
