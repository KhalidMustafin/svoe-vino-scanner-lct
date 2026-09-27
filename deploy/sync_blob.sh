#!/usr/bin/env bash
# Доведение большого файла на сервере до эталона: сравнение по блокам и досылка различий.
#
#     bash deploy/sync_blob.sh root@IP ПОРТ            # модель чтения qwen3.5:4b
#     bash deploy/sync_blob.sh root@IP ПОРТ qwen3.5:4b
#
# Зачем. Ollama качает блоб большими кусками и пишет их по местам в заранее растянутом файле:
# размер сразу полный, а непривезённые куски — нули. Поэтому «докачать хвост» нельзя, дыры
# сидят в середине. Слать заново 3,39 ГБ по срезанному каналу — сутки, а настоящих данных там
# уже немало.
#
# Здесь обе стороны считают sha256 по блокам (по умолчанию 16 МиБ), список различий считается
# на месте, и через одно соединение уезжают только несовпавшие блоки — каждый ложится по
# своему смещению. Пароль спрашивается дважды: первый раз на сверку, второй на запись.
# В конце сервер считает сумму целого файла и, если она равна имени блоба, переименовывает его
# из `-partial` в рабочий и убирает следы недокачки.
set -euo pipefail

HOST=${1:-}
PORT=${2:-22}
MODEL=${3:-qwen3.5:4b}
BS=${BS:-16777216}
[ -n "$HOST" ] || { echo "нужен адрес: bash deploy/sync_blob.sh root@IP [порт] [модель]"; exit 1; }

LOCAL_MODELS=${LOCAL_MODELS:-$HOME/.ollama/models}
REMOTE_MODELS=${REMOTE_MODELS:-/usr/share/ollama/.ollama/models}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
SSH=(ssh -p "$PORT" -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes)
PY=${PY:-python}
command -v "$PY" >/dev/null || PY=python3

NAME=${MODEL%%:*}
TAG=${MODEL##*:}
MANIFEST="$LOCAL_MODELS/manifests/registry.ollama.ai/library/$NAME/$TAG"
[ -f "$MANIFEST" ] || { echo "нет манифеста $MANIFEST"; exit 1; }

DIGESTS=$(grep -o 'sha256:[0-9a-f]\{64\}' "$MANIFEST" | sort -u)
BIG=""; BIG_SIZE=0
for d in $DIGESTS; do
  f="$LOCAL_MODELS/blobs/sha256-${d#sha256:}"
  [ -f "$f" ] || { echo "нет блоба $f"; exit 1; }
  s=$(stat -c%s "$f")
  [ "$s" -gt "$BIG_SIZE" ] && { BIG=$d; BIG_SIZE=$s; }
done
SHA=${BIG#sha256:}
LOCAL_BIG="$LOCAL_MODELS/blobs/sha256-$SHA"
REMOTE_BIG="$REMOTE_MODELS/blobs/sha256-$SHA"
BLOCKS=$(( (BIG_SIZE + BS - 1) / BS ))
say "Модель $MODEL: блоб $((BIG_SIZE / 1000000)) МБ, $BLOCKS блоков по $((BS / 1048576)) МиБ"

# Хэшер: один и тот же код считает блоки и здесь, и на сервере — иначе сравнивать нечего.
HASHER='
import hashlib, sys
path, bs = sys.argv[1], int(sys.argv[2])
try:
    handle = open(path, "rb")
except OSError:
    sys.exit(0)
index = 0
while True:
    chunk = handle.read(bs)
    if not chunk:
        break
    print(index, hashlib.sha256(chunk).hexdigest()[:16])
    index += 1
'
say "Считаю блоки здесь"
LOCAL_LIST=$(printf '%s' "$HASHER" | "$PY" - "$LOCAL_BIG" "$BS")

say "Считаю блоки на сервере (пароль 1 из 2, считает пару минут)"
# Имя выбирается на сервере: недокачанный блоб лежит под `-partial`, готовый — без него.
# Двух попыток быть не может: скрипт приходит по stdin, второй раз его уже не прочитать.
REMOTE_LIST=$(printf '%s' "$HASHER" | "${SSH[@]}" "$HOST" \
  "F='$REMOTE_BIG-partial'; [ -f \"\$F\" ] || F='$REMOTE_BIG'; python3 - \"\$F\" '$BS' 2>/dev/null || true")

MISSING=$(
  { printf '%s\n' "$LOCAL_LIST" | sed 's/^/L /'; printf '%s\n' "$REMOTE_LIST" | sed 's/^/R /'; } \
  | awk '$1=="L"{want[$2]=$3} $1=="R"{have[$2]=$3}
         END{ for (i in want) if (want[i] != have[i]) print i }' \
  | sort -n
)
COUNT=$(printf '%s' "$MISSING" | grep -c . || true)
# В одну строку: номера идут по одному в строке, а в команде для сервера перенос строки
# разрывает её на куски — и второй номер выполняется как отдельная команда.
MISSING_ARGS=$(printf '%s ' $MISSING)
if [ "$COUNT" = "0" ]; then
  say "Все блоки совпали — досылать нечего"
else
  say "Не совпало $COUNT блоков из $BLOCKS — это $((COUNT * BS / 1000000)) МБ"
fi

# Писатель на сервере: принимает блоки по порядку и кладёт каждый по своему смещению.
WRITER=$(printf '%s' '
import os, sys
path, bs = sys.argv[1], int(sys.argv[2])
indexes = [int(x) for x in sys.argv[3:]]
size = os.path.getsize(path)
handle = open(path, "r+b")
for index in indexes:
    need = min(bs, size - index * bs)
    data = sys.stdin.buffer.read(need)
    if len(data) != need:
        sys.exit("оборвалось на блоке %d" % index)
    handle.seek(index * bs)
    handle.write(data)
handle.close()
' | base64 -w0)

SMALL=""
for d in $DIGESTS; do
  [ "$d" = "$BIG" ] && continue
  SMALL="$SMALL blobs/sha256-${d#sha256:}"
done

say "Досылаю блоки и мелкие файлы (пароль 2 из 2)"
REMOTE_SHA=$(
  {
    for i in $MISSING; do
      dd "if=$LOCAL_BIG" bs="$BS" skip="$i" count=1 status=none
    done
  } | "${SSH[@]}" "$HOST" "
      set -e
      mkdir -p '$REMOTE_MODELS/blobs' '$REMOTE_MODELS/manifests/registry.ollama.ai/library/$NAME'
      [ -f '$REMOTE_BIG-partial' ] || [ -f '$REMOTE_BIG' ] || \
        python3 -c \"open('$REMOTE_BIG-partial','wb').truncate($BIG_SIZE)\"
      F='$REMOTE_BIG-partial'; [ -f \"\$F\" ] || F='$REMOTE_BIG'
      echo '$WRITER' | base64 -d > /tmp/svs_writer.py
      python3 /tmp/svs_writer.py \"\$F\" '$BS' $MISSING_ARGS
      rm -f /tmp/svs_writer.py
      sha256sum \"\$F\" | cut -d' ' -f1
    " | tail -1
)

say "Сверяю"
if [ "$REMOTE_SHA" != "$SHA" ]; then
  echo "сумма не сошлась: ждали $SHA, получили $REMOTE_SHA"
  echo "запустите команду ещё раз — сверка по блокам покажет, что осталось"
  exit 1
fi

say "Ставлю модель на место (пароль 3: мелкие файлы и манифест)"
tar cf - -C "$LOCAL_MODELS" $SMALL manifests/registry.ollama.ai/library/"$NAME"/"$TAG" \
  | "${SSH[@]}" "$HOST" "
      tar xf - -C '$REMOTE_MODELS' &&
      { [ -f '$REMOTE_BIG' ] || mv '$REMOTE_BIG-partial' '$REMOTE_BIG'; } &&
      rm -f '$REMOTE_BIG'-partial* &&
      chown -R ollama:ollama '$REMOTE_MODELS' &&
      ls -lh '$REMOTE_BIG' | awk '{print \"блоб на месте:\", \$5}'
    "

cat <<EOF

Модель собрана и сошлась по контрольной сумме. Останется только починить программу ollama,
после чего:
    curl -s localhost:11434/api/tags | grep -o '"name":"[^"]*"'
    sed -i 's/^SVS_VLM_TIMEOUT_MS=.*/SVS_VLM_TIMEOUT_MS=540000/' /etc/svs-field.env
    systemctl restart svs-field
EOF
