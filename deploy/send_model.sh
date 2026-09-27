#!/usr/bin/env bash
# Отправка модели чтения из локальной Ollama на сервер, с докачкой после обрыва.
#
#     bash deploy/send_model.sh root@IP ПОРТ
#     bash deploy/send_model.sh root@IP ПОРТ qwen3.5:4b
#
# Когда реестр Ollama с сервера недоступен (или провайдер срезал канал), модель везём с
# машины разработки: она там уже скачана. Забираются файлы из каталога Ollama как есть —
# байт в байт те же, что отдал бы реестр, и демон проверит их по контрольной сумме, потому
# что имя блоба и есть его sha256.
#
# Пароль спрашивается дважды за запуск. Оборвалось — запустите ту же команду ещё раз:
# посмотрит, сколько байт уже на сервере, и дошлёт хвост.
set -euo pipefail

HOST=${1:-}
PORT=${2:-22}
MODEL=${3:-qwen3.5:4b}
[ -n "$HOST" ] || { echo "нужен адрес: bash deploy/send_model.sh root@IP [порт] [модель]"; exit 1; }

LOCAL_MODELS=${LOCAL_MODELS:-$HOME/.ollama/models}
REMOTE_MODELS=${REMOTE_MODELS:-/usr/share/ollama/.ollama/models}
OWNER=${OWNER:-ollama:ollama}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
SSH=(ssh -p "$PORT" -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes)

NAME=${MODEL%%:*}
TAG=${MODEL##*:}
MANIFEST="$LOCAL_MODELS/manifests/registry.ollama.ai/library/$NAME/$TAG"
[ -f "$MANIFEST" ] || { echo "нет манифеста $MANIFEST — сначала скачайте модель локально: ollama pull $MODEL"; exit 1; }

# Разбираем манифест без jq: он есть не везде, а формат простой.
DIGESTS=$(grep -o 'sha256:[0-9a-f]\{64\}' "$MANIFEST" | sort -u)
BIG=""
BIG_SIZE=0
for d in $DIGESTS; do
  f="$LOCAL_MODELS/blobs/sha256-${d#sha256:}"
  [ -f "$f" ] || { echo "нет блоба $f"; exit 1; }
  s=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  [ "$s" -gt "$BIG_SIZE" ] && { BIG=$d; BIG_SIZE=$s; }
done
BIG_FILE="$LOCAL_MODELS/blobs/sha256-${BIG#sha256:}"
say "Модель $MODEL: $(echo "$DIGESTS" | wc -l) файлов, тяжёлый — $((BIG_SIZE / 1000000)) МБ"

REMOTE_MANIFEST_DIR="$REMOTE_MODELS/manifests/registry.ollama.ai/library/$NAME"
REMOTE_BIG="$REMOTE_MODELS/blobs/sha256-${BIG#sha256:}"

say "Мелкие файлы и манифест (пароль 1 из 2)"
# Всё мелкое — одним соединением: каждое отдельное стоило бы ещё одного запроса пароля.
SMALL=""
for d in $DIGESTS; do
  [ "$d" = "$BIG" ] && continue
  SMALL="$SMALL blobs/sha256-${d#sha256:}"
done
HAVE=$(
  tar cf - -C "$LOCAL_MODELS" $SMALL 2>/dev/null \
  | "${SSH[@]}" "$HOST" "
      mkdir -p '$REMOTE_MODELS/blobs' '$REMOTE_MANIFEST_DIR' &&
      tar xf - -C '$REMOTE_MODELS' &&
      stat -c%s '$REMOTE_BIG' 2>/dev/null || echo 0
    " | tail -1
)
HAVE=${HAVE:-0}
"${SSH[@]}" "$HOST" "cat > '$REMOTE_MANIFEST_DIR/$TAG'" < "$MANIFEST" || true

printf 'тяжёлый файл: всего %s МБ, на сервере %s МБ\n' "$((BIG_SIZE / 1000000))" "$((HAVE / 1000000))"
if [ "$HAVE" -gt "$BIG_SIZE" ]; then
  echo "на сервере файл больше исходного — он битый, удалите и повторите:"
  echo "    ssh -p $PORT $HOST \"rm '$REMOTE_BIG'\""
  exit 1
fi

if [ "$HAVE" -lt "$BIG_SIZE" ]; then
  LEFT=$(((BIG_SIZE - HAVE) / 1000000))
  say "Дошлю $LEFT МБ (пароль 2 из 2). При 250 КБ/с это примерно $((LEFT / 15)) минут"
  if dd --help 2>/dev/null | grep -q skip_bytes; then
    FEED=(dd "if=$BIG_FILE" bs=1M "skip=$HAVE" iflag=skip_bytes status=progress)
  else
    FEED=(tail -c "+$((HAVE + 1))" "$BIG_FILE")
  fi
  REMOTE_SHA=$("${FEED[@]}" \
    | "${SSH[@]}" "$HOST" "cat >> '$REMOTE_BIG' && sha256sum '$REMOTE_BIG' | cut -d' ' -f1" | tail -1)
else
  say "Файл уже целиком на сервере, проверяю (пароль 2 из 2)"
  REMOTE_SHA=$("${SSH[@]}" "$HOST" "sha256sum '$REMOTE_BIG' | cut -d' ' -f1" | tail -1)
fi

say "Сверяю"
# Имя блоба — это и есть его sha256, так что эталон брать неоткуда не нужно.
if [ "$REMOTE_SHA" != "${BIG#sha256:}" ]; then
  echo "sha256 не сошёлся: ждали ${BIG#sha256:}, получили $REMOTE_SHA"
  echo "удалите файл на сервере и запустите снова: ssh -p $PORT $HOST \"rm '$REMOTE_BIG'\""
  exit 1
fi

cat <<EOF

Модель на месте и сошлась по контрольной сумме. На сервере осталось:
    chown -R $OWNER $REMOTE_MODELS
    curl -s localhost:11434/api/tags | grep -o '"name":"[^"]*"'   # должна появиться $MODEL
    sed -i 's/^SVS_VLM_TIMEOUT_MS=.*/SVS_VLM_TIMEOUT_MS=540000/' /etc/svs-field.env
    systemctl restart svs-field
EOF
