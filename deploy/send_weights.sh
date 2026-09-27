#!/usr/bin/env bash
# Отправка весов SigLIP на сервер с докачкой после обрыва. Пароль спрашивается дважды за запуск.
#
#     bash deploy/send_weights.sh root@IP ПОРТ
#     bash deploy/send_weights.sh root@IP ПОРТ /путь/к/siglip-vision-hub
#
# Оборвалось на середине — запустите ту же команду ещё раз: она посмотрит, сколько байт уже
# лежит на сервере, и дошлёт только хвост. Ничего не начинается заново.
#
# Почему не scp и не куски по файлам. Длинная передача до сервера встаёт (`stalled`), а scp
# после обрыва начинает файл с нуля. Досылка кусками-файлами это чинит, но делает по два
# соединения на кусок — при входе по паролю это сотня запросов пароля. Здесь на запуск ровно
# два соединения: первое спрашивает, сколько уже доехало, второе дописывает хвост и считает
# sha256 прямо на сервере.
set -euo pipefail

HOST=${1:-}
PORT=${2:-22}
SRC=${3:-$(cd "$(dirname "$0")/.." && pwd)/../siglip-vision-hub}
[ -n "$HOST" ] || { echo "нужен адрес: bash deploy/send_weights.sh root@IP [порт] [каталог]"; exit 1; }

CACHE_NAME=models--google--siglip2-so400m-patch14-384
REMOTE_HUB=${REMOTE_HUB:-/opt/svs/hf/hub}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
# Пауза в канале не должна выглядеть как разрыв: ssh сам подёргивает соединение.
SSH=(ssh -p "$PORT" -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -o TCPKeepAlive=yes)

ROOT="$SRC/$CACHE_NAME"
[ -d "$ROOT" ] || { echo "нет каталога $ROOT — сначала scripts/trim_vision_weights.py"; exit 1; }
SHA_REF=$(tr -d '[:space:]' < "$ROOT/refs/main")
SNAPSHOT="$ROOT/snapshots/$SHA_REF"
WEIGHTS="$SNAPSHOT/model.safetensors"
[ -f "$WEIGHTS" ] || { echo "нет весов $WEIGHTS"; exit 1; }

REMOTE_SNAPSHOT="$REMOTE_HUB/$CACHE_NAME/snapshots/$SHA_REF"
DEST="$REMOTE_SNAPSHOT/model.safetensors"
TOTAL=$(stat -c%s "$WEIGHTS" 2>/dev/null || stat -f%z "$WEIGHTS")

say "Готовлю сервер и смотрю, сколько уже доехало (пароль 1 из 2)"
# Мелкие файлы едут тем же соединением: отдельные scp стоили бы ещё по запросу пароля каждый.
HAVE=$(
  tar cf - -C "$SNAPSHOT" config.json preprocessor_config.json \
  | "${SSH[@]}" "$HOST" "
      mkdir -p '$REMOTE_SNAPSHOT' '$REMOTE_HUB/$CACHE_NAME/refs' &&
      printf '%s' '$SHA_REF' > '$REMOTE_HUB/$CACHE_NAME/refs/main' &&
      tar xf - -C '$REMOTE_SNAPSHOT' &&
      stat -c%s '$DEST' 2>/dev/null || echo 0
    " | tail -1
)
HAVE=${HAVE:-0}

printf 'всего %s МБ, на сервере %s МБ\n' "$((TOTAL / 1000000))" "$((HAVE / 1000000))"
if [ "$HAVE" -gt "$TOTAL" ]; then
  echo "на сервере файл больше исходного — он битый; удалите его и запустите снова:"
  echo "    ssh -p $PORT $HOST \"rm '$DEST'\""
  exit 1
fi

if [ "$HAVE" -lt "$TOTAL" ]; then
  LEFT=$(((TOTAL - HAVE) / 1000000))
  say "Дошлю $LEFT МБ (пароль 2 из 2). При 250 КБ/с это примерно $((LEFT / 15)) минут"
  echo "оборвётся — просто запустите команду ещё раз, продолжит с места"
  # Хвост подаёт `dd`, а не `tail`: у него есть `status=progress` — строка со скоростью и
  # числом байт раз в секунду. Без неё отправка выглядит как зависание, и это невыносимо.
  # `skip_bytes` есть не во всех сборках, поэтому запасной путь — прежний `tail`.
  if dd --help 2>/dev/null | grep -q skip_bytes; then
    FEED=(dd "if=$WEIGHTS" bs=1M "skip=$HAVE" iflag=skip_bytes status=progress)
  else
    echo "(в этой сборке dd нет skip_bytes — полосы не будет)"
    FEED=(tail -c "+$((HAVE + 1))" "$WEIGHTS")
  fi
  REMOTE_SHA=$("${FEED[@]}" \
    | "${SSH[@]}" "$HOST" "cat >> '$DEST' && sha256sum '$DEST' | cut -d' ' -f1" | tail -1)
else
  say "Файл уже целиком на сервере, проверяю сумму (пароль 2 из 2)"
  REMOTE_SHA=$("${SSH[@]}" "$HOST" "sha256sum '$DEST' | cut -d' ' -f1" | tail -1)
fi

say "Сверяю"
LOCAL_SHA=$(sha256sum "$WEIGHTS" | cut -d' ' -f1)
if [ "$LOCAL_SHA" != "$REMOTE_SHA" ]; then
  echo "sha256 разошлись:"
  echo "  у нас      $LOCAL_SHA"
  echo "  на сервере $REMOTE_SHA"
  echo "Файл дописался не тем хвостом. Удалите его на сервере и начните заново:"
  echo "    ssh -p $PORT $HOST \"rm '$DEST'\""
  exit 1
fi

cat <<EOF

Веса на месте, sha256 совпал. Дальше на сервере:
    chown -R svs:svs /opt/svs/hf
    bash /opt/svs/app/deploy/setup_vps.sh
EOF
