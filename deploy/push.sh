#!/usr/bin/env bash
# Отправка стенда на VPS с машины разработки. Работает в Git Bash на Windows: нужен только
# ssh и tar, rsync в Git Bash не ставится.
#
#     bash deploy/push.sh root@192.0.2.10              # код + индекс + фото каталога (~50 МБ)
#     bash deploy/push.sh root@192.0.2.10 --weights    # то же и веса SigLIP (4,6 ГБ, долго)
#
# Веса обычно гнать не надо: `setup_vps.sh` скачивает их на сервере с Hugging Face — у ЦОД
# канал шире домашнего. `--weights` нужен, если с сервера до HF не достучаться.
set -euo pipefail

HOST=${1:-}
WITH_WEIGHTS=${2:-}
[ -n "$HOST" ] || { echo "нужен адрес: bash deploy/push.sh root@IP [--weights]"; exit 1; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
REMOTE_APP=${REMOTE_APP:-/opt/svs/app}
REMOTE_DATA=${REMOTE_DATA:-/opt/svs/data}
REMOTE_HF=${REMOTE_HF:-/opt/svs/hf}
# Код может лежать в отдельном рабочем дереве (git worktree), а данные — в основном.
DATA_SRC=${DATA_SRC:-$REPO/data}
HF_SRC=${HF_SRC:-$HOME/.cache/huggingface/hub}
MODEL_DIR=models--google--siglip2-so400m-patch14-384

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Как в deploy/pack_data.sh: карта адаптера поиска index/cv-adapter-lw.npz — обязательный файл пути
# по умолчанию (SVS_CANDIDATE, 26.09), без неё сервис не стартует.
REQUIRED=(index/visual-s2so400m.npz index/cv-adapter-lw.npz index/lexicon.json gt/gt_tokens.jsonl)
for name in "${REQUIRED[@]}"; do
  [ -f "$DATA_SRC/$name" ] || {
    echo "нет $DATA_SRC/$name"
    echo "если данные в соседнем дереве: DATA_SRC=/путь/к/svoe-vino-scanner/data bash $0 $HOST"
    exit 1
  }
done

say "Готовлю каталоги"
ssh "$HOST" "mkdir -p $REMOTE_APP $REMOTE_DATA/index $REMOTE_DATA/gt $REMOTE_DATA/catalog $REMOTE_HF/hub"

say "Код"
tar czf - -C "$REPO" \
  --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' --exclude='data' --exclude='runs' \
  . | ssh "$HOST" "tar xzf - -C $REMOTE_APP"

say "Индекс, карта адаптера, словарь, признаки каталога (~30 МБ)"
tar czf - -C "$DATA_SRC" "${REQUIRED[@]}" \
  | ssh "$HOST" "tar xzf - -C $REMOTE_DATA"

if [ -d "$DATA_SRC/catalog/photos_small" ]; then
  say "Пересжатые фото каталога (~19 МБ)"
  tar czf - -C "$DATA_SRC" catalog/photos_small | ssh "$HOST" "tar xzf - -C $REMOTE_DATA"
else
  echo "(фото каталога нет — сначала python scripts/make_photo_pack.py, иначе карточка будет без картинки)"
fi

if [ "$WITH_WEIGHTS" = "--weights" ]; then
  [ -d "$HF_SRC/$MODEL_DIR" ] || { echo "нет весов в $HF_SRC/$MODEL_DIR"; exit 1; }
  say "Веса SigLIP — 4,6 ГБ, это надолго и без возобновления"
  tar czf - -C "$HF_SRC" "$MODEL_DIR" | ssh "$HOST" "tar xzf - -C $REMOTE_HF/hub"
fi

ssh "$HOST" "chown -R svs:svs /opt/svs 2>/dev/null || true"

cat <<EOF

Готово. Дальше на сервере:
    sudo bash $REMOTE_APP/deploy/setup_vps.sh
EOF
