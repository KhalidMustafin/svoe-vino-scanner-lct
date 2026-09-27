#!/usr/bin/env bash
# Полевой стенд на VPS без видеокарты: окружение, Ollama, служба, файрвол.
#
# Запуск от root на чистой Ubuntu 22.04/24.04:
#     bash deploy/setup_vps.sh
#
# Скрипт идемпотентен — его можно гонять повторно. Он НЕ качает веса SigLIP и НЕ кладёт
# индекс: это делает deploy/push.sh с машины разработки (сервис веса не скачивает,
# local_files_only=True). Порядок: сначала push.sh, потом этот скрипт.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/svs/app}
DATA_DIR=${DATA_DIR:-/opt/svs/data}
ENV_FILE=${ENV_FILE:-/etc/svs-field.env}
SVS_USER=${SVS_USER:-svs}
VLM_MODEL=${VLM_MODEL:-qwen3.5:4b}
OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }

[ "$(id -u)" -eq 0 ] || { echo "нужен root: sudo bash $0"; exit 1; }
[ -d "$APP_DIR/app" ] || { echo "нет кода в $APP_DIR — сначала deploy/push.sh"; exit 1; }

# --------------------------------------------------------------------- железо
say "Проверяю железо"
RAM_MB=$(free -m | awk '/^Mem:/ {print $2}')
CORES=$(nproc)
DISK_GB=$(df -BG --output=avail /opt | tail -1 | tr -dc '0-9')
echo "ОЗУ ${RAM_MB} МБ · ядер ${CORES} · свободно на /opt ${DISK_GB} ГБ"
[ "$RAM_MB" -ge 7000 ] || warn "меньше 8 ГБ ОЗУ: SigLIP (2,5 ГБ) и Ollama (3,1 ГБ) не поместятся"
[ "$DISK_GB" -ge 20 ] || warn "меньше 20 ГБ свободно: окружение с torch и веса займут ~10 ГБ"
[ "$CORES" -ge 4 ] || warn "$CORES ядра: кадр пойдёт минуты — чтение этикетки на CPU упирается в ядра"

# Сосед по серверу. На 8 ГБ ОЗУ сканер и «Лоза» с её моделью одновременно не живут: у сканера
# только SigLIP и Ollama съедают около 6 ГБ. Скрипт ничего не выключает сам — это решение
# хозяина сервера, — но молчать об этом нельзя.
if command -v docker >/dev/null && docker ps --format '{{.Names}}' 2>/dev/null | grep -q .; then
  warn "на сервере работают контейнеры: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
  warn "при 8 ГБ ОЗУ на время полевого прогона их лучше погасить: docker compose down"
fi
if ss -tlnp 2>/dev/null | grep -q ':8080 '; then
  warn "порт 8080 уже занят — поменяйте SVS_PORT в $ENV_FILE"
fi

# На 8 ГБ подъём SigLIP идёт впритык, и первый же скан может уронить процесс по OOM.
# Файл подкачки дешевле, чем потерянный день полевого прогона.
if [ "$RAM_MB" -lt 12000 ] && ! swapon --show | grep -q .; then
  say "Добавляю 4 ГБ подкачки (ОЗУ меньше 12 ГБ)"
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# --------------------------------------------------------------------- пакеты
say "Ставлю пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates git jq ufw libgl1 libglib2.0-0 >/dev/null
# libgl1 и libglib — для opencv-python-headless: без них `import cv2` падает на голой Ubuntu.

id -u "$SVS_USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir /opt/svs "$SVS_USER"
mkdir -p "$DATA_DIR" /opt/svs/hf
chown -R "$SVS_USER:$SVS_USER" /opt/svs

# --------------------------------------------------------------------- uv и окружение
say "Собираю окружение (torch с индекса CPU, без CUDA-колёс)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh
fi
cd "$APP_DIR"
# --extra cpu вместо gpu: CUDA-сборка тянет 19 пакетов nvidia-* и без карты бесполезна.
sudo -u "$SVS_USER" env HOME=/opt/svs uv sync --extra cpu --extra cv --extra api
sudo -u "$SVS_USER" env HOME=/opt/svs .venv/bin/python -c "
import torch, torchvision, cv2, fastapi
from PIL import Image
import pillow_heif  # HEIC с айфона
print('torch', torch.__version__, '| torchvision', torchvision.__version__,
      '| cv2', cv2.__version__, '| heic ок')
"

# --------------------------------------------------------------------- веса SigLIP
# Сервис их не скачивает (local_files_only=True), поэтому качаем здесь: у ЦОД канал шире
# домашнего. Если с сервера до Hugging Face не достучаться — везём с машины разработки.
HF_DIR=${HF_HOME:-/opt/svs/hf}
if [ -d "$HF_DIR/hub/models--google--siglip2-so400m-patch14-384/snapshots" ]; then
  say "Веса SigLIP уже на месте"
else
  say "Качаю веса SigLIP (4,6 ГБ)"
  if ! sudo -u "$SVS_USER" env HOME=/opt/svs HF_HOME="$HF_DIR" .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/siglip2-so400m-patch14-384',
                  allow_patterns=['*.json', '*.txt', 'model.safetensors'])
print('веса скачаны')
"; then
    warn "Hugging Face недоступен. С машины разработки: bash deploy/push.sh $HOSTNAME --weights"
    warn "без весов сервис поднимется, но каждый кадр вернёт ошибку cv:"
  fi
fi

# Не «файлы на месте», а «модель поднимается». Разница поймала настоящую ошибку: веса лежали,
# а препроцессор не собирался без torchvision — и сервис жаловался на отсутствие весов.
say "Проверяю, что модель зрения действительно поднимается"
sudo -u "$SVS_USER" env HOME=/opt/svs HF_HOME="$HF_DIR" .venv/bin/python -c "
from transformers import AutoImageProcessor, SiglipVisionModel
name = 'google/siglip2-so400m-patch14-384'
model = SiglipVisionModel.from_pretrained(name, local_files_only=True)
AutoImageProcessor.from_pretrained(name, local_files_only=True)
print('зрительная башня и препроцессор в порядке:', sum(p.numel() for p in model.parameters()))
" || { echo "модель не поднимается — дальше идти бессмысленно"; exit 1; }

# --------------------------------------------------------------------- настройки
# Настройки кладём до Ollama: по ним видно, нужна ли она вообще.
if [ ! -f "$ENV_FILE" ]; then
  say "Кладу $ENV_FILE из образца — ПОМЕНЯЙТЕ SVS_FIELD_TOKEN"
  install -m 640 -o root -g "$SVS_USER" deploy/svs-field.env.example "$ENV_FILE"
fi
# Полевой контур включается флагом (с 24.09); в env, положенном раньше, его нет.
if ! grep -q '^SVS_FIELD=' "$ENV_FILE"; then
  say "Включаю полевой контур: дописываю SVS_FIELD=1 в $ENV_FILE"
  printf '\n# Полевой контур: страница съёмки и приём кадров (без флага их нет)\nSVS_FIELD=1\n' \
    >> "$ENV_FILE"
fi
VLM_TIMEOUT=$(grep -oP '^SVS_VLM_TIMEOUT_MS=\K.*' "$ENV_FILE" 2>/dev/null | tr -d '[:space:]' || true)

# --------------------------------------------------------------------- Ollama
if [ "${VLM_TIMEOUT:-}" = "0" ]; then
  # Режим «только картинка». Он придуман не для красоты: если канал до сервера не даёт
  # привезти модель чтения (3,4 ГБ), стенд всё равно собирает полевые кадры — а это и есть
  # его главная задача. Ответы будут слабее, каждый с флагом `vlm_disabled`, но сами кадры
  # потом прогоняются через полную цепочку на машине с видеокартой.
  say "Чтение этикетки выключено (SVS_VLM_TIMEOUT_MS=0) — Ollama не нужна, пропускаю"
  warn "сервис будет выбирать вино по одной картинке: точность заметно ниже замеров"
else

say "Ставлю Ollama"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh

# На сервере может уже стоять Ollama от другого проекта (у «Лозы» свой drop-in с
# OLLAMA_HOST=0.0.0.0 — иначе её контейнеры не достучатся). Адрес чужой настройки не трогаем:
# сломать соседа настройкой ради себя — худшее, что может сделать скрипт развёртывания.
mkdir -p /etc/systemd/system/ollama.service.d
NEIGHBOUR=$(grep -rl "OLLAMA_HOST" /etc/systemd/system/ollama.service.d/ 2>/dev/null \
  | grep -v 'field.conf' | head -1 || true)
{
  echo '[Service]'
  if [ -n "$NEIGHBOUR" ]; then
    warn "адрес Ollama оставляю как есть: его задаёт $NEIGHBOUR"
  else
    echo 'Environment="OLLAMA_HOST=127.0.0.1:11434"'
  fi
  # Один слот: параллельные слоты делят контекст и рушат кэш префикса, а на процессоре
  # обработка промпта и так самое дорогое место.
  echo 'Environment="OLLAMA_NUM_PARALLEL=1"'
  # Модель не выгружать: холодная загрузка на CPU — десятки секунд на каждый кадр.
  echo 'Environment="OLLAMA_KEEP_ALIVE=24h"'
  # Лимит загруженных моделей ставим, только если сосед не делит с нами Ollama: иначе мы
  # вытесняли бы его модель на каждом кадре, а он нашу.
  [ -n "$NEIGHBOUR" ] || echo 'Environment="OLLAMA_MAX_LOADED_MODELS=1"'
} > /etc/systemd/system/ollama.service.d/field.conf
systemctl daemon-reload
systemctl enable ollama >/dev/null

# Перезапуск — только если программа ollama цела. Установщик, скачанный по рваному каналу,
# оставляет обрубок: демон при этом продолжает работать из уже загруженного кода, а вот
# рестарт его убивает — systemd попытается запустить битый файл. Поэтому сначала проверка,
# и лучше остаться со старым окружением, чем без Ollama вообще.
OLLAMA_OK=1
if ! timeout 10 ollama --version >/dev/null 2>&1; then
  OLLAMA_OK=0
  warn "программа ollama не запускается (битый файл?): $(ls -l "$(command -v ollama)" 2>/dev/null | awk '{print $5" байт"}')"
  warn "службу не перезапускаю — иначе демон не поднимется; чинить: deploy/fix_ollama.sh"
fi
if [ "$OLLAMA_OK" = 1 ]; then
  # Именно restart: установщик уже поднял службу со старым окружением, и enable --now её не тронет.
  systemctl restart ollama
  sleep 3
fi

if ! curl -sf --max-time 5 http://127.0.0.1:11434/api/tags >/dev/null; then
  warn "Ollama не отвечает на 11434 — смотрите journalctl -u ollama"
fi

# Модель тянем через API демона, а не программой: API работает и при битом бинарнике, и
# докачивает начатое. Прогресс печатаем раз в несколько секунд, иначе строк будут тысячи.
if curl -sf --max-time 5 "$OLLAMA_URL/api/tags" 2>/dev/null | grep -q "\"$VLM_MODEL\""; then
  say "Модель чтения $VLM_MODEL уже на месте"
else
  say "Тяну модель чтения ($VLM_MODEL) — это 3,4 ГБ"
  curl -sN "$OLLAMA_URL/api/pull" -d "{\"model\":\"$VLM_MODEL\"}" \
    | grep --line-buffered -o '"completed":[0-9]*' \
    | awk -F: '{ now = systime(); if (now != last) { printf "  %.0f МБ\n", $2/1000000; fflush(); last = now } }'
  curl -sf --max-time 5 "$OLLAMA_URL/api/tags" | grep -q "\"$VLM_MODEL\"" \
    || { echo "модель $VLM_MODEL не скачалась"; exit 1; }
fi

fi  # конец ветки «чтение этикетки включено»

# --------------------------------------------------------------------- служба
say "Ставлю службу svs-field"
install -m 644 deploy/svs-field.service /etc/systemd/system/svs-field.service
systemctl daemon-reload
systemctl enable svs-field >/dev/null
systemctl restart svs-field

# --------------------------------------------------------------------- файрвол
say "Закрываю порты (наружу только SSH и 80)"
if command -v ufw >/dev/null; then
  ufw allow OpenSSH >/dev/null
  ufw allow 80/tcp >/dev/null
  yes | ufw enable >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------- ожидание прогрева
# Порт берём из настроек, а не из умолчания: стенд часто переносят на 80-й, и опрос
# несуществующего 8080 молча ждёт четверть часа, после чего печатает пустоту.
PORT=$(grep -oP '^SVS_PORT=\K.*' "$ENV_FILE" 2>/dev/null | tr -d '[:space:]' || true)
PORT=${PORT:-8080}
HEALTH="http://127.0.0.1:${PORT}/v1/health"
say "Жду прогрев на порту $PORT (на процессоре это минуты)"
for i in $(seq 1 90); do
  STATUS=$(curl -sf --max-time 5 "$HEALTH" | jq -r '.status // "нет"' 2>/dev/null || echo "нет")
  [ "$STATUS" = "ready" ] || [ "$STATUS" = "degraded" ] && break
  sleep 10
done
curl -sf --max-time 5 "$HEALTH" | jq '{status, degraded_reasons, warnings, formats}' \
  || warn "сервис не ответил на $HEALTH — смотрите journalctl -u svs-field -n 50"

cat <<EOF

Готово. Дальше:
  1) проверить живым кадром:  python3 scripts/preflight_field.py http://127.0.0.1:${PORT} кадр.jpg ключ
  2) открыть наружу:          поставьте Caddy (файл deploy/Caddyfile) — он даст https и пароль
  3) журнал службы:           journalctl -u svs-field -f
  4) кадры прогона:           $DATA_DIR/field
EOF
