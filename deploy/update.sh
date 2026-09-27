#!/usr/bin/env bash
# Обновление стенда на сервере: забрать код, дособрать окружение, перезапустить службу.
#
#     sudo bash /opt/svs/app/deploy/update.sh              # текущая ветка
#     sudo bash /opt/svs/app/deploy/update.sh <ветка>          # переключиться на другую ветку
#
# Данные (индекс, признаки, фото) не трогаются: они приезжают отдельным архивом и меняются
# только при пересборке каталога.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/svs/app}
SVS_USER=${SVS_USER:-svs}
BRANCH=${1:-}

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "нужен root: sudo bash $0"; exit 1; }
cd "$APP_DIR"

say "Забираю код"
sudo -u "$SVS_USER" env HOME=/opt/svs git fetch --prune origin
if [ -n "$BRANCH" ]; then
  sudo -u "$SVS_USER" env HOME=/opt/svs git checkout "$BRANCH"
fi
CURRENT=$(git rev-parse --abbrev-ref HEAD)
# Жёстко на состояние ветки: сервер — не место для правок, и «почти как в репозитории»
# хуже, чем ровно как в репозитории. Правки делаются на машине разработки и приезжают сюда.
sudo -u "$SVS_USER" env HOME=/opt/svs git reset --hard "origin/$CURRENT"
git --no-pager log -1 --format='%h %s'

say "Дособираю окружение"
sudo -u "$SVS_USER" env HOME=/opt/svs uv sync --extra cpu --extra cv --extra api

# С 24.09 полевой контур включается флагом SVS_FIELD (по умолчанию выключен: продукт кадров
# не пишет). В старом /etc/svs-field.env флага нет — без этой строки стенд молча потерял бы
# страницу и приём кадров.
if [ -f /etc/svs-field.env ] && ! grep -q '^SVS_FIELD=' /etc/svs-field.env; then
  say "Включаю полевой контур: дописываю SVS_FIELD=1 в /etc/svs-field.env"
  printf '\n# Полевой контур: страница съёмки и приём кадров (без флага их нет)\nSVS_FIELD=1\n' \
    >> /etc/svs-field.env
fi

say "Перезапускаю службу (прогрев на процессоре идёт минуты)"
systemctl restart svs-field
PORT=$(grep -oP '^SVS_PORT=\K.*' /etc/svs-field.env 2>/dev/null || echo 8080)
for _ in $(seq 1 60); do
  STATUS=$(curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/health" | jq -r '.status // "нет"' 2>/dev/null || echo "нет")
  [ "$STATUS" = "ready" ] || [ "$STATUS" = "degraded" ] && break
  sleep 10
done
curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/health" | jq '{status, degraded_reasons, warnings}' || true
