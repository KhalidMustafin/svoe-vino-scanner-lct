#!/usr/bin/env bash
# Проверка образа svoe-vino-scanner:cpu через compose.yml — без видеокарты и без Ollama.
#
# Что проверяет: тома /data, /dataset, /hf/hub доходят до сервиса и читаются под пользователем
# образа; карточки получают описания из выгрузки; фото каталога отдаётся с верным типом; тело
# predict — прежние 8 ключей; ошибка кадра — HTTP 200 и `decode:`. Чего не проверяет: чтение
# этикетки (SVS_VLM_TIMEOUT_MS=0) и видеокарту — поэтому точность и время здесь не боевые.
#
# Как запускался (Git Bash, машина разработки, 23.09.2026), из корня рабочего дерева:
#     bash research/2026-09-23_docker-smoke/smoke.sh > research/2026-09-23_docker-smoke/smoke.out 2>&1
set -u
cd "$(dirname "$0")/../.."
HERE=research/2026-09-23_docker-smoke
PY="<корень>/svoe-vino-scanner/.venv/Scripts/python.exe"
Q="<корень>/Датасет и подробное задание/unpacked/Датасет/eval/queries"
URL=http://127.0.0.1:8791
C=svs-infra-smoke-scanner-cpu-1

export MSYS_NO_PATHCONV=1   # иначе Git Bash перепишет /data в C:/Program Files/Git/data
export SVS_DATA_HOST_DIR="<корень>/svoe-vino-scanner/data"
export SVS_PUBLISH_PORT=8791 SVS_VLM_TIMEOUT_MS=0
compose() {
  docker compose --env-file .env.example -p svs-infra-smoke \
    -f compose.yml -f "$HERE/smoke.override.yml" --profile cpu "$@"
}

echo "== образ"
docker images svoe-vino-scanner:cpu --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.ID}}'
started=$(date +%s)
compose up -d --no-deps --no-build scanner-cpu 2>&1 | tail -1
until curl -s -m 3 "$URL/v1/health" > /dev/null 2>&1; do
  [ -n "$(docker ps -q --filter name=$C)" ] || { echo "контейнер упал"; docker logs $C 2>&1 | tail -30; exit 1; }
  sleep 3
done
echo "порт открыт через $(( $(date +%s) - started )) с (прогрев в lifespan)"

echo "== окружение и тома"
docker exec $C sh -c 'id; env | grep -E "^(SVS_DATA_DIR|SVS_DATASET_DIR|SVS_DEVICE|SVS_FIELD_PHOTO_DIR|HF_HOME)=" | sort'
docker inspect $C --format '{{range .Mounts}}{{.Destination}} rw={{.RW}}  {{end}}'

echo "== /v1/health"
curl -s "$URL/v1/health" | "$PY" -c "
import json, sys
h = json.load(sys.stdin)
print('status', h['status'], h['degraded_reasons'])
print('cv', h['model']['cv'], 'device', h['model']['cv_device'], '| resolve', h['model']['resolve'])
print('index', h['index']['path'], h['index']['n_slugs'], h['index']['n_vectors'])
print('cards', json.dumps(h['catalog']['cards'], ensure_ascii=False))
print('provenance.consistent', h['provenance']['consistent'], '| formats', h['formats'])
print('warm scan', json.dumps(h['warm'].get('scan', {}).get('timings_ms')))
"

echo "== фото каталога"
S=massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16
# Не -o /dev/null: при MSYS_NO_PATHCONV curl для Windows получает путь /dev/null как есть.
curl -s -o "$HERE/_photo.tmp" \
  -w "GET /v1/field/photo/$S -> %{http_code} %{content_type} %{size_download} B\n" \
  "$URL/v1/field/photo/$S"
rm -f "$HERE/_photo.tmp"

echo "== predict на трёх публичных кадрах (без чтения этикетки — слаг не боевой)"
for f in 019c68d0.jpg 02eef911.webp 096ca74e.jpg; do
  curl -s -m 60 -F "image=@$Q/$f" "$URL/v1/eval/predict" | "$PY" -c "
import json, sys
b = json.load(sys.stdin)
print('$f', 'keys', sorted(b), '|', b['slug'], b['outcome'], b['degraded'],
      'cv', b['timings_ms'].get('cv'), 'total', b['timings_ms'].get('total'))
"
done

echo "== не картинка"
curl -s -m 30 -w "  HTTP %{http_code}\n" -F "image=@README.md;type=image/jpeg" "$URL/v1/eval/predict"

echo "== журнал: ошибки записи и трассы"
docker logs $C 2>&1 | grep -ciE "Traceback|Permission denied|Read-only file system" || true

compose down -v 2>&1 | tail -1
