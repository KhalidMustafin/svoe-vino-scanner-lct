#!/usr/bin/env bash
# Один архив с данными сервиса: индекс, словарь, признаки каталога, фото выгрузки, словарь
# групп вин и данные сомелье.
#
#     bash deploy/pack_data.sh                       # положит svs-field-data.tar.gz рядом с репозиторием
#     bash deploy/pack_data.sh /путь/архив.tar.gz
#
# Зачем отдельно от git: эти файлы в `.gitignore`, и им там место. Индекс и признаки каталога
# собраны из выгрузки организатора, фото — из его же Strapi; репозиторий по условиям задачи
# становится открытым, и выкладывать туда чужие файлы нельзя. Публиковать ли сам архив —
# отдельное решение (ARCHITECTURE.md, «Данные»). Архив собирается раз: индекс меняется только
# при пересборке каталога.
#
# Без портала (решение 24.09): снимка портала (`portal/`), фото живых карточек портала и
# эталонов правки 23.09 (`catalog/packshots_fixed/` — живой портал и Роскачество) в архиве нет.
# Словарь групп кладётся урезанным: только позиции выгрузки и поля группировки — живые поля
# справочника (цвет, сахар, «опубликовано», фото портала) сервис не читает и не везёт.
# Исключение — комплект живых карточек Э3 (25.09, под флагом SVS_LIVE_CARDS): векторы SigLIP и
# признаки 71 карточки портала, без самих фото; кладётся, только если собран.
#
# Обязательное — без него сервис не стартует (`ScannerService.load` в app/api/service.py):
#     index/visual-s2so400m.npz   индекс эталонов SigLIP
#     index/cv-adapter-lw.npz     карта адаптера поиска пути по умолчанию (SVS_CANDIDATE, 26.09):
#                                 собрана под этот индекс (sha1 ccd3a01f), с другим не стартует;
#                                 собирает research/2026-09-26_fund/final/build_final.py
#     index/lexicon.json          словарь каталога для разбора этикетки
#     gt/gt_tokens.jsonl          признаки и карточки каталога
# Необязательное — кладётся, если есть:
#     catalog/photos_small/       пересжатые фото выгрузки (scripts/make_photo_pack.py)
#     catalog/wines.jsonl         словарь групп вин (урезанный, см. выше) и
#     catalog/wine_groups.json    состав групп — для рекомендаций
#     somm/                       данные сомелье (scripts/build_somm.py): пары блюд, подача
#     index/*-live71, gt/*-live71 комплект живых карточек (SVS_LIVE_CARDS=1,
#                                 scripts/build_live_set.py): индекс, словарь и gt вместе
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$REPO/../svs-field-data.tar.gz}
DATA_SRC=${DATA_SRC:-$REPO/data}
PYTHON=${PYTHON:-python}

REQUIRED=(index/visual-s2so400m.npz index/cv-adapter-lw.npz index/lexicon.json gt/gt_tokens.jsonl)
for name in "${REQUIRED[@]}"; do
  [ -f "$DATA_SRC/$name" ] || {
    echo "нет $DATA_SRC/$name"
    echo "если данные в соседнем рабочем дереве: DATA_SRC=/путь/к/svoe-vino-scanner/data bash $0"
    exit 1
  }
done

# Карта адаптера и индекс — пара: карта с чужим индексом сервис не запустит (app/api/service.py,
# load_cv_adapter). Ловим это здесь, а не на машине проверки.
INDEX_SHA1=$(sha1sum "$DATA_SRC/index/visual-s2so400m.npz" | cut -d' ' -f1)
[ "$INDEX_SHA1" = "ccd3a01f16379a3f0a8a1e7e8dc0623c7622bda5" ] || {
  echo "ОШИБКА: index/visual-s2so400m.npz — sha1 $INDEX_SHA1, а карта index/cv-adapter-lw.npz"
  echo "собрана под ccd3a01f…: путь по умолчанию с таким индексом не стартует. Либо вернуть индекс"
  echo "ccd3a01f, либо запускать сервис с SVS_CANDIDATE=off (прежний путь без адаптера)."
  exit 1
}

# Файлы данных идут в архив как есть; урезанный словарь групп — из промежуточного каталога.
ITEMS=("${REQUIRED[@]}")
STAGED=()
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

# Необязательное: кладём, что есть, и говорим, чего нет и чем это грозит.
add() {
  if [ -e "$DATA_SRC/$1" ]; then
    ITEMS+=("$1")
  else
    echo "(нет $1 — $2)"
  fi
}
add catalog/photos_small "сначала python scripts/make_photo_pack.py; без него карточка без фото"
add somm                 "данные сомелье собирает scripts/build_somm.py; без них чип блюда молчит"
# Второй комплект — живые карточки портала (SVS_LIVE_CARDS=1, scripts/build_live_set.py): три
# файла вместе, чтобы флаг переключался одной переменной без пересборки пачки.
for name in index/visual-s2so400m-live71.npz gt/gt_tokens-live71.jsonl index/lexicon-live71.json; do
  add "$name" "комплект живых карточек собирает scripts/build_live_set.py; без него SVS_LIVE_CARDS=1 не стартует"
done

# В пачке фото — только выгрузка организатора: имя файла — slug из gt_tokens.jsonl.
PHOTOS="$DATA_SRC/catalog/photos_small"
if [ -d "$PHOTOS" ]; then
  sed -n 's/^{"slug": "\([^"]*\)".*/\1/p' "$DATA_SRC/gt/gt_tokens.jsonl" | sort > "$STAGE/slugs.txt"
  find "$PHOTOS" -type f -printf '%f\n' | sed 's/\.webp$//' | sort > "$STAGE/photos.txt"
  FOREIGN=$(comm -23 "$STAGE/photos.txt" "$STAGE/slugs.txt")
  if [ -n "$FOREIGN" ]; then
    echo "ОШИБКА: в catalog/photos_small есть фото не из выгрузки (живые карточки портала?):"
    printf '%s\n' "$FOREIGN" | sed -n '1,5p'
    echo "Пересобрать с нуля: убрать catalog/photos_small (сначала копию) и запустить"
    echo "python scripts/make_photo_pack.py в дереве, где лежат эти данные: поверх старого"
    echo "каталога сборка лишних файлов не удаляет."
    exit 1
  fi
fi

# photos_small пересжимается из путей slug_photo_map.csv. Если карту правили позже, в архив
# уехали бы старые фото.
MAP="$DATA_SRC/catalog/slug_photo_map.csv"
if [ -d "$PHOTOS" ] && [ -f "$MAP" ] \
    && [ -z "$(find "$PHOTOS" -type f -newer "$MAP" -print -quit)" ]; then
  echo "ВНИМАНИЕ: catalog/photos_small старше catalog/slug_photo_map.csv — фото устарели."
  echo "          Пересобрать: python scripts/make_photo_pack.py в дереве, где лежат эти данные."
fi

# Словарь групп — только позиции выгрузки и поля группировки (app/recommend/catalog.py).
if [ -f "$DATA_SRC/catalog/wines.jsonl" ] && [ -f "$DATA_SRC/catalog/wine_groups.json" ]; then
  mkdir -p "$STAGE/catalog"
  "$PYTHON" - "$DATA_SRC/catalog" "$STAGE/catalog" <<'PY'
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
keep = ("slug", "wine_id", "canonical", "is_canonical", "in_csv", "winery_norm", "grapes",
        "grapes_src")
lines = (src / "wines.jsonl").read_text(encoding="utf-8").splitlines()
rows = [json.loads(line) for line in lines if line.strip()]
ours = [{key: row.get(key) for key in keep} for row in rows if row.get("in_csv")]
slugs = {row["slug"] for row in ours}
with (dst / "wines.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
    for row in ours:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
groups = json.loads((src / "wine_groups.json").read_text(encoding="utf-8"))
kept = {}
for wine_id, group in groups.items():
    members = [slug for slug in group.get("members") or [] if slug in slugs]
    if members:
        kept[wine_id] = {"wine_id": wine_id, "members": members}
(dst / "wine_groups.json").write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
print(f"словарь групп: {len(ours)} позиций выгрузки из {len(rows)}, групп {len(kept)}")
PY
  STAGED+=(catalog/wines.jsonl catalog/wine_groups.json)
else
  echo "(нет catalog/wines.jsonl или wine_groups.json — без групп похожие не убирают то же вино)"
fi

# GNU tar принимает путь вида `C:/...` за адрес удалённой машины (`host:path`) и лезет в сеть.
# На Windows это ровно тот путь, который приходит из Git Bash.
TAR_OPTS=()
tar --help 2>/dev/null | grep -q -- '--force-local' && TAR_OPTS+=(--force-local)
if [ "${#STAGED[@]}" -gt 0 ]; then
  tar czf "$OUT" "${TAR_OPTS[@]}" -C "$DATA_SRC" "${ITEMS[@]}" -C "$STAGE" "${STAGED[@]}"
else
  tar czf "$OUT" "${TAR_OPTS[@]}" -C "$DATA_SRC" "${ITEMS[@]}"
fi
SIZE=$(du -h "$OUT" | cut -f1)

echo
echo "В архиве:"
printf '    %s\n' "${ITEMS[@]}" "${STAGED[@]}"

cat <<EOF

Архив готов: $OUT ($SIZE)

Docker (compose.yml) — распаковать в SVS_DATA_HOST_DIR (по умолчанию data/ репозитория):
    tar xzf "$OUT" -C data

VPS (полевой стенд) — отправить и распаковать:
    scp "$OUT" root@IP:/tmp/
    ssh root@IP 'mkdir -p /opt/svs/data && tar xzf /tmp/$(basename "$OUT") -C /opt/svs/data && chown -R svs:svs /opt/svs/data && rm /tmp/$(basename "$OUT")'
EOF
