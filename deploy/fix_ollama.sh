#!/usr/bin/env bash
# Починка программы `ollama`, когда установщик оставил обрубок.
#
#     sudo bash deploy/fix_ollama.sh
#
# Официальный установщик качает архив одним куском и без докачки: на рваном канале он
# записывает поверх рабочего файла огрызок, программа начинает падать с Segmentation fault, а
# демон живёт дальше — из уже загруженного кода. Пока его не перезапускали, всё работает; после
# перезагрузки сервера Ollama просто не встанет, и стенд умрёт молча.
#
# Здесь архив берётся с GitHub (он обычно доступен там, где реестры режут), с докачкой и
# проверкой размера, и ставится рядом. Версия — та же, что у работающего демона: ставить более
# новую программу к старому демону незачем.
set -euo pipefail

OLLAMA_URL=${OLLAMA_URL:-http://127.0.0.1:11434}
PREFIX=${PREFIX:-/usr/local}
ARCHIVE=${ARCHIVE:-/tmp/ollama-linux-amd64.tgz}
#: Меньше этого — точно обрубок: настоящий архив весит сотни мегабайт.
MIN_BYTES=$((50 * 1024 * 1024))

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "нужен root: sudo bash $0"; exit 1; }

VERSION=${1:-$(curl -sf --max-time 5 "$OLLAMA_URL/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)}
[ -n "$VERSION" ] || { echo "не узнал версию демона; задайте её первым доводом: bash $0 0.33.2"; exit 1; }
say "Чиню программу под версию демона $VERSION"

BASE="https://github.com/ollama/ollama/releases/download/v${VERSION}"
# С какого-то выпуска архив стал zstd вместо gzip, и старое имя отдаёт 404. Пробуем новое,
# откатываемся на старое.
ASSET=ollama-linux-amd64.tar.zst
curl -sfIL "$BASE/$ASSET" >/dev/null 2>&1 || ASSET=ollama-linux-amd64.tgz
ARCHIVE="$(dirname "$ARCHIVE")/$ASSET"
URL="$BASE/$ASSET"

say "Качаю $URL (с докачкой)"
# На этом сервере режут скорость на соединение: один поток даёт сотни килобайт, шестнадцать —
# мегабайты. Поэтому aria2c, если он есть, и curl как запасной путь.
if command -v aria2c >/dev/null; then
  aria2c -x16 -s16 -k10M -c -d "$(dirname "$ARCHIVE")" -o "$(basename "$ARCHIVE")" "$URL"
else
  curl -L -C - --retry 10 --retry-delay 5 --retry-all-errors -o "$ARCHIVE" "$URL"
fi

SIZE=$(stat -c%s "$ARCHIVE")
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
  echo "архив всего $SIZE байт — снова обрубок, запустите ещё раз (докачает)"
  exit 1
fi
say "Архив $((SIZE / 1024 / 1024)) МБ, распаковываю в $PREFIX"
case "$ASSET" in
  *.tar.zst)
    command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; }
    tar -I zstd -xf "$ARCHIVE" -C "$PREFIX"
    ;;
  *) tar xzf "$ARCHIVE" -C "$PREFIX" ;;
esac
# Сломанный установщик оставил файл с правами 700 и владельцем root, а служба работает от
# пользователя ollama — отсюда и было «Permission denied» вместо запуска.
chmod 755 "$PREFIX/bin/ollama"

say "Проверяю"
if ! timeout 10 "$PREFIX/bin/ollama" --version; then
  echo "программа всё ещё не запускается — не перезапускайте службу, демон пока жив"
  exit 1
fi
rm -f "$ARCHIVE"

say "Перезапускаю службу"
# После сотен неудачных стартов подряд systemd отказывается пробовать снова, пока счётчик не
# сброшен: битый бинарник обычно означает именно такую петлю.
systemctl reset-failed ollama 2>/dev/null || true
systemctl restart ollama
sleep 3
curl -sf --max-time 5 "$OLLAMA_URL/api/tags" >/dev/null \
  && echo "Ollama отвечает, модели на месте:" \
  && curl -sf "$OLLAMA_URL/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4
