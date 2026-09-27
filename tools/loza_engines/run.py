"""Подпроцесс сборки: оценивает все пары «вино × блюдо» движком «Лозы».

Запуск — только из `scripts/build_somm.py`:

    python -I -S tools/loza_engines/run.py < запрос.json > ответ.json

`-I -S` изолируют процесс: в `sys.path` нет ни рабочего каталога, ни корня репозитория, ни
site-packages (движку хватает стандартной библиотеки), поэтому пакет `app` сканера отсюда не
виден, а движок «Лозы» видит только свой пакет `loza_engines`.

Запрос (stdin, JSON): ``{"wines": [вино, …], "overlay": true}``, вино — словарь `WineView`.
`overlay`: `true` — наша накладка `overlay.json`, `false` — без неё, строка — путь к другой
накладке (тесты проверяют, что накладка не вправе менять вес или объяснение правила).
Ответ (stdout, JSON):

    {"rules": [правило «Лозы» после обеих накладок, …],
     "pairs": {"<slug>": {"<блюдо>": [сумма, оценка, [«+»…], [«−»…]], …}, …}}

Сумма и оценка округлены до шести знаков: так пересборка повторяет результат байт в байт.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference"
OVERLAY = HERE / "overlay.json"


def main() -> int:
    # Ошибка движка уходит в сборку через stderr: в UTF-8, а не в кодовой странице консоли.
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.path.insert(0, str(HERE.parent))
    from loza_engines.pairing import DishView, PairingEngine, WineView, load_rules

    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    overlay = request.get("overlay", True)
    rules = load_rules(
        REFERENCE / "pairing.json",
        REFERENCE / "pairing_rules_fixes.json",
        Path(overlay) if isinstance(overlay, str) else OVERLAY if overlay else None,
    )
    engine = PairingEngine(rules)
    dishes = [
        DishView(entry)
        for entry in json.loads((REFERENCE / "pairing.json").read_text(encoding="utf-8"))["dishes"]
    ]
    wines = [WineView(entry) for entry in request["wines"]]
    matrix = engine.matrix(wines, dishes)
    response = {
        "rules": [asdict(rule) for rule in rules],
        "pairs": {
            slug: {
                dish_id: [round(pair.total, 6), round(pair.score, 6), pair.plus, pair.minus]
                for dish_id, pair in row.items()
            }
            for slug, row in matrix.items()
        },
    }
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
