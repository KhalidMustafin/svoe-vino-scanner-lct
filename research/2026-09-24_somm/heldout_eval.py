"""Барьер на отложенном наборе: сколько опасных фраз получают отказ и сколько законных — лишний.

Отложенный набор составлен 24.09 независимо от барьера, его тестов и прежних
наборов: 273 опасные фразы (цена и покупка, несовершеннолетние, беременность и
кормление, здоровье, зависимость и опьянение, вождение; прямо, окольно, разговорно и в паре с
законным вопросом) и 314 законных, из них 24 спорные — они в счёт не идут. Файлы лежат вне
репозитория и сюда не копируются: иначе следующий замер на них перестанет быть честным.

Отказ — `intent == "refuse"` сквозь весь маршрут (`Router.text` на полной сборке `data/somm`,
смысловой слой выключен, как в сервисе). «Дошло до вердикта» — опасная фраза без отказа, которую
маршрут отдал в подбор, подачу или вердикт пары, а не в «Вот что я умею». Сохраняются только
счёты — ни одной фразы.

    SVS_DATA_DIR=... .venv/Scripts/python.exe research/2026-09-24_somm/heldout_eval.py <каталог>
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from app.recommend.somm_data import load_somm_data  # noqa: E402
from app.sommelier.router import Context, Router  # noqa: E402

OUT = HERE / "heldout_eval.json"
#: Намерения, которые дают человеку вердикт или подбор: опасная фраза здесь — настоящий пропуск.
VERDICT = frozenset(
    {
        "dish_check",
        "what_to_eat",
        "softer",
        "fresher",
        "replace",
        "guided",
        "serve",
        "grape",
        "fact",
    }
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    held = Path(sys.argv[1])
    risky = json.loads((held / "risky.json").read_text(encoding="utf-8"))
    legit = json.loads((held / "legit.json").read_text(encoding="utf-8"))
    data_dir = Path(os.environ.get("SVS_DATA_DIR") or REPO / "data") / "somm"
    router = Router(load_somm_data(data_dir, fallback=REPO / "tests" / "fixtures" / "somm"))

    def route(question: str):  # noqa: ANN202
        return router.text(question, Context(), grapes=["saperavi"])

    topics: dict[str, Counter] = defaultdict(Counter)
    forms: dict[str, Counter] = defaultdict(Counter)
    for item in risky:
        got = route(item["text"])
        refused = got.intent == "refuse"
        for bucket in (topics[item["topic"]], forms[item["form"]]):
            bucket["n"] += 1
            bucket["refused"] += refused
            bucket["verdict"] += not refused and got.intent in VERDICT
    scored = [item for item in legit if not item.get("ambiguous")]
    triggers: dict[str, Counter] = defaultdict(Counter)
    false_topics: Counter = Counter()
    for item in scored:
        got = route(item["text"])
        key = item.get("trigger") or "-"
        triggers[key]["n"] += 1
        if got.intent == "refuse":
            triggers[key]["false"] += 1
            false_topics[got.refusal.topic] += 1
    ambiguous = Counter(
        f"{item.get('lean')}:{'refuse' if route(item['text']).intent == 'refuse' else 'pass'}"
        for item in legit
        if item.get("ambiguous")
    )
    n_risky = sum(c["n"] for c in topics.values())
    n_refused = sum(c["refused"] for c in topics.values())
    n_false = sum(c["false"] for c in triggers.values())
    result = {
        "risky": {
            "n": n_risky,
            "refused": n_refused,
            "recall": round(n_refused / n_risky, 4),
            "reach_verdict": sum(c["verdict"] for c in topics.values()),
            "by_topic": {key: dict(value) for key, value in sorted(topics.items())},
            "by_form": {key: dict(value) for key, value in sorted(forms.items())},
        },
        "legit": {
            "n": len(scored),
            "false": n_false,
            "rate": round(n_false / len(scored), 4),
            "by_trigger": {key: dict(value) for key, value in sorted(triggers.items())},
            "false_topics": dict(false_topics),
            "ambiguous": dict(sorted(ambiguous.items())),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"опасные: {n_refused}/{n_risky} = {n_refused / n_risky:.1%}")
    for key, value in result["risky"]["by_topic"].items():
        print(f"  {key:10} {value['refused']}/{value['n']}, до вердикта {value['verdict']}")
    print(f"законные: лишних отказов {n_false}/{len(scored)} = {n_false / len(scored):.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
