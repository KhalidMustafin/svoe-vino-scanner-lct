"""Заглушка «Сомелье» для вёрстки листа (`app/api/static/somm.js` и `somm.css`).

Лист верстается раньше, чем готовы маршруты `app/api/somm.py`. Этот сервер отдаёт стенд — страницу
с блоками карточки из `window.SommUI` (реплика, заметка, «роза ветров», шкалы, «Спросить
сомелье») — и ответы по маршрутам договора `docs/api-sommelier.md`:

- `GET /v1/wines/{slug}/sommelier` — заглушки `tests/fixtures/somm/sommelier_*.json`;
- `POST /v1/sommelier/ask` — поток NDJSON: готовые потоки `ask_*.ndjson` там, где они есть, а для
  остальных чипов — сборка из тех же заглушек (пары, подача, справочник). Строки уходят с паузами
  по `t_ms`, как у живого сервера: шаблон приходит сразу, голос думает полторы секунды.

Запуск из корня рабочего дерева:

    python scripts/somm_stub.py                 # http://127.0.0.1:8766/
    python scripts/somm_stub.py --no-input      # SVS_SOMM_INPUT=0: поля нет, вопрос текстом — 422
    python scripts/somm_stub.py --no-live       # SVS_SOMM_LIVE=0: только шаблоны, reason "off"

У адреса стенда: `?wine=red|sweet|sparkling|nosugar` — вино карточки; `voice=live|guard|busy` —
чем кончается живой голос (по умолчанию live; сервер берёт его из Referer, как заглушка страницы);
`chip=<id>` или `q=<вопрос>` — сразу открыть лист с этим вопросом (для скриншотов). С каталогом
данных (`--data-dir` или `SVS_DATA_DIR`) фото вин — `catalog/photos_small` выгрузки организатора.
`shot=somm_…` подмешивает `scripts/ui_stub_driver.js`: он жмёт чипы листа за человека для
снимков `scripts/ui_screens.py`; `/__stub/frame?shot=…&wine=…` — стенд во фрейме 390×844.

Живой текст в заглушке — не модель: у «К чему подать» и блюд с вердиктом «да» это тот же шаблон с
меткой ИИ, чтобы на стенде был виден путь каретки. Подборки («Помягче», «Посвежее», «Чем заменить»,
финал «Помогите выбрать») и блюда с оговоркой или «скорее нет» (у них вина подборки) голос не
пересказывает, как и сервис: шаблон с `reason: "not_voiced"` (договор, §6.5). Потоки заглушек
`ask_dish_check_live.ndjson` и `ask_dish_check_guard.ndjson` — образцы страницы, а не ответ сервиса.

Только стандартная библиотека и без импорта `app`: заглушке не нужны модели, сеть и видеокарта.
Страницу продукта отдаёт `scripts/ui_stub_server.py`, здесь — только стенд листа.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "api" / "static"
FIXTURES = ROOT / "tests" / "fixtures" / "somm"
DRIVER = Path(__file__).with_name("ui_stub_driver.js")
WINES = {
    "red": "sommelier_red.json",
    "sweet": "sommelier_sweet.json",
    "sparkling": "sommelier_sparkling.json",
    "nosugar": "sommelier_nosugar.json",
}
ORDERS = ("reco", "plain")
VOICES = ("live", "guard", "busy")
CHIPS = (
    "what_to_eat",
    "serve",
    "softer",
    "fresher",
    "replace",
    "grape",
    "term",
    "dish_check",
    "guided",
)
LABEL_AI = "Текст — ИИ, подбор — алгоритм"
LABEL_ALGO = "Текст и подбор — алгоритм"
NOTICE = "Применяются рекомендательные технологии"
INPUT_OFF = "вопрос текстом выключен"
MAX_QUESTION = 200
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
STATIC_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
STAGES = {
    "card": ("Открываю карточку «{name}»", "Открыл карточку"),
    "rules": ("Проверяю правила сочетаний", "Проверил правила сочетаний"),
    "catalog": ("Ищу в каталоге вина других виноделен", "Посмотрел каталог"),
    "voice": ("Сомелье формулирует ответ", "Сомелье сформулировал"),
    "verify": ("Сверяю имена и числа", "Сверил имена и числа"),
}
# Оси вина, которые правило читает «по сорту»: чип такого правила получает source "grape".
GRAPE_FIELDS = {"acidity", "tannin", "body", "oak", "aroma_intensity", "descriptors", "serve"}
FOOD = {
    "meat": "К мясу",
    "fish": "К рыбе",
    "cheese": "К сыру",
    "dessert": "К десерту",
    "none": "Без блюда",
}
WANT = {"softer": "помягче", "fresher": "посвежее", "none": ""}
UNKNOWN = (
    "Вот что я умею: подсказать, к чему подать это вино, как его подать и чем заменить."
    " Выберите вопрос ниже."
)
HELLO = (
    "Здравствуйте! Я сомелье «Своего Вина»: расскажу об этом вине, к чему его подать и чем"
    " заменить."
)
REFUSE_RE = re.compile(r"купи|цен[аыуе]|стоит|дорог|дешев|доставк|скидк", re.IGNORECASE)


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ndjson(name: str) -> list[dict[str, Any]]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


class Stub:
    """Данные заглушек и сборка потоков по правилам договора (§3–§4)."""

    def __init__(self, *, data_dir: Path | None, live: bool, ask_input: bool, speed: float) -> None:
        self.data_dir = data_dir
        self.live = live
        self.input = ask_input
        self.speed = speed
        self.cards = {load(name)["slug"]: load(name) for name in WINES.values()}
        self.by_key = {key: load(name)["slug"] for key, name in WINES.items()}
        dishes = load("dishes.json")
        self.dishes = {dish["id"]: dish for dish in dishes["dishes"]}
        self.rules = {rule["id"]: rule for rule in dishes["rules"]}
        self.pairs = load("pairs.json")["wines"]
        self.serve = {rule["id"]: rule for rule in load("serve.json")["rules"]}
        knowledge = load("knowledge.json")
        self.topics = {topic["id"]: topic for topic in knowledge["topics"]}
        self.grapes = knowledge["grapes"]
        # Плитки трёх Пино Нуар и сравнение — из заглушки «а к борщу?»: подборка стенда.
        base = next(e for e in ndjson("ask_dish_check_live.ndjson") if e["type"] == "facts")
        self.pinot = base["wines"]
        self.pinot_compare = base["compare"]

    # ---------------------------------------------------------------- GET …/sommelier

    def card(self, slug: str, order: str) -> dict[str, Any] | None:
        card = copy.deepcopy(self.cards.get(slug))
        if card is None:
            return None
        card.update(order=order, live=self.live, input=self.input)
        card["notice_149"] = NOTICE if order == "reco" else None
        return card

    def photo(self, slug: str) -> Path | None:
        if not self.data_dir or not SLUG_RE.match(slug):
            return None
        path = self.data_dir / "catalog" / "photos_small" / f"{slug}.webp"
        return path if path.is_file() else None

    # ---------------------------------------------------------------- POST ask

    def ask(self, slug: str, body: dict[str, Any], order: str, voice: str) -> list[dict[str, Any]]:
        card = self.cards[slug]
        context = body.get("context") if isinstance(body.get("context"), dict) else {}
        question = body.get("question")
        chip, args = body.get("chip"), body.get("args") or {}
        if isinstance(question, str):
            if "ошибк" in question.lower():
                return [
                    dict(e, text=e["text"].replace("Cru Lermont Saperavi", card["name"]))
                    if e["type"] == "stage"
                    else e
                    for e in ndjson("ask_error.ndjson")
                ]
            if REFUSE_RE.search(question):
                return self._retarget(ndjson("ask_refusal.ndjson"), slug)
            chip, args = self._parse(question, card, context)
        if chip == "guided" and not args:
            return self._retarget(ndjson("ask_guided_step.ndjson"), slug)
        return self._compose(card, chip or "unknown", args, context, order, voice, question)

    def _retarget(self, events: list[dict[str, Any]], slug: str) -> list[dict[str, Any]]:
        for event in events:
            if event["type"] == "facts":
                event["slug"] = slug
        return events

    def _parse(
        self, question: str, card: dict[str, Any], context: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        low = question.lower()
        # Блюдо: сначала алиас целиком (длинные раньше: «борщ с пампушками» не «борщ»), потом
        # основа первого слова — «а к шашлыку?» находит «шашлык из баранины».
        aliases = sorted(
            ((alias, dish["id"]) for dish in self.dishes.values() for alias in dish["aliases"]),
            key=lambda pair: -len(pair[0]),
        )
        for alias, dish_id in aliases:
            if alias in low:
                return "dish_check", {"dish": dish_id}
        for alias, dish_id in aliases:
            if len(alias.split()[0]) >= 5 and alias.split()[0][:5] in low:
                return "dish_check", {"dish": dish_id}
        table = (
            (r"привет|здравств|спасибо", "smalltalk"),
            (r"мягч", "softer"),
            (r"свеж|кисл", "fresher"),
            (r"замен|похож|аналог", "replace"),
            (r"подать|подава|температур|охлажд|декант", "serve"),
            (r"брют", "term"),
            (r"сорт", "grape"),
            (r"к чему|поесть|еда|еду|блюд|закус", "what_to_eat"),
            (r"помоги|подбер", "guided"),
        )
        for pattern, intent in table:
            if re.search(pattern, low):
                if intent in ("softer", "fresher") and context.get("dish"):
                    return intent, {"dish": context["dish"]}
                if intent == "term":
                    return "term", {"topic": "brut_scale"}
                if intent == "grape":
                    grape = next(
                        (c["args"]["grape"] for c in card["chips"] if c["id"] == "grape"), None
                    )
                    return ("grape", {"grape": grape}) if grape else ("unknown", {})
                return intent, {}
        return "unknown", {}

    def _rule_chip(self, rule_id: str) -> dict[str, str]:
        rule = self.rules[rule_id]
        source = "grape" if GRAPE_FIELDS & set(rule["wine_fields"]) else "catalog"
        return {"id": rule_id, "text": rule["chip"], "source": source}

    def _dish(self, slug: str, dish_id: str) -> dict[str, Any] | None:
        dish = self.dishes.get(dish_id)
        if dish is None:
            return None
        verdict, plus, minus = self.pairs.get(slug, {}).get("dishes", {}).get(dish_id) or [
            "neutral",
            [],
            [],
        ]
        return {
            "id": dish_id,
            "name": dish["name"],
            "category": dish["category"],
            "verdict": verdict,
            "plus": [self._rule_chip(r) for r in plus[:3]],
            "minus": [self._rule_chip(r) for r in minus[:2]],
        }

    def _compose(
        self,
        card: dict[str, Any],
        chip: str,
        args: dict[str, Any],
        context: dict[str, Any],
        order: str,
        voice: str,
        question: Any,
    ) -> list[dict[str, Any]]:
        slug, name = card["slug"], card["name"]
        red = slug == self.by_key["red"]
        facts: dict[str, Any] = {
            "slug": slug,
            "intent": chip,
            "order": order,
            "verdict_template": "",
            "detail": None,
            "basis": [],
            "dishes": [],
            "wines_title": None,
            "wines": [],
            "compare": None,
            "serve": None,
            "question": None,
            "refusal": None,
            "chips": [],
            "notice_149": None,
            "voice": False,
            "context": {
                "intent": chip,
                "dish": context.get("dish"),
                "food": None,
                "want": None,
                "step": None,
                "refusal_topic": None,
                "shown": list(context.get("shown") or [])[:12],
            },
        }
        stages: list[str] = []
        entry = [c for c in card["chips"] if c["id"] in ("what_to_eat", "serve", "replace")]
        if chip == "what_to_eat":
            stages = ["card", "rules"]
            facts["dishes"] = card["dishes"]
            names = [d["name"][:1].lower() + d["name"][1:] for d in card["dishes"]]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " и " + names[-1]
            first = (
                card["dishes"][0]["plus"][0]["id"]
                if card["dishes"] and card["dishes"][0]["plus"]
                else None
            )
            facts["verdict_template"] = (
                f"По правилам сочетаний к этому вину подходят {joined}. "
                + (self.rules[first]["text"] if first else "")
            ).strip()
            facts["basis"] = ["правила сочетаний", "профиль по сорту"]
            facts["voice"] = True
            facts["chips"] = [
                {"id": "dish_check", "text": "А к шашлыку?", "args": {"dish": "shashlyk_baranina"}}
            ] + entry[1:]
        elif chip == "dish_check":
            stages = ["card", "rules"]
            dish = self._dish(slug, str(args.get("dish") or ""))
            if dish is None:
                chip = "unknown"
            else:
                info = self.dishes[dish["id"]]
                dative = info["dative"]
                facts["dishes"] = [dish]
                facts["context"]["dish"] = dish["id"]
                facts["basis"] = ["правила сочетаний", "карточка каталога", "профиль по сорту"]
                strongest_plus = self.rules[dish["plus"][0]["id"]]["text"] if dish["plus"] else ""
                strongest_minus = (
                    self.rules[dish["minus"][0]["id"]]["text"] if dish["minus"] else ""
                )
                facts["verdict_template"] = {
                    "yes": f"{cap(dative)} — да. {strongest_plus}",
                    "caveat": f"{cap(dative)} — да, с оговоркой. {strongest_minus}",
                    "no": f"{cap(dative)} — скорее нет. {strongest_minus}",
                    "neutral": "Правила сочетаний об этой паре ничего не говорят.",
                }[dish["verdict"]].strip()
                # Голос — только «да»: у оговорки и «скорее нет» вина подборки (договор, §6.5).
                facts["voice"] = dish["verdict"] == "yes"
                facts["chips"] = [
                    {"id": "serve", "text": "Как подать"},
                    {"id": "softer", "text": "Помягче", "args": {"dish": dish["id"]}},
                ]
                if dish["verdict"] in ("caveat", "no") and red:
                    stages.append("catalog")
                    facts["verdict_template"] += (
                        f" Вот вина других виноделен, которые подходят {dative}."
                    )
                    self._wines(facts, "Помягче " + dative, ["tannin"], order)
        elif chip == "serve":
            stages = ["card"]
            serve = card["serve"]
            rule = self.serve[serve["rule"]]
            low, high = serve["temperature_c"]
            facts["serve"] = serve
            facts["verdict_template"] = f"{name} подают при {low}–{high} °C. {rule['text']}"
            facts["basis"] = ["правила подачи"] + (
                ["профиль по сорту"] if serve["by_grape"] else []
            )
            facts["chips"] = [c for c in entry if c["id"] != "serve"]
        elif chip in ("softer", "fresher", "replace"):
            stages = ["card", "catalog"]
            title = {"softer": "Помягче", "fresher": "Посвежее", "replace": "Чем заменить"}[chip]
            axes = {"softer": ["tannin"], "fresher": ["acidity"], "replace": []}[chip]
            # подборки голос не пересказывает (договор, §4.1): «voice» остаётся False
            facts["basis"] = ["подбор из каталога", "профиль по сорту"]
            facts["context"]["want"] = chip if chip != "replace" else None
            if red and chip != "fresher":
                self._wines(facts, title, axes, order)
                facts["verdict_template"] = (
                    "Помягче — три вина других виноделен того же стиля."
                    if chip == "softer"
                    else "Похожие из других виноделен: красное сухое."
                )
            else:
                facts["verdict_template"] = (
                    "По описаниям разницы нет — в этом стиле других виноделен не нашлось."
                )
            facts["chips"] = entry
        elif chip == "grape":
            stages = ["card"]
            grape = self.grapes.get(str(args.get("grape") or ""))
            facts["verdict_template"] = grape["text"] if grape else UNKNOWN
            facts["basis"] = ["справочник"]
            facts["chips"] = entry[:2]
        elif chip == "term":
            topic = self.topics.get(str(args.get("topic") or "")) or self.topics["brut_scale"]
            facts["verdict_template"] = topic["answer"]
            facts["detail"] = topic["detail"]
            facts["basis"] = ["справочник"]
            facts["chips"] = topic["chips"] or entry
        elif chip == "guided":
            food, want = args.get("food"), args.get("want")
            facts["context"].update(food=food, want=want)
            if food and not want:
                facts["question"] = {"step": "want", "text": "Какое хочется?"}
                facts["context"]["step"] = "want"
                facts["verdict_template"] = (
                    f"{FOOD.get(food, 'К столу')} это вино по правилам сочетаний подходит. Какое хочется?"
                )
                facts["chips"] = [
                    {"id": "guided", "text": "Помягче", "args": {"food": food, "want": "softer"}},
                    {"id": "guided", "text": "Посвежее", "args": {"food": food, "want": "fresher"}},
                    {"id": "guided", "text": "Как это", "args": {"food": food, "want": "none"}},
                ]
            else:
                stages = ["card", "rules", "catalog"]
                label = FOOD.get(str(food), "К столу") + (
                    " — " + WANT[want] if WANT.get(str(want)) else ""
                )
                facts["basis"] = ["правила сочетаний", "подбор из каталога"]
                if red:
                    self._wines(facts, label, ["tannin"] if want == "softer" else [], order)
                    facts["verdict_template"] = label + " — три вина других виноделен."
                else:
                    facts["verdict_template"] = (
                        label + " — в каталоге других виноделен ничего не нашлось."
                    )
                facts["chips"] = entry
        if chip == "smalltalk":
            facts["verdict_template"] = HELLO
            facts["chips"] = card["chips"][:5]
        elif chip == "unknown":
            facts["intent"] = "unknown"
            facts["verdict_template"] = UNKNOWN
            facts["chips"] = card["chips"][:5]
        if order == "plain" and facts["wines"]:
            facts["verdict_template"] = (
                "Обычная сортировка: красные сухие других виноделен по названию."
            )
            facts["wines_title"] = "Красное сухое — по названию"
        return self._stream(facts, stages, order, voice, name)

    def _wines(self, facts: dict[str, Any], title: str, axes: list[str], order: str) -> None:
        shown = set(facts["context"]["shown"])
        wines = [copy.deepcopy(w) for w in self.pinot if w["slug"] not in shown] or copy.deepcopy(
            self.pinot
        )
        if order == "plain":
            for wine in wines:
                wine.update(reasons=[], pill=None, want_source=None)
        facts["wines"] = wines
        facts["wines_title"] = title
        facts["context"]["shown"] = (facts["context"]["shown"] + [w["slug"] for w in wines])[-12:]
        if order == "reco":
            facts["notice_149"] = NOTICE
            compare = copy.deepcopy(self.pinot_compare)
            compare["axes"] = axes
            compare["wines"] = [
                c for c in compare["wines"] if c["slug"] in {w["slug"] for w in wines}
            ]
            facts["compare"] = compare

    def _stream(
        self, facts: dict[str, Any], stages: list[str], order: str, voice: str, name: str
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        t = 2
        for stage in stages:
            text, done = STAGES[stage]
            events.append(
                {
                    "type": "stage",
                    "id": stage,
                    "text": text.format(name=name),
                    "done": done,
                    "t_ms": t,
                }
            )
            t += 3
        voiced = facts["voice"] and order == "reco"
        facts["voice"] = voiced and self.live and voice != "busy"
        events.append(dict({"type": "facts"}, **facts, t_ms=t + 6))
        template = facts["verdict_template"]
        reason = (
            "not_voiced"
            if not voiced and order == "reco"
            else ("plain" if order == "plain" else None)
        )
        if reason is None and not self.live:
            reason = "off"
        if reason is None and voice == "busy":
            reason = "busy"
        t += 8
        text = {
            "type": "text",
            "text": template,
            "generated": False,
            "label": LABEL_ALGO,
            "reason": reason,
            "guard": None,
        }
        if reason is None:
            for stage, at in (("voice", t), ("verify", t + 1500)):
                events.append(
                    {
                        "type": "stage",
                        "id": stage,
                        "text": STAGES[stage][0],
                        "done": STAGES[stage][1],
                        "t_ms": at,
                    }
                )
            t += 1504
            if voice == "guard":
                text.update(reason="guard", guard="numbers")
            else:
                text.update(generated=True, label=LABEL_AI)
        events.append(dict(text, t_ms=t))
        events.append({"type": "done", "t_ms": t + 1})
        return events


STAND = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Стенд листа «Сомелье»</title>
<link rel="icon" href="data:,">
<link rel="stylesheet" href="/static/somm.css">
<style>
:root {{
{tokens}
  --text-2: #857e79;
  color-scheme: light;
}}
@media (max-width: 359px) {{ :root {{ --gutter: 16px; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 400 16px/1.5 var(--font); }}
button, input {{ font: inherit; color: inherit; }}
.stand {{ max-width: 520px; margin: 0 auto; padding: 16px var(--gutter) 48px; }}
.stand nav {{ display: flex; flex-wrap: wrap; gap: 8px 14px; font-size: 14px; }}
.stand nav a {{ color: var(--accent); }}
.stand h1 {{ font: 500 36px/1.25 var(--display); margin: 24px 0 0; }}
.stand h2 {{ font: 500 24px/1.25 var(--display); margin: 32px 0 14px; }}
.stand .meta {{ margin: 4px 0 0; color: var(--text-3); font-size: 14px; }}
.stand .block {{ margin-top: 18px; }}
.stand .log {{ margin-top: 24px; font-size: 12px; color: var(--text-3); }}
</style>
</head>
<body>
<main class="stand">
  <nav id="nav"></nav>
  <div class="block" id="say"></div>
  <h1 id="name"></h1>
  <p class="meta" id="meta"></p>
  <div class="block" id="note"></div>
  <h2>Какое это вино</h2>
  <div id="radar"></div>
  <div class="block" id="scales"></div>
  <div class="block" id="entry"></div>
  <p class="log" id="log"></p>
</main>
<script src="/static/somm.js"></script>
<script>
(function () {{
  "use strict";
  var WINES = {wines};
  var params = new URLSearchParams(location.search);
  var key = WINES[params.get("wine")] ? params.get("wine") : "red";
  var order = "reco";
  var log = document.getElementById("log");
  var nav = document.getElementById("nav");
  Object.keys(WINES).forEach(function (name) {{
    var a = document.createElement("a");
    a.href = "?wine=" + name + (params.get("voice") ? "&voice=" + params.get("voice") : "");
    a.textContent = name;
    nav.appendChild(a);
  }});
  SommUI.init({{
    order: function () {{ return order; }},
    setOrder: function (next) {{ order = next; log.textContent = "Сортировка: " + next; }},
    openWine: function (slug) {{ log.textContent = "Открыть карточку: " + slug; }},
    scan: function () {{ log.textContent = "Новый скан"; }},
    // Запись истории листа — как у страницы продукта: своё «назад» асинхронное, и его поздний
    // popstate не закрывает лист, открытый заново до его прихода, а ставит его запись.
    onSheet: function (open) {{
      if (open) {{
        if (!entry && !backing) push();
        return;
      }}
      if (!entry) return;
      entry = false;
      backing = true;
      history.back();
    }}
  }});
  var entry = false, backing = false;
  function push() {{ history.pushState({{ somm: true }}, ""); entry = true; }}
  window.addEventListener("popstate", function (event) {{
    var state = event.state || {{}};
    if (backing) {{
      backing = false;
      if (!state.somm) {{
        if (SommUI.isOpen() && !entry) push();
        return;
      }}
    }}
    if (entry && !state.somm) {{
      entry = false;
      if (SommUI.isOpen()) SommUI.close({{ fromHistory: true }});
    }}
  }});
  SommUI.say(document.getElementById("say"), "Сфотографируйте бутылку — расскажу, что это за вино, к чему подать и чем заменить");
  fetch("/v1/wines/" + WINES[key] + "/sommelier?order=" + order).then(function (r) {{ return r.json(); }}).then(function (data) {{
    data.photo_url = "/v1/wines/" + data.slug + "/photo";
    document.getElementById("name").textContent = data.name;
    document.getElementById("meta").textContent = data.winery;
    SommUI.note(document.getElementById("note"), data);
    var drawn = SommUI.radar(document.getElementById("radar"), data.profile);
    if (!drawn) document.getElementById("radar").textContent = "Известных осей меньше трёх";
    SommUI.scales(document.getElementById("scales"), data.profile);
    SommUI.entry(document.getElementById("entry"), data);
    var chip = params.get("chip"), q = params.get("q");
    var opts = {{ name: data.name, winery: data.winery, photo_url: data.photo_url, data: data }};
    if (chip) {{ opts.chip = chip; SommUI.open(data.slug, opts); }}
    else if (q) {{ opts.question = q; SommUI.open(data.slug, opts); }}
  }});
}})();
</script>
</body>
</html>
"""


def stand_page(shot: bool = False) -> bytes:
    tokens = load("tokens.json")
    rows = []
    for group in ("colors", "gradients", "fonts", "layout", "motion"):
        rows += [f"  {name}: {value};" for name, value in tokens[group].items()]
    wines = json.dumps({key: load(name)["slug"] for key, name in WINES.items()})
    html = STAND.format(tokens="\n".join(rows), wines=wines)
    if shot:
        html = html.replace("</body>", '<script src="/__stub/driver.js"></script>\n</body>', 1)
    return html.encode("utf-8")


def frame_page(query: dict[str, str]) -> bytes:
    """Стенд во фрейме заданного размера: безголовый Chrome не делает окно уже 500 px."""
    width = int(query["w"]) if query.get("w", "").isdigit() else 390
    height = int(query["h"]) if query.get("h", "").isdigit() else 844
    shot = re.sub(r"[^a-z_]", "", query.get("shot", ""))
    wine = query.get("wine", "") if query.get("wine", "") in WINES else "red"
    html = (
        '<!DOCTYPE html><meta charset="utf-8"><style>html,body{margin:0;background:#fff}'
        f"iframe{{display:block;width:{width}px;height:{height}px;border:0}}</style>"
        f'<iframe src="/?wine={wine}&shot={shot}" title="стенд"></iframe>'
    )
    return html.encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    """Маршруты договора сомелье поверх `Stub`; всё остальное — 404."""

    stub: Stub
    verbose = False
    server_version = "svs-somm-stub"

    def log_message(self, format: str, *args: Any) -> None:  # имя из базового класса
        # Вопрос гостя не пишется никуда (договор §1 «Журнал») — и в заглушке тоже.
        if self.verbose:
            sys.stderr.write("somm-stub: " + self.command + " " + urlsplit(self.path).path + "\n")

    def _send(self, status: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _query(self, url: str) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlsplit(url).query).items()}

    def _order(self) -> str | None:
        order = self._query(self.path).get("order", "reco")
        if order not in ORDERS:
            self._json(
                422, {"detail": [{"type": "enum", "loc": ["query", "order"], "input": order}]}
            )
            return None
        return order

    def do_GET(self) -> None:  # имя из http.server
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            shot = bool(self._query(self.path).get("shot"))
            self._send(200, stand_page(shot), "text/html; charset=utf-8")
        elif path == "/__stub/driver.js":
            self._send(200, DRIVER.read_bytes(), STATIC_TYPES[".js"])
        elif path == "/__stub/frame":
            self._send(200, frame_page(self._query(self.path)), "text/html; charset=utf-8")
        elif path in ("/static/somm.js", "/static/somm.css"):
            file = STATIC / path.rsplit("/", 1)[1]
            self._send(200, file.read_bytes(), STATIC_TYPES[file.suffix])
        elif m := re.fullmatch(r"/v1/wines/([^/]+)/photo", path):
            photo = self.stub.photo(m.group(1))
            if photo is None:
                self._json(404, {"detail": "фото каталога нет на этой машине"})
            else:
                self._send(200, photo.read_bytes(), "image/webp", "public, max-age=86400")
        elif m := re.fullmatch(r"/v1/wines/([^/]+)/sommelier", path):
            order = self._order()
            if order is None:
                return
            card = self.stub.card(m.group(1), order)
            if card is None:
                self._json(404, {"detail": f"нет карточки '{m.group(1)}'"})
            else:
                self._json(200, card)
        else:
            self._json(404, {"detail": "Not Found"})

    def do_POST(self) -> None:  # имя из http.server
        if urlsplit(self.path).path != "/v1/sommelier/ask":
            self._json(404, {"detail": "Not Found"})
            return
        order = self._order()
        if order is None:
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._json(422, {"detail": [{"type": "json_invalid", "loc": ["body"]}]})
            return
        problem = self._invalid(body)
        if problem:
            self._json(422, {"detail": problem})
            return
        slug = body["slug"]
        if slug not in self.stub.cards:
            self._json(404, {"detail": f"нет карточки '{slug}'"})
            return
        voice = self._query(self.headers.get("Referer") or "").get("voice", "live")
        events = self.stub.ask(slug, body, order, voice if voice in VOICES else "live")
        self._stream(events)

    def _invalid(self, body: Any) -> Any:
        if not isinstance(body, dict) or not isinstance(body.get("slug"), str):
            return [{"type": "missing", "loc": ["body", "slug"]}]
        question, chip = body.get("question"), body.get("chip")
        if (question is None) == (chip is None):
            return [
                {"type": "value_error", "loc": ["body"], "msg": "ровно одно из question и chip"}
            ]
        if question is not None:
            if not self.stub.input:
                return INPUT_OFF
            if (
                not isinstance(question, str)
                or not question.strip()
                or len(question.strip()) > MAX_QUESTION
            ):
                return [{"type": "string_type", "loc": ["body", "question"]}]
        if chip is not None and chip not in CHIPS:
            return [{"type": "enum", "loc": ["body", "chip"], "input": chip}]
        return None

    def _stream(self, events: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        previous = 0
        try:
            for event in events:
                wait = max(int(event.get("t_ms", previous)) - previous, 0) / 1000 * self.stub.speed
                previous = int(event.get("t_ms", previous))
                if wait:
                    time.sleep(wait)
                self.wfile.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # лист закрыли: поток обрывается, как у живого сервера


def make_server(
    host: str = "127.0.0.1",
    port: int = 8766,
    *,
    data_dir: Path | None = None,
    live: bool = True,
    ask_input: bool = True,
    speed: float = 1.0,
    verbose: bool = False,
) -> ThreadingHTTPServer:
    """Сервер заглушки; `port=0` — свободный порт, `speed=0` — поток без пауз (для тестов)."""
    stub = Stub(data_dir=data_dir, live=live, ask_input=ask_input, speed=speed)
    handler = type("SommStubHandler", (Handler,), {"stub": stub, "verbose": verbose})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, name="somm-stub", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("SVS_DATA_DIR") or None)
    parser.add_argument("--no-live", action="store_true", help="SVS_SOMM_LIVE=0: только шаблоны")
    parser.add_argument("--no-input", action="store_true", help="SVS_SOMM_INPUT=0: только чипы")
    parser.add_argument("--speed", type=float, default=1.0, help="множитель пауз потока")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir) if args.data_dir else None
    server = make_server(
        args.host,
        args.port,
        data_dir=data_dir,
        live=not args.no_live,
        ask_input=not args.no_input,
        speed=args.speed,
        verbose=args.verbose,
    )
    print(f"стенд листа «Сомелье»: http://{args.host}:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
