"""Заглушки «Сомелье» (`tests/fixtures/somm/`) соответствуют договору `docs/api-sommelier.md`.

Сомелье собирается из шести частей на этих файлах: ворота и голос, данные сборки, маршрут и
ответы, лист на странице, страница, приёмка. Расхождение заглушки с договором — это расхождение
частей между собой, поэтому формы, коды, метки и право проверяются здесь на чистом Python, без
сервиса, модели и видеокарты. Эталонное правило вопроса экрана `check` (§5) написано здесь же:
реализация в `after_layer` обязана давать те же вопросы на заглушке `scan_check_question.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.reading.contracts import Color, SugarClass
from app.reading.taxonomy import canonical_grape
from app.recommend.build import REGION_CODES, build_profile, load_priors
from app.recommend.content_filter import check
from app.recommend.profile import AXES, STYLE_AXES, RussianPGI, WineKind

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "somm"
DOC = ROOT / "docs" / "api-sommelier.md"

LABEL_AI = "Текст — ИИ, подбор — алгоритм"
LABEL_ALGO = "Текст и подбор — алгоритм"
NOTICE = "Применяются рекомендательные технологии"
PORTAL = "https://vino-svoe.ru/wines/"

INTENTS = (
    "what_to_eat",
    "serve",
    "softer",
    "fresher",
    "replace",
    "grape",
    "term",
    "dish_check",
    "guided",
    "fact",
    "refuse",
    "smalltalk",
    "unknown",
)
#: Факты карточки намерения `fact` и правила хранения по стилю (§4.5).
FACTS = ("sugar", "alcohol", "storage")
STORAGE_RULES = ("sparkling", "sweet", "white", "red_full", "red", "red_unknown", "unknown")
CHIP_ARGS: dict[str, set[str]] = {
    "what_to_eat": set(),
    "serve": set(),
    "softer": {"dish"},
    "fresher": {"dish"},
    "replace": set(),
    "grape": {"grape"},
    "term": {"topic"},
    "dish_check": {"dish"},
    "guided": {"food", "want"},
}
REQUIRED_ARGS = {"grape": {"grape"}, "term": {"topic"}, "dish_check": {"dish"}}
GUIDED_FOOD = {"meat", "fish", "cheese", "dessert", "none"}
GUIDED_WANT = {"softer", "fresher", "none"}
REFUSALS = {
    "minors": "Вино — только с восемнадцати, поэтому советовать не стану.",
    "medical": "Это вопрос к врачу, а не к сомелье: про вино при беременности, болезни и "
    "лекарствах я не консультирую.",
    "addiction": "С этим я не помогу и советовать вино не стану. Если тревожит зависимость, "
    "обратитесь к врачу.",
    "sobriety": "Про опьянение и вытрезвление не консультирую — это вопрос здоровья.",
    "driving": "Тут советовать не стану: вопрос не про вкус вина, а про безопасность.",
    "health": "О пользе и вреде алкоголя не рассуждаю. Могу рассказать, как подать это вино и "
    "к чему.",
    "other_alcohol": "Про крепкий алкоголь и пиво не расскажу — я рассказываю о винах каталога.",
    "purchase": "О покупке и стоимости не рассказываю: сервис справочный, а не магазин. Могу "
    "подсказать, как подать это вино.",
    "price": "О покупке и стоимости не рассказываю: сервис справочный, а не магазин. Могу "
    "подсказать, как подать это вино.",
    "recipe": "Рецептами не помогу — я про вино. Назовите блюдо, и я скажу, как с ним "
    "сочетается это вино.",
}
STICKY = {"minors", "medical", "addiction", "sobriety"}
GUARDS = (
    "think",
    "empty",
    "length",
    "cutoff",
    "preamble",
    "dish_as_wine",
    "false_refusal",
    "numbers",
    "entities",
    "descriptors",
    "stoplist",
    "legal",
)
EVENTS = ("stage", "facts", "text", "error", "done")
STAGES = {
    "card": ("Открываю карточку «", "Открыл карточку"),
    "rules": ("Проверяю правила сочетаний", "Проверил правила сочетаний"),
    "catalog": ("Ищу в каталоге вина других виноделен", "Посмотрел каталог"),
    "voice": ("Сомелье формулирует ответ", "Сомелье сформулировал"),
    "verify": ("Сверяю имена и числа", "Сверил имена и числа"),
}
REASONS = (
    "not_voiced",
    "plain",
    "off",
    "not_ready",
    "busy",
    "quiet",
    "locked",
    "preempted",
    "timeout",
    "unavailable",
    "guard",
    "error",
    "cancelled",
)
VERDICTS = ("yes", "caveat", "no", "neutral")
BASIS = {
    "карточка каталога",
    "правила сочетаний",
    "профиль по сорту",
    "правила подачи",
    "справочник",
    "подбор из каталога",
}
AXIS_UI = {
    "sweetness": ("Сладость", "сухое", "сладкое"),
    "acidity": ("Кислотность", "мягкая", "живая"),
    "tannin": ("Танины", "мягкие", "терпкие"),
    "body": ("Тело", "лёгкое", "плотное"),
    "alcohol": ("Крепость", "лёгкое", "крепкое"),
    "oak": ("Дуб", "без дуба", "заметный"),
    "aroma_intensity": ("Аромат", "сдержанный", "яркий"),
    "effervescence": ("Пузырьки", "тихое", "игристое"),
}
FOOD_GROUPS = ("meat", "fish", "cheese", "dessert")
#: §7.1: правила пары — по убыванию модуля веса, при равенстве — сначала правила о рыбе (в этом
#: порядке), затем по `id` (третий круг проверки 25.09).
FISH_FIRST = ("sweet_wine_on_fish", "no_tannin_with_oily_fish")


def rule_order(rule_id: str, weight: float) -> tuple[float, int, str]:
    first = FISH_FIRST.index(rule_id) if rule_id in FISH_FIRST else len(FISH_FIRST)
    return (-abs(weight), first, rule_id)


WINE_FIELDS = {*AXES, "region", "color", "grapes", "serve", "descriptors"}
SERVE_ORDER = (
    "sparkling", "sweet", "sweet_name", "orange", "white_full", "white", "rose", "red_full", "red",
)  # fmt: skip
QUESTION_FIELDS = ("sugar", "color", "sparkling", "grapes", "abv", "name")

GET_KEYS = {
    "slug",
    "name",
    "winery",
    "order",
    "note",
    "profile",
    "serve",
    "dishes",
    "chips",
    "live",
    "input",
    "notice_149",
}
FACTS_KEYS = {
    "type",
    "slug",
    "intent",
    "order",
    "verdict_template",
    "detail",
    "basis",
    "dishes",
    "wines_title",
    "wines",
    "compare",
    "serve",
    "question",
    "refusal",
    "chips",
    "notice_149",
    "voice",
    "context",
    "t_ms",
}
TEXT_KEYS = {"type", "text", "generated", "label", "reason", "guard", "t_ms"}
CONTEXT_KEYS = {"intent", "dish", "food", "want", "step", "refusal_topic", "shown"}
TILE_KEYS = {
    "slug",
    "name",
    "winery",
    "region",
    "style_label",
    "sparkling",
    "grapes",
    "photo_url",
    "portal_url",
    "reasons",
    "pill",
    "want_source",
}
CARD_KEYS = {
    "slug",
    "name",
    "winery",
    "region",
    "grapes",
    "category",
    "color",
    "sugar",
    "sugar_class",
    "photo_name",
    "color_label",
    "sugar_label",
    "sparkling",
    "style_label",
    "description",
    "description_src",
    "photo_url",
    "portal_url",
    "alcohol",
    "alcohol_max",
    "alcohol_src",
}
#: Поля карточки, пришедшие со снимка или живого портала: после решения 24.09 их нет.
PORTAL_KEYS = {
    "dishes",
    "temperature",
    "category_gradient",
    "live_category",
    "published",
    "icon_url",
}
SUGAR_WORDS = {
    "brut_nature": "брют натюр",
    "extra_brut": "экстра брют",
    "brut": "брют",
    "suhoe": "сухое",
    "polusuhoe": "полусухое",
    "polusladkoe": "полусладкое",
    "sladkoe": "сладкое",
}
STOP = re.compile(
    r"купит|купи\b|куплю|цена|цены|цену|₽|\bруб(?:л|\.|\b)|лучш|идеальн|вино недели|рейтинг"
    r"|балл|скидк|шедевр|превосходн|попробуйте|выпейте|пейте|закажите",
    re.IGNORECASE,
)
PERCENT = re.compile(r"%(?! об\.)")
ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\|/Users/|/home/")

SOMMELIER = ("sommelier_red", "sommelier_sweet", "sommelier_sparkling", "sommelier_nosugar")
STREAMS = (
    "ask_dish_check_live",
    "ask_dish_check_guard",
    "ask_what_to_eat_busy",
    "ask_refusal",
    "ask_softer_plain",
    "ask_guided_step",
    "ask_error",
)
#: Файлы ответов сервиса: всё в них видит человек. `requests.json` (там вопрос гостя),
#: `tokens.json`, `vocab.json` и `pairs.json` — не ответы.
ANSWERS = (
    *(f"{name}.json" for name in SOMMELIER),
    *(f"{name}.ndjson" for name in STREAMS),
    "wine_card_noportal.json",
    "scan_check_question.json",
    "dishes.json",
    "serve.json",
    "knowledge.json",
)


# ------------------------------------------------------------------ загрузка
def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def events(name: str) -> list[dict[str, Any]]:
    text = (FIXTURES / f"{name}.ndjson").read_text(encoding="utf-8")
    assert text.endswith("\n") and "\n\n" not in text, f"{name}: пустые строки в NDJSON"
    return [json.loads(line) for line in text.splitlines()]


def strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in strings(v)]
    return []


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dishes() -> dict[str, dict[str, Any]]:
    return {dish["id"]: dish for dish in load("dishes.json")["dishes"]}


@pytest.fixture(scope="module")
def rules() -> dict[str, dict[str, Any]]:
    return {rule["id"]: rule for rule in load("dishes.json")["rules"]}


@pytest.fixture(scope="module")
def knowledge() -> dict[str, Any]:
    return load("knowledge.json")


# ------------------------------------------------------------------ общие проверки объектов
def check_profile(profile: list[dict[str, Any]]) -> None:
    assert [item["axis"] for item in profile] == list(AXES)
    for item in profile:
        assert set(item) == {"axis", "label", "left", "right", "value", "source"}
        assert (item["label"], item["left"], item["right"]) == AXIS_UI[item["axis"]]
        assert item["source"] in ("catalog", "grape", "style", None)
        assert (item["value"] is None) == (item["source"] is None), item
        # По стилю — только пять осей «по сорту»: сахар и крепость — факты карточки.
        if item["source"] == "style":
            assert item["axis"] in STYLE_AXES, item
        if item["value"] is not None:
            assert 0.0 <= item["value"] <= 1.0
            assert round(item["value"], 2) == item["value"]


def check_chip(chip: dict[str, Any], dishes: dict, knowledge: dict) -> None:
    assert set(chip) <= {"id", "text", "args"} and chip["text"]
    assert chip["id"] in CHIP_ARGS, chip
    args = chip.get("args", {})
    assert set(args) <= CHIP_ARGS[chip["id"]], chip
    assert REQUIRED_ARGS.get(chip["id"], set()) <= set(args), chip
    if "dish" in args:
        assert args["dish"] in dishes, chip
    if "grape" in args:
        assert args["grape"] in knowledge["grapes"], chip
    if "topic" in args:
        assert args["topic"] in {topic["id"] for topic in knowledge["topics"]}, chip
    if "food" in args:
        assert args["food"] in GUIDED_FOOD, chip
    if "want" in args:
        assert args["want"] in GUIDED_WANT, chip


def check_dish(dish: dict[str, Any], dishes: dict, rules: dict) -> None:
    assert set(dish) == {"id", "name", "category", "verdict", "plus", "minus", "source"}
    # Источник блюда — общий источник всех его чипов; разные или чипов нет — null (§1).
    sources = {chip["source"] for chip in (*dish["plus"], *dish["minus"])}
    assert dish["source"] == (next(iter(sources)) if len(sources) == 1 else None), dish
    assert dish["id"] in dishes
    assert dish["name"] == dishes[dish["id"]]["name"]
    assert dish["category"] == dishes[dish["id"]]["category"]
    assert dish["verdict"] in VERDICTS
    assert len(dish["plus"]) <= 3 and len(dish["minus"]) <= 2
    for sign, chips in (("+", dish["plus"]), ("−", dish["minus"])):
        weights = [abs(rules[chip["id"]]["weight"]) for chip in chips]
        assert weights == sorted(weights, reverse=True), dish["id"]
        for chip in chips:
            assert set(chip) == {"id", "text", "source"}
            assert rules[chip["id"]]["sign"] == sign, chip
            assert chip["text"] == rules[chip["id"]]["chip"]
            assert chip["source"] in ("catalog", "grape", "type")
    if dish["verdict"] == "yes":
        assert not dish["minus"]


def check_tile(tile: dict[str, Any], order: str) -> None:
    assert set(tile) == TILE_KEYS
    assert tile["portal_url"] == PORTAL + tile["slug"]
    assert tile["photo_url"] in (None, f"/v1/wines/{tile['slug']}/photo")
    assert len(tile["reasons"]) <= 4
    assert tile["want_source"] in ("catalog", "grape", None)
    if order == "plain":
        assert tile["reasons"] == [] and tile["pill"] is None and tile["want_source"] is None


# ------------------------------------------------------------------ договор и заглушки согласны
def test_doc_names_every_code(doc: str) -> None:
    """Каждый код, который шлёт или ждёт сервис, назван в договоре — иначе части сервиса его не знают."""
    codes = [
        *INTENTS,
        *CHIP_ARGS,
        *REFUSALS,
        *GUARDS,
        *EVENTS,
        *STAGES,
        *REASONS,
        *VERDICTS,
        *FOOD_GROUPS,
        *SERVE_ORDER,
        *QUESTION_FIELDS,
        *FACTS,
        *STORAGE_RULES,
    ]
    missing = [code for code in codes if f"`{code}`" not in doc]
    assert not missing, f"не названы в docs/api-sommelier.md: {missing}"
    for text in (LABEL_AI, LABEL_ALGO, NOTICE, *REFUSALS.values()):
        assert text in doc, text
    tokens = load("tokens.json")
    names = [
        name
        for group in ("colors", "gradients", "fonts", "layout", "motion")
        for name in tokens[group]
    ]
    assert [name for name in names if f"`{name}`" not in doc] == []
    for name, value in tokens["colors"].items():
        if value.startswith("#"):
            assert value in doc, name


def test_requests_cover_every_answer() -> None:
    requests = load("requests.json")
    expected = {f"{name}.ndjson" for name in STREAMS} | {f"{name}.json" for name in SOMMELIER}
    assert expected | {"wine_card_noportal.json"} == set(requests)
    for name in STREAMS:
        request = requests[f"{name}.ndjson"]
        assert (request["method"], request["path"]) == ("POST", "/v1/sommelier/ask")
        body = request["body"]
        assert ("question" in body) != ("chip" in body), name
        assert len(body.get("question", "")) <= 200
        assert request["query"]["order"] in ("reco", "plain")
        facts = [event for event in events(name) if event["type"] == "facts"]
        for event in facts:
            assert event["order"] == request["query"]["order"], name
            assert event["slug"] == body["slug"]
    for name in SOMMELIER:
        request = requests[f"{name}.json"]
        body = load(f"{name}.json")
        assert request["path"] == f"/v1/wines/{body['slug']}/sommelier"
        assert body["order"] == request["query"]["order"]


# ------------------------------------------------------------------ GET /v1/wines/{slug}/sommelier
@pytest.mark.parametrize("name", SOMMELIER)
def test_sommelier_get(name: str, dishes: dict, rules: dict, knowledge: dict) -> None:
    body = load(f"{name}.json")
    assert set(body) == GET_KEYS
    assert body["order"] in ("reco", "plain")
    note = body["note"]
    assert set(note) == {"text", "basis", "generated", "label"}
    assert note["generated"] is False and note["label"] == LABEL_ALGO
    assert note["text"].startswith(f"{body['name']} — ")
    check_profile(body["profile"])
    serve = body["serve"]
    assert set(serve) == {"temperature_c", "source", "rule", "by_grape"}
    assert serve["source"] == "rule" and serve["rule"] in SERVE_ORDER
    low, high = serve["temperature_c"]
    assert isinstance(low, int) and isinstance(high, int) and low < high
    rule = next(item for item in load("serve.json")["rules"] if item["id"] == serve["rule"])
    assert rule["temperature_c"] == serve["temperature_c"]
    assert f"Подают при {low}–{high} °C." in note["text"]
    assert len(body["dishes"]) <= 3
    for dish in body["dishes"]:
        check_dish(dish, dishes, rules)
        assert dish["verdict"] in ("yes", "caveat")
    pairs = load("pairs.json")["wines"][body["slug"]]
    assert [dish["id"] for dish in body["dishes"]] == pairs["top"][:3]
    for chip in body["chips"]:
        check_chip(chip, dishes, knowledge)
    ids = [chip["id"] for chip in body["chips"]]
    assert ids[:5] == ["what_to_eat", "serve", "softer", "fresher", "replace"]
    assert ids[-1] == "guided"
    assert isinstance(body["live"], bool) and isinstance(body["input"], bool)
    assert body["notice_149"] == (NOTICE if body["order"] == "reco" else None)


def test_sommelier_sources_follow_facts() -> None:
    """Сладкое без сорта в приорах — пять осей «по сорту» оценены по стилю; без сахара и крепости
    — их оси пусты: по стилю они не оцениваются."""
    sweet = {item["axis"]: item for item in load("sommelier_sweet.json")["profile"]}
    assert {axis for axis, item in sweet.items() if item["source"] == "style"} == set(STYLE_AXES)
    assert sweet["sweetness"]["source"] == sweet["alcohol"]["source"] == "catalog"
    # Белое не бывает танинным и по стилю.
    assert sweet["tannin"]["value"] <= round(0.3 / 5, 2)
    nosugar = {item["axis"]: item for item in load("sommelier_nosugar.json")["profile"]}
    assert nosugar["sweetness"]["value"] is None and nosugar["alcohol"]["value"] is None
    red = load("sommelier_red.json")["profile"]
    assert {item["source"] for item in red} == {"catalog", "grape"}


@pytest.mark.parametrize(
    "name", ["sommelier_red", "sommelier_sweet", "sommelier_sparkling", "sommelier_nosugar"]
)
def test_radar_has_three_axes_everywhere(name: str) -> None:
    """«Роза ветров» обязательна (решение 24.09): у каждой заглушки не меньше трёх осей."""
    profile = load(f"{name}.json")["profile"]
    assert sum(item["value"] is not None for item in profile) >= 3


def test_red_profile_is_build_profile() -> None:
    """Профиль заглушки — `build_profile` по фактам выгрузки карточки без портала, ось / 5."""
    card = load("wine_card_noportal.json")
    body = load("sommelier_red.json")
    assert body["slug"] == card["slug"]
    grapes = [code for code in (canonical_grape(label) for label in card["grapes"]) if code]
    profile = build_profile(
        grapes,
        Color(card["color_label"]),
        SugarClass(card["sugar_class"]),
        WineKind.SPARKLING if card["sparkling"] else WineKind.STILL,
        load_priors(),
        region=REGION_CODES.get(card["region"], RussianPGI.OTHER),
        name=card["name"],
        abv=card["alcohol"],
    )
    for item in body["profile"]:
        assert item["source"] == profile.source(item["axis"]), item["axis"]
        assert item["value"] == round(profile.axis(item["axis"]) / 5, 2), item["axis"]


# ------------------------------------------------------------------ POST /v1/sommelier/ask
@pytest.mark.parametrize("name", STREAMS)
def test_stream(name: str, dishes: dict, rules: dict, knowledge: dict) -> None:
    stream = events(name)
    kinds = "".join(event["type"][0] for event in stream)  # stage/facts/text/error/done
    assert re.fullmatch(r"s*(fs*t|e)d", kinds), f"{name}: грамматика потока нарушена: {kinds}"
    times = [event["t_ms"] for event in stream]
    assert all(isinstance(t, int) for t in times) and times == sorted(times)
    for event in stream:
        assert event["type"] in EVENTS
        if event["type"] == "stage":
            assert set(event) == {"type", "id", "text", "done", "t_ms"}
            text, done = STAGES[event["id"]]
            assert event["text"].startswith(text) and event["done"] == done
        if event["type"] == "error":
            assert set(event) == {"type", "code", "text", "t_ms"}
            assert event["code"] in ("internal", "not_ready")
    facts = next((event for event in stream if event["type"] == "facts"), None)
    if facts is None:
        return
    assert set(facts) == FACTS_KEYS
    assert facts["t_ms"] <= 150, "facts позже 150 мс"
    assert facts["intent"] in INTENTS
    assert set(facts["basis"]) <= BASIS
    order = facts["order"]
    for dish in facts["dishes"]:
        check_dish(dish, dishes, rules)
    for tile in facts["wines"]:
        check_tile(tile, order)
    assert len(facts["wines"]) <= 3
    assert len({tile["winery"] for tile in facts["wines"]}) == len(facts["wines"])
    assert (facts["wines_title"] is None) == (not facts["wines"])
    expected_notice = NOTICE if facts["wines"] and order == "reco" else None
    assert facts["notice_149"] == expected_notice
    for chip in facts["chips"]:
        check_chip(chip, dishes, knowledge)
    assert set(facts["context"]) == CONTEXT_KEYS
    assert facts["context"]["intent"] == facts["intent"]
    assert len(facts["context"]["shown"]) <= 12
    compare = facts["compare"]
    if compare is not None:
        assert order == "reco" and facts["wines"]
        assert set(compare) == {"anchor", "axes", "wines", "overlay"}
        assert compare["overlay"] in (None, *(item["slug"] for item in compare["wines"]))
        assert compare["anchor"]["slug"] == facts["slug"]
        check_profile(compare["anchor"]["profile"])
        assert [item["slug"] for item in compare["wines"]] == [t["slug"] for t in facts["wines"]]
        for item in compare["wines"]:
            check_profile(item["profile"])
        assert set(compare["axes"]) <= set(AXES)
    if facts["question"] is not None:
        assert facts["intent"] == "guided" and facts["question"]["step"] in ("food", "want")
    refusal = facts["refusal"]
    if refusal is not None:
        assert facts["intent"] == "refuse"
        assert refusal["text"] == REFUSALS[refusal["topic"]] == facts["verdict_template"]
        sticky = facts["context"]["refusal_topic"]
        assert sticky == (refusal["topic"] if refusal["topic"] in STICKY else None)
        assert not facts["voice"]
    text = next(event for event in stream if event["type"] == "text")
    assert set(text) == TEXT_KEYS
    after = stream[stream.index(facts) + 1 : stream.index(text)]
    voiced_stages = [event["id"] for event in after]
    if text["generated"]:
        assert text["label"] == LABEL_AI and text["reason"] is None and text["guard"] is None
        assert facts["voice"] and order == "reco"
        assert voiced_stages == ["voice", "verify"]
    else:
        assert text["label"] == LABEL_ALGO and text["reason"] in REASONS
        assert text["text"] == facts["verdict_template"]
        assert (text["guard"] in GUARDS) if text["reason"] == "guard" else text["guard"] is None
        if not facts["voice"]:
            assert voiced_stages == [] and text["reason"] in ("not_voiced", "plain", "off")
    if order == "plain":
        assert text["reason"] == "plain" and not facts["voice"]


# ------------------------------------------------------------------ данные data/somm
def test_dishes_and_rules(dishes: dict, rules: dict) -> None:
    data = load("dishes.json")
    assert set(data) == {"version", "source", "categories", "dishes", "rules"}
    for dish in dishes.values():
        assert set(dish) == {"id", "name", "dative", "category", "food", "family", "aliases"}
        assert dish["category"] in data["categories"]
        assert dish["dative"].startswith("к ") and dish["aliases"]
        assert dish["food"] in (*FOOD_GROUPS, None)
        category = dish["category"]
        if category == "main_meat":
            assert dish["food"] == "meat"
        if category in ("seafood", "seafood_raw"):
            assert dish["food"] == "fish"
        if category in ("dessert", "sweet_preserve"):
            assert dish["food"] == "dessert"
    assert len(rules) == 39
    order = [rule["id"] for rule in data["rules"]]
    assert order == sorted(order, key=lambda rid: (-rules[rid]["weight"], rid))
    for rule in rules.values():
        assert set(rule) == {"id", "sign", "weight", "name", "chip", "text", "wine_fields"}
        assert rule["sign"] == ("+" if rule["weight"] > 0 else "−")
        assert 0 < len(rule["chip"]) <= 30
        assert rule["text"].endswith(".")
        assert rule["wine_fields"] and set(rule["wine_fields"]) <= WINE_FIELDS
    assert "portal_editorial_prior" not in rules


def test_pairs(dishes: dict, rules: dict) -> None:
    data = load("pairs.json")
    assert set(data) == {"version", "built", "wines"}
    sweet = load("sommelier_sweet.json")["slug"]
    nosugar = load("sommelier_nosugar.json")["slug"]
    for slug, entry in data["wines"].items():
        assert set(entry) == {"top", "dishes"}
        for dish_id, (verdict, plus, minus) in entry["dishes"].items():
            assert dish_id in dishes and verdict in VERDICTS
            assert all(rules[rid]["sign"] == "+" for rid in plus)
            assert all(rules[rid]["sign"] == "−" for rid in minus)
            for group in (plus, minus):
                assert group == sorted(group, key=lambda rid: rule_order(rid, rules[rid]["weight"]))
            assert (verdict == "neutral") == (not plus and not minus)
            if verdict == "yes":
                assert not minus
            if any(rules[rid]["weight"] <= -3.0 for rid in minus):
                assert verdict == "no", (slug, dish_id)
        top = entry["top"]
        assert 1 <= len(top) <= 5 and len(set(top)) == len(top)
        assert all(entry["dishes"][dish_id][0] in ("yes", "caveat") for dish_id in top)
        assert len({dishes[dish_id]["family"] for dish_id in top}) == len(top)
        categories = {dishes[dish_id]["category"] for dish_id in top}
        if slug == sweet:
            assert categories <= {"dessert", "sweet_preserve", "appetizer"}
        if slug == nosugar:
            assert not categories & {"dessert", "sweet_preserve"}


def test_serve_rules() -> None:
    data = load("serve.json")
    assert [rule["id"] for rule in data["rules"]] == list(SERVE_ORDER)
    for rule in data["rules"]:
        assert set(rule) == {"id", "when", "temperature_c", "text"}
        when = set(rule["when"])
        assert when <= {"sparkling", "sugar", "color", "body_min", "sweet_name"} and when
        low, high = rule["temperature_c"]
        assert 4 <= low < high <= 20
        assert rule["text"].endswith(".")


def test_vocab_and_knowledge(dishes: dict, knowledge: dict) -> None:
    vocab = load("vocab.json")
    keys = {
        "wineries",
        "winery_words",
        "grapes",
        "dishes",
        "cooking",
        "regions",
        "descriptors",
        "generic",
        "taste_stems",
    }
    assert set(vocab) == {"version", *keys}
    for key in keys:
        assert vocab[key] and all(isinstance(word, str) and word for word in vocab[key]), key
    assert all(word == word.lower() for word in vocab["winery_words"])
    for dish in dishes.values():
        assert dish["name"].lower() in vocab["dishes"]
    assert set(knowledge) == {"version", "source", "topics", "grapes"}
    for topic in knowledge["topics"]:
        assert set(topic) == {"id", "name", "triggers", "context", "answer", "detail", "chips"}
        for chip in topic["chips"]:
            check_chip(chip, dishes, knowledge)
    for code, grape in knowledge["grapes"].items():
        assert set(grape) == {"label", "text", "count", "regions"}, code
        assert grape["count"] > 0 and grape["regions"]


# ------------------------------------------------------------------ вопрос экрана check (§5)
def _alternatives(labels: list[str]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} или {labels[1]}"
    return f"{', '.join(labels[:-1])} или {labels[-1]}"


def _number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".").replace(".", ",")


def question_of(read: dict[str, Any], candidates: list[dict[str, Any]]) -> dict | None:
    """Эталонное правило §5 договора: один вопрос по полю, которым различаются варианты."""
    if len(candidates) < 2:
        return None

    def agrees(candidate: dict[str, Any]) -> bool:
        for key in ("sugar", "color", "sparkling", "abv"):
            ours, theirs = read.get(key), candidate["facts"].get(key)
            if ours is not None and theirs is not None and ours != theirs:
                return False
        return True

    agreeing = [c for c in candidates if agrees(c)]
    pool = agreeing if len(agreeing) >= 2 else candidates
    read_fields = {
        key for key in ("sugar", "color", "sparkling", "abv") if read.get(key) is not None
    }
    for field in QUESTION_FIELDS:
        if field in read_fields:
            continue
        if field == "grapes" and any(len(c["facts"]["grapes"]) > 2 for c in pool):
            continue
        options: dict[Any, str] = {}
        for c in pool:
            value = c["name"] if field == "name" else c["facts"].get(field)
            if isinstance(value, list):
                value = tuple(value) or None
            if value is not None and value not in options:
                options[value] = c["slug"]
        if len(options) < 2:
            continue
        values = list(options)[:4]
        if field == "sugar":
            labels = [SUGAR_WORDS[v] for v in values]
            text = f"Что на этикетке: {_alternatives(labels)}?"
        elif field == "color":
            labels = [v.lower() for v in values]
            text = f"Какого цвета вино: {_alternatives(labels)}?"
        elif field == "sparkling":
            labels = ["игристое" if v else "тихое" for v in values]
            text = "Вино игристое или тихое?"
        elif field == "grapes":
            labels = [" и ".join(v) for v in values]
            text = f"Какой сорт на этикетке: {_alternatives(labels)}?"
        elif field == "abv":
            numbers = [_number(v) for v in values]
            labels = [f"{n} % об." for n in numbers]
            text = f"Какая крепость на этикетке: {_alternatives(numbers)} % об.?"
        else:
            labels = list(values)
            text = f"Какое название на этикетке: {_alternatives([f'«{v}»' for v in values])}?"
        return {
            "field": field,
            "text": text,
            "options": [
                {"label": label, "slug": options[value]}
                for label, value in zip(labels, values, strict=True)
            ],
        }
    return None


@pytest.mark.parametrize("case", load("scan_check_question.json")["cases"], ids=lambda c: c["name"])
def test_check_question(case: dict[str, Any]) -> None:
    assert question_of(case["read"], case["candidates"]) == case["question"]
    question = case["question"]
    if question is not None:
        assert question["field"] in QUESTION_FIELDS and question["text"].endswith("?")
        slugs = [c["slug"] for c in case["candidates"]]
        assert all(option["slug"] in slugs for option in question["options"])
        assert 2 <= len(question["options"]) <= 4


# ------------------------------------------------------------------ карточка без портала
def test_card_without_portal() -> None:
    card = load("wine_card_noportal.json")
    assert set(card) == CARD_KEYS
    assert not PORTAL_KEYS & set(card)
    assert card["portal_url"] == PORTAL + card["slug"]
    assert card["photo_url"] == f"/v1/wines/{card['slug']}/photo"
    assert card["description_src"] in ("catalog", None)
    assert card["alcohol_src"] in ("catalog", None)
    assert card["style_label"] == f"{card['color_label']} {card['sugar_label']}"
    assert card["sugar"] == card["sugar_label"] == SUGAR_WORDS[card["sugar_class"]]


# ------------------------------------------------------------------ право
@pytest.mark.parametrize("name", ANSWERS)
def test_legal(name: str) -> None:
    """Ни стоп-слов, ни `%` вне «% об.», ни абсолютных путей; каждая строка проходит фильтр."""
    path = FIXTURES / name
    raw = path.read_text(encoding="utf-8")
    assert not ABS_PATH.search(raw), name
    if path.suffix == ".ndjson":
        texts = [s for line in raw.splitlines() for s in strings(json.loads(line))]
    else:
        texts = strings(json.loads(raw))
    for text in texts:
        assert not STOP.search(text), f"{name}: {text}"
        assert not PERCENT.search(text), f"{name}: {text}"
        verdict = check(text)
        assert verdict.clean, f"{name}: {verdict.violations}: {text}"
