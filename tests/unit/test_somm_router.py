"""Маршрут вопроса сомелье (`app/sommelier/router.py`), тексты и шаблоны ответов.

Главная приёмка дорожки — точность маршрута на наборе `tests/fixtures/somm/routing_cases.json`
(≥ 150 фраз, часть — из `routing_cases.json` «Лозы» с ожиданиями под намерения сомелье на
карточке): не ниже 90 %. Промахи набора печатаются в сообщении проверки — их видно, а не
прячут подгонкой ожиданий. Данные — заглушки договора (24 блюда, 4 темы, 4 портрета сорта).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.recommend.content_filter import check
from app.recommend.somm_data import load_somm_data
from app.sommelier import smalltalk
from app.sommelier import templates as t
from app.sommelier.router import (
    CHIP_ARGS,
    INTENTS,
    Context,
    ContextError,
    DishIndex,
    Route,
    Router,
    context_of,
)
from app.sommelier.text import (
    clean_question,
    count_wines,
    normalize,
    plural,
    words_match,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "somm"
CASES = json.loads((FIXTURES / "routing_cases.json").read_text(encoding="utf-8"))
#: Приёмка маршрута: не ниже 90 % на наборе.
MIN_ACCURACY = 0.90
MIN_CASES = 150


@pytest.fixture(scope="module")
def data() -> Any:
    return load_somm_data(None, fallback=FIXTURES, strict=True)


@pytest.fixture(scope="module")
def router(data: Any) -> Router:
    return Router(data)


def slots_of(route: Route) -> dict[str, Any]:
    return {
        "dish": route.dish,
        "grape": route.grape,
        "topic": route.topic,
        "food": route.food,
        "want": route.want,
        "refusal": route.refusal.topic if route.refusal else None,
        "unknown_dish": route.unknown_dish,
        "fact": route.fact,
    }


def route_case(router: Router, data: Any, case: dict[str, Any]) -> Route:
    context = context_of(case.get("context"), data.dishes)
    return router.text(case["question"], context, grapes=CASES["grapes"])


def matches(route: Route, case: dict[str, Any]) -> bool:
    slots = slots_of(route)
    return route.intent == case["intent"] and all(
        slots[key] == value for key, value in (case.get("slots") or {}).items()
    )


# ------------------------------------------------------------------ набор фраз
def test_cases_are_well_formed(data: Any) -> None:
    cases = CASES["cases"]
    assert len(cases) >= MIN_CASES
    questions = [case["question"] for case in cases if not case.get("context")]
    assert len(questions) == len(set(questions)), "повтор фразы в наборе"
    assert {case["intent"] for case in cases} == set(INTENTS), "каждое намерение в наборе"
    assert sum(case["source"] == "loza" for case in cases) >= 60
    for case in cases:
        assert case["intent"] in INTENTS, case
        assert case["source"] in ("loza", "contract", "new"), case
        slots = case.get("slots") or {}
        if "dish" in slots:
            assert slots["dish"] in data.dishes, case
        if "grape" in slots:
            assert slots["grape"] in data.grapes, case
        if "topic" in slots:
            assert slots["topic"] in data.topics, case
        assert len(case["question"]) <= 200


def test_routing_accuracy(router: Router, data: Any) -> None:
    """Приёмка: ≥ 90 % фраз набора получают своё намерение и слоты."""
    misses = []
    for case in CASES["cases"]:
        route = route_case(router, data, case)
        if not matches(route, case):
            misses.append(f"{case['question']!r}: ждали {case['intent']}, получили {route}")
    accuracy = 1 - len(misses) / len(CASES["cases"])
    assert accuracy >= MIN_ACCURACY, f"точность {accuracy:.1%}; промахи:\n" + "\n".join(misses)


@pytest.mark.parametrize("intent", ["refuse", "smalltalk", "dish_check"])
def test_routing_per_intent_is_perfect_where_safety_matters(router, data, intent) -> None:
    """Отказы, беседа и блюда набора — без промахов: здесь ошибка стоит дороже всего."""
    cases = [case for case in CASES["cases"] if case["intent"] == intent]
    wrong = [c["question"] for c in cases if not matches(route_case(router, data, c), c)]
    assert not wrong


# ------------------------------------------------------------------ настоящие данные
#: Полная сборка `scripts/build_somm.py` (82 блюда, 32 темы): где её нет — проверка пропускается.
REAL_SOMM = Path(os.environ.get("SVS_DATA_DIR") or "__нет__") / "somm"
CASES_REAL = json.loads((FIXTURES / "routing_cases_real.json").read_text(encoding="utf-8"))
needs_real = pytest.mark.skipif(
    not (REAL_SOMM / "pairs.json").is_file(), reason="нужны данные сомелье: SVS_DATA_DIR"
)


@pytest.fixture(scope="module")
def real_data() -> Any:
    return load_somm_data(REAL_SOMM, fallback=None, strict=True)


def real_cases() -> list[dict[str, Any]]:
    """Набор на настоящих данных: основной с ожиданиями `real` и фразы только для них."""
    cases = [{**case, **case["real"]} if case.get("real") else case for case in CASES["cases"]]
    return cases + CASES_REAL["cases"]


def test_real_cases_are_well_formed() -> None:
    questions = [case["question"] for case in CASES_REAL["cases"]]
    assert len(questions) == len(set(questions)) and len(questions) >= 40
    assert not set(questions) & {case["question"] for case in CASES["cases"]}
    for case in (*CASES_REAL["cases"], *(c["real"] for c in CASES["cases"] if c.get("real"))):
        assert case["intent"] in INTENTS, case
        assert set(case.get("slots") or {}) <= {"dish", "topic", "grape", "food", "want",
                                                "unknown_dish", "fact"}, case  # fmt: skip


@needs_real
def test_routing_accuracy_on_real_data(real_data: Any) -> None:
    """Приёмка на полной сборке: ≥ 90 %, отказы, беседа и блюда — без промахов.

    Полный справочник не должен уводить маршрут: слово «вино» — к общей теме, «салат» или
    «торт» — к чужому блюду. Слоты сверяются с настоящими блюдами и темами.
    """
    router = Router(real_data)
    cases = real_cases()
    misses = []
    for case in cases:
        slots = case.get("slots") or {}
        assert slots.get("dish") in (None, *real_data.dishes), case
        assert slots.get("topic") in (None, *real_data.topics), case
        route = route_case(router, real_data, case)
        if not matches(route, case):
            misses.append((case, route))
    text = "\n".join(f"{c['question']!r}: ждали {c['intent']}, получили {r}" for c, r in misses)
    accuracy = 1 - len(misses) / len(cases)
    assert accuracy >= MIN_ACCURACY, f"точность {accuracy:.1%}; промахи:\n{text}"
    safety = [c for c, _ in misses if c["intent"] in ("refuse", "smalltalk", "dish_check")]
    assert not safety, text


@needs_real
def test_refusal_sets_through_the_router_on_real_data(real_data: Any) -> None:
    """Наборы отказов барьера — сквозь весь маршрут на полной сборке: блюдо ищется раньше
    барьера, и 82 блюда с алиасами не должны открыть вопрос о цене или возрасте."""
    import test_somm_barrier as sets

    router = Router(real_data)
    phrases = [(topic, q) for topic, qs in {**sets.REFUSE, **sets.OTHER_TOPICS}.items() for q in qs]
    phrases += [(None, q) for q in sets.ADVERSARIAL]
    phrases += [(None, p + t) for p in sets.DANGEROUS_PREFIXES for t in sets.ANSWERABLE_TAILS]
    wrong = []
    for topic, question in phrases:
        route = router.text(question, Context(), grapes=CASES["grapes"])
        refused = route.intent == "refuse" and route.refusal is not None
        if not refused or (topic is not None and route.refusal.topic != topic):
            wrong.append((question, route.intent))
    assert not wrong
    legal = [q for q in (*sets.LEGAL, *sets.LEGAL_NEAR_NEW_RULES)
             if router.text(q, Context(), grapes=CASES["grapes"]).intent == "refuse"]  # fmt: skip
    assert not legal


def test_broad_topic_answers_only_the_broad_question(data: Any) -> None:
    """Общая тема («Что такое вино») не забирает вопрос о частном, узкая тема сильнее этикетки."""
    topic = type(next(iter(data.topics.values())))

    def make(topic_id: str, triggers: tuple[str, ...], context: tuple[str, ...] = ()) -> Any:
        return topic(topic_id, topic_id, triggers, context, "Ответ.", "", ())

    topics = {
        "what_is_wine": make("what_is_wine", ("вино", "вина"), ("такое", "значит", "вообще")),
        "orange_wine": make("orange_wine", ("оранжев",)),
        "decanting": make("decanting", ("декант", "графин")),
        "zgu_znmp": make("zgu_znmp", ("згу",)),
        "read_label": make("read_label", ("этикетк",), ("чита", "значит")),
    }
    router = Router(replace(data, topics=topics))
    asked = {
        "что такое вино?": "what_is_wine",
        "что такое оранжевое вино?": "orange_wine",
        "а графин для вина вообще для чего нужен": "decanting",
        "что значит ЗГУ на этикетке?": "zgu_znmp",
        "что значит надпись на этикетке?": "read_label",
    }
    for question, topic_id in asked.items():
        assert router.text(question, Context()).topic == topic_id, question


# ------------------------------------------------------------------ порядок разбора
def test_sticky_refusal_answers_every_question_and_chip(router: Router) -> None:
    context = Context(intent="refuse", refusal_topic="medical")
    for route in (
        router.text("а к борщу?", context),
        router.text("привет", context),
        router.chip("what_to_eat", {}, context),
        router.chip("guided", {"food": "meat"}, context),
    ):
        assert route.intent == "refuse" and route.refusal is not None
        assert route.refusal.topic == "medical"


def test_smalltalk_is_whole_phrase_only(router: Router) -> None:
    assert router.text("пока", Context()).intent == "smalltalk"
    assert router.text("пока вино дышит, что приготовить?", Context()).intent == "what_to_eat"
    assert smalltalk.detect("Спасибо!") == "thanks"
    assert smalltalk.detect("спасибо, а к борщу?") is None


def test_negated_dish_is_not_a_dish(router: Router) -> None:
    route = router.text("нет у меня борща, что ещё к нему подать?", Context())
    assert route.intent == "what_to_eat" and route.dish is None
    assert router.text("борща не будет", Context()).dish is None
    assert router.text("не борщ, а селёдка", Context()).dish == "solenaya_seld"


def test_longest_dish_name_wins(router: Router) -> None:
    assert router.text("а к борщу с пампушками?", Context()).dish == "borsch_s_pampushkami"
    assert router.text("а к борщу?", Context()).dish == "borsch"


def test_grape_word_is_not_a_fruit_plate(router: Router) -> None:
    """«Виноград» — алиас «Фруктов и ягод», но в вопросе о вине это само вино."""
    route = router.text("из какого винограда это вино?", Context(), grapes=["saperavi"])
    assert route.dish is None and route.intent != "dish_check"


def test_ellipsis_takes_dish_from_context(router: Router) -> None:
    context = Context(intent="dish_check", dish="borsch")
    assert router.text("а помягче?", context) == Route(
        "softer", dish="borsch", want="softer", source="text"
    )
    assert router.chip("fresher", {}, context).dish == "borsch"
    assert router.chip("fresher", {"dish": "oysters"}, context).dish == "oysters"


def test_guided_steps_from_text(router: Router) -> None:
    step_food = Context(intent="guided", step="food")
    assert router.text("к устрицам", step_food).food == "fish"
    assert router.text("без блюда", step_food).food == "none"
    step_want = Context(intent="guided", step="want", food="cheese")
    route = router.text("посвежее", step_want)
    assert (route.intent, route.food, route.want) == ("guided", "cheese", "fresher")


def test_chips_route_directly(router: Router) -> None:
    for chip_id in CHIP_ARGS:
        args = {"dish": "borsch"} if chip_id == "dish_check" else {}
        if chip_id == "grape":
            args = {"grape": "saperavi"}
        if chip_id == "term":
            args = {"topic": "tannins"}
        route = router.chip(chip_id, args, Context())
        assert route.intent == chip_id and route.source == "chip"
    route = router.chip("guided", {"want": "softer"}, Context())
    assert (route.food, route.want) == (None, None), "без блюда — первый шаг, направление ждёт"


def test_unknown_dish_and_unknown(router: Router) -> None:
    assert router.text("а к хинкали?", Context()) == Route(
        "unknown", unknown_dish=True, source="text"
    )
    assert router.text("к рислингу что лучше всего идёт", Context()).unknown_dish is False
    assert router.text("напиши стихотворение", Context()).intent == "unknown"


def dish(name: str, *aliases: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, aliases=list(aliases))


#: Блюда `dishes.json` с уточнением в названии и без — как в полной сборке «Лозы».
VARIANT_DISHES = {
    "pasta_tomat": dish(
        "Паста с томатным соусом", "паста", "макароны", "макароны с сыром", "паста с овощами"
    ),
    "vareniki": dish("Вареники с картошкой", "вареники", "вареники с картофелем"),
    "utka_s_yablokami": dish("Запечённая утка с яблоками", "утка", "утке", "утка с яблоками"),
    "griby_v_smetane": dish("Грибы в сметане", "грибы", "грибам", "белые грибы"),
    "solenaya_seld": dish(
        "Солёная сельдь", "сельдь", "селёдка", "сельдь с луком", "селёдка с картошкой"
    ),
    "steik_ribay": dish("Стейк рибай", "стейк"),
    "kartoshka_zapechennaya": dish(
        "Запечённая картошка",
        "картошка",
        "картошкой",
        "пюре",
        "печёная картошка",
        "картошка в духовке",
    ),
    "buzhenina": dish("Буженина", "буженина"),
    "grecheskiy_salat": dish("Греческий салат", "салат", "салат с фетой", "овощной салат"),
    "ovoshchi_gril": dish("Овощи на гриле", "овощи", "запечённые овощи"),
    "kurinaya_grudka": dish("Куриная грудка", "курица", "запечённая курица", "отварная курица"),
    "tushenaya_kapusta": dish("Тушёная капуста", "капуста", "тушёная капуста с мясом"),
    "zharenaya_kartoshka_s_gribami": dish("Жареная картошка с грибами", "картошка с грибами"),
    "kurnik": dish("Курник", "пирог с курицей"),
    "kotlety_farsh": dish("Котлеты из фарша", "котлеты", "котлета", "котлеты с пюре"),
}


@pytest.mark.parametrize(
    ("question", "dish_id", "variant"),
    [
        # Проверка на видеокарте 25.09: «к пасте с грибами» отвечали про томатный соус.
        ("к пасте с грибами подойдёт?", None, True),
        ("паста с белыми грибами", None, True),
        ("к пасте с креветками", None, True),
        ("вареники с вишней", None, True),
        ("утка с апельсинами", None, True),
        # Главное слово уточнения после чужого определения — другой соус, а не томатный.
        ("к пасте с грибным соусом", None, True),
        ("к пасте со сливочным соусом", None, True),
        ("нет у меня пасты с грибами", None, False),
        # Уточнение из правил, то же слово в другой форме или вовсе без уточнения — то же блюдо.
        ("а к пасте?", "pasta_tomat", False),
        ("к пасте с томатным соусом", "pasta_tomat", False),
        ("к пасте с томатами", "pasta_tomat", False),
        ("а к пасте с соусом?", "pasta_tomat", False),
        ("макароны с сыром", "pasta_tomat", False),
        ("макароны с тёртым сыром", "pasta_tomat", False),
        ("к пасте с друзьями", "pasta_tomat", False),
        ("к утке с яблоками подойдёт?", "utka_s_yablokami", False),
        ("вареники с картофелем и грибами", "vareniki", False),
        ("вареники с жареной картошкой", "vareniki", False),
        # У блюда без уточнения в названии «с …» — гарнир или компания: блюдо то же.
        ("селёдка с горчицей", "solenaya_seld", False),
        ("стейк с картошкой", "steik_ribay", False),
        ("что скажешь про буженину с этим вином", "buzhenina", False),
        ("грибы с луком", "griby_v_smetane", False),
        # Проверка 25.09, вечер: слово правил — начинка или гарнир чужого блюда.
        ("рыба с овощами", None, True),
        ("к рыбе на гриле с овощами", None, True),
        ("к мясу с грибами", None, True),
        ("к мясу с белыми грибами", None, True),
        ("к пирогу с грибами", None, True),
        ("к рису с овощами", None, True),
        ("к мясу с картошкой", None, True),
        ("а с грибами?", "griby_v_smetane", False),
        ("это вино с грибами пойдёт?", "griby_v_smetane", False),
        ("к рыбе, а с грибами?", "griby_v_smetane", False),
        ("к пирогу с курицей", "kurnik", False),
        ("курица с картошкой", "kurinaya_grudka", False),
        # Алиас правил задом наперёд — то же блюдо: «котлеты с пюре», «селёдка с картошкой».
        ("пюре с котлетой подойдёт?", "kotlety_farsh", False),
        ("картошка с селёдкой", "solenaya_seld", False),
        ("к пюре с грибами", None, True),
        # Родовое слово блюда правил с чужим составом — другое блюдо; алиас с «с» — то же.
        ("к салату с курицей", None, True),
        ("к картошке с мясом", None, True),
        ("овощи с сыром", None, True),
        ("а к салату?", "grecheskiy_salat", False),
        ("к салату с фетой", "grecheskiy_salat", False),
        ("к салату с этим вином", "grecheskiy_salat", False),
        ("к капусте с мясом", "tushenaya_kapusta", False),
        ("картошка с грибами", "zharenaya_kartoshka_s_gribami", False),
        # Способ готовки в названии блюда правил и другой — в вопросе.
        ("к жареной картошке с луком", None, True),
        ("к жареной картошке", None, True),
        ("а к картошке варёной?", None, True),
        ("к картошке фри", None, True),
        ("к печёной картошке", "kartoshka_zapechennaya", False),
        ("к картошке в духовке", "kartoshka_zapechennaya", False),
        ("к жареной картошке с грибами", "zharenaya_kartoshka_s_gribami", False),
        # У «Куриной грудки» способа в названии нет — жареная курица остаётся курицей.
        ("к жареной курице", "kurinaya_grudka", False),
    ],
)
def test_dish_with_another_qualifier_is_not_the_rules_dish(
    question: str, dish_id: str | None, variant: bool
) -> None:
    """Блюдо правил с уточнением в названии и другое уточнение в вопросе — другое блюдо; так же
    начинка чужого блюда, родовое слово с чужим составом и другой способ готовки (проверка 25.09,
    вечер): «Такого блюда в правилах сочетаний нет», а не ответ о другом блюде."""
    found = DishIndex(VARIANT_DISHES).find(question)
    assert (found.dish, found.variant) == (dish_id, variant)
    route = Router(SimpleNamespace(dishes=VARIANT_DISHES, topics={}, grapes={})).text(
        question, Context()
    )
    if variant:
        assert route == Route("unknown", unknown_dish=True, source="text")
    elif dish_id is not None:
        assert (route.intent, route.dish) == ("dish_check", dish_id)


# ------------------------------------------------------------------ контекст
def test_context_is_sanitized(data: Any) -> None:
    context = context_of(
        {
            "intent": "dish_check",
            "dish": "нет-такого",
            "food": "meat",
            "want": "sweeter",
            "step": "want",
            "refusal_topic": "price",
            "shown": ["a", "b", "a"],
            "extra": {"ignored": True},
        },
        data.dishes,
    )
    assert context == Context(intent="dish_check", food="meat", step="want", shown=("a", "b"))
    assert context_of(None, data.dishes) == Context()
    assert context_of({"refusal_topic": "minors"}, data.dishes).refusal_topic == "minors"


@pytest.mark.parametrize(
    "raw",
    [
        {"shown": ["s"] * 13},
        {"shown": "slug"},
        {"shown": [1, 2]},
        {"dish": "x" * 201},
        {"intent": 5},
        {"shown": ["x" * 201]},
    ],
)
def test_bad_context_is_an_error(raw: dict[str, Any], data: Any) -> None:
    with pytest.raises(ContextError):
        context_of(raw, data.dishes)


# ------------------------------------------------------------------ текст
def test_clean_question() -> None:
    assert clean_question("  а\tк\u200b  борщу?\n ") == "а к борщу?"
    assert clean_question("\x00\x07") == ""
    assert clean_question("Саперави") == "Саперави"


def test_word_forms_and_plural() -> None:
    assert words_match("борщу", "борщ") and words_match("гуся", "гусь")
    assert not words_match("паста", "пастила")
    assert normalize("Ёжик, «Брют»!") == "ежик брют"
    assert [plural(n, ("вино", "вина", "вин")) for n in (1, 3, 5, 11, 21)] == [
        "вино",
        "вина",
        "вин",
        "вин",
        "вино",
    ]
    assert count_wines(3) == "три вина" and count_wines(1) == "одно вино"


# ------------------------------------------------------------------ шаблоны
def test_style_and_plain_texts() -> None:
    assert t.style_words("Белое брют", True) == "игристое белое брют"
    assert t.style_words("Белое игристое", True) == "белое игристое"
    assert t.style_words("Красное сухое", False) == "красное сухое"
    assert t.plural_style("Красное сухое") == "красные сухие"
    assert t.plural_style("Белое полусладкое") == "белые полусладкие"
    assert (
        t.plain_text("Розовое брют")
        == "Обычная сортировка: розовые брют других виноделен по названию."
    )


def test_note_template_shapes() -> None:
    text = t.note_text(
        name="Вино",
        style="белое",
        grapes=["А", "Б", "В", "Г"],
        winery="Винодельня",
        region="",
        temperature=(6, 8),
        dishes=["Сырная тарелка"],
    )
    assert text == (
        "Вино — белое. Сорта — А, Б, В и другие. Винодельня. Подают при 6–8 °C. "
        "По правилам сочетаний к нему подходит сырная тарелка."
    )
    bare = t.note_text(
        name="Вино", style="", grapes=[], winery="", region="", temperature=None, dishes=[]
    )
    assert bare == "Вино."


def template_strings() -> list[str]:
    texts = [
        t.UNKNOWN,
        t.UNKNOWN_DISH,
        t.UNKNOWN_GRAPE,
        t.NO_PAIRS,
        t.NEUTRAL,
        t.NO_SERVE,
        t.NO_SIMILAR,
        t.GUIDED_FOOD_QUESTION,
        t.GUIDED_WANT_QUESTION,
        *t.ERROR_TEXTS.values(),
        *smalltalk.TEXTS.values(),
        *t.CHIP_TEXTS.values(),
        *t.TERM_CHIP_TEXTS.values(),
        *(text for _, text in t.GUIDED_FOODS),
        *(text for _, text in t.GUIDED_WANTS),
        *(t.caveat_tail(want, "к борщу") for want in t.CAVEAT_WORDS),
        *(t.caveat_title(want, "к борщу") for want in t.CAVEAT_WORDS),
        *(t.no_tail(want, "к устрицам") for want in t.CAVEAT_WORDS),
        t.matching_tail("к борщу"),
        t.dish_verdict("caveat", "к борщу", "Правило.", t.caveat_tail("softer", "к борщу")),
        t.dish_verdict("no", "к устрицам", "Правило.", t.matching_tail("к устрицам")),
        t.dish_verdict("no", "к устрицам", "Правило.", t.no_tail("softer", "к устрицам")),
        t.dish_verdict("yes", "к гусю", "Правило.", None),
        t.direction_text("softer", "к борщу", 3, expanded=False),
        t.direction_text("fresher", None, 1, expanded=True),
        t.direction_empty("softer", None, no_difference=True),
        t.direction_empty("fresher", "к рыбе", no_difference=False),
        t.replace_text("красное сухое", "Саперави"),
        t.guided_food_verdict("meat", "caveat"),
        t.guided_food_verdict("dessert", None),
        t.guided_text("meat", "softer", 3),
        t.guided_text(None, None, 2),
        t.guided_empty("fish", "fresher"),
        t.guided_title("cheese", None),
        t.plain_text("Красное сухое"),
        t.serve_text("Вино", (16, 18), "Плотные красные подают при 16–18 °C."),
        t.what_to_eat(["Борщ", "Гусь запечённый"], "Правило."),
        *(t.stage(stage_id, "Cru Lermont Saperavi")["text"] for stage_id in t.STAGES),
        *(done for _, done in t.STAGES.values()),
    ]
    return texts


@pytest.mark.parametrize("text", template_strings())
def test_templates_are_legal(text: str) -> None:
    """Каждая строка шаблонов проходит `content_filter` и стоп-слова, `%` — только «% об.»."""
    assert check(text).clean, text
    lowered = text.lower()
    for stop in ("купи", "цена", "₽", "лучш", "идеальн", "вино недели", "попробуйте", "рубл"):
        assert stop not in lowered, text
    assert "%" not in text.replace("% об.", "")


# ------------------------------------------------------------------ вопрос экрана check (§5)
@pytest.mark.parametrize(
    "case",
    json.loads((FIXTURES / "scan_check_question.json").read_text(encoding="utf-8"))["cases"],
    ids=lambda case: case["name"],
)
def test_check_question_matches_contract(case: dict[str, Any]) -> None:
    assert t.check_question(case["read"], case["candidates"]) == case["question"]


def test_check_question_other_fields() -> None:
    def cand(slug: str, **facts: Any) -> dict[str, Any]:
        base = {"sugar": "suhoe", "color": "Красное", "sparkling": False, "grapes": [], "abv": 13}
        return {"slug": slug, "name": slug.upper(), "facts": {**base, **facts}}

    empty = {"sugar": None, "color": None, "sparkling": None, "abv": None, "grapes": []}
    question = t.check_question(empty, [cand("a", abv=12), cand("b", abv=13.5)])
    assert question == {
        "field": "abv",
        "text": "Какая крепость на этикетке: 12 или 13,5 % об.?",
        "options": [{"label": "12 % об.", "slug": "a"}, {"label": "13,5 % об.", "slug": "b"}],
    }
    question = t.check_question(
        empty, [cand("a", grapes=["Мерло"]), cand("b", grapes=["Каберне Совиньон"]), cand("c")]
    )
    assert question is not None and question["field"] == "grapes"
    assert question["text"] == "Какой сорт на этикетке: Мерло или Каберне Совиньон?"
    question = t.check_question(empty, [cand("a", sparkling=True), cand("b")])
    assert question is not None and question["text"] == "Вино игристое или тихое?"
    assert [o["label"] for o in question["options"]] == ["игристое", "тихое"]
    for q in (question,):
        assert check(q["text"]).clean
