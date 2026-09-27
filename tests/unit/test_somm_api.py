"""Маршруты «Сомелье» через HTTP (`app/api/somm.py`) против договора `docs/api-sommelier.md`.

Справочник — семь реальных позиций выгрузки организатора из заглушек договора (Cru Lermont
Saperavi, три Пино Нуар, Мускатель белый, Alveus Ultra Cuvee Brut, Cru Lermont Рислинг) с
фактами без портала; данные сомелье — заглушки `tests/fixtures/somm`. Проверяется:

* `GET …/sommelier` совпадает с заглушками договора целиком (заметка, профиль, подача, блюда,
  чипы) — это и есть проверка, что шаблон собран из фактов, а не написан руками;
* поток `POST /v1/sommelier/ask`: грамматика, этапы, метки, причины голоса, `facts` ≤ 150 мс,
  совпадение с потоками-заглушками, обычная сортировка без голоса, выключатели, 404 и 422;
* вопрос гостя не попадает ни в журнал, ни в пакет для модели;
* на настоящих данных (`SVS_DATA_DIR`) — GET ≤ 20 мс и `facts` ≤ 150 мс по p95.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.somm import SommSettings, ask_events, register_somm_routes
from app.recommend.catalog import RecoCatalog, card_sugar, sweet_name, wine_of
from app.recommend.content_filter import check
from app.recommend.facts import tile
from app.recommend.shelf import Shelf
from app.recommend.somm_data import load_somm_data
from app.sommelier import answers
from app.sommelier import templates as t
from app.sommelier.answers import FactsPackage, Sommelier
from app.sommelier.entity_lock import EntityLock
from app.sommelier.gate import SommGate
from app.sommelier.guards import stoplist_hits
from app.sommelier.ollama_text import ChatResult
from app.sommelier.router import Context, Route, Router
from app.sommelier.templates import check_question
from app.sommelier.voice import Voice

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "somm"
ASK = "/v1/sommelier/ask"
LABEL_AI = "Текст — ИИ, подбор — алгоритм"
LABEL_ALGO = "Текст и подбор — алгоритм"
NOTICE = "Применяются рекомендательные технологии"
RED = "fanagoriya-cru-lermont-saperavi-saperavi-krasnoe-suhoe-135"
SWEET = "massandra-muskatel-belyy-belye-sorta-vinograda-beloe-sladkoe-16"
SPARKLING = "fanagoriya-alveus-ultra-cuvee-brut-shardone-beloe-bryut-12"
NOSUGAR = "cru-lermont-risling"
#: Слово-метка вопроса гостя: его не должно быть ни в журнале, ни в пакете модели.
MARKER = "фиолетовыйзебрик"
DESCRIPTION = "описаниевыгрузкиметка"


# ------------------------------------------------------------------ справочник на заглушках
def wine(
    slug: str,
    winery: str,
    title: str,
    grapes: list[str],
    labels: list[str],
    color: str,
    sugar: str | None,
    abv: float | None,
    region: str,
    *,
    sparkling: bool = False,
) -> dict[str, Any]:
    return {
        "row": {
            "slug": slug,
            "wine_id": f"W-{slug}",
            "canonical": slug,
            "is_canonical": True,
            "in_csv": True,
            "winery": winery,
            "winery_norm": winery.casefold(),
            "title": title,
            "grapes": grapes,
            "grapes_src": "csv",
            "color": color,
            "sugar": sugar,
            "sparkling": sparkling,
            "abv": abv,
            "region": region,
            "photo": f"{slug}.webp",
        },
        "labels": labels,
    }


WINES = [
    wine(RED, "Фанагория", "Cru Lermont Saperavi", ["saperavi"], ["Саперави"], "Красное",
         "suhoe", 13.5, "Кубань"),
    wine("a-gordienko-m-nikolaev-pino-nuar-krasnoe-suhoe-135", "А. Гордиенко & М. Николаев",
         "Пино Нуар", ["pinot_noir"], ["Пино Нуар"], "Красное", "suhoe", 13.5, "Кубань"),
    wine("alma-valley-pino-nuar-krasnoe-suhoe-13", "Alma Valley", "Пино Нуар", ["pinot_noir"],
         ["Пино Нуар"], "Красное", "suhoe", 13.0, "Крым"),
    wine("belbek-pino-nuar-krasnoe-suhoe-133", "Бельбек", "Пино Нуар", ["pinot_noir"],
         ["Пино Нуар"], "Красное", "suhoe", 13.3, "Крым"),
    wine(SWEET, "Массандра", "Мускатель белый", [], ["Белые сорта винограда"], "Белое",
         "sladkoe", 16.0, "Крым"),
    wine(SPARKLING, "Фанагория", "Alveus Ultra Cuvee Brut", ["riesling", "chardonnay"],
         ["Рислинг", "Шардоне"], "Белое", "brut", 12.0, "Кубань", sparkling=True),
    wine(NOSUGAR, "Фанагория", "Cru Lermont Рислинг", ["riesling"], ["Рислинг Рейнский"],
         "Белое", None, None, "Кубань"),
]  # fmt: skip
SUGAR_WORDS = {"suhoe": "сухое", "sladkoe": "сладкое", "brut": "брют"}


def card_of(item: dict[str, Any]) -> dict[str, Any]:
    """Карточка без портала (договор «после поиска», §3) — поля, которые читает сомелье."""
    row = item["row"]
    sugar = SUGAR_WORDS.get(row["sugar"] or "", "")
    return {
        "slug": row["slug"],
        "name": row["title"],
        "winery": row["winery"],
        "region": row["region"],
        "grapes": item["labels"],
        "color_label": row["color"],
        "sugar_class": row["sugar"],
        "sparkling": row["sparkling"],
        "style_label": f"{row['color']} {sugar}".strip(),
        "alcohol": row["abv"],
        "description": f"Описание {DESCRIPTION}",
        "portal_url": f"https://vino-svoe.ru/wines/{row['slug']}",
    }


class FakeAfter:
    """Слой «после поиска» на семи винах: справочник, «Сомелье у полки», карточка, плитка.

    Справочник — строки фактов в форме `organizer_row` (выгрузка организатора, без портала).
    """

    def __init__(self, extra: Sequence[dict[str, Any]] = ()) -> None:
        items = [*WINES, *extra]
        self.catalog = RecoCatalog([wine_of(item["row"]) for item in items])
        self.sommelier = Shelf(self.catalog)
        self.cards = {item["row"]["slug"]: card_of(item) for item in items}

    def card(self, slug: str) -> dict[str, Any] | None:
        return self.cards.get(slug)

    def tile(self, wine: Any, reasons: Any = ()) -> dict[str, Any]:
        return tile(wine, reasons, photo_url=f"/v1/wines/{wine.slug}/photo")


@pytest.fixture(scope="module")
def after() -> FakeAfter:
    return FakeAfter()


@pytest.fixture(scope="module")
def data() -> Any:
    return load_somm_data(None, fallback=FIXTURES, strict=True)


# ------------------------------------------------------------------ голос-подделка
@dataclass(frozen=True)
class Spoken:
    text: str
    generated: bool
    label: str
    reason: str | None
    guard: str | None


class FakeVoice:
    """Голос по договору §6.3 без модели: причина, живой текст, брак проверки, задержка."""

    def __init__(
        self,
        *,
        reason: str | None = None,
        text: str | None = None,
        guard: str | None = None,
        live: bool = True,
        delay: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.reason, self.text, self.guard = reason, text, guard
        self.live, self.delay, self.fail = live, delay, fail
        self.packages: list[Any] = []
        self.cancelled = False

    def template(self, facts: Any, reason: str, guard: str | None = None) -> Spoken:
        return Spoken(facts.verdict_template, False, LABEL_ALGO, reason, guard)

    async def speak(self, facts: Any, *, on_stage=None, disconnected=None) -> Spoken:
        self.packages.append(facts)
        if self.fail:
            raise RuntimeError(f"голос упал {MARKER}")
        if not facts.voiced:
            order = facts.public["order"]
            return self.template(facts, "plain" if order == "plain" else "not_voiced")
        if not self.live:
            return self.template(facts, "off")
        if self.reason in ("busy", "quiet", "locked", "not_ready"):
            return self.template(facts, self.reason)
        await on_stage("voice")
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        await on_stage("verify")
        if self.reason == "guard":
            return self.template(facts, "guard", self.guard)
        return Spoken(self.text or "Живой текст.", True, LABEL_AI, None, None)


def settings(**overrides: Any) -> Any:
    values = {"live": True, "input": True, "data_dir": FIXTURES, "timeout_ms": 4000}
    values.update(overrides)
    values.setdefault("quiet_s", 15.0)
    return SommSettings(**values)


def make_client(after: FakeAfter, data: Any, *, voice: Any = None, **overrides: Any) -> TestClient:
    app = FastAPI()
    app.state.service = SimpleNamespace(after=after)
    register_somm_routes(
        app, settings=settings(**overrides), data=data, voice=voice or FakeVoice(), preload=False
    )
    return TestClient(app)


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fixture_stream(name: str) -> list[dict[str, Any]]:
    text = (FIXTURES / f"{name}.ndjson").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def ask(client: TestClient, body: dict[str, Any], order: str = "reco") -> list[dict[str, Any]]:
    response = client.post(ASK, params={"order": order}, json=body)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/x-ndjson; charset=utf-8"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.endswith("\n") and "\n\n" not in response.text
    return [json.loads(line) for line in response.text.splitlines()]


def kind(stream: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(event for event in stream if event["type"] == name)


def grammar(stream: list[dict[str, Any]]) -> str:
    return "".join(event["type"][0] for event in stream)


def strip_times(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_times(v) for k, v in obj.items() if k != "t_ms"}
    if isinstance(obj, list):
        return [strip_times(v) for v in obj]
    return obj


def chip_keys(chips: list[dict[str, Any]]) -> list[tuple[str, tuple]]:
    return [(chip["id"], tuple(sorted((chip.get("args") or {}).items()))) for chip in chips]


def strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in strings(v)]
    return []


# ------------------------------------------------------------------ GET …/sommelier
@pytest.mark.parametrize(
    "name", ["sommelier_red", "sommelier_sweet", "sommelier_sparkling", "sommelier_nosugar"]
)
def test_get_matches_contract_fixture(name: str, after: FakeAfter, data: Any) -> None:
    """Тело GET совпадает с заглушкой договора целиком: шаблон — из фактов, а не руками."""
    expected = load(f"{name}.json")
    client = make_client(after, data)
    response = client.get(f"/v1/wines/{expected['slug']}/sommelier")
    assert response.status_code == 200
    assert response.json() == expected


def test_get_plain_and_flags(after: FakeAfter, data: Any) -> None:
    client = make_client(after, data, live=False, input=False)
    body = client.get(f"/v1/wines/{RED}/sommelier", params={"order": "plain"}).json()
    assert body["order"] == "plain" and body["notice_149"] is None
    assert (body["live"], body["input"]) == (False, False)
    names = [dish["name"] for dish in body["dishes"]]
    assert names == sorted(names, key=str.casefold)
    assert body["note"]["label"] == LABEL_ALGO and body["note"]["generated"] is False


def test_get_errors(after: FakeAfter, data: Any) -> None:
    client = make_client(after, data)
    missing = client.get("/v1/wines/net-takogo/sommelier")
    assert missing.status_code == 404 and missing.json() == {"detail": "нет карточки 'net-takogo'"}
    assert client.get(f"/v1/wines/{RED}/sommelier", params={"order": "x"}).status_code == 422
    app = FastAPI()
    register_somm_routes(app, settings=settings(), data=data, voice=FakeVoice(), preload=False)
    assert TestClient(app).get(f"/v1/wines/{RED}/sommelier").status_code == 503


def test_get_is_legal(after: FakeAfter, data: Any) -> None:
    client = make_client(after, data)
    for slug in (item["row"]["slug"] for item in WINES):
        body = client.get(f"/v1/wines/{slug}/sommelier").json()
        for text in strings(body):
            assert check(text).clean, text
            assert "%" not in text.replace("% об.", ""), text
        assert DESCRIPTION not in json.dumps(body, ensure_ascii=False)


# ------------------------------------------------------------------ POST ask: заглушки договора
@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("ask_dish_check_live", {"question": "а к борщу подойдёт?"}),
        ("ask_dish_check_guard", {"chip": "dish_check", "args": {"dish": "borsch"}}),
    ],
)
def test_dish_check_with_caveat_is_a_template(
    name: str, body: dict[str, Any], after: FakeAfter, data: Any
) -> None:
    """«А к борщу?» — оговорка и три Пино Нуар помягче: ответ заглушки договора, но без голоса.

    Решение 25.09 (договор, §6.5): `dish_check` с вердиктом не «да» — это всегда пакет с винами
    подборки, и модель в нём путает вино карточки с подборкой. Заглушки с живым и забракованным
    текстом остались образцами страницы; сервис на тот же вопрос отдаёт те же `facts` с
    `voice: false`, без этапов `voice` и `verify`, а `text` — шаблон с `reason: "not_voiced"`,
    даже когда голос наготове.
    """
    expected = fixture_stream(name)
    voice = FakeVoice(text=kind(expected, "text")["text"])
    stream = ask(make_client(after, data, voice=voice), {"slug": RED, **body})
    assert [e.get("id") or e["type"] for e in stream] == [
        e.get("id") or e["type"] for e in expected if e.get("id") not in ("voice", "verify")
    ]
    facts, want = strip_times(kind(stream, "facts")), strip_times(kind(expected, "facts"))
    for key in ("intent", "order", "verdict_template", "detail", "basis", "dishes",
                "wines_title", "serve", "question", "refusal", "notice_149"):  # fmt: skip
        assert facts[key] == want[key], key
    assert facts["voice"] is False and want["voice"] is True
    assert chip_keys(facts["chips"]) == chip_keys(want["chips"])
    assert {w["slug"] for w in facts["wines"]} == {w["slug"] for w in want["wines"]}
    by_slug = {w["slug"]: w for w in want["wines"]}
    for tile_body in facts["wines"]:
        assert tile_body == by_slug[tile_body["slug"]]
    assert facts["compare"]["axes"] == ["tannin"]
    assert facts["compare"]["anchor"] == want["compare"]["anchor"]
    assert strip_times(kind(stream, "text")) == {
        "type": "text",
        "text": want["verdict_template"],
        "generated": False,
        "label": LABEL_ALGO,
        "reason": "not_voiced",
        "guard": None,
    }
    assert [package.voiced for package in voice.packages] == [False]


class EchoOllama:
    """Клиент голоса без сети и видеокарты: записывает каждый вызов модели и отвечает готовым
    ответом пакета — шаблоном, который проверки голоса пропускают."""

    model = "qwen3.5:4b"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self) -> bool:
        return True

    def chat(self, messages, *, timeout_s, cancel, on_abort=None) -> ChatResult:
        user = messages[-1]["content"]
        self.calls.append(user)
        answer = user.split("Готовый ответ: ", 1)[1].rsplit("\n/no_think", 1)[0]
        return ChatResult("ok", answer, 0, "stop", 24, 300, 5)


def live_voice(data: Any) -> tuple[Voice, EchoOllama]:
    """Настоящий голос: ворота связаны с замком видеокарты и пускают, клиент — `EchoOllama`."""
    gate = SommGate(quiet_s=0.0)
    gate.bind(threading.Lock())
    client = EchoOllama()
    return Voice(gate, client, EntityLock(data.vocab), live=True), client  # type: ignore[arg-type]


def test_dish_check_is_voiced_only_for_yes_without_wines(after: FakeAfter, data: Any) -> None:
    """Голос у блюд — только «да» без вин подборки (решение 25.09, договор §6.5).

    Сквозь маршрут и настоящий `Voice`: при оговорке, «скорее нет» и «правила молчат» модель не
    зовётся ни разу — `facts.voice: false`, этапов голоса нет, `text` — шаблон `not_voiced`. «Да»
    и «К чему подать» идут в модель, как раньше.
    """
    voice, client = live_voice(data)
    http = make_client(after, data, voice=voice)
    cases = [
        (SPARKLING, {"chip": "dish_check", "args": {"dish": "borsch"}}, "yes", True),
        (SPARKLING, {"chip": "dish_check", "args": {"dish": "oysters"}}, "yes", True),
        (RED, {"chip": "what_to_eat"}, None, True),
        (RED, {"chip": "dish_check", "args": {"dish": "borsch"}}, "caveat", False),
        (RED, {"question": "а к устрицам подойдёт?"}, "no", False),
        (SPARKLING, {"chip": "dish_check", "args": {"dish": "medovik"}}, "no", False),
        (SWEET, {"chip": "dish_check", "args": {"dish": "oysters"}}, "no", False),
        (NOSUGAR, {"chip": "dish_check", "args": {"dish": "napoleon"}}, "neutral", False),
    ]
    for slug, body, verdict, voiced in cases:
        before = len(client.calls)
        stream = ask(http, {"slug": slug, **body})
        check_stream(stream)
        facts, text = kind(stream, "facts"), kind(stream, "text")
        stages = [event["id"] for event in stream if event["type"] == "stage"]
        if verdict is not None:
            assert [dish["verdict"] for dish in facts["dishes"]] == [verdict], (slug, body)
        assert facts["voice"] is voiced, (slug, body)
        assert len(client.calls) - before == int(voiced), (slug, body)
        if voiced:
            assert not facts["wines"]
            assert stages[-2:] == ["voice", "verify"]
            assert (text["generated"], text["label"], text["reason"]) == (True, LABEL_AI, None)
        else:
            assert "voice" not in stages and "verify" not in stages
            assert (text["generated"], text["label"], text["reason"]) == (
                False,
                LABEL_ALGO,
                "not_voiced",
            )
            assert text["text"] == facts["verdict_template"]
    # Оговорка и «скорее нет» к борщу и устрицам — с подборкой: её голос и путал с якорем.
    assert voice.stats()["reasons"] == {"not_voiced": 5}


@pytest.mark.parametrize(
    ("slug", "dish", "verdict", "want", "head", "title", "axis", "pill"),
    [
        # «Сладость спорит с рыбой» — «скорее нет» (третий круг проверки 25.09), подборка суше:
        # брют и сухие, и хвост не «если хочется», а «вот вина других виноделен суше».
        (SWEET, "solenaya_seld", "no", "drier",
         "Вот вина других виноделен суше, которые подходят к солёной сельди.",
         "Суше к солёной сельди", "sweetness", "а не сладкое"),
        # «Сладость перебивает блюдо» — оговорка, суше.
        (SWEET, "kholodets", "caveat", "drier", "Если хочется суше —", "Суше к холодцу",
         "sweetness", "а не сладкое"),
        # «Блюдо насыщеннее вина» — помощнее: крепость выше.
        ("alma-valley-pino-nuar-krasnoe-suhoe-13", "steik_ribay", "caveat", "fuller",
         "Если хочется вино помощнее —", "Помощнее к стейку рибай", "body", "крепость выше"),
        # «Бульон подчёркивает терпкость» — помягче, как было.
        (RED, "borsch", "caveat", "softer", "Если хочется мягче —", "Помягче к борщу", "tannin",
         "танины мягче — по сорту"),
    ],
)  # fmt: skip
def test_selection_follows_the_reason(
    slug: str,
    dish: str,
    verdict: str,
    want: str,
    head: str,
    title: str,
    axis: str,
    pill: str,
    after,
    data,
) -> None:
    """Проверка 25.09, вечер: заголовок подборки оговорки — от причины, а не всегда «Если хочется
    мягче», и вина подборки сдвинуты именно туда: у каждой плитки причина направления из
    карточки или по сорту, мини-шкала — по оси причины. Третий круг проверки 25.09: так же и после
    «скорее нет»."""
    stream = ask(
        make_client(after, data), {"slug": slug, "chip": "dish_check", "args": {"dish": dish}}
    )
    facts = kind(stream, "facts")
    item = facts["dishes"][0]
    assert item["verdict"] == verdict
    ways = answers.REASONS[item["minus"][0]["id"]]
    assert want in {way[1] for way in ways}
    assert head in facts["verdict_template"]
    assert facts["wines_title"] == title and facts["wines"]
    assert facts["compare"]["axes"] == [axis]
    for tile_body in facts["wines"]:
        assert tile_body["pill"] and pill in tile_body["pill"], tile_body
        assert tile_body["want_source"] in ("catalog", "grape")
    # В контексте разговора — только направления чипов листа.
    assert facts["context"]["want"] == (want if want in ("softer", "fresher") else None)


def test_every_minus_rule_has_a_reason(data) -> None:
    """Каждое правило «−» — и мягкое (оговорка), и жёсткое («скорее нет») — даёт направление
    подборки (`REASONS`, третий круг проверки 25.09); направление понимает шаблон хвоста обоих
    вердиктов, а пороги вариантов — оси профиля."""
    minus = {rule.id for rule in data.rules.values() if rule.weight < 0}
    assert minus == set(answers.REASONS)
    evidence = set(answers.EVIDENCE_AXES)
    for ways in answers.REASONS.values():
        for trigger, want, by in ways:
            assert trigger is None or trigger[0] in answers.AXIS_UI
            assert want in t.CAVEAT_WORDS and by and set(by) <= evidence
            assert t.caveat_tail(want, "к борщу").startswith("Если хочется ")
            assert t.no_tail(want, "к борщу").startswith("Вот вина других виноделен ")


def test_recorded_texts_of_non_yes_verdicts_are_never_voiced() -> None:
    """Настоящие тексты повторного прогона на видеокарте 25.09: все 12 прошедших проверки ответов
    `dish_check` с вердиктом не «да» — из пакетов, которые голос больше не пересказывает.

    Среди 8 при «скорее нет» 6 грубо неверны (вердикт перевёрнут, вино карточки выдано за вино
    подборки) — проверки текста их пропустили, закрывают их только ворота `voice_fits`. Ответы
    «да» без подборки (их в том прогоне прошло 426) идут в голос, как раньше.
    """
    recorded = json.loads(
        (ROOT / "tests" / "fixtures" / "somm_recorded" / "dish_verdict_texts_2509.json").read_text(
            encoding="utf-8"
        )
    )
    cases = recorded["cases"]
    labels: dict[str, list[str]] = {}
    for case in cases:
        public = case["package"]["public"]
        (dish,) = public["dishes"]
        labels.setdefault(dish["verdict"], []).append(case["label"])
        assert dish["verdict"] != "yes" and case["text"], case["id"]
        assert not answers.voice_fits(public["intent"], public["dishes"], public["wines"]), case
    assert len(cases) == 12
    assert (len(labels["no"]), labels["no"].count("major")) == (8, 6)
    assert (len(labels["caveat"]), labels["caveat"].count("major")) == (4, 0)
    assert sum(recorded["passed_yes"].values()) == 426
    assert answers.voice_fits("dish_check", [{"verdict": "yes"}], [])
    assert answers.voice_fits("what_to_eat", [{"verdict": "yes"}] * 3, [])
    assert not answers.voice_fits("dish_check", [{"verdict": "yes"}], [{"slug": "x"}])
    assert not answers.voice_fits("dish_check", [], [])
    assert not answers.voice_fits("softer", [], [])


def test_unknown_sugar_is_never_called_dry(after: FakeAfter, data: Any) -> None:
    """Сахара вина нет в карточке: ни шаблон, ни чип правила не называют его сухим (аудит 25.09).

    К десерту такой паре правила не ставятся (`build_somm.sugar_decides`), и шаблон говорит
    прямо, что решает сахар, которого в карточке нет.
    """
    http = make_client(after, data)
    dry = re.compile(r"\bсух(?:ое|ого|ому|им|ом|ие|их)\b", re.IGNORECASE)
    for dish_id, dish in data.dishes.items():
        body = {"slug": NOSUGAR, "chip": "dish_check", "args": {"dish": dish_id}}
        facts = kind(ask(http, body), "facts")
        chips = [c["text"] for item in facts["dishes"] for c in (*item["plus"], *item["minus"])]
        assert not dry.search(" ".join([facts["verdict_template"], *chips])), dish_id
        if dish.food == "dessert":
            assert facts["dishes"][0]["verdict"] == "neutral" and not chips, dish_id
            assert facts["verdict_template"] == t.sweet_dish_sugar_unknown(dish.dative)
            assert not stoplist_hits(facts["verdict_template"])
    # С известным сахаром «сухое» к десерту — правда, и правило остаётся.
    red = kind(
        ask(http, {"slug": RED, "chip": "dish_check", "args": {"dish": "napoleon"}}), "facts"
    )
    assert red["dishes"][0]["minus"][0]["id"] == "dry_wine_on_dessert"


def test_what_to_eat_busy_stream(after: FakeAfter, data: Any) -> None:
    expected = fixture_stream("ask_what_to_eat_busy")
    client = make_client(after, data, voice=FakeVoice(reason="busy"))
    stream = ask(client, {"slug": RED, "chip": "what_to_eat"})
    assert [e.get("id") or e["type"] for e in stream] == [
        e.get("id") or e["type"] for e in expected
    ]
    facts, want = strip_times(kind(stream, "facts")), strip_times(kind(expected, "facts"))
    assert {**facts, "chips": None} == {**want, "chips": None}
    assert chip_keys(facts["chips"]) == chip_keys(want["chips"])
    assert strip_times(kind(stream, "text")) == strip_times(kind(expected, "text"))


@pytest.mark.parametrize(
    ("name", "order"),
    [("ask_refusal", "reco"), ("ask_softer_plain", "plain"), ("ask_guided_step", "reco")],
)
def test_streams_match_fixtures_exactly(name: str, order: str, after: FakeAfter, data: Any):
    request = load("requests.json")[f"{name}.ndjson"]
    assert request["query"]["order"] == order
    client = make_client(after, data)
    stream = ask(client, request["body"], order)
    assert strip_times(stream) == strip_times(fixture_stream(name))


def test_error_stream(after: FakeAfter, data: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Исключение сборки после этапа `card` — `error` + `done`, как в заглушке.

    Сбой — в ответе «Как подать»: у него после карточки этапов нет, и поток выходит тот же,
    что в заглушке (`card` → `error` → `done`).
    """

    def broken(self: Any, draft: Any, anchor: Any) -> None:
        raise ValueError(f"сломалось {MARKER}")

    monkeypatch.setattr(Sommelier, "_serve", broken)
    client = make_client(after, data)
    stream = ask(client, {"slug": RED, "chip": "serve"})
    assert strip_times(stream) == strip_times(fixture_stream("ask_error"))


# ------------------------------------------------------------------ POST ask: поток и голос
def check_stream(stream: list[dict[str, Any]]) -> None:
    """Грамматика договора: stage* (facts stage* text | error) done; время не убывает."""
    assert re.fullmatch(r"s*(fs*t|e)d", grammar(stream)), grammar(stream)
    times = [event["t_ms"] for event in stream]
    assert all(isinstance(t, int) for t in times) and times == sorted(times)
    for event in stream:
        for text in strings(strip_times(event)):
            assert check(text).clean, text


CHIPS: list[dict[str, Any]] = [
    {"chip": "what_to_eat"},
    {"chip": "serve"},
    {"chip": "softer"},
    {"chip": "fresher"},
    {"chip": "replace"},
    {"chip": "grape", "args": {"grape": "saperavi"}},
    {"chip": "term", "args": {"topic": "brut_scale"}},
    {"chip": "dish_check", "args": {"dish": "oysters"}},
    {"chip": "dish_check", "args": {"dish": "borodinsky_hleb_s_salom"}},
    {"chip": "dish_check", "args": {"dish": "napoleon"}},
    {"chip": "softer", "args": {"dish": "borsch"}},
    {"chip": "guided"},
    {"chip": "guided", "args": {"food": "fish"}},
    {"chip": "guided", "args": {"food": "meat", "want": "softer"}},
    {"chip": "guided", "args": {"food": "none", "want": "none"}},
    {"chip": "guided", "args": {"food": "dessert", "want": "fresher"}},
]
QUESTIONS = (
    "привет",
    "а к хинкали?",
    "что за погода",
    "как подать?",
    "мне 16, можно?",
    "насколько оно сладкое?",
    "оно крепкое?",
    "сколько его можно хранить?",
    "есть послаще?",
    "к новогоднему столу?",
)


@pytest.fixture
def quiet_gc() -> Any:
    """Сборщик мусора не вмешивается в замер `facts` ≤ 150 мс на запрос.

    Сам путь `facts` на заглушках — 1–2 мс (p95 1 мс), а пауза поколения 2 на куче всей сессии
    тестов без данных — около 170 мс, и где она случится, решает счёт аллокаций: пять вопросов
    факта сдвинули её внутрь этого замера (проверка кода). Время кода сверяется здесь, паузы
    сборщика на куче сервиса — замером на настоящем справочнике (`test_latency_on_real_data`, p95).
    """
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.mark.parametrize("slug", [RED, SWEET, SPARKLING, NOSUGAR])
@pytest.mark.parametrize("order", ["reco", "plain"])
def test_every_chip_and_question_streams_by_contract(slug, order, after, data, quiet_gc) -> None:
    client = make_client(after, data)
    bodies = [{"slug": slug, **chip} for chip in CHIPS]
    bodies += [{"slug": slug, "question": q} for q in QUESTIONS]
    for body in bodies:
        stream = ask(client, body, order)
        check_stream(stream)
        facts = kind(stream, "facts")
        text = kind(stream, "text")
        assert facts["t_ms"] <= 150, (body, facts["t_ms"])
        assert facts["order"] == order and facts["slug"] == slug
        assert len(facts["wines"]) <= 3
        assert len({w["winery"] for w in facts["wines"]}) == len(facts["wines"])
        assert facts["notice_149"] == (NOTICE if facts["wines"] and order == "reco" else None)
        assert (facts["wines_title"] is None) == (not facts["wines"])
        assert slug not in {w["slug"] for w in facts["wines"]}
        if order == "plain":
            assert facts["voice"] is False and text["reason"] == "plain"
            assert facts["compare"] is None
            assert all(w["reasons"] == [] and w["pill"] is None for w in facts["wines"])
        if not facts["voice"]:
            assert text["reason"] in ("not_voiced", "plain", "off")
        assert facts["context"]["intent"] == facts["intent"]


def fact_of(client: TestClient, slug: str, question: str) -> tuple[dict[str, Any], list[str]]:
    stream = ask(client, {"slug": slug, "question": question})
    check_stream(stream)
    facts = kind(stream, "facts")
    text = kind(stream, "text")
    assert facts["intent"] == "fact" and facts["voice"] is False
    assert (text["reason"], text["generated"]) == ("not_voiced", False)
    assert facts["wines"] == [] and facts["dishes"] == [] and facts["notice_149"] is None
    assert [e["id"] for e in stream if e["type"] == "stage"] == ["card"]
    return facts, [chip["id"] for chip in facts["chips"]]


def test_card_facts_by_question(after, data) -> None:
    """Простые вопросы о карточке — не «Вот что я умею»: сахар, крепость и срок хранения по
    нашему правилу стиля, без модели и без чисел, кроме «% об.»."""
    client = make_client(after, data)
    facts, chips = fact_of(client, RED, "насколько оно сладкое?")
    assert facts["verdict_template"] == "По карточке каталога это красное сухое."
    assert facts["basis"] == ["карточка каталога"] and chips[-1] == "what_to_eat"
    facts, chips = fact_of(client, RED, "есть послаще?")
    assert facts["verdict_template"] == (
        "По карточке каталога это красное сухое. Подбора «послаще» в листе нет — могу подобрать "
        "помягче или посвежее."
    )
    assert {"softer", "fresher"} <= set(chips)
    facts, _ = fact_of(client, SWEET, "есть послаще?")
    assert facts["verdict_template"].endswith("Это вино уже сладкое.")
    facts, _ = fact_of(client, NOSUGAR, "насколько оно сладкое?")
    assert facts["verdict_template"] == "Сахар этого вина в карточке каталога не указан."
    facts, chips = fact_of(client, SPARKLING, "оно сладкое?")
    assert "брют" in facts["verdict_template"] and "term" in chips  # «Что значит «брют»?»
    facts, _ = fact_of(client, RED, "оно крепкое?")
    assert facts["verdict_template"] == (
        "Крепость по карточке каталога — 13,5 % об. Для вина это средняя крепость."
    )
    facts, _ = fact_of(client, SWEET, "какая у него крепость?")
    assert facts["verdict_template"].endswith("16 % об. Для вина это высокая крепость.")
    facts, _ = fact_of(client, NOSUGAR, "оно крепкое?")
    assert facts["verdict_template"] == "Крепость этого вина в карточке каталога не указана."


@pytest.mark.parametrize(
    ("slug", "rule"),
    [(RED, "red_full"), (SWEET, "sweet"), (SPARKLING, "sparkling"), (NOSUGAR, "white")],
)
def test_storage_by_style_rule(slug: str, rule: str, after, data) -> None:
    """Срока хранения в выгрузке нет: ответ — наше правило по стилю, и это сказано прямо."""
    client = make_client(after, data)
    facts, chips = fact_of(client, slug, "сколько его можно хранить?")
    assert facts["verdict_template"] == f"{t.STORAGE_HEAD} {t.STORAGE_TEXTS[rule]}"
    assert facts["basis"][:2] == ["карточка каталога", "справочник"]
    assert chips == ["what_to_eat"]  # тем хранения в заглушках нет — их чипов тоже


def test_storage_rule_order_and_neutral_body() -> None:
    """Игристое → сладкое и креплёное → белое, розовое, оранжевое → красное по телу; тело
    неизвестно — нейтральная фраза, а не «лёгкие красные»."""
    rule = t.storage_rule
    assert rule(color="Красное", sugar="sladkoe", sparkling=True, sweet_name=False, body=4) == (
        "sparkling"
    )
    assert rule(color="Белое", sugar=None, sparkling=False, sweet_name=True, body=None) == "sweet"
    assert rule(color="Розовое", sugar="suhoe", sparkling=False, sweet_name=False, body=4) == (
        "white"
    )
    assert rule(color="Красное", sugar="suhoe", sparkling=False, sweet_name=False, body=3.0) == (
        "red"
    )
    assert rule(color="Красное", sugar=None, sparkling=False, sweet_name=False, body=None) == (
        "red_unknown"
    )
    assert rule(color=None, sugar=None, sparkling=False, sweet_name=False, body=None) == "unknown"
    for text in (*t.STORAGE_TEXTS.values(), t.STORAGE_HEAD, t.SWEETER_NONE, t.SWEETER_ALREADY):
        assert check(text).clean, text
        assert not re.search(r"\d", text), text


def test_sweet_description_is_sweet_by_card(data) -> None:
    """Приёмочная проверка 25.09: сахара нет ни в названии, ни в slug, а «Описание» — «Сладкое
    розовое вино». Для сомелье это сахар по карточке, а не догадка по названию: стиль «Розовое
    сладкое», подача и хранение сладкого, сладость профиля и чип о сахаре — «из карточки»,
    позиция справочника для подбора — с этим сахаром. Справочник и карточка «после поиска» — как
    были. Тот же мускат без таких слов в «Описании» — догадка: сахар в карточке не указан."""
    late = wine("x-late", "Икс", "Мускат позднего сбора розовый", [], ["Мускат розовый"],
                "Розовое", None, 16.0, "Крым")  # fmt: skip
    guess = wine("x-guess", "Игрек", "Мускат позднего сбора розовый", [], ["Мускат розовый"],
                 "Розовое", None, 16.0, "Крым")  # fmt: skip
    after = FakeAfter([late, guess])
    after.cards["x-late"]["description"] = "Сладкое розовое вино ЗГУ Крым"
    sommelier = Sommelier(data, answers.WineSource.of_after(after), live=False, input_enabled=True)
    rule = data.rules["sweet_wine_on_fish"]

    anchor = sommelier.anchor("x-late")
    assert (anchor.sugar, anchor.sweet_name, anchor.style_label) == (
        "sladkoe",
        False,
        "Розовое сладкое",
    )
    assert anchor.wine is not None and anchor.wine.sugar == "sladkoe"
    assert after.catalog.get("x-late").sugar is None and after.card("x-late")["sugar_class"] is None
    assert anchor.profile.source("sweetness") == "catalog" and anchor.profile.sweetness >= 4.0
    assert anchor.serve is not None and anchor.serve.rule.id == "sweet"
    assert sommelier.chip_source(rule, anchor) == "catalog"
    guessed = sommelier.anchor("x-guess")
    assert (guessed.sugar, guessed.sweet_name, guessed.style_label) == (None, True, "Розовое")
    assert guessed.serve is not None and guessed.serve.rule.id == "sweet_name"
    assert sommelier.chip_source(rule, guessed) == "type"

    client = make_client(after, data)
    facts, _ = fact_of(client, "x-late", "оно сладкое?")
    assert facts["verdict_template"] == "По карточке каталога это розовое сладкое."
    facts, _ = fact_of(client, "x-guess", "оно сладкое?")
    assert facts["verdict_template"] == t.SUGAR_UNKNOWN
    facts, _ = fact_of(client, "x-late", "сколько его можно хранить?")
    assert facts["verdict_template"] == f"{t.STORAGE_HEAD} {t.STORAGE_TEXTS['sweet']}"
    body = client.get("/v1/wines/x-late/sommelier").json()
    assert body["note"]["text"].startswith("Мускат позднего сбора розовый — розовое сладкое.")
    assert body["serve"]["rule"] == "sweet"
    sweetness = next(item for item in body["profile"] if item["axis"] == "sweetness")
    assert sweetness["source"] == "catalog"


def test_fact_texts_pass_our_own_stoplist() -> None:
    """Наши фразы факта и подачи не задевают стоп-лист голоса: «после покупки» у игристых ловил
    `покуп` (проверка кода) — шаблон не должен быть грязнее живого ответа, который им заменяют."""
    texts = [
        *t.STORAGE_TEXTS.values(),
        t.STORAGE_HEAD,
        t.SUGAR_UNKNOWN,
        t.SWEETER_ALREADY,
        t.SWEETER_NONE,
        t.SWEETER_UNKNOWN,
        t.ALCOHOL_UNKNOWN,
        t.PROFILE_SAME,
        t.alcohol_fact(9.5),
        t.alcohol_fact(12.0, 13.5),
        t.alcohol_fact(16.0),
        t.sugar_fact("красное сухое", "suhoe", sweeter=True),
        t.sugar_fact("белое сладкое", "sladkoe", sweeter=True),
        t.sugar_fact("", None, sweeter=True),
        *(t.serve_neutral(color) for color in ("Красное", "Белое", "Розовое", None)),
    ]
    for text in texts:
        assert not stoplist_hits(text), (text, stoplist_hits(text))
        assert check(text).clean, text


def test_live_text_replaces_template_only_when_generated(after, data) -> None:
    client = make_client(after, data, voice=FakeVoice(text="К борщу — да, с оговоркой."))
    stream = ask(client, {"slug": RED, "chip": "what_to_eat"})
    text = kind(stream, "text")
    assert (text["generated"], text["label"], text["reason"]) == (True, LABEL_AI, None)
    assert [e["id"] for e in stream if e["type"] == "stage"] == [
        "card",
        "rules",
        "voice",
        "verify",
    ]


def test_generated_text_failing_the_filter_falls_back(after, data) -> None:
    """Последняя сетка: текст с призывом к покупке не уходит, даже если голос его пропустил."""
    client = make_client(after, data, voice=FakeVoice(text="Купите это вино сегодня."))
    text = kind(ask(client, {"slug": RED, "chip": "what_to_eat"}), "text")
    assert (text["generated"], text["reason"], text["guard"]) == (False, "guard", "legal")
    assert text["label"] == LABEL_ALGO


def test_voice_exception_is_a_template(after, data, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    client = make_client(after, data, voice=FakeVoice(fail=True))
    stream = ask(client, {"slug": RED, "question": f"а к борщу {MARKER}?"})
    text = kind(stream, "text")
    assert (text["generated"], text["reason"]) == (False, "error")
    assert MARKER not in caplog.text


def test_live_off_means_off(after, data) -> None:
    client = make_client(after, data, live=False, voice=FakeVoice(live=False))
    # «off» — у намерений с голосом; подборка («Чем заменить») голоса не имеет вовсе
    for chip, reason in (("what_to_eat", "off"), ("replace", "not_voiced")):
        stream = ask(client, {"slug": RED, "chip": chip})
        assert kind(stream, "facts")["voice"] is False
        assert kind(stream, "text")["reason"] == reason


def test_plain_order_has_no_voice(after, data) -> None:
    voice = FakeVoice(text="Живой текст.")
    client = make_client(after, data, voice=voice)
    for chip in ({"chip": "softer"}, {"chip": "replace"}, {"chip": "what_to_eat"}):
        stream = ask(client, {"slug": RED, **chip}, "plain")
        text = kind(stream, "text")
        assert text["reason"] == "plain" and text["generated"] is False
        assert [e["id"] for e in stream if e["type"] == "stage"][-1] != "voice"
        facts = kind(stream, "facts")
        if facts["wines"]:
            names = [(w["name"].casefold(), w["slug"]) for w in facts["wines"]]
            assert names == sorted(names)
            assert facts["verdict_template"].startswith("Обычная сортировка:")
    assert all(not package.voiced for package in voice.packages)


def test_selection_is_never_voiced(after, data) -> None:
    """Подборки — всегда шаблон (решение 25.09): в прогоне на видеокарте модель выдавала вино
    карточки за вино подборки. Голос пересказывает только блюда."""
    assert answers.VOICED == {"what_to_eat", "dish_check"}
    voice = FakeVoice(text="Живой текст.")
    client = make_client(after, data, voice=voice)
    bodies = (
        {"chip": "softer"},
        {"chip": "fresher"},
        {"chip": "replace"},
        {"chip": "guided", "args": {"food": "meat", "want": "softer"}},
    )
    with_wines = 0
    for body in bodies:
        stream = ask(client, {"slug": RED, **body})
        facts, text = kind(stream, "facts"), kind(stream, "text")
        with_wines += bool(facts["wines"])
        assert facts["voice"] is False
        assert (text["generated"], text["label"], text["reason"]) == (
            False,
            LABEL_ALGO,
            "not_voiced",
        )
        assert text["text"] == facts["verdict_template"]
        assert "voice" not in [e["id"] for e in stream if e["type"] == "stage"]
    assert with_wines >= 3  # шаблон из-за намерения, а не из-за пустой подборки
    assert not any(package.voiced for package in voice.packages)
    dish_yes = {"chip": "dish_check", "args": {"dish": "borodinsky_hleb_s_salom"}}
    for body in ({"chip": "what_to_eat"}, dish_yes):
        stream = ask(client, {"slug": RED, **body})
        assert kind(stream, "facts")["voice"] is True
        assert kind(stream, "text")["generated"] is True


def test_direction_without_facts_says_it_cannot_be_evaluated(data) -> None:
    """Купаж без сахара, крепости и сортов в приорах: «помягче» и «посвежее» — не «разницы нет»,
    а честно «подобрать не по чему», без вин и без голоса."""
    blend = wine("x-blanc", "Икс", "Икс Бланк", [], ["Белые сорта винограда"], "Белое", None,
                 None, "Крым")  # fmt: skip
    client = make_client(FakeAfter([blend]), data)
    expected = {
        "softer": "Сахар этого вина в каталоге не указан, а по сорту мягкость не оценить — "
        "помягче подобрать не по чему.",
        "fresher": "Сахар и крепость этого вина в каталоге не указаны, а по сорту свежесть не "
        "оценить — посвежее подобрать не по чему.",
    }
    for chip, text in expected.items():
        stream = ask(client, {"slug": "x-blanc", "chip": chip})
        facts = kind(stream, "facts")
        assert facts["verdict_template"] == text and facts["wines"] == []
        assert "разницы нет" not in facts["verdict_template"]
        assert kind(stream, "text")["reason"] == "not_voiced"
    guided = kind(
        ask(
            client,
            {"slug": "x-blanc", "chip": "guided", "args": {"food": "meat", "want": "softer"}},
        ),
        "facts",
    )
    assert guided["verdict_template"] == expected["softer"]


def test_note_does_not_repeat_the_grape_of_the_name(data) -> None:
    """«Саперави — красное сухое. Сорт — Саперави.» — повтор: сорт из названия не пишется."""
    assert t.grapes_not_in_name(["Саперави"], "Саперави") == []
    assert t.grapes_not_in_name(["Каберне Фран"], "Каберне-Фран Резерв") == []
    assert t.grapes_not_in_name(["Саперави"], "Cru Lermont Saperavi") == []
    assert t.grapes_not_in_name(["Рислинг Рейнский"], "Cru Lermont Рислинг") == []
    assert t.grapes_not_in_name(["Мерло", "Саперави"], "Мерло Терруар") == ["Саперави"]
    assert t.grapes_not_in_name(["Рислинг", "Шардоне"], "Alveus Ultra Cuvee Brut") == [
        "Рислинг",
        "Шардоне",
    ]
    same = wine("x-saperavi", "Икс", "Саперави", ["saperavi"], ["Саперави"], "Красное", "suhoe",
                13.0, "Кубань")  # fmt: skip
    client = make_client(FakeAfter([same]), data)
    note = client.get("/v1/wines/x-saperavi/sommelier").json()["note"]["text"]
    assert note.startswith("Саперави — красное сухое. Икс, Кубань.") and "Сорт" not in note


def test_serve_is_neutral_when_the_body_is_unknown(data) -> None:
    """Красное без сорта в приорах: правило `red` выбрано не по телу, а за его неизвестностью —
    текст не утверждает, что вино лёгкое (финальная проверка 24.09)."""
    blend = wine("x-rouge", "Икс", "Икс Руж", [], ["Красные сорта винограда"], "Красное",
                 "suhoe", 13.0, "Крым")  # fmt: skip
    client = make_client(FakeAfter([blend]), data)
    facts = kind(ask(client, {"slug": "x-rouge", "chip": "serve"}), "facts")
    assert facts["serve"]["rule"] == "red" and facts["serve"]["by_grape"] is False
    assert facts["verdict_template"] == (
        "Икс Руж подают при 12–14 °C. Тело этого вина по сорту не оценить, поэтому температура — "
        "общая для красных вин."
    )
    assert "Лёгкие" not in facts["verdict_template"]
    red = kind(ask(client, {"slug": RED, "chip": "serve"}), "facts")
    assert red["serve"]["rule"] == "red_full"
    # Правило само называет 16–18 °C: температура — раз, за ней объяснение (проверка 25.09).
    assert red["verdict_template"].count("16–18 °C") == 1
    assert red["verdict_template"].endswith(
        " подают при 16–18 °C: тепло раскрывает аромат, а холод сделал бы танины жёстче."
    )


def test_serve_text_does_not_repeat_the_rule_temperature() -> None:
    """«Как подать» не пишет температуру дважды, когда её называет само правило подачи."""
    rule = "Плотные красные подают при 16–18 °C: тепло раскрывает аромат."
    assert t.serve_text("Саперави", (16, 18), rule) == (
        "Саперави подают при 16–18 °C: тепло раскрывает аромат."
    )
    assert t.serve_text("Вино", (16, 18), "Плотные красные подают при 16–18 °C.") == (
        "Вино подают при 16–18 °C."
    )
    # Другая температура в тексте правила — не повтор: текст остаётся целиком.
    assert t.serve_text("Вино", (12, 14), "Лёгкие красные подают слегка охлаждёнными.") == (
        "Вино подают при 12–14 °C. Лёгкие красные подают слегка охлаждёнными."
    )


def test_overlay_is_the_most_different_wine(after, data) -> None:
    """«Чем заменить»: на «розу» ложится вино, сильнее всех отличное по общим осям; все совпали
    — `overlay: null` и фраза, что профиль по сорту тот же."""
    somm = Sommelier(data, answers.WineSource.of_after(after), live=True, input_enabled=True)
    anchor = somm.anchor(RED)
    assert anchor is not None
    stream = ask(make_client(after, data), {"slug": RED, "chip": "replace"})
    facts = kind(stream, "facts")
    slugs = [item["slug"] for item in facts["compare"]["wines"]]
    distances = somm._distances(anchor, slugs)
    best = max(range(len(slugs)), key=lambda i: (distances[i][0], -i))
    assert facts["compare"]["overlay"] == slugs[best] and distances[best][0] > 0
    assert t.PROFILE_SAME not in facts["verdict_template"]
    # Само вино — «похожее» на себя: всё совпало, накладывать нечего.
    assert somm._overlay(anchor, [RED]) is None and somm._same_profile(anchor, [RED]) is True
    # У сладкого без сорта в приорах общих осей по сорту нет — «тот же профиль» не про него.
    assert somm._same_profile(anchor, [RED, SWEET]) is False
    assert t.replace_text("красное сухое", "Саперави", same_profile=True).endswith(t.PROFILE_SAME)
    assert check(t.PROFILE_SAME).clean


def test_dish_source_is_printed_once(after, data) -> None:
    """У блюда, все чипы которого по сорту, источник — один раз в строке блюда (`source`)."""
    body = make_client(after, data).get(f"/v1/wines/{RED}/sommelier").json()
    for dish in body["dishes"]:
        sources = {chip["source"] for chip in (*dish["plus"], *dish["minus"])}
        assert dish["source"] == (sources.pop() if len(sources) == 1 else None)
    assert {dish["source"] for dish in body["dishes"]} == {"grape"}


def test_sticky_refusal_through_context(after, data) -> None:
    client = make_client(after, data)
    first = kind(ask(client, {"slug": RED, "question": "я беременна, можно глоток?"}), "facts")
    assert first["refusal"]["topic"] == "medical" and first["context"]["refusal_topic"] == "medical"
    assert first["chips"] == []
    second = kind(
        ask(client, {"slug": RED, "chip": "what_to_eat", "context": first["context"]}), "facts"
    )
    assert second["intent"] == "refuse" and second["refusal"]["topic"] == "medical"


def test_conversation_context_round_trip(after, data) -> None:
    """«а к борщу?» → «а помягче?»: помягче к борщу, показанные вина не повторяются."""
    client = make_client(after, data)
    first = kind(ask(client, {"slug": RED, "question": "а к борщу подойдёт?"}), "facts")
    assert first["context"]["dish"] == "borsch"
    second = kind(
        ask(client, {"slug": RED, "question": "а помягче?", "context": first["context"]}),
        "facts",
    )
    assert second["intent"] == "softer" and second["context"]["dish"] == "borsch"
    assert not {w["slug"] for w in second["wines"]} & set(first["context"]["shown"])
    guided = kind(ask(client, {"slug": RED, "chip": "guided"}), "facts")
    step = kind(
        ask(client, {"slug": RED, "question": "к мясу", "context": guided["context"]}), "facts"
    )
    assert (step["intent"], step["question"]["step"]) == ("guided", "want")


# ------------------------------------------------------------------ выключатели и ошибки
def test_input_off_rejects_text_but_not_chips(after, data) -> None:
    client = make_client(after, data, input=False)
    response = client.post(ASK, json={"slug": RED, "question": "а к борщу?"})
    assert response.status_code == 422
    assert response.json() == {"detail": "вопрос текстом выключен"}
    assert kind(ask(client, {"slug": RED, "chip": "serve"}), "facts")["intent"] == "serve"
    assert client.get(f"/v1/wines/{RED}/sommelier").json()["input"] is False


@pytest.mark.parametrize(
    "body",
    [
        {"slug": RED},
        {"slug": RED, "question": "а к борщу?", "chip": "serve"},
        {"slug": RED, "question": "   \n\t "},
        {"slug": RED, "question": "а" * 201},
        {"slug": RED, "chip": "nope"},
        {"slug": RED, "chip": "dish_check"},
        {"slug": RED, "chip": "dish_check", "args": {"dish": "hinkali"}},
        {"slug": RED, "chip": "serve", "args": {"dish": "borsch"}},
        {"slug": RED, "chip": "guided", "args": {"food": "soup"}},
        {"slug": RED, "chip": "guided", "args": {"food": "meat", "want": "sweeter"}},
        {"slug": RED, "chip": "grape", "args": {"grape": "merlot"}},
        {"slug": RED, "chip": "term", "args": {"topic": "terroir"}},
        {"slug": RED, "chip": "serve", "context": {"shown": ["s"] * 13}},
        {"slug": RED, "chip": "serve", "context": {"dish": "x" * 201}},
        {"slug": RED, "chip": "serve", "extra": 1},
    ],
)
def test_bad_bodies_are_422(body: dict[str, Any], after, data) -> None:
    response = make_client(after, data).post(ASK, json=body)
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "body",
    [
        {"slug": RED, "question": [MARKER, "а к борщу?"]},
        {"slug": RED, "question": {"text": MARKER}},
        {"slug": RED, "chip": "dish_check", "args": {"dish": [MARKER]}},
        {"slug": RED, "chip": "serve", "context": [MARKER]},
        {"slug": [MARKER], "chip": "serve"},
        # Ключ, а не значение: pydantic кладёт его в `loc` (проверка кода).
        {"slug": RED, "chip": "serve", MARKER: 1},
        {"slug": RED, "chip": "dish_check", "args": {MARKER: [1]}},
    ],
)
def test_422_never_echoes_what_was_sent(body: dict[str, Any], after, data) -> None:
    """Pydantic кладёт присланное в `detail[].input`: у маршрутов сомелье его там нет."""
    response = make_client(after, data).post(ASK, json=body)
    assert response.status_code == 422, response.text
    assert MARKER not in response.text
    for error in response.json()["detail"]:
        assert set(error) <= {"type", "loc", "msg"} and error["type"] and error["msg"]
        assert error["loc"][0] == "body"


def test_422_keeps_the_field_name_in_loc(after, data) -> None:
    """Имя поля договора в `loc` остаётся: странице и прокси видно, что именно не так."""
    response = make_client(after, data).post(ASK, json={"slug": RED, "question": [MARKER]})
    assert response.json()["detail"][0]["loc"] == ["body", "question"]
    extra = make_client(after, data).post(ASK, json={"slug": RED, "chip": "serve", MARKER: 1})
    assert extra.json()["detail"][0] == {
        "type": "extra_forbidden",
        "loc": ["body"],
        "msg": "Extra inputs are not permitted",
    }


def test_422_elsewhere_keeps_the_fastapi_body(after, data) -> None:
    """Обработчик сомелье трогает только свои пути: чужой маршрут отвечает стандартным 422."""
    client = make_client(after, data)

    @client.app.get("/v1/other")
    async def other(limit: int) -> dict[str, int]:
        return {"limit": limit}

    body = client.get("/v1/other", params={"limit": MARKER}).json()
    assert body["detail"][0]["input"] == MARKER
    bad_query = client.get(f"/v1/wines/{RED}/sommelier", params={"order": MARKER})
    assert bad_query.status_code == 422 and MARKER not in bad_query.text


def test_question_is_cleaned_before_the_limit(after, data) -> None:
    client = make_client(after, data)
    padded = "  " + "а к борщу?" + " " * 300
    assert kind(ask(client, {"slug": RED, "question": padded}), "facts")["intent"] == "dish_check"


def test_unknown_slug_and_order(after, data) -> None:
    client = make_client(after, data)
    response = client.post(ASK, json={"slug": "net-takogo", "chip": "serve"})
    assert response.status_code == 404
    assert response.json() == {"detail": "нет карточки 'net-takogo'"}
    assert (
        client.post(ASK, params={"order": "x"}, json={"slug": RED, "chip": "serve"}).status_code
        == 422
    )


def test_client_gone_mid_voice_cancels_voice_and_stage_wait() -> None:
    """Клиент ушёл, пока поток ждёт этап голоса: отменяются и голос, и ожидание этапа —
    ни одной задачи не остаётся висеть до сборки мусора (проверка маршрута)."""
    package = FactsPackage(
        public={"intent": "serve", "order": "reco"},
        verdict_template="Шаблон.",
        voiced=True,
        intent_label="как подать",
        voice_input=(),
        allowed_entities=frozenset(),
        allowed_numbers=frozenset(),
        allowed_text="",
    )

    def build(route: Any) -> Any:
        return package
        yield  # генератор этапов без этапов

    class Hang:
        cancelled = False

        async def speak(self, facts: Any, *, on_stage=None, disconnected=None) -> Any:
            await on_stage("voice")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                Hang.cancelled = True
                raise

    async def scenario() -> list[asyncio.Task]:
        request = SimpleNamespace(is_disconnected=lambda: asyncio.sleep(0, result=False))
        events = ask_events(
            request,  # type: ignore[arg-type]
            started=time.perf_counter(),
            route=lambda: Route("serve", chip="serve"),
            answer=build,
            voice=Hang(),
            slug=RED,
            name="Cru Lermont Saperavi",
            chip="serve",
        )
        assert json.loads(await events.__anext__())["type"] == "facts"
        assert json.loads(await events.__anext__())["id"] == "voice"
        waiting = asyncio.ensure_future(events.__anext__())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        for _ in range(3):
            await asyncio.sleep(0)
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]

    assert asyncio.run(scenario()) == []
    assert Hang.cancelled


def test_not_ready_stream(data) -> None:
    app = FastAPI()
    register_somm_routes(app, settings=settings(), data=data, voice=FakeVoice(), preload=False)
    stream = ask(TestClient(app), {"slug": RED, "chip": "serve"})
    assert grammar(stream) == "ed"
    assert (stream[0]["code"], stream[0]["text"]) == (
        "not_ready",
        "Сомелье запускается — повторите через минуту",
    )


# ------------------------------------------------------------------ журнал и пакет модели
def test_question_is_never_logged(after, data, caplog, monkeypatch) -> None:
    """Вопрос гостя не пишется ни при ответе, ни при отказе, ни при ошибке сборки."""
    caplog.set_level(logging.DEBUG)
    voice = FakeVoice(text="Живой текст.")
    client = make_client(after, data, voice=voice)
    questions = [
        f"а к борщу {MARKER} подойдёт?",
        f"где купить {MARKER}?",
        f"{MARKER}",
        f"помягче {MARKER}",
        f"мне 16 {MARKER}",
    ]
    for question in questions:
        ask(client, {"slug": RED, "question": question})
    client.post(ASK, json={"slug": RED, "question": MARKER * 20})  # 422 — тоже без эха в журнал

    def broken(self: Any, *args: Any) -> None:
        raise KeyError(MARKER)

    monkeypatch.setattr(Sommelier, "_dish_check", broken)
    ask(client, {"slug": RED, "question": f"а к борщу {MARKER}?"})
    assert MARKER not in caplog.text
    for record in caplog.records:
        assert MARKER not in record.getMessage()
        assert MARKER not in str(record.args)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("somm ask ")]
    assert len(lines) == len(questions) + 1
    pattern = (
        r"somm ask slug=\S+ intent=\S+ input=(chip|text) chip=\S+ voice=\S+ guard=\S+ "
        r"facts_ms=-?\d+ total_ms=\d+"
    )
    assert all(re.fullmatch(pattern, line) for line in lines), lines
    # Модель не видит ни вопроса, ни описания выгрузки.
    for package in voice.packages:
        blob = json.dumps(
            [package.voice_input, package.allowed_text, package.intent_label, package.public],
            ensure_ascii=False,
        )
        assert MARKER not in blob and DESCRIPTION not in blob


def test_facts_package_for_the_model(after, data) -> None:
    """Пакет собирается и там, где голоса нет: «К борщу — да, с оговоркой» с подборкой модели не
    уходит (договор, §6.5), но имена и числа пакета — те же, что сверяли бы проверки."""
    voice = FakeVoice(text="Живой текст.")
    client = make_client(after, data, voice=voice)
    ask(client, {"slug": RED, "question": "а к борщу подойдёт?"})
    package = voice.packages[-1]
    assert not package.voiced and package.intent_label == "подходит ли вино к блюду «Борщ»"
    assert {"16", "18"} <= package.allowed_numbers
    assert {"Cru Lermont Saperavi", "Фанагория", "Кубань", "Саперави", "Борщ"} <= (
        package.allowed_entities
    )
    assert "Пино Нуар" in package.allowed_entities and "Бельбек" in package.allowed_entities
    assert sum(len(line) + 1 for line in package.voice_input) <= answers.MAX_VOICE_CHARS
    joined = "\n".join(package.voice_input)
    assert "0,8" not in joined and "Танины —" not in joined  # чисел профиля нет
    assert all("reasons" not in line for line in package.voice_input)


# ------------------------------------------------------------------ сквозь create_app
def test_routes_are_registered_in_the_service(tmp_path) -> None:
    from api_env import make_service, settings_for

    from app.api.main import create_app

    service = make_service(settings=settings_for(tmp_path))
    client = TestClient(create_app(service=service, warm=False))
    with client:
        assert client.get("/v1/wines/net-takogo/sommelier").status_code == 404
        body = client.get("/v1/wines/beta-merlot/sommelier").json()
        assert [item["axis"] for item in body["profile"]][:2] == ["sweetness", "acidity"]
        assert body["note"]["text"].startswith("Мерло Терруар — ")
        stream = ask(client, {"slug": "beta-merlot", "chip": "serve"})
        check_stream(stream)


# ------------------------------------------------------------------ настоящие данные
REAL = Path(os.environ.get("SVS_DATA_DIR") or "__нет__")
needs_real = pytest.mark.skipif(
    not (REAL / "catalog" / "wines.jsonl").is_file(), reason="нужны данные сервиса: SVS_DATA_DIR"
)


@pytest.fixture(scope="module")
def real_after() -> Any:
    """Слой «после поиска» на настоящем справочнике выгрузки и карточках без портала.

    Карточки — с «Описанием» выгрузки, если она есть (`SVS_DATASET_DIR`), как у сервиса: по нему
    снимается догадка «сладкое по названию» (`catalog.sweet_name`, третий круг проверки 25.09).
    """
    from app.api.after_layer import AfterSearch
    from app.api.cards import CatalogCards
    from app.api.service import load_catalog
    from app.config import get_settings

    records, attrs = load_catalog(REAL / "gt" / "gt_tokens.jsonl")
    csv_path = get_settings().dataset_dir / "strapi_output0709.csv"
    cards = CatalogCards.build(
        records,
        csv_path=csv_path if csv_path.is_file() else None,
        photo_map=REAL / "catalog" / "slug_photo_map.csv",
    )
    catalog = RecoCatalog.load(
        REAL / "gt" / "gt_tokens.jsonl",
        wines_path=REAL / "catalog" / "wines.jsonl",
        groups_path=REAL / "catalog" / "wine_groups.json",
    )
    return AfterSearch(catalog, cards, attrs), sorted({record["slug"] for record in records})


@needs_real
def test_every_wine_has_a_radar_on_real_data(real_after: Any) -> None:
    """«Роза ветров» обязательна: у каждой из 2 103 позиций выгрузки — не меньше трёх осей.

    Без оценки по стилю у 31 позиции (сорт не в приорах, сахар и крепость неизвестны) осей было
    одна-две. Оси «по сорту» и из карточки оценка по стилю не трогает.
    """
    after, slugs = real_after
    sommelier = Sommelier(
        load_somm_data(REAL / "somm"), answers.WineSource.of_after(after), live=False,
        input_enabled=True,
    )  # fmt: skip
    assert len(slugs) == 2103
    short, described = [], []
    for slug in slugs:
        body = sommelier.card(slug, "reco")
        assert body is not None, slug
        known = [item for item in body["profile"] if item["value"] is not None]
        if len(known) < 3:
            short.append(slug)
        profile = after.sommelier.profiles.get(slug)
        anchor = sommelier.anchor(slug)
        if anchor.wine is not None and anchor.sugar != after.catalog.get(slug).sugar:
            # Сладкое по «Описанию» (`catalog.card_sugar`, приёмочная проверка): профиль сомелье —
            # со сладостью из карточки, остальные оси — как у «Сомелье у полки».
            described.append(slug)
            assert anchor.profile.source("sweetness") == "catalog", slug
            assert profile is not None and profile.source("sweetness") is None, slug
            profile = anchor.profile
        for item in body["profile"]:
            if profile is not None and item["source"] in ("catalog", "grape"):
                assert item["source"] == profile.source(item["axis"]), (slug, item)
                assert item["value"] == round(profile.axis(item["axis"]) / 5, 2), (slug, item)
    assert not short, short
    with_csv = bool(after.cards.sources.get("csv"))
    assert described == (["muskat-pozdnego-sbora-rozovyj"] if with_csv else [])


@needs_real
def test_latency_on_real_data(real_after: Any) -> None:
    """GET …/sommelier ≤ 20 мс и `facts` ≤ 150 мс по p95 на настоящем справочнике (CPU)."""
    after, _ = real_after
    catalog = after.catalog
    service = SimpleNamespace(after=after)
    app = FastAPI()
    app.state.service = service
    data = load_somm_data(REAL / "somm")
    register_somm_routes(app, settings=settings(), data=data, voice=FakeVoice(), preload=False)
    client = TestClient(app)
    slugs = sorted(wine.slug for wine in catalog.pool)[::20][:100]
    get_ms: list[float] = []
    for slug in slugs:
        started = time.perf_counter()
        assert client.get(f"/v1/wines/{slug}/sommelier").status_code == 200
        get_ms.append((time.perf_counter() - started) * 1000)
    facts_ms: list[int] = []
    for slug in slugs[:40]:
        for chip in CHIPS[:5] + CHIPS[11:14]:
            stream = ask(client, {"slug": slug, **chip})
            facts_ms.append(kind(stream, "facts")["t_ms"])
    p95_get = sorted(get_ms)[int(len(get_ms) * 0.95) - 1]
    p95_facts = sorted(facts_ms)[int(len(facts_ms) * 0.95) - 1]
    print(f"GET p95 {p95_get:.1f} мс, facts p95 {p95_facts} мс, max {max(facts_ms)} мс")
    assert p95_get <= 20, p95_get
    assert p95_facts <= 150, p95_facts


def answer_of(sommelier: Sommelier, anchor: Any, route: Route) -> FactsPackage:
    """Пакет ответа без потока: этапы генератора пропускаются."""
    builder = sommelier.answer(anchor, route, "reco", Context())
    while True:
        try:
            next(builder)
        except StopIteration as stop:
            return stop.value


@needs_real
def test_audit_rule_findings_on_real_data(real_after: Any) -> None:
    """Пять ошибок правил из аудита прогона на видеокарте 25.09 — на тех же винах и вопросах.

    «К пасте с грибами» отвечали про томатный соус; «Бульон подчёркивает терпкость» стоял у
    сельди и сырной тарелки; утку с яблоками правила считали десертом; вину без сахара в
    выгрузке шаблон говорил «сухое»; «Пузырьки освежают жареное» — у малосольной сёмги.
    """
    after, slugs = real_after
    data = load_somm_data(REAL / "somm")
    sommelier = Sommelier(data, answers.WineSource.of_after(after), live=True, input_enabled=True)
    router = Router(data)

    def ask_text(slug: str, question: str) -> FactsPackage:
        anchor = sommelier.anchor(slug)
        assert anchor is not None, slug
        return answer_of(sommelier, anchor, router.text(question, Context(), grapes=anchor.grapes))

    def chips(package: FactsPackage) -> list[str]:
        return [
            c["text"] for dish in package.public["dishes"] for c in (*dish["plus"], *dish["minus"])
        ]

    pasta = ask_text(
        "imenie-sikory-shardone-semeynyy-rezerv-beloe-suhoe-13", "к пасте с грибами подойдёт?"
    )
    assert pasta.public["intent"] == "unknown" and pasta.verdict_template == t.UNKNOWN_DISH
    seld = ask_text("glu-glu-mr-mouse-2022", "с селёдкой нормально?")
    assert seld.public["dishes"][0]["verdict"] == "no" and "Танины спорят с рыбой" in chips(seld)
    cheese = ask_text("gunko-winery-saperavi-gunko-winery-krasnoe-suhoe-135", "к сыру подойдёт?")
    assert "Бульон подчёркивает терпкость" not in chips(seld) + chips(cheese)
    assert "бульон" not in (seld.verdict_template + cheese.verdict_template).lower()
    duck = ask_text("abrau-dyurso-imperial-kyuve-pino-nuar-rozovoe-bryut-125", "а к утке?")
    assert "десерт" not in (duck.verdict_template + " ".join(chips(duck))).lower()
    assert not any("десерт" in text.lower() for text in chips(duck))
    # Сахара в выгрузке нет — к шоколаду правила не подскажут. У мускатного розового из аудита
    # название мускатное: с проверки 25.09, вечер, правила судят его как сладкое (`rule_sugar`
    # сборки), и «сухим» его тоже не называют.
    chocolate = ask_text("daniel-22", "подойдёт к шоколаду?")
    assert chocolate.public["dishes"][0]["verdict"] == "neutral" and not chocolate.voiced
    assert chocolate.verdict_template == t.sweet_dish_sugar_unknown("к шоколаду")
    muscat = ask_text("zhemchuzhnaya-9-muskat-rozovyj-kaberne-sovinon", "подойдёт к шоколаду?")
    assert muscat.public["dishes"][0]["verdict"] != "neutral"
    for question in ("а к малосольной сёмге?", "а к блинам с икрой?", "а к селёдке под шубой?"):
        brut = ask_text("fanagoriya-alveus-ultra-cuvee-brut-shardone-beloe-bryut-12", question)
        assert "Пузырьки освежают жареное" not in chips(brut), question
        assert "жарен" not in brut.verdict_template, question
    # Ни одному из 376 вин без сахара в выгрузке сладкое блюдо не отвечает «сухим»; у 351 без
    # догадки «возможно сладкое по названию» и без «сладкого вина» в «Описании» сладкие блюда —
    # `neutral`. Догадка — у 24 из 27 креплёных, десертных и мускатных названий: «Мускат Оранж» —
    # оранжевое, у «Жемчужная 9 … Пино гри» в «Описании» — «абсолютная сухость», у «Мускат
    # позднего сбора розовый» — «Сладкое розовое вино», сладкое по карточке (приёмочная проверка).
    # Без выгрузки описания нет — догадок 26, сладких по описанию нет.
    dry = re.compile(r"\bсух(?:ое|ого|ому|им|ом|ие|их)\b", re.IGNORECASE)
    desserts = [dish_id for dish_id, dish in data.dishes.items() if dish.food == "dessert"]
    unknown = [slug for slug in slugs if (after.card(slug) or {}).get("sugar_class") is None]
    assert len(desserts) == 11 and len(unknown) == 376
    by_name = {slug for slug in unknown if sommelier.anchor(slug).sweet_name}
    described = {slug for slug in unknown if sommelier.anchor(slug).sugar is not None}
    with_csv = bool(after.cards.sources.get("csv"))
    assert len(by_name) == (24 if with_csv else 26)
    assert described == ({"muskat-pozdnego-sbora-rozovyj"} if with_csv else set())
    assert "perovskih_muskat_orange" not in by_name
    for slug in unknown:
        anchor = sommelier.anchor(slug)
        for dish_id in desserts:
            package = answer_of(sommelier, anchor, Route("dish_check", dish=dish_id))
            if slug not in by_name | described:
                assert package.public["dishes"][0]["verdict"] == "neutral", (slug, dish_id)
            assert not dry.search(package.verdict_template + " ".join(chips(package)))


@needs_real
def test_pair_rules_review_on_real_data(real_after: Any) -> None:
    """Проверка 60 случайных пар глазами 25.09, вечер — на тех же винах и вопросах.

    Портвейну без сахара в выгрузке «подходили» устрицы; полусладкому — пицца и карбонара;
    «салат с курицей» уходил в «Греческий салат», «рыба с овощами» — в «Овощи на гриле»; у
    солёных огурцов шаблон писал «деликатное блюдо», у грибов в сметане — «сахар спорит с солью и
    мясом»; подборка оговорки всегда шла «Если хочется мягче».
    """
    after, _ = real_after
    data = load_somm_data(REAL / "somm")
    sommelier = Sommelier(data, answers.WineSource.of_after(after), live=True, input_enabled=True)
    router = Router(data)

    def ask_dish(slug: str, dish: str) -> FactsPackage:
        anchor = sommelier.anchor(slug)
        assert anchor is not None, slug
        return answer_of(sommelier, anchor, Route("dish_check", dish=dish))

    def texts(package: FactsPackage, side: str) -> list[str]:
        return [chip["text"] for dish in package.public["dishes"] for chip in dish[side]]

    # 1. Креплёное, десертное и мускатное название без сахара — сладкое для правил: рыба не «да».
    fish = [dish_id for dish_id, dish in data.dishes.items() if dish.food == "fish"]
    for slug in ("portvejn-krymskij", "massandra-heres", "madera-krymskaya", "pozdnij-sbor-beloe"):
        assert not [d for d in fish if data.pairs[slug].pair(d).verdict == "yes"], slug
    oysters = ask_dish("portvejn-krymskij", "oysters")
    # Сахар портвейна — догадка по названию: оговорка, а не «скорее нет» (третий круг проверки).
    assert oysters.public["dishes"][0]["verdict"] == "caveat"
    assert "Если сладкое — спорит с рыбой" in texts(oysters, "minus")
    assert "Сладость спорит с рыбой" not in texts(oysters, "minus")
    assert ask_dish("portvejn-krymskij", "shokolad").public["dishes"][0]["verdict"] == "yes"

    # 2. Полусладкое к несладкому солёному — оговорка «Сладость спорит с солёным», подборка суше;
    # острое, десерты, фрукты и сыр — как было.
    semi = "fanagoriya-beloe-polusladkoe"
    for dish_id in ("pizza", "pasta_carbonara"):
        package = ask_dish(semi, dish_id)
        assert package.public["dishes"][0]["verdict"] == "caveat", dish_id
        assert "Сладость спорит с солёным" in texts(package, "minus"), dish_id
        assert "Если хочется суше — вот вина" in package.verdict_template, dish_id
        assert package.public["wines"], dish_id
        for tile_body in package.public["wines"]:
            assert tile_body["pill"].endswith("а не полусладкое"), tile_body
    semis = [
        s
        for s, e in data.pairs.items()
        if (after.card(s) or {}).get("sugar_class") == "polusladkoe"
    ]
    kept = ("shaverma", "kharcho", "lyulya_kebab", "pastroma", "cheese_plate", "frukty", "medovik")
    for slug in semis:
        for dish_id in kept:
            assert "semisweet_wine_on_savoury_dish" not in data.pairs[slug].pair(dish_id).minus

    # 3. Блюдо с чужой начинкой, составом или способом готовки — «такого блюда нет» с чипами.
    for question in ("к салату с курицей?", "рыба с овощами?", "к жареной картошке с луком?",
                     "к пирогу с грибами?", "к мясу с грибами?", "к картошке фри?"):  # fmt: skip
        route = router.text(question, Context(), grapes=("saperavi",))
        package = answer_of(sommelier, sommelier.anchor(RED), route)
        assert package.public["intent"] == "unknown", question
        assert package.verdict_template == t.UNKNOWN_DISH, question
        chips_ids = [chip["id"] for chip in package.public["chips"]]
        assert chips_ids and set(chips_ids) == {"dish_check"}, question
    for question, dish_id in (("а к салату?", "grecheskiy_salat"),
                              ("курица с картошкой пойдёт?", "kurinaya_grudka"),
                              ("к жареной картошке с грибами?", "zharenaya_kartoshka_s_gribami"),
                              # Алиас задом наперёд: «котлеты с пюре», «селёдка с картошкой».
                              ("пюре с котлетой подойдёт?", "kotlety_farsh"),
                              ("картошка с селёдкой?", "solenaya_seld")):  # fmt: skip
        assert router.text(question, Context(), grapes=("saperavi",)).dish == dish_id, question

    # 4. Фраза правила — правда о блюде: ни «деликатного», ни «мяса», ни «горячего», ни
    # «сытного» (то же правило у овощей на гриле, которые «Вино мощнее блюда» зовёт лёгкими).
    for rule in data.rules.values():
        said = f"{rule.chip} {rule.text}".lower()
        assert not re.search(r"деликатн|мяс|горяч|сытн|уксус|закуск|углей", said), rule.id
    # К рыбе подборка даёт и лёгкие красные: фраза правила не требует «белое, розовое, игристое».
    assert "белое, розовое" not in data.rules["no_tannin_with_oily_fish"].text
    pickles = ask_dish("chateau-de-talu-kaberne-fran-rezerv-krasnoe-suhoe-13", "solenye_ogurcy")
    assert "лёгкого блюда" in pickles.verdict_template
    mushrooms = ask_dish("fanagoriya-ice-wine-kaberne-kaberne-sovinon-krasnoe-sladkoe-11",
                         "griby_v_smetane")  # fmt: skip
    assert "Сладость спорит с блюдом" in texts(mushrooms, "minus")

    # 5. Подборка оговорки — от причины: вину слабее утки — помощнее, и вина правда помощнее.
    duck = ask_dish("aya-organic-wine-vineyards-evolution-pinot-noir-pino-nuar-krasnoe-suhoe-125",
                    "utka_s_yablokami")  # fmt: skip
    assert texts(duck, "minus")[0] == "Блюдо насыщеннее вина"
    assert "Если хочется вино помощнее — вот вина" in duck.verdict_template
    assert duck.public["wines"]
    for tile_body in duck.public["wines"]:
        assert tile_body["pill"] in ("крепость выше", "тело плотнее — по сорту"), tile_body


@needs_real
def test_pair_rules_third_round_on_real_data(real_after: Any) -> None:
    """Третий круг проверки пар 25.09 — на настоящих данных.

    1. Сладкое к рыбе — «скорее нет»; креплёное, десертное, мускатное название без сахара
       («возможно сладкое») — оговорка «если вино сладкое…» (третий круг проверки), а «скорее
       нет» — только по другому правилу.
    2. Портвейн, херес, мадера и другие сладкие к сырной тарелке и орехам — «да».
    3. Подборка после «скорее нет» и оговорки — в сторону причины и не держится цвета вина
       карточки: белому Шардоне к шашлыку из баранины — помощнее и красные, терпкому красному к
       устрицам — помягче, белые и розовые.
    4. Направление — не шум порога: «полегче» и «помощнее» по крепости — на полградуса и больше,
       по телу — на 0,5 по сорту; «полегче» по причине крепости — правда слабее; подпись плитки —
       факт причины.
    """
    after, slugs = real_after
    data = load_somm_data(REAL / "somm")
    sommelier = Sommelier(data, answers.WineSource.of_after(after), live=True, input_enabled=True)
    catalog, profiles = after.catalog, after.sommelier.profiles

    def ask_dish(slug: str, dish: str) -> FactsPackage:
        anchor = sommelier.anchor(slug)
        assert anchor is not None, slug
        return answer_of(sommelier, anchor, Route("dish_check", dish=dish))

    def minus(package: FactsPackage) -> list[str]:
        return [chip["text"] for dish in package.public["dishes"] for chip in dish["minus"]]

    # 1. Сладкое к рыбе — «скорее нет» у каждого сладкого по карточке вина и каждого рыбного
    # блюда; у сладкого только по названию — оговорка, «скорее нет» — не из-за догадки.
    fish = [dish_id for dish_id, dish in data.dishes.items() if dish.food == "fish"]
    known = [s for s in slugs if (after.card(s) or {}).get("sugar_class") == "sladkoe"]
    guessed = [
        s
        for s in slugs
        if sweet_name((card := after.card(s) or {}).get("name"), None, card.get("description"))
        and card.get("sugar_class") is None
    ]
    # Приёмочная проверка: «Сладкое розовое вино» в «Описании» — сладкое по карточке, не догадка.
    described = [
        s
        for s in slugs
        if (card := after.card(s) or {}).get("sugar_class") is None
        and card_sugar(None, card.get("description")) == "sladkoe"
    ]
    sweet = known + described + guessed
    with_csv = bool(after.cards.sources.get("csv"))
    assert len(known) == 59 and len(guessed) == (24 if with_csv else 26)
    assert described == (["muskat-pozdnego-sbora-rozovyj"] if with_csv else [])
    for slug in known + described:
        assert {data.pairs[slug].pair(d).verdict for d in fish} == {"no"}, slug
    for slug in described:
        late = ask_dish(slug, "solenaya_seld")
        assert late.public["dishes"][0]["verdict"] == "no"
        assert late.public["dishes"][0]["minus"][0] == {
            "id": "sweet_wine_on_fish",
            "text": "Сладость спорит с рыбой",
            "source": "catalog",
        }
        assert "может быть сладким" not in late.verdict_template
    hedge = "maybe_sweet_wine_on_fish"
    for slug in guessed:
        for dish_id in fish:
            pair = data.pairs[slug].pair(dish_id)
            assert hedge in pair.minus and "sweet_wine_on_fish" not in pair.minus, (slug, dish_id)
            assert pair.verdict == "caveat" or (pair.verdict == "no" and pair.minus[0] != hedge)
    oysters = ask_dish("portvejn-krymskij", "oysters")
    assert oysters.public["dishes"][0]["verdict"] == "caveat"
    assert minus(oysters) == ["Если сладкое — спорит с рыбой"]
    assert "может быть сладким: если так, сладость будет спорить с рыбой" in (
        oysters.verdict_template
    )
    assert "Если хочется суше — вот вина других виноделен, которые подходят к устрицам." in (
        oysters.verdict_template
    )
    assert oysters.public["wines_title"] == "Суше к устрицам" and oysters.public["wines"]
    for tile_body in oysters.public["wines"]:
        # «Суше» — по сахару карточки кандидата, а не по догадке.
        assert catalog.get(tile_body["slug"]).sugar not in (None, "sladkoe"), tile_body
        assert tile_body["pill"].endswith("по карточке"), tile_body
    sweet_oysters = ask_dish(known[0], "oysters")
    assert sweet_oysters.public["dishes"][0]["verdict"] == "no"
    assert minus(sweet_oysters)[0] == "Сладость спорит с рыбой"
    assert "Вот вина других виноделен суше, которые подходят к устрицам." in (
        sweet_oysters.verdict_template
    )

    # 2. Сладкое и креплёное к сыру и орехам — «да».
    for slug in ("portvejn-krymskij", "massandra-heres", "madera-krymskaya",
                 "massandra-portveyn-krasnyy-livadiya-kaberne-sovinon-krasnoe-sladkoe-185"):  # fmt: skip
        for dish_id in ("cheese_plate", "orehi"):
            package = ask_dish(slug, dish_id)
            item = package.public["dishes"][0]
            assert item["verdict"] == "yes" and not item["minus"], (slug, dish_id)
            assert "Сладкое к сыру и орехам" in [chip["text"] for chip in item["plus"]]
    for slug in sweet:
        for dish_id in ("cheese_plate", "orehi"):
            assert data.pairs[slug].pair(dish_id).verdict == "yes", (slug, dish_id)

    # 3. Подборка — в сторону причины и в стиль, куда зовёт причина.
    for slug, verdict in (("donskoe-vinodelcheskoe-hozyaystvo-elbuzd-shardone-beloe-suhoe-13",
                           "caveat"),
                          ("abrau-dyurso-brut-dor-blanc-de-blancs-shardone-beloe-bryut-12", "no")):  # fmt: skip
        lamb = ask_dish(slug, "shashlyk_baranina")
        assert lamb.public["dishes"][0]["verdict"] == verdict
        assert lamb.public["wines_title"] == "Помощнее к шашлыку из баранины"
        assert lamb.public["wines"], slug
        for tile_body in lamb.public["wines"]:
            assert catalog.get(tile_body["slug"]).color == "Красное", tile_body
            assert tile_body["pill"] in ("крепость выше", "тело плотнее — по сорту"), tile_body
    tannic = ask_dish(RED, "oysters")
    assert tannic.public["dishes"][0]["verdict"] == "no"
    assert minus(tannic)[0] == "Танины спорят с рыбой"
    assert "Вот вина других виноделен помягче, которые подходят к устрицам." in (
        tannic.verdict_template
    )
    assert tannic.public["wines_title"] == "Помягче к устрицам" and tannic.public["wines"]
    for tile_body in tannic.public["wines"]:
        assert catalog.get(tile_body["slug"]).color in ("Белое", "Розовое"), tile_body
        assert tile_body["pill"].endswith(", а не красное"), tile_body

    # 4. Направление — не шум порога, подпись — факт причины. Все «−» пар части выгрузки.
    def degrees(slug: str) -> tuple[float, float]:
        alcohol = catalog.alcohol(slug)
        assert alcohol.value is not None, slug
        return alcohol.value, alcohol.max if alcohol.max is not None else alcohol.value

    def grape_gap(anchor: str, other: str, axis: str) -> float:
        mine, theirs = profiles[anchor], profiles[other]
        assert mine.source(axis) == theirs.source(axis) == "grape", (anchor, other, axis)
        return round(theirs.axis(axis) - mine.axis(axis), 6)

    checked: Counter[str] = Counter()
    for slug in slugs[::23]:
        anchor = sommelier.anchor(slug)
        if anchor is None or anchor.wine is None:
            continue
        for dish_id in sorted(data.dishes)[::3]:
            pair = sommelier.pair(anchor, dish_id)
            if pair.verdict not in ("caveat", "no"):
                continue
            reason = sommelier._reason(pair, anchor, dish_id)
            package = ask_dish(slug, dish_id)
            title = package.public["wines_title"] or ""
            if reason is None or not title.startswith(t.CAVEAT_WORDS[reason.want][1]):
                continue
            for tile_body in package.public["wines"]:
                other, pill = tile_body["slug"], tile_body["pill"]
                checked[pill] += 1
                if pill == "крепость ниже":
                    assert degrees(slug)[0] - degrees(other)[1] >= 0.5 - 1e-9, (slug, other)
                elif pill == "крепость выше":
                    assert degrees(other)[0] - degrees(slug)[1] >= 0.5 - 1e-9, (slug, other)
                elif pill == "тело легче — по сорту":
                    assert grape_gap(slug, other, "body") <= -0.5, (slug, other)
                elif pill == "тело плотнее — по сорту":
                    assert grape_gap(slug, other, "body") >= 0.5, (slug, other)
                elif pill == "кислотность выше — по сорту":
                    assert reason.want == "fresher", (slug, dish_id, reason)
                    assert grape_gap(slug, other, "acidity") >= 0.5, (slug, other)
                if reason.by == ("strength",):
                    assert pill == "крепость ниже", (slug, dish_id, pill)
                if reason.want == "fresher":
                    assert pill == "кислотность выше — по сорту", (slug, dish_id, pill)
    assert checked["крепость ниже"] and checked["тело плотнее — по сорту"], checked
    assert checked["крепость выше"] and checked["танины мягче — по сорту"], checked

    # Причина — крепость: «полегче» только слабее на полградуса; мягкая кислотность — посвежее
    # по кислотности, а не «крепость ниже» (так было до третьего круга).
    strong = "bakla-vines-kaberne-fran-krasnoe-suhoe-143"
    nuts = ask_dish(strong, "orehi")
    assert minus(nuts) == ["Крепость перекрывает блюдо"]
    assert nuts.public["wines_title"] == "Полегче к орехам" and nuts.public["wines"]
    for tile_body in nuts.public["wines"]:
        assert tile_body["pill"] == "крепость ниже", tile_body
        assert degrees(strong)[0] - degrees(tile_body["slug"])[1] >= 0.5 - 1e-9, tile_body
    flat = "vinodelnya-myshako-sesto-senso-tropicheskiy-vzryv-gevyurtstraminer-beloe-suhoe-12"
    for dish_id, title in (("solyanka", "Посвежее к солянке"), ("burger", "Посвежее к бургеру")):
        package = ask_dish(flat, dish_id)
        assert package.public["wines_title"] == title and package.public["wines"], dish_id
        for tile_body in package.public["wines"]:
            assert tile_body["pill"] == "кислотность выше — по сорту", tile_body
            assert grape_gap(flat, tile_body["slug"], "acidity") >= 0.5, tile_body


# ------------------------------------------------------------------ вопрос экрана check (§5)
def test_question_candidates_use_card_facts(after: FakeAfter) -> None:
    """Факты вариантов — поля карточки без портала; заглушка «Белые сорта винограда» — не сорт."""
    from app.api.after_layer import question_candidates

    result = SimpleNamespace(
        top5=[SimpleNamespace(slug=slug) for slug in (SWEET, SPARKLING, "net-takogo", NOSUGAR)]
    )
    candidates = question_candidates(result, after)  # type: ignore[arg-type]
    assert [c["slug"] for c in candidates] == [SWEET, SPARKLING, NOSUGAR]
    assert candidates[0]["facts"] == {
        "sugar": "sladkoe",
        "color": "Белое",
        "sparkling": False,
        "grapes": [],
        "abv": 16.0,
    }
    read = {"winery": None, "color": "Белое", "sugar": None, "grapes": [], "abv": None,
            "sparkling": None}  # fmt: skip
    question = check_question(read, candidates)
    assert question is not None and question["field"] == "sugar"
    assert question["text"] == "Что на этикетке: сладкое или брют?"


def test_scan_check_screen_gets_a_question(tmp_path) -> None:
    """`/v1/scan` на экране `check` несёт `after.question`, на `found` — `null`; predict — прежний."""
    from reco_env import make_reco_service

    from app.api.main import create_app

    service = make_reco_service(tmp_path)
    from api_env import image_bytes

    files = {"image": ("q.png", image_bytes(), "application/octet-stream")}
    with TestClient(create_app(service=service, warm=False)) as client:
        found = client.post("/v1/scan", files=files).json()
        service.after.suggest_max = 1.0  # подсказка переводит кадр на экран check
        flagged = client.post("/v1/scan", files=files).json()
        predict = client.post("/v1/eval/predict", files=files).json()
    assert found["after"]["state"] == "found" and found["after"]["question"] is None
    assert flagged["after"]["state"] == "check"
    question = flagged["after"]["question"]
    slugs = [cand["slug"] for cand in flagged["candidates"]]
    if question is not None:
        assert all(option["slug"] in slugs for option in question["options"])
        assert check(question["text"]).clean
    assert "question" not in predict and "after" not in predict
