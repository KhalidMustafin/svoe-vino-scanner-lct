"""Смысловой слой барьера (`app/sommelier/safety.py`): голосование, односторонность, без модели.

Логика голосования проверяется на подставном кодировщике: фразы — заранее заданные векторы,
поэтому видно, какой сосед голосует и почему. Настоящая rubert-tiny2 — только если её веса есть в
локальном кэше Hugging Face (скачивания нет): тогда проверяется, что слой поднимается на CPU и
отвечает быстро. Замер порогов на живой речи — `research/2026-09-24_somm/safety_eval.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from app.recommend.somm_data import load_somm_data
from app.sommelier import barrier
from app.sommelier.router import Context, Router
from app.sommelier.safety import (
    ANCHORS_PATH,
    CLASSES,
    MODEL_NAME,
    TOPIC_OF_CLASS,
    SafetyLayer,
    load_anchors,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "somm"

#: Направления подставного пространства: у каждого класса — своя ось.
AXES = {"зависимость": 0, "возраст": 1, "ok": 2, "офтоп": 3, "шум": 4}
ANCHORS = {
    "зависимость": ["з1", "з2", "з3"],
    "возраст": ["в1"],
    "ok": ["о1", "о2", "о3"],
    "офтоп": ["ф1"],
}


def unit(**weights: float) -> np.ndarray:
    vector = np.zeros(len(AXES), dtype=np.float32)
    for name, weight in weights.items():
        vector[AXES[name]] = weight
    return vector / np.linalg.norm(vector)


#: Эталоны лежат ровно на осях; вопросы — рядом с нужной осью или между осями.
VECTORS = {
    **{text: unit(**{name: 1.0}) for name, texts in ANCHORS.items() for text in texts},
    "тянет каждый вечер": unit(зависимость=1.0, шум=0.2),  # близко к «зависимости»: ≈ 0,98
    "что к борщу": unit(ok=1.0, шум=0.2),
    "между": unit(зависимость=1.0, ok=1.0),  # ≈ 0,71 к обоим — ниже порога
    "спорный": unit(возраст=1.05, офтоп=1.0),  # голоса ≈ 0,58 за возраст и 0,42 за офтоп
    "далеко": unit(шум=1.0),
}


def fake_encoder(texts):
    return np.stack([VECTORS[text] for text in texts])


@pytest.fixture
def anchors_path(tmp_path: Path) -> Path:
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps({"anchors": ANCHORS}, ensure_ascii=False), encoding="utf-8")
    return path


def layer(anchors_path: Path, **kwargs) -> SafetyLayer:
    return SafetyLayer(anchors_path=anchors_path, encoder=fake_encoder, **kwargs)


# ------------------------------------------------------------------ голосование
def test_close_risk_anchors_give_their_topic(anchors_path: Path) -> None:
    safety = layer(anchors_path, near=0.75)
    name, share = safety.classify("тянет каждый вечер")
    assert name == "зависимость" and share == 1.0
    assert safety.refusal_topic("тянет каждый вечер") == "addiction"
    assert safety.stats()["refused"] == {"addiction": 1}


def test_ok_and_offtopic_neighbours_never_refuse(anchors_path: Path) -> None:
    safety = layer(anchors_path, near=0.75)
    assert safety.classify("что к борщу")[0] == "ok"
    assert safety.refusal_topic("что к борщу") is None


def test_background_similarity_does_not_vote(anchors_path: Path) -> None:
    """≈ 0,71 к двум классам — ниже порога: соседей нет, риска нет."""
    assert layer(anchors_path, near=0.75).classify("между") is None
    assert layer(anchors_path, near=0.75).classify("далеко") is None


def test_winner_needs_its_share_of_votes(anchors_path: Path) -> None:
    """Возраст набирает ≈ 0,58 голосов против офтопа: отказ при доле 0,55 и тишина при 0,6."""
    assert layer(anchors_path, near=0.6).refusal_topic("спорный") == "minors"
    assert layer(anchors_path, near=0.6, min_share=0.6).refusal_topic("спорный") is None


def test_anchor_does_not_vote_for_itself_when_excluded(anchors_path: Path) -> None:
    safety = layer(anchors_path, near=0.75)
    assert safety.classify("в1")[0] == "возраст"
    assert safety.classify("в1", exclude=frozenset({"в1"})) is None


def test_disabled_layer_is_silent_and_loads_nothing(anchors_path: Path) -> None:
    safety = layer(anchors_path, enabled=False)
    safety.start()
    assert safety.refusal_topic("тянет каждый вечер") is None
    assert safety.stats()["status"] == "off" and safety.stats()["anchors"] == 0


def test_missing_model_is_rules_only(anchors_path: Path) -> None:
    """Весов нет в локальном кэше — `missing`, ни исключения, ни отказа; сети нет."""
    safety = SafetyLayer(anchors_path=anchors_path, model_name="svs-test/no-such-model")
    assert safety.load() is False
    stats = safety.stats()
    assert stats["status"] == "missing" and stats["error"]
    assert safety.refusal_topic("тянет каждый вечер") is None


def test_broken_anchors_are_an_error_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps({"anchors": {"ok": ["о1"]}}), encoding="utf-8")
    safety = SafetyLayer(anchors_path=path, encoder=fake_encoder)
    assert safety.load() is False and safety.stats()["status"] == "error"


def test_shipped_anchors_are_loza_classes() -> None:
    anchors = load_anchors(ANCHORS_PATH)
    assert set(anchors) == set(CLASSES)
    assert all(anchors[name] for name in TOPIC_OF_CLASS)
    assert len(anchors["ok"]) >= 50
    assert set(TOPIC_OF_CLASS.values()) <= set(barrier.TOPIC_ORDER)


# ------------------------------------------------------------------ односторонность в маршруте
class AlwaysAddiction:
    """Слой, который видит зависимость в любом вопросе: так видно, что он может и чего нет."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def refusal_topic(self, question: str) -> str | None:
        self.asked.append(question)
        return "addiction"


@pytest.fixture(scope="module")
def data():
    return load_somm_data(None, fallback=FIXTURES, strict=True)


def test_layer_only_adds_refusals_never_replaces_rules(data) -> None:
    safety = AlwaysAddiction()
    router = Router(data, safety=safety)
    # Правила отказали — тема правил, слой не спрашивается.
    route = router.text("мне 16, к борщу?", Context())
    assert route.intent == "refuse" and route.refusal.topic == "minors"
    assert safety.asked == []
    # Светская беседа — раньше барьера, слой не спрашивается.
    assert router.text("спасибо", Context()).intent == "smalltalk"
    assert safety.asked == []
    # Правила промолчали — слой может добавить отказ, и он липкий, как у правил.
    route = router.text("а к борщу подойдёт?", Context())
    assert route.intent == "refuse" and route.refusal == barrier.refusal("addiction")
    assert route.refusal.sticky and safety.asked == ["а к борщу подойдёт?"]


def test_silent_layer_changes_nothing(data) -> None:
    class Silent:
        def refusal_topic(self, question: str) -> None:
            return None

    rules, both = Router(data), Router(data, safety=Silent())
    for question in ("а к борщу подойдёт?", "где купить подешевле?", "мне 16", "как подать"):
        assert both.text(question, Context()) == rules.text(question, Context())


def test_chips_never_ask_the_layer(data) -> None:
    safety = AlwaysAddiction()
    assert Router(data, safety=safety).chip("serve", {}, Context()).intent == "serve"
    assert safety.asked == []


# ------------------------------------------------------------------ настоящая модель
def _cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(MODEL_NAME, "config.json"), str)


@pytest.mark.skipif(not _cached(), reason="rubert-tiny2 нет в локальном кэше Hugging Face")
def test_real_model_loads_from_disk_and_answers_fast_on_cpu() -> None:
    safety = SafetyLayer()
    assert safety.load() is True
    assert safety.stats()["anchors"] == sum(len(v) for v in load_anchors().values())
    # Эталон находит свой класс (без исключения себя — это проверка сборки, а не замер).
    assert safety.classify("не могу остановиться на одном бокале")[0] == "зависимость"
    assert safety.refusal_topic("а к борщу подойдёт?") is None
    started = time.perf_counter()
    for _ in range(20):
        safety.classify("что подать к утке с яблоками")
    assert (time.perf_counter() - started) / 20 < 0.05  # < 50 мс на вопрос на CPU
