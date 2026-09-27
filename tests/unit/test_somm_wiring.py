"""«Сомелье» в собранном сервисе: настоящие модули дорожек и одни настройки на всех.

После слияния дорожек (данные, ворота и голос, маршрут, страница) проверяется стык:

* `SVS_SOMM_*` читает только `SommSettings` (`app/sommelier/settings.py`), а сервис держит тот же
  объект в `ServiceSettings.somm`: каталог данных у «Сомелье у полки» и у маршрутов один, и
  неверный флаг — отказ старта `SettingsError`, как у остальных `SVS_*`;
* ворота видеокарты `create_app` берут тихое окно из `SVS_SOMM_QUIET_S`;
* старт сервиса отдаёт маршрутам данные слоя «после поиска» (тот же объект, без второго чтения
  20 МБ пар) и собирает настоящий голос: `Voice` с воротами приложения,
  `OllamaText.mirroring(читатель этикетки)` и `EntityLock(vocab)`;
* `/v1/health` несёт блок `somm`: выключатели, данные, ворота и голос;
* выключатели работают сквозь `create_app`, VLM выключен — голос `not_ready`;
* двойника `app/sommelier/interim.py` больше нет, пачка данных везёт `somm/` и не везёт портал.

Сервис — на фейках (`reco_env`, `api_env`), без моделей, видеокарты и сети: Ollama не зовётся.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from api_env import make_service, settings_for
from fastapi.testclient import TestClient
from reco_env import make_reco_service

from app.api.config import ServiceSettings, SettingsError
from app.api.main import create_app
from app.config import REPO_ROOT
from app.recommend.somm_data import SOMM_FILES, load_somm_data
from app.sommelier.entity_lock import EntityLock
from app.sommelier.gate import SommGate
from app.sommelier.ollama_text import OllamaText
from app.sommelier.settings import SettingsError as SommSettingsError
from app.sommelier.settings import SommSettings
from app.sommelier.voice import Voice

ASK = "/v1/sommelier/ask"
SLUG = "beta-merlot"
LABEL_ALGO = "Текст и подбор — алгоритм"


def stream(client: TestClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.post(ASK, json=body)
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines()]


def text_of(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event["type"] == "text")


# ------------------------------------------------------------------ настройки
def test_settings_defaults_are_the_contract() -> None:
    settings = SommSettings.from_env({})
    assert settings == SommSettings()
    assert (settings.live, settings.input) == (True, True)
    assert settings.data_dir == REPO_ROOT / "data" / "somm"
    assert (settings.timeout_ms, settings.quiet_s, settings.timeout_s) == (4000, 15.0, 4.0)


def test_settings_from_env(tmp_path: Path) -> None:
    env = {
        "SVS_SOMM_LIVE": "Off",
        "SVS_SOMM_INPUT": "no",
        "SVS_SOMM_TIMEOUT_MS": "2500",
        "SVS_SOMM_QUIET_S": "7.5",
        "SVS_DATA_DIR": str(tmp_path),
    }
    settings = SommSettings.from_env(env)
    assert (settings.live, settings.input) == (False, False)
    assert (settings.timeout_ms, settings.quiet_s) == (2500, 7.5)
    assert settings.data_dir == tmp_path.resolve() / "somm"
    own = tmp_path / "own"
    assert SommSettings.from_env({**env, "SVS_SOMM_DIR": str(own)}).data_dir == own.resolve()
    assert SommSettings.from_env({"SVS_SOMM_LIVE": " YES "}).live is True
    assert SommSettings.from_env({}).safety is False
    assert SommSettings.from_env({"SVS_SOMM_SAFETY": "1"}).safety is True


@pytest.mark.parametrize(
    "env",
    [
        {"SVS_SOMM_LIVE": "2"},
        {"SVS_SOMM_INPUT": "maybe"},
        {"SVS_SOMM_TIMEOUT_MS": "4s"},
        {"SVS_SOMM_TIMEOUT_MS": "0"},
        {"SVS_SOMM_QUIET_S": "-1"},
        {"SVS_SOMM_SAFETY": "semantic"},
    ],
)
def test_bad_values_stop_the_start(env: dict[str, str]) -> None:
    with pytest.raises(SommSettingsError):
        SommSettings.from_env(env)
    # Сервис читает их тем же разбором и падает своей ошибкой настроек: `python -m app.api` — код 2.
    with pytest.raises(SettingsError):
        ServiceSettings.from_env(env)


def test_service_settings_hold_the_same_somm_settings(tmp_path: Path) -> None:
    """Одна переменная — одно значение: `somm_dir` слоя «после поиска» и данные маршрутов."""
    env = {"SVS_DATA_DIR": str(tmp_path), "SVS_SOMM_QUIET_S": "3", "SVS_SOMM_LIVE": "0"}
    service = ServiceSettings.from_env(env)
    assert service.somm == SommSettings.from_env(env)
    assert service.somm_dir == service.somm.data_dir == tmp_path.resolve() / "somm"
    assert ServiceSettings.from_env({}) == ServiceSettings()
    assert service.public()["somm"]["data_dir"] == str(service.somm_dir)
    json.dumps(service.public())  # настройки для журнала — чистый JSON


# ------------------------------------------------------------------ старт сервиса
def test_gate_quiet_window_comes_from_settings(tmp_path: Path) -> None:
    somm = SommSettings(data_dir=tmp_path / "somm", quiet_s=3.0)
    service = make_service(settings=settings_for(tmp_path, somm=somm))
    app = create_app(service=service, warm=False)
    assert app.state.somm_gate.quiet_s == 3.0
    assert app.state.somm.settings is somm


def test_start_shares_after_data_and_builds_the_real_voice(tmp_path: Path) -> None:
    service = make_reco_service(tmp_path)
    app = create_app(service=service, warm=False)
    runtime = app.state.somm
    with TestClient(app):
        assert runtime.data() is service.after.somm  # те же данные, второго чтения нет
        voice = runtime.voice(app, service)
    assert isinstance(voice, Voice)
    assert voice.gate is app.state.somm_gate and isinstance(voice.gate, SommGate)
    assert voice.gate.bound
    assert isinstance(voice.client, OllamaText) and isinstance(voice.lock, EntityLock)
    reader = service.vlm.reader  # type: ignore[union-attr]
    assert (voice.client.model, voice.client.keep_alive, voice.client.num_ctx) == (
        reader.model,
        reader.keep_alive,
        reader.num_ctx,
    )
    assert voice.live is True and voice.timeout_s == 4.0


def test_other_somm_dir_is_read_on_its_own(tmp_path: Path) -> None:
    """Каталог маршрутов не тот, что у слоя «после поиска», — данные читаются отдельно."""
    service = make_reco_service(tmp_path)
    other = SommSettings(data_dir=tmp_path / "другой")
    app = create_app(settings_for(tmp_path, somm=other), service, warm=False)
    with TestClient(app):
        data = app.state.somm.data()
    assert data is not service.after.somm
    assert set(data.sources.values()) == {"fixture"}


def test_health_has_the_somm_block(tmp_path: Path) -> None:
    service = make_reco_service(tmp_path)
    app = create_app(service=service, warm=False)
    with TestClient(app) as client:
        stream(client, {"slug": SLUG, "chip": "serve"})
        somm = client.get("/v1/health").json()["somm"]
    assert set(somm) == {
        "status",
        "degraded_reasons",
        "live",
        "input",
        "data",
        "gate",
        "voice",
        "safety",
        "asked",
    }
    # Смысловой слой по умолчанию выключен: замер 24.09 не нашёл порога без лишних отказов.
    assert somm["safety"]["enabled"] is False and somm["safety"]["status"] == "off"
    assert (somm["live"], somm["input"], somm["asked"]) == (True, True, 1)
    assert list(somm["data"]) == list(SOMM_FILES)
    assert somm["data"]["pairs.json"]["source"] == "data"
    assert somm["data"]["vocab.json"]["source"] == "fixture"
    assert somm["gate"]["bound"] is True and somm["gate"]["quiet_s"] == 15.0
    assert somm["voice"]["calls"] == 1 and somm["voice"]["model"] == service.settings.vlm_model


def test_health_before_start_has_the_block_too() -> None:
    app = create_app(service=make_service(), warm=False)
    somm = TestClient(app).get("/v1/health").json()["somm"]  # без lifespan: сервис не собран
    assert somm["data"] is None and somm["voice"] is None and somm["gate"]["bound"] is False
    assert (somm["status"], somm["degraded_reasons"]) == ("starting", [])


def test_fixture_data_is_degraded_and_warned(tmp_path: Path) -> None:
    """Заглушки вместо `data/somm` — не тихая подмена: `somm.status = degraded`, причины по файлам
    и предупреждение в общем `warnings`, которое читает проверка готовности `run_eval.sh`."""
    service = make_reco_service(tmp_path)
    with TestClient(create_app(service=service, warm=False)) as client:
        health = client.get("/v1/health").json()
    somm = health["somm"]
    assert somm["status"] == "degraded"
    assert "data:vocab.json=fixture" in somm["degraded_reasons"]
    assert "data:pairs.json=data" not in " ".join(somm["degraded_reasons"])
    warned = [w for w in health["warnings"] if w.startswith("Сомелье на неполных данных")]
    assert len(warned) == 1 and "vocab.json=fixture" in warned[0]
    assert "somm" not in " ".join(health["degraded_reasons"])  # статус сканера — свой


def test_full_data_is_ok_and_silent(tmp_path: Path) -> None:
    """Все пять файлов в `data/somm` — `ok`, без причин и без предупреждения."""
    somm_dir = tmp_path / "somm"
    somm_dir.mkdir(parents=True, exist_ok=True)
    fixtures = REPO_ROOT / "tests" / "fixtures" / "somm"
    service = make_reco_service(tmp_path)
    for name in SOMM_FILES:
        if not (somm_dir / name).is_file():
            (somm_dir / name).write_bytes((fixtures / name).read_bytes())
    service.after.somm = load_somm_data(somm_dir, fallback=None)
    with TestClient(create_app(service=service, warm=False)) as client:
        health = client.get("/v1/health").json()
    assert (health["somm"]["status"], health["somm"]["degraded_reasons"]) == ("ok", [])
    assert not [w for w in health["warnings"] if w.startswith("Сомелье")]


# ------------------------------------------------------------------ выключатели сквозь сервис
def test_switches_through_create_app(tmp_path: Path) -> None:
    somm = SommSettings(data_dir=tmp_path / "somm", live=False, input=False)
    service = make_reco_service(tmp_path, somm=somm)
    app = create_app(service=service, warm=False)
    with TestClient(app) as client:
        card = client.get(f"/v1/wines/{SLUG}/sommelier").json()
        assert (card["live"], card["input"]) == (False, False)
        refused = client.post(ASK, json={"slug": SLUG, "question": "а к борщу?"})
        assert refused.status_code == 422
        assert refused.json() == {"detail": "вопрос текстом выключен"}
        reasons = set()
        for chip in ("what_to_eat", "serve", "softer", "replace", "guided"):
            text = text_of(stream(client, {"slug": SLUG, "chip": chip}))
            assert text["generated"] is False and text["label"] == LABEL_ALGO
            reasons.add(text["reason"])
    assert "off" in reasons and reasons <= {"off", "not_voiced"}
    assert app.state.somm.voice(app, service).stats()["calls"] >= 1


def test_safety_switch_puts_the_layer_under_the_router(tmp_path: Path) -> None:
    """`SVS_SOMM_SAFETY=1` — маршрут спрашивает смысловой слой; без весов модели он молчит, а
    `/v1/health.somm.safety.status` пишет `missing`: вопросы идут по одним правилам."""
    somm = SommSettings(data_dir=tmp_path / "somm", safety=True)
    service = make_reco_service(tmp_path, somm=somm)
    app = create_app(service=service, warm=False)
    runtime = app.state.somm
    runtime.safety.model_name = "svs-test/no-such-model"  # только локальный кэш: не найдётся
    with TestClient(app) as client:
        assert runtime.router().safety is runtime.safety
        events = stream(client, {"slug": SLUG, "question": "а к борщу подойдёт?"})
        facts = next(event for event in events if event["type"] == "facts")
        assert facts["intent"] == "dish_check"
        refusal = stream(client, {"slug": SLUG, "question": "мне 16, к борщу?"})
        assert next(e for e in refusal if e["type"] == "facts")["intent"] == "refuse"
        safety = client.get("/v1/health").json()["somm"]["safety"]
    assert safety["enabled"] is True and safety["status"] == "missing"
    assert safety["refused"] == {}


def test_safety_off_keeps_the_router_rules_only(tmp_path: Path) -> None:
    service = make_reco_service(tmp_path)
    app = create_app(service=service, warm=False)
    with TestClient(app):
        assert app.state.somm.router().safety is None


def test_voice_without_vlm_is_not_ready(tmp_path: Path) -> None:
    service = make_service(settings=settings_for(tmp_path), vlm=False)
    app = create_app(service=service, warm=False)
    with TestClient(app):
        voice = app.state.somm.voice(app, service)
    assert voice.client is None and voice.live is True


# ------------------------------------------------------------------ после слияния
def test_interim_stand_ins_are_gone() -> None:
    assert importlib.util.find_spec("app.sommelier.interim") is None


def test_pack_ships_somm_and_never_the_portal() -> None:
    script = (REPO_ROOT / "deploy" / "pack_data.sh").read_text(encoding="utf-8")
    code = [line for line in script.splitlines() if line.strip() and not line.startswith("#")]
    added = [line.split()[1] for line in code if line.startswith("add ")]
    assert "somm" in added
    assert not any("portal" in item or "packshots_fixed" in item for item in added)
    assert not any("portal/" in line for line in code if "ITEMS" in line or "tar " in line)
