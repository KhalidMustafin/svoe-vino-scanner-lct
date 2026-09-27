"""Ворота видеокарты для голоса сомелье (`app/sommelier/gate.py`) и хуки в `app/api/main.py`.

Скан всегда первый: голос не входит при скане и в тихом окне после predict, берёт замок только
без ожидания и уходит по первому скану. Приёмка дорожки A: с поддельной медленной Ollama скан
ждёт замок видеокарты не дольше 300 мс (p95 на 50 прогонах), пока голос держит её, — и в фазе
«обработки промпта», и посреди потока токенов.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from api_env import RED, image_bytes, make_service
from api_env import FakeOllama as ReaderOllama
from fastapi.testclient import TestClient
from somm_voice_env import (
    ChatPlan,
    FakeOllama,
    dish_check_facts,
    lock,
    words_as_tokens,
)

from app.api.field import FieldSettings
from app.api.main import create_app
from app.sommelier.gate import SommGate
from app.sommelier.ollama_text import OllamaText
from app.sommelier.voice import Voice


class Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def bound(quiet_s: float = 15.0) -> tuple[SommGate, threading.Lock, Clock]:
    clock = Clock()
    gate = SommGate(quiet_s=quiet_s, clock=clock)
    gpu = threading.Lock()
    gate.bind(gpu)
    return gate, gpu, clock


# ============================================================ вход и отказы
def test_до_связи_с_сервисом_голос_не_входит():
    gate = SommGate()
    assert gate.try_enter() == (None, "not_ready")


def test_свободные_ворота_дают_слот_и_держат_замок():
    gate, gpu, _ = bound()
    slot, reason = gate.try_enter()
    assert slot is not None and reason is None
    assert gpu.locked()
    slot.release()
    assert not gpu.locked()


def test_второй_голос_не_входит_пока_первый_держит_слот():
    gate, _, _ = bound()
    slot, _ = gate.try_enter()
    assert gate.try_enter() == (None, "locked")
    slot.release()
    second, reason = gate.try_enter()
    assert second is not None and reason is None
    second.release()


def test_при_скане_голос_не_входит():
    gate, gpu, _ = bound()
    gate.scan_started()
    assert gate.try_enter() == (None, "busy")
    assert not gpu.locked()
    gate.scan_finished()
    slot, reason = gate.try_enter()
    assert slot is not None and reason is None
    slot.release()


def test_два_скана_держат_ворота_до_конца_последнего():
    gate, _, _ = bound()
    gate.scan_started()
    gate.scan_started()
    gate.scan_finished()
    assert gate.try_enter() == (None, "busy")
    gate.scan_finished()
    slot, _ = gate.try_enter()
    assert slot is not None
    slot.release()


def test_счёт_сканов_не_уходит_в_минус():
    """Лишний `scan_finished` не должен открыть ворота во время следующего скана."""
    gate, _, _ = bound()
    gate.scan_finished()
    gate.scan_finished()
    assert gate.active_scans == 0
    gate.scan_started()
    assert gate.try_enter() == (None, "busy")


def test_тихое_окно_после_predict():
    gate, _, clock = bound(quiet_s=15)
    gate.predict_done()
    assert gate.try_enter() == (None, "quiet")
    clock.now += 14.9
    assert gate.try_enter() == (None, "quiet")
    clock.now += 0.2
    slot, reason = gate.try_enter()
    assert slot is not None and reason is None
    slot.release()


def test_каждый_predict_продлевает_тихое_окно():
    gate, _, clock = bound(quiet_s=15)
    gate.predict_done()
    clock.now += 10
    gate.predict_done()
    clock.now += 10
    assert gate.try_enter() == (None, "quiet")
    clock.now += 5.1
    slot, _ = gate.try_enter()
    assert slot is not None
    slot.release()


def test_скан_важнее_тихого_окна_в_причине():
    gate, _, _ = bound()
    gate.predict_done()
    gate.scan_started()
    assert gate.try_enter() == (None, "busy")


def test_занятый_замок_голос_не_ждёт():
    """Замок держит прогрев или скан — голос уходит сразу, а не встаёт в очередь."""
    gate, gpu, _ = bound()
    gpu.acquire()
    try:
        started = time.perf_counter()
        assert gate.try_enter() == (None, "locked")
        assert time.perf_counter() - started < 0.05
    finally:
        gpu.release()


# ============================================================ обрыв
def test_скан_обрывает_голос_и_зовёт_колбэк_один_раз():
    gate, _, _ = bound()
    slot, _ = gate.try_enter()
    calls = []
    slot.on_cancel(lambda: calls.append("сокет"))
    gate.scan_started()
    assert slot.cancelled.is_set() and slot.reason == "scan"
    gate.scan_started()
    assert calls == ["сокет"]
    slot.release()
    assert gate.stats()["cancelled"]["scan"] == 1


def test_колбэк_после_обрыва_зовётся_сразу():
    gate, _, _ = bound()
    slot, _ = gate.try_enter()
    gate.preempt("client")
    calls = []
    slot.on_cancel(lambda: calls.append(1))
    assert calls == [1] and slot.reason == "client"
    slot.release()


def test_упавший_колбэк_не_роняет_скан():
    gate, _, _ = bound()
    slot, _ = gate.try_enter()

    def boom() -> None:
        raise RuntimeError("колбэк упал")

    slot.on_cancel(boom)
    gate.scan_started()  # не бросает
    assert slot.cancelled.is_set()
    slot.release()


def test_колбэк_может_сразу_отдать_слот():
    """Мьютекс ворот реентерабельный: отдача слота из колбэка обрыва не встаёт в самоблокировку."""
    gate, gpu, _ = bound()
    slot, _ = gate.try_enter()
    slot.on_cancel(slot.release)
    done = threading.Event()
    threading.Thread(target=lambda: (gate.scan_started(), done.set()), daemon=True).start()
    assert done.wait(2)
    assert not gpu.locked() and slot.released


def test_отдача_слота_повторно_ничего_не_делает():
    gate, gpu, _ = bound()
    slot, _ = gate.try_enter()
    with slot:
        pass
    slot.release()
    assert not gpu.locked()
    assert gate.stats()["released"] == 1


def test_отданный_слот_не_обрывается():
    gate, _, _ = bound()
    slot, _ = gate.try_enter()
    slot.release()
    assert slot.cancel("scan") is False
    assert not slot.cancelled.is_set()


def test_ворота_не_бросают_на_чужом_замке():
    """Кто-то отдал замок за голос — ворота пишут в журнал, но не бросают в путь predict."""
    gate, gpu, _ = bound()
    slot, _ = gate.try_enter()
    gpu.release()
    slot.release()  # RuntimeError внутри поймана
    assert gate.try_enter()[0] is not None


def test_из_многих_потоков_входит_один():
    gate, _, _ = bound()
    results = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        results.append(gate.try_enter())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    slots = [slot for slot, _ in results if slot is not None]
    assert len(slots) == 1
    assert sorted(reason for _, reason in results if reason) == ["locked"] * 7
    slots[0].release()


def test_справка_ворот():
    gate, _, clock = bound(quiet_s=15)
    slot, _ = gate.try_enter()
    clock.now += 0.25
    gate.scan_started()
    clock.now += 0.05
    slot.release()
    gate.scan_finished()
    gate.predict_done()
    stats = gate.stats()
    assert stats["bound"] and stats["quiet_s"] == 15
    assert stats["active_scans"] == 0 and not stats["voice_active"]
    assert stats["scans"] == 1 and stats["predicts"] == 1 and stats["entered"] == 1
    assert stats["quiet_left_s"] == 15
    assert stats["hold_ms"]["last"] == pytest.approx(300.0)
    assert stats["release_after_cancel_ms"]["last"] == pytest.approx(50.0)


# ============================================================ приёмка: скан ждёт ≤ 300 мс
RUNS = 50
GOOD = (
    "К борщу — да, с оговоркой: вино и борщ равны по силе вкуса, но наваристый бульон сделает "
    "терпкость Саперави заметнее. Если хочется мягче — вот вина других виноделен."
)


class VoiceLoop:
    """Голос в своём цикле событий — как в сервисе, где его зовёт маршрут `ask`."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def speak(self, voice: Voice, facts) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(voice.speak(facts), self.loop)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(2)


def _scan_wait_ms(gate: SommGate, gpu: threading.Lock) -> float:
    """Скан сервиса: хук `run_scan`, затем замок видеокарты с таймаутом, как в `_search`."""
    gate.scan_started()
    started = time.perf_counter()
    try:
        assert gpu.acquire(timeout=5.0), "скан не дождался замка"
        waited = (time.perf_counter() - started) * 1000
        gpu.release()
        return waited
    finally:
        gate.scan_finished()


@pytest.mark.parametrize(
    ("phase", "plan"),
    [
        # Ollama ещё считает промпт: заголовков ответа нет, клиент ждёт в `select`.
        ("prompt", ChatPlan(tokens=words_as_tokens(GOOD), delay_first_s=10.0)),
        # Модель пишет токен раз в 30 мс — ответ шёл бы 10 с.
        ("tokens", ChatPlan(tokens=("слово ",) * 330, token_interval_s=0.03)),
    ],
)
def test_скан_ждёт_замок_не_дольше_300_мс_пока_говорит_голос(phase, plan):
    gate = SommGate(quiet_s=0)
    gpu = threading.Lock()
    gate.bind(gpu)
    facts = dish_check_facts()
    loop = VoiceLoop()
    waits, reasons = [], []
    try:
        with FakeOllama(plan) as fake:
            client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
            for _ in range(RUNS // 2):
                fake.reset()
                voice = Voice(gate, client, lock(), live=True, timeout_s=15.0, cache_size=0)
                future = loop.speak(voice, facts)
                started = fake.streaming if phase == "tokens" else fake.received
                assert started.wait(5), "голос не дошёл до Ollama"
                assert gpu.locked(), "голос должен держать замок видеокарты"
                waits.append(_scan_wait_ms(gate, gpu))
                result = future.result(5)
                reasons.append(result.reason)
                assert not result.generated and result.text == facts.verdict_template
    finally:
        loop.close()
    waits.sort()
    p95 = waits[round(0.95 * len(waits)) - 1]
    assert p95 <= 300, f"p95 ожидания скана {p95:.1f} мс: {waits}"
    assert set(reasons) == {"preempted"}
    assert not gpu.locked()


def test_пятьдесят_прогонов_приёмки():
    """Две фазы по 25 прогонов — вместе 50, как в приёмке дорожки A."""
    assert RUNS == 50


# ============================================================ хуки в app/api/main.py
class ScanWatch(ReaderOllama):
    """Транспорт читателя, который смотрит на ворота в момент чтения этикетки."""

    def __init__(self, text: str = "Бета Холмы\nМерло") -> None:
        super().__init__(text)
        self.app = None
        self.seen: list[int] = []

    def __call__(self, payload, *, timeout_s):
        self.seen.append(self.app.state.somm_gate.active_scans)
        return super().__call__(payload, timeout_s=timeout_s)


def _post(client: TestClient, path: str):
    return client.post(path, files={"image": ("q.png", image_bytes(RED), "image/png")})


def test_ворота_создаются_с_приложением_и_связываются_в_lifespan():
    service = make_service()
    app = create_app(service=service, warm=False)
    gate = app.state.somm_gate
    assert isinstance(gate, SommGate) and not gate.bound
    with TestClient(app):
        assert gate.bound
        slot, reason = gate.try_enter()
        assert reason is None and service.gpu_lock.locked()
        slot.release()


@pytest.mark.parametrize("path", ["/v1/eval/predict", "/v1/scan"])
def test_скан_считается_на_время_run_scan(path):
    watch = ScanWatch()
    service = make_service(watch)
    app = create_app(service=service, warm=False)
    watch.app = app
    with TestClient(app) as client:
        assert _post(client, path).status_code == 200
    assert watch.seen == [1]
    assert app.state.somm_gate.active_scans == 0


def test_тихое_окно_ставит_только_predict():
    service = make_service()
    app = create_app(service=service, warm=False)
    gate = app.state.somm_gate
    with TestClient(app) as client:
        _post(client, "/v1/scan")
        slot, reason = gate.try_enter()
        assert reason is None
        slot.release()
        _post(client, "/v1/eval/predict")
        assert gate.try_enter() == (None, "quiet")
    assert gate.stats()["predicts"] == 1


def test_тихое_окно_и_при_упавшем_predict(monkeypatch):
    service = make_service()
    app = create_app(service=service, warm=False)
    monkeypatch.setattr(service, "scan", lambda data: 1 / 0)
    with TestClient(app) as client:
        assert _post(client, "/v1/eval/predict").json()["slug"] is None
    assert app.state.somm_gate.stats()["predicts"] == 1
    assert app.state.somm_gate.active_scans == 0


def test_счёт_сканов_отпускается_и_без_кадра():
    service = make_service()
    app = create_app(service=service, warm=False)
    with TestClient(app) as client:
        assert client.post("/v1/eval/predict", data={"x": "1"}).status_code == 200
    assert app.state.somm_gate.active_scans == 0


def test_скан_через_приложение_обрывает_голос_и_получает_замок():
    service = make_service()
    app = create_app(service=service, warm=False)
    gate = app.state.somm_gate
    with TestClient(app) as client:
        slot, _ = gate.try_enter()
        # Поток голоса отдаёт слот, как только сокет к Ollama закрыт.
        slot.on_cancel(lambda: threading.Timer(0.05, slot.release).start())
        body = _post(client, "/v1/scan").json()
    assert slot.cancelled.is_set() and slot.reason == "scan"
    assert body["slug"] is not None
    assert not service.gpu_lock.locked()


def test_полевой_приём_кадра_тоже_скан_но_без_тихого_окна(tmp_path):
    watch = ScanWatch()
    service = make_service(watch)
    field = FieldSettings(directory=tmp_path / "field", enabled=True)
    app = create_app(service=service, warm=False, field=field)
    watch.app = app
    gate = app.state.somm_gate
    with TestClient(app) as client:
        slot, _ = gate.try_enter()
        slot.on_cancel(slot.release)
        assert _post(client, "/v1/field/scan").status_code == 200
        assert slot.reason == "scan"
        free, reason = gate.try_enter()
        assert reason is None
        free.release()
    assert watch.seen == [1]
    assert gate.active_scans == 0 and gate.stats()["predicts"] == 0


def test_без_полевого_контура_обёртки_нет():
    app = create_app(service=make_service(), warm=False)
    assert not any(m.cls.__name__ == "FieldScanGate" for m in app.user_middleware)
