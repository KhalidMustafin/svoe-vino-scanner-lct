"""Живой голос сомелье (`app/sommelier/voice.py`) и клиент Ollama (`ollama_text.py`) без видеокарты.

Ollama — поддельная, на 127.0.0.1 (`somm_voice_env.FakeOllama`): поток NDJSON по токену, пауза
«обработки промпта», отказы. Проверяется порядок `speak` по договору (§6.3): причины шаблона,
этапы `voice` и `verify`, кэш, обрыв сканом и клиентом, отдача замка видеокарты при любом
исходе; что модель видит только пакет фактов; что в журнал не попадает ни текст модели, ни
факты. Отдельно — зонд `scripts/somm_probe.py` против той же подделки.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import pytest
from somm_voice_env import (
    FIXTURES,
    ChatPlan,
    FakeOllama,
    build_facts,
    dish_check_facts,
    facts_event,
    lock,
    probe,
    what_to_eat_facts,
    words_as_tokens,
)

from app.sommelier import answers, guards
from app.sommelier.gate import SommGate
from app.sommelier.ollama_text import MAX_NUM_PREDICT, OllamaText
from app.sommelier.voice import (
    LABEL_GENERATED,
    LABEL_TEMPLATE,
    MAX_INPUT_CHARS,
    REASONS,
    SYSTEM_PROMPT,
    Voice,
    VoiceResult,
    build_messages,
)

GOOD = (
    "К борщу — да, с оговоркой: вино и борщ равны по силе вкуса, но наваристый бульон сделает "
    "терпкость Саперави заметнее. Если хочется мягче — вот вина других виноделен."
)
BAD_NUMBER = "К борщу — да, с оговоркой: подавайте при 12 °C, бульон подчёркивает терпкость."
QUESTION = "а к борщу подойдёт?"
MSGS = [{"role": "system", "content": "с"}, {"role": "user", "content": "в"}]


@pytest.fixture
def fake():
    with FakeOllama(ChatPlan(tokens=words_as_tokens(GOOD))) as server:
        yield server


def open_gate(quiet_s: float = 0.0) -> tuple[SommGate, threading.Lock]:
    gate = SommGate(quiet_s=quiet_s)
    gpu = threading.Lock()
    gate.bind(gpu)
    return gate, gpu


def make_voice(fake, gate=None, **kw) -> Voice:
    gate = gate or open_gate()[0]
    client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
    kw.setdefault("timeout_s", 2.0)
    return Voice(gate, client, lock(), live=True, **kw)


def speak(voice: Voice, facts=None, **kw):
    return asyncio.run(voice.speak(facts or dish_check_facts(), **kw))


class Stages:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, stage: str) -> None:
        self.seen.append(stage)


# ============================================================ причины шаблона без модели
def test_не_пересказываемое_намерение():
    facts = build_facts({**facts_event("ask_guided_step.ndjson"), "voice": False})
    voice = Voice(open_gate()[0], None, lock(), live=True)
    result = speak(voice, facts)
    assert (result.reason, result.generated, result.label) == ("not_voiced", False, LABEL_TEMPLATE)
    assert result.text == facts.verdict_template


def test_обычная_сортировка():
    facts = build_facts(facts_event("ask_softer_plain.ndjson"))
    result = speak(Voice(open_gate()[0], None, lock(), live=True), facts)
    assert result.reason == "plain" and result.text == facts.verdict_template


def test_голос_выключен(fake):
    voice = make_voice(fake)
    voice.live = False
    assert speak(voice).reason == "off"
    assert fake.payloads == []


def test_нет_клиента_модели():
    assert speak(Voice(open_gate()[0], None, lock(), live=True)).reason == "not_ready"


def test_ворота_не_связаны(fake):
    assert speak(make_voice(fake, gate=SommGate())).reason == "not_ready"
    assert fake.payloads == []


@pytest.mark.parametrize("state", ["busy", "quiet", "locked"])
def test_ворота_не_впустили(fake, state):
    gate, gpu = open_gate(quiet_s=15)
    if state == "busy":
        gate.scan_started()
    elif state == "quiet":
        gate.predict_done()
    else:
        gpu.acquire()
    stages = Stages()
    result = speak(make_voice(fake, gate=gate), on_stage=stages)
    assert result.reason == state and not result.generated
    assert stages.seen == [] and fake.payloads == []


# ============================================================ живой текст
def test_живой_текст_проходит_проверки(fake):
    gate, gpu = open_gate()
    stages = Stages()
    voice = make_voice(fake, gate=gate)
    result = speak(voice, on_stage=stages)
    assert result == VoiceResult(GOOD, True, LABEL_GENERATED, None, None)
    assert stages.seen == ["voice", "verify"]
    assert not gpu.locked()


def test_модель_видит_только_пакет_фактов(fake):
    facts = dish_check_facts()
    speak(make_voice(fake), facts)
    (payload,) = fake.payloads
    system, user = payload["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["content"].startswith(f"Намерение: {facts.intent_label}\nФакты:\n- Вино: ")
    assert f"Готовый ответ: {facts.verdict_template}\n/no_think" in user["content"]
    assert QUESTION not in json.dumps(payload, ensure_ascii=False)
    assert "Описание" not in user["content"] and "context" not in user["content"]
    assert payload["think"] is False and payload["stream"] is True


def test_кэш_отдаёт_прошедший_текст_без_модели_и_этапов(fake):
    voice = make_voice(fake)
    assert speak(voice).generated
    stages = Stages()
    again = speak(voice, on_stage=stages)
    assert again.generated and again.text == GOOD and again.reason is None
    assert stages.seen == [] and len(fake.payloads) == 1
    assert voice.stats()["cache_hits"] == 1


def test_кэш_работает_и_когда_видеокарта_занята(fake):
    gate, _ = open_gate()
    voice = make_voice(fake, gate=gate)
    assert speak(voice).generated
    gate.scan_started()
    assert speak(voice).generated


def test_брак_проверки_даёт_шаблон_и_не_кэшируется(fake):
    fake.plan = ChatPlan(tokens=words_as_tokens(BAD_NUMBER))
    voice = make_voice(fake)
    stages = Stages()
    result = speak(voice, on_stage=stages)
    facts = dish_check_facts()
    assert result == VoiceResult(facts.verdict_template, False, LABEL_TEMPLATE, "guard", "numbers")
    assert stages.seen == ["voice", "verify"]
    speak(voice)
    assert len(fake.payloads) == 2
    assert voice.stats()["guards"] == {"numbers": 2}


def test_рассуждение_без_ответа_это_брак_think(fake):
    fake.plan = ChatPlan(tokens=(), thinking="Сначала подумаю о борще.")
    stages = Stages()
    result = speak(make_voice(fake), on_stage=stages)
    assert (result.reason, result.guard) == ("guard", "think")
    assert stages.seen == ["voice"]  # пустой content — этапа проверки нет


def test_обрыв_по_num_predict_это_брак_cutoff(fake):
    fake.plan = ChatPlan(tokens=words_as_tokens(GOOD), done_reason="length")
    assert speak(make_voice(fake)).guard == "cutoff"


def test_таймаут(fake):
    fake.plan = ChatPlan(tokens=words_as_tokens(GOOD), delay_first_s=5)
    gate, gpu = open_gate()
    started = time.perf_counter()
    result = speak(make_voice(fake, gate=gate, timeout_s=0.3))
    assert result.reason == "timeout"
    assert time.perf_counter() - started < 1.5
    assert not gpu.locked()


def test_нет_модели_и_пауза_без_вызовов(fake):
    """«Лоза»: недоступная модель даёт шаблон. Здесь ещё 30 с голос Ollama не зовёт."""
    fake.plan = ChatPlan(status=404)
    clock = [100.0]
    voice = make_voice(fake, clock=lambda: clock[0])
    assert speak(voice).reason == "unavailable"
    assert speak(voice).reason == "unavailable"
    assert len(fake.payloads) == 1
    clock[0] += 31
    fake.plan = ChatPlan(tokens=words_as_tokens(GOOD))
    assert speak(voice).generated


def test_имя_ollama_разрешается_до_замка_видеокарты(fake, monkeypatch):
    """`getaddrinfo` не оборвать: медленный DNS (`ollama` в compose) под замком держал бы скан.

    Голос разрешает имя до входа в ворота, запоминает адреса и соединяется по ним; отказ
    соединения сбрасывает запомненное. Имя не разрешилось — шаблон `unavailable`, ворот не
    трогали.
    """
    from app.sommelier import ollama_text

    gate, gpu = open_gate()
    held: list[bool] = []
    real = ollama_text.socket.getaddrinfo

    def watched(*args, **kwargs):
        held.append(gpu.locked())
        return real(*args, **kwargs)

    monkeypatch.setattr(ollama_text.socket, "getaddrinfo", watched)
    voice = make_voice(fake, gate)
    assert speak(voice).generated is True
    voice._cache.clear()
    assert speak(voice).generated is True
    assert held == [False], "имя разрешается один раз и вне замка"

    def broken(*args, **kwargs):
        raise OSError("имя не разрешилось")

    monkeypatch.setattr(ollama_text.socket, "getaddrinfo", broken)
    blind = make_voice(fake, gate)
    fake.payloads.clear()
    assert speak(blind).reason == "unavailable"
    assert fake.payloads == [] and gate.stats()["entered"] == 2 and not gpu.locked()


def test_поток_без_done_это_ошибка(fake):
    fake.plan = ChatPlan(tokens=words_as_tokens(GOOD), drop=True)
    assert speak(make_voice(fake)).reason == "error"


def test_ошибка_ollama_в_потоке(fake):
    fake.plan = ChatPlan(error="model runner has unexpectedly stopped")
    assert speak(make_voice(fake)).reason == "error"


def test_голос_не_бросает(fake):
    class Broken(OllamaText):
        def chat(self, *args, **kwargs):
            raise RuntimeError("сломался клиент")

    gate, gpu = open_gate()
    voice = Voice(gate, Broken(fake.url, "m", keep_alive="24h", num_ctx=4096), lock(), live=True)
    result = speak(voice)
    assert result.reason == "error" and result.text == dish_check_facts().verdict_template
    assert not gpu.locked()


# ============================================================ обрывы
def _slow(fake) -> None:
    fake.plan = ChatPlan(tokens=("слово ",) * 300, token_interval_s=0.02)


def test_скан_обрывает_голос(fake):
    _slow(fake)
    gate, gpu = open_gate()
    voice = make_voice(fake, gate=gate, timeout_s=10)

    async def scenario():
        task = asyncio.ensure_future(voice.speak(dish_check_facts()))
        await asyncio.to_thread(fake.streaming.wait, 5)
        gate.scan_started()
        return await task

    started = time.perf_counter()
    result = asyncio.run(scenario())
    assert result.reason == "preempted" and not result.generated
    assert time.perf_counter() - started < 2
    assert not gpu.locked()


def test_клиент_закрыл_поток(fake):
    _slow(fake)
    gate, gpu = open_gate()
    voice = make_voice(fake, gate=gate, timeout_s=10)

    async def gone() -> bool:
        return fake.streaming.is_set()

    result = speak(voice, disconnected=gone)
    assert result.reason == "cancelled"
    assert not gpu.locked()
    assert gate.stats()["cancelled"]["client"] == 1


def test_упавший_этап_значит_клиент_ушёл(fake):
    gate, gpu = open_gate()

    async def broken(stage: str) -> None:
        raise ConnectionResetError("поток закрыт")

    assert speak(make_voice(fake, gate=gate), on_stage=broken).reason == "cancelled"
    assert fake.payloads == [] and not gpu.locked()


def test_отмена_задачи_обрывает_слот_и_отдаёт_замок(fake):
    _slow(fake)
    gate, gpu = open_gate()
    voice = make_voice(fake, gate=gate, timeout_s=10)

    async def scenario():
        task = asyncio.ensure_future(voice.speak(dish_check_facts()))
        await asyncio.to_thread(fake.streaming.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    deadline = time.monotonic() + 2
    while gpu.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not gpu.locked()
    assert voice.stats()["reasons"] == {"cancelled": 1}


# ============================================================ вход модели
def test_вход_не_длиннее_потолка_срезаются_строки_вин():
    public = facts_event("ask_dish_check_live.ndjson")
    wine = public["wines"][0]
    many = [
        {**wine, "name": f"Пино Нуар {i}", "pill": "танины мягче — по сорту"} for i in range(80)
    ]
    facts = build_facts({**public, "wines": many})
    messages = build_messages(facts)
    total = sum(len(message["content"]) for message in messages)
    assert total <= MAX_INPUT_CHARS
    user = messages[1]["content"]
    assert "- Вино: Cru Lermont Saperavi" in user and "- Блюдо: Борщ" in user
    assert "Пино Нуар 79" not in user and facts.verdict_template in user


def test_потолок_входа_около_700_токенов_русского_текста():
    """≈ 3 символа на токен у русского текста в Qwen: 700 токенов — не больше 2 100 символов
    (перепроверка: прежние 2 800 были бы ≈ 900 токенов). Строкам фактов остаётся потолок без промпта."""
    assert MAX_INPUT_CHARS <= 700 * 3
    assert answers.MAX_VOICE_CHARS == MAX_INPUT_CHARS - len(SYSTEM_PROMPT) > 1000
    public = facts_event("ask_dish_check_live.ndjson")
    messages = build_messages(build_facts(public))
    user = messages[1]["content"]
    assert sum(len(message["content"]) for message in messages) <= MAX_INPUT_CHARS
    assert "- Вино: Cru Lermont Saperavi" in user and "Готовый ответ:" in user


def test_в_промпте_нет_цифр():
    """Цифровой замок сверяет ответ с пакетом: число из промпта было бы в ответе чужим."""
    assert not any(ch.isdigit() for ch in SYSTEM_PROMPT)
    assert "/no_think" in SYSTEM_PROMPT


def test_журнал_не_видит_ни_текста_модели_ни_фактов(fake, caplog):
    caplog.set_level(logging.DEBUG)
    fake.plan = ChatPlan(tokens=words_as_tokens(BAD_NUMBER))
    speak(make_voice(fake))
    fake.plan = ChatPlan(tokens=words_as_tokens(GOOD))
    speak(make_voice(fake))
    logged = " ".join(record.getMessage() for record in caplog.records)
    for fragment in ("бульон", "Саперави", "борщ", "12 °C", "Cru Lermont"):
        assert fragment not in logged


def test_коды_причин_из_договора():
    contract = (FIXTURES.parents[2] / "docs" / "api-sommelier.md").read_text(encoding="utf-8")
    for reason in REASONS:
        assert f"| `{reason}` |" in contract, reason


def test_справка_голоса(fake):
    voice = make_voice(fake)
    speak(voice)
    speak(voice)
    voice.live = False
    speak(voice)
    stats = voice.stats()
    assert stats["calls"] == 3 and stats["generated"] == 2 and stats["cache_hits"] == 1
    assert stats["reasons"] == {"off": 1} and stats["model"] == fake.model
    assert stats["last_call"]["status"] == "ok"


# ============================================================ клиент Ollama
def test_клиент_собирает_поток_и_сводку(fake):
    client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
    result = client.chat(MSGS, timeout_s=2, cancel=threading.Event())
    assert result.status == "ok" and result.content == GOOD
    assert result.done_reason == "stop" and result.prompt_eval_count == 300
    assert result.load_ms == 12 and result.first_token_ms is not None
    assert result.error is None


@pytest.mark.parametrize("phase", ["prompt", "tokens"])
def test_обрыв_будит_клиент_сразу(fake, phase):
    if phase == "prompt":
        fake.plan = ChatPlan(delay_first_s=10)
    else:
        _slow(fake)
    client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
    aborts: list = []
    out: dict = {}

    def run() -> None:
        out["result"] = client.chat(
            MSGS, timeout_s=15, cancel=threading.Event(), on_abort=aborts.append
        )
        out["at"] = time.perf_counter()

    thread = threading.Thread(target=run)
    thread.start()
    assert (fake.received if phase == "prompt" else fake.streaming).wait(5)
    started = time.perf_counter()
    aborts[0]()
    thread.join(5)
    assert out["result"].status == "cancelled" and out["result"].content == ""
    assert out["at"] - started < 0.2


def test_событие_cancel_без_будильника_тоже_обрывает(fake):
    _slow(fake)
    client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    started = time.perf_counter()
    assert client.chat(MSGS, timeout_s=10, cancel=cancel).status == "cancelled"
    assert time.perf_counter() - started < 1


def test_уже_оборванный_вызов_не_ходит_в_ollama(fake):
    client = OllamaText(fake.url, fake.model, keep_alive="24h", num_ctx=4096)
    cancel = threading.Event()
    cancel.set()
    assert client.chat(MSGS, timeout_s=2, cancel=cancel).status == "cancelled"
    assert fake.payloads == []


def test_порт_не_слушает():
    """Ollama не поднята: `unavailable`. На Windows отказ соединения идёт около 2 с."""
    client = OllamaText("http://127.0.0.1:9", "m", keep_alive="24h", num_ctx=4096)
    result = client.chat(MSGS, timeout_s=5, cancel=threading.Event())
    assert result.status == "unavailable" and result.error == "ConnectionRefusedError"


class _SlowRefusal:
    """Сокет, которому отказывают через `DELAY` с, как Windows на закрытом порту (около 2 с).

    Соединения нет вовсе — ни с 127.0.0.1, ни с портом Ollama: адреса даёт подставной резолвер.
    """

    DELAY = 0.3
    made = 0

    def __init__(self, family: int, kind: int, proto: int = 0) -> None:
        type(self).made += 1
        self.refused_at = 0.0

    def setblocking(self, flag: bool) -> None:
        pass

    def connect_ex(self, address: object) -> int:
        import errno

        self.refused_at = time.perf_counter() + self.DELAY
        return errno.EINPROGRESS

    def getsockopt(self, level: int, option: int) -> int:
        import errno

        return errno.ECONNREFUSED

    def close(self) -> None:
        pass


@pytest.fixture
def two_slow_addresses(monkeypatch):
    """Имя с двумя адресами (как `localhost` на Windows: ::1 и 127.0.0.1), и оба отказывают
    медленно. Подменяются только ссылки модуля клиента на `socket` и `select`."""
    import select as real_select_module
    import socket as real_socket
    import types

    from app.sommelier import ollama_text

    addresses = [
        (real_socket.AF_INET6, real_socket.SOCK_STREAM, 6, "", ("::1", 9, 0, 0)),
        (real_socket.AF_INET, real_socket.SOCK_STREAM, 6, "", ("127.0.0.1", 9)),
    ]
    fake_socket = types.SimpleNamespace(
        **{
            name: getattr(real_socket, name)
            for name in dir(real_socket)
            if not name.startswith("__")
        }
    )
    fake_socket.socket = _SlowRefusal
    fake_socket.getaddrinfo = lambda *args, **kwargs: list(addresses)

    def fake_select(read, write, broken, timeout):
        slow = [sock for sock in (*write, *broken) if isinstance(sock, _SlowRefusal)]
        if not slow:
            return real_select_module.select(read, write, broken, timeout)
        left = slow[0].refused_at - time.perf_counter()
        if left > timeout:
            time.sleep(timeout)
            return [], [], []
        time.sleep(max(0.0, left))
        return [], [slow[0]], []

    monkeypatch.setattr(ollama_text, "socket", fake_socket)
    monkeypatch.setattr(ollama_text, "select", types.SimpleNamespace(select=fake_select))
    _SlowRefusal.made = 0
    return addresses


def test_недоступный_хост_с_двумя_адресами_это_unavailable(two_slow_addresses):
    """Отказы двух адресов вместе дольше срока голоса: исход — `unavailable`, а не `timeout`
    (финальная проверка 24.09). Адреса забываются: контейнер Ollama мог сменить адрес."""
    client = OllamaText("http://ollama:9", "m", keep_alive="24h", num_ctx=4096)
    assert client.resolve() is True
    started = time.perf_counter()
    result = client.chat(MSGS, timeout_s=0.5, cancel=threading.Event())
    assert (result.status, result.error) == ("unavailable", "connect_timeout")
    assert time.perf_counter() - started < 1.0
    assert _SlowRefusal.made == 2  # первый адрес отказал, второй не успел
    assert client._addresses is None


def test_недоступный_хост_даёт_паузу_30_секунд(two_slow_addresses):
    """Голос после такого отказа 30 с не зовёт Ollama: ворота и замок видеокарты не заняты зря."""
    gate, gpu = open_gate()
    clock = [100.0]
    client = OllamaText("http://ollama:9", "m", keep_alive="24h", num_ctx=4096)
    voice = Voice(gate, client, lock(), live=True, timeout_s=0.5, clock=lambda: clock[0])
    assert speak(voice).reason == "unavailable"
    made = _SlowRefusal.made
    assert speak(voice).reason == "unavailable" and _SlowRefusal.made == made  # пауза
    assert voice.stats()["unavailable_left_s"] == 30.0 and not gpu.locked()
    clock[0] += 31
    assert speak(voice).reason == "unavailable" and _SlowRefusal.made > made  # снова пробует


def test_ответ_не_кусками_тоже_читается():
    """Ollama за прокси может ответить `Content-Length` или до закрытия — строки те же."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    lines = [
        {"message": {"content": "Ответ "}, "done": False},
        {"message": {"content": "готов."}, "done": True, "done_reason": "stop"},
    ]
    body = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaText(
            f"http://127.0.0.1:{server.server_address[1]}", "m", keep_alive="24h", num_ctx=4096
        )
        result = client.chat(MSGS, timeout_s=2, cancel=threading.Event())
    finally:
        server.shutdown()
        server.server_close()
    assert result.status == "ok" and result.content == "Ответ готов."


def test_адрес_и_длина_ответа_проверяются():
    with pytest.raises(ValueError):
        OllamaText("http://127.0.0.1:11434", "m", keep_alive="24h", num_ctx=4096, num_predict=161)
    with pytest.raises(ValueError):
        OllamaText("ftp://ollama", "m", keep_alive="24h", num_ctx=4096)
    client = OllamaText("http://127.0.0.1:11434/v1/", "m", keep_alive="24h", num_ctx=4096)
    assert client.url == "http://127.0.0.1:11434"
    assert client.payload(MSGS)["options"]["num_predict"] == MAX_NUM_PREDICT == 160


# ============================================================ зонд для окна видеокарты
def test_пакеты_зонда_из_заглушек():
    """Зонд мерит только то, что голос сервиса пересказывает: у заглушек «а к борщу?» оговорка и
    подборка — такие пакеты с 25.09 шаблон (`answers.voice_fits`), в замер они не идут."""
    packages = dict(probe.fixture_packages())
    intents = sorted(facts.public["intent"] for facts in packages.values())
    assert intents == ["what_to_eat"]
    assert all(facts.voiced for facts in packages.values())
    for name in ("ask_dish_check_live.ndjson", "ask_dish_check_guard.ndjson"):
        public = facts_event(name)
        assert public["voice"] and public["wines"] and not probe._fits(public)


def test_пакет_зонда_как_у_тестов():
    facts = what_to_eat_facts()
    assert facts.voice_input[:6] == (
        "Вино: Cru Lermont Saperavi",
        "Винодельня: Фанагория",
        "Регион: Кубань",
        "Стиль: красное сухое",
        "Сорта: Саперави",
        "Подача: 16–18 °C",
    )
    assert {"16", "18"} <= facts.allowed_numbers
    assert {"Гусь запечённый", "Фанагория", "Саперави"} <= facts.allowed_entities


def test_зонд_против_подделки(fake, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SVS_OLLAMA_URL", fake.url)
    monkeypatch.setenv("SVS_VLM_MODEL", fake.model)
    monkeypatch.setenv("SVS_SOMM_DIR", str(tmp_path))  # vocab.json нет — берётся заглушка
    out = tmp_path / "report.json"
    assert (
        probe.main(["--runs", "2", "--preempt", "2", "--preempt-after-ms", "0", "--out", str(out)])
        == 0
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    options = report["options"]
    assert all(options[key] for key in options if key.endswith(("_equal", "_options", "_false")))
    calls = 2 * len(probe.fixture_packages())
    assert report["summary"]["calls"] == calls and report["summary"]["answered"] == calls
    expected = sum(
        guards.check(GOOD, facts, lock()) is None for _, facts in probe.fixture_packages()
    )
    assert report["summary"]["passed"] == 2 * expected
    assert report["summary"]["reloads"] == 0
    assert report["ps"]["loaded_before"] and report["ps"]["expires_not_shortened"]
    assert report["preempt"]["cancel_return_ms"]["n"] == 2
    assert "text" in report["calls"][0]
    assert "прошли проверки" in capsys.readouterr().out


def test_зонд_насквозь_через_маршруты_сомелье():
    """Режим `--service`: заглушка маршрутов отдаёт карточку и поток из заглушек договора."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    card = (FIXTURES / "sommelier_red.json").read_bytes()
    stream = (FIXTURES / "ask_dish_check_live.ndjson").read_bytes()
    asked: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, kind: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._send(card, "application/json")

        def do_POST(self) -> None:
            asked.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._send(stream, "application/x-ndjson")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        records = probe.run_service(base, ["slug"], timeout_s=2)
    finally:
        server.shutdown()
        server.server_close()
    chips = json.loads(card)["chips"]
    assert [body["chip"] for body in asked] == [chip["id"] for chip in chips]
    summary = probe.summarize_service(records)
    assert summary["asks"] == len(chips) and summary["errors"] == 0
    assert summary["pass_rate"] == 1.0
    assert records[0]["stages"] == ["card", "rules", "catalog", "voice", "verify"]
