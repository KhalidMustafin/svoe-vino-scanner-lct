"""Фейки голоса сомелье: пакет фактов по заглушкам договора и поддельная Ollama на 127.0.0.1.

`build_facts` собирает подобие `FactsPackage` (договор, §6.4) из события `facts` заглушек
`tests/fixtures/somm/ask_*.ndjson` и карточки якоря: строки фактов для модели, имена, числа и
общий текст пакета. Сборщик — из зонда `scripts/somm_probe.py`: тесты проверяют ровно то, что
зонд мерит в окне видеокарты. Настоящий пакет собирает `app/sommelier/answers.py`; проверкам и
голосу нужны только эти поля (`guards.FactsLike`).

`FakeOllama` — HTTP-сервер на свободном порту: `/api/chat` отвечает потоком NDJSON по строке на
токен, как Ollama, с задержкой «обработки промпта» и паузой между токенами; `/api/ps` и
`/api/tags` — список моделей. Сеть наружу, модель и видеокарта не нужны.

Модуль не собирается pytest (имя не начинается с `test_`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

from app.sommelier.entity_lock import EntityLock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "somm_probe.py"
MODEL = "qwen3.5:4b"


def _probe() -> Any:
    """Зонд `scripts/somm_probe.py`: его сборщик пакета — один на тесты и замер в окне GPU."""
    module = sys.modules.get("somm_probe")
    if module is None:
        spec = importlib.util.spec_from_file_location("somm_probe", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


probe = _probe()
FIXTURES: Path = probe.FIXTURES
Facts = probe.ProbeFacts


def events(name: str) -> list[dict[str, Any]]:
    path = FIXTURES / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def facts_event(name: str) -> dict[str, Any]:
    """Тело события `facts` заглушки — `FactsPackage.public`."""
    (event,) = [e for e in events(name) if e["type"] == "facts"]
    return {k: v for k, v in event.items() if k not in {"type", "t_ms"}}


def lock() -> EntityLock:
    return EntityLock.from_mapping(json.loads((FIXTURES / "vocab.json").read_text("utf-8")))


def build_facts(
    public: Mapping[str, Any],
    anchor: Mapping[str, Any] | None = None,
    *,
    template: str | None = None,
) -> Any:
    """Пакет фактов по телу `facts` и карточке якоря (договор, §6.4) — сборщик зонда."""
    return probe.facts_from_event(public, anchor or probe.fixture_anchor(), template=template)


def dish_check_facts() -> Any:
    """«А к борщу подойдёт?» к Cru Lermont Saperavi: оговорка и три Пино Нуар помягче."""
    return build_facts(facts_event("ask_dish_check_live.ndjson"))


def what_to_eat_facts() -> Any:
    """«К чему подать» Cru Lermont Saperavi: три блюда без подборки вин."""
    return build_facts(facts_event("ask_what_to_eat_busy.ndjson"))


# ---------------------------------------------------------------------- поддельная Ollama
@dataclass
class ChatPlan:
    """Как отвечает `/api/chat`: токены, паузы, итог."""

    tokens: Sequence[str] = ("Ответ.",)
    thinking: str = ""
    #: «Обработка промпта»: столько сервер молчит до заголовков ответа.
    delay_first_s: float = 0.0
    #: Пауза между строками потока.
    token_interval_s: float = 0.0
    done_reason: str = "stop"
    status: int = 200
    load_ms: int = 12
    #: Строка с `error` вместо ответа.
    error: str | None = None
    #: Оборвать поток без `done`.
    drop: bool = False


@dataclass
class FakeOllama:
    """Ollama на 127.0.0.1 со счётом запросов и обрывов со стороны клиента."""

    plan: ChatPlan = field(default_factory=ChatPlan)
    model: str = MODEL
    payloads: list[dict[str, Any]] = field(default_factory=list)
    received: threading.Event = field(default_factory=threading.Event)
    streaming: threading.Event = field(default_factory=threading.Event)
    disconnects: int = 0
    finished: int = 0
    #: Поколение прогона: обработчик прошлого запроса ещё пишет в мёртвый сокет и не должен
    #: ставить события следующего.
    generation: int = 0
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None
    _stopping: threading.Event = field(default_factory=threading.Event)

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def reset(self, plan: ChatPlan | None = None) -> None:
        if plan is not None:
            self.plan = plan
        self.generation += 1
        self.received.clear()
        self.streaming.clear()

    def __enter__(self) -> Self:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, status: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_GET(self) -> None:
                if self.path in {"/api/ps", "/api/tags"}:
                    self._json(200, {"models": [{"name": fake.model, "model": fake.model,
                                                 "expires_at": "2026-09-25T12:00:00Z",
                                                 "size_vram": 1}]})  # fmt: skip
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.payloads.append(payload)
                generation = fake.generation
                fake.received.set()
                plan = fake.plan
                if self.path != "/api/chat" or plan.status != 200:
                    self._json(plan.status if plan.status != 200 else 404, {"error": "нет модели"})
                    return
                if not fake._sleep(plan.delay_first_s):
                    return
                self.close_connection = True
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for line in fake._lines(plan, payload.get("model", fake.model)):
                        data = (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")
                        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()
                        if generation == fake.generation:
                            fake.streaming.set()
                        if not fake._sleep(plan.token_interval_s):
                            return
                    if not plan.drop:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    fake.finished += 1
                except OSError:
                    fake.disconnects += 1

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def _sleep(self, seconds: float) -> bool:
        """Пауза, прерываемая остановкой сервера; `False` — сервер останавливается."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stopping.wait(min(0.01, max(0.0, end - time.monotonic()))):
                return False
        return not self._stopping.is_set()

    def _lines(self, plan: ChatPlan, model: str) -> Iterator[dict[str, Any]]:
        if plan.error:
            yield {"error": plan.error}
            return
        if plan.thinking:
            message = {"role": "assistant", "content": "", "thinking": plan.thinking}
            yield {"model": model, "message": message, "done": False}
        for token in plan.tokens:
            yield {"model": model, "message": {"role": "assistant", "content": token},
                   "done": False}  # fmt: skip
        if plan.drop:
            return
        yield {
            "model": model,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": plan.done_reason,
            "eval_count": len(plan.tokens),
            "prompt_eval_count": 300,
            "load_duration": plan.load_ms * 1_000_000,
        }


def words_as_tokens(text: str) -> tuple[str, ...]:
    """Текст потоком «по токену»: слово с пробелом за ним."""
    parts = text.split(" ")
    return tuple(part + (" " if i < len(parts) - 1 else "") for i, part in enumerate(parts))
