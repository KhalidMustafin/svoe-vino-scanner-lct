"""Живой голос через родной `/api/chat` Ollama: поток токенов, мгновенный обрыв, опции читателя.

Перенос родной ветки клиента «Лозы» (`Code/backend/app/llm/client.py:107-154`) с тремя
отличиями, и каждое — ради скана:

- **опции зеркалят читателя этикетки.** Модель, `keep_alive` и `num_ctx` берутся у живого
  `OllamaVlmReader` (`mirroring`), ключи `options` те же, что у него. Другой `num_ctx` или
  параметр загрузки (`num_gpu`, `num_batch`…) заставил бы Ollama перезагрузить модель, и
  следующий скан заплатил бы 3–4 с. Прогрева раз в 210 с и OpenAI-слоя «Лозы» здесь нет;
- **поток, а не один ответ** (`stream: true`): генерацию можно оборвать между токенами и даже
  во время обработки промпта. Колбэк ворот будит поток голоса, тот закрывает сокет, Ollama видит
  разрыв и отдаёт слот, а `chat` сразу возвращает `cancelled`, не дожидаясь конца ответа;
- **HTTP/1.1 прямо на сокете стандартной библиотеки.** `http.client` здесь не годится: на
  Windows `socket.shutdown` из чужого потока не будит заблокированный `recv` (замер: 2,8 с до
  ответа сервера), а поток `http.client` после таймаута сокета непригоден, и опрашивать его
  короткими таймаутами нельзя. Поэтому поток голоса ждёт в `select` сразу сокет Ollama и
  будильник (пару сокетов): байт в будильнике — обрыв, и сокет закрывает тот же поток, что его
  читает. Соединение тоже неблокирующее: пока Ollama не слушает порт, Windows повторяет SYN
  около 2 с, и скан не должен ждать их под замком видеокарты. По той же причине имя хоста
  (`ollama` в compose) разрешается заранее (`resolve`, голос зовёт его до входа в ворота), а не
  при соединении: `getaddrinfo` не оборвать, и медленный DNS держал бы замок.

`think: false` обязателен. Пустой `content` при непустом `thinking` — сбой `thinking_only`, как
у читателя (`ollama_vlm.py`, `classify`): думать модель не должна, и такой ответ не текст.

Текст модели и сообщения клиент в журнал не пишет: в `error` результата — только класс сбоя.
"""

from __future__ import annotations

import errno
import json
import select
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

#: Потолок длины ответа: голос пересказывает 2–3 фразы, а больше токенов — дольше держит слот.
MAX_NUM_PREDICT = 160
#: Одна строка потока или заголовка Ollama — маленькая; длиннее — не Ollama.
MAX_LINE_BYTES = 64 * 1024
#: Как часто поток голоса сверяет `cancel` и срок, если его не будят (`on_abort` не задан).
POLL_S = 0.1
_CONNECTING = {
    0,
    errno.EINPROGRESS,
    errno.EWOULDBLOCK,
    errno.EALREADY,
    getattr(errno, "WSAEWOULDBLOCK", -1),
}

ChatStatus = Literal["ok", "cancelled", "timeout", "unavailable", "error", "thinking_only"]


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Итог одного вызова. `content` — сырой текст модели, его судят проверки голоса."""

    status: ChatStatus
    content: str
    thinking_chars: int
    done_reason: str | None
    eval_count: int | None
    prompt_eval_count: int | None
    elapsed_ms: int
    #: От запроса до первого токена (`content` или `thinking`); замер зонда.
    first_token_ms: int | None = None
    #: `load_duration` Ollama: больше секунды — модель грузилась заново, опции разошлись.
    load_ms: int | None = None
    #: Класс сбоя без текста модели: «HTTP 404», «ConnectionRefusedError», «no_done».
    error: str | None = None


class _Stop(Exception):
    """Ожидание прервано: обрыв (`cancel`) или срок (`timeout`)."""

    def __init__(self, why: str) -> None:
        super().__init__(why)
        self.why = why


class _Abort:
    """Будильник обрыва: колбэк ворот пишет байт в пару сокетов, поток голоса ждёт на ней.

    Первый обрыв задаёт причину; после `close()` обрыв ничего не делает.
    """

    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self.reader, self._writer = socket.socketpair()
        self.reader.setblocking(False)
        self._writer.setblocking(False)
        self._closed = False
        self.why: str | None = None

    def cancel(self) -> None:
        with self._mutex:
            if self._closed or self.why is not None:
                return
            self.why = "cancel"
            try:
                self._writer.send(b"x")
            except OSError:
                pass  # будильник переполнен или закрыт — причина уже записана

    def close(self) -> None:
        with self._mutex:
            self._closed = True
            self.reader.close()
            self._writer.close()


class _Wire:
    """HTTP/1.1 поверх неблокирующего сокета: каждое ожидание — `select` с будильником."""

    def __init__(self, deadline: float, cancel: threading.Event, abort: _Abort) -> None:
        self.deadline = deadline
        self.cancel = cancel
        self.abort = abort
        self.sock: socket.socket | None = None
        self.buf = bytearray()

    def _check(self) -> float:
        if self.abort.why is not None or self.cancel.is_set():
            raise _Stop("cancel")
        remaining = self.deadline - time.perf_counter()
        if remaining <= 0:
            raise _Stop("timeout")
        return remaining

    def _wait(self, *, write: bool = False) -> None:
        sock = self.sock
        assert sock is not None
        while True:
            timeout = min(self._check(), POLL_S)
            if write:
                ready, writable, broken = select.select(
                    [self.abort.reader], [sock], [sock], timeout
                )
            else:
                ready, writable, broken = select.select([sock, self.abort.reader], [], [], timeout)
            self._check()
            if ready or writable or broken:
                return

    def connect(self, host: str, port: int, addresses: Sequence[Any] | None = None) -> None:
        """Неблокирующее соединение по адресам имени по очереди, как `socket.create_connection`.

        `addresses` — уже разрешённые адреса (`OllamaText.resolve`); нет их — `getaddrinfo` здесь.
        """
        code = -1
        if addresses is None:
            addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        for family, kind, proto, _, address in addresses:
            self.sock = socket.socket(family, kind, proto)
            self.sock.setblocking(False)
            code = self.sock.connect_ex(address)
            if code and code in _CONNECTING:
                self._wait(write=True)
                code = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if code == 0:
                return
            self.close()
        raise ConnectionRefusedError(code, "соединение с Ollama не установлено")

    def send(self, data: bytes) -> None:
        assert self.sock is not None
        view = memoryview(data)
        while view:
            self._wait(write=True)
            try:
                sent = self.sock.send(view)
            except BlockingIOError:
                continue
            view = view[sent:]

    def _fill(self) -> None:
        assert self.sock is not None
        self._wait()
        try:
            data = self.sock.recv(65536)
        except BlockingIOError:
            return
        if not data:
            raise EOFError("соединение закрыто")
        self.buf += data

    def line(self) -> bytes:
        while True:
            end = self.buf.find(b"\n")
            if end >= 0:
                line = bytes(self.buf[: end + 1])
                del self.buf[: end + 1]
                return line
            if len(self.buf) > MAX_LINE_BYTES:
                raise ValueError("строка длиннее предела")
            self._fill()

    def exact(self, size: int) -> bytes:
        while len(self.buf) < size:
            self._fill()
        data = bytes(self.buf[:size])
        del self.buf[:size]
        return data

    def some(self) -> bytes:
        if not self.buf:
            self._fill()
        data = bytes(self.buf)
        self.buf.clear()
        return data

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


@dataclass
class _Stream:
    content: str = ""
    thinking: str = ""
    done_reason: str | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    load_ms: int | None = None
    first_token_ms: int | None = None
    error: str | None = None


class OllamaText:
    """Клиент текста к родному `/api/chat` с опциями читателя этикетки."""

    def __init__(
        self,
        url: str,
        model: str,
        *,
        keep_alive: str,
        num_ctx: int,
        num_predict: int = MAX_NUM_PREDICT,
        temperature: float = 0.3,
    ) -> None:
        if not 1 <= num_predict <= MAX_NUM_PREDICT:
            raise ValueError(f"num_predict от 1 до {MAX_NUM_PREDICT}, получено {num_predict}")
        base = url.rstrip("/").removesuffix("/v1")
        parts = urllib.parse.urlsplit(base)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError(f"адрес Ollama должен быть http://хост:порт, получено {url!r}")
        self.url = base
        self.model = model
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.temperature = temperature
        self._host = parts.hostname
        self._port = parts.port or 80
        self._path = parts.path.rstrip("/") + "/api/chat"
        #: Адреса хоста Ollama: разрешаются раз (`resolve`), сбрасываются при отказе соединения.
        self._addresses: list[Any] | None = None

    @classmethod
    def mirroring(cls, reader: Any, *, num_predict: int = MAX_NUM_PREDICT) -> OllamaText:
        """Клиент с адресом, моделью, `keep_alive` и `num_ctx` живого читателя этикетки.

        `reader` — `OllamaVlmReader` или обёртка сервиса под замком (`LockedReader`, у неё
        читатель в `.reader`).
        """
        inner = getattr(reader, "reader", reader)
        return cls(
            inner.url,
            inner.model,
            keep_alive=inner.keep_alive,
            num_ctx=inner.num_ctx,
            num_predict=num_predict,
        )

    def resolve(self) -> bool:
        """Разрешить имя хоста заранее — вне замка видеокарты. `False` — имя не разрешилось.

        Результат запоминается: второй вопрос DNS не ждёт. Отказ соединения сбрасывает его —
        контейнер Ollama мог перезапуститься с другим адресом.
        """
        if self._addresses is not None:
            return True
        try:
            self._addresses = list(
                socket.getaddrinfo(self._host, self._port, type=socket.SOCK_STREAM)
            )
        except OSError:
            return False
        return bool(self._addresses)

    def options(self) -> dict[str, Any]:
        """Ключи — ровно те, что у читателя: без параметров загрузки модели."""
        return {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

    def payload(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": str(message["role"]), "content": str(message["content"])}
                for message in messages
            ],
            "options": self.options(),
        }

    def request_bytes(self, messages: Sequence[Mapping[str, str]]) -> bytes:
        body = json.dumps(self.payload(messages), ensure_ascii=False).encode("utf-8")
        head = (
            f"POST {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Content-Type: application/json\r\n"
            "Accept: application/x-ndjson\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        return head.encode("ascii") + body

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_s: float,
        cancel: threading.Event,
        on_abort: Callable[[Callable[[], None]], None] | None = None,
    ) -> ChatResult:
        """Один ответ модели. Исключений нет: любой сбой — статус.

        `timeout_s` — предел от запроса до последнего токена. `cancel` сверяется не реже раза в
        `POLL_S`. `on_abort` получает функцию обрыва — голос регистрирует её в слоте ворот
        (`slot.on_cancel`), и скан будит поток голоса мгновенно, а не через `POLL_S`.
        """
        started = time.perf_counter()
        state = _Stream()
        abort = _Abort()
        wire = _Wire(started + max(0.0, timeout_s), cancel, abort)
        if on_abort is not None:
            on_abort(abort.cancel)
        try:
            status = self._exchange(wire, messages, state, started)
        except Exception as exc:  # noqa: BLE001 — последняя страховка: голос не бросает
            status, state.error = "error", type(exc).__name__
        finally:
            wire.close()
            abort.close()
        return ChatResult(
            status=status,
            content=state.content if status in {"ok", "thinking_only"} else "",
            thinking_chars=len(state.thinking),
            done_reason=state.done_reason,
            eval_count=state.eval_count,
            prompt_eval_count=state.prompt_eval_count,
            elapsed_ms=_ms_since(started),
            first_token_ms=state.first_token_ms,
            load_ms=state.load_ms,
            error=state.error,
        )

    def _exchange(
        self,
        wire: _Wire,
        messages: Sequence[Mapping[str, str]],
        state: _Stream,
        started: float,
    ) -> ChatStatus:
        try:
            wire.connect(self._host, self._port, self._addresses)
        except _Stop as stop:
            if stop.why == "cancel":
                return "cancelled"
            # Срок вышел, а соединения нет: Ollama недоступна, а не медленна. У имени с
            # несколькими адресами (`localhost` на Windows — ::1 и 127.0.0.1) каждый отказ длится
            # около 2 с, и вместе они съедают весь срок — без этого исход был бы `timeout`, голос не
            # делал бы паузу 30 с и держал бы ворота на каждом вопросе (финальная проверка 24.09).
            self._addresses = None
            state.error = "connect_timeout"
            return "unavailable"
        except OSError as exc:  # порт не слушает, имя не разрешилось
            self._addresses = None
            state.error = type(exc).__name__
            return "unavailable"
        try:
            wire.send(self.request_bytes(messages))
            status_code, headers = _read_head(wire)
            if status_code != 200:
                state.error = f"HTTP {status_code}"
                return "unavailable" if status_code == 404 else "error"
            for line in _lines(_body(wire, headers)):
                if not line.strip():
                    continue
                if _feed(json.loads(line), state, started):
                    break
            else:
                state.error = "no_done"  # поток кончился без done: Ollama упала или разрыв
                return "error"
        except _Stop as stop:
            return "cancelled" if stop.why == "cancel" else "timeout"
        except (OSError, EOFError, ValueError) as exc:
            state.error = "no_done" if isinstance(exc, EOFError) else type(exc).__name__
            return "error"
        if state.error:
            return "error"
        if not state.content.strip() and state.thinking.strip():
            return "thinking_only"
        return "ok"


def _read_head(wire: _Wire) -> tuple[int, dict[str, str]]:
    parts = wire.line().decode("latin-1").split()
    if len(parts) < 2 or not parts[0].startswith("HTTP/") or not parts[1].isdigit():
        raise ValueError("не HTTP-ответ")
    headers: dict[str, str] = {}
    while True:
        line = wire.line().decode("latin-1").strip()
        if not line:
            return int(parts[1]), headers
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()


def _body(wire: _Wire, headers: Mapping[str, str]) -> Iterator[bytes]:
    """Тело ответа кусками: chunked (так отвечает Ollama), по длине или до закрытия."""
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size = int(wire.line().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                return
            yield wire.exact(size)
            wire.line()  # CRLF после куска
    elif "content-length" in headers:
        left = int(headers["content-length"])
        while left > 0:
            piece = wire.some()[:left]
            left -= len(piece)
            yield piece
    else:
        while True:
            try:
                yield wire.some()
            except EOFError:
                return


def _lines(pieces: Iterator[bytes]) -> Iterator[bytes]:
    """Строки NDJSON из кусков тела: кусок может резать строку где угодно."""
    pending = b""
    for piece in pieces:
        pending += piece
        *complete, pending = pending.split(b"\n")
        yield from complete
        if len(pending) > MAX_LINE_BYTES:
            raise ValueError("строка длиннее предела")
    if pending.strip():
        yield pending


def _feed(data: Any, state: _Stream, started: float) -> bool:
    """Строка потока в состояние; `True` — пришёл `done`, дальше читать нечего."""
    if not isinstance(data, dict):
        state.error = "bad_line"
        return True
    if data.get("error"):
        state.error = "ollama_error"
        return True
    message = data.get("message") or {}
    piece = str(message.get("content") or "")
    thought = str(message.get("thinking") or "")
    if (piece or thought) and state.first_token_ms is None:
        state.first_token_ms = _ms_since(started)
    state.content += piece
    state.thinking += thought
    if not data.get("done"):
        return False
    state.done_reason = data.get("done_reason")
    state.eval_count = _int_or_none(data.get("eval_count"))
    state.prompt_eval_count = _int_or_none(data.get("prompt_eval_count"))
    load_ns = _int_or_none(data.get("load_duration"))
    state.load_ms = None if load_ns is None else round(load_ns / 1e6)
    return True


def _ms_since(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
