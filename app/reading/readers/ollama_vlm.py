"""Модель зрения через родной /api/chat Ollama: построчный текст этикетки и статус чтения.

Ответ — обычный текст, не `format` JSON: кавычки и запятые JSON съедают num_predict, и длинная
этикетка обрезается в битый JSON. Выдумка модели узнаётся по почерку — повторам и мусору.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

from app.reading.contracts import CropName, Reading, ReadStatus, TextLine, params_hash
from app.reading.readers.base import (
    elapsed_ms_since,
    encode_jpeg_b64,
    ensure_rgb,
    long_side,
    make_reading,
    resize_long_side,
)

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "На фото бутылка вина. Перепиши построчно текст с этикетки бутылки, которая стоит "
    "в центре кадра. Текст соседних бутылок и ценников не переписывай. Пиши только то, "
    "что видно; нечитаемое пропускай. Ничего не добавляй и не переводи."
)

MAX_LINES = 24
JPEG_QUALITY = 92
# Меняется при любой правке разбора ответа: старые чтения в кэше перестают совпадать.
POSTPROCESS_VERSION = "1"

_LOOP_REPEATS = 3  # одна строка столько раз — зацикливание
_WORD_RUN = 4  # одно слово подряд столько раз — зацикливание без переводов строк
_GARBAGE_SHARE = 0.5
_CHAR_RUN = 8

_PREAMBLE_RE = re.compile(
    r"^(текст на этикетке|на этикетке|этикетка|текст|label text|the label reads)\s*[:\-—]\s*",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•·]|\d+[.)])\s+")
_EMPTY_ANSWERS = frozenset({"", "пусто", "пустой ответ", "не видно", "нет текста", "none", "n/a"})
_ALNUM_RE = re.compile(r"[^\W_]")


class Transport(Protocol):
    def __call__(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]: ...


class OllamaHttpError(RuntimeError):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}: {body[:200]}")
        self.code = code


class UrllibTransport:
    """POST JSON через stdlib: httpx в окружении может не оказаться."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def __call__(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OllamaHttpError(exc.code, exc.read().decode("utf-8", "replace")) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(str(exc.reason)) from exc
            raise ConnectionError(str(exc.reason)) from exc


def _split(text: str) -> list[str]:
    lines: list[str] = []
    for part in re.split(r"[\n|]+", text or ""):
        part = _PREAMBLE_RE.sub("", part.strip())
        part = _BULLET_RE.sub("", part)
        part = part.strip(" \t\"'«»`*")
        if part.lower() in _EMPTY_ANSWERS:
            continue
        # Строка без букв и цифр — сбой генерации («@@@@» под нехваткой видеопамяти).
        if not _ALNUM_RE.search(part):
            continue
        lines.append(part)
    return lines


def _is_loop(lines: list[str]) -> bool:
    if not lines:
        return False
    if Counter(line.lower() for line in lines).most_common(1)[0][1] >= _LOOP_REPEATS:
        return True
    words = " ".join(lines).lower().split()
    run = 1
    for prev, cur in pairwise(words):
        run = run + 1 if cur == prev else 1
        if run >= _WORD_RUN:
            return True
    return False


def clean_lines(text: str, max_lines: int = MAX_LINES) -> list[str]:
    """Строки ответа без обёрток, маркеров и повторов. Пусто — ничего не прочитано или петля."""
    lines = _split(text)
    if not lines or _is_loop(lines):
        return []
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line.lower() not in seen:
            seen.add(line.lower())
            unique.append(line)
    return unique[:max_lines]


def is_garbage(text: str) -> bool:
    """Больше половины непробельных символов — не буквы и не цифры, или один символ подряд."""
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    junk = sum(1 for ch in chars if not _ALNUM_RE.match(ch))
    if junk / len(chars) > _GARBAGE_SHARE:
        return True
    longest = run = 1
    for prev, cur in pairwise(chars):
        run = run + 1 if cur == prev else 1
        longest = max(longest, run)
    return longest >= _CHAR_RUN and longest / len(chars) > _GARBAGE_SHARE


def classify(content: str, thinking: str = "") -> tuple[ReadStatus, list[str]]:
    """Статус и строки по тексту ответа модели.

    Пустой ответ при непустом `thinking` — сбой среды, а не итог модели: think:false не
    сработал, и ответ ушёл в рассуждение. Такое чтение получает `error`: оно не кэшируется
    и видно в `degraded`, а не выглядит пустой этикеткой.
    """
    if not content.strip():
        return ("error" if thinking.strip() else "empty"), []
    if is_garbage(content):
        return "garbage", []
    if _is_loop(_split(content)):
        return "loop", []
    lines = clean_lines(content)
    return ("ok", lines) if lines else ("empty", [])


def _failure_status(exc: BaseException) -> ReadStatus:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, OllamaHttpError):
        return "unavailable" if exc.code == 404 else "error"
    if isinstance(exc, (ConnectionError, OSError)):
        return "unavailable"
    name = type(exc).__name__.lower()  # httpx и прочие транспорты
    if "timeout" in name:
        return "timeout"
    if "connect" in name:
        return "unavailable"
    return "error"


class OllamaVlmReader:
    id = "vlm"

    def __init__(
        self,
        model: str,
        url: str = "http://127.0.0.1:11434",
        *,
        long_side: int = 1024,
        num_predict: int = 160,
        temperature: float = 0,
        # Модель остаётся в памяти весь прогон: пауза между кадрами не выгружает её.
        keep_alive: str = "30m",
        timeout_s: float = 8.0,
        num_ctx: int = 4096,
        prompt: str = DEFAULT_PROMPT,
        transport: Transport | None = None,
        probe: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        base = url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.url = base
        self.version = model
        self.long_side = long_side
        self.num_predict = num_predict
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.prompt = prompt
        self._transport: Transport = transport or UrllibTransport(f"{base}/api/chat")
        self._probe = probe
        self.params_hash = params_hash(
            {
                "model": model,
                "prompt": prompt,
                "long_side": long_side,
                "num_predict": num_predict,
                "temperature": temperature,
                "num_ctx": num_ctx,
                "jpeg_quality": JPEG_QUALITY,
                "postprocess": POSTPROCESS_VERSION,
                "max_lines": MAX_LINES,
            }
        )

    def crop_px_for(self, image: np.ndarray) -> int:
        return min(self.long_side, long_side(image))

    def available(self) -> bool:
        try:
            tags = self._probe() if self._probe else self._fetch_tags()
        except Exception:  # noqa: BLE001 — недоступность штатная
            return False
        names = {m.get("name") or m.get("model") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def _fetch_tags(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.url}/api/tags", timeout=1.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def payload(self, image_b64: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "messages": [{"role": "user", "content": self.prompt, "images": [image_b64]}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }

    def read(self, image: np.ndarray, *, crop: CropName, budget_ms: int) -> Reading:
        """Бюджет `budget_ms` <= 0 — сразу timeout без вызова модели."""
        return self._read(image, crop=crop, budget_ms=budget_ms, timeout_s=self.timeout_s)

    def warm(self, image: np.ndarray, *, budget_ms: int) -> Reading:
        """Прогрев: тот же запрос, но ждать можно весь бюджет, а не только `timeout_s`.

        Холодная загрузка модели бывает дольше `timeout_s`, а оборванный запрос прогрева
        ничего не даёт. Дальше модель держит `keep_alive`.
        """
        return self._read(image, crop="full", budget_ms=budget_ms, timeout_s=budget_ms / 1000)

    def _read(
        self, image: np.ndarray, *, crop: CropName, budget_ms: int, timeout_s: float
    ) -> Reading:
        ensure_rgb(image)
        t0 = time.perf_counter()

        def done(status: ReadStatus, raw: dict[str, Any], **kw: Any) -> Reading:
            return make_reading(
                self,
                image,
                params=self.params_hash,
                crop=crop,
                status=status,
                elapsed_ms=elapsed_ms_since(t0),
                raw=json.dumps(raw, ensure_ascii=False),
                **kw,
            )

        if budget_ms <= 0:
            return done("timeout", {"error": "budget exhausted"})
        picture = encode_jpeg_b64(resize_long_side(image, self.long_side), JPEG_QUALITY)
        remaining = budget_ms / 1000 - (time.perf_counter() - t0)
        timeout = min(timeout_s, remaining)
        if timeout <= 0:
            return done("timeout", {"error": "budget exhausted"})
        try:
            data = self._transport(self.payload(picture), timeout_s=timeout)
        except Exception as exc:  # noqa: BLE001 — любая осечка превращается в статус
            return done(_failure_status(exc), {"error": f"{type(exc).__name__}: {exc}"})
        if not isinstance(data, dict) or data.get("error"):
            error = data.get("error") if isinstance(data, dict) else repr(data)
            return done("error", {"error": str(error)})

        message = data.get("message") or {}
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        status, lines = classify(content, thinking)
        raw: dict[str, Any] = {
            "content": content,
            "done_reason": data.get("done_reason"),
            "eval_count": data.get("eval_count"),
        }
        if thinking:
            raw["thinking_chars"] = len(thinking)
            # Ловушка тегов с рассуждением: think:false не сработал, ответ ушёл в thinking.
            raw["thinking_only"] = not content.strip()
            if raw["thinking_only"]:
                raw["error"] = "thinking_only: think:false не сработал, ответ ушёл в рассуждение"
        if data.get("done_reason") == "length":
            raw["truncated"] = True
        return done(
            status,
            raw,
            lines=[TextLine(id=i, text=text) for i, text in enumerate(lines)],
            prompt_tokens=data.get("prompt_eval_count"),
        )
