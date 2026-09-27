import base64
import json
from pathlib import Path

import numpy as np
import pytest

from app.reading.readers.ollama_vlm import (
    DEFAULT_PROMPT,
    OllamaHttpError,
    OllamaVlmReader,
    classify,
    clean_lines,
    is_garbage,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ollama"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeTransport:
    def __init__(self, response=None, exc: BaseException | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[tuple[dict, float]] = []

    def __call__(self, payload, *, timeout_s):
        self.calls.append((payload, timeout_s))
        if self.exc is not None:
            raise self.exc
        return self.response


def frame(h: int = 1600, w: int = 900) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def reader(transport, **kw) -> OllamaVlmReader:
    return OllamaVlmReader("qwen2.5vl:3b", "http://127.0.0.1:11434", transport=transport, **kw)


# --- перенесено из «Лозы»: TestCleanLines ---


class TestCleanLines:
    def test_splits_on_newline_and_bar(self):
        assert clean_lines("SEGHESIO\nZinfandel | 2022") == ["SEGHESIO", "Zinfandel", "2022"]

    def test_preamble_and_bullets_removed(self):
        text = "Текст на этикетке:\n- Массандра\n2. Кагор Южнобережный"
        assert clean_lines(text) == ["Массандра", "Кагор Южнобережный"]

    def test_loop_means_fabrication(self):
        text = "Bordeaux | Château | Château | Château | Château | Château"
        assert clean_lines(text) == []

    @pytest.mark.parametrize("text", ["", "Не видно", "пустой ответ", "  \n  "])
    def test_honest_nothing_stays_nothing(self, text):
        assert clean_lines(text) == []

    def test_repeat_twice_is_not_loop(self):
        assert clean_lines("БЮРНЬЕ | БЮРНЬЕ | КАБЕРНЕ ФРАН") == ["БЮРНЬЕ", "КАБЕРНЕ ФРАН"]

    def test_quotes_stripped(self):
        assert clean_lines('«esse»\n"rosé"') == ["esse", "rosé"]

    def test_line_without_letters_or_digits_dropped(self):
        assert clean_lines("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@") == []
        assert clean_lines("*** | АБРАУ | ---") == ["АБРАУ"]


# --- правила статусов на синтетических строках ---


def test_word_run_without_newlines_is_loop():
    assert classify("ВИНО ВИНО ВИНО ВИНО ВИНО")[0] == "loop"


def test_numeric_label_lines_are_not_garbage():
    # Годы, объём и крепость — цифры, а не мусор.
    assert not is_garbage("2021\n0,75 л\n12%\n1987")
    assert classify("2021\n0,75 л\n12%")[0] == "ok"


def test_mostly_punctuation_is_garbage():
    assert is_garbage("!!!! ?? --- АБ")


# --- статусы на записанных ответах ---


def test_ok_response_lines_and_meta():
    reading = reader(FakeTransport(load("ok"))).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == "ok"
    assert reading.text.splitlines()[0] == "ВИНОДЕЛЬНЯ СОЛНЕЧНАЯ ДОЛИНА"
    assert all(line.conf is None and line.box is None for line in reading.lines)
    assert reading.prompt_tokens == 1133
    assert json.loads(reading.raw)["content"].startswith("ВИНОДЕЛЬНЯ")
    assert reading.reader == "vlm" and reading.version == "qwen2.5vl:3b"


def test_thinking_only_is_error_with_flag():
    # think:false не сработал: это сбой среды, а не пустая этикетка — не кэшируется и в degraded.
    reading = reader(FakeTransport(load("thinking_only"))).read(
        frame(), crop="full", budget_ms=5000
    )
    raw = json.loads(reading.raw)
    assert reading.status == "error"
    assert reading.lines == []
    assert raw["thinking_only"] is True and raw["thinking_chars"] > 0
    assert raw["truncated"] is True and raw["error"].startswith("thinking_only")


def test_empty_content_is_empty_only_without_thinking():
    assert classify("", "") == ("empty", [])
    assert classify("  \n", "рассуждение модели") == ("error", [])


def test_loop_response():
    reading = reader(FakeTransport(load("loop"))).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == "loop" and reading.lines == []
    assert "Château" in json.loads(reading.raw)["content"]


def test_garbage_response():
    reading = reader(FakeTransport(load("garbage"))).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == "garbage" and reading.lines == []


def test_empty_answer_response():
    reading = reader(FakeTransport(load("empty"))).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == "empty"


def test_timeout_from_transport():
    transport = FakeTransport(exc=TimeoutError("timed out"))
    reading = reader(transport).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == "timeout"
    assert "TimeoutError" in json.loads(reading.raw)["error"]


def test_timeout_is_min_of_budget_and_timeout_s():
    transport = FakeTransport(load("ok"))
    reader(transport, timeout_s=8.0).read(frame(), crop="full", budget_ms=2000)
    assert 1.0 < transport.calls[0][1] <= 2.0
    transport = FakeTransport(load("ok"))
    reader(transport, timeout_s=0.5).read(frame(), crop="full", budget_ms=2000)
    assert transport.calls[0][1] == pytest.approx(0.5)


def test_exhausted_budget_skips_call():
    transport = FakeTransport(load("ok"))
    reading = reader(transport).read(frame(), crop="full", budget_ms=0)
    assert reading.status == "timeout" and transport.calls == []


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (ConnectionRefusedError("refused"), "unavailable"),
        (OllamaHttpError(404, "model not found"), "unavailable"),
        (OllamaHttpError(500, "boom"), "error"),
        (ValueError("bad json"), "error"),
    ],
)
def test_transport_failures_map_to_status(exc, status):
    reading = reader(FakeTransport(exc=exc)).read(frame(), crop="full", budget_ms=5000)
    assert reading.status == status


def test_error_field_in_response():
    reading = reader(FakeTransport({"error": "llama runner process has terminated"})).read(
        frame(), crop="full", budget_ms=5000
    )
    assert reading.status == "error"


def test_payload_shape():
    transport = FakeTransport(load("ok"))
    reader(transport, num_predict=160, keep_alive="10m").read(
        frame(1600, 900), crop="label", budget_ms=5000
    )
    payload = transport.calls[0][0]
    assert payload["think"] is False and payload["stream"] is False
    assert "format" not in payload
    assert payload["keep_alive"] == "10m"
    assert payload["options"] == {"temperature": 0, "num_predict": 160, "num_ctx": 4096}
    message = payload["messages"][0]
    assert message["content"] == DEFAULT_PROMPT
    assert len(message["images"]) == 1
    jpeg = base64.b64decode(message["images"][0])
    assert jpeg[:2] == b"\xff\xd8"


def test_default_keep_alive_holds_model_for_the_run():
    transport = FakeTransport(load("ok"))
    reader(transport).read(frame(), crop="full", budget_ms=5000)
    assert transport.calls[0][0]["keep_alive"] == "30m"


def test_warm_waits_whole_budget_not_timeout_s():
    transport = FakeTransport(load("ok"))
    vlm = reader(transport, timeout_s=1.0)
    warmed = vlm.warm(frame(), budget_ms=30_000)
    assert warmed.status == "ok" and warmed.crop == "full"
    assert 25.0 < transport.calls[0][1] <= 30.0
    vlm.read(frame(), crop="full", budget_ms=30_000)
    assert transport.calls[1][1] <= 1.0  # обычное чтение по-прежнему ограничено timeout_s
    assert transport.calls[0][0]["messages"] == transport.calls[1][0]["messages"]


def test_image_shrunk_to_long_side_and_crop_px():
    from io import BytesIO

    from PIL import Image

    transport = FakeTransport(load("ok"))
    reading = reader(transport, long_side=768).read(frame(1600, 900), crop="label", budget_ms=5000)
    image = Image.open(BytesIO(base64.b64decode(transport.calls[0][0]["messages"][0]["images"][0])))
    assert max(image.size) == 768
    assert reading.crop_px == 768
    small = reader(FakeTransport(load("ok")), long_side=768).read(
        frame(400, 300), crop="label", budget_ms=5000
    )
    assert small.crop_px == 400


def test_params_hash_tracks_output_affecting_params():
    base = reader(FakeTransport()).params_hash
    assert reader(FakeTransport(), long_side=768).params_hash != base
    assert reader(FakeTransport(), num_predict=96).params_hash != base
    assert reader(FakeTransport(), temperature=0.2).params_hash != base
    assert reader(FakeTransport(), prompt="другой").params_hash != base
    assert OllamaVlmReader("qwen2.5vl:7b", transport=FakeTransport()).params_hash != base
    assert reader(FakeTransport(), timeout_s=1.0, keep_alive="0").params_hash == base


def test_url_v1_suffix_stripped():
    assert OllamaVlmReader("m", "http://ollama:11434/v1/").url == "http://ollama:11434"


def test_available_uses_probe():
    tags = {"models": [{"name": "qwen2.5vl:3b"}, {"name": "llava:latest"}]}
    assert OllamaVlmReader("qwen2.5vl:3b", probe=lambda: tags).available()
    assert OllamaVlmReader("llava", probe=lambda: tags).available()
    assert not OllamaVlmReader("qwen2.5vl:7b", probe=lambda: tags).available()

    def down():
        raise ConnectionRefusedError

    assert not OllamaVlmReader("qwen2.5vl:3b", probe=down).available()
