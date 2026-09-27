"""Голос сомелье не заставляет Ollama перезагрузить модель читателя этикетки (договор, §6.2).

Ollama держит одну загрузку модели на набор параметров загрузки: другой `num_ctx`, другой
`keep_alive` или любой параметр вроде `num_gpu` — и модель грузится заново, а следующий скан
платит 3–4 с. Поэтому клиент голоса собирается из живого читателя (`OllamaText.mirroring`), и
тест сверяет с `OllamaVlmReader.payload` модель, `keep_alive`, `num_ctx` и набор ключей — и в
теле, которое голос собирает, и в том, что на самом деле ушло в сокет.
"""

from __future__ import annotations

import threading

import pytest
from somm_voice_env import ChatPlan, FakeOllama

from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.sommelier.ollama_text import MAX_NUM_PREDICT, OllamaText

pytest.importorskip("fastapi")

from api_env import make_service, settings_for

from app.api.config import ServiceSettings

LOAD_OPTIONS = {"num_gpu", "num_thread", "num_batch", "use_mmap", "use_mlock", "main_gpu"}
MESSAGES = [{"role": "user", "content": "проверка"}]


def reader_like_service(settings: ServiceSettings) -> OllamaVlmReader:
    """Читатель ровно так, как его собирает `ScannerService.load`."""
    return OllamaVlmReader(
        settings.vlm_model,
        settings.ollama_url,
        timeout_s=settings.vlm_timeout_ms / 1000,
        keep_alive=settings.vlm_keep_alive,
    )


def assert_same_load(reader: OllamaVlmReader, text: OllamaText) -> None:
    ours = text.payload(MESSAGES)
    theirs = reader.payload("")
    assert ours["model"] == theirs["model"]
    assert ours["keep_alive"] == theirs["keep_alive"]
    assert ours["options"]["num_ctx"] == theirs["options"]["num_ctx"]
    assert set(ours) == set(theirs)
    assert set(ours["options"]) == set(theirs["options"])
    assert not set(ours["options"]) & LOAD_OPTIONS
    assert ours["think"] is False and ours["stream"] is True
    assert ours["options"]["num_predict"] <= MAX_NUM_PREDICT == 160


def test_голос_зеркалит_читателя_сервиса():
    settings = ServiceSettings.from_env({"SVS_OLLAMA_URL": "http://127.0.0.1:11434"})
    reader = reader_like_service(settings)
    text = OllamaText.mirroring(reader)
    assert_same_load(reader, text)
    assert text.model == settings.vlm_model == "qwen3.5:4b"
    assert text.keep_alive == settings.vlm_keep_alive == "24h"
    assert text.num_ctx == reader.num_ctx == 4096
    assert text.url == reader.url


@pytest.mark.parametrize(
    "env",
    [
        {"SVS_VLM_KEEP_ALIVE": "-1", "SVS_VLM_MODEL": "qwen3-vl:4b-instruct"},
        {"SVS_OLLAMA_URL": "http://ollama:11434/v1"},
    ],
)
def test_голос_следует_за_настройками_читателя(env):
    reader = reader_like_service(ServiceSettings.from_env(env))
    text = OllamaText.mirroring(reader)
    assert_same_load(reader, text)


def test_голос_зеркалит_читателя_под_замком_сервиса(tmp_path):
    """В сервисе читатель обёрнут `LockedReader`; зеркало берёт настоящего из `.reader`."""
    service = make_service(settings=settings_for(tmp_path))
    text = OllamaText.mirroring(service.vlm)
    assert_same_load(service.vlm.reader, text)


def test_в_сокет_уходят_те_же_ключи():
    reader = OllamaVlmReader("qwen3.5:4b", keep_alive="24h")
    with FakeOllama(ChatPlan(tokens=("Да.",))) as fake:
        reader.url = fake.url
        text = OllamaText.mirroring(reader)
        assert text.chat(MESSAGES, timeout_s=2, cancel=threading.Event()).status == "ok"
    (sent,) = fake.payloads
    assert sent == text.payload(MESSAGES)
    assert set(sent["options"]) == set(reader.payload("")["options"])
    assert sent["keep_alive"] == "24h" and sent["options"]["num_ctx"] == 4096


def test_длинный_ответ_не_разрешён():
    reader = OllamaVlmReader("qwen3.5:4b")
    with pytest.raises(ValueError):
        OllamaText.mirroring(reader, num_predict=MAX_NUM_PREDICT + 1)
