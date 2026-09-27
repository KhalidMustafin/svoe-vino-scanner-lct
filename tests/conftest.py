"""Общие настройки тестов: только CPU, GPU-тесты — по явному `SVS_RUN_GPU_TESTS=1`.

Голос сомелье (`app/sommelier/ollama_text.py`) в тестах не ходит в настоящую Ollama: сервис на
фейках собирает живой `Voice`, и его клиент зеркалит адрес читателя этикетки — по умолчанию
`127.0.0.1:11434`, где на машине разработчика Ollama и слушает. Соединение с портом Ollama
отказывает сразу (`unavailable`); поддельные Ollama тестов слушают свободные порты и работают.
"""

from __future__ import annotations

import os

import pytest

GPU_TESTS = os.environ.get("SVS_RUN_GPU_TESTS") == "1"

# До импорта torch и onnxruntime в модулях тестов: юнит-тесты не занимают видеокарту.
if not GPU_TESTS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#: Порт настоящей Ollama (`SVS_OLLAMA_URL` по умолчанию и в compose): в тестах он закрыт.
OLLAMA_PORT = 11434


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Клиент голоса не соединяется с портом Ollama: ни модели, ни видеокарты в юнит-тестах."""
    if GPU_TESTS:
        return
    from app.sommelier import ollama_text

    connect = ollama_text._Wire.connect

    def guarded(self: object, host: str, port: int, *args: object) -> None:
        if port == OLLAMA_PORT:
            raise ConnectionRefusedError(0, "тесты не ходят в настоящую Ollama")
        connect(self, host, port, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(ollama_text._Wire, "connect", guarded)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: нужна видеокарта, модель или Ollama; запуск при SVS_RUN_GPU_TESTS=1"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if GPU_TESTS:
        return
    skip = pytest.mark.skip(reason="нужна видеокарта или модель: SVS_RUN_GPU_TESTS=1")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
