"""Сервис на фейках: цепочка, деградация, прогрев, отказ старта. Без моделей, GPU и сети."""

from __future__ import annotations

import threading

import pytest
from api_env import (
    BLUE,
    CV_MODEL,
    FAKE_READER_KEY,
    RED,
    BrokenEmbedder,
    FakeClock,
    FakeOllama,
    SlowEmbedder,
    image_bytes,
    make_index,
    make_model,
    make_service,
    settings_for,
    write_files,
)
from fakes import ColorEmbedder

import app.api.service as service_module
from app.api.config import MEASURED_VLM_READER
from app.api.service import (
    LockedReader,
    ScannerService,
    StartupError,
    provenance,
    reader_failure,
    reader_identity,
    reading_status,
    text_read_of,
)
from app.reading.contracts import LabelFields, Reading, ReadResult, TextLine
from app.reading.readers.base import make_reading
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.resolve.attrs import CatalogAttrs
from bench.ocr_bench import overall_status

BETA_TEXT = "Бета Холмы\nМерло"


PREDICT_KEYS = [
    "slug",
    "confidence",
    "margin",
    "top5",
    "outcome",
    "degraded",
    "timings_ms",
    "error",
]


def assert_cv_fallback(result, reason: str) -> None:
    """Ответ запасного пути Э2: CV top-1, top-5 и счёты CV, вероятности нет, slug не null."""
    cv = result.evidence["cv"]["top5"]
    assert result.slug == cv[0]["slug"] == "alfa-muskat"
    assert [item.slug for item in result.top5] == [c["slug"] for c in cv]
    assert [item.score for item in result.top5] == [round(c["score"], 4) for c in cv]
    assert result.confidence.top1 is None and result.confidence.top5 is None
    assert result.margin == round(cv[0]["score"] - cv[1]["score"], 4)
    assert result.outcome == "matched" and result.error is None
    assert "resolve_error" not in result.degraded
    assert result.evidence["resolve"] == {
        "fallback": reason,
        "cv_top1": "alfa-muskat",
        "changed_cv_top1": False,
        "text_read": False,
    }
    assert list(result.predict_body()) == PREDICT_KEYS


def test_reading_without_catalog_words_goes_through_resolve_and_is_ambiguous():
    """Строка прочитана, но слов каталога в ней нет: путь прежний — решает слой выбора."""
    service = make_service(FakeOllama("Урожай позапрошлого года"))
    result = service.scan(image_bytes(RED))
    assert result.evidence["vlm"]["status"] == "ok" and result.evidence["vlm"]["lines"]
    assert result.slug == "alfa-muskat"
    assert result.outcome == "ambiguous"  # «Мерло» вплотную: p(top-1) ниже половины
    assert 0 < result.confidence.top1 < 0.5
    assert result.confidence.top5 >= result.confidence.top1
    assert [item.slug for item in result.top5][:2] == ["alfa-muskat", "beta-merlot"]
    assert result.error is None and result.degraded == []
    assert "fallback" not in result.evidence["resolve"]
    assert {"decode", "views", "cv", "vlm", "resolve", "total"} <= set(result.timings_ms)


def test_empty_reading_answers_cv_top1_without_probability():
    """Э2: VLM ответил пустотой (0 строк) — CV top-1 и top-5 CV, слой выбора не зовётся."""
    service = make_service(FakeOllama(""))
    result = service.scan(image_bytes(RED))
    assert result.evidence["vlm"]["status"] == "empty" and result.evidence["vlm"]["lines"] == []
    assert_cv_fallback(result, "reader_empty")
    assert result.degraded == []  # `empty` — не сбой транспорта: флага чтения нет и не было
    assert {"decode", "views", "cv", "vlm", "resolve", "total"} <= set(result.timings_ms)


def test_label_text_moves_answer_to_the_read_winery():
    transport = FakeOllama(BETA_TEXT)
    service = make_service(transport)
    result = service.scan(image_bytes(RED))
    assert result.slug == "beta-merlot"
    assert result.outcome == "matched"
    assert result.confidence.top1 > 0.9
    assert result.margin > 0.5
    assert result.degraded == []
    resolve = result.evidence["resolve"]
    assert resolve["cv_top1"] == "alfa-muskat" and resolve["changed_cv_top1"] is True
    assert resolve["bonus_flip_blocked"] is False  # P1: у модели фейков бонуса соседу нет
    assert result.evidence["vlm"]["fields"]["winery"] == ["Бета Холмы"]
    # чтение шло целым кадром 1024 с бюджетом сервиса (5 с: точность важнее скорости)
    assert transport.calls and transport.calls[0]["timeout_s"] == pytest.approx(5.0, abs=0.05)


def test_vlm_timeout_answers_cv_top1():
    service = make_service(FakeOllama(BETA_TEXT, error=TimeoutError("timed out")))
    result = service.scan(image_bytes(RED))
    assert result.evidence["vlm"]["status"] == "timeout"
    assert_cv_fallback(result, "reader_timeout")  # без текста решает картинка
    assert result.degraded == ["vlm_timeout"]  # флаг чтения остаётся


def test_vlm_unavailable_and_thinking_only_answers_are_flags_not_errors():
    down = make_service(FakeOllama(error=ConnectionError("refused"))).scan(image_bytes(RED))
    assert_cv_fallback(down, "reader_unavailable")
    assert "vlm_unavailable" in down.degraded
    thinking = make_service(FakeOllama("", thinking="Хм, на этикетке…")).scan(image_bytes(RED))
    assert_cv_fallback(thinking, "reader_error")
    assert "vlm_error" in thinking.degraded


class StatusReader:
    """Читатель без Ollama: каждое чтение — заданный статус и заданные строки."""

    id = "vlm"
    version = "fake:vlm"
    params_hash = "status-reader"

    def __init__(self, status: str, lines: tuple[str, ...] = ()) -> None:
        self.status = status
        self.lines = lines
        self.calls = 0

    def available(self) -> bool:
        return True

    def read(self, image, *, crop, budget_ms):
        self.calls += 1
        return make_reading(
            self,
            image,
            params=self.params_hash,
            crop=crop,
            status=self.status,
            elapsed_ms=5,
            lines=[TextLine(id=i, text=text) for i, text in enumerate(self.lines)],
        )


def with_reader(service: ScannerService, reader: StatusReader) -> ScannerService:
    service.vlm = LockedReader(reader, service.gpu_lock)
    return service


def forbid_resolve(monkeypatch) -> None:
    """Слой выбора и правило спорного кадра (H5) в запасном пути звать нельзя."""

    def forbidden(*args, **kwargs):
        raise AssertionError("при сбое читателя слой выбора не вызывается")

    monkeypatch.setattr(service_module, "rank_query_detailed", forbidden)
    monkeypatch.setattr(service_module, "rerank_ambiguous", forbidden)


@pytest.mark.parametrize(
    ("status", "flag"),
    [
        ("timeout", "vlm_timeout"),
        ("unavailable", "vlm_unavailable"),
        ("error", "vlm_error"),
        ("loop", "vlm_loop"),
        ("garbage", "vlm_garbage"),
        ("empty", None),
    ],
)
def test_reader_failure_answers_cv_top1_without_resolve(monkeypatch, status, flag):
    """Э2: любой статус чтения, кроме ok, — ответ CV top-1; модель и H5 не применяются."""
    forbid_resolve(monkeypatch)
    reader = StatusReader(status)
    result = with_reader(make_service(), reader).scan(image_bytes(RED))
    assert reader.calls == 1
    assert_cv_fallback(result, f"reader_{status}")
    assert result.degraded == ([flag] if flag else [])


def test_ok_reading_without_lines_answers_cv_top1(monkeypatch):
    """Статус ok, но ни одной строки — тоже сбой чтения: CV top-1."""
    forbid_resolve(monkeypatch)
    result = with_reader(make_service(), StatusReader("ok")).scan(image_bytes(RED))
    assert result.evidence["vlm"]["status"] == "ok" and result.evidence["vlm"]["lines"] == []
    assert_cv_fallback(result, "reader_no_lines")
    assert result.degraded == []


def test_reading_exception_answers_cv_top1(monkeypatch):
    forbid_resolve(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("сломался разбор")

    monkeypatch.setattr(service_module, "read_label", boom)
    result = make_service(FakeOllama(BETA_TEXT)).scan(image_bytes(RED))
    assert result.evidence["vlm"]["status"] == "error"
    assert_cv_fallback(result, "reader_error")
    assert result.degraded == ["vlm_exception"]


def test_partial_reading_keeps_the_resolve_path():
    """Строк ≥ 1 — путь прежний: прочитанная винодельня уводит ответ от CV top-1."""
    reader = StatusReader("ok", ("Бета Холмы",))
    result = with_reader(make_service(), reader).scan(image_bytes(RED))
    assert result.slug == "beta-merlot" and result.confidence.top1 > 0.5
    assert result.evidence["resolve"]["changed_cv_top1"] is True
    assert "fallback" not in result.evidence["resolve"]


def test_ollama_down_never_gives_null_slug():
    """Ollama лежит: каждый кадр — slug CV top-1, ни одного null и ни одной ошибки."""
    service = make_service(FakeOllama(error=ConnectionError("refused")))
    service.warm()
    results = [service.scan(image_bytes(color)) for color in (RED, BLUE, RED)]
    assert all(r.slug is not None and r.outcome == "matched" and r.error is None for r in results)
    assert all(r.slug == r.evidence["cv"]["top5"][0]["slug"] for r in results)
    assert all("vlm_unavailable" in r.degraded for r in results)
    assert service.health()["scans"]["null_slug"] == 0


@pytest.mark.parametrize(
    ("vlm", "expected"),
    [
        (None, None),
        ({}, None),
        ({"status": "not_needed"}, None),
        ({"status": "ok", "lines": ["БЕТА"]}, None),
        ({"status": "ok", "lines": []}, "no_lines"),
        ({"status": "ok"}, "no_lines"),
        ({"status": "timeout", "lines": []}, "timeout"),
        ({"status": "timeout", "lines": ["x"]}, "timeout"),
        ({"status": "empty", "lines": []}, "empty"),
        ({"status": "disabled"}, "disabled"),
        ({"status": "skipped", "budget_ms": 0}, "skipped"),
        ({"lines": ["x"]}, "unknown"),
    ],
)
def test_reader_failure_rule(vlm, expected):
    assert reader_failure(vlm) == expected


def test_resolve_failure_falls_back_to_cv_top1(monkeypatch):
    def broken(*args, **kwargs):
        raise KeyError("vlm35.winery_match")

    monkeypatch.setattr(service_module, "rank_query_detailed", broken)
    result = make_service(FakeOllama(BETA_TEXT)).scan(image_bytes(RED))
    assert result.slug == "alfa-muskat"
    assert "resolve_error" in result.degraded
    assert result.confidence.top1 is None and result.confidence.top5 is None
    assert result.top5[0].slug == "alfa-muskat" and result.top5[0].score == pytest.approx(
        0.99, 1e-3
    )
    assert result.outcome == "matched"


def test_cv_failure_gives_null_slug_with_error():
    result = make_service(FakeOllama(BETA_TEXT), embedder=BrokenEmbedder(CV_MODEL)).scan(
        image_bytes(RED)
    )
    assert result.slug is None
    assert result.outcome == "error"
    assert result.error.startswith("cv: RuntimeError")


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\0" * 40])
def test_decode_failure_gives_null_slug_with_error(data):
    result = make_service().scan(data)
    assert result.slug is None and result.outcome == "error"
    assert result.error.startswith("decode:")


def test_unexpected_exception_never_leaves_scan(monkeypatch):
    service = make_service()

    def boom(*args, **kwargs):
        raise AssertionError("баг")

    monkeypatch.setattr(service, "_resolve", boom)
    result = service.scan(image_bytes(RED))
    assert result.slug is None and result.error.startswith("internal: AssertionError")


def test_budget_spent_on_cv_skips_vlm():
    clock = FakeClock()
    transport = FakeOllama(BETA_TEXT)
    settings = settings_for(budget_ms=6000, vlm_timeout_ms=2500)
    service = make_service(
        transport, embedder=SlowEmbedder(clock, 5.5), clock=clock, settings=settings
    )
    result = service.scan(image_bytes(RED))
    assert "vlm_skipped_budget" in result.degraded
    assert transport.calls == []
    assert_cv_fallback(result, "reader_skipped")


def test_budget_left_after_cv_cuts_vlm_timeout():
    clock = FakeClock()
    transport = FakeOllama(BETA_TEXT)
    settings = settings_for(budget_ms=6000, vlm_timeout_ms=2500)
    service = make_service(
        transport, embedder=SlowEmbedder(clock, 3.3), clock=clock, settings=settings
    )
    result = service.scan(image_bytes(RED))
    assert "vlm_budget_cut" in result.degraded
    # 6000 − 3300 − запас 400 = 2300 мс вместо 2500
    assert transport.calls[0]["timeout_s"] <= 2.3 + 1e-6
    assert result.slug == "beta-merlot"


def test_gpu_lock_held_elsewhere_times_out_within_budget():
    settings = settings_for(budget_ms=200, vlm_timeout_ms=100)
    service = make_service(FakeOllama(BETA_TEXT), settings=settings)
    service.gpu_lock.acquire()
    try:
        result = service.scan(image_bytes(RED))
    finally:
        service.gpu_lock.release()
    assert result.slug is None and result.error.startswith("cv: GpuBusy")


def test_locked_reader_returns_timeout_when_lock_is_busy():
    lock = threading.Lock()
    inner = OllamaVlmReader("fake:vlm", transport=FakeOllama(BETA_TEXT))
    reader = LockedReader(inner, lock)
    lock.acquire()
    try:
        reading = reader.read(image_frame(), crop="full", budget_ms=50)
    finally:
        lock.release()
    assert reading.status == "timeout" and reading.reader == "vlm"
    assert reader.read(image_frame(), crop="full", budget_ms=2500).status == "ok"


def image_frame():
    import numpy as np

    return np.full((32, 32, 3), 200, dtype=np.uint8)


def test_abstain_ooc_only_returns_null_slug_for_unknown_position():
    settings = settings_for(abstain="ooc_only")
    # Винодельня «Гамма» прочитана уверенно, но «Мерло» у неё нет, а картинка слаба (синий кадр).
    service = make_service(FakeOllama("Гамма Берег\nМерло"), settings=settings)
    result = service.scan(image_bytes(BLUE))
    assert result.slug is None
    assert result.outcome == "out_of_catalog"
    assert result.evidence["abstain"]["fired"] is True
    assert result.top5, "подсказки «может быть» остаются и при отказе"


def test_abstain_off_always_answers_with_slug():
    """При `off` условия правила пишутся в evidence (для подсказки страницы), slug не трогается."""
    service = make_service(FakeOllama("Гамма Берег\nМерло"))
    result = service.scan(image_bytes(BLUE))
    assert result.slug is not None
    assert result.outcome != "out_of_catalog"
    check = result.evidence["abstain"]
    assert check["mode"] == "off" and check["fired"] is False
    # те же условия, при которых `ooc_only` отказал бы (test_abstain_ooc_only_…)
    assert check["winery_confident"] and check["no_position_matches"] and check["below_floor"]


def test_abstain_failure_under_off_does_not_touch_degraded(monkeypatch):
    """`degraded` уходит в predict: сбой проверки при `off` не должен его менять."""

    def boom(*args, **kwargs):
        raise RuntimeError("сломалось правило")

    monkeypatch.setattr(service_module, "abstain_check", boom)
    off = make_service(FakeOllama("Гамма Берег\nМерло")).scan(image_bytes(BLUE))
    assert off.degraded == [] and off.slug is not None
    assert "error" in off.evidence["abstain"]
    enforced = make_service(
        FakeOllama("Гамма Берег\nМерло"), settings=settings_for(abstain="ooc_only")
    ).scan(image_bytes(BLUE))
    assert "abstain_error" in enforced.degraded and enforced.slug is not None


def test_health_before_and_after_warm():
    service = make_service(FakeOllama("CHATEAU WARMUP"))
    assert service.health()["status"] == "starting"
    assert service.health()["warmed_at"] is None
    report = service.warm()
    health = service.health()
    assert health["status"] == "ready"
    assert health["warmed_at"] is not None
    assert report["cv"]["status"] == "ok" and report["vlm"]["status"] == "ok"
    assert report["scan"]["outcome"] in ("matched", "ambiguous")
    assert health["model"]["cv"] == CV_MODEL and health["model"]["resolve_reader"] == "vlm35"


class LatentOllama(FakeOllama):
    """Ollama, которой нужно `latency` секунд на ответ: не дождались — TimeoutError.

    Прогрев ждёт 60 с, а скан — 2,5 с: ответ за 4 с проходит на прогреве и никогда в скане.
    `latencies` — задержка по очереди вызовов (последняя повторяется).
    """

    def __init__(self, text: str, latencies: list[float]) -> None:
        super().__init__(text)
        self.latencies = list(latencies)

    def __call__(self, payload, *, timeout_s):
        latency = self.latencies.pop(0) if len(self.latencies) > 1 else self.latencies[0]
        if latency > timeout_s:
            self.calls.append({"model": payload["model"], "timeout_s": timeout_s})
            raise TimeoutError("timed out")
        return super().__call__(payload, timeout_s=timeout_s)


def test_vlm_answering_only_within_warmup_budget_is_degraded_not_ready():
    """VLM за 4 с — прогрев с бюджетом 60 с проходит, каждый скан — vlm_timeout.

    Раньше health говорил `ready`, и отчётный прогон молча шёл по одной картинке.
    """
    service = make_service(
        LatentOllama(BETA_TEXT, [4.0]), settings=settings_for(vlm_timeout_ms=2500)
    )
    report = service.warm()
    health = service.health()
    assert report["vlm"]["status"] == "ok"  # прогрев VLM сам по себе удался
    assert report["scan"]["degraded"] == ["vlm_timeout"] and report["scan"]["attempt"] == 2
    assert health["status"] == "degraded"
    assert "warm_scan:vlm_timeout" in health["degraded_reasons"]
    assert service.scan(image_bytes(RED)).slug == "alfa-muskat"  # CV top-1, текста нет


def test_cv_eating_the_budget_on_warm_scan_is_degraded():
    """CV 6,5 с (CPU): на VLM бюджета нет, ответ дольше бюджета — это не `ready`."""
    clock = FakeClock()
    settings = settings_for(budget_ms=6000, vlm_timeout_ms=2500)
    service = make_service(
        FakeOllama(BETA_TEXT), embedder=SlowEmbedder(clock, 6.5), clock=clock, settings=settings
    )
    service.warm()
    reasons = service.health()["degraded_reasons"]
    assert service.health()["status"] == "degraded"
    assert {"warm_scan:vlm_skipped_budget", "warm_scan:over_budget"} <= set(reasons)


def test_one_slow_warm_scan_is_retried_before_calling_it_degraded():
    # прогрев VLM (60 с) → первый скан не успел → второй успел
    service = make_service(
        LatentOllama(BETA_TEXT, [4.0, 4.0, 0.5]), settings=settings_for(vlm_timeout_ms=2500)
    )
    report = service.warm()
    assert report["scan"]["attempt"] == 2 and report["scan"]["degraded"] == []
    assert service.health()["status"] == "ready"
    assert service.health()["degraded_reasons"] == []


class DeviceEmbedder(ColorEmbedder):
    """Эмбеддер, который «поднялся» на заданном устройстве, как `SiglipEmbedder.device`."""

    def __init__(self, device: str) -> None:
        super().__init__(CV_MODEL)
        self.device = device


@pytest.mark.parametrize(
    ("requested", "actual", "fallback"),
    [
        ("cuda", "cpu", "cv_device_fallback:cuda->cpu"),
        ("cuda", "cuda", None),
        ("cuda", "cuda:0", None),
        ("cpu", "cpu", None),
    ],
)
def test_cv_device_fallback_is_shown_and_degrades(requested, actual, fallback):
    """SigLIP без CUDA молча уходит на CPU — health показывает фактическое устройство."""
    settings = settings_for(device=requested)
    service = make_service(
        FakeOllama("CHATEAU WARMUP"), settings=settings, embedder=DeviceEmbedder(actual)
    )
    service.warm()
    health = service.health()
    assert health["model"]["cv_device"] == actual
    assert health["settings"]["device"] == requested
    if fallback:
        assert health["status"] == "degraded" and fallback in health["degraded_reasons"]
    else:
        assert health["status"] == "ready"


def test_scan_counters_show_answers_without_text_and_skip_warmup():
    transport = LatentOllama(BETA_TEXT, [0.1])
    service = make_service(transport)
    service.warm()
    assert service.health()["scans"]["total"] == 0  # прогревочные сканы не в счёт
    service.scan(image_bytes(RED))
    transport.latencies = [9.0]  # дальше VLM не успевает
    service.scan(image_bytes(RED))
    service.scan(b"junk")
    scans = service.health()["scans"]
    assert scans["total"] == 3 and scans["text_read"] == 1
    assert scans["errors"] == 1 and scans["null_slug"] == 1
    assert scans["degraded"]["vlm_timeout"] == 1


def test_warm_without_ollama_is_degraded_not_fatal():
    service = make_service(FakeOllama(error=ConnectionError("refused")))
    service.warm()
    assert service.health()["status"] == "degraded"
    assert service.health()["vlm"]["warm"]["status"] == "unavailable"
    assert "vlm_warm_unavailable" in service.health()["degraded_reasons"]


def test_warm_with_broken_cv_is_degraded():
    service = make_service(FakeOllama("x"), embedder=BrokenEmbedder(CV_MODEL))
    report = service.warm()
    assert report["cv"]["status"] == "error" and "scan" not in report
    assert service.health()["status"] == "degraded"


def test_vlm_disabled_marks_every_answer():
    settings = settings_for(vlm_timeout_ms=0)
    service = make_service(settings=settings, vlm=False)
    result = service.scan(image_bytes(RED))
    assert "vlm_disabled" in result.degraded
    assert_cv_fallback(result, "reader_disabled")
    service.warm()
    assert service.health()["status"] == "degraded"


def test_index_model_mismatch_refuses_start(tmp_path):
    write_files(tmp_path, index_model="fake/other")
    with pytest.raises(StartupError, match="fake/other"):
        ScannerService.load(settings_for(tmp_path))


def test_constructor_refuses_index_of_other_model():
    settings = settings_for(cv_model="google/siglip2-so400m-patch14-384")
    with pytest.raises(StartupError, match="SVS_CV_MODEL"):
        make_service(settings=settings)


def test_top_k_other_than_training_refuses_start(tmp_path):
    write_files(tmp_path)
    with pytest.raises(StartupError, match="top-20"):
        ScannerService.load(settings_for(tmp_path, top_k=10))


def test_missing_files_refuse_start(tmp_path):
    write_files(tmp_path)
    (tmp_path / "lexicon.json").unlink()
    with pytest.raises(StartupError, match="словаря"):
        ScannerService.load(settings_for(tmp_path))
    with pytest.raises(StartupError, match="индекса"):
        ScannerService.load(settings_for(tmp_path, index_path=tmp_path / "nope.npz"))


def test_load_builds_service_with_cards_from_files(tmp_path):
    write_files(tmp_path)
    service = ScannerService.load(settings_for(tmp_path))
    assert service.health()["status"] == "starting"
    assert service.embedder.model_name == CV_MODEL  # SigLIP по паспорту индекса, веса не грузились
    card = service.card("beta-merlot")
    assert card["description"] == "Вишня и слива." and card["color"] == "Рубиновый"
    assert card["description_src"] == "catalog"
    # путь к фото остаётся внутри сервиса: в карточке только адрес нашего маршрута
    assert "photo_path" not in card
    assert service.cards.get("beta-merlot").photo_path == "C:/photos/beta-merlot.webp"
    assert card["photo_url"] is None  # файла C:/photos/… на этой машине нет
    assert card["region"] == "Кубань" and card["grapes"] == ["Мерло"]
    # сахар — по правилу выгрузки (название, затем slug), а не класс разметки
    assert service.card("alfa-muskat")["sugar"] == ""
    assert service.card("no-such-wine") is None
    # справочника рекомендаций в tmp_path нет — сервис всё равно собрался, без похожих
    assert service.health()["after"]["pool"] == 0


@pytest.mark.parametrize("candidate", ["adapter-lw-ranker", "off"])
def test_resolve_model_file_loads_with_expected_reader(candidate):
    """Модели слоя выбора обоих путей из configs/resolve загружаются и ждут `vlm35` на top-20."""
    from app.api.config import (
        CANDIDATE_RESOLVE_MODELS,
        CV_ADAPTER_SHA1,
        DEFAULT_RESOLVE_MODEL,
        GOAL_RESOLVE_MODEL,
    )
    from app.resolve.features import readers_of
    from app.resolve.learned import LogisticRanker

    path = CANDIDATE_RESOLVE_MODELS[candidate]
    model = LogisticRanker.load(path)
    assert model.fitted
    assert readers_of(model.feature_names) == ("vlm35",)
    assert model.meta["top_k"] == 20
    # Чьи чтения видела модель, записано в ней самой: сверка не опирается на запасной ключ.
    assert model.meta["reader_keys"] == {"vlm35": [MEASURED_VLM_READER]}
    if path == GOAL_RESOLVE_MODEL:
        assert model.meta["split"] == "dev" and "cv_adapter_sha1" not in model.meta
    else:
        # модель по умолчанию училась на пуле по выдаче карты адаптера — и без неё не стартует
        assert path == DEFAULT_RESOLVE_MODEL and model.meta["cv_adapter_sha1"] == CV_ADAPTER_SHA1
        assert model.meta["queries"] == 1517 and "kr_dev" in model.meta["sets"]


def test_measured_test_model_does_not_load_with_the_current_features():
    """Модель итогового замера test училась на признаках `resolve-features/2` (поля до правки
    родов цвета и сахара). Нынешний код считает признаки иначе, и она не загружается, а не
    работает молча на чужих признаках."""
    from app.api.config import REPO_ROOT
    from app.resolve.features import FEATURE_VERSION
    from app.resolve.learned import LogisticRanker, ModelFormatError

    measured = REPO_ROOT / "configs" / "resolve" / "s2so400m-vlm35.json"
    assert FEATURE_VERSION != "resolve-features/2"
    with pytest.raises(ModelFormatError):
        LogisticRanker.load(measured)


def test_goal_model_scans_on_fakes():
    """Модель пути `off` проходит цепочку сервиса: текст «Бета Холмы / Мерло» поднимает Мерло.

    Модель по умолчанию (`-lw-pool`) на подделках не проверить: она стартует только со своей
    картой адаптера на 1152-мерных векторах индекса (`test_fund_candidates.py`, данные сервиса).
    """
    from app.api.config import DEFAULT_VLM_MODEL, GOAL_RESOLVE_MODEL
    from app.resolve.learned import LogisticRanker

    model = LogisticRanker.load(GOAL_RESOLVE_MODEL)
    settings = settings_for(vlm_model=DEFAULT_VLM_MODEL)
    service = make_service(FakeOllama(BETA_TEXT), settings=settings, model=model)
    result = service.scan(image_bytes(RED))
    assert result.slug == "beta-merlot" and result.degraded == [] and result.error is None
    assert result.confidence.top1 is not None and result.confidence.top1 > 0.5
    report = service.health()["provenance"]
    assert report["consistent"] is True and report["vlm_reader_trained"] == [MEASURED_VLM_READER]
    without_text = make_service(FakeOllama(""), settings=settings, model=model)
    assert without_text.scan(image_bytes(RED)).slug == "alfa-muskat"  # без текста — CV-top1


def _reading(status, text=""):
    return Reading(
        reader="vlm",
        version="v",
        params_hash="p",
        image_sha1="0" * 40,
        crop="full",
        crop_px=1024,
        lines=[TextLine(id=0, text=text)] if text else [],
        status=status,
        elapsed_ms=10,
    )


@pytest.mark.parametrize(
    "statuses",
    [[], ["ok"], ["error"], ["timeout"], ["error", "ok"], ["empty", "error"], ["unavailable"]],
)
def test_reading_status_matches_ocr_bench(statuses):
    readings = [_reading(status) for status in statuses]
    assert reading_status(readings) == overall_status(readings)


def test_text_read_like_training():
    fields = LabelFields()
    ok = ReadResult(readings=[_reading("ok", "БЕТА ХОЛМЫ")], fields=fields)
    assert text_read_of(ok).raw_text == "БЕТА ХОЛМЫ" and text_read_of(ok).fields is fields
    failed = ReadResult(readings=[_reading("error")], fields=fields)
    assert text_read_of(failed).fields is None and text_read_of(failed).raw_text is None


def test_index_fixture_ranks_red_query_as_designed():
    from fakes import ColorEmbedder

    from app.features.views import from_query
    from app.normalize import decode_image

    index = make_index()
    views = from_query(decode_image(image_bytes(RED)), None)
    result = index.search(views, ColorEmbedder(CV_MODEL), top_k=3)
    assert [c.slug for c in result.candidates] == ["alfa-muskat", "beta-merlot", "beta-shardone"]


def test_provenance_flags_catalog_other_than_training(tmp_path):
    write_files(tmp_path)
    service = ScannerService.load(settings_for(tmp_path))
    assert service.health()["provenance"]["consistent"] is True
    model = make_model()
    model.meta["gt_tokens_sha1"] = "0" * 40  # обучена на другой разметке каталога
    other = make_service(model=model)
    other.attrs.meta["sha1"] = "f" * 40
    report = provenance(model, other.attrs, other.lexicon)
    assert report["consistent"] is False
    assert report["model_gt_tokens_sha1"] == "0" * 40


def test_other_vlm_than_the_model_was_trained_on_is_inconsistent():
    """SVS_VLM_MODEL другой модели — чтение чужой VLM под ключом `vlm35`."""
    settings = settings_for(vlm_model="qwen3-vl:4b-instruct")
    service = make_service(FakeOllama(BETA_TEXT), settings=settings)
    report = service.health()["provenance"]
    assert report["consistent"] is False
    assert report["vlm_reader"].startswith("vlm@qwen3-vl:4b-instruct|")
    assert report["vlm_reader_trained"] == [FAKE_READER_KEY]
    assert make_service(FakeOllama(BETA_TEXT)).health()["provenance"]["consistent"] is True


def test_model_without_reader_keys_is_checked_against_the_measured_reader():
    """У модели замера `meta.reader_keys` нет: ключ `vlm35` сверяется с читателем замера."""
    model = make_model(reader_keys={})
    report = provenance(
        model, CatalogAttrs.from_records([]), None, reader_key="vlm35", vlm_reader=FAKE_READER_KEY
    )
    assert report["vlm_reader_trained"] == [MEASURED_VLM_READER]
    assert report["consistent"] is False
    other_key = provenance(
        model, CatalogAttrs.from_records([]), None, reader_key="rapid", vlm_reader=FAKE_READER_KEY
    )
    assert other_key["vlm_reader_trained"] is None and other_key["consistent"] is True


def test_default_reader_is_the_one_whose_readings_trained_the_default_model():
    """Строка читателя сервиса по умолчанию — та, что стоит в прогонах OCR обучения модели.

    Хэш параметров собран из модели, промпта, размера кадра, `num_predict`, `num_ctx`,
    температуры и постобработки: поменяли промпт в коде — этот тест упадёт раньше, чем цифры
    замера тихо перестанут относиться к сервису.
    """
    from types import SimpleNamespace

    from app.api.config import (
        CV_ADAPTER_INDEX_SHA1,
        CV_ADAPTER_SHA1,
        DEFAULT_RESOLVE_MODEL,
        DEFAULT_VLM_MODEL,
        GOAL_RESOLVE_MODEL,
    )
    from app.resolve.learned import LogisticRanker

    reader = OllamaVlmReader(DEFAULT_VLM_MODEL, timeout_s=2.5, keep_alive="24h")
    assert reader_identity(reader) == MEASURED_VLM_READER == "vlm@qwen3.5:4b|f3a017317f04"
    passport = SimpleNamespace(sha1=CV_ADAPTER_SHA1, index_sha1=CV_ADAPTER_INDEX_SHA1)
    for path, adapter in ((DEFAULT_RESOLVE_MODEL, passport), (GOAL_RESOLVE_MODEL, None)):
        report = provenance(
            LogisticRanker.load(path),
            CatalogAttrs.from_records([]),
            None,
            reader_key="vlm35",
            vlm_reader=reader_identity(LockedReader(reader, threading.Lock())),
            cv_adapter=adapter,  # type: ignore[arg-type]
        )
        assert report["consistent"] is True, (path.name, report)
        assert report["vlm_reader_trained"] == [MEASURED_VLM_READER]


def test_live_index_keeps_csv_scores_and_can_answer_a_portal_card(tmp_path):
    """Э3: индекс с маской `base_rows` — счёт позиций CSV прежний, новая карточка встаёт в выдачу.

    Сервис грузится с диска, как `ScannerService.load` на стенде; SigLIP подменён цветовым
    эмбеддером, чтение этикетки выключено — ни весов, ни Ollama.
    """
    import numpy as np

    from app.features.index import VisualIndex
    from app.features.views import from_query
    from app.normalize import decode_image

    write_files(tmp_path)
    base = VisualIndex.load(tmp_path / "index.npz")
    live = VisualIndex(
        [*base.slugs, "portal-novoe"],
        [*base.views, "bottle"],
        np.vstack([base.vectors, np.array([[1.0, 0.0, 0.0]], dtype=np.float32)]),
        base.meta.model_copy(update={"n_slugs": base.n_slugs + 1, "n_vectors": len(base) + 1}),
        base_rows=np.arange(len(base) + 1) < len(base),
    )
    live.save(tmp_path / "live.npz")
    services = {}
    for name, path in (("csv", tmp_path / "index.npz"), ("live", tmp_path / "live.npz")):
        settings = settings_for(
            tmp_path, index_path=path, vlm_timeout_ms=0, live_cards=name == "live"
        )
        services[name] = ScannerService.load(settings)
        services[name].embedder = ColorEmbedder(CV_MODEL)
    health = {name: service.health() for name, service in services.items()}
    assert health["csv"]["index"]["n_base"] == health["csv"]["index"]["n_vectors"] == len(base)
    assert health["live"]["index"]["n_base"] == len(base)
    assert health["live"]["index"]["n_vectors"] == len(base) + 1
    assert health["live"]["settings"]["live_cards"] is True

    views = from_query(decode_image(image_bytes(RED)), None)
    found = {
        name: service.index.search(views, service.embedder, top_k=10, per_slug="zmax")
        for name, service in services.items()
    }
    assert found["live"].candidates[0].slug == "portal-novoe"
    kept = [(c.slug, c.score) for c in found["live"].candidates if c.slug != "portal-novoe"]
    assert kept == [(c.slug, pytest.approx(c.score, abs=1e-6)) for c in found["csv"].candidates]

    result = services["live"].scan(image_bytes(RED))
    assert result.slug == "portal-novoe" and result.outcome != "error"
    body = services["live"].scan_body(result)  # карточки в gt нет — ответ без неё, не сбой
    assert body["slug"] == "portal-novoe"
