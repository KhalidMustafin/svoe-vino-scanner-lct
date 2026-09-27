"""Настройки сервиса из `SVS_*`: значения замера по умолчанию и отказ на кривых значениях."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api.config import (
    CV_ADAPTER_NAME,
    CV_PER_SLUG,
    DEFAULT_CV_MODEL,
    DEFAULT_RESOLVE_MODEL,
    LIVE_CARDS_DEFAULT,
    REPO_ROOT,
    ServiceSettings,
    SettingsError,
)


def test_defaults_are_the_service_chain():
    """Схема замера (индекс, CV, VLM, кроп, бюджет, top-K) и путь поиска и выбора сервиса.

    С 26.09 по умолчанию — кандидат 2 фундаментального трека: адаптер поиска LW (карта рядом с
    индексом) и модель `-lw-pool`, обученная на его выдаче; принят по замороженному kr-test
    (`research/2026-09-26_fund/`). Прежний путь (`-goal` без адаптера) — `SVS_CANDIDATE=off`.
    """
    settings = ServiceSettings.from_env({})
    assert settings.cv_model == DEFAULT_CV_MODEL == "google/siglip2-so400m-patch14-384"
    assert settings.index_path.name == "visual-s2so400m.npz"
    assert settings.candidate == "adapter-lw-ranker"
    assert settings.cv_adapter == REPO_ROOT / "data" / "index" / CV_ADAPTER_NAME
    assert settings.cv_adapter.parent == settings.index_path.parent  # карта — рядом с индексом
    assert settings.resolve_model == DEFAULT_RESOLVE_MODEL
    assert settings.resolve_model.parts[-3:] == (
        "configs",
        "resolve",
        "s2so400m-vlm35-lw-pool.json",
    )
    assert CV_PER_SLUG == "zmax"  # счёты CV, на которых училась модель по умолчанию
    assert settings.vlm_model == "qwen3.5:4b"
    assert (settings.budget_ms, settings.vlm_timeout_ms, settings.top_k) == (8000, 5000, 20)
    # `from_env` без переменных = поля класса: так сервис стартует через `python -m app.api`
    assert settings == ServiceSettings()
    assert settings.abstain == "off"
    assert (settings.host, settings.port, settings.device) == ("0.0.0.0", 8080, "cuda")
    assert settings.max_upload_bytes == 25 * 1024 * 1024
    assert settings.warnings() == []


def test_environment_overrides(tmp_path):
    env = {
        "SVS_INDEX_PATH": str(tmp_path / "i.npz"),
        "SVS_CV_MODEL": "google/siglip2-base-patch16-224",
        "SVS_BUDGET_MS": "8000",
        "SVS_VLM_TIMEOUT_MS": "0",
        "SVS_ABSTAIN": "ooc_only",
        "SVS_PORT": "18080",
        "SVS_DEVICE": "cpu",
        "SVS_DATA_DIR": str(tmp_path / "data"),
        "SVS_DATASET_DIR": str(tmp_path / "dataset"),
        "SVS_MAX_UPLOAD_MB": "5",
    }
    settings = ServiceSettings.from_env(env)
    assert settings.index_path == (tmp_path / "i.npz").resolve()
    assert settings.budget_ms == 8000 and settings.port == 18080 and settings.device == "cpu"
    assert not settings.vlm_enabled
    assert settings.abstain == "ooc_only"
    assert settings.lexicon_path == Path(tmp_path / "data" / "index" / "lexicon.json").resolve()
    assert settings.catalog_csv.parent == (tmp_path / "dataset").resolve()
    assert settings.max_upload_bytes == 5 * 1024 * 1024
    assert any("SVS_ABSTAIN" in warning for warning in settings.warnings())
    assert settings.public()["index_path"] == str(settings.index_path)


@pytest.mark.parametrize(
    "env",
    [
        {"SVS_BUDGET_MS": "fast"},
        {"SVS_BUDGET_MS": "0"},
        {"SVS_ABSTAIN": "always"},
        {"SVS_VLM_TIMEOUT_MS": "9000"},  # больше общего бюджета 8000
        {"SVS_TOP_K": "0"},
        {"SVS_PORT": "70000"},
        {"SVS_LIVE_CARDS": "maybe"},
    ],
)
def test_bad_values_are_refused(env):
    with pytest.raises(SettingsError):
        ServiceSettings.from_env(env)


def test_budget_near_script_timeout_is_a_warning():
    settings = ServiceSettings.from_env({"SVS_BUDGET_MS": "10000"})
    assert any("10000" in warning for warning in settings.warnings())


def test_other_vlm_model_is_a_warning():
    """`SVS_VLM_MODEL` общий со стендами — чужая VLM не должна пройти молча."""
    settings = ServiceSettings.from_env({"SVS_VLM_MODEL": "qwen3-vl:4b-instruct"})
    assert any("SVS_VLM_MODEL=qwen3-vl:4b-instruct" in warning for warning in settings.warnings())
    # без чтения этикетки модель VLM не важна
    off = ServiceSettings.from_env({"SVS_VLM_MODEL": "x:1", "SVS_VLM_TIMEOUT_MS": "0"})
    assert not any("SVS_VLM_MODEL" in warning for warning in off.warnings())


@pytest.mark.parametrize(("budget", "warned"), [("5500", True), ("6899", True), ("6900", False)])
def test_budget_leaving_too_little_for_decode_and_cv_is_a_warning(budget, warned):
    """8000 − 5000 − 400 = 2600 мс на разжатие и CV; при 5500 — 100 мс, и VLM режется всегда."""
    settings = ServiceSettings.from_env({"SVS_BUDGET_MS": budget})
    assert settings.cv_room_ms == int(budget) - 5000 - 400
    assert any("vlm_budget_cut" in warning for warning in settings.warnings()) is warned


@pytest.mark.parametrize("value", ["1", "on", "TRUE", " yes "])
def test_live_cards_switch_index_gt_and_lexicon_together(tmp_path, value):
    """Э3: один флаг — три файла комплекта «CSV + живые карточки» (scripts/build_live_set.py)."""
    data = tmp_path / "data"
    settings = ServiceSettings.from_env(
        {"SVS_DATA_DIR": str(data), "SVS_LIVE_CARDS": value, "SVS_CANDIDATE": "off"}
    )
    assert settings.live_cards is True
    assert settings.index_path == (data / "index" / "visual-s2so400m-live71.npz").resolve()
    assert settings.attrs_path == (data / "gt" / "gt_tokens-live71.jsonl").resolve()
    assert settings.lexicon_path == (data / "index" / "lexicon-live71.json").resolve()
    assert settings.public()["live_cards"] is True


@pytest.mark.parametrize("value", ["0", "off", "No", "false"])
def test_live_cards_off_is_the_csv_set(tmp_path, value):
    data = tmp_path / "data"
    settings = ServiceSettings.from_env({"SVS_DATA_DIR": str(data), "SVS_LIVE_CARDS": value})
    assert settings.live_cards is False
    assert settings.index_path == (data / "index" / "visual-s2so400m.npz").resolve()
    assert settings.attrs_path == (data / "gt" / "gt_tokens.jsonl").resolve()
    assert settings.lexicon_path == (data / "index" / "lexicon.json").resolve()


def test_live_cards_default_and_explicit_paths(tmp_path):
    """Пусто — умолчание кода; явный путь главнее флага, но только для своего файла."""
    data = tmp_path / "data"
    assert ServiceSettings.from_env({"SVS_DATA_DIR": str(data)}).live_cards is LIVE_CARDS_DEFAULT
    own = tmp_path / "own.npz"
    settings = ServiceSettings.from_env(
        {
            "SVS_DATA_DIR": str(data),
            "SVS_LIVE_CARDS": "1",
            "SVS_CANDIDATE": "off",  # у живых карточек свой индекс: карта адаптера к нему не идёт
            "SVS_INDEX_PATH": str(own),
        }
    )
    assert settings.index_path == own.resolve()
    assert settings.attrs_path.name == "gt_tokens-live71.jsonl"
    assert settings.lexicon_path.name == "lexicon-live71.json"
