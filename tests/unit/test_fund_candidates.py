"""Путь поиска и выбора `SVS_CANDIDATE` (`research/2026-09-26_fund/PREREG_final.md`).

С 26.09 по умолчанию — кандидат 2 `adapter-lw-ranker`, принятый по замороженному kr-test: линейный
адаптер поиска (`app/features/adapter.py`) — строки индекса проецируются при загрузке, запрос — в
`_rank`, дальше тот же `zmax` — и модель `-lw-pool`, обученная на его выдаче. Модель без «своей»
карты не стартует, карта с чужим индексом — тоже. `off` (`none`) — прежний путь без адаптера:
равенство ответов с `after-search` на v2, kr-dev, ooc_v2 и R-оригиналах проверяет
`research/2026-09-26_fund/ship/ship_gate.py` (данные вне репо). Паспорт настоящих файлов карты и
модели сверяется с данными сервиса, если задан `SVS_DATA_DIR`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from api_env import (
    API_RECORDS,
    CV_MODEL,
    RED,
    ColorEmbedder,
    FakeOllama,
    image_bytes,
    make_index,
    make_model,
    make_service,
    settings_for,
)

from app.api.cards import CatalogCards
from app.api.config import (
    CANDIDATE_DEFAULT,
    CANDIDATE_RESOLVE_MODELS,
    CV_ADAPTER_INDEX_SHA1,
    CV_ADAPTER_NAME,
    CV_ADAPTER_SHA1,
    DEFAULT_RESOLVE_MODEL,
    GOAL_RESOLVE_MODEL,
    MEASURED_READER_NAME,
    MEASURED_VLM_READER,
    POOL_RESOLVE_MODEL,
    REPO_ROOT,
    ServiceSettings,
    SettingsError,
)
from app.api.service import (
    ScannerService,
    StartupError,
    load_catalog,
    load_cv_adapter,
    load_lexicon,
    load_resolve_model,
    provenance,
)
from app.features.adapter import ADAPTER_FORMAT, AdapterError, LinearAdapter, file_sha1
from app.features.contracts import IndexMeta
from app.features.index import IndexMismatch, VisualIndex
from app.reading.lexicon.build import build_from_records
from app.reading.readers.ollama_vlm import OllamaVlmReader
from app.resolve.attrs import CatalogAttrs

#: Файл карты кандидата 2 (`PREREG_final.md`, §2.1): sha1 файла целиком, вместе с `meta`.
CV_ADAPTER_FILE_SHA1 = "206627c7a95a759a781db21b8ee5c70d9e0c758b"
#: sha1 модели `-lw-pool` по содержимому с LF (`PREREG_final.md`, §2.3).
POOL_MODEL_SHA1_LF = "b82580e22d89aa3d5fd0e9a9678ed29e1fecb809"
#: Разметка каталога, на которой обучена модель `-lw-pool` (правки карточек Э4).
FIXED_GT = "899d5db3747fd315b7c8bfedf121f6ae79799061"


def rotation(dim: int = 3, seed: int = 0) -> LinearAdapter:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(dim, dim)).astype(np.float32)
    mean = (0.01 * rng.normal(size=dim)).astype(np.float32)
    return LinearAdapter(mean, W, {"kind": "lw", "name": "test"})


# ------------------------------------------------------------------ настройки
def test_candidate_2_is_the_default(tmp_path):
    """Без переменной — кандидат 2: карта рядом с индексом, модель `-lw-pool`, без предупреждений."""
    s = ServiceSettings.from_env({"SVS_DATA_DIR": str(tmp_path)})
    assert CANDIDATE_DEFAULT == s.candidate == "adapter-lw-ranker"
    assert s.cv_adapter == tmp_path.resolve() / "index" / CV_ADAPTER_NAME
    assert s.resolve_model == DEFAULT_RESOLVE_MODEL == POOL_RESOLVE_MODEL
    assert s.warnings() == []
    explicit = {"SVS_DATA_DIR": str(tmp_path), "SVS_CANDIDATE": "adapter-lw-ranker"}
    assert ServiceSettings.from_env(explicit) == s
    # поля класса — те же умолчания: так стартует `python -m app.api`
    plain = ServiceSettings()
    assert plain.candidate == "adapter-lw-ranker" and plain.resolve_model == POOL_RESOLVE_MODEL
    assert plain.cv_adapter == REPO_ROOT / "data" / "index" / CV_ADAPTER_NAME


@pytest.mark.parametrize("word", ["off", "none", "OFF", " None "])
def test_off_and_none_are_the_old_path(tmp_path, word):
    """`off` / `none` — прежний продукт: без карты, модель `-goal`, без предупреждений."""
    s = ServiceSettings.from_env({"SVS_DATA_DIR": str(tmp_path), "SVS_CANDIDATE": word})
    assert s.candidate == "off" and s.cv_adapter is None
    assert s.resolve_model == GOAL_RESOLVE_MODEL == CANDIDATE_RESOLVE_MODELS["off"]
    assert s.warnings() == []
    assert s.public()["candidate"] == "off" and s.public()["cv_adapter"] is None


def test_adapter_lw_keeps_the_goal_ranker_and_warns(tmp_path):
    """Кандидат 1 не принят (R-оригиналы 61 < 62): только для повторов, с предупреждением."""
    s = ServiceSettings.from_env({"SVS_DATA_DIR": str(tmp_path), "SVS_CANDIDATE": "adapter-lw"})
    assert s.cv_adapter == tmp_path.resolve() / "index" / CV_ADAPTER_NAME
    assert s.resolve_model == GOAL_RESOLVE_MODEL
    assert any("SVS_CANDIDATE=adapter-lw" in w and "не принятый" in w for w in s.warnings())


def test_explicit_adapter_and_model_paths(tmp_path):
    explicit = tmp_path / "a.npz"
    s = ServiceSettings.from_env({"SVS_CV_ADAPTER": str(explicit)})
    assert s.candidate == "adapter-lw-ranker" and s.cv_adapter == explicit.resolve()
    own = tmp_path / "own.json"
    s2 = ServiceSettings.from_env({"SVS_CANDIDATE": "off", "SVS_RESOLVE_MODEL": str(own)})
    assert s2.resolve_model == own.resolve()


@pytest.mark.parametrize(
    "env",
    [
        {"SVS_CANDIDATE": "lw"},
        {"SVS_CANDIDATE": "off", "SVS_CV_ADAPTER": "x.npz"},
        {"SVS_CANDIDATE": "none", "SVS_CV_ADAPTER": "x.npz"},
        # у живых карточек свой индекс: карта адаптера к нему не подходит — нужен off
        {"SVS_LIVE_CARDS": "1"},
        {"SVS_LIVE_CARDS": "1", "SVS_CANDIDATE": "adapter-lw-ranker"},
    ],
)
def test_bad_candidate_settings_are_refused(env):
    with pytest.raises(SettingsError):
        ServiceSettings.from_env(env)


def test_live_cards_need_the_old_path():
    with pytest.raises(SettingsError, match="SVS_CANDIDATE=off"):
        ServiceSettings.from_env({"SVS_LIVE_CARDS": "1"})
    s = ServiceSettings.from_env({"SVS_LIVE_CARDS": "1", "SVS_CANDIDATE": "off"})
    assert s.live_cards and s.cv_adapter is None and s.resolve_model == GOAL_RESOLVE_MODEL


def test_settings_object_refuses_half_a_candidate(tmp_path):
    with pytest.raises(SettingsError):
        settings_for(tmp_path, candidate="adapter-lw")
    with pytest.raises(SettingsError):
        settings_for(tmp_path, cv_adapter=tmp_path / "a.npz")
    with pytest.raises(SettingsError, match="SVS_LIVE_CARDS"):
        settings_for(
            tmp_path, candidate="adapter-lw-ranker", cv_adapter=tmp_path / "a.npz", live_cards=True
        )


# ------------------------------------------------------------------ паспорт модели по умолчанию
def test_default_model_declares_its_adapter_gt_and_reader():
    """Модель `-lw-pool` в репозитории — та, что предрегистрирована и принята (sha1 с LF), и сама
    называет свою карту, разметку каталога и читателя: по ним сервис сверяет сборку при старте."""
    import hashlib

    raw = POOL_RESOLVE_MODEL.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha1(raw).hexdigest() == POOL_MODEL_SHA1_LF
    meta = json.loads(raw)["meta"]
    assert meta["cv_adapter_sha1"] == CV_ADAPTER_SHA1
    assert meta["cv_adapter_file"] == CV_ADAPTER_NAME
    assert meta["gt_tokens_sha1"] == FIXED_GT
    assert meta["reader_keys"] == {MEASURED_READER_NAME: [MEASURED_VLM_READER]}
    assert meta["top_k"] == 20 and meta["feature_version"] == "resolve-features/3"
    assert "kr_test" in meta and "krasnostop_v1" not in meta["sets"]  # kr-test не в обучении


# ------------------------------------------------------------------ адаптер
def test_apply_is_the_track_formula():
    a = rotation()
    X = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float32)
    Y = (X - a.mean) @ a.W
    Y = Y / np.maximum(np.linalg.norm(Y, axis=-1, keepdims=True), 1e-12)
    assert np.array_equal(a.apply(X), Y.astype(np.float32))
    assert np.allclose(np.linalg.norm(a.apply(X), axis=1), 1.0, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path):
    a = rotation()
    path = a.save(tmp_path / "a.npz")
    b = LinearAdapter.load(path)
    assert b.sha1 == a.sha1
    assert np.array_equal(b.W, a.W) and np.array_equal(b.mean, a.mean)
    assert b.meta["format"] == ADAPTER_FORMAT and b.meta["content_sha1"] == a.sha1


def test_load_refuses_foreign_or_tampered_files(tmp_path):
    a = rotation()
    np.savez(tmp_path / "raw.npz", mean=a.mean, W=a.W)
    with pytest.raises(AdapterError):
        LinearAdapter.load(tmp_path / "raw.npz")
    np.savez(
        tmp_path / "bad.npz",
        mean=a.mean,
        W=a.W * 2,
        meta=np.array(f'{{"format": "{ADAPTER_FORMAT}", "content_sha1": "{a.sha1}"}}'),
    )
    with pytest.raises(AdapterError):
        LinearAdapter.load(tmp_path / "bad.npz")
    with pytest.raises(AdapterError):
        LinearAdapter(a.mean.astype(np.float64), a.W)


# ------------------------------------------------------------------ индекс
def random_index(seed: int = 0) -> tuple[VisualIndex, np.ndarray]:
    rng = np.random.default_rng(seed)
    slugs = [f"s{i}" for i in range(12) for _ in range(2)]
    views = ["bottle", "label"] * 12
    V = rng.normal(size=(24, 3)).astype(np.float32)
    meta = IndexMeta(model=CV_MODEL, dim=3, views=["bottle", "label"], n_slugs=12, n_vectors=24)
    Q = rng.normal(size=(4, 3)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return VisualIndex(slugs, views, V, meta), Q


@pytest.mark.parametrize("per_slug", ["max", "zmax"])
def test_index_without_adapter_is_unchanged(per_slug):
    idx, Q = random_index()
    assert idx.adapter is None
    again, _ = random_index()
    assert idx._rank(Q, 5, per_slug) == again._rank(Q, 5, per_slug)


def test_adapted_index_ranks_in_the_adapter_space():
    idx, Q = random_index()
    a = rotation()
    ad = idx.with_adapter(a)
    assert idx.adapter is None and ad.adapter is a  # исходный индекс не тронут
    assert np.array_equal(ad.vectors, a.apply(idx.vectors))
    cands, margin = ad._rank(Q, 5, "zmax")
    # то же руками: проекция запроса, косинусы, zmax
    from app.features.index import align_pairs

    sims = align_pairs(ad.vectors @ a.apply(Q).T, ad.view_groups, None)
    best = sims.max(axis=1)
    scores = np.full(ad.n_slugs, -np.inf, dtype=np.float32)
    np.maximum.at(scores, ad.slug_ids, best)
    order = np.argsort(-scores, kind="stable")
    assert [c.slug for c in cands] == [ad.slug_order[i] for i in order[:5]]
    assert margin == pytest.approx(float(scores[order[0]] - scores[order[1]]))
    with pytest.raises(IndexMismatch):
        ad.with_adapter(a)
    with pytest.raises(IndexMismatch):
        ad.save("unused.npz")


# ------------------------------------------------------------------ сервис
def adapted_service(tmp_path, adapter: LinearAdapter, model=None, **meta) -> ScannerService:
    adapter = LinearAdapter(adapter.mean, adapter.W, {**adapter.meta, **meta})
    path = adapter.save(tmp_path / "a.npz")
    settings = settings_for(tmp_path, candidate="adapter-lw", cv_adapter=path)
    reader = OllamaVlmReader(
        settings.vlm_model,
        settings.ollama_url,
        transport=FakeOllama("Бета Холмы\nМерло"),
        probe=lambda: {"models": [{"name": settings.vlm_model}]},
    )
    return ScannerService(
        settings,
        index=make_index().with_adapter(LinearAdapter.load(path)),
        embedder=ColorEmbedder(CV_MODEL),
        lexicon=build_from_records(API_RECORDS),
        attrs=CatalogAttrs.from_records(API_RECORDS),
        model=model or make_model(settings.top_k),
        vlm=reader,
        cards=CatalogCards.build(API_RECORDS),
    )


def identity() -> LinearAdapter:
    return LinearAdapter(np.zeros(3, np.float32), np.eye(3, dtype=np.float32), {"name": "id"})


def test_identity_adapter_gives_the_same_answer(tmp_path):
    plain = make_service(FakeOllama("Бета Холмы\nМерло"))
    ad = adapted_service(tmp_path, identity())
    a, b = plain.scan(image_bytes(RED)), ad.scan(image_bytes(RED))
    assert a.slug == b.slug == "beta-merlot"
    assert ad.health()["settings"]["candidate"] == "adapter-lw"
    assert ad.health()["model"]["cv_adapter"]["sha1"] == identity().sha1
    assert plain.health()["model"]["cv_adapter"] is None


def test_adapter_thresholds_replace_the_scale_constants(tmp_path):
    plain = make_service()
    ad = adapted_service(
        tmp_path, identity(), suggest_not_found_visual_max=0.5, abstain_visual_floor=0.4
    )
    assert ad.after.suggest_max == 0.5 and plain.after.suggest_max != 0.5
    assert ad._abstain_cfg.visual_floor == 0.4 and ad._abstain_cfg_off.visual_floor == 0.4
    assert plain._abstain_cfg.visual_floor == 0.75


def test_recalibrated_suggest_threshold_overrides_the_adapter_meta(tmp_path, monkeypatch):
    """Порог подсказки карты из `SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER` сильнее её меты."""
    import app.api.service as service_mod

    table = {identity().sha1: 0.61}
    monkeypatch.setattr(service_mod, "SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER", table)
    ad = adapted_service(tmp_path, identity(), suggest_not_found_visual_max=0.5)
    assert ad.after.suggest_max == 0.61
    assert ad.health()["model"]["suggest_not_found_visual_max"] == 0.61
    assert ad.health()["model"]["cv_adapter"]["suggest_not_found_visual_max"] == 0.5


def test_recalibrated_suggest_threshold_is_for_the_shipped_adapter_only():
    """Перекалибровка 26.09 привязана к содержимому карты кандидата 2 и лежит ниже порога продукта."""
    from app.api.config import SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER

    assert set(SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER) == {CV_ADAPTER_SHA1}
    assert 0.3475 < SUGGEST_NOT_FOUND_VISUAL_MAX_BY_ADAPTER[CV_ADAPTER_SHA1] < 0.8024


def test_model_trained_on_an_adapter_needs_that_adapter(tmp_path):
    model = make_model()
    model.meta["cv_adapter_sha1"] = identity().sha1
    with pytest.raises(StartupError, match="адаптер"):
        make_service(model=model)
    other = rotation()
    with pytest.raises(StartupError, match="адаптер"):
        adapted_service(tmp_path, other, model=model)
    assert adapted_service(tmp_path, identity(), model=model).model is model


def test_candidate_without_adapter_does_not_start(tmp_path):
    path = identity().save(tmp_path / "a.npz")
    settings = settings_for(tmp_path, candidate="adapter-lw", cv_adapter=path)
    with pytest.raises(StartupError):
        make_service(settings=settings)


def test_adapter_for_another_index_is_refused(tmp_path):
    idx_path = tmp_path / "index.npz"
    make_index().save(idx_path)
    a = LinearAdapter(identity().mean, identity().W, {"index_sha1": "0" * 40})
    path = a.save(tmp_path / "a.npz")
    with pytest.raises(StartupError, match="индекса.*SVS_CANDIDATE=off"):
        load_cv_adapter(path, idx_path)
    good = LinearAdapter(a.mean, a.W, {"index_sha1": file_sha1(idx_path)}).save(tmp_path / "b.npz")
    assert load_cv_adapter(good, idx_path).index_sha1 == file_sha1(idx_path)


def test_provenance_names_the_adapter_the_model_was_trained_on(tmp_path):
    """`/v1/health.provenance`: карта сервиса = карта обучения модели — согласовано; модель без
    записи о карте (кандидат 1: `-goal` на выдаче карты) — `mismatch: ["cv_adapter"]`."""
    model = make_model()
    model.meta["cv_adapter_sha1"] = identity().sha1
    trained = adapted_service(tmp_path, identity(), model=model).health()["provenance"]
    assert trained["consistent"] is True
    assert trained["model_cv_adapter_sha1"] == trained["cv_adapter_sha1"] == identity().sha1
    other = adapted_service(tmp_path, identity()).health()["provenance"]
    assert other["consistent"] is False and other["mismatch"] == ["cv_adapter"]
    plain = make_service().health()["provenance"]
    assert plain["consistent"] is True and plain["cv_adapter_sha1"] is None


def test_refusals_say_how_to_run_the_old_path(tmp_path):
    """Нет карты или карта от другого индекса — отказ старта с подсказкой `SVS_CANDIDATE=off`."""
    idx_path = tmp_path / "index.npz"
    make_index().save(idx_path)
    with pytest.raises(StartupError, match="SVS_CANDIDATE=off") as missing:
        load_cv_adapter(tmp_path / "none.npz", idx_path)
    assert "pack_data" in str(missing.value)
    foreign = LinearAdapter(identity().mean, identity().W, {"index_sha1": "0" * 40})
    with pytest.raises(StartupError, match="SVS_CANDIDATE=off"):
        load_cv_adapter(foreign.save(tmp_path / "a.npz"), idx_path)


# ------------------------------------------------------------------ данные сервиса
DATA = os.environ.get("SVS_DATA_DIR")
needs_data = pytest.mark.skipif(
    not DATA or not (Path(DATA) / "gt" / "gt_tokens.jsonl").is_file(),
    reason="нужны данные сервиса: SVS_DATA_DIR",
)


@needs_data
def test_data_pack_carries_the_adapter_of_the_default_path():
    """Пачка данных пути по умолчанию: карта `index/cv-adapter-lw.npz` — предрегистрированный файл,
    собранный под этот индекс, и модель `-lw-pool` с этой разметкой согласована без шагов."""
    assert DATA is not None
    settings = ServiceSettings.from_env({"SVS_DATA_DIR": DATA})
    assert settings.cv_adapter is not None
    assert settings.cv_adapter.is_file(), (
        f"нет {settings.cv_adapter}: путь по умолчанию без карты не стартует (deploy/pack_data.sh)"
    )
    assert file_sha1(settings.cv_adapter) == CV_ADAPTER_FILE_SHA1
    assert file_sha1(settings.index_path) == CV_ADAPTER_INDEX_SHA1
    adapter = load_cv_adapter(settings.cv_adapter, settings.index_path)
    assert adapter.sha1 == CV_ADAPTER_SHA1 and adapter.index_sha1 == CV_ADAPTER_INDEX_SHA1
    assert adapter.dim_in == adapter.dim_out == 1152
    _, attrs = load_catalog(settings.attrs_path)
    report = provenance(
        load_resolve_model(settings.resolve_model),
        attrs,
        load_lexicon(settings.lexicon_path),
        reader_key=MEASURED_READER_NAME,
        vlm_reader=MEASURED_VLM_READER,
        cv_adapter=adapter,
    )
    assert report["consistent"] is True and report["derivation"] is None, report
    assert report["gt_tokens_sha1"] == FIXED_GT


@needs_data
def test_real_adapter_refuses_another_index():
    """Живой индекс (`SVS_LIVE_CARDS`) — другой файл: настоящая карта с ним не стартует."""
    assert DATA is not None
    live_index = Path(DATA) / "index" / "visual-s2so400m-live71.npz"
    adapter_file = Path(DATA) / "index" / CV_ADAPTER_NAME
    if not (live_index.is_file() and adapter_file.is_file()):
        pytest.skip("нет живого индекса или карты в данных сервиса")
    with pytest.raises(StartupError, match=f"{CV_ADAPTER_INDEX_SHA1[:8]}.*SVS_CANDIDATE=off"):
        load_cv_adapter(adapter_file, live_index)
