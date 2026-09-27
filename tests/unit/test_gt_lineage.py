"""Сверка разметки каталога с моделью resolve: gt обучения или его объявленная производная.

Модель пути `SVS_CANDIDATE=off` (`-goal`) обучена на `gt_tokens.jsonl` bf6a8823. Правки карточек
Э4 (`gt_fixes.tsv`) дают 899d5db3, комплект живых карточек Э3 поверх них — 003db090. Оба шага
объявлены в `configs/resolve/gt_lineage.json`: с ними `provenance.consistent=true` и `derivation`
называет шаги, живой шаг — только при `SVS_LIVE_CARDS=1`. Любой другой gt — `consistent=false`.
Модель по умолчанию (`-lw-pool`, кандидат 2 фундаментального трека) обучена прямо на 899d5db3:
с данными сервиса она согласована без шагов.

Здесь же — что объявленные sha1 воспроизводятся из файлов репозитория (и из данных сервиса, если
задан `SVS_DATA_DIR`), и что `scripts/run_eval.sh` с такой сборкой идёт без `--allow-degraded`.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from api_env import API_RECORDS, CV_MODEL, RED, image_bytes, make_index, make_model, settings_for
from fakes import ColorEmbedder
from fastapi.testclient import TestClient

import app.api.lineage as lineage_module
from app.api.cards import CatalogCards
from app.api.config import (
    CV_ADAPTER_INDEX_SHA1,
    CV_ADAPTER_SHA1,
    DEFAULT_RESOLVE_MODEL,
    GOAL_RESOLVE_MODEL,
    GT_LINEAGE_PATH,
    MEASURED_READER_NAME,
    MEASURED_VLM_READER,
    REPO_ROOT,
    ServiceSettings,
)
from app.api.lineage import GtLineage, LineageError, LineageStep, default_gt_lineage
from app.api.main import create_app
from app.api.service import (
    ScannerService,
    StartupError,
    load_catalog,
    load_cv_adapter,
    load_lexicon,
    load_resolve_model,
    provenance,
)
from app.reading.lexicon.build import build_from_records
from app.resolve.attrs import CatalogAttrs
from app.resolve.learned import LogisticRanker

TRAINED = "bf6a8823a45004b1e13e586e5a92e05b8beb07b0"
FIXED = "899d5db3747fd315b7c8bfedf121f6ae79799061"
LIVE = "003db090393691e0e7c50c6e810ed1764f70443d"
#: Живой комплект поверх эталона без правок (`after-search`): в сборке с Э4 не объявлен.
LIVE_OLD = "06a294ff0e28b993f5ed4bad15ca8123f4467ef8"
FIXES_SHA1 = "2d7050bfb69c4fa7b936c4734bf875f6ce592391"
LIVE_TAIL_SHA1 = "312a0bc5b7e9e4377f7ec03a7408e8085f6e8474"
OTHER = "f" * 40

FIX_LABEL = "gt_fixes@2d7050bf"
LIVE_LABEL = "gt_fixes@2d7050bf+live71@312a0bc5"


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _goal_model() -> LogisticRanker:
    """Модель пути `off`: обучена на bf6a8823, данные сервиса — её объявленная производная."""
    return LogisticRanker.load(GOAL_RESOLVE_MODEL)


def _report(gt: str | None, *, live: bool = False, lexicon_gt: str | None = "same", **kw):
    """`provenance` модели `-goal` с разметкой `gt` и словарём из `lexicon_gt`."""
    lexicon_source = gt if lexicon_gt == "same" else lexicon_gt
    return provenance(
        _goal_model(),
        CatalogAttrs.from_records([], meta={"sha1": gt} if gt else None),
        build_from_records([], meta={"source_sha1": lexicon_source} if lexicon_source else None),
        reader_key=MEASURED_READER_NAME,
        vlm_reader=MEASURED_VLM_READER,
        live_cards=live,
        **kw,
    )


# ------------------------------------------------------------------ объявление в репозитории
def test_repo_declares_the_e4_fixes_and_the_live_set_over_the_models_training_gt():
    lineage = default_gt_lineage()
    assert lineage == GtLineage.load(GT_LINEAGE_PATH)
    assert [
        (s.name, s.source, s.target, s.input_sha1, s.requires_live_cards) for s in lineage.steps
    ] == [
        ("gt_fixes", TRAINED, FIXED, FIXES_SHA1, False),
        ("live71", FIXED, LIVE, LIVE_TAIL_SHA1, True),
    ]
    for name in ("s2so400m-vlm35-goal.json", "s2so400m-vlm35-m2.json", "s2so400m-vlm35.json"):
        meta = json.loads((REPO_ROOT / "configs" / "resolve" / name).read_text("utf-8"))["meta"]
        assert meta["gt_tokens_sha1"] == TRAINED
    # модель по умолчанию (кандидат 2) обучена уже на разметке с правками Э4 — шаг ей не нужен
    pool = json.loads(DEFAULT_RESOLVE_MODEL.read_text("utf-8"))["meta"]
    assert DEFAULT_RESOLVE_MODEL.name == "s2so400m-vlm35-lw-pool.json"
    assert pool["gt_tokens_sha1"] == FIXED


def test_declared_fix_table_is_the_one_in_the_repo():
    """sha1 входа шага `gt_fixes` — таблица правок репозитория без CR (как в `gt_summary.json`)."""
    step = default_gt_lineage().steps[0]
    table = (REPO_ROOT / "data" / "gt" / "gt_fixes.tsv").read_bytes().replace(b"\r\n", b"\n")
    assert _sha1(table) == step.input_sha1 == FIXES_SHA1


# ------------------------------------------------------------------ provenance
def test_training_gt_is_consistent_without_derivation():
    report = _report(TRAINED)
    assert report["consistent"] is True
    assert report["derivation"] is None and report["mismatch"] == []


@pytest.mark.parametrize("live", [False, True])
def test_e4_gt_is_a_declared_derivation(live):
    report = _report(FIXED, live=live)
    assert report["consistent"] is True and report["derivation"] == FIX_LABEL
    assert (report["gt_tokens_sha1"], report["model_gt_tokens_sha1"]) == (FIXED, TRAINED)


def test_live_set_is_consistent_only_with_the_flag_on():
    on = _report(LIVE, live=True)
    assert on["consistent"] is True and on["derivation"] == LIVE_LABEL
    off = _report(LIVE, live=False)
    assert off["consistent"] is False and off["derivation"] is None
    assert off["mismatch"] == ["gt_tokens"]


@pytest.mark.parametrize("gt", [OTHER, LIVE_OLD])
@pytest.mark.parametrize("live", [False, True])
def test_undeclared_gt_stays_inconsistent(gt, live):
    report = _report(gt, live=live)
    assert report["consistent"] is False and report["derivation"] is None
    assert report["mismatch"] == ["gt_tokens"]


def test_lexicon_must_come_from_the_loaded_gt():
    report = _report(FIXED, lexicon_gt=TRAINED)
    assert report["consistent"] is False and report["mismatch"] == ["lexicon"]
    # без хэша файла признаков разметка сервиса — источник словаря
    assert _report(None, lexicon_gt=FIXED)["derivation"] == FIX_LABEL
    assert _report(None, lexicon_gt=OTHER)["mismatch"] == ["gt_tokens"]


def test_without_declared_steps_the_check_is_strict_equality():
    assert _report(FIXED, lineage=GtLineage())["mismatch"] == ["gt_tokens"]
    assert _report(TRAINED, lineage=GtLineage())["consistent"] is True


def test_reader_mismatch_is_reported_next_to_a_derivation():
    report = provenance(
        _goal_model(),
        CatalogAttrs.from_records([], meta={"sha1": FIXED}),
        build_from_records([], meta={"source_sha1": FIXED}),
        reader_key=MEASURED_READER_NAME,
        vlm_reader="vlm@qwen3-vl:4b-instruct|0",
    )
    assert report["derivation"] == FIX_LABEL
    assert report["consistent"] is False and report["mismatch"] == ["vlm_reader"]


# ------------------------------------------------------------------ модель по умолчанию (кандидат 2)
def _passport(sha1: str = CV_ADAPTER_SHA1) -> SimpleNamespace:
    """Паспорт карты адаптера: `provenance` читает только sha1 содержимого и sha1 индекса."""
    return SimpleNamespace(sha1=sha1, index_sha1=CV_ADAPTER_INDEX_SHA1)


def _pool_report(gt: str, *, adapter: SimpleNamespace | None = None, live: bool = False):
    return provenance(
        LogisticRanker.load(DEFAULT_RESOLVE_MODEL),
        CatalogAttrs.from_records([], meta={"sha1": gt}),
        build_from_records([], meta={"source_sha1": gt}),
        reader_key=MEASURED_READER_NAME,
        vlm_reader=MEASURED_VLM_READER,
        live_cards=live,
        cv_adapter=adapter if adapter is not None else _passport(),  # type: ignore[arg-type]
    )


def test_default_model_is_consistent_with_the_e4_gt_and_its_adapter():
    report = _pool_report(FIXED)
    assert report["consistent"] is True and report["derivation"] is None
    assert report["model_gt_tokens_sha1"] == FIXED
    assert report["model_cv_adapter_sha1"] == report["cv_adapter_sha1"] == CV_ADAPTER_SHA1
    assert report["cv_adapter_index_sha1"] == CV_ADAPTER_INDEX_SHA1
    assert report["vlm_reader_trained"] == [MEASURED_VLM_READER]


def test_default_model_on_the_gt_before_e4_is_inconsistent():
    """Шаги объявлены только вперёд: разметка до правок Э4 — не производная 899d5db3."""
    assert _pool_report(TRAINED)["mismatch"] == ["gt_tokens"]


def test_search_adapter_is_part_of_the_provenance():
    """Модель `-lw-pool` училась на выдаче карты 5d25e5c6, `-goal` — на поиске без карты."""
    assert _pool_report(FIXED, adapter=_passport("0" * 40))["mismatch"] == ["cv_adapter"]
    without = provenance(
        LogisticRanker.load(DEFAULT_RESOLVE_MODEL),
        CatalogAttrs.from_records([], meta={"sha1": FIXED}),
        None,
    )
    assert without["mismatch"] == ["cv_adapter"] and without["cv_adapter_sha1"] is None
    goal = CatalogAttrs.from_records([], meta={"sha1": FIXED})
    assert provenance(_goal_model(), goal, None)["consistent"] is True
    with_map = provenance(_goal_model(), goal, None, cv_adapter=_passport())  # type: ignore[arg-type]
    assert with_map["mismatch"] == ["cv_adapter"]  # кандидат 1: -goal на выдаче карты


# ------------------------------------------------------------------ цепочки и формат файла
def _step(name: str, source: str, target: str, live: bool = False) -> LineageStep:
    return LineageStep(name, source, target, _sha1(name.encode()), live)


A, B, C, D = (_sha1(letter.encode()) for letter in "abcd")


def test_derive_walks_a_chain_and_honours_the_live_flag():
    lineage = GtLineage((_step("one", A, B), _step("two", B, C, live=True), _step("three", C, D)))
    assert lineage.derive(A, A, live_cards=False) == ()
    assert [s.name for s in lineage.derive(A, D, live_cards=True)] == ["one", "two", "three"]
    assert lineage.derive(A, D, live_cards=False) is None  # шаг «two» только с флагом
    assert [s.name for s in lineage.derive(C, D, live_cards=False)] == ["three"]
    assert lineage.derive(B, A, live_cards=True) is None  # назад не выводится
    assert lineage.derive(D, A, live_cards=True) is None


def test_ambiguous_or_cyclic_steps_are_refused():
    with pytest.raises(LineageError, match="двумя шагами"):
        GtLineage((_step("one", A, C), _step("two", B, C)))
    with pytest.raises(LineageError, match="цикл"):
        GtLineage((_step("one", A, B), _step("two", B, A)))


def _raw(**step_fields):
    step = {"name": "gt_fixes", "from": A, "to": B, "input_sha1": C, **step_fields}
    return {"format": "svs-gt-lineage/1", "steps": [step]}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"format": "other", "steps": []}, "format"),
        ({"format": "svs-gt-lineage/1", "steps": "gt_fixes"}, "steps"),
        (_raw(name="Правки"), "name"),
        (_raw(to="899d5db3"), "to"),
        (_raw(input_sha1=None), "input_sha1"),
        (_raw(to=A), "совпадают"),
        (_raw(requires_live_cards="yes"), "requires_live_cards"),
    ],
)
def test_malformed_lineage_is_an_error(data, message):
    with pytest.raises(LineageError, match=message):
        GtLineage.from_dict(data)


def test_lineage_file_missing_means_strict_and_broken_file_is_an_error(tmp_path):
    assert GtLineage.load(tmp_path / "none.json") == GtLineage()
    broken = tmp_path / "gt_lineage.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(LineageError, match="gt_lineage.json"):
        GtLineage.load(broken)


# ------------------------------------------------------------------ сервис и /v1/health
def _service(gt: str, *, live: bool = False, lineage: GtLineage | None = None) -> ScannerService:
    model = make_model()
    model.meta["gt_tokens_sha1"] = TRAINED
    return ScannerService(
        settings_for(live_cards=live),
        index=make_index(),
        embedder=ColorEmbedder(CV_MODEL),
        lexicon=build_from_records(API_RECORDS, meta={"source_sha1": gt}),
        attrs=CatalogAttrs.from_records(API_RECORDS, meta={"sha1": gt}),
        model=model,
        vlm=None,
        cards=CatalogCards.build(API_RECORDS),
        lineage=lineage,
    )


@pytest.mark.parametrize(
    ("gt", "live", "consistent", "derivation"),
    [
        (TRAINED, False, True, None),
        (FIXED, False, True, FIX_LABEL),
        (LIVE, True, True, LIVE_LABEL),
        (LIVE, False, False, None),
        (OTHER, False, False, None),
    ],
)
def test_health_reports_the_derivation(gt, live, consistent, derivation):
    service = _service(gt, live=live)
    with TestClient(create_app(service=service, warm=False)) as client:
        report = client.get("/v1/health").json()["provenance"]
    assert (report["consistent"], report["derivation"]) == (consistent, derivation)
    assert report["gt_tokens_sha1"] == gt and report["model_gt_tokens_sha1"] == TRAINED


@pytest.fixture
def broken_repo_lineage(tmp_path, monkeypatch):
    """Файл шагов репозитория испорчен: `default_gt_lineage` читает его заново."""
    path = tmp_path / "gt_lineage.json"
    path.write_text(json.dumps({"format": "svs-gt-lineage/0", "steps": []}), encoding="utf-8")
    monkeypatch.setattr(lineage_module, "GT_LINEAGE_PATH", path)
    default_gt_lineage.cache_clear()
    yield path
    default_gt_lineage.cache_clear()


def test_service_start_refuses_a_broken_lineage(broken_repo_lineage):
    with pytest.raises(StartupError, match="объявленные производные.*format"):
        _service(FIXED)
    assert _service(TRAINED).provenance["consistent"] is True  # шаги не нужны — файл не читается


def test_derivation_changes_no_answer():
    """Сверка только отчитывается: ответ и счёт кандидатов те же, что без объявленных шагов."""
    frame = image_bytes(RED)
    declared, strict = _service(FIXED).scan(frame), _service(FIXED, lineage=GtLineage()).scan(frame)
    assert declared.slug == strict.slug and declared.outcome == strict.outcome
    assert [(i.slug, i.score) for i in declared.top5] == [(i.slug, i.score) for i in strict.top5]
    assert declared.confidence == strict.confidence


# ------------------------------------------------------------------ run_eval.sh
@cache
def _bash() -> str | None:
    """bash Git for Windows (или системный на Linux) с curl, jq, awk, mktemp в /tmp."""
    candidates: list[str | None] = []
    git = shutil.which("git")
    if os.name == "nt" and git:
        candidates.append(str(Path(git).resolve().parents[1] / "bin" / "bash.exe"))
    candidates.append(shutil.which("bash"))
    probe = (
        'export PATH="$HOME/bin:$PATH"; for c in curl jq awk mktemp; do command -v "$c" '
        ">/dev/null || exit 3; done; command -v sha256sum >/dev/null || command -v shasum "
        '>/dev/null || exit 3; d=$(mktemp -d) || exit 3; rm -rf "$d"; case "$d" in /tmp/*|'
        "/private/tmp/*|/var/tmp/*|/var/folders/*|/private/var/folders/*) ;; *) exit 3 ;; esac"
    )
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file() or "system32" in candidate.lower():
            continue
        try:
            done = subprocess.run(
                [candidate, "-c", probe], capture_output=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return candidate
    return None


class _Health(http.server.BaseHTTPRequestHandler):
    body = b"{}"

    def do_GET(self) -> None:
        if self.path != "/v1/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args: object) -> None:
        pass


FAKE_PARTICIPANT = """#!/usr/bin/env bash
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in --output) out="$2"; shift 2 ;; *) shift 2 ;; esac
done
printf '{"query_id":"q-1","image_path":"a.jpg","image_sha256":"0","predicted_slug":"alfa-muskat","latency_ms":5}\\n' > "$out"
"""


@pytest.mark.parametrize(
    ("gt", "live", "allow", "code"),
    [
        (FIXED, False, False, 0),  # эталон Э4 — отчётный прогон без --allow-degraded
        (LIVE, True, False, 0),  # живой комплект при включённом флаге
        (LIVE, False, False, 2),
        (OTHER, False, False, 2),
        (OTHER, False, True, 0),  # --allow-degraded снимает именно эту проверку
    ],
)
def test_run_eval_readiness_follows_provenance(tmp_path, gt, live, allow, code):
    bash = _bash()
    if bash is None:
        pytest.skip("нужен bash с curl, jq, awk, sha256sum и mktemp в /tmp")
    health = {
        "status": "ready",
        "degraded_reasons": [],
        "warnings": [],
        "warm": {"scan": {"degraded": []}},
        "provenance": _report(gt, live=live),
        "scans": {"total": 0, "text_read": 0},
    }
    handler = type("Health", (_Health,), {"body": json.dumps(health).encode("utf-8")})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        (tmp_path / "imgs").mkdir()
        (tmp_path / "queries.tsv").write_text("q-1\ta.jpg\n", encoding="utf-8")
        (tmp_path / "gt.tsv").write_text("query_id\tslug\nq-1\talfa-muskat\n", encoding="utf-8")
        script = tmp_path / "participant_test.sh"
        script.write_bytes(FAKE_PARTICIPANT.encode("utf-8"))
        port = server.server_address[1]
        args = [
            bash, (REPO_ROOT / "scripts" / "run_eval.sh").as_posix(),
            "--images-dir", (tmp_path / "imgs").as_posix(),
            "--manifest", (tmp_path / "queries.tsv").as_posix(),
            "--gt", (tmp_path / "gt.tsv").as_posix(),
            "--out", (tmp_path / "out").as_posix(),
            "--endpoint", f"http://127.0.0.1:{port}/v1/eval/predict",
            "--script", script.as_posix(),
            "--python", Path(sys.executable).as_posix(),
        ] + (["--allow-degraded"] if allow else [])  # fmt: skip
        env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
        env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        done = subprocess.run(args, capture_output=True, timeout=180, env=env, check=False)
    finally:
        server.shutdown()
        server.server_close()
    out = done.stdout.decode("utf-8", "replace")
    err = done.stderr.decode("utf-8", "replace")
    assert done.returncode == code, out + err
    if code == 0 and not allow:
        assert "готово:" in out and "ВНИМАНИЕ: сервис не на цепочке замера" not in out
        assert (tmp_path / "out" / "judge.json").is_file()
    elif code == 0:
        assert "ВНИМАНИЕ: сервис не на цепочке замера" in out
    else:
        assert "сборка не та, на которой обучен resolve" in err
        assert "сервис не готов к отчётному прогону" in err


# ------------------------------------------------------------------ данные сервиса
DATA = os.environ.get("SVS_DATA_DIR")
needs_data = pytest.mark.skipif(
    not DATA or not (Path(DATA) / "gt" / "gt_tokens.jsonl").is_file(),
    reason="нужны данные сервиса: SVS_DATA_DIR",
)

#: (sha1 разметки в данных сервиса, живые карточки, путь `SVS_CANDIDATE`) → (согласована ли с
#: моделью этого пути, derivation). Живые карточки — только путь `off` (у карты свой индекс).
EXPECTED_ON_DATA = {
    (TRAINED, False, "off"): (True, None),
    (FIXED, False, "off"): (True, FIX_LABEL),
    (LIVE, True, "off"): (True, LIVE_LABEL),
    (LIVE_OLD, True, "off"): (False, None),  # данные после отката на bf6a88
    (FIXED, False, "adapter-lw-ranker"): (True, None),
    (TRAINED, False, "adapter-lw-ranker"): (False, None),
}


@needs_data
@pytest.mark.parametrize(
    ("live", "candidate"), [(False, "adapter-lw-ranker"), (False, "off"), (True, "off")]
)
def test_service_data_is_the_training_gt_or_a_declared_derivation(live, candidate):
    assert DATA is not None
    settings = ServiceSettings.from_env(
        {"SVS_DATA_DIR": DATA, "SVS_LIVE_CARDS": "1" if live else "0", "SVS_CANDIDATE": candidate}
    )
    if not (settings.attrs_path.is_file() and settings.lexicon_path.is_file()):
        pytest.skip(f"нет {settings.attrs_path.name} или {settings.lexicon_path.name}")
    _, attrs = load_catalog(settings.attrs_path)
    # Карта адаптера — часть данных пути по умолчанию: нет её или она от другого индекса — отказ.
    adapter = (
        load_cv_adapter(settings.cv_adapter, settings.index_path)
        if settings.cv_adapter is not None
        else None
    )
    report = provenance(
        load_resolve_model(settings.resolve_model),
        attrs,
        load_lexicon(settings.lexicon_path),
        reader_key=MEASURED_READER_NAME,
        vlm_reader=MEASURED_VLM_READER,
        live_cards=settings.live_cards,
        cv_adapter=adapter,
    )
    key = (report["gt_tokens_sha1"], live, candidate)
    if key not in EXPECTED_ON_DATA:
        pytest.skip(f"разметка {key[0][:8]} — не из сборок 25.09")
    assert (report["consistent"], report["derivation"]) == EXPECTED_ON_DATA[key], report


@needs_data
def test_declared_steps_match_the_service_data_files():
    """Файлы данных — то, что объявлено: таблица правок в `gt_summary.json`, хвост живого gt."""
    assert DATA is not None
    gt_dir = Path(DATA) / "gt"
    fixes, live = default_gt_lineage().steps
    base = (gt_dir / "gt_tokens.jsonl").read_bytes()
    if _sha1(base) != fixes.target:
        pytest.skip("в данных не эталон Э4")
    summary = json.loads((gt_dir / "gt_summary.json").read_text(encoding="utf-8"))
    assert summary["gt_fixes"]["sha1"] == fixes.input_sha1
    previous = gt_dir / "bf6a88" / "gt_tokens.jsonl"
    if previous.is_file():
        assert _sha1(previous.read_bytes()) == fixes.source
    live_path = gt_dir / "gt_tokens-live71.jsonl"
    if live_path.is_file():
        live_bytes = live_path.read_bytes()
        assert _sha1(live_bytes) == live.target and live.source == fixes.target
        assert live_bytes.startswith(base)
        assert _sha1(live_bytes[len(base) :]) == live.input_sha1
