"""Инструменты цикла улучшений (`scripts/goal.py`): правило приёмки, откат, состояние, наборы.

Прогоны `bench.train_resolve` здесь синтетические: `metrics.json` и `oof_predictions.jsonl` в том
виде, в каком их пишет стенд, с заданными верными ответами по наборам.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from app.reading.contracts import TextLine
from app.reading.readers.base import make_reading
from app.reading.readers.cache import CachedReader

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
N = 20  # запросов на набор в синтетическом прогоне
BASE_RIGHT = 10  # верны первые 10
BASE_TOP5 = 15  # в top-5 первые 15
#: Паспорт `metrics.json → run`, как у `runs/resolve-s2so400m-m2`.
PASSPORT = {
    "split": "dev",
    "folds": 5,
    "signed": True,
    "top_k": 20,
    "reader_keys": {"vlm35": ["vlm@qwen3.5:4b|f3a017317f04"]},
    "cv_sha1": ["cv-pairs", "cv-phone"],
    "feature_version": "resolve-features/3",
}


@pytest.fixture(scope="module")
def goal():
    """Скрипт как модуль, как в остальных тестах `scripts/`; в `sys.modules` — ради dataclass."""
    spec = importlib.util.spec_from_file_location("script_goal", SCRIPTS / "goal.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OK_PROVENANCE = {"consistent": True, "startup_problems": []}


# ------------------------------------------------------------------ синтетические прогоны
def flags(fix: int = 0, brk: int = 0) -> list[bool]:
    """Верные ответы кандидата: из базы (первые 10 верны) исправлены `fix`, сломаны `brk`."""
    right = [i < BASE_RIGHT for i in range(N)]
    for i in range(BASE_RIGHT, BASE_RIGHT + fix):
        right[i] = True
    for i in range(brk):
        right[i] = False
    return right


def write_run(
    path: Path,
    right: dict[str, list[bool]],
    *,
    top5: dict[str, list[bool]] | None = None,
    ece_all: float = 0.0326,
    drop: tuple[str, str] | None = None,
    run: dict | None = None,
    test_used: bool = False,
) -> Path:
    """Прогон train_resolve: вариант vlm35 и шумный вариант none, который вердикт не читает."""
    path.mkdir(parents=True, exist_ok=True)
    rows, by_set = [], {}
    for set_name, correct in right.items():
        hit5 = (top5 or {}).get(set_name) or [i < BASE_TOP5 for i in range(len(correct))]
        kept = 0
        for i, ok in enumerate(correct):
            qid = f"q{i:02d}"
            if drop == (set_name, qid):
                continue
            kept += 1
            slug = f"wine-{i}"
            top1 = slug if ok else f"other-{i}"
            learned_top5 = [top1, *([slug] if hit5[i] and not ok else []), "x-1", "x-2"][:5]
            common = {"set": set_name, "query_id": qid, "split": "dev", "slug": slug}
            rows.append(
                {
                    "variant": "vlm35",
                    **common,
                    "learned_top1": top1,
                    "learned_top5": learned_top5,
                    "learned_correct": ok,
                    "p_top1": 0.9,
                }
            )
            rows.append({"variant": "none", **common, "learned_correct": not ok})
        n = kept
        hits1 = sum(ok for i, ok in enumerate(correct) if drop != (set_name, f"q{i:02d}"))
        hits5 = sum(
            ok or hit5[i] for i, ok in enumerate(correct) if drop != (set_name, f"q{i:02d}")
        )
        by_set[set_name] = {
            "n": n,
            "learned": {"top1": round(hits1 / n, 6), "top5": round(hits5 / n, 6), "n": n},
            "calibration": {"ece": 0.04},
        }
    metrics = {
        "test_used": test_used,
        "run": {**PASSPORT, **(run or {})},
        "variants": {
            "vlm35": {"oof": {"by_set": by_set, "all": {"calibration": {"ece": ece_all}}}},
            "none": {"oof": {"by_set": {}, "all": {}}},
        },
    }
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (path / "oof_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


@pytest.fixture
def base_run(tmp_path):
    return write_run(tmp_path / "base", {"pairs": flags(), "pairs_phone": flags()})


def verdict(goal, base: Path, cand: Path, *, p95=None, provenance=OK_PROVENANCE, changed=()):
    return goal.decide(
        goal.load_resolve_run(base),
        goal.load_resolve_run(cand),
        p95_ms=p95,
        provenance=provenance,
        changed=changed,
    )


# ------------------------------------------------------------------ правило приёмки
@pytest.mark.parametrize(
    ("studio", "phone", "accepted"),
    [
        ((3, 0), (3, 0), True),  # ровно 3 и 3
        ((4, 1), (3, 0), True),  # net 3 при поломке
        ((2, 0), (3, 0), True),  # 5 в сумме
        ((5, 0), (0, 0), True),  # 5 в сумме, второй набор 0 — не отрицательный
        ((4, 0), (0, 0), False),  # 4 в сумме
        ((2, 0), (2, 0), False),
        ((6, 0), (0, 1), False),  # сумма 5, но «телефон» −1
        ((3, 0), (2, 0), True),  # 5 в сумме, 3 и 2
    ],
)
def test_gain_rule(goal, tmp_path, base_run, studio, phone, accepted):
    cand = write_run(tmp_path / "cand", {"pairs": flags(*studio), "pairs_phone": flags(*phone)})
    result = verdict(goal, base_run, cand)
    assert result["accepted"] is accepted
    assert result["rule"]["net_studio"] == studio[0] - studio[1]
    assert result["rule"]["net_phone"] == phone[0] - phone[1]
    assert result["checks"]["gain"] is accepted


def test_fixed_and_broken_lists_and_numbers(goal, tmp_path, base_run):
    cand = write_run(tmp_path / "cand", {"pairs": flags(4, 1), "pairs_phone": flags(3)})
    result = verdict(goal, base_run, cand)
    studio = result["sets"]["pairs"]
    assert studio["fixed"] == ["q10", "q11", "q12", "q13"] and studio["broken"] == ["q00"]
    assert studio["base"]["top1"] == 10 and studio["cand"]["top1"] == 13
    assert result["sets"]["pairs_phone"]["fixed"] == ["q10", "q11", "q12"]
    assert result["mean_top1_pct"] == {"base": 50.0, "cand": 65.0, "delta": 15.0}
    assert "ПРИНЯТО" in result["markdown"] and "+4/−1" in result["markdown"]


def test_top5_guard(goal, tmp_path, base_run):
    """+3/+3, но q14 студии выпал из top-5: guard отклоняет."""
    studio_top5 = [i < BASE_TOP5 - 1 for i in range(N)]
    cand = write_run(
        tmp_path / "cand",
        {"pairs": flags(3), "pairs_phone": flags(3)},
        top5={"pairs": studio_top5},
    )
    result = verdict(goal, base_run, cand)
    assert not result["accepted"]
    assert result["checks"]["studio_top5"] is False and result["checks"]["phone_top5"] is True
    assert any("top-5 студии" in reason for reason in result["reasons"])


@pytest.mark.parametrize(
    ("base_ece", "cand_ece", "ok"),
    [
        (0.0326, 0.0326, True),
        (0.0326, 0.0327, True),  # оба 0,033 — как база записана в правиле приёмки
        (0.0326, 0.0334, True),
        (0.0326, 0.0335, False),  # 0,034: половина — вверх
        (0.0326, 0.0301, True),
    ],
)
def test_ece_guard(goal, tmp_path, base_ece, cand_ece, ok):
    base = write_run(
        tmp_path / "base", {"pairs": flags(), "pairs_phone": flags()}, ece_all=base_ece
    )
    cand = write_run(
        tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)}, ece_all=cand_ece
    )
    result = verdict(goal, base, cand)
    assert result["checks"]["ece"] is ok and result["accepted"] is ok


@pytest.mark.parametrize(("p95", "ok"), [(None, True), (2999.0, True), (3000.0, False)])
def test_p95_guard(goal, tmp_path, base_run, p95, ok):
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)})
    result = verdict(goal, base_run, cand, p95=p95)
    assert result["accepted"] is ok
    assert result["checks"]["p95"] is (None if p95 is None else ok)


@pytest.mark.parametrize(
    "provenance",
    [
        {"consistent": False, "startup_problems": []},
        {"startup_problems": []},
        {"consistent": True, "startup_problems": ["модель обучена на top-10"]},
    ],
)
def test_provenance_guard(goal, tmp_path, base_run, provenance):
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)})
    result = verdict(goal, base_run, cand, provenance=provenance)
    assert not result["accepted"] and result["reasons"]


def test_query_sets_must_match(goal, tmp_path, base_run):
    cand = write_run(
        tmp_path / "cand",
        {"pairs": flags(3), "pairs_phone": flags(3)},
        drop=("pairs_phone", "q05"),
    )
    with pytest.raises(goal.GoalError, match="не совпадают"):
        verdict(goal, base_run, cand)


def test_metrics_must_match_oof_rows(goal, tmp_path, base_run):
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)})
    metrics = json.loads((cand / "metrics.json").read_text(encoding="utf-8"))
    metrics["variants"]["vlm35"]["oof"]["by_set"]["pairs"]["learned"]["top1"] = 0.9
    (cand / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(goal.GoalError, match="не сходится"):
        verdict(goal, base_run, cand)


@pytest.mark.parametrize(
    ("who", "run", "test_used", "message"),
    [
        ("cand", {"folds": 3}, False, "фолдов 3"),
        ("cand", {"folds": 10}, False, "фолдов 10"),
        ("base", {"folds": 4}, False, "фолдов 4"),
        ("cand", {"signed": False}, False, "--free-signs"),
        ("cand", {"split": "all"}, False, "обучение на 'all'"),
        ("cand", {}, True, "--final-test"),
    ],
)
def test_runs_must_share_the_goal_metric(goal, tmp_path, who, run, test_used, message):
    """другое число фолдов, --free-signs или --final-test — не та метрика."""
    runs = {}
    for name, fix in (("base", 0), ("cand", 3)):
        own = name == who
        runs[name] = write_run(
            tmp_path / name,
            {"pairs": flags(fix), "pairs_phone": flags(fix)},
            run=run if own else None,
            test_used=test_used and own,
        )
    with pytest.raises(goal.GoalError, match="не сравнимы") as err:
        verdict(goal, runs["base"], runs["cand"], p95=1000.0)
    assert message in str(err.value)


@pytest.mark.parametrize(
    ("run", "changed", "what"),
    [
        ({"reader_keys": {"vlm35": ["vlm@qwen3.5:4b|0123456789ab"]}}, (), "читатель"),
        ({"cv_sha1": ["cv-new", "cv-phone"]}, (), "прогоны CV"),
        ({"top_k": 10}, (), "top-K 20 → 10"),
        ({}, ("app/api/config.py",), "app/api/config.py"),
        ({}, ("app/features/views.py",), "app/features/views.py"),
    ],
)
def test_latency_change_requires_p95(goal, tmp_path, base_run, run, changed, what):
    """сменился читатель, CV, top-K или код на пути запроса — без --p95 нельзя."""
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)}, run=run)
    with pytest.raises(goal.GoalError, match="нужен замер p95") as err:
        verdict(goal, base_run, cand, changed=changed)
    assert what in str(err.value)
    measured = verdict(goal, base_run, cand, changed=changed, p95=2500.0)
    assert measured["accepted"] and measured["latency_changes"]
    assert verdict(goal, base_run, cand, changed=changed, p95=3000.0)["accepted"] is False


def test_nothing_on_request_path_changed_p95_optional(goal, tmp_path, base_run):
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)})
    result = verdict(goal, base_run, cand, changed=("app/reading/fields.py", "log.md"))
    assert (
        result["accepted"] and result["latency_changes"] == [] and result["checks"]["p95"] is None
    )


@pytest.mark.parametrize(
    "path", ["data/gt/public_gt.tsv", "data/raw/pairs/x.webp", "bench/judge.py", "bench/metrics.py"]
)
def test_forbidden_paths_stop_the_verdict(goal, tmp_path, base_run, path):
    cand = write_run(tmp_path / "cand", {"pairs": flags(3), "pairs_phone": flags(3)})
    with pytest.raises(goal.GoalError, match="запрещает"):
        verdict(goal, base_run, cand, changed=(path,))


def test_feature_edit_needs_feature_version_bump(goal, tmp_path, base_run):
    """правка признаков без подъёма FEATURE_VERSION — не сравнивать."""
    same = write_run(tmp_path / "same", {"pairs": flags(3), "pairs_phone": flags(3)})
    with pytest.raises(goal.GoalError, match="FEATURE_VERSION"):
        verdict(goal, base_run, same, changed=("app/resolve/features.py",))
    bumped = write_run(
        tmp_path / "bumped",
        {"pairs": flags(3), "pairs_phone": flags(3)},
        run={"feature_version": "resolve-features/4"},
    )
    assert verdict(goal, base_run, bumped, changed=("app/resolve/features.py",))["accepted"]


def test_matching(goal):
    paths = ["app/reading/fields.py", "app/reading/readers/ollama_vlm.py", "app/reading/crops.py"]
    assert goal.matching(paths, goal.READER_PATHS) == ["app/reading/readers/ollama_vlm.py"]
    assert goal.matching(paths, goal.LATENCY_PATHS) == [
        "app/reading/crops.py",
        "app/reading/readers/ollama_vlm.py",
    ]
    assert goal.matching(["data/gtx.tsv", "data/gt/a.tsv"], goal.FORBIDDEN_PATHS) == [
        "data/gt/a.tsv"
    ]


def _service_model() -> Path:
    """Модель сервиса по умолчанию: после принятой правки признаков это уже не -m2."""
    from app.api.config import DEFAULT_RESOLVE_MODEL

    return DEFAULT_RESOLVE_MODEL


@pytest.mark.skipif(
    not (_service_model().is_file() and (ROOT / "data" / "gt" / "gt_tokens.jsonl").is_file()),
    reason="нужны модель сервиса по умолчанию и разметка каталога data/gt/gt_tokens.jsonl",
)
def test_top_k_startup_problem_says_how_to_change_k(goal, tmp_path):
    """при правке K причина отказа говорит, где поменять K."""
    from app.resolve.features import FEATURE_VERSION

    model = json.loads(_service_model().read_text(encoding="utf-8"))
    if model["meta"].get("feature_version") != FEATURE_VERSION:
        pytest.skip("итерация правит признаки: модель по умолчанию ещё на прежней версии признаков")
    model["meta"]["top_k"] = 10
    path = tmp_path / "vlm35.json"
    path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    problems = goal.service_provenance(path, {})["startup_problems"]
    assert len(problems) == 1 and "app/api/config.py" in problems[0]
    assert "--env SVS_TOP_K=10" in problems[0]
    assert goal.service_provenance(path, {"SVS_TOP_K": "10"})["startup_problems"] == []


def test_verdict_cli(goal, tmp_path, base_run, monkeypatch, capsys):
    monkeypatch.setattr(goal, "service_provenance", lambda model, env: dict(OK_PROVENANCE))
    monkeypatch.setattr(goal, "load_state", goal.default_state)
    monkeypatch.setattr(goal, "changed_paths", lambda repo=None: [])
    good = write_run(tmp_path / "good", {"pairs": flags(3), "pairs_phone": flags(3)})
    weak = write_run(tmp_path / "weak", {"pairs": flags(1), "pairs_phone": flags(1)})
    out = tmp_path / "v.json"
    argv = ["verdict", "--base", str(base_run), "--out", str(out), "--iter", "4"]
    assert goal.main([*argv, "--cand", str(good), "--p95", "1200"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["verdict"] == "accept" and saved["p95_ms"] == 1200.0
    assert saved["markdown"].startswith("iter 04 — ")
    assert "ПРИНЯТО" in capsys.readouterr().out
    assert goal.main([*argv, "--cand", str(weak)]) == 10
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "reject"
    mismatched = write_run(
        tmp_path / "mism", {"pairs": flags(3), "pairs_phone": flags(3)}, drop=("pairs", "q00")
    )
    assert goal.main([*argv, "--cand", str(mismatched)]) == 2
    new_reader = write_run(
        tmp_path / "reader",
        {"pairs": flags(3), "pairs_phone": flags(3)},
        run={"reader_keys": {"vlm35": ["vlm@qwen3.5:4b|0123456789ab"]}},
    )
    assert goal.main([*argv, "--cand", str(new_reader)]) == 2  # задержку не мерили
    assert goal.main([*argv, "--cand", str(new_reader), "--p95", "900"]) == 0
    folds3 = write_run(
        tmp_path / "folds3", {"pairs": flags(3), "pairs_phone": flags(3)}, run={"folds": 3}
    )
    assert goal.main([*argv, "--cand", str(folds3)]) == 2


# ------------------------------------------------------------------ откат
def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    git(root, "config", "core.autocrlf", "false")
    files = {
        ".gitignore": "runs/\n",
        "app.py": "x = 1\n",
        "log.md": "# журнал\n",
        "REPORT.md": "# отчёт\n",
        "data/gt/table.tsv": "a\tb\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8", newline="\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def test_revert_keeps_journal_and_saves_patch(goal, repo):
    goal.begin(repo, 3)
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8", newline="\n")
    (repo / "log.md").write_text("# журнал\niter 03: отклонено\n", encoding="utf-8")
    (repo / "REPORT.md").write_text("# отчёт\nновое\n", encoding="utf-8")
    (repo / "data/gt/table.tsv").write_text("a\tc\n", encoding="utf-8", newline="\n")
    (repo / "new_mod").mkdir()
    (repo / "new_mod/extra.py").write_text("y = 3\n", encoding="utf-8", newline="\n")
    (repo / "staged.py").write_text("z = 4\n", encoding="utf-8", newline="\n")
    git(repo, "add", "staged.py")
    (repo / "runs/goal").mkdir(parents=True, exist_ok=True)  # begin уже создал iter03
    (repo / "runs/goal/state.json").write_text("{}", encoding="utf-8")

    report = goal.revert(repo, 3)

    assert (repo / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert "отклонено" in (repo / "log.md").read_text(encoding="utf-8")
    assert "новое" in (repo / "REPORT.md").read_text(encoding="utf-8")
    assert not (repo / "new_mod").exists() and not (repo / "staged.py").exists()
    assert "staged.py" not in git(repo, "ls-files")
    assert (repo / "data/gt/table.tsv").read_text(encoding="utf-8") == "a\tc\n"  # не тронут
    assert (repo / "runs/goal/state.json").is_file()
    assert report["restored"] == ["app.py"]
    assert report["deleted"] == ["new_mod/extra.py", "staged.py"]
    assert report["kept"] == ["REPORT.md", "log.md"]
    assert report["protected_untouched"] == ["data/gt/table.tsv"]
    assert report["forbidden"] == ["data/gt/table.tsv"]
    assert report["leftover"] == [] and report["untracked_changed"] == []

    patch = repo / "runs/goal/iter03/rejected.patch"
    text = patch.read_text(encoding="utf-8")
    assert "+x = 2" in text and "+y = 3" in text and "+z = 4" in text
    assert "log.md" not in text and "REPORT.md" not in text and "table.tsv" not in text
    git(repo, "apply", "--check", str(patch))  # правку можно вернуть одной командой
    assert set(git(repo, "status", "--porcelain").split()) >= {"M", "log.md", "REPORT.md"}


def test_revert_keeps_untracked_files_from_before_the_iteration(goal, repo):
    """незакоммиченные инструменты цикла переживают откат, созданное итерацией — нет."""
    write(repo / "scripts/goal.py", "# инструмент\n")
    write(repo / "tests/unit/test_goal.py", "# тесты\n")
    write(repo / "notes.txt", "до итерации\n")
    goal.begin(repo, 1)
    write(repo / "app.py", "x = 5\n")
    write(repo / "configs/new.json", "{}\n")
    report = goal.revert(repo, 1)
    assert report["deleted"] == ["configs/new.json"] and report["restored"] == ["app.py"]
    for kept in ("scripts/goal.py", "tests/unit/test_goal.py", "notes.txt"):
        assert (repo / kept).is_file()
    assert not (repo / "configs").exists()


def test_revert_never_deletes_the_tool(goal, repo):
    goal.begin(repo, 1)
    write(repo / "scripts/goal.py", "# появился во время итерации\n")
    assert goal.revert(repo, 1)["deleted"] == []
    assert (repo / "scripts/goal.py").is_file()


def test_revert_reports_edits_to_untracked_files_from_before(goal, repo):
    write(repo / "notes.txt", "до итерации\n")
    goal.begin(repo, 1)
    write(repo / "notes.txt", "правка итерации\n")
    assert goal.revert(repo, 1)["untracked_changed"] == ["notes.txt"]
    assert goal.main(["revert", "--iter", "1", "--repo", str(repo)]) == goal.EXIT_ERROR


def test_revert_does_not_overwrite_a_saved_patch(goal, repo):
    """повторный revert не затирает патч; новая правка — в rejected-2.patch."""
    goal.begin(repo, 1)
    write(repo / "app.py", "x = 7\n")
    first = goal.revert(repo, 1)
    patch = repo / "runs/goal/iter01/rejected.patch"
    saved = patch.read_bytes()
    assert first["patch"] == str(patch) and b"+x = 7" in saved

    again = goal.revert(repo, 1)  # дерево уже чистое
    assert again["patch"] is None and again["patch_bytes"] == 0
    assert patch.read_bytes() == saved

    write(repo / "app.py", "x = 8\n")
    third = goal.revert(repo, 1)
    assert third["patch"] == str(repo / "runs/goal/iter01/rejected-2.patch")
    assert patch.read_bytes() == saved


def test_revert_clears_changes_staged_only(goal, repo):
    """правка только в индексе (рабочий файл равен HEAD) тоже откатывается."""
    goal.begin(repo, 2)
    write(repo / "app.py", "x = 1\nstaged = True\n")
    git(repo, "add", "app.py")
    write(repo / "app.py", "x = 1\n")
    report = goal.revert(repo, 2)
    assert report["restored"] == ["app.py"] and report["leftover"] == []
    assert git(repo, "diff", "--cached", "--name-only") == ""
    assert git(repo, "status", "--porcelain") == ""
    assert "+staged = True" in (repo / "runs/goal/iter02/rejected.patch").read_text(
        encoding="utf-8"
    )


def test_revert_exit_code_on_forbidden_paths(goal, repo, capsys):
    """правка data/gt/* — код откатан, выход 5, а не 0."""
    goal.begin(repo, 4)
    write(repo / "app.py", "x = 9\n")
    write(repo / "data/gt/table.tsv", "a\tz\n")
    assert goal.main(["revert", "--iter", "4", "--repo", str(repo)]) == goal.EXIT_FORBIDDEN
    assert "data/gt/table.tsv" in capsys.readouterr().err
    assert (repo / "app.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (repo / "data/gt/table.tsv").read_text(encoding="utf-8") == "a\tz\n"


def test_revert_needs_begin_and_the_same_head(goal, repo):
    with pytest.raises(goal.GoalError, match="begin --iter 1"):
        goal.revert(repo, 1)
    goal.begin(repo, 1)
    write(repo / "app.py", "x = 3\n")
    git(repo, "commit", "-q", "-am", "commit during the iteration")
    with pytest.raises(goal.GoalError, match="HEAD сдвинулся"):
        goal.revert(repo, 1)


def test_begin_refusals(goal, repo):
    goal.begin(repo, 1)
    with pytest.raises(goal.GoalError, match="уже начата"):
        goal.begin(repo, 1)  # забыли поднять N
    write(repo / "app.py", "x = 4\n")
    with pytest.raises(goal.GoalError, match="уже есть правки"):
        goal.begin(repo, 2)
    git(repo, "checkout", "--", "app.py")
    write(repo / "log.md", "# журнал\niter 01\n")  # журнал не мешает
    head = git(repo, "rev-parse", "HEAD").strip()
    assert goal.begin(repo, 2, last_accepted=head[:7])["head"] == head
    write(repo / "app.py", "x = 6\n")
    git(repo, "commit", "-q", "-am", "unaccepted")
    with pytest.raises(goal.GoalError, match="не последний принятый"):
        goal.begin(repo, 3, last_accepted=head)
    git(repo, "reset", "-q", "--hard", head)
    write(repo / "data/gt/table.tsv", "a\tq\n")
    with pytest.raises(goal.GoalError, match="запрещённые"):
        goal.begin(repo, 4)


def test_revert_refuses_subdirectory(goal, repo):
    (repo / "sub").mkdir()
    with pytest.raises(goal.GoalError, match="не корень"):
        goal.revert(repo / "sub", 1)


# ------------------------------------------------------------------ состояние
def test_merge_patch_and_record(goal, tmp_path):
    state = goal.default_state()
    state = goal.merge_patch(
        state,
        {"inputs": {"resolve_run": "runs/goal/iter01/resolve"}, "service_env": {"A": "1"}},
    )
    assert state["inputs"]["resolve_run"] == "runs/goal/iter01/resolve"
    assert state["inputs"]["index"] == "data/index/visual-s2so400m.npz"  # соседи целы
    state = goal.merge_patch(state, {"service_env": {"A": None}})
    assert state["service_env"] == {}

    goal.record_attempt(state, family="fields", accepted=False, iteration=1)
    goal.record_attempt(state, family="cv", accepted=False, iteration=2)
    assert state["rejected_in_row"] == 2 and state["iteration"] == 2
    goal.record_attempt(
        state, family="fields", accepted=True, iteration=3, commit="abc", p95_ms=900
    )
    assert state["families"]["fields"] == {"attempts": 2, "accepted": 1, "rejected_in_row": 0}
    assert state["families"]["cv"]["rejected_in_row"] == 1
    assert state["rejected_in_row"] == 0 and state["last_accepted_commit"] == "abc"
    assert state["last_p95_ms"] == 900

    path = tmp_path / "state.json"
    goal.save_state(state, path)
    assert goal.load_state(path)["families"]["fields"]["accepted"] == 1
    assert goal.lookup(goal.load_state(path), "inputs.cv.pairs").endswith("predictions.jsonl")


def test_test_check_entry(goal, tmp_path):
    block = {"n": 209, "learned": {"top1": 0.85, "top5": 0.98}, "calibration": {"ece": 0.06}}
    metrics = {
        "test_used": True,
        "test": {
            "variants": {
                "vlm35": {
                    "by_set": {"pairs": block, "pairs_phone": block},
                    "all": {"calibration": {"ece": 0.05}},
                }
            }
        },
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    entry = goal.test_check_entry(run)
    assert entry["pairs"]["top1"] == 0.85 and entry["ece_all"] == 0.05
    metrics["test_used"] = False
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(goal.GoalError, match="без --final-test"):
        goal.test_check_entry(run)


# ------------------------------------------------------------------ dev30
def test_dev30(goal, tmp_path):
    ids = [f"w{i:02d}" for i in range(40)]
    split = {qid: "dev" if i % 2 else "test" for i, qid in enumerate(ids)}
    cv = {}
    for set_name in ("pairs", "pairs_phone"):
        cv[set_name] = tmp_path / f"cv-{set_name}.jsonl"
        cv[set_name].write_text(
            "".join(
                json.dumps({"query_id": q, "slug": f"s-{q}", "meta": {"split": split[q]}}) + "\n"
                for q in reversed(ids)
            ),
            encoding="utf-8",
        )
    studio, phone = tmp_path / "studio", tmp_path / "phone"
    studio.mkdir()
    phone.mkdir()
    for qid in ids:
        (studio / f"{qid}.webp").write_bytes(b"s" + qid.encode())
        (phone / f"{qid}.jpg").write_bytes(b"p" + qid.encode())
    header = "query_id\timage_path\n"
    (tmp_path / "sm.tsv").write_text(header + "".join(f"{q}\t{q}.webp\n" for q in ids))
    (tmp_path / "pm.tsv").write_text(header + "".join(f"{q}\t{q}.jpg\n" for q in ids))
    (tmp_path / "gt.tsv").write_text(
        "query_id\tslug\tin_catalog\n" + "".join(f"{q}\ts-{q}\t1\n" for q in ids)
    )
    kwargs = {
        "cv_studio": cv["pairs"],
        "cv_phone": cv["pairs_phone"],
        "studio_manifest": tmp_path / "sm.tsv",
        "studio_images": studio,
        "phone_manifest": tmp_path / "pm.tsv",
        "phone_images": phone,
        "gt": tmp_path / "gt.tsv",
        "out_images": tmp_path / "out" / "images",
        "out_manifest": tmp_path / "out" / "m.tsv",
        "out_gt": tmp_path / "out" / "gt.tsv",
        "count": 5,
    }
    report = goal.build_dev30(**kwargs)
    dev = sorted(q for q in ids if split[q] == "dev")  # 20 штук: w01, w03, …
    assert report["chosen"] == [dev[0], dev[4], dev[8], dev[12], dev[16]]
    manifest = (tmp_path / "out" / "m.tsv").read_text(encoding="utf-8").splitlines()
    assert manifest[0] == "query_id\timage_path" and len(manifest) == 11
    assert manifest[1] == "s_w01\ts_w01.webp" and manifest[6] == "p_w01\tp_w01.jpg"
    gt = (tmp_path / "out" / "gt.tsv").read_text(encoding="utf-8").splitlines()
    assert gt[0] == "query_id\tslug" and gt[1] == "s_w01\ts-w01" and gt[6] == "p_w01\ts-w01"
    assert (tmp_path / "out" / "images" / "p_w01.jpg").read_bytes() == b"pw01"
    with pytest.raises(goal.GoalError, match="--overwrite"):
        goal.build_dev30(**kwargs)
    assert goal.build_dev30(**kwargs, overwrite=True)["frames"] == 10


# ------------------------------------------------------------------ строгий кэш
class FakeReader:
    id = "fake"
    version = "1"
    params_hash = "p0"

    def __init__(self):
        self.calls = 0

    def available(self):
        return False

    def read(self, image, *, crop, budget_ms):
        self.calls += 1
        return make_reading(
            self,
            image,
            params=self.params_hash,
            crop=crop,
            status="ok",
            elapsed_ms=1,
            lines=[TextLine(id=0, text="АБРАУ")],
        )


def test_strict_cache_never_calls_the_model(goal, tmp_path):
    inner = FakeReader()
    cached = CachedReader(inner, tmp_path)
    seen = np.full((16, 8, 3), 200, dtype=np.uint8)
    unseen = np.full((16, 8, 3), 10, dtype=np.uint8)
    first = cached.read(seen, crop="full", budget_ms=100)  # кэш наполняется обычным путём
    with goal.strict_cache():
        assert cached.available() is True  # Ollama не нужна
        assert cached.read(seen, crop="full", budget_ms=100) == first
        with pytest.raises(goal.CacheMiss):
            cached.read(unseen, crop="full", budget_ms=100)
    assert inner.calls == 1 and cached.stats.hits == 1 and cached.stats.misses == 2
    assert cached.available() is False  # после выхода — как было


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--", "--dataset", "manifest", "--out", "x"], "--cache-dir"),
        (["--cache-dir", "c", "--warmup"], "--no-warmup"),
    ],
)
def test_ocr_cached_arguments(goal, argv, message):
    with pytest.raises(goal.GoalError, match=message):
        goal.ocr_cached(argv)


@pytest.mark.parametrize(
    "path", ["app/reading/readers/ollama_vlm.py", "app/reading/readers/cache.py"]
)
def test_ocr_cached_refuses_after_reader_edit(goal, path):
    """правка читателя без нового params_hash дала бы 100 % попаданий старым кодом."""
    argv = ["--", "--cache-dir", "runs/cache-pairs-ocr", "--out", "x"]
    with pytest.raises(goal.GoalError, match="POSTPROCESS_VERSION"):
        goal.ocr_cached(argv, changed=[path, "app/reading/fields.py"])


# ------------------------------------------------------------------ сервис и реальные кадры
def test_gt_rank(goal):
    answer = {"slug": "a", "top5": [{"slug": s} for s in ("a", "b", "c", "d", "e", "f")]}
    assert goal.gt_rank("a", answer) == 1
    assert goal.gt_rank("e", answer) == 5
    assert goal.gt_rank("f", answer) == ">5"
    assert goal.gt_rank("__none__", answer) == "вне каталога"
    assert goal.gt_rank("a", {"slug": "a", "top5": []}) == 1


FAKE_SERVICE = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    time.sleep(0.5)  # «прогрев»: порт открывается позже
    MODEL = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"status": "ready", "degraded_reasons": [], "warnings": [],
                        "warm": {"scan": {"degraded": []}},
                        "provenance": {"consistent": True}, "model": MODEL})

        def do_POST(self):
            size = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(size)
            slug = "wine-a" if b"AAA" in data else "wine-x"
            self.reply({"slug": slug, "confidence": {"top1": 0.91},
                        "top5": [{"slug": slug}, {"slug": "wine-b"}], "outcome": "matched",
                        "degraded": [], "timings_ms": {"total": 12.0}, "error": None})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_service_up_real13_down_with_fake_service(goal, tmp_path):
    script = tmp_path / "fake_service.py"
    script.write_text(FAKE_SERVICE, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    paths = {
        "pid_path": tmp_path / "service.pid",
        "info_path": tmp_path / "service.json",
    }
    code = goal.service_up(
        overrides={},
        url=url,
        timeout_s=30,
        command=[sys.executable, str(script), str(port)],
        log_path=tmp_path / "service.log",
        state={},
        poll_s=0.2,
        **paths,
    )
    try:
        assert code == goal.EXIT_OK
        pid = int(paths["pid_path"].read_text(encoding="utf-8"))
        assert goal.pid_alive(pid)
        # второй запуск не поднимает второй сервис
        assert goal.service_up(overrides={}, url=url, state={}, **paths) == goal.EXIT_SERVICE

        images = tmp_path / "images"
        images.mkdir()
        (images / "a.jpg").write_bytes(b"AAA")
        (images / "b.jpg").write_bytes(b"BBB")
        frames = [
            goal.Frame("q1", "public", images / "a.jpg", "wine-a"),
            goal.Frame("q2", "public", images / "b.jpg", "wine-b"),
            goal.Frame("q3", "rwl_src", images / "b.jpg", "__none__"),
        ]
        report = goal.run_real13(frames, url, tmp_path / "real")
        assert [row["rank"] for row in report["rows"]] == [1, 2, "вне каталога"]
        assert report["summary"]["in_catalog"] == 2 and report["summary"]["top1"] == 1
        assert report["summary"]["in_top5"] == 2 and report["health"]["problems"] == []
        assert (tmp_path / "real" / "real13.md").read_text(encoding="utf-8").startswith("| кадр")
    finally:
        unloaded = []
        down = goal.service_down(
            url=url, unload=lambda ollama, model: unloaded.append(model) or "ok", **paths
        )
    assert down == goal.EXIT_OK and unloaded
    assert not paths["pid_path"].exists() and not goal.pid_alive(pid)


def sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def wait_dead(goal, pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while goal.pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not goal.pid_alive(pid)


def test_process_identity(goal):
    proc = sleeper()
    try:
        ident = goal.process_identity(proc.pid)
        assert ident is not None and ident["created"] > 0
        assert goal.process_identity(proc.pid) == ident  # стабилен
    finally:
        proc.kill()
        proc.wait()
    assert wait_dead(goal, proc.pid) and goal.process_identity(proc.pid) is None


def test_service_down_never_kills_a_foreign_process(goal, tmp_path):
    """pid-файл устарел, pid занят чужим процессом — service-down его не трогает."""
    victim = sleeper()
    try:
        pid_path, info_path = tmp_path / "service.pid", tmp_path / "service.json"
        url = f"http://127.0.0.1:{free_port()}"
        cases = [
            ({"process": {"created": 1, "exe": "other.exe"}}, goal.EXIT_OK),  # чужой отпечаток
            ({}, goal.EXIT_SERVICE),  # отпечатка нет — не проверить, не трогать
        ]
        for extra, code in cases:
            pid_path.write_text(f"{victim.pid}\n", encoding="utf-8")
            info_path.write_text(
                json.dumps({"pid": victim.pid, "url": url, **extra}), encoding="utf-8"
            )
            down = goal.service_down(pid_path=pid_path, info_path=info_path, url=url, unload=None)
            assert down == code
            assert victim.poll() is None and not pid_path.exists()
    finally:
        victim.kill()


CRASHING_SERVICE = textwrap.dedent(
    """
    import subprocess, sys, time

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(sys.argv[1], "w", encoding="utf-8") as fh:
        fh.write(str(child.pid))
    time.sleep(4)  # успеть попасть в снимок потомков
    sys.exit(7)
    """
)


@pytest.mark.skipif(sys.platform == "darwin", reason="потомки ищутся через Toolhelp32 или /proc")
def test_service_that_dies_before_ready_is_cleaned_up(goal, tmp_path):
    """упал до ready — потомки добиты, VLM выгружена, service.json убран."""
    script = tmp_path / "crash.py"
    script.write_text(CRASHING_SERVICE, encoding="utf-8")
    child_file = tmp_path / "child.pid"
    paths = {"pid_path": tmp_path / "service.pid", "info_path": tmp_path / "service.json"}
    unloaded = []
    code = goal.service_up(
        overrides={},
        url=f"http://127.0.0.1:{free_port()}",
        timeout_s=60,
        command=[sys.executable, str(script), str(child_file)],
        log_path=tmp_path / "service.log",
        state={},
        unload=lambda ollama, model: unloaded.append(model) or "ok",
        poll_s=0.1,
        **paths,
    )
    child = int(child_file.read_text(encoding="utf-8"))
    try:
        assert code == goal.EXIT_SERVICE and unloaded
        assert wait_dead(goal, child)
        assert not paths["pid_path"].exists() and not paths["info_path"].exists()
    finally:
        if goal.pid_alive(child):
            goal.kill_process(child)


def test_expected_service_model(goal):
    state = {"inputs": {"service_model": "configs/resolve/a.json"}}
    assert goal.expected_service_model({}, state) == goal.REPO_ROOT / "configs/resolve/a.json"
    env = {"SVS_RESOLVE_MODEL": "runs/goal/iter01/resolve/models/vlm35.json"}
    assert goal.expected_service_model(env, state) == goal.REPO_ROOT / env["SVS_RESOLVE_MODEL"]
    assert goal.expected_service_model({}, {}) is None


def test_service_and_real13_check_the_accepted_model(goal, tmp_path, monkeypatch):
    """сервис не на принятой модели — service-up 4, real13 с пометкой и кодом 2."""
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps({"meta": {"trained_at": "T1"}}), encoding="utf-8")
    state = {"inputs": {"service_model": str(accepted)}}
    script = tmp_path / "fake_service.py"
    script.write_text(FAKE_SERVICE, encoding="utf-8")
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    paths = {"pid_path": tmp_path / "service.pid", "info_path": tmp_path / "service.json"}
    stale = sleeper()  # pid-файл прошлого сервиса, pid которого занял чужой процесс
    paths["pid_path"].write_text(f"{stale.pid}\n", encoding="utf-8")
    paths["info_path"].write_text(
        json.dumps({"pid": stale.pid, "process": {"created": 1, "exe": "x"}}), encoding="utf-8"
    )
    health_model = {"resolve": "accepted.json", "resolve_trained_at": "T0"}
    try:
        code = goal.service_up(
            overrides={},
            url=url,
            timeout_s=30,
            command=[sys.executable, str(script), str(port), json.dumps(health_model)],
            log_path=tmp_path / "service.log",
            state=state,
            poll_s=0.2,
            **paths,
        )
        assert code == goal.EXIT_NOT_READY
        assert stale.poll() is None  # чужой процесс цел

        image = tmp_path / "a.jpg"
        image.write_bytes(b"AAA")
        frames = [goal.Frame("q1", "public", image, "wine-a")]
        report = goal.run_real13(frames, url, tmp_path / "real", expected_model=accepted)
        assert report["health"]["model_problems"]
        assert report["markdown"].startswith("**НЕ ПРИНЯТАЯ МОДЕЛЬ")
        monkeypatch.setattr(goal, "real_frames", lambda dataset, rwl: frames)
        monkeypatch.setattr(goal, "load_state", lambda: state)
        argv = ["real13", "--out", str(tmp_path / "cli"), "--url", url]
        assert goal.main(argv) == goal.EXIT_ERROR

        accepted.write_text(json.dumps({"meta": {"trained_at": "T0"}}), encoding="utf-8")
        report = goal.run_real13(frames, url, tmp_path / "real", expected_model=accepted)
        assert report["health"]["model_problems"] == []
        assert goal.main(argv) == goal.EXIT_OK
    finally:
        goal.service_down(url=url, unload=None, **paths)
        stale.kill()
