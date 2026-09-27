"""Инструменты цикла улучшений: вердикт итерации, откат, состояние, сервис и реальные кадры.

Цикл меряет каждую правку на dev out-of-fold `bench.train_resolve` (5 фолдов, вариант `vlm35`)
и решает по одному правилу приёмки (`decide`). Стадии цикла — подкоманды ниже.
Скрипт только читает прогоны: top-1, top-5 и ECE считают `bench.train_resolve` и `bench.judge`,
здесь подсчёт не повторяется, а счёт по `oof_predictions.jsonl` сверяется с `metrics.json`.

    python scripts/goal.py begin --iter N
    python scripts/goal.py verdict --base RUN --cand RUN [--p95 MS] --out FILE
    python scripts/goal.py revert --iter N
    python scripts/goal.py state show [--key inputs.cv.pairs] | set PATCH | record … | test-check …
    python scripts/goal.py dev30
    python scripts/goal.py ocr-cached -- <аргументы bench.ocr_bench>
    python scripts/goal.py service-up [--env K=V ...] | service-down
    python scripts/goal.py real13 --out DIR [--url http://127.0.0.1:8080]

Коды выхода: 0 — готово (у verdict — принято); 10 — verdict: отклонено; 2 — ошибка аргументов
или входов (у verdict — прогоны не сравнимы или не хватает замера задержки; у revert — откат
неполный; у real13 — сервис не на принятой модели); 3 — сервис не поднялся, не отвечает или не
остановился; 4 — service-up: сервис отвечает, но не готов к отчётному прогону; ocr-cached: промах
кэша чтений; 5 — revert: правка задела запрещённые пути (`data/gt/`, `data/raw/`, `bench/judge.py`
…), код откатан, запрещённые файлы — нет.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import REPO_ROOT

GOAL_DIR = REPO_ROOT / "runs" / "goal"
STATE_PATH = GOAL_DIR / "state.json"
PID_PATH = GOAL_DIR / "service.pid"
SERVICE_INFO_PATH = GOAL_DIR / "service.json"
SERVICE_LOG = GOAL_DIR / "service.log"

VARIANT = "vlm35"
STUDIO = "pairs"
PHONE = "pairs_phone"
SETS = (STUDIO, PHONE)
SET_TITLES = {STUDIO: "студия", PHONE: "«телефон»"}

#: Правило приёмки: net = исправлено − сломано относительно последнего принятого прогона.
MIN_NET_EACH = 3
MIN_NET_SUM = 5
#: p95 задержки ответа (`bench.judge` на 30 кадрах dev), мс: строго меньше.
P95_LIMIT_MS = 3000.0
#: ECE сравнивается на трёх знаках — с той точностью, с какой база записана в
#: `configs/resolve/README.md` («ECE оба 0,033»); округление половины вверх. На четырёх знаках
#: guard отклонял бы кандидата с 0,0327 против 0,0326, хотя в README оба — 0,033.
ECE_DIGITS = 3
#: Доли в `metrics.json` округлены до шести знаков (`train_resolve.share`).
SHARE_TOLERANCE = 1e-6
#: Метрика цели — dev out-of-fold `bench.train_resolve` на пяти фолдах со знаками весов.
REQUIRED_FOLDS = 5
TRAIN_SPLIT = "dev"

EXIT_OK = 0
EXIT_REJECT = 10
EXIT_ERROR = 2
EXIT_SERVICE = 3
EXIT_NOT_READY = 4
EXIT_CACHE_MISS = 4
EXIT_FORBIDDEN = 5

#: Журнал и отчёт цикла переживают откат.
KEEP_NAMES = ("log.md", "REPORT.md")
#: Откат не трогает эти каталоги верхнего уровня ни при каких изменениях.
PROTECTED_TOP = ("runs", "data", ".venv")
#: Инструменты цикла откат не удаляет никогда, даже неотслеживаемыми.
NEVER_DELETE = ("scripts/goal.py", "tests/unit/test_goal.py")
#: «НЕ ТРОГАТЬ» цели, что видно по git (подсчёт внутри `bench/train_resolve.py` — только глазами).
FORBIDDEN_PATHS = (
    "bench/judge.py",
    "bench/metrics.py",
    "bench/datasets.py",
    "data/gt/",
    "data/raw/",
)
#: Код на пути запроса сервиса: его правка может сдвинуть задержку — нужен замер p95 (стадия e).
LATENCY_PATHS = (
    "app/api/",
    "app/normalize/",
    "app/detect/",
    "app/features/",
    "app/reading/readers/",
    "app/reading/crops.py",
)
#: Признаки resolve: любая их правка поднимает `FEATURE_VERSION`.
FEATURE_PATHS = ("app/resolve/features.py", "app/resolve/attrs.py")
#: Код, от которого зависит содержимое кэша чтений: после его правки путь «из кэша» не годится.
READER_PATHS = ("app/reading/readers/", "app/reading/contracts.py")

DEFAULT_URL = "http://127.0.0.1:8080"
SERVICE_TIMEOUT_S = 180.0
NONE_SLUG = "__none__"
OUT_OF_CATALOG = "вне каталога"
DATASET_DIR_DEFAULT = REPO_ROOT.parent / "Датасет и подробное задание" / "unpacked" / "Датасет"
RWL_DIR_DEFAULT = REPO_ROOT.parent / "russian_wine_labels_raw"
PAIRS_IMAGES_DEFAULT = REPO_ROOT.parent / "Code" / "data" / "raw" / "pairs" / "roskachestvo"
PHONE_IMAGES_DEFAULT = REPO_ROOT / "runs" / "pairs-phone-images"
PAIRS_OCR_DIR = REPO_ROOT / "runs" / "pairs-ocr"
EVAL_SETS_DIR = REPO_ROOT / "runs" / "eval-sets"
DEV30_QUERIES = 15


class GoalError(RuntimeError):
    """Вход не тот: нет файла, другой набор запросов, числа не сходятся с `metrics.json`."""


# ------------------------------------------------------------------ общее
def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def iter_dir(iteration: int, goal_dir: Path = GOAL_DIR) -> Path:
    """Каталог итерации: `runs/goal/iter07`."""
    if iteration < 0:
        raise GoalError(f"номер итерации не может быть отрицательным: {iteration}")
    return goal_dir / f"iter{iteration:02d}"


def write_json(path: Path, obj: Any) -> None:
    """JSON с кириллицей как есть; запись через временный файл, чтобы не оставить половину."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoalError(f"нет файла {path}") from exc
    except ValueError as exc:
        raise GoalError(f"{path}: не JSON ({exc})") from exc


def read_tsv(path: Path) -> list[dict[str, str]]:
    """TSV с шапкой (манифест `query_id<TAB>image_path` или эталон `query_id<TAB>slug`)."""
    if not path.is_file():
        raise GoalError(f"нет файла {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [
            {key: (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(fh, delimiter="\t")
        ]


def parse_env(items: Sequence[str] | None) -> dict[str, str]:
    """`K=V` из `--env` в словарь; пустое значение допустимо (переменная сбрасывается к умолчанию)."""
    out: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise GoalError(f"--env ждёт K=V, получено {item!r}")
        out[key.strip()] = value
    return out


def comma(value: float | None, digits: int = 1) -> str:
    """Число с запятой, как в README: 86,6."""
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def nearest_rank(values: Sequence[float], q: float) -> float | None:
    """Процентиль ближайшим рангом, без интерполяции — как `bench.judge`."""
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(q * len(ordered)) - 1)])


# ------------------------------------------------------------------ вердикт
@dataclass(frozen=True)
class ResolveRun:
    """Прогон `bench.train_resolve`: строки out-of-fold одного варианта и его блок `oof`."""

    path: Path
    variant: str
    rows: dict[tuple[str, str], dict[str, Any]]
    oof: dict[str, Any]
    test_used: bool
    #: Паспорт прогона (`metrics.json → run`): фолды, знаки, читатели, CV, версия признаков.
    run: dict[str, Any]


def load_resolve_run(path: Path, variant: str = VARIANT) -> ResolveRun:
    """`metrics.json` и строки `oof_predictions.jsonl` варианта; ключ — (набор, query_id)."""
    metrics_path = path / "metrics.json"
    oof_path = path / "oof_predictions.jsonl"
    for needed in (metrics_path, oof_path):
        if not needed.is_file():
            raise GoalError(f"нет {needed}: это не прогон bench.train_resolve")
    metrics = read_json(metrics_path)
    report = (metrics.get("variants") or {}).get(variant)
    if not isinstance(report, Mapping) or not isinstance(report.get("oof"), Mapping):
        raise GoalError(f"{metrics_path}: нет варианта {variant} с блоком oof")
    passport = metrics.get("run")
    if not isinstance(passport, Mapping):
        raise GoalError(f"{metrics_path}: нет паспорта прогона run")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with oof_path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise GoalError(f"{oof_path}:{number}: не JSON ({exc})") from exc
            if row.get("variant") != variant:
                continue
            if not row.get("set") or not row.get("query_id"):
                raise GoalError(f"{oof_path}:{number}: нет set или query_id")
            key = (str(row["set"]), str(row["query_id"]))
            if key in rows:
                raise GoalError(f"{oof_path}:{number}: запрос {key} повторяется")
            rows[key] = row
    if not rows:
        raise GoalError(f"{oof_path}: нет строк варианта {variant}")
    return ResolveRun(
        path,
        variant,
        rows,
        dict(report["oof"]),
        bool(metrics.get("test_used")),
        dict(passport),
    )


def set_stats(run: ResolveRun, name: str) -> dict[str, Any]:
    """Top-1, top-5 и ECE набора. Счёт по строкам сверяется с долями в `metrics.json`."""
    rows = [row for (set_name, _), row in run.rows.items() if set_name == name]
    if not rows:
        raise GoalError(f"{run.path}: нет запросов набора {name}")
    n = len(rows)
    top1 = sum(bool(row.get("learned_correct")) for row in rows)
    top5 = sum(row.get("slug") in (row.get("learned_top5") or [])[:5] for row in rows)
    block = (run.oof.get("by_set") or {}).get(name)
    if not isinstance(block, Mapping):
        raise GoalError(f"{run.path}: в metrics.json нет oof.by_set.{name}")
    learned = block.get("learned") or {}
    if block.get("n") != n:
        raise GoalError(f"{run.path}: {name}: n в metrics.json {block.get('n')}, строк oof {n}")
    for label, count in (("top1", top1), ("top5", top5)):
        written = learned.get(label)
        if written is None or abs(float(written) - count / n) > SHARE_TOLERANCE:
            raise GoalError(
                f"{run.path}: {name} {label} в metrics.json ({written}) не сходится с "
                f"oof_predictions.jsonl ({count} из {n})"
            )
    ece = (block.get("calibration") or {}).get("ece")
    return {
        "n": n,
        "top1": top1,
        "top5": top5,
        "top1_pct": round(100 * top1 / n, 4),
        "top5_pct": round(100 * top5 / n, 4),
        "ece": None if ece is None else float(ece),
    }


def overall_ece(run: ResolveRun) -> float:
    ece = ((run.oof.get("all") or {}).get("calibration") or {}).get("ece")
    if ece is None:
        raise GoalError(f"{run.path}: в metrics.json нет oof.all.calibration.ece")
    return float(ece)


def ece_rounded(value: float) -> Decimal:
    """ECE на `ECE_DIGITS` знаках, половина вверх (0,0325 → 0,033) — как в записи цели."""
    step = Decimal(1).scaleb(-ECE_DIGITS)
    return Decimal(repr(float(value))).quantize(step, rounding=ROUND_HALF_UP)


def matching(paths: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    """Пути из `paths`, равные элементу `prefixes` или лежащие под ним (элемент с `/` — каталог)."""
    return sorted(
        {
            path
            for path in paths
            for prefix in prefixes
            if path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        }
    )


def passport_problems(run: ResolveRun) -> list[str]:
    """Прогон посчитан не той метрикой, что задала цель: сравнивать его с базой нельзя.

    Цель — dev out-of-fold `bench.train_resolve` на пяти фолдах со знаками весов. Число фолдов
    само по себе двигает исправлено/сломано на ±10 на набор, `--free-signs` — другая модель, а
    прогон с `--final-test` уже смотрел test, которому место только в test-check.
    """
    info = run.run
    problems: list[str] = []
    if info.get("folds") != REQUIRED_FOLDS:
        problems.append(
            f"{run.path}: фолдов {info.get('folds')!r}, а метрика цели — {REQUIRED_FOLDS} "
            f"(bench.train_resolve --folds {REQUIRED_FOLDS})"
        )
    if info.get("split") != TRAIN_SPLIT:
        problems.append(f"{run.path}: обучение на {info.get('split')!r}, а не на {TRAIN_SPLIT}")
    if info.get("signed") is not True:
        problems.append(f"{run.path}: знаки весов не держались (--free-signs — только сравнение)")
    if run.test_used:
        problems.append(
            f"{run.path}: прогон с --final-test — test смотрят только в test-check (g), в "
            "отдельной папке; базу и кандидата обучают без него"
        )
    return problems


def latency_changes(base: ResolveRun, cand: ResolveRun, changed: Sequence[str]) -> list[str]:
    """Что в кандидате могло сдвинуть задержку сервиса относительно базы.

    Читатель (новый промпт или разбор — новый `params_hash`), другие прогоны CV (окна, индекс,
    агрегация), другой top-K или правка кода на пути запроса. В таких случаях guard p95 не
    пропускается молча: вердикт требует замер на dev30.
    """
    out: list[str] = []
    keys_base = sorted((base.run.get("reader_keys") or {}).get(base.variant) or [])
    keys_cand = sorted((cand.run.get("reader_keys") or {}).get(cand.variant) or [])
    if keys_base != keys_cand:
        out.append(f"читатель этикетки {keys_base} → {keys_cand}")
    if sorted(base.run.get("cv_sha1") or []) != sorted(cand.run.get("cv_sha1") or []):
        out.append("прогоны CV другие (run.cv_sha1)")
    if base.run.get("top_k") != cand.run.get("top_k"):
        out.append(f"top-K {base.run.get('top_k')} → {cand.run.get('top_k')}")
    touched = matching(changed, LATENCY_PATHS)
    if touched:
        out.append("правка кода на пути запроса: " + ", ".join(touched))
    return out


def decide(
    base: ResolveRun,
    cand: ResolveRun,
    *,
    p95_ms: float | None,
    provenance: Mapping[str, Any],
    changed: Sequence[str] = (),
) -> dict[str, Any]:
    """Правило приёмки цикла, без послаблений.

    Прирост: исправлено − сломано (`net`) относительно базы — не меньше 3 на студии и не меньше
    3 на «телефоне», или не меньше 5 в сумме, если ни один набор не отрицательный. Guard (не
    хуже базы): top-1 «телефона», top-5 обоих наборов, общий ECE на трёх знаках (`ECE_DIGITS`);
    p95 < 3000 мс; `provenance.consistent` как у сервиса при старте. Сверх буквы цели —
    `service_start`: сервис с моделью стартует (top-K и один читатель). Без старта у сервиса нет
    и `provenance`, так что это то же условие, а не новое.

    `GoalError` (код 2, а не отказ), если сравнивать нечего: множества запросов разные; прогон
    посчитан не метрикой цели (`passport_problems`); `changed` (пути, изменённые против HEAD)
    задевает запрещённое цели или признаки без подъёма `FEATURE_VERSION`; задержка могла
    сдвинуться (`latency_changes`), а `p95_ms` не передан.
    """
    if base.variant != cand.variant:
        raise GoalError(f"варианты разные: {base.variant} и {cand.variant}")
    problems = passport_problems(base) + passport_problems(cand)
    forbidden = matching(changed, FORBIDDEN_PATHS)
    if forbidden:
        problems.append("правка задела то, что цель трогать запрещает: " + ", ".join(forbidden))
    features = matching(changed, FEATURE_PATHS)
    if features and cand.run.get("feature_version") == base.run.get("feature_version"):
        problems.append(
            f"правлены признаки ({', '.join(features)}), а FEATURE_VERSION тот же, что у базы "
            f"({base.run.get('feature_version')}): поднять его и переобучить"
        )
    if problems:
        raise GoalError("прогоны не сравнимы по правилу цели: " + "; ".join(problems))
    only_base = sorted(set(base.rows) - set(cand.rows))
    only_cand = sorted(set(cand.rows) - set(base.rows))
    if only_base or only_cand:
        raise GoalError(
            f"множества запросов не совпадают: только в базе {len(only_base)} "
            f"(например {only_base[:2]}), только в кандидате {len(only_cand)} "
            f"(например {only_cand[:2]})"
        )
    sets: dict[str, dict[str, Any]] = {}
    for name in SETS:
        keys = sorted(key for key in cand.rows if key[0] == name)
        fixed = [
            q
            for s, q in keys
            if cand.rows[(s, q)].get("learned_correct")
            and not base.rows[(s, q)].get("learned_correct")
        ]
        broken = [
            q
            for s, q in keys
            if base.rows[(s, q)].get("learned_correct")
            and not cand.rows[(s, q)].get("learned_correct")
        ]
        sets[name] = {
            "base": set_stats(base, name),
            "cand": set_stats(cand, name),
            "fixed": fixed,
            "broken": broken,
            "net": len(fixed) - len(broken),
        }
    extra = sorted({key[0] for key in cand.rows} - set(SETS))
    if extra:
        raise GoalError(f"в прогонах есть наборы вне {SETS}: {extra}")
    latency = latency_changes(base, cand, changed)
    if latency and p95_ms is None:
        raise GoalError(
            "задержка могла сдвинуться (" + "; ".join(latency) + "): нужен замер p95 на dev30 — "
            "стадия (e), затем verdict --p95 MS"
        )

    net_s, net_p = sets[STUDIO]["net"], sets[PHONE]["net"]
    gain_each = net_s >= MIN_NET_EACH and net_p >= MIN_NET_EACH
    gain_sum = net_s + net_p >= MIN_NET_SUM and net_s >= 0 and net_p >= 0
    ece = {"base": overall_ece(base), "cand": overall_ece(cand)}
    s_base, s_cand = sets[STUDIO]["base"], sets[STUDIO]["cand"]
    p_base, p_cand = sets[PHONE]["base"], sets[PHONE]["cand"]
    problems = list(provenance.get("startup_problems") or [])
    checks: dict[str, bool | None] = {
        "gain": gain_each or gain_sum,
        "phone_top1": p_cand["top1"] >= p_base["top1"],
        "studio_top5": s_cand["top5"] >= s_base["top5"],
        "phone_top5": p_cand["top5"] >= p_base["top5"],
        "ece": ece_rounded(ece["cand"]) <= ece_rounded(ece["base"]),
        "p95": None if p95_ms is None else p95_ms < P95_LIMIT_MS,
        "provenance": provenance.get("consistent") is True,
        "service_start": not problems,
    }
    reasons: list[str] = []
    if not checks["gain"]:
        reasons.append(
            f"прирост мал: студия {net_s:+d}, «телефон» {net_p:+d} (нужно ≥ {MIN_NET_EACH} и "
            f"≥ {MIN_NET_EACH} или ≥ {MIN_NET_SUM} в сумме без отрицательного набора)"
        )
    if not checks["phone_top1"]:
        reasons.append(f"top-1 «телефона» ниже базы: {p_cand['top1']} < {p_base['top1']}")
    if not checks["studio_top5"]:
        reasons.append(f"top-5 студии ниже базы: {s_cand['top5']} < {s_base['top5']}")
    if not checks["phone_top5"]:
        reasons.append(f"top-5 «телефона» ниже базы: {p_cand['top5']} < {p_base['top5']}")
    if not checks["ece"]:
        reasons.append(
            f"общий ECE выше базы на {ECE_DIGITS} знаках: {ece_rounded(ece['cand'])} > "
            f"{ece_rounded(ece['base'])} ({ece['cand']:.4f} против {ece['base']:.4f})"
        )
    if checks["p95"] is False:
        reasons.append(f"p95 задержки {p95_ms:.0f} мс не меньше {P95_LIMIT_MS:.0f}")
    if not checks["provenance"]:
        reasons.append(f"provenance.consistent = {provenance.get('consistent')!r}")
    for problem in problems:
        reasons.append(f"сервис с моделью не стартует: {problem}")

    def mean_top1(side: str) -> float:
        """Среднее top-1 студии и «телефона» в % — из счёта, без двойного округления."""
        share = sum(sets[name][side]["top1"] / sets[name][side]["n"] for name in SETS)
        return round(100 * share / len(SETS), 4)

    result: dict[str, Any] = {
        "verdict": "accept" if not reasons else "reject",
        "accepted": not reasons,
        "reasons": reasons,
        "variant": cand.variant,
        "base": str(base.path),
        "cand": str(cand.path),
        "cand_test_used": cand.test_used,
        "rule": {
            "net_studio": net_s,
            "net_phone": net_p,
            "net_sum": net_s + net_p,
            "gain_each": gain_each,
            "gain_sum": gain_sum,
            "min_net_each": MIN_NET_EACH,
            "min_net_sum": MIN_NET_SUM,
        },
        "checks": checks,
        "sets": sets,
        "mean_top1_pct": {
            "base": mean_top1("base"),
            "cand": mean_top1("cand"),
            "delta": round(mean_top1("cand") - mean_top1("base"), 4),
        },
        "ece_all": ece,
        "ece_digits": ECE_DIGITS,
        "p95_ms": p95_ms,
        "latency_changes": latency,
        "provenance": dict(provenance),
    }
    result["markdown"] = verdict_markdown(result)
    return result


def verdict_markdown(result: Mapping[str, Any], iteration: int | None = None) -> str:
    """Строка журнала: числа базы → кандидата, исправлено/сломано, guard и решение."""
    s, p = result["sets"][STUDIO], result["sets"][PHONE]

    def top1(block: Mapping[str, Any]) -> str:
        return (
            f"{comma(block['base']['top1_pct'])} → {comma(block['cand']['top1_pct'])} "
            f"(+{len(block['fixed'])}/−{len(block['broken'])})"
        )

    def top5(block: Mapping[str, Any]) -> str:
        return f"{comma(block['base']['top5_pct'])} → {comma(block['cand']['top5_pct'])}"

    mean = result["mean_top1_pct"]
    ece = result["ece_all"]
    p95 = result.get("p95_ms")
    prov = "ok" if result["checks"]["provenance"] else "НЕ СХОДИТСЯ"
    head = f"iter {iteration:02d} — " if iteration is not None else ""
    line = (
        f"{head}OOF top-1: студия {top1(s)}, «телефон» {top1(p)}, среднее "
        f"{comma(mean['base'])} → {comma(mean['cand'])}; top-5: студия {top5(s)}, «телефон» "
        f"{top5(p)}; ECE {comma(ece['base'], 4)} → {comma(ece['cand'], 4)}; p95 "
        f"{'не мерилась' if p95 is None else f'{p95:.0f} мс'}; provenance {prov}; "
        f"**{'ПРИНЯТО' if result['accepted'] else 'ОТКЛОНЕНО'}**"
    )
    if result["reasons"]:
        line += " — " + "; ".join(result["reasons"])
    return line


def service_provenance(model_path: Path, environ: Mapping[str, str]) -> dict[str, Any]:
    """`provenance` модели так, как его считает сервис при старте, плюс проверки старта.

    Разметка каталога, словарь и настройки — из `ServiceSettings.from_env(environ)`; читатель —
    `OllamaVlmReader` с моделью сервиса под замком, как в `ScannerService.load`. Модель, которая не
    загружается (другая версия признаков), — `consistent: false` с причиной.
    """
    from app.api.config import ServiceSettings
    from app.api.service import (
        LockedReader,
        StartupError,
        load_catalog,
        load_lexicon,
        load_resolve_model,
        provenance,
        reader_identity,
    )
    from app.reading.readers.ollama_vlm import OllamaVlmReader
    from app.resolve.features import FEATURE_VERSION, readers_of

    settings = ServiceSettings.from_env(environ)
    try:
        model = load_resolve_model(model_path)
        _, attrs = load_catalog(settings.attrs_path)
        lexicon = load_lexicon(settings.lexicon_path)
    except StartupError as exc:
        return {
            "consistent": False,
            "error": str(exc),
            "model": str(model_path),
            "startup_problems": [str(exc)],
        }
    readers = readers_of(model.feature_names)
    reader_key = readers[0] if readers else None
    vlm_reader = None
    if reader_key is not None and settings.vlm_enabled:
        reader = OllamaVlmReader(
            settings.vlm_model,
            settings.ollama_url,
            timeout_s=settings.vlm_timeout_ms / 1000,
            keep_alive=settings.vlm_keep_alive,
        )
        vlm_reader = reader_identity(LockedReader(reader, threading.Lock()))
    report = provenance(
        model,
        attrs,
        lexicon,
        reader_key=reader_key,
        vlm_reader=vlm_reader,
        live_cards=settings.live_cards,
    )
    problems: list[str] = []
    model_k = model.meta.get("top_k")
    if model_k is not None and int(model_k) != settings.top_k:
        problems.append(
            f"модель обучена на top-{model_k}, а SVS_TOP_K={settings.top_k} — при правке K "
            f"поменять top_k по умолчанию в app/api/config.py в той же правке (сервис и "
            f"run_eval.sh стартуют с умолчаниями) или передать verdict --env SVS_TOP_K={model_k} "
            f"и при принятии записать его в service_env"
        )
    if len(readers) > 1:
        problems.append(f"модель ждёт чтения нескольких читателей: {', '.join(readers)}")
    return {
        **report,
        "model": str(model_path),
        "reader_key": reader_key,
        "feature_version": FEATURE_VERSION,
        "startup_problems": problems,
    }


# ------------------------------------------------------------------ откат
def git(repo: Path, *args: str, ok: Sequence[int] = (0,)) -> bytes:
    """git в `repo`; пути — буквально, без глобов pathspec (`[`, `*` в имени файла)."""
    env = {**os.environ, "GIT_LITERAL_PATHSPECS": "1"}
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False, env=env
    )
    if proc.returncode not in ok:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise GoalError(f"git {' '.join(args)}: код {proc.returncode}: {err}")
    return proc.stdout


def _z(blob: bytes) -> list[str]:
    return [part for part in blob.decode("utf-8").split("\0") if part]


def is_kept(path: str, keep: Sequence[str] = KEEP_NAMES) -> bool:
    return PurePosixPath(path).name in keep


def is_protected(path: str) -> bool:
    return PurePosixPath(path).parts[0] in PROTECTED_TOP


def git_root(repo: Path) -> Path:
    """`repo` — корень рабочего дерева git, иначе `GoalError`."""
    repo = repo.resolve()
    top = Path(git(repo, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if top != repo:
        raise GoalError(f"{repo} — не корень рабочего дерева git (корень {top})")
    return repo


def head_sha(repo: Path, rev: str = "HEAD") -> str:
    return git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").decode("utf-8").strip()


def name_status(repo: Path, *args: str) -> dict[str, str]:
    """`git diff … --name-status -z` против HEAD: путь → статус (M, A, D…)."""
    tokens = _z(git(repo, "diff", *args, "HEAD", "--name-status", "--no-renames", "-z"))
    return {path: status for status, path in zip(tokens[0::2], tokens[1::2], strict=True)}


def tracked_changes(repo: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Изменения против HEAD: в рабочем дереве и в индексе — отдельно.

    Правка, которая есть только в индексе (рабочий файл уже равен HEAD), в `git diff HEAD` не
    видна, а в коммит попала бы.
    """
    return name_status(repo), name_status(repo, "--cached")


def untracked_files(repo: Path) -> list[str]:
    return _z(git(repo, "ls-files", "--others", "--exclude-standard", "-z"))


def changed_paths(repo: Path = REPO_ROOT) -> list[str]:
    """Всё, что отличается от HEAD: рабочее дерево, индекс и новые файлы вне .gitignore."""
    worktree, cached = tracked_changes(repo)
    return sorted({*worktree, *cached, *untracked_files(repo)})


def blob_sha1(path: Path) -> str | None:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return None


def start_path_for(repo: Path, iteration: int) -> Path:
    return iter_dir(iteration, repo / "runs" / "goal") / "start.json"


def begin(repo: Path, iteration: int, *, last_accepted: str | None = None) -> dict[str, Any]:
    """Начало итерации: снимок HEAD и неотслеживаемых файлов в `runs/goal/iterNN/start.json`.

    По снимку `revert` удаляет только файлы, созданные итерацией: инструменты цикла, пока они не
    закоммичены, и прочие неотслеживаемые файлы до итерации переживают откат. Отказ, если
    итерация N уже начата (забыли поднять N), если HEAD не последний принятый коммит или если в
    дереве уже есть правки отслеживаемых файлов (кроме журнала, отчёта и `runs/`, `data/`,
    `.venv/`) либо запрещённых цели путей.
    """
    repo = git_root(repo)
    start = start_path_for(repo, iteration)
    if start.exists():
        raise GoalError(f"итерация {iteration:02d} уже начата ({start}): новой итерации — новый N")
    head = head_sha(repo)
    if last_accepted:
        accepted = head_sha(repo, last_accepted)
        if accepted != head:
            raise GoalError(
                f"HEAD {head[:12]} не последний принятый коммит {accepted[:12]} "
                "(state.last_accepted_commit): итерация начинается от принятого состояния"
            )
    worktree, cached = tracked_changes(repo)
    untracked = untracked_files(repo)
    forbidden = matching([*worktree, *cached, *untracked], FORBIDDEN_PATHS)
    if forbidden:
        raise GoalError("изменены запрещённые цели пути: " + ", ".join(forbidden))
    dirty = sorted(p for p in {*worktree, *cached} if not is_kept(p) and not is_protected(p))
    if dirty:
        raise GoalError(
            "до начала итерации в дереве уже есть правки (откат их бы тоже снёс): "
            + ", ".join(dirty)
        )
    snapshot = {
        path: blob_sha1(repo / path)
        for path in untracked
        if not is_kept(path) and not is_protected(path)
    }
    record = {"iteration": iteration, "head": head, "untracked": snapshot, "at": now_iso()}
    write_json(start, record)
    return {"start": str(start), "head": head, "untracked": sorted(snapshot)}


def free_patch_path(folder: Path) -> Path:
    """`rejected.patch`, а если он уже есть и не пуст — `rejected-2.patch`, `-3`…: не затирать."""
    candidate = folder / "rejected.patch"
    number = 1
    while candidate.exists() and candidate.stat().st_size > 0:
        number += 1
        candidate = folder / f"rejected-{number}.patch"
    return candidate


def revert(repo: Path, iteration: int, *, keep: Sequence[str] = KEEP_NAMES) -> dict[str, Any]:
    """Сохранить отклонённую правку в `runs/goal/iterNN/rejected.patch` и откатить её.

    Нужен снимок `begin` той же итерации: HEAD должен остаться тем же, а удаляются только
    неотслеживаемые файлы, которых в снимке не было (и никогда — `NEVER_DELETE`). В патч идут
    изменения отслеживаемых файлов против HEAD — и в рабочем дереве, и только в индексе — и новые
    файлы итерации, кроме журнала и отчёта (`KEEP_NAMES`) и путей в `runs/`, `data/`, `.venv/`.
    Потом отслеживаемые файлы возвращаются к HEAD (вместе с индексом), добавленные в индекс и
    новые — удаляются. Пустой патч не пишется; непустой прошлый не затирается (`free_patch_path`).
    Запрещённые цели пути (`FORBIDDEN_PATHS`) не откатываются, но попадают в `forbidden`.
    """
    repo = git_root(repo)
    start_file = start_path_for(repo, iteration)
    if not start_file.is_file():
        raise GoalError(
            f"нет {start_file}: итерацию начинают `goal.py begin --iter {iteration}` — без снимка "
            "откат не знает, какие неотслеживаемые файлы создала итерация"
        )
    start = read_json(start_file)
    head = head_sha(repo)
    if head != start.get("head"):
        raise GoalError(
            f"HEAD сдвинулся с начала итерации ({str(start.get('head'))[:12]} → {head[:12]}): "
            "откат рабочего дерева не вернёт коммиты — разобрать вручную"
        )
    before: dict[str, str | None] = dict(start.get("untracked") or {})
    worktree, cached = tracked_changes(repo)
    status = {**cached, **worktree}
    untracked = untracked_files(repo)

    def ours(path: str) -> bool:
        return not is_kept(path, keep) and not is_protected(path)

    everything = sorted({*status, *untracked})
    kept = [p for p in everything if is_kept(p, keep)]
    protected = [p for p in everything if is_protected(p) and not is_kept(p, keep)]
    forbidden = matching(everything, FORBIDDEN_PATHS)
    tracked = [(status[p], p) for p in sorted(status) if ours(p)]
    fresh = [
        p
        for p in untracked
        if ours(p) and p not in before and p not in status and p not in NEVER_DELETE
    ]
    untouchable = [
        p for p in untracked if ours(p) and p in before and blob_sha1(repo / p) != before[p]
    ]

    patch = b""
    for _, path in tracked:
        args = ("diff",) if path in worktree else ("diff", "--cached")
        patch += git(repo, *args, "HEAD", "--binary", "--no-renames", "--", path)
    for path in fresh:
        patch += git(repo, "diff", "--no-index", "--binary", "--", "/dev/null", path, ok=(0, 1))
    patch_path: Path | None = None
    if patch:
        folder = iter_dir(iteration, repo / "runs" / "goal")
        folder.mkdir(parents=True, exist_ok=True)
        patch_path = free_patch_path(folder)
        patch_path.write_bytes(patch)

    restored: list[str] = []
    deleted: list[str] = []
    for code, path in tracked:
        if code.startswith("A"):
            git(repo, "rm", "--cached", "--quiet", "--ignore-unmatch", "--", path)
            target = repo / path
            if target.is_file() or target.is_symlink():
                target.unlink()
            deleted.append(path)
        else:
            git(repo, "checkout", "HEAD", "--", path)
            restored.append(path)
    for path in fresh:
        target = repo / path
        if target.is_file() or target.is_symlink():
            target.unlink()
        deleted.append(path)
    for path in deleted:
        parent = (repo / path).parent
        while parent != repo and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    worktree_after, cached_after = tracked_changes(repo)
    leftover = sorted(p for p in {*worktree_after, *cached_after} if ours(p))
    return {
        "patch": str(patch_path) if patch_path else None,
        "patch_bytes": len(patch),
        "restored": restored,
        "deleted": sorted(deleted),
        "kept": kept,
        "protected_untouched": protected,
        "forbidden": forbidden,
        "leftover": leftover,
        "untracked_changed": untouchable,
    }


# ------------------------------------------------------------------ состояние
def default_state() -> dict[str, Any]:
    """Состояние до первой итерации: входы базы, принятые командой."""
    return {
        "version": 1,
        "iteration": 0,
        "inputs": {
            "cv": {
                STUDIO: "runs/cvall-s2so400m-pairs/predictions.jsonl",
                PHONE: "runs/cvall-s2so400m-pairs_phone/predictions.jsonl",
            },
            "ocr": {
                STUDIO: "runs/ocr-pairs-vlm35-m2/predictions.jsonl",
                PHONE: "runs/ocr-pairsphone-vlm35-m2/predictions.jsonl",
            },
            "index": "data/index/visual-s2so400m.npz",
            "resolve_run": "runs/resolve-s2so400m-m2",
            "service_model": "configs/resolve/s2so400m-vlm35-m2.json",
        },
        #: Переменные окружения, с которыми поднимается принятый сервис (например,
        #: SVS_INDEX_PATH после принятой правки стороны CV).
        "service_env": {},
        "last_accepted_commit": None,
        "last_p95_ms": None,
        "families": {},
        "rejected_in_row": 0,
        "test_checks": [],
        "updated_at": None,
    }


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    return read_json(path) if path.is_file() else default_state()


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    state["updated_at"] = now_iso()
    write_json(path, state)


def merge_patch(target: Any, patch: Any) -> Any:
    """JSON Merge Patch (RFC 7386): словари сливаются, null удаляет ключ, прочее заменяет."""
    if not isinstance(patch, Mapping):
        return patch
    out = dict(target) if isinstance(target, Mapping) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = merge_patch(out.get(key), value)
    return out


def record_attempt(
    state: dict[str, Any],
    *,
    family: str,
    accepted: bool,
    iteration: int | None = None,
    commit: str | None = None,
    p95_ms: float | None = None,
) -> dict[str, Any]:
    """Счётчики итерации: попытки и принятые по семейству, подряд отклонённые — своё и общее."""
    counters = state.setdefault("families", {}).setdefault(
        family, {"attempts": 0, "accepted": 0, "rejected_in_row": 0}
    )
    counters["attempts"] += 1
    if accepted:
        counters["accepted"] += 1
        counters["rejected_in_row"] = 0
        state["rejected_in_row"] = 0
        if commit:
            state["last_accepted_commit"] = commit
        if p95_ms is not None:
            state["last_p95_ms"] = p95_ms
    else:
        counters["rejected_in_row"] += 1
        state["rejected_in_row"] = int(state.get("rejected_in_row") or 0) + 1
    if iteration is not None:
        state["iteration"] = max(int(state.get("iteration") or 0), iteration)
    return state


def test_check_entry(run: Path, variant: str = VARIANT) -> dict[str, Any]:
    """Итог `--final-test` прогона `bench.train_resolve` — только для журнала."""
    metrics = read_json(run / "metrics.json")
    if not metrics.get("test_used"):
        raise GoalError(f"{run}: прогон без --final-test")
    report = (((metrics.get("test") or {}).get("variants") or {}).get(variant)) or {}
    by_set = report.get("by_set") or {}
    entry: dict[str, Any] = {"run": str(run), "at": now_iso()}
    for name in SETS:
        block = by_set.get(name)
        if not isinstance(block, Mapping):
            raise GoalError(f"{run}: нет test.variants.{variant}.by_set.{name}")
        learned = block.get("learned") or {}
        entry[name] = {
            "n": block.get("n"),
            "top1": learned.get("top1"),
            "top5": learned.get("top5"),
            "ece": (block.get("calibration") or {}).get("ece"),
        }
    entry["ece_all"] = ((report.get("all") or {}).get("calibration") or {}).get("ece")
    return entry


def lookup(state: Mapping[str, Any], dotted: str) -> Any:
    value: Any = state
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise GoalError(f"в состоянии нет ключа {dotted}")
        value = value[part]
    return value


# ------------------------------------------------------------------ dev30
def dev_query_ids(cv_path: Path) -> dict[str, str]:
    """query_id → slug строк dev прогона `bench.retrieval` (половина — в `meta.split`)."""
    out: dict[str, str] = {}
    with cv_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("meta") or {}).get("split") == "dev":
                out[str(row["query_id"])] = str(row["slug"])
    if not out:
        raise GoalError(f"{cv_path}: нет строк dev")
    return out


def pick_evenly(items: Sequence[str], count: int) -> list[str]:
    """`count` элементов с равным шагом по отсортированному списку, начиная с первого."""
    ordered = sorted(items)
    if count > len(ordered):
        raise GoalError(f"нужно {count} запросов, а есть {len(ordered)}")
    return [ordered[i * len(ordered) // count] for i in range(count)]


def build_dev30(
    *,
    cv_studio: Path,
    cv_phone: Path,
    studio_manifest: Path,
    studio_images: Path,
    phone_manifest: Path,
    phone_images: Path,
    gt: Path,
    out_images: Path,
    out_manifest: Path,
    out_gt: Path,
    count: int = DEV30_QUERIES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Набор задержки: `count` запросов dev × (студия + «телефон»), кадры с префиксом s_/p_."""
    dev = dev_query_ids(cv_studio)
    dev_phone = dev_query_ids(cv_phone)
    if set(dev) != set(dev_phone):
        raise GoalError("запросы dev студии и «телефона» в прогонах CV не совпадают")
    gt_rows = {row["query_id"]: row["slug"] for row in read_tsv(gt)}
    chosen = pick_evenly(list(dev), count)
    for qid in chosen:
        if gt_rows.get(qid) != dev[qid]:
            raise GoalError(f"{qid}: эталон {gt_rows.get(qid)!r} не совпал со slug CV {dev[qid]!r}")
    sources = {
        "s": ({r["query_id"]: r["image_path"] for r in read_tsv(studio_manifest)}, studio_images),
        "p": ({r["query_id"]: r["image_path"] for r in read_tsv(phone_manifest)}, phone_images),
    }
    existing = [p for p in (out_manifest, out_gt) if p.exists()]
    if out_images.exists() and any(out_images.iterdir()):
        existing.append(out_images)
    if existing and not overwrite:
        raise GoalError(f"уже есть {', '.join(map(str, existing))}: --overwrite")
    if overwrite and out_images.exists():
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[tuple[str, str]] = []
    gt_out: list[tuple[str, str]] = []
    for prefix, (paths, images_dir) in sources.items():
        for qid in chosen:
            if qid not in paths:
                raise GoalError(f"{qid}: нет в манифесте набора {prefix}")
            src = images_dir / paths[qid]
            if not src.is_file():
                raise GoalError(f"нет кадра {src}")
            name = f"{prefix}_{Path(paths[qid]).name}"
            shutil.copy2(src, out_images / name)
            manifest_rows.append((f"{prefix}_{qid}", name))
            gt_out.append((f"{prefix}_{qid}", dev[qid]))
    for path, header, rows in (
        (out_manifest, ("query_id", "image_path"), manifest_rows),
        (out_gt, ("query_id", "slug"), gt_out),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["\t".join(header), *("\t".join(row) for row in rows)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {
        "dev_queries": len(dev),
        "chosen": chosen,
        "frames": len(manifest_rows),
        "images": str(out_images),
        "manifest": str(out_manifest),
        "gt": str(out_gt),
    }


# ------------------------------------------------------------------ строгий кэш чтений
class CacheMiss(BaseException):
    """Промах строгого кэша: чтение ушло бы в модель зрения.

    `BaseException`, а не `Exception`: `read_label` превращает сбой читателя в статус чтения, а
    промах должен остановить прогон.
    """


@contextmanager
def strict_cache() -> Iterator[None]:
    """`CachedReader` без модели: попадание — чтение из кэша, промах — `CacheMiss`.

    Так пересборка полей из кэша не зовёт VLM и не занимает видеокарту; Ollama не нужна
    (`available` всегда истина). Прогрев идёт мимо кэша, поэтому он запрещён снаружи.
    """
    from app.reading.readers.cache import CachedReader

    original_read = CachedReader.read
    original_available = CachedReader.available

    def read(self: Any, image: Any, *, crop: Any, budget_ms: int) -> Any:
        key = self.key_for(image, crop=crop)
        cached = self.get(key)
        if cached is None:
            self.stats.misses += 1
            raise CacheMiss(key)
        self.stats.hits += 1
        return cached

    CachedReader.read = read  # type: ignore[method-assign]
    CachedReader.available = lambda self: True  # type: ignore[method-assign]
    try:
        yield
    finally:
        CachedReader.read = original_read  # type: ignore[method-assign]
        CachedReader.available = original_available  # type: ignore[method-assign]


def option_value(argv: Sequence[str], name: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def ocr_cached(argv: Sequence[str], *, changed: Sequence[str] | None = None) -> int:
    """`bench.ocr_bench` только по кэшу чтений: промах — остановка с кодом 4, модель не зовётся.

    В кэше лежит уже разобранное чтение (`CachedReader.put`), а ключ — `params_hash` читателя, в
    который из разбора ответа входит только `POSTPROCESS_VERSION`. Поэтому при правке читателя
    (`READER_PATHS`, против HEAD) путь «из кэша» дал бы 100 % попаданий со старым разбором —
    отказ: поднять `POSTPROCESS_VERSION` (или получить новый `params_hash`) и читать моделью.
    """
    args = list(argv)
    if args and args[0] == "--":
        args = args[1:]
    if option_value(args, "--cache-dir") is None:
        raise GoalError("ocr-cached: нужен --cache-dir (например runs/cache-pairs-ocr)")
    if "--warmup" in args:
        raise GoalError("ocr-cached: --warmup зовёт модель мимо кэша — только --no-warmup")
    touched = matching(changed_paths() if changed is None else changed, READER_PATHS)
    if touched:
        raise GoalError(
            f"ocr-cached: правлен читатель ({', '.join(touched)}) — чтения в кэше собраны старым "
            "кодом. Поднять POSTPROCESS_VERSION (разбор ответа) или получить новый params_hash "
            "(промпт, параметры) и прогнать (c) с моделью"
        )
    if "--no-warmup" not in args:
        args.append("--no-warmup")
    out = option_value(args, "--out")
    from bench import ocr_bench

    try:
        with strict_cache():
            code = ocr_bench.main(args)
    except CacheMiss as miss:
        print(
            f"ПРОМАХ КЭША: {miss} — чтения этого кадра нет, модель не вызывалась; прогон "
            f"{out or ''} неполный. Нужен прогон с моделью (bench.ocr_bench с прогревом, GPU)",
            file=sys.stderr,
        )
        return EXIT_CACHE_MISS
    if code == 0 and out:
        run = read_json(Path(out) / "metrics.json").get("run") or {}
        print(f"кэш: {json.dumps(run.get('cache'), ensure_ascii=False)}", file=sys.stderr)
    return code


# ------------------------------------------------------------------ HTTP
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(
    url: str, *, data: bytes | None = None, content_type: str | None = None, timeout: float = 10.0
) -> dict[str, Any]:
    """GET (или POST, если есть тело) без прокси; ответ — JSON."""
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if content_type:
        request.add_header("Content-Type", content_type)
    with _OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def multipart(field: str, path: Path) -> tuple[bytes, str]:
    boundary = f"svs-goal-{uuid.uuid4().hex}"
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; '
        f'filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'
    ).encode()
    return head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(), (
        f"multipart/form-data; boundary={boundary}"
    )


# ------------------------------------------------------------------ сервис
def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@functools.cache
def _kernel32() -> Any:
    """kernel32 с объявленными типами (HANDLE — указатель, а не int) — отдельно от `windll`."""
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4]
    k.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    return k


def _proc_stat(pid: int | str) -> list[str] | None:
    """Поля `/proc/<pid>/stat` после имени процесса: [0] — состояние, [1] — ppid, [19] — старт."""
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def process_identity(pid: int) -> dict[str, Any] | None:
    """Отпечаток живого процесса: время создания и исполняемый файл.

    PID в Windows переиспользуется: pid-файл, переживший сервис, может указывать на чужой процесс.
    Пара «pid + время создания» уникальна. `None` — процесса нет или ОС его не показывает.
    """
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k = _kernel32()
        handle = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not k.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
                return None
            created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
            size = wintypes.DWORD(32768)
            buf = ctypes.create_unicode_buffer(size.value)
            ok = k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            return {"created": created, "exe": buf.value if ok else ""}
        finally:
            k.CloseHandle(handle)
    fields = _proc_stat(pid)
    if fields is None or len(fields) < 20 or fields[0] == "Z":
        return None
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = ""
    return {"created": int(fields[19]), "exe": exe}


def process_parents() -> dict[int, int]:
    """pid → pid родителя у всех живых процессов (Windows — Toolhelp32, Linux — /proc)."""
    out: dict[int, int] = {}
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        k = _kernel32()
        snap = k.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if not snap or snap == ctypes.c_void_p(-1).value:
            return out
        try:
            entry = Entry()
            entry.dwSize = ctypes.sizeof(Entry)
            ok = k.Process32FirstW(snap, ctypes.byref(entry))
            while ok:
                out[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                ok = k.Process32NextW(snap, ctypes.byref(entry))
        finally:
            k.CloseHandle(snap)
        return out
    for stat in Path("/proc").glob("[0-9]*/stat"):
        fields = _proc_stat(stat.parent.name)
        if fields is not None and len(fields) > 1 and fields[1].isdigit():
            out[int(stat.parent.name)] = int(fields[1])
    return out


def descendants(pid: int) -> list[dict[str, Any]]:
    """Живые потомки `pid` с отпечатками. Ребёнок не старше родителя — иначе ppid переиспользован."""
    root = process_identity(pid)
    if root is None:
        return []
    parents = process_parents()
    out: list[dict[str, Any]] = []
    seen = {pid}
    frontier = [(pid, root["created"])]
    while frontier:
        parent, parent_created = frontier.pop()
        for child, ppid in parents.items():
            if ppid != parent or child in seen:
                continue
            ident = process_identity(child)
            if ident is None or ident["created"] < parent_created:
                continue
            seen.add(child)
            out.append({"pid": child, **ident})
            frontier.append((child, ident["created"]))
    return out


def pid_owner(pid: int, info: Mapping[str, Any]) -> tuple[str, str]:
    """Чей процесс под `pid` из pid-файла: `ours`, `dead`, `foreign` или `unknown` (+ пояснение).

    Свой — только если отпечаток из `service.json` (записан при запуске) совпал с живым процессом.
    """
    if not pid or not pid_alive(pid):
        return "dead", "процесса нет"
    expected = info.get("process") if info.get("pid") == pid else None
    if not expected:
        return "unknown", "в service.json нет отпечатка этого pid — не проверить, что он наш"
    current = process_identity(pid)
    if current != expected:
        return "foreign", f"сейчас это {current}, а сервис был {expected}"
    return "ours", ""


def read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def kill_process(pid: int) -> None:
    """Убить один процесс (на Windows — с деревом потомков)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def kill_recorded_descendants(info: Mapping[str, Any]) -> list[int]:
    """Добить потомков сервиса из `service.json`, которые пережили лаунчер (с тем же отпечатком)."""
    killed: list[int] = []
    for item in info.get("descendants") or []:
        pid = int(item.get("pid") or 0)
        expected = {"created": item.get("created"), "exe": item.get("exe")}
        if pid and process_identity(pid) == expected:
            kill_process(pid)
            killed.append(pid)
    return killed


def kill_tree(pid: int, *, wait_s: float = 15.0) -> bool:
    """Остановить процесс со всеми потомками. `.venv\\Scripts\\python.exe` у uv — лаунчер,
    сам сервис — его дочерний процесс, поэтому на Windows — `taskkill /T`."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.2)
    if os.name != "nt":
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return not pid_alive(pid)


def spawn_detached(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path):
    """Процесс, который переживает скрипт: вывод — в журнал, своя группа процессов."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        fh.write(f"\n=== {now_iso()} {' '.join(command)}\n".encode())
        fh.flush()
        kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "env": dict(env),
            "stdin": subprocess.DEVNULL,
            "stdout": fh,
            "stderr": subprocess.STDOUT,
        }
        if os.name == "nt":
            flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            try:  # CREATE_BREAKAWAY_FROM_JOB: не умереть вместе с заданием оболочки
                return subprocess.Popen(command, creationflags=flags | 0x01000000, **kwargs)
            except OSError:
                return subprocess.Popen(command, creationflags=flags, **kwargs)
        return subprocess.Popen(command, start_new_session=True, **kwargs)


def health_problems(health: Mapping[str, Any]) -> list[str]:
    """То же условие, что у `run_eval.sh` перед отчётным прогоном."""
    problems: list[str] = []
    if health.get("status") != "ready":
        problems.append(f"status={health.get('status')}")
    if health.get("degraded_reasons"):
        problems.append(f"degraded_reasons: {', '.join(health['degraded_reasons'])}")
    warm_flags = ((health.get("warm") or {}).get("scan") or {}).get("degraded") or []
    if warm_flags:
        problems.append(f"прогревочный скан с флагами: {', '.join(warm_flags)}")
    if health.get("warnings"):
        problems.append("предупреждения настроек: " + " | ".join(health["warnings"]))
    if (health.get("provenance") or {}).get("consistent") is False:
        problems.append("provenance.consistent = false")
    return problems


def service_environment(
    overrides: Mapping[str, str], state: Mapping[str, Any], base: Mapping[str, str] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Окружение сервиса: текущее + `service_env` состояния + `--env`. Возвращает и заметки."""
    env = dict(os.environ if base is None else base)
    notes: list[str] = []
    if not env.get("SVS_DATASET_DIR") and DATASET_DIR_DEFAULT.is_dir():
        env["SVS_DATASET_DIR"] = str(DATASET_DIR_DEFAULT)
    env.update({str(k): str(v) for k, v in (state.get("service_env") or {}).items()})
    env.update(overrides)
    if env.get("CUDA_VISIBLE_DEVICES") == "-1" and "CUDA_VISIBLE_DEVICES" not in overrides:
        env.pop("CUDA_VISIBLE_DEVICES")
        notes.append("CUDA_VISIBLE_DEVICES=-1 из окружения снят: сервис считает на видеокарте")
    env["PYTHONIOENCODING"] = "utf-8"
    return env, notes


def unload_vlm(ollama_url: str, model: str) -> str:
    """Выгрузить модель зрения из памяти Ollama (`keep_alive: 0`) и сказать, что там осталось."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    try:
        http_json(
            f"{ollama_url.rstrip('/')}/api/generate",
            data=body,
            content_type="application/json",
            timeout=60,
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return f"{model} не выгружена: {exc}"
    try:
        loaded = [m.get("name") for m in http_json(f"{ollama_url}/api/ps").get("models", [])]
    except (urllib.error.URLError, OSError, ValueError):
        loaded = ["(api/ps не ответил)"]
    return f"{model} выгружена; в памяти Ollama: {', '.join(map(str, loaded)) or 'пусто'}"


def expected_service_model(env: Mapping[str, str], state: Mapping[str, Any]) -> Path | None:
    """Модель resolve, на которой должен работать сервис: `SVS_RESOLVE_MODEL` окружения, иначе
    принятая `state.inputs.service_model`. Относительные пути — от корня репозитория (оттуда
    стартует сервис)."""
    value = (env.get("SVS_RESOLVE_MODEL") or "").strip() or str(
        (state.get("inputs") or {}).get("service_model") or ""
    )
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def service_model_problems(health: Mapping[str, Any], expected: Path | None) -> list[str]:
    """Сервис поднялся не на той модели: имя файла и `meta.trained_at` против `/v1/health`."""
    if expected is None:
        return []
    if not expected.is_file():
        return [f"нет файла ожидаемой модели resolve {expected}"]
    meta = read_json(expected).get("meta") or {}
    model = health.get("model") or {}
    if model.get("resolve") == expected.name and model.get("resolve_trained_at") == meta.get(
        "trained_at"
    ):
        return []
    message = (
        f"сервис на модели {model.get('resolve')!r} (trained_at "
        f"{model.get('resolve_trained_at')!r}), а ждали {expected.name} (trained_at "
        f"{meta.get('trained_at')!r}): DEFAULT_RESOLVE_MODEL или SVS_RESOLVE_MODEL не те"
    )
    return [message]


def service_up(
    *,
    overrides: Mapping[str, str],
    url: str = DEFAULT_URL,
    timeout_s: float = SERVICE_TIMEOUT_S,
    command: Sequence[str] | None = None,
    pid_path: Path = PID_PATH,
    info_path: Path = SERVICE_INFO_PATH,
    log_path: Path = SERVICE_LOG,
    state: Mapping[str, Any] | None = None,
    stop: Callable[[], Any] | None = None,
    unload: Callable[[str, str], str] | None = unload_vlm,
    poll_s: float = 2.0,
) -> int:
    """Поднять `python -m app.api` в фоне и дождаться `/v1/health` после прогрева.

    Сервис, упавший до готовности или не успевший за `timeout_s`, останавливается целиком
    (`stop`, по умолчанию `service_down`: дерево процессов, потомки из `service.json` и выгрузка
    VLM). Готовый сервис сверяется с ожидаемой моделью resolve (`expected_service_model`).
    """
    state = state if state is not None else load_state()
    if pid_path.is_file():
        old = read_pid(pid_path)
        info = read_json(info_path) if info_path.is_file() else {}
        owner, why = pid_owner(old, info)
        if owner == "ours":
            print(f"сервис уже запущен (pid {old}): сначала service-down", file=sys.stderr)
            return EXIT_SERVICE
        if owner == "unknown":
            print(
                f"pid {old} из {pid_path} жив, но {why}: остановить вручную "
                f"(taskkill /PID {old} /T /F), если это сервис, и удалить pid-файл",
                file=sys.stderr,
            )
            return EXIT_SERVICE
        print(f"pid-файл устарел (pid {old}: {why}) — удаляю, процесс не трогаю")
        pid_path.unlink()
        info_path.unlink(missing_ok=True)
    try:
        http_json(f"{url}/v1/health", timeout=3)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    else:
        print(f"{url} уже отвечает, а pid-файла нет: чужой процесс на порту", file=sys.stderr)
        return EXIT_SERVICE
    env, notes = service_environment(overrides, state)
    for note in notes:
        print(note)
    shown = {k: env[k] for k in sorted(env) if k.startswith("SVS_") or k == "CUDA_VISIBLE_DEVICES"}
    print(f"окружение сервиса: {json.dumps(shown, ensure_ascii=False)}")
    proc = spawn_detached(
        list(command or [sys.executable, "-m", "app.api"]), cwd=REPO_ROOT, env=env, log=log_path
    )
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    info: dict[str, Any] = {
        "pid": proc.pid,
        "process": process_identity(proc.pid),
        "descendants": [],
        "url": url,
        "started_at": now_iso(),
        "overrides": dict(overrides),
        "resolve_model": env.get("SVS_RESOLVE_MODEL") or None,
        "vlm_model": env.get("SVS_VLM_MODEL") or None,
        "ollama_url": env.get("SVS_OLLAMA_URL") or None,
        "log": str(log_path),
    }
    write_json(info_path, info)
    halt = stop or (
        lambda: service_down(pid_path=pid_path, info_path=info_path, url=url, unload=unload)
    )

    def track() -> None:
        """Запомнить потомков лаунчера: если он умрёт первым, service-down найдёт их по отпечатку."""
        known = {(d["pid"], d["created"]) for d in info["descendants"]}
        fresh = [d for d in descendants(proc.pid) if (d["pid"], d["created"]) not in known]
        if fresh:
            info["descendants"] = [*info["descendants"], *fresh]
            write_json(info_path, info)

    print(f"pid {proc.pid}, журнал {log_path}; жду /v1/health до {timeout_s:.0f} с…")
    deadline = time.monotonic() + timeout_s
    health: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        track()
        if proc.poll() is not None:
            print(f"сервис завершился с кодом {proc.returncode} — прибираю:\n{log_tail(log_path)}")
            halt()
            return EXIT_SERVICE
        try:
            health = http_json(f"{url}/v1/health", timeout=5)
        except (urllib.error.URLError, OSError, ValueError):
            health = None
        if health is not None and health.get("status") != "starting":
            break
        time.sleep(poll_s)
    if health is None or health.get("status") == "starting":
        print(f"за {timeout_s:.0f} с /v1/health не готов — останавливаю:\n{log_tail(log_path)}")
        halt()
        return EXIT_SERVICE
    track()
    summary = {
        "status": health.get("status"),
        "degraded_reasons": health.get("degraded_reasons"),
        "warnings": health.get("warnings"),
        "model": health.get("model"),
        "index": (health.get("index") or {}).get("path"),
        "vlm_warm": (health.get("vlm") or {}).get("warm"),
        "warm_scan": (health.get("warm") or {}).get("scan"),
        "provenance": health.get("provenance"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    problems = health_problems(health)
    if problems:
        print("не готов к отчётному прогону (как проверяет run_eval.sh): " + "; ".join(problems))
    wrong_model = service_model_problems(health, expected_service_model(env, state))
    if wrong_model:
        print("не та модель resolve: " + "; ".join(wrong_model))
    if problems or wrong_model:
        print("сервис оставлен работать; остановка — service-down")
        return EXIT_NOT_READY
    print("готов: ready, без предупреждений, provenance сходится, модель resolve ожидаемая")
    return EXIT_OK


def log_tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(журнала нет)"
    return "\n".join(text.splitlines()[-lines:])


def service_down(
    *,
    pid_path: Path = PID_PATH,
    info_path: Path = SERVICE_INFO_PATH,
    url: str | None = None,
    unload: Callable[[str, str], str] | None = unload_vlm,
) -> int:
    """Остановить сервис по pid-файлу и выгрузить VLM из Ollama.

    Убивается только свой процесс: pid из файла и отпечаток из `service.json` (время создания и
    исполняемый файл) должны совпасть с живым процессом. Устаревший pid-файл, чей pid занял
    чужой процесс, просто удаляется. Потомки сервиса, пережившие лаунчер, добиваются по своим
    отпечаткам.
    """
    from app.api.config import DEFAULT_VLM_MODEL

    info = read_json(info_path) if info_path.is_file() else {}
    code = EXIT_OK
    if pid_path.is_file():
        pid = read_pid(pid_path)
        owner, why = pid_owner(pid, info)
        if owner == "ours":
            stopped = kill_tree(pid)
            print(f"сервис pid {pid}: {'остановлен' if stopped else 'НЕ остановился'}")
            code = EXIT_OK if stopped else EXIT_SERVICE
        elif owner == "dead":
            print(f"pid {pid} уже не работает")
        elif owner == "foreign":
            print(f"pid {pid} занят чужим процессом ({why}): pid-файл устарел, процесс не трогаю")
        else:
            print(
                f"pid {pid} жив, но {why}: не трогаю — если это сервис, остановить вручную "
                f"(taskkill /PID {pid} /T /F)"
            )
            code = EXIT_SERVICE
        pid_path.unlink(missing_ok=True)
    else:
        print("pid-файла нет: останавливать нечего")
    orphans = kill_recorded_descendants(info)
    if orphans:
        print(f"добиты процессы сервиса, пережившие лаунчер: {', '.join(map(str, orphans))}")
    health_url = url or info.get("url") or DEFAULT_URL
    try:
        http_json(f"{health_url}/v1/health", timeout=3)
    except (urllib.error.URLError, OSError, ValueError):
        pass
    else:
        print(f"ВНИМАНИЕ: {health_url} всё ещё отвечает — на порту чужой процесс")
        code = EXIT_SERVICE
    if unload is not None:
        model = info.get("vlm_model") or os.environ.get("SVS_VLM_MODEL") or DEFAULT_VLM_MODEL
        ollama = (
            info.get("ollama_url") or os.environ.get("SVS_OLLAMA_URL") or "http://127.0.0.1:11434"
        )
        print(unload(ollama, model))
    info_path.unlink(missing_ok=True)
    return code


# ------------------------------------------------------------------ 13 реальных кадров
@dataclass(frozen=True)
class Frame:
    query_id: str
    source: str
    path: Path
    gt: str


def frames_from(source: str, manifest: Path, images_dir: Path, gt_path: Path) -> list[Frame]:
    gt = {row["query_id"]: row["slug"] for row in read_tsv(gt_path)}
    frames: list[Frame] = []
    for row in read_tsv(manifest):
        qid = row["query_id"]
        if qid not in gt:
            raise GoalError(f"{source}: у {qid} нет эталона в {gt_path}")
        path = images_dir / row["image_path"]
        if not path.is_file():
            raise GoalError(f"{source}: нет кадра {path}")
        frames.append(Frame(qid, source, path, gt[qid]))
    return frames


def real_frames(dataset_dir: Path, rwl_dir: Path) -> list[Frame]:
    """3 кадра организатора и 10 исходных кадров russian_wine_labels_raw."""
    return [
        *frames_from(
            "public",
            dataset_dir / "eval" / "queries.tsv",
            dataset_dir / "eval" / "queries",
            REPO_ROOT / "data" / "gt" / "public_gt.tsv",
        ),
        *frames_from(
            "rwl_src",
            REPO_ROOT / "runs" / "rwl" / "src_manifest.tsv",
            rwl_dir,
            EVAL_SETS_DIR / "rwl_src_gt.tsv",
        ),
    ]


def gt_rank(gt: str, answer: Mapping[str, Any]) -> int | str:
    """Место эталона в top-5 ответа: 1–5, «>5» или «вне каталога» (эталон `__none__`)."""
    if gt == NONE_SLUG:
        return OUT_OF_CATALOG
    slugs = [item.get("slug") for item in (answer.get("top5") or [])][:5]
    if gt in slugs:
        return slugs.index(gt) + 1
    if answer.get("slug") == gt:
        return 1
    return ">5"


def scan_frames(frames: Sequence[Frame], url: str, *, timeout: float = 30.0) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        body, ctype = multipart("image", frame.path)
        started = time.perf_counter()
        try:
            answer = http_json(f"{url}/v1/scan", data=body, content_type=ctype, timeout=timeout)
            error = answer.get("error")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            answer, error = {}, f"http: {exc}"
        latency = round((time.perf_counter() - started) * 1000, 1)
        rows.append(
            {
                "query_id": frame.query_id,
                "source": frame.source,
                "image": frame.path.name,
                "gt": frame.gt,
                "slug": answer.get("slug"),
                "confidence": (answer.get("confidence") or {}).get("top1"),
                "top5": [item.get("slug") for item in (answer.get("top5") or [])],
                "rank": gt_rank(frame.gt, answer),
                "outcome": answer.get("outcome"),
                "degraded": answer.get("degraded") or [],
                "latency_ms": latency,
                "service_total_ms": (answer.get("timings_ms") or {}).get("total"),
                "error": error,
            }
        )
    return rows


def real13_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    inside = [row for row in rows if row["gt"] != NONE_SLUG]
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "frames": len(rows),
        "in_catalog": len(inside),
        "top1": sum(row["rank"] == 1 for row in inside),
        "in_top5": sum(isinstance(row["rank"], int) for row in inside),
        "out_of_catalog": len(rows) - len(inside),
        "errors": sum(bool(row["error"]) for row in rows),
        "degraded": sum(bool(row["degraded"]) for row in rows),
        "latency_ms": {
            "p50": nearest_rank(latencies, 0.5),
            "p95": nearest_rank(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
    }


def real13_markdown(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    lines = [
        "| кадр | набор | эталон | ответ | p(top-1) | ранг эталона | мс |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        conf = "—" if row["confidence"] is None else comma(row["confidence"], 3)
        answer = row["slug"] or f"null ({row['error'] or row['outcome']})"
        lines.append(
            f"| {row['query_id']} | {row['source']} | {row['gt']} | {answer} | {conf} | "
            f"{row['rank']} | {row['latency_ms']:.0f} |"
        )
    lat = summary["latency_ms"]
    lines.append("")
    lines.append(
        f"В каталоге {summary['in_catalog']}: top-1 {summary['top1']}, в top-5 "
        f"{summary['in_top5']}; вне каталога {summary['out_of_catalog']}; ошибок "
        f"{summary['errors']}, с флагами degraded {summary['degraded']}; задержка p50 "
        f"{comma(lat['p50'], 0)} / p95 {comma(lat['p95'], 0)} / max {comma(lat['max'], 0)} мс."
    )
    return "\n".join(lines)


def run_real13(
    frames: Sequence[Frame], url: str, out: Path, *, expected_model: Path | None = None
) -> dict[str, Any]:
    """13 кадров через `/v1/scan`. `expected_model` — принятая модель resolve: если сервис не
    на ней, отчёт пишется с пометкой в `health.model_problems` и первой строкой `real13.md`."""
    try:
        health = http_json(f"{url}/v1/health", timeout=5)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise GoalError(f"сервис не отвечает на {url}/v1/health: {exc}") from exc
    wrong_model = service_model_problems(health, expected_model)
    rows = scan_frames(frames, url)
    summary = real13_summary(rows)
    report = {
        "at": now_iso(),
        "url": url,
        "health": {
            "status": health.get("status"),
            "problems": health_problems(health),
            "model": health.get("model"),
            "expected_model": str(expected_model) if expected_model else None,
            "model_problems": wrong_model,
            "provenance": health.get("provenance"),
        },
        "summary": summary,
        "rows": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "real13.json", report)
    markdown = real13_markdown(rows, summary)
    if wrong_model:
        markdown = "**НЕ ПРИНЯТАЯ МОДЕЛЬ:** " + "; ".join(wrong_model) + "\n\n" + markdown
    (out / "real13.md").write_text(markdown + "\n", encoding="utf-8", newline="\n")
    report["markdown"] = markdown
    return report


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/goal.py", description="Инструменты цикла улучшений сканера."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "verdict", help="сравнить два прогона bench.train_resolve по правилу приёмки"
    )
    p.add_argument("--base", type=Path, required=True, help="последний принятый прогон")
    p.add_argument("--cand", type=Path, required=True, help="прогон кандидата")
    p.add_argument("--p95", type=float, help="p95 задержки из judge.json на dev30, мс")
    p.add_argument("--out", type=Path, required=True, help="куда записать JSON вердикта")
    p.add_argument("--variant", default=VARIANT)
    p.add_argument("--iter", type=int, help="номер итерации для строки журнала")
    p.add_argument(
        "--env",
        action="append",
        metavar="K=V",
        help="переменные SVS_* для provenance сверх текущего окружения и service_env состояния",
    )

    p = sub.add_parser(
        "begin", help="начать итерацию: снимок HEAD и неотслеживаемых файлов для revert"
    )
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)

    p = sub.add_parser("revert", help="сохранить правку в rejected.patch и откатить её")
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)

    p = sub.add_parser("state", help="runs/goal/state.json: принятые входы и счётчики")
    state_sub = p.add_subparsers(dest="action", required=True)
    s = state_sub.add_parser("show", help="напечатать состояние (или одно значение)")
    s.add_argument("--key", help="путь через точку, например inputs.cv.pairs")
    s = state_sub.add_parser("set", help="применить JSON Merge Patch (RFC 7386)")
    s.add_argument(
        "patch", help='например \'{"inputs": {"resolve_run": "runs/goal/iter03/resolve"}}\''
    )
    s = state_sub.add_parser("record", help="учесть итерацию в счётчиках семейства")
    s.add_argument("--family", required=True)
    s.add_argument("--verdict", required=True, choices=["accept", "reject"])
    s.add_argument("--iter", type=int)
    s.add_argument("--commit", help="принятый коммит")
    s.add_argument("--p95", type=float, help="p95 принятого сервиса, мс")
    s = state_sub.add_parser("test-check", help="дописать итог --final-test (только журнал)")
    s.add_argument("--run", type=Path, required=True, help="прогон train_resolve с --final-test")
    s.add_argument("--variant", default=VARIANT)

    p = sub.add_parser("dev30", help="собрать набор задержки: 15 запросов dev × студия/«телефон»")
    p.add_argument(
        "--cv-studio", type=Path, default=REPO_ROOT / "runs/cvall-s2so400m-pairs/predictions.jsonl"
    )
    p.add_argument(
        "--cv-phone",
        type=Path,
        default=REPO_ROOT / "runs/cvall-s2so400m-pairs_phone/predictions.jsonl",
    )
    p.add_argument("--studio-manifest", type=Path, default=PAIRS_OCR_DIR / "pairs_manifest.tsv")
    p.add_argument("--studio-images", type=Path, default=PAIRS_IMAGES_DEFAULT)
    p.add_argument(
        "--phone-manifest", type=Path, default=PAIRS_OCR_DIR / "pairs_phone_manifest.tsv"
    )
    p.add_argument("--phone-images", type=Path, default=PHONE_IMAGES_DEFAULT)
    p.add_argument("--gt", type=Path, default=PAIRS_OCR_DIR / "pairs_gt.tsv")
    p.add_argument("--out-images", type=Path, default=EVAL_SETS_DIR / "dev30-images")
    p.add_argument("--out-manifest", type=Path, default=EVAL_SETS_DIR / "dev30_manifest.tsv")
    p.add_argument("--out-gt", type=Path, default=EVAL_SETS_DIR / "dev30_gt.tsv")
    p.add_argument("--count", type=int, default=DEV30_QUERIES)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser(
        "ocr-cached", help="bench.ocr_bench только из кэша чтений: промах — стоп, модель не зовётся"
    )
    p.add_argument("bench_args", nargs=argparse.REMAINDER, help="-- и аргументы bench.ocr_bench")

    p = sub.add_parser("service-up", help="поднять python -m app.api в фоне и дождаться health")
    p.add_argument("--env", action="append", metavar="K=V", help="переменные сервиса поверх state")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--timeout", type=float, default=SERVICE_TIMEOUT_S)

    p = sub.add_parser("service-down", help="остановить сервис и выгрузить VLM из Ollama")
    p.add_argument("--url", help="адрес сервиса для проверки, что порт освободился")

    p = sub.add_parser("real13", help="13 реальных кадров через /v1/scan запущенного сервиса")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--dataset-dir", type=Path, help="датасет организатора (SVS_DATASET_DIR)")
    p.add_argument("--rwl-dir", type=Path, default=RWL_DIR_DEFAULT)
    p.add_argument(
        "--expect-model",
        help="модель resolve, на которой должен быть сервис; по умолчанию принятая из state "
        "(service_env.SVS_RESOLVE_MODEL или inputs.service_model)",
    )
    return parser


def cmd_verdict(args: argparse.Namespace) -> int:
    base = load_resolve_run(args.base, args.variant)
    cand = load_resolve_run(args.cand, args.variant)
    env = {**os.environ, **(load_state().get("service_env") or {}), **parse_env(args.env)}
    prov = service_provenance(args.cand / "models" / f"{args.variant}.json", env)
    result = decide(base, cand, p95_ms=args.p95, provenance=prov, changed=changed_paths(REPO_ROOT))
    result["markdown"] = verdict_markdown(result, args.iter)
    result["at"] = now_iso()
    write_json(args.out, result)
    for name in SETS:
        block = result["sets"][name]
        print(
            f"{SET_TITLES[name]}: исправлено {block['fixed'] or '—'}; сломано {block['broken'] or '—'}",
            file=sys.stderr,
        )
    print(result["markdown"])
    return EXIT_OK if result["accepted"] else EXIT_REJECT


def cmd_state(args: argparse.Namespace) -> int:
    state = load_state()
    if args.action == "show":
        value = lookup(state, args.key) if args.key else state
        print(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1))
        return EXIT_OK
    if args.action == "set":
        try:
            patch = json.loads(args.patch)
        except ValueError as exc:
            raise GoalError(f"патч не JSON: {exc}") from exc
        if not isinstance(patch, Mapping):
            raise GoalError("патч должен быть объектом JSON")
        state = merge_patch(state, patch)
    elif args.action == "record":
        record_attempt(
            state,
            family=args.family,
            accepted=args.verdict == "accept",
            iteration=args.iter,
            commit=args.commit,
            p95_ms=args.p95,
        )
    elif args.action == "test-check":
        entry = test_check_entry(args.run, args.variant)
        entry["iteration"] = state.get("iteration")
        state.setdefault("test_checks", []).append(entry)
        s, p = entry[STUDIO], entry[PHONE]
        print(
            f"TEST (только журнал): студия top-1 {comma(100 * s['top1'])} / top-5 "
            f"{comma(100 * s['top5'])}, «телефон» top-1 {comma(100 * p['top1'])} / top-5 "
            f"{comma(100 * p['top5'])}, ECE {entry['ece_all']}"
        )
    save_state(state)
    print(json.dumps(state, ensure_ascii=False, indent=1))
    return EXIT_OK


def cmd_begin(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    state = load_state(repo / "runs" / "goal" / "state.json")
    report = begin(repo, args.iter, last_accepted=state.get("last_accepted_commit"))
    print(f"итерация {args.iter:02d}: HEAD {report['head'][:12]}, снимок {report['start']}")
    print(
        f"неотслеживаемые до итерации (откат их не удалит): {', '.join(report['untracked']) or '—'}"
    )
    return EXIT_OK


def cmd_revert(args: argparse.Namespace) -> int:
    report = revert(args.repo, args.iter)
    if report["patch"]:
        print(f"патч: {report['patch']} ({report['patch_bytes']} байт)")
    else:
        print("патч не записан: откатывать нечего")
    print(f"возвращены к HEAD: {', '.join(report['restored']) or '—'}")
    print(f"удалены: {', '.join(report['deleted']) or '—'}")
    print(f"оставлены (журнал и отчёт): {', '.join(report['kept']) or '—'}")
    if report["protected_untouched"]:
        print(
            "НЕ тронуты (runs/, data/, .venv/), хотя изменены: "
            + ", ".join(report["protected_untouched"])
        )
    code = EXIT_OK
    if report["untracked_changed"]:
        print(
            "ОТКАТ НЕПОЛНЫЙ: итерация правила неотслеживаемые файлы, бывшие до неё (вернуть "
            "нечем): " + ", ".join(report["untracked_changed"]),
            file=sys.stderr,
        )
        code = EXIT_ERROR
    if report["leftover"]:
        print(
            "ОТКАТ НЕПОЛНЫЙ: после отката отличаются от HEAD: " + ", ".join(report["leftover"]),
            file=sys.stderr,
        )
        code = EXIT_ERROR
    if report["forbidden"]:
        print(
            "НАРУШЕНИЕ «НЕ ТРОГАТЬ»: изменены " + ", ".join(report["forbidden"]) + " — код "
            "откатан, эти файлы нет; вернуть их вручную (git checkout HEAD -- <путь>) и записать "
            "в журнал",
            file=sys.stderr,
        )
        code = EXIT_FORBIDDEN
    return code


def cmd_real13(args: argparse.Namespace) -> int:
    dataset = args.dataset_dir or Path(os.environ.get("SVS_DATASET_DIR") or DATASET_DIR_DEFAULT)
    frames = real_frames(dataset, args.rwl_dir)
    if len(frames) != 13:
        print(f"ВНИМАНИЕ: кадров {len(frames)}, а не 13", file=sys.stderr)
    if args.expect_model:
        expected = expected_service_model({"SVS_RESOLVE_MODEL": args.expect_model}, {})
    else:
        state = load_state()
        expected = expected_service_model(state.get("service_env") or {}, state)
    try:
        report = run_real13(frames, args.url.rstrip("/"), args.out, expected_model=expected)
    except GoalError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_SERVICE
    if report["health"]["problems"]:
        print("ВНИМАНИЕ: сервис не на цепочке замера: " + "; ".join(report["health"]["problems"]))
    print(report["markdown"])
    if report["health"]["model_problems"]:
        print(
            "ОШИБКА: сервис не на принятой модели — в журнал этот прогон не идёт", file=sys.stderr
        )
        return EXIT_ERROR
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verdict":
            return cmd_verdict(args)
        if args.command == "begin":
            return cmd_begin(args)
        if args.command == "revert":
            return cmd_revert(args)
        if args.command == "state":
            return cmd_state(args)
        if args.command == "dev30":
            report = build_dev30(
                cv_studio=args.cv_studio,
                cv_phone=args.cv_phone,
                studio_manifest=args.studio_manifest,
                studio_images=args.studio_images,
                phone_manifest=args.phone_manifest,
                phone_images=args.phone_images,
                gt=args.gt,
                out_images=args.out_images,
                out_manifest=args.out_manifest,
                out_gt=args.out_gt,
                count=args.count,
                overwrite=args.overwrite,
            )
            print(json.dumps(report, ensure_ascii=False, indent=1))
            return EXIT_OK
        if args.command == "ocr-cached":
            return ocr_cached(args.bench_args)
        if args.command == "service-up":
            return service_up(
                overrides=parse_env(args.env), url=args.url.rstrip("/"), timeout_s=args.timeout
            )
        if args.command == "service-down":
            return service_down(url=args.url)
        if args.command == "real13":
            return cmd_real13(args)
    except GoalError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
