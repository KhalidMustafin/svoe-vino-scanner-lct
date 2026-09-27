"""Судья прогона скрипта организатора: `predictions.jsonl` против эталона.

    python -m bench.judge --pred runs/eval-public/predictions.jsonl --gt data/gt/public_gt.tsv
    python -m bench.judge --pred predictions.jsonl --gt field_gt.tsv --out judge.json --json

Вход — ровно то, что пишет `participant_test.sh`: строка на кадр с `query_id`, `image_path`,
`image_sha256`, `predicted_slug` (строка или null) и `latency_ms`. Эталон — TSV с заголовком
`query_id<TAB>slug[<TAB>…]`, где `__none__` — вина нет в каталоге.

Что считается:

    top1_in_catalog   доля верных slug среди кадров, чьё вино есть в каталоге (цифра ТЗ);
                      кадр без строки в предсказаниях — промах
    null_share        доля ответов null — всего и отдельно по кадрам в каталоге и вне его
    out_of_catalog    кадры `__none__`: сколько из них получили null. Как организатор считает
                      такие кадры, неизвестно, поэтому «null = верно» — отдельная строка
                      `if_null_is_correct_for_none` с оговоркой, а не итог
    latency_ms        p50 / p95 (ближайший ранг, как у стендов) / max и число ответов дольше
                      3 000 мс (SLA ТЗ) и 10 000 мс (`curl --max-time 10`: ответ потерян)

Формат тоже проверяется: `\\r` в файле или в slug (jq.exe на Windows пишет CRLF, и slug
приезжает с хвостом `\\r` — такой slug у организатора не совпадёт ни с чем), повтор
`query_id`, битые строки. Найдена проблема формата — код выхода 1.

Коды выхода: 0 — готово; 1 — предсказания с проблемами формата; 2 — ошибка аргументов или
файлов.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bench.metrics import NONE_SLUG, dumps, percentile, print_json

SLA_MS = 3000
CURL_MAX_MS = 10_000
#: Сколько примеров показать в списках промахов и проблем.
SHOW = 10


class JudgeError(ValueError):
    """Файл не разобрался: судить нечего."""


def load_gt(path: Path) -> dict[str, str]:
    """Эталон: `query_id` → slug или `__none__`. Заголовок обязателен, `\\r` снимается."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise JudgeError(f"эталон не читается: {exc}") from exc
    if not lines:
        raise JudgeError(f"{path}: пустой эталон")
    header = [cell.strip() for cell in lines[0].split("\t")]
    if header[:2] != ["query_id", "slug"]:
        raise JudgeError(f"{path}: заголовок должен начинаться с query_id<TAB>slug, а не {header}")
    gt: dict[str, str] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split("\t")]
        if len(cells) < 2 or not cells[0] or not cells[1]:
            raise JudgeError(f"{path}:{number}: ожидается query_id<TAB>slug")
        if cells[0] in gt:
            raise JudgeError(f"{path}:{number}: повтор query_id {cells[0]}")
        gt[cells[0]] = cells[1]
    if not gt:
        raise JudgeError(f"{path}: в эталоне нет строк")
    return gt


def load_predictions(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Строки `predictions.jsonl` и найденные проблемы формата (не исключения)."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise JudgeError(f"предсказания не читаются: {exc}") from exc
    problems: list[str] = []
    carriage = raw.count(b"\r")
    if carriage:
        problems.append(f"в файле {carriage} символов \\r (CRLF от jq.exe?)")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    text = raw.decode("utf-8", errors="replace")
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"строка {number}: не JSON ({exc.msg})")
            continue
        if not isinstance(row, dict) or not isinstance(row.get("query_id"), str):
            problems.append(f"строка {number}: нет query_id")
            continue
        qid = row["query_id"]
        if qid in seen:
            problems.append(f"строка {number}: повтор query_id {qid}")
            continue
        seen.add(qid)
        slug = row.get("predicted_slug")
        if slug is not None and not isinstance(slug, str):
            problems.append(f"строка {number}: predicted_slug не строка и не null")
            slug = None
        if isinstance(slug, str) and (slug != slug.strip() or "\r" in slug):
            problems.append(f"строка {number}: slug {slug!r} с пробелом или \\r по краям")
        latency = row.get("latency_ms")
        if not isinstance(latency, int | float) or isinstance(latency, bool):
            problems.append(f"строка {number}: latency_ms не число")
            latency = None
        rows.append({"query_id": qid, "predicted_slug": slug, "latency_ms": latency})
    return rows, problems


def _share(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def judge(gt: Mapping[str, str], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Метрики прогона. Предсказание без эталона в счёт не идёт, но называется."""
    by_qid = {str(row["query_id"]): row for row in rows}
    in_catalog = [qid for qid, slug in gt.items() if slug != NONE_SLUG]
    out_catalog = [qid for qid, slug in gt.items() if slug == NONE_SLUG]
    missing = [qid for qid in gt if qid not in by_qid]
    extra = [qid for qid in by_qid if qid not in gt]

    def predicted(qid: str) -> str | None:
        row = by_qid.get(qid)
        return row.get("predicted_slug") if row else None

    correct = [qid for qid in in_catalog if predicted(qid) == gt[qid]]
    wrong = [
        {"query_id": qid, "slug": gt[qid], "predicted": predicted(qid)}
        for qid in in_catalog
        if predicted(qid) != gt[qid]
    ]
    in_nulls = sum(qid in by_qid and predicted(qid) is None for qid in in_catalog)
    out_nulls = sum(qid in by_qid and predicted(qid) is None for qid in out_catalog)
    judged = [by_qid[qid] for qid in gt if qid in by_qid]
    nulls = sum(row.get("predicted_slug") is None for row in judged)
    latencies = [float(row["latency_ms"]) for row in judged if row.get("latency_ms") is not None]
    null_correct = len(correct) + out_nulls
    return {
        "queries": len(gt),
        "predictions": len(rows),
        "judged": len(judged),
        "missing": len(missing),
        "missing_examples": missing[:SHOW],
        "extra": len(extra),
        "extra_examples": extra[:SHOW],
        "in_catalog": {
            "n": len(in_catalog),
            "correct": len(correct),
            "top1": _share(len(correct), len(in_catalog)),
            "null": in_nulls,
            "null_share": _share(in_nulls, len(in_catalog)),
            "wrong_examples": wrong[:SHOW],
        },
        "out_of_catalog": {
            "n": len(out_catalog),
            "null": out_nulls,
            "null_share": _share(out_nulls, len(out_catalog)),
            "answered": len(out_catalog) - out_nulls - sum(q not in by_qid for q in out_catalog),
        },
        "null_share": _share(nulls, len(judged)),
        "if_null_is_correct_for_none": {
            "accuracy": _share(null_correct, len(gt)),
            "note": "допущение: кадр вне каталога засчитан, если ответ null. Как считает "
            "организатор, неизвестно — это не итоговая цифра",
        },
        "latency_ms": {
            "n": len(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
            f"over_{SLA_MS}": sum(value > SLA_MS for value in latencies),
            f"over_{CURL_MAX_MS}": sum(value > CURL_MAX_MS for value in latencies),
        },
    }


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    inside, outside, latency = report["in_catalog"], report["out_of_catalog"], report["latency_ms"]

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f} %"

    lines = [
        (
            f"кадров в эталоне: {report['queries']}, строк в предсказаниях: "
            f"{report['predictions']}, без предсказания: {report['missing']}, "
            f"лишних: {report['extra']}"
        ),
        (
            f"top-1 в каталоге: {pct(inside['top1'])} ({inside['correct']} из {inside['n']}), "
            f"null среди них: {inside['null']}"
        ),
        (
            f"вне каталога: {outside['n']}, из них null: {outside['null']} "
            f"({pct(outside['null_share'])})"
        ),
        f"доля null всего: {pct(report['null_share'])}",
        (
            "если null на __none__ — верно: "
            f"{pct(report['if_null_is_correct_for_none']['accuracy'])} (допущение, не итог)"
        ),
        (
            f"задержка, мс: p50 {latency['p50']}, p95 {latency['p95']}, max {latency['max']}; "
            f"> {SLA_MS}: {latency[f'over_{SLA_MS}']}, "
            f"> {CURL_MAX_MS}: {latency[f'over_{CURL_MAX_MS}']}"
        ),
    ]
    for problem in report.get("format_problems", []):
        lines.append(f"ФОРМАТ: {problem}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.judge",
        description="Точность и задержки прогона participant_test.sh против эталона.",
    )
    parser.add_argument("--pred", type=Path, required=True, help="predictions.jsonl скрипта")
    parser.add_argument("--gt", type=Path, required=True, help="TSV query_id<TAB>slug|__none__")
    parser.add_argument("--out", type=Path, help="куда записать отчёт JSON")
    parser.add_argument("--json", action="store_true", help="печатать JSON вместо сводки")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        gt = load_gt(args.gt)
        rows, problems = load_predictions(args.pred)
    except JudgeError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    report = judge(gt, rows)
    report["format_problems"] = problems[: SHOW * 2]
    report["format_problems_total"] = len(problems)
    report["files"] = {"pred": str(args.pred), "gt": str(args.gt)}
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(dumps(report) + "\n", encoding="utf-8")
    if args.json:
        print_json(report)
    else:
        for line in summary_lines(report):
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", "backslashreplace").decode("ascii"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
