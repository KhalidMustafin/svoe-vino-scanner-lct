"""Судья прогона `participant_test.sh`: точность, доля null, задержки и формат файла."""

from __future__ import annotations

import json

import pytest

from bench.judge import judge, load_gt, load_predictions, main

GT = (
    "query_id\tslug\tin_catalog\n"
    "q-1\tkokur-suhoe-2025\t1\n"
    "q-2\tmassandra-muskatel\t1\n"
    "q-3\tabrau-risling\t1\n"
    "q-4\t__none__\t0\n"
    "q-5\tfanagoria-saperavi\t1\n"
)
PREDICTIONS = [
    {"query_id": "q-1", "predicted_slug": "kokur-suhoe-2025", "latency_ms": 900},
    {"query_id": "q-2", "predicted_slug": "massandra-muskatel", "latency_ms": 2100},
    {"query_id": "q-3", "predicted_slug": "abrau-risling-2020", "latency_ms": 3500},
    {"query_id": "q-4", "predicted_slug": None, "latency_ms": 10050},
    {"query_id": "q-extra", "predicted_slug": "x", "latency_ms": 100},
]  # q-5 нет: промах


def write_pred(path, rows, newline="\n"):
    lines = [
        json.dumps({"image_path": "a.jpg", "image_sha256": "0" * 64, **row}, separators=(",", ":"))
        for row in rows
    ]
    path.write_bytes((newline.join(lines) + newline).encode("utf-8"))
    return path


@pytest.fixture
def files(tmp_path):
    gt = tmp_path / "gt.tsv"
    gt.write_text(GT, encoding="utf-8")
    return gt, write_pred(tmp_path / "predictions.jsonl", PREDICTIONS)


def test_known_predictions(files):
    gt_path, pred_path = files
    rows, problems = load_predictions(pred_path)
    assert problems == []
    report = judge(load_gt(gt_path), rows)
    inside = report["in_catalog"]
    assert inside["n"] == 4 and inside["correct"] == 2 and inside["top1"] == 0.5
    assert report["missing"] == 1 and report["missing_examples"] == ["q-5"]
    assert report["extra"] == 1
    assert report["out_of_catalog"] == {"n": 1, "null": 1, "null_share": 1.0, "answered": 0}
    assert report["null_share"] == 0.25  # один null из четырёх оценённых строк
    assert report["if_null_is_correct_for_none"]["accuracy"] == 0.6  # (2 + 1) / 5
    latency = report["latency_ms"]
    assert latency["n"] == 4 and latency["max"] == 10050
    assert latency["p50"] == 3500  # ближайший ранг, без интерполяции
    assert latency["over_3000"] == 2 and latency["over_10000"] == 1
    wrong = {item["query_id"]: item["predicted"] for item in inside["wrong_examples"]}
    assert wrong == {"q-3": "abrau-risling-2020", "q-5": None}


def test_crlf_and_trailing_cr_in_slug_are_format_problems(tmp_path, files):
    gt_path, _ = files
    bad = [{"query_id": "q-1", "predicted_slug": "kokur-suhoe-2025\r", "latency_ms": 1}]
    pred = write_pred(tmp_path / "crlf.jsonl", bad, newline="\r\n")
    rows, problems = load_predictions(pred)
    assert any("\\r" in problem for problem in problems)
    assert len(problems) == 2  # CR в файле и хвост \r у slug
    report = judge(load_gt(gt_path), rows)
    assert report["in_catalog"]["correct"] == 0  # slug с \r не совпадает ни с чем
    assert main(["--pred", str(pred), "--gt", str(gt_path)]) == 1


def test_duplicates_and_broken_lines(tmp_path):
    pred = tmp_path / "p.jsonl"
    pred.write_text(
        '{"query_id":"q-1","predicted_slug":"kokur-suhoe-2025","latency_ms":5}\n'
        '{"query_id":"q-1","predicted_slug":"other","latency_ms":5}\n'
        "not json\n"
        '{"query_id":"q-2","predicted_slug":7,"latency_ms":"slow"}\n',
        encoding="utf-8",
        newline="\n",  # иначе Windows сам допишет \r
    )
    rows, problems = load_predictions(pred)
    assert [row["query_id"] for row in rows] == ["q-1", "q-2"]
    assert rows[1]["predicted_slug"] is None and rows[1]["latency_ms"] is None
    assert len(problems) == 4


def test_cli_writes_json_report(files, tmp_path, capsys):
    gt_path, pred_path = files
    out = tmp_path / "out" / "judge.json"
    assert main(["--pred", str(pred_path), "--gt", str(gt_path), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["in_catalog"]["top1"] == 0.5 and report["format_problems"] == []
    printed = capsys.readouterr().out
    assert "top-1 в каталоге: 50.0 %" in printed
    assert main(["--pred", str(pred_path), "--gt", str(gt_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["latency_ms"]["over_10000"] == 1


def test_bad_gt_is_exit_2(tmp_path, files):
    _, pred_path = files
    gt = tmp_path / "bad.tsv"
    gt.write_text("id\tanswer\nq-1\tx\n", encoding="utf-8")
    assert main(["--pred", str(pred_path), "--gt", str(gt)]) == 2
    assert main(["--pred", str(tmp_path / "none.jsonl"), "--gt", str(gt)]) == 2


def test_public_gt_parses_with_crlf(tmp_path):
    gt = tmp_path / "public.tsv"
    gt.write_bytes(b"query_id\tslug\tin_catalog\r\nq-000001\t__none__\t0\r\nq-000002\tm-16\t1\r\n")
    assert load_gt(gt) == {"q-000001": "__none__", "q-000002": "m-16"}
