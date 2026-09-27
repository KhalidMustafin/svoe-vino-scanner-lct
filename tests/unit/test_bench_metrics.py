import json

import pytest

from bench import metrics
from bench.metrics import (
    NONE_SLUG,
    answered_by_frame_neighbor,
    best_cer,
    cv_vs_cv_ocr,
    evaluate,
    expected_label_tokens,
    field_metrics,
    latency_summary,
    lev,
    norm,
    ooc_reject_rate,
    percentile,
    synth_predictions,
    text_topk,
    token_cer_table,
    twin_resolved_top1,
)


def _record(slug, *, winery="Фанагория", grape="мерло", sugar="suhoe", year=None, mates=()):
    tokens = [
        {"field": "winery", "text": winery.lower(), "primary": True},
        {"field": "grape", "text": grape, "primary": True},
        {"field": "grape", "text": "merlot", "primary": False},
        {"field": "sugar", "text": "сухое", "primary": True},
    ]
    if year:
        tokens.append({"field": "year", "text": str(year), "primary": True})
    return {
        "slug": slug,
        "name": grape,
        "winery": winery,
        "fields": {
            "winery": {"key_tokens": [winery.lower()], "variants": [], "brands": []},
            "cuvee": {"tokens": [], "variants": []},
            "grape": {"variants": [grape]},
            "sugar": {"class": sugar},
            "year": {"value": year},
            "serial": {"tokens": [], "keywords": []},
        },
        "visual_mates": list(mates),
        "expected_tokens": tokens,
    }


GTOK = {
    "wine-a": _record("wine-a", year=2023, mates=("wine-b",)),
    "wine-b": _record("wine-b", grape="каберне", year=2022, mates=("wine-a",)),
    "wine-c": _record("wine-c", winery="Массандра", grape="мускат", sugar="sladkoe"),
}


def test_norm_folds_case_yo_diacritics_and_punctuation():
    assert norm("Château  ЁЛКА-Брют") == "chateau елка брют"
    assert norm("Фантом 30/70!") == "фантом 30/70"
    assert norm("Мускат Белый Й") == "мускат белый й"
    assert norm(None) == ""


def test_lev_counts_unit_edits():
    assert lev("брют", "брут") == 1
    assert lev("", "abc") == 3
    assert lev("same", "same") == 0


def test_best_cer_ignores_line_order():
    assert best_cer("Пино Нуар", "2025\nПИНО НУАР\nТабия") == 0.0


def test_best_cer_merged_words_empty_and_cap():
    assert best_cer("пино нуар", "ПИНОНУАР") == pytest.approx(1 / 9)
    assert best_cer("", "что угодно") is None
    assert best_cer("брют", "") == 1.0
    assert best_cer("abc", "zzzzzzzzzz") == 1.0


def test_ooc_reject_rate_counts_only_out_of_catalog():
    rows = [("q1", NONE_SLUG), ("q2", NONE_SLUG), ("q3", "wine-a"), ("q4", NONE_SLUG)]
    preds = {"q1": {"text_top5": []}, "q2": {"text_top5": ["wine-b"]}, "q3": {"text_top5": []}}
    # q4 без предсказания — ответа нет, это отказ
    assert ooc_reject_rate(rows, preds) == pytest.approx(0.667)
    assert ooc_reject_rate([("q3", "wine-a")], preds) is None


def test_text_topk_skips_out_of_catalog():
    rows = [("q1", "wine-a"), ("q2", "wine-b"), ("q3", NONE_SLUG)]
    preds = {
        "q1": {"text_top5": ["wine-a", "wine-b"]},
        "q2": {"text_top5": ["wine-a", "wine-b"]},
        "q3": {"text_top5": ["wine-c"]},
    }
    assert text_topk(rows, preds) == {"top1": 0.5, "top5": 1.0, "n": 2}


def test_twin_resolved_top1_and_answered_by_mate():
    rows = [("q1", "wine-a"), ("q2", "wine-b"), ("q3", "wine-c")]
    preds = {"q1": {"text_top5": ["wine-a"]}, "q2": {"text_top5": ["wine-a"]}}
    result = twin_resolved_top1(rows, preds, GTOK)
    assert result == {
        "twin_queries": 2,
        "twin_resolved_top1": 0.5,
        "twin_answered_by_mate": 0.5,
    }


def test_answered_by_frame_neighbor_includes_out_of_catalog():
    rows = [("q1", NONE_SLUG), ("q2", "wine-a"), ("q3", "wine-c")]
    ann = {
        "q1": {"neighbors": [{"slug": "wine-c"}, {"slug": None}]},
        "q2": {"neighbors": [{"slug": "wine-b"}]},
    }
    preds = {"q1": {"text_top5": ["wine-c"]}, "q2": {"text_top5": ["wine-a"]}}
    assert answered_by_frame_neighbor(rows, preds, ann) == {
        "neighbor_queries": 2,
        "answered_by_frame_neighbor": 0.5,
    }


def test_percentiles_nearest_rank():
    values = [400, 100, 1000, 300, 200]
    assert percentile(values, 0.5) == 300
    assert percentile(values, 0.95) == 1000
    assert latency_summary([]) == {"p50": None, "p95": None, "n": 0}


def test_field_metrics_correct_wrong_abstain():
    rows = [("q1", "wine-a"), ("q2", "wine-a"), ("q3", "wine-a")]
    preds = {
        "q1": {"fields": {"winery": "ФАНАГОРИЯ", "sugar": "suhoe", "year": 2023}},
        "q2": {"fields": {"winery": "Фанагорея", "sugar": "polusuhoe", "year": "2O23"}},
        "q3": {"fields": {"sugar": ["suhoe", "brut"]}},
    }
    result = field_metrics(rows, preds, GTOK)
    assert result["winery"] == {
        "correct": 2,
        "wrong": 0,
        "abstain": 1,
        "acc": 0.667,
        "wrong_rate": 0.0,
    }
    assert (result["sugar"]["correct"], result["sugar"]["wrong"]) == (1, 2)
    assert (result["year"]["correct"], result["year"]["wrong"], result["year"]["abstain"]) == (
        1,
        1,
        1,
    )
    assert "serial" not in result  # у эталона нет серии — поле не оценивается


def test_expected_label_tokens_prefers_transcription():
    ann = {
        "front_label_text": [
            {"field": "winery", "text": "МАССАНДРА"},
            {"field": "year", "text": ""},
        ]
    }
    assert expected_label_tokens(ann, GTOK["wine-c"]) == (
        "front_label_text",
        [("winery", "МАССАНДРА")],
    )
    source, tokens = expected_label_tokens({}, GTOK["wine-a"])
    assert source == "expected_tokens"
    assert ("grape", "merlot") not in tokens and ("sugar", "сухое") not in tokens
    assert expected_label_tokens(None, None) == ("none", [])


def test_token_cer_table_has_key_field_aggregate():
    table = token_cer_table([([("winery", "табия"), ("sugar", "полусухое")], "ТАБИЯ ПОЛУСУХОИ")])
    assert table["winery"]["mean"] == 0.0
    assert table["key_fields"]["n"] == 1
    assert table["sugar"]["share_le_0.2"] == 1.0


def test_cv_vs_cv_ocr_placeholder():
    rows = [("q1", "wine-a"), ("q2", NONE_SLUG)]
    assert cv_vs_cv_ocr(rows, {"q1": {"text_top5": ["wine-a"]}}) is None
    preds = {
        "q1": {"cv_top5": ["wine-b", "wine-a"], "final_top5": ["wine-a"]},
        "q2": {"cv_top5": ["wine-c"], "final_top5": []},
    }
    result = cv_vs_cv_ocr(rows, preds)
    assert result["cv_top1"] == 0.0 and result["cv_ocr_top1"] == 1.0
    assert result["delta_top1"] == 1.0
    assert result["cv_ocr_ooc_reject_rate"] == 1.0


def test_selftest_predictions_are_deterministic_and_bounded():
    rows = [("s1", "wine-a"), ("s2", "wine-b"), ("s3", "wine-c"), ("s4", NONE_SLUG)]
    ann = {"s1": {"neighbors": [{"slug": "wine-b"}]}, "s4": {"neighbors": [{"slug": "wine-c"}]}}
    first = evaluate(rows, ann, synth_predictions(rows, ann, GTOK), GTOK)
    second = evaluate(rows, ann, synth_predictions(rows, ann, GTOK), GTOK)
    assert first == second
    assert (first["queries"], first["in_catalog"], first["out_of_catalog"]) == (4, 3, 1)
    for key in ("text_only_top1", "twin_resolved_top1", "answered_by_frame_neighbor"):
        assert 0.0 <= first[key] <= 1.0
    assert first["latency_ms"]["n"] == 4


def _write_eval_files(tmp_path):
    gt = tmp_path / "gt.tsv"
    gt.write_text("query_id\tslug\tin_catalog\nq1\twine-a\t1\nq2\t__none__\t0\n", encoding="utf-8")
    gtok = tmp_path / "gt_tokens.jsonl"
    gtok.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in GTOK.values()) + "\n", encoding="utf-8"
    )
    return gt, gtok


def test_cli_selftest_writes_json(tmp_path):
    gt, gtok = _write_eval_files(tmp_path)
    out = tmp_path / "res" / "selftest.json"
    assert (
        metrics.main(["--gt", str(gt), "--gt-tokens", str(gtok), "--selftest", "--out", str(out)])
        == 0
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["queries"] == 2 and "text_only_top1" in result


def test_cli_scores_prediction_file(tmp_path):
    gt, gtok = _write_eval_files(tmp_path)
    pred = tmp_path / "pred.jsonl"
    rows = [
        {
            "query_id": "q1",
            "raw_text": "ФАНАГОРИЯ\nМЕРЛО 2023",
            "text_top5": ["wine-a"],
            "latency_ms": 900,
        },
        {"query_id": "q2", "raw_text": "", "text_top5": [], "latency_ms": 700},
    ]
    pred.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = tmp_path / "m.json"
    assert (
        metrics.main(
            ["--gt", str(gt), "--gt-tokens", str(gtok), "--pred", str(pred), "--out", str(out)]
        )
        == 0
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["text_only_top1"] == 1.0
    assert result["ooc_reject_rate"] == 1.0
    assert result["key_token_cer"]["key_fields"]["mean"] == 0.0
    assert result["latency_ms"] == {"p50": 700, "p95": 900, "n": 2}  # ближайший ранг
