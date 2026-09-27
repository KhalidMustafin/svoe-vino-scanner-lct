"""Стенд обучения resolve: фолды, сопоставление прогонов, test под замком, обучение на синтетике."""

import json
from pathlib import Path

import numpy as np
import pytest
from learned_synth import (
    SYNTH_RECORDS,
    cv_record,
    label_fields,
    ocr_record,
    synth_queries,
    write_gt,
    write_jsonl,
)

import bench.train_resolve as train
from app.reading.contracts import Evidence, LabelFields
from app.resolve.attrs import CatalogAttrs
from app.resolve.features import (
    DEFAULT_GROUPS,
    FeatureOptions,
    TextRead,
    cv_query,
    feature_names,
    feature_sign,
)
from app.resolve.learned import LogisticRanker, rank_query
from bench.datasets import DatasetError, split_for
from bench.metrics import load_gt_tokens

FAST = ["--folds", "3", "--l2", "0.01,0.1", "--loss", "listwise", "--top-k", "10"]


@pytest.fixture
def gt_path(tmp_path: Path) -> Path:
    return write_gt(tmp_path / "gt_tokens.jsonl")


@pytest.fixture
def gt(gt_path: Path) -> dict:
    return load_gt_tokens(gt_path)


def write_run(
    tmp_path: Path, *, n: int = 48, seed: int = 11, poison_test: bool = False, name: str = "run"
) -> tuple[Path, Path]:
    """Прогон CV и прогон OCR одного набора: две трети запросов dev, треть — test."""
    cv_rows, ocr_rows = [], []
    for i, (cv, ocr, _) in enumerate(synth_queries(n, seed=seed)):
        split = "test" if i % 3 == 2 else "dev"
        cv["meta"]["split"] = split
        if split == "test" and poison_test:
            cv["top"] = "POISON"
            ocr["label_fields"] = "POISON"
        cv_rows.append(cv)
        ocr_rows.append(ocr)
    return (
        write_jsonl(tmp_path / f"{name}_cv.jsonl", cv_rows),
        write_jsonl(tmp_path / f"{name}_ocr.jsonl", ocr_rows),
    )


def run_main(tmp_path: Path, gt_path: Path, cv: Path, ocr: Path, *extra: str) -> tuple[int, Path]:
    out = tmp_path / "out"
    args = ["--cv", str(cv), "--ocr", f"r:synthetic={ocr}", "--gt-tokens", str(gt_path), *FAST]
    code = train.main([*args, "--out", str(out), "--overwrite", *extra])
    return code, out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------------ фолды
def test_group_folds_keep_groups_together_and_balance():
    groups = [f"cluster:{i % 7}" for i in range(70)] + [f"slug:s{i}" for i in range(30)]
    folds = train.group_folds(groups, 5)
    by_group: dict[str, set[int]] = {}
    for group, fold in zip(groups, folds, strict=True):
        by_group.setdefault(group, set()).add(int(fold))
    assert all(len(f) == 1 for f in by_group.values())
    sizes = np.bincount(folds, minlength=5)
    assert sizes.max() - sizes.min() <= 10


def test_group_folds_do_not_depend_on_row_order():
    groups = [f"g{i % 9}" for i in range(45)]
    folds = train.group_folds(groups, 3)
    order = list(reversed(range(45)))
    shuffled = train.group_folds([groups[i] for i in order], 3)
    assert {g: int(f) for g, f in zip(groups, folds, strict=True)} == {
        groups[i]: int(f) for i, f in zip(order, shuffled, strict=True)
    }
    with pytest.raises(ValueError):
        train.group_folds(["a", "b"], 3)


def test_pairs_and_phone_copy_of_a_frame_share_a_fold(tmp_path: Path, gt: dict):
    rows = synth_queries(30, seed=4)
    cv_a = write_jsonl(tmp_path / "a.jsonl", [r[0] for r in rows])
    cv_b = write_jsonl(
        tmp_path / "b.jsonl",
        [{**r[0], "meta": {**r[0]["meta"], "set": "synthetic_phone"}} for r in rows],
    )
    queries, _ = train.load_split([cv_a, cv_b], [], gt, split="dev", top_k=10)
    folds = train.group_folds([q.group for q in queries], 3)
    fold_of: dict[str, set[int]] = {}
    for query, fold in zip(queries, folds, strict=True):
        fold_of.setdefault(query.query_id, set()).add(int(fold))
    assert len(queries) == 60 and all(len(f) == 1 for f in fold_of.values())


# ------------------------------------------------------------------ входные файлы
def test_parse_ocr_spec():
    spec = train.parse_ocr_spec("vlm35:pairs_phone=runs/ocr-pairsphone-vlm35/predictions.jsonl")
    assert (spec.reader, spec.set_name) == ("vlm35", "pairs_phone")
    assert spec.path == Path("runs/ocr-pairsphone-vlm35/predictions.jsonl")
    for bad in ("vlm35=path", "vlm35:pairs", "both:pairs=p", "vl m:pairs=p"):
        with pytest.raises(ValueError):
            train.parse_ocr_spec(bad)
    with pytest.raises(ValueError):
        train.parse_losses("pointwise,hinge")


def test_ocr_is_matched_by_set_and_query_id(tmp_path: Path, gt: dict):
    ranked = [("alfa-muskat", 0.9), ("beta-merlot", 0.8)]
    cv_pairs = write_jsonl(tmp_path / "cv1.jsonl", [cv_record("q1", "alfa-muskat", ranked)])
    cv_phone = write_jsonl(
        tmp_path / "cv2.jsonl",
        [cv_record("q1", "alfa-muskat", ranked, set_name="synthetic_phone")],
    )
    ocr_pairs = write_jsonl(
        tmp_path / "o1.jsonl", [ocr_record("q1", "alfa-muskat", label_fields("Альфа Долина"), "")]
    )
    ocr_phone = write_jsonl(
        tmp_path / "o2.jsonl", [ocr_record("q1", "alfa-muskat", label_fields("Бета Холмы"), "")]
    )
    specs = [
        train.OcrSpec("r", "synthetic", ocr_pairs),
        train.OcrSpec("r", "synthetic_phone", ocr_phone),
    ]
    queries, _ = train.load_split([cv_pairs, cv_phone], specs, gt, split="dev", top_k=10)
    reads = {q.set_name: q.reads["r"].fields.producer[0].value for q in queries}
    assert reads == {"synthetic": "Альфа Долина", "synthetic_phone": "Бета Холмы"}


def test_ocr_with_other_target_is_rejected(tmp_path: Path, gt: dict):
    cv = write_jsonl(
        tmp_path / "cv.jsonl", [cv_record("q1", "alfa-muskat", [("alfa-muskat", 0.9)])]
    )
    ocr = write_jsonl(tmp_path / "o.jsonl", [ocr_record("q1", "beta-merlot", LabelFields(), "")])
    with pytest.raises(DatasetError, match="эталон"):
        train.load_split([cv], [train.OcrSpec("r", "synthetic", ocr)], gt, split="dev", top_k=10)


def test_pairs_split_must_match_cluster_hash(tmp_path: Path, gt: dict):
    slug = "beta-merlot"
    hashed = split_for(slug, "q1")
    wrong = "test" if hashed == "dev" else "dev"
    record = cv_record("q1", slug, [(slug, 0.9)], set_name="pairs", split=wrong)
    path = write_jsonl(tmp_path / "cv.jsonl", [record])
    with pytest.raises(DatasetError, match="cluster_B"):
        train.load_cv(path, gt, splits=frozenset({"dev"}), top_k=10)
    record["meta"].pop("split")
    path = write_jsonl(tmp_path / "cv.jsonl", [record])
    queries, info = train.load_cv(path, gt, splits=frozenset({hashed}), top_k=10)
    assert len(queries) == 1 and queries[0].split == hashed and info["rows_other_splits"] == 0


def test_merge_fields_keeps_first_reader_single_values():
    first = LabelFields(
        producer=[Evidence[str](value="Альфа Долина")], vintage=Evidence[int](value=2023)
    )
    second = LabelFields(
        producer=[Evidence[str](value="Альфа Долина"), Evidence[str](value="Бета Холмы")],
        vintage=Evidence[int](value=2024),
    )
    merged = train.merge_fields([first, None, second])
    assert [e.value for e in merged.producer] == ["Альфа Долина", "Бета Холмы"]
    assert merged.vintage.value == 2023
    assert train.merge_fields([None, None]) is None


# ------------------------------------------------------------------ обучение на синтетике
def test_learned_model_does_not_let_misread_winery_beat_confident_cv(tmp_path: Path, gt: dict):
    attrs = CatalogAttrs.from_records(SYNTH_RECORDS)
    names = feature_names(["r"], DEFAULT_GROUPS)
    parts = {}
    for name, seed in (("train", 1), ("hold", 2)):
        rows = synth_queries(200, seed=seed)
        cv = write_jsonl(tmp_path / f"{name}_cv.jsonl", [r[0] for r in rows])
        ocr = write_jsonl(tmp_path / f"{name}_ocr.jsonl", [r[1] for r in rows])
        queries, _ = train.load_split(
            [cv], [train.OcrSpec("r", "synthetic", ocr)], gt, split="dev", top_k=20
        )
        kinds = {r[0]["query_id"]: r[2] for r in rows}
        design = train.build_design(
            queries, ["r"], attrs, names, top_k=20, options=FeatureOptions()
        )
        parts[name] = (queries, design, kinds)
    queries, design, _ = parts["train"]
    model = train.fit_on(design, list(range(len(queries))), names, 0.01, "listwise")
    queries, design, kinds = parts["hold"]
    scores = train.score_on(model, design, list(range(len(queries))))
    learned = [train.is_correct(design, i, scores[i]) for i in range(len(queries))]
    manual = [train.manual_top5(q, ["r"], attrs, 20)[0] == q.slug for q in queries]
    cv_only = [q.cv.candidates[0].slug == q.slug for q in queries]
    misread = [i for i, q in enumerate(queries) if kinds[q.query_id] == "confident:misread"]
    assert len(misread) >= 20
    # Уверенная картинка против ошибочно прочитанной винодельни: модель держит CV-top1,
    # ручные веса (−0,45 за спор и +0,30 за совпадение) его переворачивают.
    assert np.mean([learned[i] for i in misread]) >= 0.95
    assert np.mean([manual[i] for i in misread]) <= 0.2
    assert np.mean(learned) > max(np.mean(manual), np.mean(cv_only)) + 0.1
    # Сервисный вызов на одном запросе даёт тот же порядок, что стенд.
    for i in misread[:5]:
        ranking = rank_query(model, queries[i].cv, queries[i].reads, attrs, top_k=20)
        expected = [design.slugs[i][j] for j in train.order_of(scores[i])]
        assert list(ranking.slugs) == expected and 0 < ranking.p_top1 <= 1


# ------------------------------------------------------------------ CLI и test под замком
def test_main_without_final_test_never_reads_test_rows(tmp_path, gt_path, monkeypatch):
    cv, ocr = write_run(tmp_path, poison_test=True)
    calls: list[frozenset[str]] = []
    real_load_cv = train.load_cv

    def spy(path, gt_tokens, *, splits, top_k):
        calls.append(splits)
        return real_load_cv(path, gt_tokens, splits=splits, top_k=top_k)

    monkeypatch.setattr(train, "load_cv", spy)
    code, out = run_main(tmp_path, gt_path, cv, ocr)
    assert code == 0
    assert calls and all(splits == frozenset({"dev"}) for splits in calls)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_used"] is False and "test" not in metrics
    rows = read_jsonl(out / "oof_predictions.jsonl")
    assert rows and {row["split"] for row in rows} == {"dev"}
    assert not (out / "test_predictions.jsonl").exists()
    # Отравленные строки test тронуты только с флагом — и тогда прогон падает на них.
    code, _ = run_main(tmp_path, gt_path, cv, ocr, "--final-test")
    assert code == 2


def test_main_writes_reports_models_and_final_test(tmp_path, gt_path):
    cv, ocr = write_run(tmp_path)
    code, out = run_main(tmp_path, gt_path, cv, ocr, "--final-test")
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_used"] is True and "TEST" in metrics["test_warning"]
    assert set(metrics["variants"]) == {"r", "none"}
    report = metrics["variants"]["r"]
    block = report["oof"]["by_set"]["synthetic"]
    assert {"cv_only", "manual", "learned", "vs_cv", "calibration", "ceiling_k"} <= set(block)
    assert len(block["calibration"]["bins"]) == 10
    assert report["selection"]["grid"] and report["model"]["weights"]
    test_block = metrics["test"]["variants"]["r"]["by_set"]["synthetic"]
    assert test_block["n"] == 16 and test_block["learned"]["top1"] is not None
    assert {row["split"] for row in read_jsonl(out / "test_predictions.jsonl")} == {"test"}
    model = LogisticRanker.load(out / "models" / "r.json")
    assert model.meta["split"] == "dev" and model.meta["variant"] == "r"
    assert "## dev" in (out / "tables.md").read_text(encoding="utf-8")


def test_main_is_deterministic(tmp_path, gt_path):
    cv, ocr = write_run(tmp_path, seed=21)
    args = ["--cv", str(cv), "--ocr", f"r:synthetic={ocr}", "--gt-tokens", str(gt_path)]
    args += ["--folds", "3", "--l2", "0.01,0.1", "--loss", "pointwise,listwise", "--top-k", "10"]
    outs = [tmp_path / "out0", tmp_path / "out1"]
    for out in outs:
        assert train.main([*args, "--out", str(out)]) == 0
    first, second = (json.loads((o / "metrics.json").read_text(encoding="utf-8")) for o in outs)
    for variant, report in first["variants"].items():
        other = second["variants"][variant]
        # Путь модели свой у каждого прогона; всё остальное, включая веса, совпадает.
        assert {k: v for k, v in report.items() if k != "model"} == {
            k: v for k, v in other.items() if k != "model"
        }
        assert report["model"]["weights"] == other["model"]["weights"]
    oof = [(o / "oof_predictions.jsonl").read_text(encoding="utf-8") for o in outs]
    assert oof[0] == oof[1]
    a, b = (LogisticRanker.load(o / "models" / "r.json") for o in outs)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    assert a.temperature_ == b.temperature_


def test_main_refuses_existing_run_and_bad_variant(tmp_path, gt_path):
    cv, ocr = write_run(tmp_path)
    code, out = run_main(tmp_path, gt_path, cv, ocr)
    assert code == 0
    base = ["--cv", str(cv), "--ocr", f"r:synthetic={ocr}", "--gt-tokens", str(gt_path), *FAST]
    assert train.main([*base, "--out", str(out)]) == 2
    assert train.main([*base, "--variants", "both", "--out", str(tmp_path / "o2")]) == 2
    assert train.main([*base, "--groups", "winery", "--out", str(tmp_path / "o3")]) == 2


# ------------------------------------------------------------------ сервисный вызов
def test_rank_query_without_the_models_reader_falls_back_to_cv(tmp_path: Path, gt: dict):
    """Сбой или таймаут OCR: читателя модели нет в `reads` — это пустое чтение, как при
    обучении, а не `KeyError`."""
    attrs = CatalogAttrs.from_records(SYNTH_RECORDS)
    rows = synth_queries(60, seed=3)
    cv = write_jsonl(tmp_path / "cv.jsonl", [r[0] for r in rows])
    ocr = write_jsonl(tmp_path / "ocr.jsonl", [r[1] for r in rows])
    queries, _ = train.load_split(
        [cv], [train.OcrSpec("r", "synthetic", ocr)], gt, split="dev", top_k=10
    )
    names = feature_names(["r"], DEFAULT_GROUPS)
    design = train.build_design(queries, ["r"], attrs, names, top_k=10, options=FeatureOptions())
    model = train.fit_on(design, list(range(len(queries))), names, 0.1, "listwise")
    record = rows[0][0]
    empty = rank_query(model, record, {"r": TextRead()}, attrs, top_k=10)
    assert rank_query(model, record, {}, attrs, top_k=10) == empty
    assert rank_query(model, record, {"r": None}, attrs, top_k=10) == empty
    other = {"other": TextRead(label_fields("Бета Холмы"))}  # чужой читатель модели не нужен
    assert rank_query(model, record, other, attrs, top_k=10) == empty


# ------------------------------------------------------------------ OCR того же кадра
def frame_runs(tmp_path: Path, *, swap: bool) -> tuple[list[Path], list[train.OcrSpec]]:
    """Набор и его порченая копия с путями кадров, как у настоящих прогонов.

    CV портит исходный кадр на лету (`meta.phone_seed`, путь — исходный), OCR читает
    сохранённую порченую копию. `swap` путает прогоны OCR двух наборов.
    """
    ranked = [("alfa-muskat", 0.9), ("beta-merlot", 0.8)]
    studio = "C:/data/raw/pairs/roskachestvo/q1.webp"
    studio_cv = {**cv_record("q1", "alfa-muskat", ranked, set_name="s"), "image_path": studio}
    phone_cv = {**cv_record("q1", "alfa-muskat", ranked, set_name="s_phone"), "image_path": studio}
    phone_cv["meta"]["phone_seed"] = 0
    fields = label_fields("Альфа Долина")
    studio_ocr = {**ocr_record("q1", "alfa-muskat", fields, ""), "image_path": studio}
    phone_ocr = {
        **ocr_record("q1", "alfa-muskat", fields, ""),
        "image_path": "runs/pairs-phone-images/q1.jpg",
    }
    cvs = [
        write_jsonl(tmp_path / "cv_s.jsonl", [studio_cv]),
        write_jsonl(tmp_path / "cv_p.jsonl", [phone_cv]),
    ]
    ocr_s = write_jsonl(tmp_path / "ocr_s.jsonl", [studio_ocr])
    ocr_p = write_jsonl(tmp_path / "ocr_p.jsonl", [phone_ocr])
    if swap:
        ocr_s, ocr_p = ocr_p, ocr_s
    return cvs, [train.OcrSpec("r", "s", ocr_s), train.OcrSpec("r", "s_phone", ocr_p)]


def test_swapped_ocr_runs_of_a_set_and_its_phone_copy_are_rejected(tmp_path: Path, gt: dict):
    cvs, specs = frame_runs(tmp_path, swap=False)
    queries, _ = train.load_split(cvs, specs, gt, split="dev", top_k=10)
    assert {q.set_name for q in queries if "r" in q.reads} == {"s", "s_phone"}
    cvs, specs = frame_runs(tmp_path, swap=True)
    with pytest.raises(DatasetError, match="перепутаны"):
        train.load_split(cvs, specs, gt, split="dev", top_k=10)


def test_ocr_of_the_other_half_is_rejected(tmp_path: Path, gt: dict):
    cv = write_jsonl(
        tmp_path / "cv.jsonl", [cv_record("q1", "alfa-muskat", [("alfa-muskat", 0.9)])]
    )
    ocr = write_jsonl(
        tmp_path / "o.jsonl",
        [{**ocr_record("q1", "alfa-muskat", LabelFields(), ""), "split": "test"}],
    )
    with pytest.raises(DatasetError, match="split=test"):
        train.load_split([cv], [train.OcrSpec("r", "synthetic", ocr)], gt, split="dev", top_k=10)


def test_same_image_survives_a_moved_dataset():
    assert train.same_image("C:/a/roskachestvo/x.webp", "D:/b/roskachestvo/x.webp")
    assert not train.same_image("C:/a/roskachestvo/x.webp", "runs/pairs-phone-images/x.jpg")


# ------------------------------------------------------------------ ничьи CV
def tie_eval(qid: str, ranked, slug: str, learned: str) -> train.QueryEval:
    query = train.Query(
        set_name="s",
        query_id=qid,
        slug=slug,
        split="dev",
        group=qid,
        cv=cv_query([{"slug": s, "score": v} for s, v in ranked]),
    )
    cv_top = [s for s, _ in ranked]
    return train.QueryEval(query, cv_top, cv_top, [learned], True, 0.5)


def test_exact_cv_ties_are_counted_apart_and_credited_one_in_k():
    evals = [
        # CV «угадал» порядком файла, обученный выбрал второго из равных — жребий.
        tie_eval("q1", [("a", 0.9), ("b", 0.9), ("c", 0.5)], "a", "b"),
        tie_eval("q2", [("a", 0.9), ("b", 0.8)], "b", "b"),  # честное исправление
    ]
    assert [e.cv_tie for e in evals] == [2, 1]
    counts = train.fix_break(evals, "learned")
    assert (counts["fixed"], counts["broke"]) == (1, 1)
    assert (counts["fixed_in_ties"], counts["broke_in_ties"]) == (0, 1)
    stats = train.column_stats(evals, "cv_only")
    assert stats["top1"] == 0.5 and stats["top1_tie_fair"] == 0.25
    assert train.fix_break_cell(counts) == "+1/−1 (н. +0/−1)"
    assert train.summarize(evals)["all"]["cv_ties"] == 1
    rows = list(train.prediction_rows("r", evals))
    assert [row["cv_tie_k"] for row in rows] == [2, 1]


# ------------------------------------------------------------------ двойники по обе стороны
def test_cross_split_twins_are_named_by_query_id_without_year():
    def query(qid: str, group: str, set_name: str = "pairs") -> train.Query:
        return train.Query(set_name, qid, "slug", "?", group, cv_query([]))

    dev = [
        query("novyy-svet-pino-nuar-2020", "cluster:2"),
        query("abrau-blan-2020", "cluster:113"),
        query("novyy-svet-pino-nuar-2020", "cluster:2", "pairs_phone"),
    ]
    test = [
        query("novyy-svet-pino-nuar-2018", "cluster:87"),
        query("abrau-blan-2021", "cluster:113"),  # одна группа — разбиение их не делит
        query("rkatsiteli", "cluster:5"),
    ]
    twins = train.cross_split_twins(dev, test)
    assert [(t["set"], t["dev"], t["test"]) for t in twins] == [
        ("pairs", "novyy-svet-pino-nuar-2020", "novyy-svet-pino-nuar-2018")
    ]


# ------------------------------------------------------------------ знаки и контрасты
def test_saved_model_keeps_signs_and_reports_group_contrasts(tmp_path, gt_path):
    cv, ocr = write_run(tmp_path)
    code, out = run_main(tmp_path, gt_path, cv, ocr, "--final-test")
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["signed"] is True and "cross_split_twins" in metrics["test"]
    model = LogisticRanker.load(out / "models" / "r.json")
    for name, sign, weight in zip(model.feature_names, model.signs, model.coef_, strict=True):
        assert sign == feature_sign(name) and sign * weight >= 0
    report = metrics["variants"]["r"]["model"]
    raw = {w["name"]: w["raw_weight"] for w in report["weights"]}
    assert report["contrasts"]["r.winery"]["match"] == pytest.approx(
        raw["r.winery_match"] + raw["r.winery_match_conf"], abs=2e-4
    )
    assert "Контрасты групп" in (out / "tables.md").read_text(encoding="utf-8")
    code, out = run_main(tmp_path, gt_path, cv, ocr, "--free-signs")
    assert code == 0
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["run"]["signed"] is False


def test_model_meta_names_the_readers_whose_readings_it_saw(tmp_path, gt_path):
    """Модель помнит, какой VLM и с какими параметрами читали её обучение.

    Строка читателя записи OCR — `vlm@модель|params_hash|<sha1 кадра>|full|1024`: в `meta`
    уходят первые два звена, по ним сервис сверяет свой читатель при старте.
    """
    cv, ocr = write_run(tmp_path)
    rows = read_jsonl(ocr)
    for i, row in enumerate(rows):
        row["reader"] = f"vlm@qwen3.5:4b|f3a017317f04|{i:040d}|full|1024"
        if i == 0:  # запись с несколькими читателями — берутся все
            row["readers"] = [
                {"reader": f"vlm@qwen3.5:4b|f3a017317f04|{i:040d}|full|1024"},
                {"reader": f"rapid@v5|0123abcd|{i:040d}|full|1024"},
            ]
    write_jsonl(ocr, rows)
    code, out = run_main(tmp_path, gt_path, cv, ocr)
    assert code == 0
    model = LogisticRanker.load(out / "models" / "r.json")
    assert model.meta["reader_keys"] == {"r": ["rapid@v5|0123abcd", "vlm@qwen3.5:4b|f3a017317f04"]}


def test_ocr_rows_without_reader_leave_no_reader_keys(tmp_path, gt_path):
    cv, ocr = write_run(tmp_path)
    code, out = run_main(tmp_path, gt_path, cv, ocr)
    assert code == 0
    assert LogisticRanker.load(out / "models" / "r.json").meta["reader_keys"] == {}
