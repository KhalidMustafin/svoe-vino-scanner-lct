"""Стенд поиска с текстом этикетки: `--with-ocr`, колонка «CV+OCR» и разница в пунктах.

Модель зрения здесь подставная, а индекс либо крошечный, либо заглушка: проверяется
проводка стенда — что поля прогона OCR доходят до resolve и что метрики считаются дважды.
"""

import json

import numpy as np
import pytest
from catalog import TOY_RECORDS, toy_attrs
from fakes import PixelEmbedder

from app.features.contracts import Candidate, IndexMeta, VisualResult
from app.features.index import VisualIndex
from app.reading.contracts import Color, LabelFields, SugarClass
from app.resolve.rerank import DEFAULT_CONFIG
from bench.datasets import DatasetError
from bench.queries import QueryItem
from bench.retrieval import (
    CV_KEY,
    FINAL_KEY,
    Pipeline,
    column_metrics,
    compute_metrics,
    cv_vs_cv_ocr,
    delta_pp,
    fields_from_flat,
    fields_of_record,
    load_read_fields,
    main,
    search_item,
)

TWINS = ("dolina-aligote-2023", "dolina-aligote-2024")


# ------------------------------------------------------------------ поля прогона OCR
def test_fields_from_flat():
    fields = fields_from_flat(
        {
            "winery": "Тестовая Долина",
            "cuvee": "Баррель",
            "grape": "aligote",
            "sugar": ["suhoe"],
            "year": 2024,
            "serial": ["XXIV"],
            "abv": 12.5,
            "color": "Белое",
        }
    )
    assert fields.producer[0].value == "Тестовая Долина" and fields.cuvee[0].value == "Баррель"
    assert fields.sugar[0].value is SugarClass.DRY and fields.color.value is Color.WHITE
    assert fields.vintage.value == 2024 and fields.abv.value == 12.5
    assert fields.serial[0].value == "XXIV"
    # У плоской записи нет уверенности чтения: правило отказа по ней не сработает.
    assert fields.producer[0].conf is None


def test_fields_from_flat_ignores_unknown_codes_and_empties():
    fields = fields_from_flat({"winery": "", "sugar": ["нет"], "color": "Синее", "serial": []})
    assert fields.producer == [] and fields.sugar == [] and fields.color is None
    assert fields_from_flat({}) == LabelFields()


def test_fields_of_record_prefers_full_fields():
    full = LabelFields.model_validate(
        {"producer": [{"value": "Другая Винодельня", "conf": 1.0}], "vintage": None}
    )
    record = {
        "label_fields": full.model_dump(mode="json"),
        "fields": {"winery": "Тестовая Долина"},
    }
    assert fields_of_record(record).producer[0].value == "Другая Винодельня"
    assert fields_of_record({"fields": {"winery": "Тестовая Долина"}}).producer[0].value == (
        "Тестовая Долина"
    )
    assert fields_of_record({"query_id": "q1"}) is None


def test_load_read_fields_skips_broken_frames(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"query_id": "q1", "status": "ok", "fields": {"winery": "Тестовая Долина"}},
        {"query_id": "q2", "status": "error", "fields": {"winery": "Другая Винодельня"}},
        {"query_id": "q3", "status": "ok"},  # полей нет — кадр пропускается
        {"status": "ok", "fields": {"winery": "Третья Марка"}},  # без query_id
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n", encoding="utf-8"
    )
    reads = load_read_fields(path)
    assert set(reads) == {"q1"} and reads["q1"].producer[0].value == "Тестовая Долина"


def test_load_read_fields_needs_a_file(tmp_path):
    with pytest.raises(DatasetError, match="нет прогона OCR"):
        load_read_fields(tmp_path / "нет.jsonl")
    broken = tmp_path / "predictions.jsonl"
    broken.write_text("{не json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r":1: не JSON"):
        load_read_fields(broken)


# ------------------------------------------------------------------ один кадр
class StubIndex:
    """Индекс-заглушка: отдаёт заранее заданную выдачу и запоминает виды запроса."""

    def __init__(self, pairs: list[tuple[str, float]]) -> None:
        self.meta = IndexMeta(model="fake/pixel", dim=4, n_slugs=len(pairs), n_vectors=len(pairs))
        self.pairs = pairs
        self.views: dict = {}

    def search(self, views, embedder, *, top_k=20, per_slug="max") -> VisualResult:
        self.views = views
        candidates = [
            Candidate(slug=slug, score=score, view="bottle", rank=rank)
            for rank, (slug, score) in enumerate(self.pairs[:top_k], start=1)
        ]
        margin = self.pairs[0][1] - self.pairs[1][1] if len(self.pairs) > 1 else 0.0
        return VisualResult(candidates=candidates, margin=margin, model="fake/pixel")


@pytest.fixture
def frame(tmp_path):
    """Маленький настоящий кадр: стенд его действительно декодирует и режет на виды."""
    from PIL import Image

    path = tmp_path / "q1.webp"  # имя как в наборе pairs; формат определяется по сигнатуре
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:, :, 0] = np.arange(64, dtype=np.uint8)
    Image.fromarray(array).save(path, "PNG")
    return path


def test_search_item_adds_resolve_columns(frame):
    item = QueryItem(query_id="q1", image_path=frame, slug=TWINS[1])
    pipeline = Pipeline(
        index=StubIndex([(TWINS[0], 0.82), ("dolina-merlot", 0.60)]),
        embedder=PixelEmbedder(dim=4),
        attrs=toy_attrs(),
        cfg=DEFAULT_CONFIG,
    )
    fields = fields_from_flat({"winery": "Тестовая Долина", "year": 2024})
    record = search_item(item, pipeline, fields)
    assert record["status"] == "ok" and record["read_fields"] is True
    assert record[CV_KEY] == [TWINS[0], "dolina-merlot"]
    # Двойник пришёл из серии и выиграл по году с этикетки.
    assert record["slug_pred"] == TWINS[1] and record[FINAL_KEY][0] == TWINS[1]
    assert record["outcome"] == "matched" and record["evidence"]["features"]["year"]
    assert record["timings_ms"]["resolve"] >= 0


def test_search_item_without_attrs_stays_visual(frame):
    item = QueryItem(query_id="q1", image_path=frame, slug=TWINS[0])
    pipeline = Pipeline(index=StubIndex([(TWINS[0], 0.8)]), embedder=PixelEmbedder(dim=4))
    record = search_item(item, pipeline)
    assert FINAL_KEY not in record and "outcome" not in record
    assert record[CV_KEY] == [TWINS[0]] and record["hit_rank"] == 1


def test_search_item_without_read_fields_still_resolves(frame):
    # Кадра нет в прогоне OCR: resolve получает пустые поля и опирается на один CV.
    item = QueryItem(query_id="q1", image_path=frame, slug=TWINS[0])
    pipeline = Pipeline(
        index=StubIndex([(TWINS[0], 0.82), ("dolina-merlot", 0.60)]),
        embedder=PixelEmbedder(dim=4),
        attrs=toy_attrs(),
    )
    record = search_item(item, pipeline)
    assert record["read_fields"] is False
    assert record[FINAL_KEY][0] == TWINS[0] and record["evidence"]["text_empty"] is True


# ------------------------------------------------------------------ две колонки метрик
def items_and_predictions():
    """Четыре запроса: двойник, чужая бутылка, кадр вне каталога и отказ resolve."""
    items = [
        QueryItem(query_id="q1", image_path="a.webp", slug=TWINS[1]),
        QueryItem(query_id="q2", image_path="b.webp", slug="dolina-merlot"),
        QueryItem(query_id="q3", image_path="c.webp", slug=None),
    ]
    predictions = [
        {
            "query_id": "q1",
            "status": "ok",
            "top": [{"slug": TWINS[0], "score": 0.82}, {"slug": TWINS[1], "score": 0.81}],
            CV_KEY: [TWINS[0], TWINS[1]],
            FINAL_KEY: [TWINS[1], TWINS[0]],  # текст переставил двойников
            "outcome": "matched",
            "resolve_margin": 0.3,
            "read_fields": True,
        },
        {
            "query_id": "q2",
            "status": "ok",
            "top": [{"slug": "dolina-merlot", "score": 0.77}],
            CV_KEY: ["dolina-merlot"],
            FINAL_KEY: ["dolina-merlot"],
            "outcome": "matched",
            "resolve_margin": 0.1,
            "read_fields": True,
        },
        {
            "query_id": "q3",
            "status": "ok",
            "top": [{"slug": "drugaya-kaberne", "score": 0.44}],
            CV_KEY: ["drugaya-kaberne"],
            FINAL_KEY: [],  # отказ: вина нет в каталоге
            "outcome": "out_of_catalog",
            "resolve_margin": 0.0,
            "read_fields": True,
        },
    ]
    return items, predictions


def test_cv_vs_cv_ocr_counts_both_columns():
    items, predictions = items_and_predictions()
    by_qid = {p["query_id"]: p for p in predictions}
    gt_tokens = {record["slug"]: record for record in TOY_RECORDS}
    block = cv_vs_cv_ocr(
        items, by_qid, gt_tokens, source="runs/ocr/predictions.jsonl", series_pool=True
    )
    assert block["cv_only"]["by_card"]["top1"] == 0.5  # верна только «Мерло»
    assert block["cv_ocr"]["by_card"]["top1"] == 1.0  # текст поднял верного двойника
    assert block["delta_pp"]["by_card"]["top1"] == 50.0  # разница в пунктах
    # Макет у двойников общий: по нему CV не ошибался и разницы нет.
    group = ("by_visual_group", "by_winery_line")
    assert all(block["cv_only"][k]["top1"] == block["cv_ocr"][k]["top1"] == 1.0 for k in group)
    # Строка по линейке помечена: пул resolve наполняется тем же ключом cluster_B.
    assert block["series_pool"] is True and "тавтологична" in block["by_winery_line_note"]
    assert block["cv_only"]["ooc_reject_rate"] == 0.0
    assert block["cv_ocr"]["ooc_reject_rate"] == 1.0
    assert block["outcomes"] == {"matched": 2, "out_of_catalog": 1}
    assert block["with_read_fields"] == 3 and block["read_coverage"] == 1.0
    assert block["with_ocr"].endswith("predictions.jsonl")
    assert "не подобраны на данных" in block["note"]


def test_a_broken_frame_is_not_a_refusal():
    """Сбой разжатия тоже даёт пустую выдачу, но отказом канала он не является."""
    items = [
        QueryItem(query_id="q1", image_path="a.webp", slug=None),
        QueryItem(query_id="q2", image_path="b.webp", slug=None),
    ]
    by_qid = {
        "q1": {"query_id": "q1", "status": "error", CV_KEY: [], FINAL_KEY: []},
        "q2": {
            "query_id": "q2",
            "status": "ok",
            "top": [{"slug": "dolina-merlot", "score": 0.4}],
            CV_KEY: ["dolina-merlot"],
            FINAL_KEY: ["dolina-merlot"],
        },
    }
    gt_tokens = {record["slug"]: record for record in TOY_RECORDS}
    column = column_metrics(items, by_qid, gt_tokens, key=CV_KEY)
    # Один кадр вне каталога дошёл до конца и ответил: отказов ноль, а не половина.
    assert column["ooc_scored"] == 1 and column["ooc_reject_rate"] == 0.0


def test_delta_pp_skips_counts_and_none():
    before = {"by_card": {"top1": 0.5, "n": 2}, "ooc_reject_rate": None}
    after = {"by_card": {"top1": 0.75, "n": 2}, "ooc_reject_rate": 1.0}
    assert delta_pp(before, after) == {"by_card": {"top1": 25.0}}


def test_compute_metrics_adds_the_block_only_with_resolve():
    items, predictions = items_and_predictions()
    gt_tokens = {record["slug"]: record for record in TOY_RECORDS}
    metrics = compute_metrics(items, predictions, gt_tokens, top_k=20, with_ocr="p.jsonl")
    assert metrics["cv_vs_cv_ocr"]["delta_pp"]["by_card"]["top1"] == 50.0

    visual_only = [{k: v for k, v in p.items() if k != FINAL_KEY} for p in predictions]
    assert compute_metrics(items, visual_only, gt_tokens, top_k=20)["cv_vs_cv_ocr"] is None


# ------------------------------------------------------------------ прогон целиком
def tiny_index(path, slugs):
    """Индекс подставной модели: строка на slug, вектора хватает, чтобы стенд собрался."""
    vectors = np.eye(4, dtype=np.float32)[[i % 4 for i in range(len(slugs))]]
    meta = IndexMeta(
        model="fake/pixel", dim=4, views=["bottle"], n_slugs=len(slugs), n_vectors=len(slugs)
    )
    VisualIndex(list(slugs), ["bottle"] * len(slugs), vectors, meta).save(path)
    return path


def ocr_setup(tmp_path, read_ids=("w1", "w2")):
    """Пары, разметка каталога, прогон OCR и индекс на диске — всё, что нужно `main`."""
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    pairs = [{"wine_id": "w1", "portal_slug": TWINS[0]}, {"wine_id": "w2", "portal_slug": TWINS[1]}]
    for number, pair in enumerate(pairs):
        array = np.full((48, 48, 3), 40 * (number + 1), dtype=np.uint8)
        array[0, :4, 0] = [10, 200, 30, 40]
        Image.fromarray(array).save(images / f"{pair['wine_id']}.webp", "PNG")
    pairs_json = tmp_path / "benchmark.json"
    pairs_json.write_text(json.dumps(pairs), encoding="utf-8")

    gt_tokens = tmp_path / "gt_tokens.jsonl"
    gt_tokens.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in TOY_RECORDS) + "\n",
        encoding="utf-8",
    )
    fields = {"w1": {"winery": "Тестовая Долина"}, "w2": {"year": 2024}}
    reads = tmp_path / "ocr_predictions.jsonl"
    reads.write_text(
        "\n".join(
            json.dumps(
                {"query_id": qid, "status": "ok", "fields": fields.get(qid, {"year": 2024})},
                ensure_ascii=False,
            )
            for qid in read_ids
        )
        + "\n",
        encoding="utf-8",
    )
    index = tiny_index(tmp_path / "visual.npz", [record["slug"] for record in TOY_RECORDS])
    return {
        "pairs": pairs_json,
        "images": images,
        "gt": gt_tokens,
        "reads": reads,
        "index": index,
        "out": tmp_path / "run",
    }


def ocr_argv(files, *extra):
    return [
        "--queries",
        "pairs",
        "--pairs-json",
        str(files["pairs"]),
        "--pairs-images",
        str(files["images"]),
        "--index",
        str(files["index"]),
        "--gt-tokens",
        str(files["gt"]),
        "--with-ocr",
        str(files["reads"]),
        "--out",
        str(files["out"]),
        "--split",
        "all",  # две игрушечные пары меряются целиком, а не половиной
        "--unseal-test",
        *extra,
    ]


def test_main_runs_with_ocr(tmp_path):
    files = ocr_setup(tmp_path)
    out = files["out"]
    code = main(ocr_argv(files), embedder=PixelEmbedder(dim=4))
    assert code == 0
    predictions = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(predictions) == 2
    assert all(p["status"] == "ok" and p["read_fields"] for p in predictions)
    assert all(FINAL_KEY in p and p["outcome"] for p in predictions)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    block = metrics["cv_vs_cv_ocr"]
    assert set(block) >= {"cv_only", "cv_ocr", "delta_pp", "outcomes"}
    assert metrics["run"]["with_ocr"].endswith("ocr_predictions.jsonl")
    assert metrics["run"]["with_ocr_fields"] == 2
    assert metrics["run"]["with_ocr_covered"] == 2
    assert metrics["run"]["resolve"]["cfg"]["top_k"] == DEFAULT_CONFIG.top_k
    assert metrics["run"]["resolve"]["attrs"]["slugs"] == len(TOY_RECORDS)
    # Оговорка «веса не подобраны» лежит рядом с числами, а не только внутри блока сравнения.
    assert "не подобраны на данных" in metrics["run"]["resolve"]["weights_note"]


def test_main_refuses_ocr_that_does_not_overlap_the_set(tmp_path, capsys):
    """Прогон OCR с чужими query_id: прирост дал бы один добор серий, а подписан как текст."""
    files = ocr_setup(tmp_path, read_ids=("other-1", "other-2"))
    assert main(ocr_argv(files), embedder=PixelEmbedder(dim=4)) == 2
    assert "ни один query_id" in capsys.readouterr().err
    assert not files["out"].exists()


def test_main_reports_partial_ocr_coverage(tmp_path, capsys):
    files = ocr_setup(tmp_path, read_ids=("w1",))
    assert main(ocr_argv(files), embedder=PixelEmbedder(dim=4)) == 0
    assert "есть у 1 кадров из 2" in capsys.readouterr().err
    metrics = json.loads((files["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["with_ocr_covered"] == 1
    assert metrics["cv_vs_cv_ocr"]["read_coverage"] == 0.5


def test_main_without_ocr_stays_visual(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.full((48, 48, 3), 80, dtype=np.uint8)).save(images / "w1.webp", "PNG")
    pairs_json = tmp_path / "benchmark.json"
    pairs_json.write_text(
        json.dumps([{"wine_id": "w1", "portal_slug": TWINS[0]}]), encoding="utf-8"
    )
    gt_tokens = tmp_path / "gt_tokens.jsonl"
    gt_tokens.write_text(json.dumps(TOY_RECORDS[0], ensure_ascii=False) + "\n", encoding="utf-8")
    index = tiny_index(tmp_path / "visual.npz", [TWINS[0]])
    out = tmp_path / "run"
    code = main(
        [
            "--queries",
            "pairs",
            "--pairs-json",
            str(pairs_json),
            "--pairs-images",
            str(images),
            "--index",
            str(index),
            "--gt-tokens",
            str(gt_tokens),
            "--out",
            str(out),
            "--split",
            "all",
        ],
        embedder=PixelEmbedder(dim=4),
    )
    assert code == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cv_vs_cv_ocr"] is None and metrics["run"]["resolve"] is None
    record = json.loads((out / "predictions.jsonl").read_text(encoding="utf-8"))
    assert FINAL_KEY not in record


def test_main_needs_catalog_markup_for_ocr(tmp_path, capsys):
    out = main(
        [
            "--queries",
            "pairs",
            "--index",
            str(tmp_path / "нет.npz"),
            "--gt-tokens",
            str(tmp_path / "нет.jsonl"),
            "--with-ocr",
            str(tmp_path / "ocr.jsonl"),
            "--out",
            str(tmp_path / "run"),
        ],
        embedder=PixelEmbedder(dim=4),
    )
    assert out == 3 and "требует разметки каталога" in capsys.readouterr().err
