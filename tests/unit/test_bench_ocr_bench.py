import json
from pathlib import Path

import numpy as np
import pytest

from app.reading.contracts import Box, LexHit, Reading, TextLine, image_sha1, params_hash
from app.reading.lexicon.build import build_from_records
from bench import ocr_bench
from bench.datasets import SEALED_METRICS_FILE, BenchItem
from bench.ocr_bench import (
    TEXT_TOP_NOTE,
    BenchSetupError,
    Pipeline,
    annotation_box,
    annotation_target,
    import_attr,
    parse_reader_spec,
    run_bench,
    text_top_slugs,
)


class FakeReader:
    id = "fake"
    version = "0"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0
        self.crops = []

    def available(self):
        return True

    def read(self, image, *, crop, budget_ms):
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        self.crops.append((crop, image.shape))
        return Reading(
            reader=self.id,
            version=self.version,
            params_hash=params_hash({"budget_ms": budget_ms}),
            image_sha1=image_sha1(image),
            crop=crop,
            crop_px=max(image.shape[:2]),
            lines=[TextLine(id=i, text=t) for i, t in enumerate(text.split("\n"))],
            raw=text,
            elapsed_ms=100 * self.calls,
            prompt_tokens=600,
        )


def _decode(data: bytes) -> np.ndarray:
    if data == b"bad":
        raise ValueError("битый кадр")
    return np.zeros((16, 12, 3), np.uint8)


def _fake_set(tmp_path, *, with_images=True):
    images = tmp_path / "images"
    images.mkdir()
    if with_images:
        (images / "a.jpg").write_bytes(b"img-a")
        (images / "b.webp").write_bytes(b"bad")
        (images / "c.jpg").write_bytes(b"img-c")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "query_id\timage_path\nq-1\ta.jpg\nq-2\tb.webp\nq-3\tc.jpg\n", encoding="utf-8"
    )
    gt = tmp_path / "gt.tsv"
    gt.write_text(
        "query_id\tslug\tin_catalog\nq-1\twine-a\t1\nq-2\t__none__\t0\nq-3\twine-b\t1\n",
        encoding="utf-8",
    )
    ann = tmp_path / "ann.jsonl"
    rows = [
        {
            "query_id": "q-1",
            "target_slug": "wine-a",
            "front_label_text": [
                {"field": "winery", "text": "ТАБИЯ"},
                {"field": "year", "text": "2025"},
            ],
        },
        {
            "query_id": "q-2",
            "target_slug": None,
            "front_label_text": [{"field": "cuvee", "text": "DONUM"}],
        },
    ]
    ann.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    args = [
        "--dataset",
        "manifest",
        "--manifest",
        str(manifest),
        "--gt",
        str(gt),
        "--ann",
        str(ann),
        "--images-dir",
        str(images),
        "--gt-tokens",
        str(tmp_path / "absent.jsonl"),
        "--lexicon",
        str(tmp_path / "absent_lexicon.json"),
    ]
    return args


def _forbid_pipeline(*_args, **_kwargs):
    raise AssertionError("dry-run не должен собирать конвейер")


def _record(slug, winery, key, sugar=None):
    """Запись `gt_tokens.jsonl` с винодельней и сахаром."""
    return {
        "slug": slug,
        "winery": winery,
        "fields": {
            "winery": {"key_tokens": key, "variants": [], "brands": []},
            "cuvee": {"tokens": [], "variants": []},
            "grape": {"values": [], "codes": [], "variants": []},
            "sugar": {"class": sugar, "variants": []},
            "serial": {"tokens": [], "keywords": []},
            "color": {"class": None, "variants": []},
            "year": {"value": None},
        },
    }


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("vlm:qwen3-vl:4b-instruct", ("vlm", "qwen3-vl:4b-instruct")),
        ("easyocr", ("easyocr", None)),
        ("rapidocr", ("rapidocr", None)),
        ("rapidocr:server", ("rapidocr", "server")),
        ("vlm", ("vlm", None)),
    ],
)
def test_parse_reader_spec(spec, expected):
    assert parse_reader_spec(spec) == expected


@pytest.mark.parametrize("spec", ["tesseract", "", ":vlm"])
def test_parse_reader_spec_rejects_unknown(spec):
    with pytest.raises(ValueError):
        parse_reader_spec(spec)


def test_import_attr_explains_missing_module():
    with pytest.raises(BenchSetupError, match="ещё не написан"):
        import_attr("app.reading.no_such_module_for_bench", "make_crop")
    with pytest.raises(BenchSetupError, match="нет no_such_attr"):
        import_attr("json", "no_such_attr")


def test_import_attr_explains_missing_dependency(tmp_path, monkeypatch):
    (tmp_path / "svs_bench_broken_mod.py").write_text(
        "import svs_missing_dep_xyz\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(BenchSetupError, match="svs_missing_dep_xyz"):
        import_attr("svs_bench_broken_mod", "anything")


def test_dry_run_checks_fake_set_without_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", _forbid_pipeline)
    out = tmp_path / "run"
    assert ocr_bench.main([*_fake_set(tmp_path), "--dry-run", "--out", str(out)]) == 0
    report = json.loads((out / "dataset_check.json").read_text(encoding="utf-8"))
    assert report["ok"] and report["dry_run"]
    assert (report["items"], report["in_catalog"], report["out_of_catalog"]) == (3, 2, 1)
    assert report["token_sources"] == {"front_label_text": 2, "none": 1}
    assert report["lexicon"]["exists"] is False
    assert not (out / "predictions.jsonl").exists()


def test_dry_run_fails_on_missing_images(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", _forbid_pipeline)
    assert ocr_bench.main([*_fake_set(tmp_path, with_images=False), "--dry-run"]) == 1


def test_run_writes_predictions_and_metrics(tmp_path, monkeypatch):
    reader = FakeReader(["ТАБИЯ\n2025", "ARISTOV\nDONUM"])
    pipeline = Pipeline(decode=_decode, readers=(reader,), target=annotation_target)
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", lambda *args, **kwargs: pipeline)
    out = tmp_path / "run"
    args = [*_fake_set(tmp_path), "--reader", "easyocr", "--crop", "label", "--crop-px", "512"]
    argv = [*args, "--split", "all", "--unseal-test", "--out", str(out), "--no-warmup"]
    assert ocr_bench.main(argv) == 0

    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    preds = [json.loads(line) for line in lines]
    assert [p["query_id"] for p in preds] == ["q-1", "q-2", "q-3"]
    assert [p["status"] for p in preds] == ["ok", "error", "ok"]
    assert preds[0]["raw_text"] == "ТАБИЯ\n2025" and preds[0]["prompt_tokens"] == 600
    assert preds[0]["reader"].startswith("fake@0|") and preds[0]["lines"][1]["text"] == "2025"
    assert preds[1]["error"].startswith("decode: ValueError")
    # Рамки цели в разметке нет: label уходит в весь кадр, это видно в предсказании.
    assert preds[0]["crop_used"] == "full" and "crop_fallback_full" in preds[0]["degraded"]
    assert preds[0]["fields"]["year"] == 2025 and preds[0]["text_top5"] == []
    assert preds[0]["label_fields"]["vintage"]["value"] == 2025

    result = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert result["statuses"] == {"error": 1, "ok": 2}
    assert result["key_token_cer"]["winery"]["mean"] == 0.0
    assert result["key_token_cer"]["key_fields"]["n"] == 2
    assert result["latency_ms"] == {"p50": 100, "p95": 200, "n": 2}
    assert result["latency_ms_warm"]["n"] == 1
    assert result["run"]["crop"] == "label" and result["run"]["reader_id"] == "fake"
    # Бюджет стенда по умолчанию — сервисный, прогон с ним в пределах SLA.
    assert (result["run"]["budget_ms"], result["run"]["sla_budget"]) == (2500, True)
    assert result["run"]["split"] == "all" and result["run"]["warmup"] is None
    assert result["fields_filled"]["year"] == 0.5
    assert "diagnostic_text_only" not in result  # без словаря текстового top-k нет
    assert reader.calls == 2


class WarmAwareReader(FakeReader):
    """Большой синтетический кадр прогрева «читается» 9 с, маленькие кадры набора — 100 мс."""

    params_hash = "p0"

    def crop_px_for(self, image):
        return max(image.shape[:2])

    def read(self, image, *, crop, budget_ms):
        reading = super().read(image, crop=crop, budget_ms=budget_ms)
        elapsed = 9000 if image.shape[0] >= 512 else 100
        return reading.model_copy(update={"elapsed_ms": elapsed, "params_hash": self.params_hash})


def test_warmup_runs_before_frames_bypasses_cache_and_stays_out_of_latency(tmp_path):
    from app.reading.readers.cache import CachedReader
    from app.reading.warmup import synthetic_label

    set_args = _fake_set(tmp_path)
    items = ocr_bench.load_manifest(
        Path(set_args[3]), Path(set_args[5]), Path(set_args[7]), images_dir=Path(set_args[9])
    )
    inner = WarmAwareReader(["ТАБИЯ\n2025"])
    cached = CachedReader(inner, tmp_path / "cache")
    metrics = run_bench(
        [items[0], items[2]],
        Pipeline(decode=_decode, readers=(cached,)),
        crop="full",
        crop_px=1024,
        budget_ms=1000,
        out_dir=tmp_path / "run",
        gt_tokens={},
        warmup=True,
    )
    warm = metrics["run"]["warmup"]
    assert list(warm) == ["fake"] and warm["fake"]["status"] == "ok"
    assert isinstance(warm["fake"]["elapsed_ms"], int)
    # Прогрев — первым и синтетическим кадром размера crop_px; оба кадра набора одинаковы.
    assert inner.crops == [("full", (1024, 1024, 3)), ("full", (16, 12, 3))]
    assert metrics["latency_ms"] == {"p50": 100, "p95": 100, "n": 2}
    assert metrics["latency_ms_warm"] == {"p50": 100, "p95": 100, "n": 1}
    assert (cached.stats.writes, cached.stats.hits) == (1, 1)
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1
    assert cached.get(cached.key_for(synthetic_label(1024), crop="full")) is None


@pytest.mark.parametrize(("flag", "calls", "warmed"), [([], 3, True), (["--no-warmup"], 2, False)])
def test_cli_warmup_is_on_by_default(tmp_path, monkeypatch, flag, calls, warmed):
    reader = WarmAwareReader(["ТАБИЯ\n2025"])
    pipeline = Pipeline(decode=_decode, readers=(reader,))
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", lambda *args, **kwargs: pipeline)
    out = tmp_path / "run"
    args = [*_fake_set(tmp_path), "--reader", "easyocr", "--split", "all", "--out", str(out)]
    assert ocr_bench.main([*args, "--unseal-test", *flag]) == 0
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert reader.calls == calls  # два кадра набора читаются, третий не декодируется
    assert (metrics["run"]["warmup"] is not None) is warmed
    assert metrics["latency_ms"] == {"p50": 100, "p95": 100, "n": 2}


def test_run_with_lexicon_records_fields_and_diagnostic_top(tmp_path):
    lexicon = build_from_records(
        [
            _record("tabiya-brut", "Табия", ["табия"], sugar="brut"),
            _record("aristov-red", "Аристов", ["аристов"]),
        ]
    )
    set_args = _fake_set(tmp_path)
    items = ocr_bench.load_manifest(
        Path(set_args[3]), Path(set_args[5]), Path(set_args[7]), images_dir=Path(set_args[9])
    )
    reader = FakeReader(["ТАБИЯ\nБрют 2025"])
    pipeline = Pipeline(decode=_decode, readers=(reader,), lexicon=lexicon)
    gt_tokens = {
        "wine-a": {"slug": "wine-a", "fields": _record("wine-a", "Табия", ["табия"])["fields"]}
    }
    metrics = run_bench(
        items[:1],
        pipeline,
        crop="full",
        crop_px=1024,
        budget_ms=1000,
        out_dir=tmp_path / "run",
        gt_tokens=gt_tokens,
    )
    pred = json.loads((tmp_path / "run" / "predictions.jsonl").read_text(encoding="utf-8"))
    assert pred["fields"]["winery"] == "Табия" and pred["fields"]["sugar"] == ["brut"]
    assert pred["text_top5"][0] == "tabiya-brut"
    assert pred["text_top5_scores"][0][0] == "tabiya-brut"
    assert metrics["diagnostic_text_only"]["note"] == TEXT_TOP_NOTE
    assert metrics["field_accuracy"]["winery"]["correct"] == 1
    assert "GRAPE_SYNONYMS" in metrics["field_accuracy_notes"]["cuvee_or_grape"]
    assert metrics["run"]["lexicon_entries"] == len(lexicon)


def test_run_without_split_reads_only_dev(tmp_path, monkeypatch):
    reader = FakeReader(["Шардоне 2020"])
    pipeline = Pipeline(decode=_decode, readers=(reader,))
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", lambda *args, **kwargs: pipeline)
    args = _fake_set(tmp_path)
    (tmp_path / "gt.tsv").write_text(
        "query_id\tslug\tin_catalog\tsplit\n"
        "q-1\twine-a\t1\tdev\nq-2\t__none__\t0\ttest\nq-3\twine-b\t1\ttest\n",
        encoding="utf-8",
    )
    out = tmp_path / "run"
    assert ocr_bench.main([*args, "--reader", "easyocr", "--out", str(out)]) == 0
    preds = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [p["query_id"] for p in preds] == ["q-1"]
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["split"] == "dev" and metrics["splits"] == {"dev": 1}
    # Проверка набора без --split смотрит весь набор: согласованность split видна целиком.
    check = tmp_path / "check"
    assert ocr_bench.main([*args, "--dry-run", "--out", str(check)]) == 0
    report = json.loads((check / "dataset_check.json").read_text(encoding="utf-8"))
    assert report["split"] == "all" and report["items"] == 3


def test_text_top_slugs_prefers_rare_entities_and_counts_each_once():
    common = LexHit(canonical="brut", field="sugar", cost=0.0, slugs=frozenset({"a", "b", "c"}))
    rare = LexHit(canonical="Табия", field="producer", cost=0.0, slugs=frozenset({"b"}))
    fuzzy = LexHit(canonical="Аристов", field="producer", cost=1.0, slugs=frozenset({"c"}))
    hits = [((0,), common), ((1,), common), ((2,), common), ((3,), rare), ((4,), fuzzy)]
    top = text_top_slugs(hits, n_slugs=100)
    assert [slug for slug, _ in top] == ["b", "c", "a"]
    only_common = text_top_slugs([((0,), common), ((1,), common)], n_slugs=100)
    assert only_common[0][1] == text_top_slugs([((0,), common)], n_slugs=100)[0][1]
    assert text_top_slugs([], n_slugs=100) == []


def test_existing_run_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", _forbid_pipeline)
    out = tmp_path / "run"
    out.mkdir()
    (out / "predictions.jsonl").write_text("{}\n", encoding="utf-8")
    assert ocr_bench.main([*_fake_set(tmp_path), "--reader", "easyocr", "--out", str(out)]) == 2


def test_missing_pipeline_module_exits_with_setup_code(tmp_path, monkeypatch, capsys):
    def fake_import(module, attr):
        raise BenchSetupError(f"модуль {module} ещё не написан")

    monkeypatch.setattr(ocr_bench, "import_attr", fake_import)
    out = tmp_path / "run"
    args = [*_fake_set(tmp_path), "--reader", "rapidocr", "--split", "all"]
    assert ocr_bench.main([*args, "--out", str(out)]) == 3
    assert "стенд не готов" in capsys.readouterr().err


def test_broken_lexicon_exits_with_setup_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", _forbid_pipeline)
    broken = tmp_path / "lexicon.json"
    broken.write_text("{}", encoding="utf-8")
    args = [*_fake_set(tmp_path), "--reader", "easyocr", "--lexicon", str(broken), "--split", "all"]
    assert ocr_bench.main([*args, "--out", str(tmp_path / "run")]) == 3
    assert "словарь" in capsys.readouterr().err


def test_run_requires_reader_and_out(tmp_path):
    with pytest.raises(SystemExit):
        ocr_bench.main(_fake_set(tmp_path))


def test_dry_run_with_label_crop_requires_target_boxes(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", _forbid_pipeline)
    out = tmp_path / "run"
    args = [*_fake_set(tmp_path), "--crop", "label", "--dry-run", "--out", str(out)]
    assert ocr_bench.main(args) == 1
    report = json.loads((out / "dataset_check.json").read_text(encoding="utf-8"))
    assert report["crop_boxes"] == {
        "source": "annotation",
        "key": "target_bbox_visible",
        "missing": 3,
    }
    (tmp_path / "d").mkdir()
    args = [*_fake_set(tmp_path / "d"), "--crop", "label", "--target", "detector", "--dry-run"]
    assert ocr_bench.main(args) == 0


def _boxed_item(**annotations):
    return BenchItem(
        query_id="q",
        image_path=Path("q.jpg"),
        slug=None,
        in_catalog=False,
        annotations=annotations,
        split="smoke",
    )


def test_annotation_box_converts_pixels_to_fractions():
    item = _boxed_item(target_bbox_visible=[30, 0, 270, 400])
    box = annotation_box(item, 300, 400)
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((0.1, 0.0, 0.9, 1.0))
    assert annotation_box(_boxed_item(target_bbox_visible=[60, 400, 50, 420]), 300, 400) is None
    assert annotation_box(_boxed_item(), 300, 400) is None


def test_annotation_target_is_confident_and_crops_label_in_read_label(tmp_path):
    image = np.full((400, 300, 3), 200, np.uint8)
    (tmp_path / "q.jpg").write_bytes(b"img")
    item = _boxed_item(target_bbox_visible=[60, 0, 240, 400]).model_copy(
        update={"image_path": tmp_path / "q.jpg"}
    )
    target = annotation_target(image, item)
    assert target.confident and target.target == Box(x0=0.2, y0=0.0, x1=0.8, y1=1.0)
    assert annotation_target(image, _boxed_item()) is None

    reader = FakeReader(["DONUM"])
    pipeline = Pipeline(decode=lambda data: image, readers=(reader,), target=annotation_target)
    record = ocr_bench.read_item(item, pipeline, crop="label", crop_px=1024, budget_ms=1000)
    crop, shape = reader.crops[0]
    assert (
        crop == "label"
        and record["crop_used"] == "label"
        and record["degraded"] == ["lexicon_missing"]
    )
    assert shape[0] == 280  # нижние 70 % высоты бутылки
    assert 180 <= shape[1] <= 210  # ширина рамки с отступом 8 % с каждой стороны


def test_split_all_seals_test_metrics_in_an_envelope(tmp_path, monkeypatch, capsys):
    reader = FakeReader(["ТАБИЯ\n2025", "ARISTOV\nDONUM"])
    pipeline = Pipeline(decode=_decode, readers=(reader,), target=annotation_target)
    monkeypatch.setattr(ocr_bench, "resolve_pipeline", lambda *args, **kwargs: pipeline)
    out = tmp_path / "run"
    args = [*_fake_set(tmp_path), "--reader", "easyocr", "--split", "all", "--no-warmup"]
    assert ocr_bench.main([*args, "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "запечатаны" in captured.err
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # предсказания — по всему набору
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    envelope = json.loads((out / SEALED_METRICS_FILE).read_text(encoding="utf-8"))
    sealed = metrics["sealed_test"]["items"]
    assert sealed >= 1 and metrics["run"]["test_sealed"] is True
    assert metrics["items"] == 3 - sealed and "test" not in metrics["splits"]
    assert "test" not in metrics["key_token_cer_by_split"]
    assert envelope["items"] == 3 and envelope["splits"]["test"] == sealed
    assert "КОНВЕРТ" in envelope["warning"]
    assert "sealed_test" in json.loads(captured.out)
