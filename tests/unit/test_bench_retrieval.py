"""Стенд поиска: порча кадра, наборы запросов, счёт метрик и CLI.

Индекс здесь игрушечный: пять «вин» — сплошные цвета, а фейковый эмбеддер считает средний
цвет кадра. Любой кроп одноцветного кадра даёт тот же вектор, поэтому косинусы, порядок
кандидатов и все метрики известны заранее — без весов, видеокарты и сети.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fakes import ColorEmbedder
from PIL import Image

from app.features.index import VisualIndex
from bench.datasets import SEALED_METRICS_FILE, DatasetError
from bench.phone_shot import noise_image, perspective_coeffs, phone_shot, phone_shot_rgb
from bench.queries import QueryItem, load_pairs, load_queries, missing_images, summarize
from bench.retrieval import (
    Pipeline,
    assign_splits,
    credited_slugs,
    default_split,
    group_key,
    line_key,
    main,
    margin_table,
    margins_by_correctness,
    rank_columns,
    rank_table,
    run_retrieval,
    search_item,
    take_limit,
    tie_rate,
    twin_key,
    unpublished_answers,
    visual_group_metrics,
)

# Каталог: цвет «бутылки» и её место в группах двойников.
CATALOG: dict[str, tuple[int, int, int]] = {
    "alpha": (255, 0, 0),
    "beta": (0, 255, 0),
    "gamma": (0, 0, 255),
    "twin-a": (255, 200, 0),  # пара двойников: цвета почти совпадают, как макет этикетки
    "twin-b": (255, 190, 0),
}
GT_TOKENS: dict[str, dict[str, object]] = {
    "alpha": {"cluster_B": None, "visual_group": None, "visual_mates": []},
    "beta": {"cluster_B": None, "visual_group": None, "visual_mates": []},
    "gamma": {"cluster_B": None, "visual_group": None, "visual_mates": []},
    "twin-a": {"cluster_B": 7, "visual_group": [7, 0], "visual_mates": ["twin-b"]},
    "twin-b": {"cluster_B": 7, "visual_group": [7, 0], "visual_mates": ["twin-a"]},
}
#: Запросы: цвет кадра и ожидаемый slug. Ответы посчитаны по углам между цветами.
QUERIES: list[tuple[str, tuple[int, int, int], str | None]] = [
    ("q-red", (255, 0, 0), "alpha"),  # точное попадание
    ("q-green", (0, 255, 0), "beta"),  # точное попадание
    ("q-twin", (255, 190, 0), "twin-a"),  # первым встанет двойник twin-b
    ("q-gray", (128, 128, 128), None),  # вина нет в каталоге
    ("q-absent", (0, 255, 255), "zeta"),  # цели нет в индексе вовсе
]


def solid(color: tuple[int, int, int], size: tuple[int, int] = (24, 40)) -> np.ndarray:
    """Одноцветный кадр RGB uint8 (ширина, высота)."""
    width, height = size
    return np.full((height, width, 3), color, dtype=np.uint8)


def write_solid(path: Path, color: tuple[int, int, int]) -> Path:
    """Одноцветный кадр на диск в WebP без потерь: цвет должен вернуться тем же."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(solid(color)).save(path, format="WEBP", lossless=True)
    return path


@pytest.fixture
def embedder() -> ColorEmbedder:
    return ColorEmbedder()


@pytest.fixture
def index(embedder: ColorEmbedder) -> VisualIndex:
    entries = (
        (slug, {name: solid(color) for name in ("bottle", "label", "band")})
        for slug, color in CATALOG.items()
    )
    return VisualIndex.build(entries, embedder)


@pytest.fixture
def pipeline(index: VisualIndex, embedder: ColorEmbedder) -> Pipeline:
    return Pipeline(index=index, embedder=embedder, top_k=3)


@pytest.fixture
def items(tmp_path: Path) -> list[QueryItem]:
    out = []
    for query_id, color, slug in QUERIES:
        path = write_solid(tmp_path / "queries" / f"{query_id}.webp", color)
        out.append(QueryItem(query_id=query_id, image_path=path, slug=slug, meta={"set": "toy"}))
    return out


# ------------------------------------------------------------------ порча кадра
def sample_photo(seed: int = 3, size: tuple[int, int] = (40, 60)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_phone_shot_repeats_itself_on_the_same_seed():
    photo = sample_photo()
    first = np.asarray(phone_shot(photo, 7))
    second = np.asarray(phone_shot(photo, 7))
    assert first.shape == second.shape and np.array_equal(first, second)


def test_phone_shot_differs_between_seeds():
    photo = sample_photo()
    first = np.asarray(phone_shot(photo, 7))
    second = np.asarray(phone_shot(photo, 8))
    assert first.shape != second.shape or not np.array_equal(first, second)


def test_phone_shot_spoils_the_frame():
    photo = sample_photo()
    spoiled = phone_shot(photo, 1)
    # Вокруг бутылки появился фон, кадр повернули и обрезали: тех же пикселей не осталось.
    assert spoiled.mode == "RGB" and spoiled.size != photo.size


def test_phone_shot_rgb_keeps_the_array_contract():
    spoiled = phone_shot_rgb(np.asarray(sample_photo()), 2)
    assert spoiled.dtype == np.uint8 and spoiled.ndim == 3 and spoiled.shape[2] == 3
    with pytest.raises(ValueError, match="RGB uint8"):
        phone_shot_rgb(np.zeros((4, 4), dtype=np.uint8), 2)


def test_noise_image_repeats_itself_unlike_pillow():
    first = np.asarray(noise_image((16, 12), 30.0, 5))
    second = np.asarray(noise_image((16, 12), 30.0, 5))
    assert first.shape == (12, 16) and np.array_equal(first, second)
    assert not np.array_equal(first, np.asarray(noise_image((16, 12), 30.0, 6)))


def test_perspective_coeffs_of_untouched_corners_are_identity():
    corners = [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
    assert np.allclose(perspective_coeffs(corners, corners), [1, 0, 0, 0, 1, 0, 0, 0], atol=1e-9)


# ------------------------------------------------------------------ наборы запросов
def write_pairs(tmp_path: Path, records: list[dict[str, str]]) -> tuple[Path, Path]:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path, tmp_path / "roskachestvo"


def test_load_pairs_takes_the_query_from_the_other_source(tmp_path: Path):
    path, images = write_pairs(
        tmp_path,
        [
            {"wine_id": "wine-1", "portal_slug": "alpha"},
            {"wine_id": "wine-2", "portal_slug": "beta"},
        ],
    )
    items = load_pairs(path, images)
    assert [i.query_id for i in items] == ["wine-1", "wine-2"]
    assert [i.slug for i in items] == ["alpha", "beta"]
    # Запрос — снимок Роскачества, а не фото портала: оно и есть эталон каталога.
    assert items[0].image_path == images / "wine-1.webp"
    assert items[0].meta["source"] == "roskachestvo" and items[0].phone_seed is None


def test_load_pairs_phone_seeds_are_record_numbers(tmp_path: Path):
    path, images = write_pairs(
        tmp_path, [{"wine_id": f"wine-{n}", "portal_slug": f"slug-{n}"} for n in range(4)]
    )
    items = load_pairs(path, images, phone=True)
    assert [i.phone_seed for i in items] == [0, 1, 2, 3]
    # --limit режет хвост, сиды первых записей не сдвигаются.
    assert [i.phone_seed for i in items[:2]] == [0, 1]


@pytest.mark.parametrize(
    "records",
    [
        [{"wine_id": "wine-1"}],
        [{"portal_slug": "alpha"}],
        [
            {"wine_id": "wine-1", "portal_slug": "alpha"},
            {"wine_id": "wine-1", "portal_slug": "beta"},
        ],
        [],
    ],
)
def test_load_pairs_refuses_broken_files(tmp_path: Path, records: list[dict[str, str]]):
    path, images = write_pairs(tmp_path, records)
    with pytest.raises(DatasetError):
        load_pairs(path, images)


def test_load_queries_refuses_an_unknown_set():
    with pytest.raises(DatasetError, match="неизвестный набор"):
        load_queries("pairs_studio")


def test_load_queries_reads_the_rwl_manifest(tmp_path: Path):
    directory, images = tmp_path / "rwl", tmp_path / "images"
    directory.mkdir()
    (directory / "src_manifest.tsv").write_text(
        "query_id\timage_path\nsrc_1\ta.jpg\nsrc_2\tb.jpg\n", encoding="utf-8"
    )
    (directory / "src_gt.tsv").write_text(
        "query_id\tslug\tin_catalog\nsrc_1\talpha\t1\nsrc_2\t__none__\t0\n", encoding="utf-8"
    )
    (directory / "src_ann.jsonl").write_text(
        '{"query_id": "src_1", "target_slug": "alpha"}\n{"query_id": "src_2", "target_slug": null}\n',
        encoding="utf-8",
    )
    items = load_queries("rwl_src", rwl_dir=directory, rwl_images=images)
    assert [(i.query_id, i.slug) for i in items] == [("src_1", "alpha"), ("src_2", None)]
    assert items[0].image_path == images / "a.jpg" and items[0].meta["set"] == "rwl_src"


def test_summarize_and_missing_images_see_the_set(items: list[QueryItem], tmp_path: Path):
    assert summarize(items) == {
        "queries": 5,
        "in_catalog": 4,
        "out_of_catalog": 1,
        "slugs": 4,
        "phone_shots": 0,
    }
    assert missing_images(items) == []
    absent = QueryItem(query_id="q-x", image_path=tmp_path / "nope.webp", slug="alpha")
    assert missing_images([*items, absent]) == [f"q-x: нет файла {tmp_path / 'nope.webp'}"]


# ------------------------------------------------------------------ счёт метрик
def test_rank_table_counts_places():
    table = rank_table([1, 2, 4, 11, None, None], top_k=20)
    assert table["top1"] == rate_of(1, 6) and table["top3"] == rate_of(2, 6)
    assert table["top5"] == rate_of(3, 6) and table["top10"] == rate_of(3, 6)
    assert table["recall_at_20"] == rate_of(4, 6) and table["n"] == 6


def rate_of(num: int, den: int) -> float:
    return round(num / den, 3)


def test_margin_table_is_median_and_tenth_percentile():
    table = margin_table([0.5, 0.1, 0.3, 0.2, 0.4])
    assert table == {"median": 0.3, "p10": 0.1, "n": 5}
    assert margin_table([]) == {"median": None, "p10": None, "n": 0}


def test_series_and_group_keys_keep_lonely_wines_alone():
    assert line_key("twin-a", GT_TOKENS) == line_key("twin-b", GT_TOKENS) == "B7"
    assert line_key("alpha", GT_TOKENS) == "slug:alpha"  # вне кластера вино равно только себе
    assert line_key("zeta", GT_TOKENS) == "slug:zeta" and line_key(None, GT_TOKENS) is None
    assert group_key("twin-a", GT_TOKENS) == "7-0" and group_key("alpha", GT_TOKENS) is None
    assert twin_key("twin-a", GT_TOKENS) == "7-0" and twin_key("alpha", GT_TOKENS) == "slug:alpha"


def test_ranks_by_card_and_by_group_differ_on_twins():
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="twin-a"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="alpha"),
    ]
    preds = {
        "q1": {"top": [{"slug": "twin-b"}, {"slug": "twin-a"}]},
        "q2": {"top": [{"slug": "beta"}, {"slug": "gamma"}]},
    }
    columns = rank_columns(items, preds, GT_TOKENS)
    assert columns["by_card"] == [2, None]  # своя карточка вторая, у alpha её в выдаче нет
    assert columns["by_visual_group"] == [1, None]  # но макет угадан с первого места
    assert columns["by_winery_line"] == [1, None]


def test_margins_split_correct_answers_from_wrong():
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="alpha"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="twin-a"),
        QueryItem(query_id="q3", image_path=Path("c"), slug=None),
    ]
    preds = {
        "q1": {"status": "ok", "margin": 0.4, "top": [{"slug": "alpha"}]},
        "q2": {"status": "ok", "margin": 0.01, "top": [{"slug": "twin-b"}]},
        "q3": {"status": "ok", "margin": 0.9, "top": [{"slug": "alpha"}]},
    }
    table = margins_by_correctness(items, preds)
    assert table["correct"] == {"median": 0.4, "p10": 0.4, "n": 1}
    assert table["wrong"] == {"median": 0.01, "p10": 0.01, "n": 1}


def test_visual_group_metrics_count_answers_by_a_twin():
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="twin-a"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="twin-b"),
        QueryItem(query_id="q3", image_path=Path("c"), slug="alpha"),  # вне групп: не считается
    ]
    preds = {
        "q1": {"top": [{"slug": "twin-b"}]},
        "q2": {"top": [{"slug": "twin-b"}]},
        "q3": {"top": [{"slug": "alpha"}]},
    }
    assert visual_group_metrics(items, preds, GT_TOKENS) == {
        "queries": 2,
        "groups_touched": 1,
        "groups_in_catalog": 1,
        "top1": 0.5,
        "answered_by_mate": 0.5,
    }


# ------------------------------------------------------------------ поблажки метрик
#: Каталог, где линейка винодельни шире визуальной группы — как `cluster_B` в разметке:
#: «Дербент Каберне» и «Дербент Шардоне» лежат в одном B0, хотя это красное и белое.
WIDE_TOKENS: dict[str, dict[str, object]] = {
    "line-cabernet": {"cluster_B": 0, "visual_group": None, "visual_mates": []},
    "line-chardonnay": {"cluster_B": 0, "visual_group": None, "visual_mates": []},
    "twin-2024": {"cluster_B": 0, "visual_group": [0, 1], "visual_mates": ["twin-2025"]},
    "twin-2025": {"cluster_B": 0, "visual_group": [0, 1], "visual_mates": ["twin-2024"]},
    "lonely": {"cluster_B": None, "visual_group": None, "visual_mates": []},
}


def test_winery_line_is_wider_than_the_visual_group():
    # Ответ «Шардоне» на запрос «Каберне» — промах по карточке и по макету, но попадание
    # по линейке: cluster_B склеивает вина одной винодельни, а не двойников.
    items = [QueryItem(query_id="q1", image_path=Path("a"), slug="line-cabernet")]
    preds = {"q1": {"top": [{"slug": "line-chardonnay"}, {"slug": "line-cabernet"}]}}
    columns = rank_columns(items, preds, WIDE_TOKENS)
    assert columns["by_card"] == [2]
    assert columns["by_visual_group"] == [2]  # своя группа — только сама карточка
    assert columns["by_winery_line"] == [1]  # а линейка засчитывает чужой сорт и цвет


def test_visual_group_still_forgives_the_year():
    items = [QueryItem(query_id="q1", image_path=Path("a"), slug="twin-2025")]
    preds = {"q1": {"top": [{"slug": "twin-2024"}]}}
    columns = rank_columns(items, preds, WIDE_TOKENS)
    assert columns["by_card"] == [None] and columns["by_visual_group"] == [1]


def test_credited_slugs_show_the_size_of_the_forgiveness():
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="line-cabernet"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="twin-2024"),
        QueryItem(query_id="q3", image_path=Path("c"), slug="lonely"),
    ]
    # Пулы: 1 (сама карточка), 2 (двойники), 1 (одиночка).
    assert credited_slugs(items, WIDE_TOKENS, twin_key) == {
        "median": 1,
        "max": 2,
        "mean": 1.333,
        "n": 3,
    }
    line = credited_slugs(items, WIDE_TOKENS, line_key)
    assert line["max"] == 4 and line["median"] == 4  # весь B0 зачтён одному запросу


def test_empty_columns_say_there_is_nothing_to_measure():
    # Раньше `rate` делил на max(1, den) и печатал 0.0 — «канал не угадал ничего».
    table = rank_table([], top_k=20)
    assert table == {
        "top1": None,
        "top3": None,
        "top5": None,
        "top10": None,
        "recall_at_20": None,
        "n": 0,
    }
    assert credited_slugs([], WIDE_TOKENS, twin_key)["n"] == 0


def test_ties_are_counted_as_the_ceiling_of_by_card():
    # Семь файлов каталога стоят у двух slug: их векторы совпадают до бита.
    items = [QueryItem(query_id=f"q{i}", image_path=Path("a"), slug="lonely") for i in (1, 2, 3)]
    preds = {
        "q1": {"status": "ok", "top": [{"score": 0.9}, {"score": 0.9}]},
        "q2": {"status": "ok", "top": [{"score": 0.9}, {"score": 0.5}]},
        "q3": {"status": "error", "top": []},
    }
    table = tie_rate(items, preds)
    assert table["queries"] == 2 and table["ties"] == 1 and table["rate"] == 0.5


def test_unpublished_answers_are_counted():
    tokens = {
        "hidden": {"published": False, "cluster_B": None},
        "shown": {"published": True, "cluster_B": None},
    }
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="shown"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="shown"),
    ]
    preds = {
        "q1": {"status": "ok", "top": [{"slug": "hidden"}]},
        "q2": {"status": "ok", "top": [{"slug": "shown"}]},
    }
    assert unpublished_answers(items, preds, tokens) == {
        "in_catalog": 1,
        "answered": 2,
        "top1_unpublished": 1,
        "rate": 0.5,
    }
    assert unpublished_answers(items, preds, {"shown": {"published": True}}) is None


def test_splits_group_twins_together_and_stay_stable():
    items = [
        QueryItem(query_id="q1", image_path=Path("a"), slug="twin-2024"),
        QueryItem(query_id="q2", image_path=Path("b"), slug="twin-2025"),
        QueryItem(query_id="q3", image_path=Path("c"), slug="lonely", meta={"split": "smoke"}),
    ]
    split = assign_splits(items, WIDE_TOKENS)
    assert {i.query_id: i.meta["split"] for i in split}["q3"] == "smoke"  # чужое не трогаем
    # Двойники одного кластера обязаны попасть в одну половину.
    assert split[0].meta["split"] == split[1].meta["split"]
    assert split[0].meta["split"] in ("dev", "test")
    assert [i.meta["split"] for i in assign_splits(items, WIDE_TOKENS)] == [
        i.meta["split"] for i in split
    ]


def test_take_limit_samples_with_a_seed_instead_of_the_prefix():
    items = [QueryItem(query_id=f"q{i}", image_path=Path("a")) for i in range(10)]
    assert [i.query_id for i in take_limit(items, 3, None)] == ["q0", "q1", "q2"]
    sampled = [i.query_id for i in take_limit(items, 3, 42)]
    assert len(sampled) == 3 and sampled != ["q0", "q1", "q2"]
    assert [i.query_id for i in take_limit(items, 3, 42)] == sampled  # сид держит выборку


# ------------------------------------------------------------------ прогон
def test_search_item_finds_the_wine_and_records_timings(items, pipeline: Pipeline):
    record = search_item(items[0], pipeline)
    assert record["status"] == "ok" and record["top"][0]["slug"] == "alpha"
    assert record["top"][0]["score"] == pytest.approx(1.0) and record["hit_rank"] == 1
    assert [c["rank"] for c in record["top"]] == [1, 2, 3]  # top_k=3
    assert record["views"] == ["band", "bottle", "full", "label"]
    assert set(record["timings_ms"]) >= {"decode", "embed", "match", "total"}


def test_search_item_puts_the_twin_first_and_leaves_a_tiny_margin(items, pipeline: Pipeline):
    record = search_item(items[2], pipeline)
    assert [c["slug"] for c in record["top"][:2]] == ["twin-b", "twin-a"]
    assert record["hit_rank"] == 2 and record["margin"] < 0.01


def test_search_item_writes_down_a_broken_frame(tmp_path: Path, pipeline: Pipeline):
    broken = tmp_path / "broken.webp"
    broken.write_bytes(b"not an image at all")
    record = search_item(QueryItem(query_id="q-broken", image_path=broken, slug="alpha"), pipeline)
    assert record["status"] == "error" and record["error"].startswith("decode: DecodeError")
    assert record["top"] == [] and record["timings_ms"]["total"] >= 0


def test_search_item_spoils_the_frame_when_a_seed_is_given(items, pipeline: Pipeline):
    plain = search_item(items[0], pipeline)
    spoiled = QueryItem(
        query_id="q-red-phone",
        image_path=items[0].image_path,
        slug="alpha",
        meta={"phone_seed": 0},
    )
    record = search_item(spoiled, pipeline)
    assert record["status"] == "ok"
    # Фон, наклон и блик сдвигают средний цвет: тот же кадр даёт другой счёт.
    assert record["top"][0]["score"] != plain["top"][0]["score"]


def test_run_writes_one_prediction_per_query(items, pipeline: Pipeline, tmp_path: Path):
    out = tmp_path / "run"
    run_retrieval(items, pipeline, out_dir=out, gt_tokens=GT_TOKENS)
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["query_id"] for r in records] == [q[0] for q in QUERIES]
    assert all(r["status"] == "ok" for r in records)
    assert records[0]["cv_top5"][:1] == ["alpha"] and records[0]["meta"]["set"] == "toy"
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["by_card"]["n"] == 4 and saved["run"]["top_k"] == 3


def test_metrics_of_the_toy_run_are_known_in_advance(items, pipeline: Pipeline, tmp_path: Path):
    metrics = run_retrieval(items, pipeline, out_dir=tmp_path / "run", gt_tokens=GT_TOKENS)
    assert metrics["queries"] == 5 and metrics["failed"] == 0
    # Своя карточка первой у двух из четырёх, серия — у трёх: двойника различает не снимок.
    assert metrics["by_card"]["top1"] == 0.5 and metrics["by_card"]["top3"] == 0.75
    assert metrics["by_visual_group"]["top1"] == 0.75
    assert metrics["by_winery_line"]["top1"] == 0.75
    assert metrics["by_card"]["recall_at_3"] == 0.75  # zeta в индексе нет вовсе
    assert metrics["margin"]["correct"]["n"] == 2 and metrics["margin"]["wrong"]["n"] == 2
    assert metrics["margin"]["correct"]["median"] > metrics["margin"]["wrong"]["median"]
    assert metrics["visual_groups"] == {
        "queries": 1,
        "groups_touched": 1,
        "groups_in_catalog": 1,
        "top1": 0.0,
        "answered_by_mate": 1.0,
    }
    assert metrics["out_of_catalog_scores"]["queries"] == 1
    assert metrics["latency_ms"]["total"]["n"] == 5


def test_metrics_count_a_failed_frame_without_losing_the_run(
    items, pipeline: Pipeline, tmp_path: Path
):
    broken = tmp_path / "broken.webp"
    broken.write_bytes(b"still not an image")
    spoiled = [*items, QueryItem(query_id="q-broken", image_path=broken, slug="alpha")]
    metrics = run_retrieval(spoiled, pipeline, out_dir=tmp_path / "run", gt_tokens=GT_TOKENS)
    assert metrics["failed"] == 1 and metrics["errors"][0].startswith("q-broken: decode")
    assert metrics["by_card"]["n"] == 5  # сбойный кадр считается промахом, а не пропуском


def test_compute_metrics_survives_an_empty_gt(items, pipeline: Pipeline, tmp_path: Path):
    metrics = run_retrieval(items, pipeline, out_dir=tmp_path / "run", gt_tokens={})
    # Без gt_tokens и макет, и линейка вырождаются в сам slug, а групп нет вовсе.
    assert metrics["by_visual_group"]["top1"] == 0.5
    assert metrics["by_winery_line"]["top1"] == 0.5
    assert metrics["visual_groups"]["queries"] == 0
    assert metrics["visual_groups"]["top1"] is None  # мерить нечего, а не «ноль попаданий»
    assert metrics["gt_tokens_loaded"] == 0


# ------------------------------------------------------------------ CLI
@pytest.fixture
def cli(tmp_path: Path, index: VisualIndex) -> dict[str, Path]:
    """Индекс, пары и эталонные токены на диске — всё, что нужно `main`."""
    index_path = index.save(tmp_path / "index" / "visual.npz")
    images = tmp_path / "roskachestvo"
    records = []
    for query_id, color, slug in QUERIES:
        if slug is None:
            continue
        write_solid(images / f"{query_id}.webp", color)
        records.append({"wine_id": query_id, "portal_slug": slug})
    pairs = tmp_path / "benchmark.json"
    pairs.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    gt = tmp_path / "gt_tokens.jsonl"
    gt.write_text(
        "".join(
            json.dumps({"slug": s, **r}, ensure_ascii=False) + "\n" for s, r in GT_TOKENS.items()
        ),
        encoding="utf-8",
    )
    return {
        "index": index_path,
        "pairs": pairs,
        "images": images,
        "gt": gt,
        "out": tmp_path / "run",
    }


def argv_for(cli: dict[str, Path], *extra: str) -> list[str]:
    flags = {
        "--index": cli["index"],
        "--queries": "pairs",
        "--pairs-json": cli["pairs"],
        "--pairs-images": cli["images"],
        "--gt-tokens": cli["gt"],
        "--out": cli["out"],
        "--top-k": "3",
        # Игрушечный набор меряется целиком: по умолчанию стенд взял бы только dev, а при
        # all запечатал бы test — здесь печать снята явно.
        "--split": "all",
    }
    return [str(cell) for pair in flags.items() for cell in pair] + ["--unseal-test", *extra]


def test_main_runs_the_set_and_writes_the_run(cli, embedder: ColorEmbedder, capsys):
    assert main(argv_for(cli), embedder=embedder) == 0
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["queries"] == 4 and metrics["by_card"]["top1"] == 0.5
    assert metrics["run"]["queries_set"] == "pairs" and metrics["run"]["target"] == "none"
    assert metrics["run"]["notes"] and "самопоиск" in metrics["run"]["notes"][0]
    assert capsys.readouterr().out.strip()


def test_main_takes_the_aligned_aggregation_of_the_service(cli, embedder: ColorEmbedder):
    """`--per-slug zmax` — так считает сервис; паспорт прогона это записывает."""
    assert main(argv_for(cli, "--per-slug", "zmax"), embedder=embedder) == 0
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["per_slug"] == "zmax" and metrics["queries"] == 4


def test_main_limit_cuts_the_tail(cli, embedder: ColorEmbedder):
    assert main(argv_for(cli, "--limit", "2"), embedder=embedder) == 0
    lines = (cli["out"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["query_id"] for line in lines] == ["q-red", "q-green"]
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["queries"] == 2 and metrics["run"]["limit"] == 2


def test_main_samples_with_a_seed_and_marks_the_prefix(cli, embedder: ColorEmbedder, capsys):
    assert main(argv_for(cli, "--limit", "2"), embedder=embedder) == 0
    assert "не замер" in capsys.readouterr().err
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert "префикс набора" in metrics["run"]["limit_note"]
    assert (
        main(argv_for(cli, "--limit", "2", "--sample-seed", "5", "--overwrite"), embedder=embedder)
        == 0
    )
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert (
        metrics["run"]["sample_seed"] == 5 and "случайная выборка" in metrics["run"]["limit_note"]
    )


def test_main_splits_the_set_and_records_the_shares(cli, embedder: ColorEmbedder):
    assert main(argv_for(cli), embedder=embedder) == 0
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["split"] == "all"
    assert set(metrics["run"]["split_shares"]) <= {"dev", "test"}
    assert sum(metrics["run"]["split_shares"].values()) == 4
    # Разрез по половинам считается всегда: подбирать пороги можно только на dev.
    assert set(metrics["by_split"]) == set(metrics["run"]["split_shares"])
    assert sum(part["queries"] for part in metrics["by_split"].values()) == 4


def test_main_runs_one_half_of_the_set(cli, embedder: ColorEmbedder):
    shares = {}
    for half in ("dev", "test"):
        code = main(argv_for(cli, "--split", half, "--overwrite"), embedder=embedder)
        if code == 1:  # в игрушечном наборе половина может оказаться пустой
            shares[half] = 0
            continue
        metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["run"]["split"] == half and set(metrics["by_split"]) == {half}
        shares[half] = metrics["queries"]
    assert sum(shares.values()) == 4


def test_default_split_matches_the_ocr_bench_discipline():
    """Дисциплина проекта та же, что у `bench.ocr_bench`: test смотрят один раз и явно."""
    assert default_split("pairs") == default_split("pairs_phone") == "dev"
    assert default_split("rwl_aug") == "dev" and default_split("public") == "all"


def test_main_takes_dev_without_a_split_flag(cli, embedder: ColorEmbedder, capsys):
    # Все пять игрушечных slug хэш кладёт в test, поэтому dev тут пуст — и это видно.
    argv = [a for a in argv_for(cli) if a not in ("--split", "all")]
    assert main(argv, embedder=embedder) == 1
    assert "половины dev" in capsys.readouterr().err and not cli["out"].exists()
    assert main(argv_for(cli, "--split", "test"), embedder=embedder) == 0
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run"]["split"] == "test" and metrics["queries"] == 4


def test_pairs_notes_admit_the_funnel_and_the_in_sample_windows(cli, embedder: ColorEmbedder):
    assert main(argv_for(cli), embedder=embedder) == 0
    notes = " ".join(
        json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))["run"]["notes"]
    )
    assert "580" in notes and "358" in notes  # воронка разметки пар
    assert "in-sample" in notes  # окна запроса выбраны на этом же наборе


def test_main_refuses_to_overwrite_a_run(cli, embedder: ColorEmbedder):
    assert main(argv_for(cli), embedder=embedder) == 0
    assert main(argv_for(cli), embedder=embedder) == 2
    assert main(argv_for(cli, "--overwrite"), embedder=embedder) == 0


def test_main_reports_a_missing_frame(cli, embedder: ColorEmbedder):
    (cli["images"] / "q-red.webp").unlink()
    assert main(argv_for(cli), embedder=embedder) == 1
    assert not cli["out"].exists()


def test_main_refuses_an_index_from_another_model(cli, tmp_path: Path):
    assert main(argv_for(cli), embedder=ColorEmbedder(model_name="fake/other")) == 3


def test_main_refuses_a_missing_index(cli, embedder: ColorEmbedder, tmp_path: Path):
    argv = argv_for(cli)
    argv[argv.index("--index") + 1] = str(tmp_path / "nope.npz")
    assert main(argv, embedder=embedder) == 3


def test_split_all_seals_test_metrics_in_an_envelope(cli, embedder: ColorEmbedder, capsys):
    """`--split all` — ради предсказаний по всему набору; метрики test в `metrics.json` и в
    консоль не попадают, они лежат в конверте."""
    argv = [a for a in argv_for(cli) if a != "--unseal-test"]
    assert main(argv, embedder=embedder) == 0
    captured = capsys.readouterr()
    assert "запечатаны" in captured.err
    lines = (cli["out"] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # предсказания — по всему набору
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    # Все игрушечные slug хэш кладёт в test: открытая часть пуста.
    assert metrics["queries"] == 0 and metrics["by_card"]["n"] == 0
    assert "test" not in metrics["by_split"] and metrics["run"]["test_sealed"] is True
    assert metrics["sealed_test"]["queries"] == 4
    assert metrics["sealed_test"]["file"] == SEALED_METRICS_FILE
    assert "test" not in json.loads(captured.out)["by_split"]
    envelope = json.loads((cli["out"] / SEALED_METRICS_FILE).read_text(encoding="utf-8"))
    assert envelope["queries"] == 4 and envelope["by_split"]["test"]["queries"] == 4
    assert "КОНВЕРТ" in envelope["warning"]


def test_explicit_test_run_is_not_sealed(cli, embedder: ColorEmbedder):
    argv = [a for a in argv_for(cli, "--split", "test") if a != "--unseal-test"]
    assert main(argv, embedder=embedder) == 0
    metrics = json.loads((cli["out"] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["queries"] == 4 and "sealed_test" not in metrics
    assert not (cli["out"] / SEALED_METRICS_FILE).exists()
