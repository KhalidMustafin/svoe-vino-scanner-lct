import json

import numpy as np
import pytest
from fakes import HashEmbedder, PixelEmbedder, frame

from app.features.contracts import IndexMeta
from app.features.embedder import unit_rows
from app.features.index import IndexMismatch, VisualIndex, align_pairs

# Игрушечный каталог: пять вин, у каждого три вида. Векторы заданы пикселями кадра,
# поэтому косинусы известны заранее.
CATALOG: dict[str, dict[str, tuple[float, ...]]] = {
    "alpha": {
        "bottle": (1.0, 0.0, 0.0, 0.0),
        "label": (0.9, 0.4, 0.0, 0.0),
        "band": (0.8, 0.6, 0.0, 0.0),
    },
    "beta": {
        "bottle": (0.0, 1.0, 0.0, 0.0),
        "label": (0.0, 0.9, 0.4, 0.0),
        "band": (0.0, 0.8, 0.6, 0.0),
    },
    "gamma": {
        "bottle": (0.0, 0.0, 1.0, 0.0),
        "label": (0.0, 0.0, 0.9, 0.4),
        "band": (0.0, 0.0, 0.8, 0.6),
    },
    "delta": {
        "bottle": (0.0, 0.0, 0.0, 1.0),
        "label": (0.4, 0.0, 0.0, 0.9),
        "band": (0.6, 0.0, 0.0, 0.8),
    },
    "epsilon": {
        "bottle": (0.5, 0.5, 0.5, 0.5),
        "label": (0.6, 0.6, 0.4, 0.4),
        "band": (0.4, 0.4, 0.6, 0.6),
    },
}


def entries(catalog=CATALOG):
    for slug, views in catalog.items():
        yield slug, {name: frame(values) for name, values in views.items()}


@pytest.fixture
def embedder() -> PixelEmbedder:
    return PixelEmbedder(dim=4)


@pytest.fixture
def index(embedder: PixelEmbedder) -> VisualIndex:
    return VisualIndex.build(entries(), embedder, source_sha1="deadbeef")


def test_build_makes_one_vector_per_view(index: VisualIndex):
    assert len(index) == 15 and index.n_slugs == 5
    assert index.slugs[:3] == ["alpha", "alpha", "alpha"]
    assert index.views[:3] == ["bottle", "label", "band"]  # канонический порядок видов
    assert index.vectors.shape == (15, 4) and index.vectors.dtype == np.float32


def test_build_normalizes_rows(index: VisualIndex):
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_build_fills_the_passport(index: VisualIndex):
    meta = index.meta
    assert meta.model == "fake/pixel" and meta.dim == 4
    assert meta.views == ["bottle", "label", "band"]
    assert (meta.n_slugs, meta.n_vectors) == (5, 15)
    assert meta.source_sha1 == "deadbeef" and meta.built_at.endswith("+00:00")


def test_build_splits_into_batches(embedder: PixelEmbedder):
    VisualIndex.build(entries(), embedder, batch_size=4)
    assert embedder.batches == [4, 4, 4, 3]  # 15 кадров порциями по четыре


def test_build_of_empty_catalog_is_empty(embedder: PixelEmbedder):
    empty = VisualIndex.build(iter([]), embedder)
    assert len(empty) == 0 and empty.n_slugs == 0 and empty.meta.dim == 4
    assert empty.search({"full": frame((1.0, 0, 0, 0))}, embedder).candidates == []


def test_build_rejects_entry_without_views(embedder: PixelEmbedder):
    with pytest.raises(ValueError, match="ни одного вида"):
        VisualIndex.build([("alpha", {})], embedder)


def test_save_and_load_roundtrip(index: VisualIndex, tmp_path):
    path = index.save(tmp_path / "sub" / "visual.npz")
    assert path.is_file() and path.name == "visual.npz"
    loaded = VisualIndex.load(path)
    assert loaded.slugs == index.slugs and loaded.views == index.views
    assert loaded.meta == index.meta
    # float16 в файле: разница с исходным вектором ниже шума кадра
    assert np.allclose(loaded.vectors, index.vectors, atol=1e-3)


def test_saved_file_holds_float16_and_meta(index: VisualIndex, tmp_path):
    path = index.save(tmp_path / "visual.npz")
    with np.load(path, allow_pickle=False) as data:
        assert data["vectors"].dtype == np.float16
        assert set(data.files) == {"slugs", "views", "vectors", "meta"}
        assert json.loads(str(data["meta"].item()))["model"] == "fake/pixel"


def test_load_rejects_other_model(index: VisualIndex, tmp_path):
    path = index.save(tmp_path / "visual.npz")
    with pytest.raises(IndexMismatch, match="fake/pixel"):
        VisualIndex.load(path, model="google/siglip-base-patch16-224")
    assert VisualIndex.load(path, model="fake/pixel").n_slugs == 5


def test_load_rejects_file_without_fields(tmp_path):
    path = tmp_path / "broken.npz"
    with path.open("wb") as fh:
        np.savez(fh, slugs=np.array(["alpha"]))
    with pytest.raises(IndexMismatch, match="нет полей"):
        VisualIndex.load(path)


def test_load_rejects_dim_that_contradicts_the_passport(index: VisualIndex, tmp_path):
    path = tmp_path / "visual.npz"
    meta = index.meta.model_copy(update={"dim": 7})
    with path.open("wb") as fh:
        np.savez(
            fh,
            slugs=np.array(index.slugs),
            views=np.array(index.views),
            vectors=index.vectors.astype(np.float16),
            meta=np.array(meta.model_dump_json()),
        )
    with pytest.raises(IndexMismatch, match="против dim 7"):
        VisualIndex.load(path)


def test_load_rejects_unknown_view(index: VisualIndex, tmp_path):
    path = tmp_path / "visual.npz"
    views = ["neck"] + index.views[1:]
    with path.open("wb") as fh:
        np.savez(
            fh,
            slugs=np.array(index.slugs),
            views=np.array(views),
            vectors=index.vectors.astype(np.float16),
            meta=np.array(index.meta.model_dump_json()),
        )
    with pytest.raises(IndexMismatch, match="неизвестные виды"):
        VisualIndex.load(path)


def test_search_finds_the_exact_view(index: VisualIndex, embedder: PixelEmbedder):
    result = index.search({"full": frame((0.0, 0.9, 0.4, 0.0))}, embedder)
    assert result.candidates[0].slug == "beta"
    assert result.candidates[0].score == pytest.approx(1.0, abs=1e-3)
    assert result.candidates[0].view == "label"  # именно этот вид эталона и совпал
    assert result.candidates[0].rank == 1
    assert [c.rank for c in result.candidates] == list(range(1, len(result.candidates) + 1))


def test_search_collapses_views_of_one_slug(index: VisualIndex, embedder: PixelEmbedder):
    result = index.search({"bottle": frame((1.0, 0.0, 0.0, 0.0))}, embedder)
    slugs = [c.slug for c in result.candidates]
    assert len(slugs) == len(set(slugs)) == 5  # у каждого вина одна строка, а не три


def test_search_takes_the_best_pair_of_views(index: VisualIndex, embedder: PixelEmbedder):
    """Совпал хоть один вид запроса с одним видом эталона — вино узнано."""
    query = {"bottle": frame((0.0, 0.0, 0.0, 1.0)), "band": frame((0.0, 0.8, 0.6, 0.0))}
    result = index.search(query, embedder)
    assert {c.slug for c in result.candidates[:2]} == {"beta", "delta"}
    assert result.candidates[0].score == pytest.approx(1.0, abs=1e-3)


def test_search_honours_top_k(index: VisualIndex, embedder: PixelEmbedder):
    result = index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, embedder, top_k=2)
    assert len(result.candidates) == 2 and result.candidates[0].slug == "alpha"


def test_margin_is_the_gap_between_first_and_second(index: VisualIndex, embedder: PixelEmbedder):
    result = index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, embedder)
    gap = result.candidates[0].score - result.candidates[1].score
    assert result.margin == pytest.approx(gap, abs=1e-4) and result.margin > 0


def test_margin_does_not_depend_on_top_k(index: VisualIndex, embedder: PixelEmbedder):
    """Отрыв считается по всему каталогу: при `top_k=1` срез выдачи длиной в одного."""
    query = {"full": frame((1.0, 0.0, 0.0, 0.0))}
    full = index.search(query, embedder, top_k=len(index.slug_order)).margin
    assert full > 0
    for top_k in (1, 2, 3):
        result = index.search(query, embedder, top_k=top_k)
        assert len(result.candidates) == min(top_k, index.n_slugs)
        assert result.margin == pytest.approx(full, abs=1e-6)


def test_margin_of_twins_is_near_zero(embedder: PixelEmbedder):
    """Две позиции с одной этикеткой: CV их не разводит, и отрыв это показывает."""
    twins = {"left": CATALOG["alpha"], "right": CATALOG["alpha"]}
    index = VisualIndex.build(entries(twins), embedder)
    result = index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, embedder)
    assert result.margin == pytest.approx(0.0, abs=1e-6)


def test_mean_aggregation_differs_from_max(index: VisualIndex, embedder: PixelEmbedder):
    query = {"full": frame((1.0, 0.0, 0.0, 0.0))}
    by_max = index.search(query, embedder, per_slug="max").candidates[0]
    by_mean = index.search(query, embedder, per_slug="mean").candidates[0]
    assert by_max.slug == by_mean.slug == "alpha"
    assert by_mean.score < by_max.score  # среднее по трём видам ниже лучшего


def unit(cos: float) -> tuple[float, float]:
    """Единичный вектор с косинусом `cos` к запросу (1, 0)."""
    return cos, float(np.sqrt(1.0 - cos * cos))


def background_index() -> VisualIndex:
    """Этикетки каталога все похожи на запрос (фон пары высокий), бутылки — разные.

    У alpha этикетка чуть выше фона своей пары (0,83 при 0,81–0,82), у beta бутылка далеко
    выше фона своей (0,80 при 0,10–0,30). Сырой максимум отдаёт alpha, выровненный — beta.
    """
    labels = {"alpha": 0.83, "beta": 0.82, "gamma": 0.82, "delta": 0.82, "epsilon": 0.81}
    bottles = {"alpha": 0.30, "beta": 0.80, "gamma": 0.20, "delta": 0.10, "epsilon": 0.15}
    slugs, views, vectors = [], [], []
    for slug in labels:
        for name, table in (("bottle", bottles), ("label", labels)):
            slugs.append(slug)
            views.append(name)
            vectors.append(unit(table[slug]))
    meta = IndexMeta(model="fake/pixel", dim=2, n_slugs=5, n_vectors=10)
    return VisualIndex(slugs, views, np.array(vectors, dtype=np.float32), meta)


def test_align_pairs_brings_every_pair_to_the_distribution_of_the_query():
    rng = np.random.default_rng(0)
    sims = np.concatenate([rng.normal(0.8, 0.02, (6, 3)), rng.normal(0.4, 0.2, (5, 3))])
    groups = [np.arange(6), np.arange(6, 11)]
    aligned = align_pairs(sims, groups)
    for rows in groups:
        assert np.allclose(aligned[rows].mean(axis=0), sims.mean(), atol=1e-5)
        assert np.allclose(aligned[rows].std(axis=0), sims.std(), atol=1e-5)
        # Внутри пары порядок кандидатов тот же: меняется только то, какая пара выиграет.
        for window in range(3):
            assert (np.argsort(aligned[rows, window]) == np.argsort(sims[rows, window])).all()


def test_align_pairs_leaves_a_flat_pair_as_it_is():
    """Вид с одним вектором или одинаковыми векторами: σ пары ноль, выравнивать не по чему."""
    sims = np.array([[0.5, 0.9], [0.5, 0.1], [0.7, 0.3]], dtype=np.float32)
    aligned = align_pairs(sims, [np.array([0, 1]), np.array([2])])
    assert aligned[0, 0] == aligned[1, 0] == pytest.approx(0.5)  # плоская пара — как была
    assert aligned[2].tolist() == pytest.approx([0.7, 0.3])  # один вектор вида — как был
    assert aligned[0, 1] > aligned[1, 1]  # живая пара выровнена, порядок держится


def test_view_groups_follow_the_rows_of_the_index(index: VisualIndex):
    groups = dict(zip(("bottle", "label", "band"), index.view_groups, strict=True))
    assert groups["bottle"].tolist() == [0, 3, 6, 9, 12]
    assert groups["band"].tolist() == [2, 5, 8, 11, 14]


def test_zmax_prefers_a_pair_that_stands_out_over_a_pair_with_a_high_background():
    index = background_index()
    query = {"full": frame((1.0, 0.0))}
    by_max = index.search(query, PixelEmbedder(dim=2), per_slug="max")
    by_zmax = index.search(query, PixelEmbedder(dim=2), per_slug="zmax")
    assert by_max.candidates[0].slug == "alpha" and by_max.candidates[0].view == "label"
    assert by_zmax.candidates[0].slug == "beta" and by_zmax.candidates[0].view == "bottle"
    # Выровненный счёт в единицах косинуса запроса и выше 1 не обрезается: иначе у сильных
    # совпадений были бы ничьи на самом верху выдачи.
    assert by_zmax.candidates[0].score > 1.0
    gap = by_zmax.candidates[0].score - by_zmax.candidates[1].score
    assert by_zmax.margin == pytest.approx(gap, abs=1e-4) and by_zmax.margin > 0


def test_zmax_keeps_the_order_of_max_when_there_is_one_pair(embedder: PixelEmbedder):
    """Один вид эталона и одно окно запроса: выравнивание — возрастающее отображение."""
    bottles = {slug: {"bottle": views["bottle"]} for slug, views in CATALOG.items()}
    index = VisualIndex.build(entries(bottles), embedder)
    query = {"full": frame((0.7, 0.3, 0.5, 0.2))}
    by_max = [c.slug for c in index.search(query, embedder, per_slug="max").candidates]
    by_zmax = [c.slug for c in index.search(query, embedder, per_slug="zmax").candidates]
    assert by_zmax == by_max


def test_negative_cosine_is_clipped_to_zero():
    """Косинус ниже нуля — «совсем не то»; в контракте score живёт в 0..1."""
    embedder = PixelEmbedder(dim=2)
    index = VisualIndex.build([("alpha", {"bottle": frame((1.0, 0.0))})], embedder)
    index.vectors = np.array([[-1.0, 0.0]], dtype=np.float32)
    result = index.search({"full": frame((1.0, 0.0))}, embedder)
    assert result.candidates[0].score == 0.0


class BrokenEmbedder:
    """Модель, вернувшая NaN: так выглядит переполнение float16 на видеокарте."""

    model_name = "fake/pixel"
    dim = 4

    def embed(self, images):
        return np.full((len(images), 4), np.nan, dtype=np.float32)


def test_search_refuses_a_nan_vector_instead_of_answering(index: VisualIndex):
    """Раньше запрос с NaN получал первый slug индекса со счётом 0,0 и статусом «ok»."""
    with pytest.raises(ValueError, match="NaN или Inf"):
        index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, BrokenEmbedder())


def test_search_rejects_another_model(index: VisualIndex):
    with pytest.raises(IndexMismatch, match="fake/other"):
        index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, PixelEmbedder(4, "fake/other"))


def test_search_rejects_another_dim(index: VisualIndex):
    other = PixelEmbedder(dim=8, model_name="fake/pixel")
    with pytest.raises(IndexMismatch, match="против индекса 4"):
        index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, other)


def test_search_rejects_empty_query_and_bad_arguments(index: VisualIndex, embedder: PixelEmbedder):
    query = {"full": frame((1.0, 0.0, 0.0, 0.0))}
    with pytest.raises(ValueError, match="ни одного вида запроса"):
        index.search({}, embedder)
    with pytest.raises(ValueError, match="top_k"):
        index.search(query, embedder, top_k=0)
    with pytest.raises(ValueError, match="per_slug"):
        index.search(query, embedder, per_slug="median")


def test_search_reports_timings_and_model(index: VisualIndex, embedder: PixelEmbedder):
    result = index.search({"full": frame((1.0, 0.0, 0.0, 0.0))}, embedder)
    assert set(result.timings_ms) == {"embed", "match", "total"}
    # Миллисекунды с долями: плоский поиск занимает 1,4 мс, и в целых он был бы нулём.
    assert all(isinstance(v, float) and v >= 0 for v in result.timings_ms.values())
    assert result.model == "fake/pixel" and result.top is not None


def test_index_rejects_mismatched_columns():
    meta = IndexMeta(model="fake/pixel", dim=2, n_slugs=1, n_vectors=1)
    with pytest.raises(ValueError, match="не сходится"):
        VisualIndex(["alpha", "beta"], ["bottle"], np.zeros((2, 2), dtype=np.float32), meta)


def test_hash_embedder_index_is_stable():
    """Сборка не зависит от порядка батчей: тот же каталог — тот же индекс."""
    first = VisualIndex.build(entries(), HashEmbedder(dim=8), batch_size=2)
    second = VisualIndex.build(entries(), HashEmbedder(dim=8), batch_size=7)
    assert np.allclose(first.vectors, second.vectors)
    assert first.slugs == second.slugs and first.views == second.views


# ------------------------------------------------------------------ заморозка нормировки (Э3)
def signs(rng: np.random.Generator, rows: int, dim: int = 16) -> np.ndarray:
    """Единичные векторы из ±1/4: косинусы кратны 1/16 и считаются точно при любом порядке сумм."""
    return rng.choice(np.array([-0.25, 0.25], dtype=np.float32), size=(rows, dim))


def extended_index(n_base: int = 12, n_extra: int = 6, dim: int = 16, seed: int = 0):
    """База и та же база с дополнением: у дополнения свои slug, те же три вида."""
    rng = np.random.default_rng(seed)
    names = ("bottle", "label", "band")
    slugs = [f"csv-{i // 3}" for i in range(n_base)] + [f"live-{i // 3}" for i in range(n_extra)]
    views = [names[i % 3] for i in range(n_base)] + [names[i % 3] for i in range(n_extra)]
    vectors = signs(rng, n_base + n_extra, dim)
    meta = IndexMeta(model="fake/pixel", dim=dim, n_slugs=len(set(slugs)), n_vectors=len(slugs))
    base = VisualIndex(slugs[:n_base], views[:n_base], vectors[:n_base], meta)
    mask = np.arange(n_base + n_extra) < n_base
    live = VisualIndex(slugs, views, vectors, meta, base_rows=mask)
    return base, live, rng


def test_align_pairs_with_a_full_mask_is_the_unmasked_path_bit_for_bit():
    rng = np.random.default_rng(1)
    sims = rng.normal(0.5, 0.2, (15, 4)).astype(np.float32)
    groups = [np.arange(0, 15, 3), np.arange(1, 15, 3), np.arange(2, 15, 3)]
    full = np.ones(15, dtype=bool)
    assert np.array_equal(align_pairs(sims, groups, full), align_pairs(sims, groups))


def test_frozen_base_rows_keep_the_scores_of_the_base_rows():
    """Дополнение не сдвигает выровненные косинусы строк базы: mu, sd и M, s — только по базе."""
    base, live, rng = extended_index()
    sims = live.vectors @ unit_rows(signs(rng, 4)).T
    frozen = align_pairs(sims, live.view_groups, live.base_rows)
    alone = align_pairs(sims[:12], base.view_groups)
    assert np.array_equal(frozen[:12], alone)
    # Без маски та же матрица нормируется по всем строкам — и база сдвигается.
    assert not np.allclose(align_pairs(sims, live.view_groups)[:12], alone)


def test_extended_index_ranks_base_slugs_exactly_as_the_base_index():
    base, live, rng = extended_index(seed=3)
    for _ in range(5):
        queries = unit_rows(signs(rng, 4))
        base_cands, _ = base._rank(queries, 20, "zmax")
        live_cands, _ = live._rank(queries, 20, "zmax")
        kept = [(c.slug, c.score, c.view) for c in live_cands if c.slug.startswith("csv-")]
        assert kept == [(c.slug, c.score, c.view) for c in base_cands]
        assert len(live_cands) == 6  # 4 slug базы + 2 дополнения


def test_group_without_base_rows_stays_raw():
    sims = np.array([[0.2, 0.4], [0.6, 0.1], [0.9, 0.8]], dtype=np.float32)
    mask = np.array([True, True, False])
    aligned = align_pairs(sims, [np.array([0, 1]), np.array([2])], mask)
    assert aligned[2].tolist() == pytest.approx([0.9, 0.8])


def test_base_rows_survive_save_and_load(tmp_path):
    _, live, _ = extended_index()
    path = live.save(tmp_path / "live.npz")
    loaded = VisualIndex.load(path)
    assert loaded.base_rows is not None and loaded.base_rows.tolist() == live.base_rows.tolist()
    assert loaded.n_base == 12 and len(loaded) == 18
    base = VisualIndex.load(
        VisualIndex(loaded.slugs, loaded.views, loaded.vectors, loaded.meta).save(
            tmp_path / "base.npz"
        )
    )
    assert base.base_rows is None and base.n_base == len(base)
    with np.load(tmp_path / "base.npz") as data:
        assert "base_rows" not in data.files  # индекс без добавлений пишется как прежде


def test_full_mask_is_no_mask_and_bad_masks_are_refused():
    base, live, _ = extended_index()
    same = VisualIndex(base.slugs, base.views, base.vectors, base.meta, base_rows=np.ones(12, bool))
    assert same.base_rows is None
    with pytest.raises(IndexMismatch, match="длины 18"):
        VisualIndex(live.slugs, live.views, live.vectors, live.meta, base_rows=np.ones(5, bool))
    with pytest.raises(IndexMismatch, match="bool"):
        VisualIndex(live.slugs, live.views, live.vectors, live.meta, base_rows=np.ones(18))
    with pytest.raises(IndexMismatch, match="ни одной"):
        VisualIndex(live.slugs, live.views, live.vectors, live.meta, base_rows=np.zeros(18, bool))
