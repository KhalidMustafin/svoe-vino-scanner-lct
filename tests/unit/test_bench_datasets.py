import json

import pytest

from app.config import REPO_ROOT, Settings
from bench.datasets import (
    DatasetError,
    check_images,
    load_manifest,
    load_public,
    load_synth,
    split_for,
    summarize,
    twin_clusters,
)
from bench.synth.make_field_synth import write_outputs

PUBLIC_QUERIES = "query_id\timage_path\nq-000001\t019c68d0.jpg\nq-000002\t02eef911.webp\nq-000003\t096ca74e.jpg\n"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _field_set(tmp_path, *, gt_rows, manifest_rows=None, ann=None):
    manifest_rows = manifest_rows or [(qid, f"{qid}.jpg") for qid, *_ in gt_rows]
    manifest = _write(
        tmp_path / "manifest.tsv",
        "query_id\timage_path\n" + "".join(f"{q}\t{p}\n" for q, p in manifest_rows),
    )
    header = "query_id\tslug\tin_catalog\tsplit\n"
    gt = _write(tmp_path / "gt.tsv", header + "".join("\t".join(r) + "\n" for r in gt_rows))
    ann_path = None
    if ann is not None:
        ann_path = _write(
            tmp_path / "ann.jsonl", "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in ann)
        )
    return manifest, gt, ann_path


def test_split_is_deterministic_and_close_to_40_60():
    slugs = [f"wine-{i}" for i in range(4000)]
    splits = [split_for(s, "q") for s in slugs]
    assert splits == [split_for(s, "other-query") for s in slugs]
    assert 0.37 <= splits.count("dev") / len(splits) <= 0.43


def test_split_groups_bottles_and_splits_unlabelled_frames_by_query():
    assert len({split_for("massandra-muskatel", f"q-{i}") for i in range(50)}) == 1
    assert len({split_for(None, f"q-{i}", group="bottle-7") for i in range(50)}) == 1
    assert len({split_for(None, f"q-{i}") for i in range(50)}) == 2


def test_load_public_reads_repo_annotations(tmp_path):
    _write(tmp_path / "eval" / "queries.tsv", PUBLIC_QUERIES)
    settings = Settings(dataset_dir=tmp_path, data_dir=REPO_ROOT / "data")
    items = load_public(settings)
    assert [item.query_id for item in items] == ["q-000001", "q-000002", "q-000003"]
    assert {item.split for item in items} == {"smoke"}
    by_id = {item.query_id: item for item in items}
    assert by_id["q-000002"].in_catalog and by_id["q-000002"].slug.startswith("massandra-")
    assert by_id["q-000001"].slug is None and not by_id["q-000001"].in_catalog
    assert by_id["q-000003"].image_path == tmp_path / "eval" / "queries" / "096ca74e.jpg"
    assert all(item.annotations["front_label_text"] for item in items)


def test_load_public_without_dataset_is_clear(tmp_path):
    settings = Settings(dataset_dir=tmp_path / "absent", data_dir=REPO_ROOT / "data")
    with pytest.raises(DatasetError, match="SVS_DATASET_DIR"):
        load_public(settings)


def test_load_manifest_infers_catalog_flag_and_hash_split(tmp_path):
    manifest, gt, ann = _field_set(
        tmp_path,
        gt_rows=[
            ("f-1", "wine-a", "", ""),
            ("f-2", "__none__", "0", ""),
            ("f-3", "wine-a", "1", ""),
        ],
        ann=[{"query_id": "f-2", "bottle_id": "b-9", "front_label_text": []}],
    )
    items = load_manifest(manifest, gt, ann)
    assert [item.in_catalog for item in items] == [True, False, True]
    assert items[0].split == items[2].split == split_for("wine-a", "f-1")
    assert items[1].split == split_for(None, "f-2", group="b-9")
    assert items[0].image_path == tmp_path / "f-1.jpg"


def test_explicit_split_column_wins(tmp_path):
    manifest, gt, _ = _field_set(
        tmp_path, gt_rows=[("f-1", "wine-a", "1", "test"), ("f-2", "wine-b", "1", "dev")]
    )
    assert [item.split for item in load_manifest(manifest, gt)] == ["test", "dev"]


@pytest.mark.parametrize(
    ("manifest_text", "match"),
    [
        ("id\tpath\nq1\ta.jpg\n", "заголовок"),
        ("query_id\timage_path\nq1\ta.jpg\nq1\tb.jpg\n", "повтор"),
        ("query_id\timage_path\nq1\t../a.jpg\n", "небезопасный"),
        ("query_id\timage_path\nq 1\ta.jpg\n", "query_id"),
        ("query_id\timage_path\n", "нет ни одного"),
    ],
)
def test_manifest_format_is_validated(tmp_path, manifest_text, match):
    manifest = _write(tmp_path / "m.tsv", manifest_text)
    gt = _write(tmp_path / "gt.tsv", "query_id\tslug\nq1\twine-a\n")
    with pytest.raises(DatasetError, match=match):
        load_manifest(manifest, gt)


def test_manifest_requires_consistent_ground_truth(tmp_path):
    manifest, gt, _ = _field_set(
        tmp_path,
        gt_rows=[("f-1", "wine-a", "1", "")],
        manifest_rows=[("f-1", "a.jpg"), ("f-2", "b.jpg")],
    )
    with pytest.raises(DatasetError, match="нет эталона"):
        load_manifest(manifest, gt)
    manifest, gt, _ = _field_set(tmp_path, gt_rows=[("f-1", "__none__", "1", "")])
    with pytest.raises(DatasetError, match="противоречит"):
        load_manifest(manifest, gt)
    manifest, gt, ann = _field_set(
        tmp_path,
        gt_rows=[("f-1", "wine-a", "1", "")],
        ann=[{"query_id": "f-1", "target_slug": "wine-b"}],
    )
    with pytest.raises(DatasetError, match="target_slug"):
        load_manifest(manifest, gt, ann)


def test_load_synth_reads_generator_layout(tmp_path):
    anns = [
        {
            "query_id": "s-000001",
            "image_path": "aa.webp",
            "target_slug": "wine-a",
            "in_catalog": True,
        },
        {
            "query_id": "s-000002",
            "image_path": "bb.jpg",
            "target_slug": "wine-b",
            "in_catalog": True,
        },
    ]
    write_outputs(anns, tmp_path)
    items = load_synth(tmp_path)
    assert [item.slug for item in items] == ["wine-a", "wine-b"]
    assert all(item.split in ("dev", "test") for item in items)
    assert items[1].image_path == tmp_path / "bb.jpg"


def test_summarize_flags_split_conflicts_and_missing_images(tmp_path):
    manifest, gt, _ = _field_set(
        tmp_path, gt_rows=[("f-1", "wine-a", "1", "dev"), ("f-2", "wine-a", "1", "test")]
    )
    items = load_manifest(manifest, gt)
    _write(tmp_path / "f-1.jpg", "x")
    summary = summarize(items)
    assert summary["split_conflicts"] == ["wine-a"]
    assert summary["slugs"] == 1 and summary["in_catalog"] == 2
    problems = check_images(items)
    assert len(problems) == 1 and problems[0].startswith("f-2")


def test_twins_of_one_cluster_share_a_split(tmp_path):
    slugs = [f"wine-{i}" for i in range(40)]
    in_dev = next(s for s in slugs if split_for(s, "q") == "dev")
    in_test = next(s for s in slugs if split_for(s, "q") == "test")
    clusters = twin_clusters(
        {in_dev: {"cluster_B": 7}, in_test: {"cluster_B": 7}, "solo": {"cluster_B": None}}
    )
    assert clusters == {in_dev: "7", in_test: "7"}
    manifest, gt, _ = _field_set(
        tmp_path, gt_rows=[("f-1", in_dev, "1", ""), ("f-2", in_test, "1", "")]
    )
    assert {item.split for item in load_manifest(manifest, gt)} == {"dev", "test"}
    items = load_manifest(manifest, gt, clusters=clusters)
    assert len({item.split for item in items}) == 1
    assert summarize(items, clusters)["cluster_conflicts"] == []


def test_summarize_flags_twins_split_by_explicit_column(tmp_path):
    manifest, gt, _ = _field_set(
        tmp_path, gt_rows=[("f-1", "wine-a", "1", "dev"), ("f-2", "wine-b", "1", "test")]
    )
    items = load_manifest(manifest, gt)
    assert summarize(items, {"wine-a": "3", "wine-b": "3"})["cluster_conflicts"] == ["3"]
    assert summarize(items)["cluster_conflicts"] == []
