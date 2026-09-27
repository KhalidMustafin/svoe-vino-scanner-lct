import random

import numpy as np
from PIL import Image

from bench.synth import make_field_synth as synth


def _fake_bottle(path):
    """Бутылка 80x320 с альфой: узкое горлышко, тело, светлая полоса этикетки."""
    h, w = 320, 80
    arr = np.zeros((h, w, 4), np.uint8)
    for y in range(h):
        half = 10 if y < 0.3 * h else 36
        arr[y, w // 2 - half : w // 2 + half] = (30, 80, 40, 255)
    arr[int(0.55 * h) : int(0.85 * h), 4:76, :3] = (240, 235, 220)
    arr[int(0.60 * h) : int(0.63 * h), 20:60, :3] = 15
    Image.fromarray(arr, "RGBA").save(path)


def _catalog(tmp_path):
    photo = tmp_path / "bottle.png"
    _fake_bottle(photo)
    stat = {
        "bbox": [4, 0, 72, 320],
        "alpha_kind": "alpha",
        "big_components": 1,
        "bottle_aspect": 0.25,
    }
    gt = {
        "wine-a": {
            "name": "Вино А",
            "winery": "Винодельня",
            "visual_mates": ["wine-b"],
            "cluster_B": None,
        },
        "wine-b": {
            "name": "Вино Б",
            "winery": "Винодельня",
            "visual_mates": ["wine-a"],
            "cluster_B": None,
        },
    }
    return synth.Catalog(
        stats={"wine-a": stat, "wine-b": stat},
        paths={"wine-a": str(photo), "wine-b": str(photo)},
        gt=gt,
        members={},
    )


def test_make_frame_on_small_frame_is_deterministic(tmp_path):
    catalog = _catalog(tmp_path)
    ok = {"wine-a", "wine-b"}
    ann, thumb = synth.make_frame(
        "s-000001", "wine-a", 42, catalog, ok, tmp_path / "one", frame_size=(192, 256)
    )
    again, _ = synth.make_frame(
        "s-000001", "wine-a", 42, catalog, ok, tmp_path / "two", frame_size=(192, 256)
    )

    assert (tmp_path / "one" / ann["image_path"]).is_file()
    assert thumb.size == synth.THUMB_SIZE
    assert ann["target_slug"] == "wine-a" and ann["label_rows_method"] == "color_run"
    assert [n["relation"] for n in ann["neighbors"]] == ["visual_twin"]
    assert 0.0 < ann["label_area_share"] <= 1.0
    assert ann["params"]["frame_size"] == [192, 256]
    ann.pop("gen_ms"), again.pop("gen_ms")
    assert ann == again
    assert (tmp_path / "one" / ann["image_path"]).read_bytes() == (
        tmp_path / "two" / ann["image_path"]
    ).read_bytes()


def test_pick_neighbors_prefers_visual_twin():
    gt = {
        "t": {"visual_mates": ["twin"], "cluster_B": 1, "winery": "W"},
        "twin": {"visual_mates": ["t"], "cluster_B": 1, "winery": "W"},
        "mate": {"visual_mates": [], "cluster_B": 1, "winery": "W"},
        "other": {"visual_mates": [], "cluster_B": None, "winery": "X"},
    }
    ok = set(gt)
    for seed in range(20):
        picked = synth.pick_neighbors("t", random.Random(seed), gt, {1: ["t", "twin", "mate"]}, ok)
        assert picked[0] == ("twin", "visual_twin")
        assert all(slug != "t" for slug, _ in picked)


def test_pick_targets_is_seeded_and_respects_explicit_list():
    gt = {f"w{i}": {"winery": f"winery-{i % 5}", "visual_mates": [f"w{i + 1}"]} for i in range(30)}
    ok = set(gt)
    first = synth.pick_targets(gt, ok, n=4, seed=7)
    assert first == synth.pick_targets(gt, ok, n=4, seed=7)
    assert len(first) == 4 and len({gt[s]["winery"] for s in first}) == 4
    assert synth.pick_targets(gt, ok, n=2, seed=7, explicit=["w3", "w9", "w1"]) == ["w3", "w9"]


def test_write_outputs_uses_participant_manifest_format(tmp_path):
    synth.write_outputs(
        [{"query_id": "s-000001", "image_path": "ab.jpg", "target_slug": "wine-a"}], tmp_path
    )
    assert (tmp_path / "synth_manifest.tsv").read_text(
        encoding="utf-8"
    ) == "query_id\timage_path\ns-000001\tab.jpg\n"
    assert (tmp_path / "synth_gt.tsv").read_text(encoding="utf-8").splitlines()[
        1
    ] == "s-000001\twine-a\t1"
