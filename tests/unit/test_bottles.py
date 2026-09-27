import os
import random
import sys

import numpy as np
import pytest

from app.detect import (
    CENTER_BAND,
    COCO_BOTTLE,
    BottleDetector,
    Detection,
    detections_from_output,
    select_target,
)
from app.reading.contracts import Box


def _det(x0: float, y0: float, x1: float, y1: float, score: float = 0.9) -> Detection:
    return Detection(box=Box(x0=x0, y0=y0, x1=x1, y1=y1), score=score)


LEFT = _det(0.02, 0.10, 0.28, 0.95, 0.95)
CENTER = _det(0.35, 0.05, 0.65, 1.00, 0.90)
RIGHT = _det(0.72, 0.10, 0.98, 0.95, 0.97)


def test_no_detections_falls_back_to_center_band():
    selection = select_target([])
    assert selection.method == "center_fallback"
    assert selection.target == CENTER_BAND
    assert selection.target.x1 - selection.target.x0 == pytest.approx(0.4)
    assert selection.neighbors == [] and selection.score == 0.0
    assert selection.confident is False


def test_single_central_bottle_is_confident():
    selection = select_target([CENTER])
    assert selection.method == "detector"
    assert selection.target == CENTER.box and selection.score == pytest.approx(0.9)
    assert selection.confident is True


def test_low_score_is_not_confident():
    selection = select_target([_det(0.35, 0.05, 0.65, 1.0, score=0.4)])
    assert selection.method == "detector" and selection.confident is False


def test_target_not_covering_center_is_not_confident():
    selection = select_target([_det(0.55, 0.1, 0.95, 0.9, score=0.99)])
    assert selection.method == "detector" and selection.confident is False


def test_center_bottle_wins_over_two_neighbors():
    selection = select_target([RIGHT, CENTER, LEFT])
    assert selection.target == CENTER.box
    assert selection.neighbors == [LEFT.box, RIGHT.box]
    assert selection.confident is True


def test_selection_does_not_depend_on_input_order():
    detections = [LEFT, CENTER, RIGHT]
    expected = select_target(detections)
    rng = random.Random(0)
    for _ in range(5):
        rng.shuffle(detections)
        assert select_target(detections) == expected


def test_center_weight_trades_area_for_centrality():
    big_side = _det(0.55, 0.0, 1.0, 1.0)
    small_center = _det(0.4, 0.2, 0.6, 0.8)
    assert select_target([big_side, small_center], center_weight=0.0).target == big_side.box
    assert select_target([big_side, small_center], center_weight=4.0).target == small_center.box
    with pytest.raises(ValueError):
        select_target([big_side], center_weight=-1.0)


def test_box_inside_target_is_not_a_neighbor():
    label_part = _det(0.38, 0.45, 0.62, 0.9, score=0.5)
    selection = select_target([CENTER, label_part, RIGHT])
    assert selection.target == CENTER.box
    assert selection.neighbors == [RIGHT.box]


def test_detections_from_output_filters_and_normalizes():
    boxes = [
        [100, 50, 300, 390],  # бутылка
        [0, 0, 400, 400],  # не бутылка
        [10, 10, 60, 390],  # бутылка с низкой уверенностью
        [350, 20, 420, 380],  # бутылка за правым краем кадра
        [0, 0, 20, 20],  # мелкая бутылка на дальней полке
        [200, 100, 200, 300],  # вырожденная рамка
    ]
    labels = [COCO_BOTTLE, 1, COCO_BOTTLE, COCO_BOTTLE, COCO_BOTTLE, COCO_BOTTLE]
    scores = [0.8, 0.99, 0.2, 0.95, 0.9, 0.9]
    found = detections_from_output(np.array(boxes), labels, scores, 400, 400)
    assert [d.score for d in found] == pytest.approx([0.95, 0.8])
    assert found[0].box.x1 == 1.0
    assert found[1].box == Box(x0=0.25, y0=0.125, x1=0.75, y1=0.975)


def test_detections_from_output_respects_thresholds():
    boxes, labels, scores = [[0, 0, 40, 40]], [COCO_BOTTLE], [0.5]
    assert detections_from_output(boxes, labels, scores, 100, 100) != []
    assert detections_from_output(boxes, labels, scores, 100, 100, min_area=0.2) == []
    assert detections_from_output(boxes, labels, scores, 100, 100, score_thresh=0.6) == []


def test_detector_without_cached_weights_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    monkeypatch.setenv("SVS_DEVICE", "cpu")
    if "torch" in sys.modules:  # кэш torch.hub мог быть переназначен другим тестом
        monkeypatch.setattr(sys.modules["torch"].hub, "_hub_dir", None, raising=False)
    detector = BottleDetector()
    assert detector.requested_device == "cpu"
    assert detector.available() is False
    assert detector.detect(np.zeros((64, 48, 3), dtype=np.uint8)) == []


def test_detect_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        BottleDetector(device="cpu").detect(np.zeros((64, 48), dtype=np.uint8))


@pytest.mark.skipif(
    os.environ.get("SVS_RUN_GPU_TESTS") != "1", reason="нужна модель: SVS_RUN_GPU_TESTS=1"
)
def test_detector_finds_bottles_on_public_frames():
    from app.config import get_settings
    from app.normalize import decode_image

    queries = get_settings().dataset_dir / "eval" / "queries"
    frames = sorted(p for p in queries.glob("*") if p.is_file()) if queries.is_dir() else []
    if not frames:
        pytest.skip(f"нет публичных кадров в {queries}")
    detector = BottleDetector()
    if not detector.available():
        pytest.skip("детектор не поднялся: нет torchvision или весов")
    found_any = False
    for path in frames:
        detections = detector.detect(decode_image(path.read_bytes()))
        assert all(isinstance(d, Detection) for d in detections)
        found_any = found_any or bool(detections)
        selection = select_target(detections)
        assert selection.method == ("detector" if detections else "center_fallback")
    assert found_any
