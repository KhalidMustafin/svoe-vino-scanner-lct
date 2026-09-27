"""Детектор бутылок и выбор цели на кадре."""

from app.detect.bottles import (
    CENTER_BAND,
    COCO_BOTTLE,
    BottleDetector,
    Detection,
    TargetSelection,
    center_closeness,
    covers_center,
    detections_from_output,
    select_target,
    weights_path,
)

__all__ = [
    "CENTER_BAND",
    "COCO_BOTTLE",
    "BottleDetector",
    "Detection",
    "TargetSelection",
    "center_closeness",
    "covers_center",
    "detections_from_output",
    "select_target",
    "weights_path",
]
