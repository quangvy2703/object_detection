# rml.vision.object_detection.models.yolov8.ultralytics YOLO 🚀, AGPL-3.0 license

from rml.vision.object_detection.models.yolov8.ultralytics.models.yolo import classify, detect, pose, segment

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
