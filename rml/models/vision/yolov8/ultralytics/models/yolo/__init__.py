# rml.vision.object_detection.models.yolov8.ultralytics YOLO 🚀, AGPL-3.0 license

from rml.models.vision.yolov8.ultralytics.models.yolo import classify, detect, pose
from . import segment

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'pose', 'YOLO'
