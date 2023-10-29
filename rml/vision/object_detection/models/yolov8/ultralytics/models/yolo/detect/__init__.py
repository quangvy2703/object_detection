# rml.vision.object_detection.models.yolov8.ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

__all__ = 'DetectionPredictor', 'DetectionTrainer', 'DetectionValidator'
