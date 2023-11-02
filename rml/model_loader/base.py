import logging
import os
import yaml
from typing import List

from rml.models.vision.object_detection.yolov8.ultralytics import YOLO

from rml.domain.inference_input import ObjectDetectionInferenceInput


class ModelLoader:
    def train(self, data_config_path: str, train_config_path: str):
        pass

    def inference(self, inference_input: ObjectDetectionInferenceInput):
        pass
