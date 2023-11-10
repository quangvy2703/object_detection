import logging
import os
import yaml
from typing import List

from rml.models.vision.object_detection.yolov8.ultralytics import YOLO

from rml.domain.inference_input import ObjectDetectionInferenceInput

from rml.model_loader.base import ModelLoader


class YOLOv8ModelLoader(ModelLoader):
    @staticmethod
    def from_pretrained(model_path: str):
        return YOLOv8ModelLoader(
            pretrained_model_path=model_path
        )

    @staticmethod
    def load_training_config(train_config_path: str):
        if os.path.exists(train_config_path):
            with open(train_config_path, "r") as stream:
                training_config = yaml.safe_load(stream)
            return training_config
        else:
            logging.warning(f"Training config file {train_config_path} is not found")
            return {}

    @staticmethod
    def merge_configs(config_a: dict, config_b: dict) -> dict:
        for config in config_a.keys():
            config_a[config] = config_b.get(config, config_a[config])
        return config_a

    @staticmethod
    def update_data_config_file(
            data_config_files: List[str],
            data_dirs: List[str] = None,
            # trains: List[str] = None,
            # vals: List[str] = None,
            # tests: List[str] = None
    ):
        def check_valid_update(
                data_config_files: List[str],
                data_dirs: List[str] = None,
                # trains: List[str] = None,
                # vals: List[str] = None,
                # tests: List[str] = None
        ):
            num_data_config_files = len(data_config_files)
            if num_data_config_files > 1:
                if data_dirs:
                    assert len(data_dirs) == num_data_config_files, \
                        "Multiple datasets training, num of updated paths must be equal to num of data_config_files"
                # if trains:
                #     assert len(trains) == num_data_config_files - 1, "Multiple datasets training, num of updated trains must be equal to num of data_config_files - 1"
                # if vals:
                #     assert len(vals) == num_data_config_files - 1, "Multiple datasets training, num of updated vals must be equal to num of data_config_files - 1"
                # if tests:
                #     assert len(tests) == num_data_config_files - 1, "Multiple datasets training, num of updated tests must be equal to num of data_config_files - 1"

        try:
            check_valid_update(data_config_files, data_dirs)
            for data_config, data_dir in zip(data_config_files, data_dirs):
                with open(data_config, "r") as stream:
                    configs = yaml.safe_load(stream)
                    configs["data_dir"] = data_dir if data_dir else configs["data_dir"]
                with open(data_config, "w") as stream:
                    yaml.dump(configs, stream)
        except Exception as e:
            logging.error(f"Update config files {data_config_files} failed. Due to", exc_info=e)
            raise ValueError(f"Update config files {data_config_files} failed. Due to {str(e)}")

    def __init__(
            self,
            model_config_path: str = None,
            pretrained_model_path: str = None
    ):
        self.model_config_path: str = model_config_path
        self.pretrained_model_path: str = pretrained_model_path
        assert self.model_config_path is not None or self.pretrained_model_path is not None, \
            "model_config_path or pretrained_model_path is required"
        self.model = self._load(self.model_config_path, self.pretrained_model_path)

    def train(self, training_data_config_paths: List[str], train_configs: dict):
        # training_config = YOLOv8ModelLoader.load_training_config(train_config_path)
        self.model.train(data=training_data_config_paths, **train_configs)

    def inference(self, inference_input: ObjectDetectionInferenceInput, show: bool = False, save: bool = False):
        results = self.model.predict(source=inference_input.images, show=show, save=save)
        return results

    def validate(self):
        metrics = self.model.val()
        print(metrics)

    def export(self, format="onnx"):
        return self.model.export(format=format)

    def _load(
            self,
            model_config_path: str,
            pretrained_model_path: str
    ):
        if model_config_path:
            model = YOLO(model_config_path)
        if pretrained_model_path:
            model = YOLO(pretrained_model_path)

        return model

    def _check_train_config(self) -> bool:
        if self.model_name is None and self.pretrained_model_path is None:
            logging.error(f"Training failed. model_name or pretrained_model_path is required")
            return False

        return True


class VisionTransformerLoader(ModelLoader):
    def __init__(self):
        pass
