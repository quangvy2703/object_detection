import logging
import os
import yaml
from typing import List

from rml.vision.object_detection.models.yolov8.ultralytics import YOLO

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
            paths: List[str] = None,
            # trains: List[str] = None,
            # vals: List[str] = None,
            # tests: List[str] = None
    ):
        def check_valid_update(
                data_config_files: List[str],
                paths: List[str] = None,
                # trains: List[str] = None,
                # vals: List[str] = None,
                # tests: List[str] = None
        ):
            num_data_config_files = len(data_config_files)
            if num_data_config_files > 1:
                if paths:
                    assert len(paths) == num_data_config_files - 1, \
                        "Multiple datasets training, num of updated paths must be equal to num of data_config_files - 1"
                # if trains:
                #     assert len(trains) == num_data_config_files - 1, "Multiple datasets training, num of updated trains must be equal to num of data_config_files - 1"
                # if vals:
                #     assert len(vals) == num_data_config_files - 1, "Multiple datasets training, num of updated vals must be equal to num of data_config_files - 1"
                # if tests:
                #     assert len(tests) == num_data_config_files - 1, "Multiple datasets training, num of updated tests must be equal to num of data_config_files - 1"

        try:
            check_valid_update(data_config_files, paths)
            for data_config, path in zip(
                    data_config_files[1:] if len(data_config_files) > 1 else data_config_files,
                    paths,
                    # trains,
                    # vals,
                    # tests
            ):
                print(data_config, path, os.path.exists(data_config))

                # if os.path.exists(data_config):
                with open(data_config, "r") as stream:
                    configs = yaml.safe_load(stream)
                    configs["path"] = path if path else configs["path"]
                    # configs["train"] = train if path else configs["train"]
                    # configs["val"] = val if path else configs["val"]
                    # configs["test"] = test if path else configs["test"]
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
