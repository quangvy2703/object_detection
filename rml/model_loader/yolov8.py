import logging
import os
import yaml
from typing import List, Dict
from tqdm import tqdm

from rml.models.vision.yolov8.ultralytics import YOLO

from rml.domain.inference_input import ImageInferenceInput

from rml.model_loader.base import ModelLoader
from rml.utils.validator import ClassificationScore
from rml.evaluation_scores.score_evaluator import ScoreEvaluator


class YOLOv8ModelLoader(ModelLoader):
    CLASSIFY = "classify"
    DETECTION = "detect"

    @staticmethod
    def from_pretrained(model_path: str, task: str):
        return YOLOv8ModelLoader(
            pretrained_model_path=model_path,
            task=task
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
                data_dirs: List[str] = None
        ):
            num_data_config_files = len(data_config_files)
            if num_data_config_files > 1:
                if data_dirs:
                    assert len(data_dirs) == num_data_config_files, \
                        "Multiple datasets training, num of updated paths must be equal to num of data_config_files"

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
            pretrained_model_path: str = None,
            task: str = None
    ):
        self.model_config_path: str = model_config_path
        self.pretrained_model_path: str = pretrained_model_path
        self.task: str = task
        assert self.model_config_path is not None or self.pretrained_model_path is not None, \
            "model_config_path or pretrained_model_path is required"
        self.model = self._load(self.model_config_path, self.pretrained_model_path, task)
        assert self.model is not None, "Loaded model failed"

    def train(self, training_data_config_paths: List[str], train_configs: dict):
        self.model.train(data=training_data_config_paths, **train_configs)

    def inference(self, inference_input: ImageInferenceInput, show: bool = False, save: bool = False):
        results = self.model.predict(source=inference_input.images, show=show, save=save)
        return results

    def validate(
            self,
            validation_data_dir: str,
            names: Dict[int, str],
            mapping_ids: Dict[int, int],
            mapping_names: Dict[int, str],
    ):
        if self.task == YOLOv8ModelLoader.CLASSIFY:
            scores = self._validate_classify(
                data_dir=validation_data_dir,
                names=names,
                mapping_ids=mapping_ids,
                mapping_names=mapping_names
            )
            for label, score in scores.items():
                print(label, score.to_dict())
            return scores
        elif self.task == YOLOv8ModelLoader.DETECTION:
            scores = self.model.val(data_dir=validation_data_dir)
            print(scores)

    def export(self, format="torchscript"):
        return self.model.export(format=format)

    def _validate_classify(
            self,
            data_dir: str,
            names: Dict[int, str],
            mapping_ids: Dict[int, int],
            mapping_names: Dict[int, str],
    ) -> Dict[str, ClassificationScore]:
        reversed_names = {label: label_id for label_id, label in names.items()}
        labels = os.listdir(data_dir)
        images = {}
        for label in labels:
            if "DS_Store" in label:
                continue
            label_id = reversed_names[label]
            mapped_label_id = mapping_ids[label_id]
            if mapped_label_id == -1:
                continue
            # mapped_label = mapping_names[mapped_label_id]
            images[mapped_label_id] = [
                os.path.join(data_dir, label, image)
                for image in os.listdir(os.path.join(data_dir, label))
            ]

        scores = {}
        overall_true_labels = []
        overall_predicted_labels = []
        for label_id, label_images in images.items():
            true_labels = []
            predicted_labels = []
            for image in tqdm(label_images, desc=f"Evaluating {mapping_names[label_id]}..."):
                true_labels.append(label_id)
                overall_true_labels.append(label_id)
                result = self.inference(inference_input=ImageInferenceInput.from_paths([image]))
                predicted_labels.append(result[0].probs.top1)
                overall_predicted_labels.append(result[0].probs.top1)

            scores[mapping_names[label_id]] = ClassificationScore(
                precision=ScoreEvaluator.precisions(true_labels, predicted_labels),
                recall=ScoreEvaluator.recalls(true_labels, predicted_labels),
                accuracy=ScoreEvaluator.accuracy(true_labels, predicted_labels)
            )
        scores[ScoreEvaluator.OVERALL] = ClassificationScore(
            precision=ScoreEvaluator.precisions(overall_true_labels, overall_predicted_labels),
            recall=ScoreEvaluator.recalls(overall_true_labels, overall_predicted_labels),
            accuracy=ScoreEvaluator.accuracy(overall_true_labels, overall_predicted_labels)
        )

        return scores

    def _validate_detect(
            self,
            data_dir: str,
            names: Dict[int, str],
            mapping_ids: Dict[int, int],
            mapping_names: Dict[int, str],
    ) -> Dict[str, ClassificationScore]:
        reversed_names = {label: label_id for label_id, label in names.items()}
        labels = os.listdir(data_dir)
        images = {}
        for label in labels:
            if "DS_Store" in label:
                continue
            label_id = reversed_names[label]
            mapped_label_id = mapping_ids[label_id]
            if mapped_label_id == -1:
                continue
            # mapped_label = mapping_names[mapped_label_id]
            images[mapped_label_id] = [
                os.path.join(data_dir, label, image)
                for image in os.listdir(os.path.join(data_dir, label))
            ]

        scores = {}
        overall_true_labels = []
        overall_predicted_labels = []
        for label_id, label_images in images.items():
            true_labels = []
            predicted_labels = []
            for image in tqdm(label_images, desc=f"Evaluating {mapping_names[label_id]}..."):
                true_labels.append(label_id)
                overall_true_labels.append(label_id)
                result = self.inference(inference_input=ImageInferenceInput.from_paths([image]))
                predicted_labels.append(result[0].probs.top1)
                overall_predicted_labels.append(result[0].probs.top1)

            scores[mapping_names[label_id]] = ClassificationScore(
                precision=ScoreEvaluator.precisions(true_labels, predicted_labels),
                recall=ScoreEvaluator.recalls(true_labels, predicted_labels),
                accuracy=ScoreEvaluator.accuracy(true_labels, predicted_labels)
            )
        scores[ScoreEvaluator.OVERALL] = ClassificationScore(
            precision=ScoreEvaluator.precisions(overall_true_labels, overall_predicted_labels),
            recall=ScoreEvaluator.recalls(overall_true_labels, overall_predicted_labels),
            accuracy=ScoreEvaluator.accuracy(overall_true_labels, overall_predicted_labels)
        )

        return scores

    def _load(
            self,
            model_config_path: str,
            pretrained_model_path: str,
            task: str
    ):
        if model_config_path:
            model = YOLO(model_config_path, task)
        if pretrained_model_path:
            model = YOLO(pretrained_model_path, task)

        return model

    def _check_train_config(self) -> bool:
        if self.model_name is None and self.pretrained_model_path is None:
            logging.error(f"Training failed. model_name or pretrained_model_path is required")
            return False

        return True


class VisionTransformerLoader(ModelLoader):
    def __init__(self):
        pass
