import logging
import os
import yaml
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from datasets import load_metric

from rml.domain.inference_input import ObjectDetectionInferenceInput

from rml.model_loader.base import ModelLoader
from rml.dataloader.image_classification import ImageClassificationDataset


class VisionTransformerModelLoader(ModelLoader):
    @staticmethod
    def from_pretrained(model_path: str, label_names: List[str]):
        return VisionTransformerModelLoader(
            model_path=model_path,
            label_names=label_names
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

    def __init__(
            self,
            model_path: str,
            label_names: List[str]
    ):
        self.model_path: str = model_path
        self.label_names: List[str] = label_names
        self.model = self._load(self.model_path, self.label_names)

    def train(self, image_dirs: List[str], train_configs: dict):
        data_loader = ImageClassificationDataset.from_image_pretrained(
            image_dirs=image_dirs,
            pretrained_path=train_configs.get('pretrained_model_path', None)
        )

        train_args = TrainingArguments(
            output_dir=train_configs["output_dir"],
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            # learning_rate=5e-5,
            learning_rate=train_configs["lr"],
            per_device_train_batch_size=train_configs["batch_size"],
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=train_configs["batch_size"],
            num_train_epochs=train_configs["epochs"],
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        trainer = Trainer(
            self.model,
            train_args,
            train_dataset=data_loader.train_dataset,
            eval_dataset=data_loader.val_dataset,
            tokenizer=data_loader.image_processor,
            compute_metrics=self._compute_metrics,
            data_collator=self._collate_fn,
        )

        train_results = trainer.train()
        # rest is optional but nice to have
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

    def inference(self, inference_input: ObjectDetectionInferenceInput, show: bool = False, save: bool = False):
        results = self.model.predict(source=inference_input.images, show=show, save=save)
        return results

    def _load(self, pretrained_model_path: str, label_names: List[str]):
        label2id, id2label = self._ids_from_labels(label_names)
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_path,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        return model

    def _ids_from_labels(self, label_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        label2id, id2label = dict(), dict()
        for i, label in enumerate(label_names):
            label2id[label] = i
            id2label[i] = label
        return label2id, id2label

    def _collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def _compute_metrics(self, eval_pred):
        metric = load_metric("accuracy")
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


class VisionTransformerLoader(ModelLoader):
    def __init__(self):
        pass
