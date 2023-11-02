from typing import List
import os
import pathlib
import pandas as pd
import yaml
import shutil
from tqdm import tqdm

from rml.data_converter.base import DatasetConverter
from rml.utils.dataset_utils import load_open_image_metadata

from rml.domain.label import OIBox, COCOBox


class CocoaDatasetConverter(DatasetConverter):
    def __init__(
            self,
            converted_dir: str,
            classes: List[str]
    ):
        self.converted_dir: str = converted_dir
        self.classes: List[str] = classes

    @staticmethod
    def from_open_image(open_image_dir: str, converted_dir: str, classes: List[str], split: str = "train"):
        converted_split = "valid" if split == "validation" else split
        pathlib.Path(os.path.join(converted_dir, "labels", converted_split)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(converted_dir, "images", converted_split)).mkdir(parents=True, exist_ok=True)

        classes_mapping, labels = load_open_image_metadata(open_image_dir, split)
        class_name_to_class_id = dict([(_class, idx) for idx, _class in enumerate(classes)])

        for image_id, labeled_boxes in tqdm(labels.groupby(["ImageID"])):
            image_id = image_id[0]
            lines = []
            for index, box in labeled_boxes.iterrows():
                label_name = classes_mapping.get(box["LabelName"], "").strip()
                if label_name in classes:
                    label_id = class_name_to_class_id.get(label_name, None)
                    coco_box = COCOBox.from_oi_box(OIBox.from_dict(box))
                    lines.append(
                        f"{label_id} {coco_box.x_center} {coco_box.y_center} {coco_box.width} {coco_box.height}\n")
            if len(lines) > 0:
                with open(os.path.join(converted_dir, "labels", converted_split, f"{image_id}.txt"), "w") as f:
                    f.writelines(lines)
                shutil.copy(
                    os.path.join(open_image_dir, split, "data", f"{image_id}.jpg"),
                    os.path.join(converted_dir, "images", converted_split, f"{image_id}.jpg")
                )

        with open(f'{converted_dir}/dataset.yaml', 'w') as file:
            yaml.dump(
                {
                    "path": converted_dir,
                    "train": "images/train",
                    "val": "images/valid",
                    "names": dict([(label_id, label_name) for label_name, label_id in class_name_to_class_id.items()]),
                },
                file
            )

    @staticmethod
    def from_rever_image(open_image_dir: str, converted_dir: str, classes: List[str], split: str = "train"):
        converted_split = "valid" if split == "validation" else split
        pathlib.Path(os.path.join(converted_dir, "labels", converted_split)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(converted_dir, "images", converted_split)).mkdir(parents=True, exist_ok=True)

        classes_mapping, labels = load_open_image_metadata(open_image_dir, split)
        class_name_to_class_id = dict([(_class, idx) for idx, _class in enumerate(classes)])

        for image_id, labeled_boxes in tqdm(labels.groupby(["ImageID"])):
            image_id = image_id[0]
            lines = []
            for index, box in labeled_boxes.iterrows():
                label_name = classes_mapping.get(box["LabelName"], "").strip()
                if label_name in classes:
                    label_id = class_name_to_class_id.get(label_name, None)
                    coco_box = COCOBox.from_oi_box(OIBox.from_dict(box))
                    lines.append(
                        f"{label_id} {coco_box.x_center} {coco_box.y_center} {coco_box.width} {coco_box.height}\n")
            if len(lines) > 0:
                with open(os.path.join(converted_dir, "labels", converted_split, f"{image_id}.txt"), "w") as f:
                    f.writelines(lines)
                shutil.copy(
                    os.path.join(open_image_dir, split, "data", f"{image_id}.jpg"),
                    os.path.join(converted_dir, "images", converted_split, f"{image_id}.jpg")
                )
