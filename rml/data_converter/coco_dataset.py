from typing import List
import os
import pathlib

import cv2
import concurrent
from concurrent.futures import ThreadPoolExecutor
import yaml
import shutil
from tqdm.autonotebook import tqdm

from rml.data_converter.base import DatasetConverter
from rml.utils.dataset_utils import load_open_image_metadata, load_lvis_metadata

from rml.domain.label import OIBox, COCOBox, LVISBox


def _convert(dataset_dir: str, converted_dir: str, image_id: str, split: str, annotations: List[dict]):
    source = "train2017"
    img = cv2.imread(os.path.join(dataset_dir, "coco2017", source, f"{image_id:012d}.jpg"))
    if img is None:
        source = "val2017"
        img = cv2.imread(os.path.join(dataset_dir, "coco2017", source, f"{image_id:012d}.jpg"))
    if img is None:
        source = "test2017"
        img = cv2.imread(os.path.join(dataset_dir, "coco2017", source, f"{image_id:012d}.jpg"))
    img_w, img_h = img.shape[1], img.shape[0]
    with open(os.path.join(converted_dir, split, "labels", f"{image_id:012d}.txt"), "w") as f:
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            coco_box = COCOBox.from_lvis_box(
                lvis_box=LVISBox.from_array(annotation["bbox"]),
                image_w=img_w,
                image_h=img_h
            )
            f.write(
                f"{category_id} {coco_box.x_center} {coco_box.y_center} {coco_box.width} {coco_box.height}\n")

    if os.path.exists(os.path.join(converted_dir, split, "images", f"{image_id:012d}.jpg")):
        return True
    shutil.copy(
        src=os.path.join(dataset_dir, "coco2017", source, f"{image_id:012d}.jpg"),
        dst=os.path.join(converted_dir, split, "images", f"{image_id:012d}.jpg")
    )
    return True


class CocoDatasetConverter(DatasetConverter):
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

    @staticmethod
    def from_lvis(dataset_dir: str, converted_dir: str, classes: List[str], split: str = "train"):

        converted_split = "valid" if split == "validation" else split
        pathlib.Path(os.path.join(converted_dir, converted_split, "labels")).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(converted_dir, converted_split, "images")).mkdir(parents=True, exist_ok=True)

        classes_mapping, annotations = load_lvis_metadata(dataset_dir)
        annotation_by_image_id = {}
        bboxes, category_ids, image_ids = [], [], []
        for annotation in tqdm(annotations):
            bboxes.append(annotation["bbox"])
            category_ids.append(annotation["category_id"])
            image_ids.append(annotation["image_id"])
            if annotation["image_id"] in annotation_by_image_id:
                annotation_by_image_id[annotation["image_id"]].append(
                    {
                        "bbox": annotation["bbox"],
                        "category_id": annotation["category_id"]
                    }
                )
            else:
                annotation_by_image_id[annotation["image_id"]] = [
                    {
                        "bbox": annotation["bbox"],
                        "category_id": annotation["category_id"]
                    }
                ]

        executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=16)
        tasks = []
        for image_id, annotations in tqdm(annotation_by_image_id.items()):
            _convert(
                dataset_dir=dataset_dir,
                converted_dir=converted_dir,
                image_id=image_id,
                split=split,
                annotations=annotations
            )
        #     tasks.append(executor.submit(
        #         _convert,
        #         dataset_dir=dataset_dir,
        #         converted_dir=converted_dir,
        #         image_id=image_id,
        #         split=split,
        #         annotations=annotations
        #     ))
        # pbar = tqdm(total=len(tasks), ascii=' =')
        # for idx, future in enumerate(concurrent.futures.as_completed(tasks)):
        #     if future.result():
        #         pbar.update(idx)

