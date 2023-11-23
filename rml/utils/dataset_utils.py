import pandas as pd
import os
from typing import List, Dict
from tqdm import tqdm

def load_open_image_metadata(open_image_dir: str, split: str):
    classes_mapping_df = pd.read_csv(
        filepath_or_buffer=os.path.join(open_image_dir, split, "metadata", "classes.csv"),
        header=None
    )
    labels = pd.read_csv(
        filepath_or_buffer=os.path.join(open_image_dir, split, "labels", "detections.csv"),
        usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"],
        dtype=dict(zip(
            ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"],
            ["str", "str", "float16", "float16", "float16", "float16"]
        ))
    )

    classes_mapping = {}
    for index, class_mapping in classes_mapping_df.iterrows():
        class_id, class_name = class_mapping[0], class_mapping[1]
        classes_mapping[class_id] = class_name

    return classes_mapping, labels


def plot_classes(data_dirs: List[str], mapping_names: List[Dict[int, int]]):
    labels = {}
    for data_idx, data_dir in enumerate(data_dirs):
        for label_file in tqdm(os.listdir(data_dir)):
            with open(os.path.join(data_dir, label_file)) as f:
                lines = f.readlines()
                for line in lines:
                    label = int(line.split(" ")[0])
                    label = mapping_names[data_idx][label]
                    if label in labels:
                        labels[label] += 1
                    else:
                        labels[label] = 0

    return labels

