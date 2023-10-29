import pandas as pd
import os


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

