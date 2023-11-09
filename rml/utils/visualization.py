import os.path

import cv2
import pandas as pd
from numpy import ndarray


def draw_coco_bboxes(image_path: str, annotation_path: str, save: bool = False, show: bool = False):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[0], img.shape[1]
    bboxes = open(annotation_path, 'r').readlines()

    window_name = 'Image'

    for bbox in bboxes:
        infos = bbox.replace('\n', '').split(' ')
        infos = list(map(float, infos))
        width = int(infos[3] * img_w)
        height = int(infos[4] * img_h)
        xmin = int(infos[1] * img_w) - width // 2
        ymin = int(infos[2] * img_h) - height // 2

        start_point = (xmin, ymin)
        end_point = (xmin + width, ymin + height)

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.putText(
            img,
            str(infos[0]),
            start_point,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = (255, 255, 255),
            thickness=2
        )
    # Displaying the image
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def draw_bboxes(image_path: str, bboxes: ndarray, labels: ndarray, save: bool = False, show: bool = False):
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[0], img.shape[1]

    window_name = 'Image'

    for idx, bbox in enumerate(bboxes):
        width = int(bbox[2] * img_w)
        height = int(bbox[3] * img_h)
        xmin = int(bbox[0] * img_w) - width // 2
        ymin = int(bbox[1] * img_h) - height // 2

        start_point = (xmin, ymin)
        end_point = (xmin + width, ymin + height)

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.putText(
            img,
            str(labels[idx]),
            start_point,
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.6,
            color = (255, 255, 255),
            thickness=2
        )
    # Displaying the image
    cv2.imshow(window_name, img)
    cv2.waitKey(0)



def draw_open_image_bboxes(data_dir: str, image_id: str, split: str, save: bool = False, show: bool = False):
    img = cv2.imread(os.path.join(data_dir, split, 'data', f"{image_id}.jpg"))

    img_h, img_w = img.shape[0], img.shape[1]
    labels = pd.read_csv(
        filepath_or_buffer=os.path.join(data_dir, split, "labels", "detections.csv"),
        usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"],
        dtype=dict(zip(
            ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"],
            ["str", "str", "float16", "float16", "float16", "float16"]
        ))
    )

    bboxes = labels.loc[labels["ImageID"] == image_id]
    window_name = 'Image'

    for idx, bbox in bboxes.iterrows():
        xmin = int(bbox["XMin"] * img_w)
        xmax = int(bbox["XMax"] * img_w)
        ymin = int(bbox["YMin"] * img_h)
        ymax = int(bbox["YMax"] * img_h)

        start_point = (xmin, ymin)
        end_point = (xmax, ymax)

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

    # Displaying the image
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
