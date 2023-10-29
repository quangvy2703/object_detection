import cv2
import numpy as np
from numpy import ndarray
from typing import List, Union

from sklearn import metrics

from rml.domain.label import COCOBox


class EvaluationScore:
    def __init__(self):
        pass

    @staticmethod
    def iou(ground_truths: List[COCOBox], predictions: List[COCOBox]) -> List[float]:
        n_boxes = len(ground_truths)
        ground_truth_boxes = np.array([
            [
                ground_truth.x_center - ground_truth.width / 2,
                ground_truth.y_center - ground_truth.height / 2,
                ground_truth.x_center + ground_truth.width / 2,
                ground_truth.y_center + ground_truth.width / 2,
            ]
            for ground_truth in ground_truths]
        )

        prediction_boxes = np.array([
            [
                prediction.x_center - prediction.width / 2,
                prediction.y_center - prediction.height / 2,
                prediction.x_center + prediction.width / 2,
                prediction.y_center + prediction.width / 2,
            ]
            for prediction in predictions]
        )

        rights_min = np.min(np.array([ground_truth_boxes, prediction_boxes]), axis=0)[:, 2]
        bottoms_min = np.min(np.array([ground_truth_boxes, prediction_boxes]), axis=0)[:, 3]
        lefts_max = np.max(np.array([ground_truth_boxes, prediction_boxes]), axis=0)[:, 0]
        tops_max = np.max(np.array([ground_truth_boxes, prediction_boxes]), axis=0)[:, 1]

        ground_truth_box_areas = compute_box_areas(ground_truth_boxes)
        prediction_box_areas = compute_box_areas(prediction_boxes)

        inter_areas = np.max(np.array([np.zeros(n_boxes), rights_min - lefts_max + np.ones(n_boxes)]), axis=0) \
                      * np.max(np.array([np.zeros(n_boxes), bottoms_min - tops_max + np.ones(n_boxes)]), axis=0)
        ious = inter_areas / (ground_truth_box_areas + prediction_box_areas - inter_areas)

        return ious

    @staticmethod
    def precisions(ground_truths: Union[List[str], List[int]], predictions: Union[List[str], List[int]]) -> List[float]:
        return [
            metrics.precision_score(ground_truth, prediction, average="micro")
            for ground_truth, prediction in zip(ground_truths, predictions)
        ]

    @staticmethod
    def recalls(ground_truths: Union[List[str], List[int]], predictions: Union[List[str], List[int]]) -> List[float]:
        return [
            metrics.recall_score(ground_truth, prediction, average="micro")
            for ground_truth, prediction in zip(ground_truths, predictions)
        ]

    @staticmethod
    def average_precision_score(ground_truths: List[ndarray], prediction_scores: List[ndarray]) -> List[float]:
        return [
            metrics.average_precision_score(ground_truth, prediction_score, average="micro")
            for ground_truth, prediction_score in zip(ground_truths, prediction_scores)
        ]


def compute_box_areas(bboxes: ndarray) -> ndarray:
    return (bboxes[:, 2] - bboxes[:, 0] + np.ones(bboxes.shape[0])) * (
            bboxes[:, 3] - bboxes[:, 1] + np.ones(bboxes.shape[0]))
