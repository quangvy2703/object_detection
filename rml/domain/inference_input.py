from numpy import ndarray
from typing import List
import cv2

from rml.utils.image_downloader import ImageDownloader


class InferenceInput:
    pass


class ObjectDetectionInferenceInput(InferenceInput):
    image_downloader = ImageDownloader(n_processors=4)

    @staticmethod
    def from_ndarray(images: List[ndarray]):
        return ObjectDetectionInferenceInput(images)

    @staticmethod
    def from_paths(image_paths: List[str]):
        images = [cv2.imread(image_path) for image_path in image_paths]
        return ObjectDetectionInferenceInput(images)

    @staticmethod
    def from_urls(image_urls: List[str]):
        images_dict = ObjectDetectionInferenceInput.image_downloader.bulk_read_images(image_urls)
        images = [images_dict[image_url] for image_url in list(images_dict.keys())]
        return ObjectDetectionInferenceInput(images)

    def __init__(self, images: List[ndarray]):
        self.images: List[ndarray] = images
