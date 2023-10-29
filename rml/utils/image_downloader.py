import logging
import cv2
import numpy as np
import requests
from numpy import ndarray
from typing import List, Dict, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import urllib.parse
from urllib.parse import urlparse, urlunparse


def encode_url(url: str):
    base_url_parse = urlparse(url)
    base_path = base_url_parse.path
    new_path = urllib.parse.quote(base_path)
    new_url_parse = base_url_parse._replace(path=new_path)
    url = urlunparse(new_url_parse)
    return url

def _read_from_local(path: str) -> Tuple[str, ndarray]:
    return path, cv2.imread(path)


def _read_from_url(url: str) -> Tuple[str, ndarray]:
    image = None
    raw = None
    response = None
    try:
        url_encoded = encode_url(url)
        response = requests.get(url_encoded, cookies={'ssid_admin': 'abcd'})
        # response = requests.get(url_encoded, verify=False)
        raw = bytearray(response.content)
        image = np.array(raw, dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        logging.info(f"Unable to read image from {url}", exc_info=e)
    # finally:
    #     del raw
    #     del response

    return url, image


class ImageDownloader:
    def __init__(self, n_processors: int) -> None:
        self.executor = ThreadPoolExecutor(max_workers=n_processors)

    def bulk_read_images(self, urls: List[str]) -> Dict[str, ndarray]:
        is_url = True if urls[0][:4] == 'http' else False
        return self._bulk_read_from_url(urls) if is_url else self._bulk_read_from_local(urls)

    def _bulk_read_from_url(self, urls: List[str]) -> Dict[str, ndarray]:
        tasks = []
        for url in urls:
            tasks.append(self.executor.submit(_read_from_url, url=url))

        loaded_images = {}
        for task in concurrent.futures.as_completed(tasks):
            url, image = task.result()
            if image is not None:
                loaded_images[url] = image

        return loaded_images

    def _bulk_read_from_local(self, paths: List[str]) -> Dict[str, ndarray]:
        tasks = []
        for path in paths:
            tasks.append(self.executor.submit(_read_from_local, path=path))

        images = {}
        for task in concurrent.futures.as_completed(tasks):
            path, image = task.result()
            if image is not None:
                images[path] = image
        return images
