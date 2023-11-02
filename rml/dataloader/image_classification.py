from datasets import load_dataset, concatenate_datasets
from base import Dataset

from typing import List, Tuple
from transformers import AutoImageProcessor

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor
)


class ImageClassificationDataset(Dataset):
    @staticmethod
    def from_image_pretrained(image_dirs: List[str], pretrained_path: str):
        image_processor = AutoImageProcessor.from_pretrained(pretrained_path)
        crop_size = (256, 256)
        if "height" in image_processor.size:
            size = (image_processor.size["height"], image_processor.size["width"])
            crop_size = size
        elif "shortest_edge" in image_processor.size:
            size = image_processor.size["shortest_edge"]
            crop_size = (size, size)

        return ImageClassificationDataset(
            image_dirs=image_dirs,
            image_mean=image_processor.image_mean,
            image_std=image_processor.image_std,
            crop_size=crop_size,
            image_processor=image_processor
        )

    def __init__(
            self,
            image_dirs: List[str],
            image_mean: float,
            image_std: float,
            crop_size: Tuple[int, int],
            image_processor=None
    ):
        self.dataset = concatenate_datasets(
            [load_dataset("imagefolder", data_dir=image_dir) for image_dir in image_dirs]
        )
        self.train_transforms, self.val_transforms = self.get_transform(image_mean, image_std, crop_size)

        splits = self.dataset["train"].train_test_split(test_size=0.1)
        self.train_dataset = splits['train']
        self.val_dataset = splits['test']

        self.train_dataset.set_transform(self.train_transform)
        self.val_dataset.set_transform(self.val_transform)
        self.image_processor = image_processor

    def get_transform(self, image_mean: float, image_std: float, crop_size: Tuple[int, int]):
        normalize = Normalize(mean=image_mean, std=image_std)
        train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

        val_transforms = Compose(
            [
                Resize(crop_size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

        return train_transforms, val_transforms

    def train_transform(self, example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.train_transform(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def val_transform(self, example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            self.val_transform(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch
