from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO

from rml.models.vision.yolov8.ultralytics.utils.downloads import Path, download
from rml.models.vision.yolov8.ultralytics.utils.ops import xyxy2xywhn
from rml.models.vision.yolov8.ultralytics.utils.checks import check_requirements

check_requirements('pycocotools>=2.0')


class DatasetDownloader:
    OBJECT365 = "Object365"

    @staticmethod
    def download(root_dir: Path, dataset_name: str):
        if dataset_name == DatasetDownloader.OBJECT365:
            Object365DatasetDownloader(root_dir)


class Object365DatasetDownloader:
    def __init__(self, root_dir: Path):
        # dir = Path("/Volumes/ExtendDisk/Object365")  # dataset root dir
        for p in 'images', 'labels':
            (root_dir / p).mkdir(parents=True, exist_ok=True)
            for q in 'train', 'val':
                (root_dir / p / q).mkdir(parents=True, exist_ok=True)

        # Train, Val Splits
        for split, patches in [('train', 50 + 1), ('val', 43 + 1)]:
            print(f"Processing {split} in {patches} patches ...")
            images, labels = root_dir / 'images' / split, root_dir / 'labels' / split

            # Download
            url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
            if split == 'train':
                download([f'{url}zhiyuan_objv2_{split}.tar.gz'], dir=root_dir, delete=False)  # annotations json
                download([f'{url}patch{i}.tar.gz' for i in range(patches)], dir=images, curl=True, delete=False,
                         threads=8)
            elif split == 'val':
                download([f'{url}zhiyuan_objv2_{split}.json'], dir=root_dir, delete=False)  # annotations json
                download([f'{url}images/v1/patch{i}.tar.gz' for i in range(15 + 1)], dir=images, curl=True,
                         delete=False,
                         threads=8)
                download([f'{url}images/v2/patch{i}.tar.gz' for i in range(16, patches)], dir=images, curl=True,
                         delete=False,
                         threads=8)

            # Move
            for f in tqdm(images.rglob('*.jpg'), desc=f'Moving {split} images'):
                f.rename(images / f.name)  # move to /images/{split}

            # Labels
            coco = COCO(root_dir / f'zhiyuan_objv2_{split}.json')
            names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
            for cid, cat in enumerate(names):
                catIds = coco.getCatIds(catNms=[cat])
                imgIds = coco.getImgIds(catIds=catIds)
                for im in tqdm(coco.loadImgs(imgIds), desc=f'Class {cid + 1}/{len(names)} {cat}'):
                    width, height = im["width"], im["height"]
                    path = Path(im["file_name"])  # image filename
                    try:
                        with open(labels / path.with_suffix('.txt').name, 'a') as file:
                            annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=False)
                            for a in coco.loadAnns(annIds):
                                x, y, w, h = a['bbox']  # bounding box in xywh (xy top-left corner)
                                xyxy = np.array([x, y, x + w, y + h])[None]  # pixels(1,4)
                                x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]  # normalized and clipped
                                file.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
                    except Exception as e:
                        print(e)
