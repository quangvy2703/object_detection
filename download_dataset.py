from rml.utils.dataset_downloader.dataset_downloader import DatasetDownloader
from pathlib import Path


root_dir = Path("/Volumes/ExtendDisk/Object365")
DatasetDownloader.download(
    root_dir=root_dir,
    dataset_name=DatasetDownloader.OBJECT365
)
