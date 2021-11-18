from typing import Optional
from pathlib import Path

from pytorch_lightning import LightningDataModule
from fastai.data.transforms import get_image_files
from fastai.vision.data import ImageDataLoaders

from .download import DataDownloader


class SensumClassificationDataModule(LightningDataModule):
    """Handles all things related to data preparation for training classification models."""

    def __init__(self, task: str, downloader: DataDownloader, batch_size: int = 64, val_perc: float = 0.2, seed: int = 47):
        """[summary]

        Args:
            task (str): Which dataset to use. Expects one of "capsule" or "softgel".
            downloader (DataDownloader): Downloader to use for retrieving data.
            batch_size (int, optional): Batch size to use in data loaders. Defaults to 64.
            val_perc (float, optional): Percentage of data to be used for validation. Defaults to 0.2.
            seed (int, optional): Random seed to use for splitting data. Defaults to 47.

        Raises:
            ValueError: If task contains unexpected value.
        """
        super().__init__()
        self.task = task
        self.downloader = downloader
        self.batch_size = batch_size
        self.val_perc = val_perc
        self.seed = seed

        if self.task not in ["softgel", "capsule"]:
            raise ValueError(
                f"Task {self.task} is not supported for this dataset.")

    def prepare_data(self):
        """Retrieves data and saves path to raw data."""
        self.downloader.download()

    def setup(self, stage: Optional[str] = None):
        """Collects image files and creates data loaders.

        Args:
            stage (Optional[str], optional): Stage to be used in Trainer. Defaults to None.
        """
        if stage in (None, "fit"):
            self.data_dir = self.downloader.get_data_dir()
            pos_fnames = get_image_files(self.data_dir/"softgel/positive/data")
            neg_fnames = get_image_files(self.data_dir/"softgel/negative/data")
            fnames = pos_fnames + neg_fnames

            def label_func(fn: Path) -> bool:
                image_class = str(fn).split("/")[-3]
                return image_class == "positive"

            self.dls = ImageDataLoaders.from_path_func(
                self.data_dir, fnames, label_func, bs=self.batch_size, seed=self.seed)

    def train_dataloader(self):
        return self.dls.train

    def val_dataloader(self):
        return self.dls.valid

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
