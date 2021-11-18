from pytorch_lightning import LightningDataModule

from .download import DataDownloader


class SensumClassificationDataModule(LightningDataModule):
    """Handles all things related to data preparation for training classification models."""

    def __init__(self, task: str, downloader: DataDownloader, batch_size: int = 64):
        """[summary]

        Args:
            task (str): Which dataset to use. Expects one of "capsule" or "softgel".
            downloader (DataDownloader): Downloader to use for retrieving data.
            batch_size (int, optional): Batch size to use in data loaders. Defaults to 64.

        Raises:
            ValueError: If task contains unexpected value.
        """
        super().__init__()
        self.task = task
        self.downloader = downloader
        self.batch_size = batch_size

        if self.task not in ["softgel", "capsule"]:
            raise ValueError(
                f"Task {self.task} is not supported for this dataset.")

    def prepare_data(self):
        """Retrieves data and saves path to raw data."""
        self.data_dir = self.downloader.download()

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass
