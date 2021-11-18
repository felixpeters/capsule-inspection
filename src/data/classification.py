from typing import Callable, Any
from pathlib import Path

from pytorch_lightning import LightningDataModule


class ClassificationDataModule(LightningDataModule):
    """Handles all things related to data preparation for model training."""

    def __init__(self):
        pass

    def setup(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
