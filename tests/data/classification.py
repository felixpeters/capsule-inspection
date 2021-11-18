import pytest

from src.data.classification import SensumClassificationDataModule

from .download import sensum_downloader


@pytest.fixture
def classification_data_module(sensum_downloader):
    return SensumClassificationDataModule("capsule", sensum_downloader, batch_size=8)


def test_valid_init(sensum_downloader):
    dm = SensumClassificationDataModule("capsule", sensum_downloader)
    assert dm.batch_size == 64
    assert dm.task == "capsule"


def test_invalid_init(sensum_downloader):
    with pytest.raises(ValueError):
        dm = SensumClassificationDataModule("invalid", sensum_downloader)


def test_prepare_data(classification_data_module):
    classification_data_module.prepare_data()
    assert classification_data_module.data_dir.exists()
    assert (classification_data_module.data_dir/"capsule").exists()
