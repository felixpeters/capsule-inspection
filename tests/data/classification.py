import pytest

from src.data.classification import SensumClassificationDataModule

from .download import sensum_downloader


@pytest.fixture(scope="session")
def classification_data_module(sensum_downloader):
    dm = SensumClassificationDataModule(
        "softgel", sensum_downloader, batch_size=8)
    dm.prepare_data()
    dm.setup()
    return dm


def test_valid_init(sensum_downloader):
    dm = SensumClassificationDataModule("softgel", sensum_downloader)
    assert dm.batch_size == 64
    assert dm.task == "softgel"


def test_invalid_init(sensum_downloader):
    with pytest.raises(ValueError):
        dm = SensumClassificationDataModule("invalid", sensum_downloader)


def test_prepare_data(classification_data_module):
    assert classification_data_module.downloader.data_dir.exists()
    assert (classification_data_module.downloader.data_dir/"softgel").exists()


def test_setup(classification_data_module):
    assert len(classification_data_module.dls.loaders) == 2


def test_train_dataloader(classification_data_module):
    dl = classification_data_module.train_dataloader()
    x, y = dl.one_batch()
    assert x.numpy().shape == (8, 3, 144, 144)
    assert y.numpy().shape == (8,)
    assert (x.numpy() >= 0).all()
    assert (x.numpy() <= 1).all()
    assert (y.numpy() >= 0).all()
    assert (y.numpy() <= 1).all()


def test_valid_dataloader(classification_data_module):
    dl = classification_data_module.val_dataloader()
    x, y = dl.one_batch()
    assert x.numpy().shape == (8, 3, 144, 144)
    assert y.numpy().shape == (8,)
    assert (x.numpy() >= 0).all()
    assert (x.numpy() <= 1).all()
    assert (y.numpy() >= 0).all()
    assert (y.numpy() <= 1).all()
