import pytest
from pathlib import Path

from src.data.download import URLDownloader
from src.config import URLs


@pytest.fixture(scope="session")
def sensum_downloader():
    return URLDownloader(URLs.SENSUM_SODF)


@pytest.fixture(scope="session")
def data_path():
    downloader = URLDownloader(URLs.SENSUM_SODF)
    downloader.download()
    return downloader.get_data_dir()


def test_url_downloader(sensum_downloader):
    sensum_downloader.download()
    path = sensum_downloader.get_data_dir()
    assert path.exists()
    assert (path/"softgel").exists()
    assert (path/"capsule").exists()
