import pytest
from pathlib import Path

from src.data.download import URLDownloader
from src.config import URLs


@pytest.fixture
def sensum_downloader():
    return URLDownloader(URLs.SENSUM_SODF)


@pytest.fixture
def data_path():
    downloader = URLDownloader(URLs.SENSUM_SODF)
    path = downloader.download()
    return path


def test_url_downloader(sensum_downloader):
    path = sensum_downloader.download()
    assert path.exists()
    assert (path/"softgel").exists()
    assert (path/"capsule").exists()
