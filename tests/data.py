from pathlib import Path

from src.data import download_data
from src.config import URLs


def test_download_data():
    path = download_data(URLs.SENSUM_SODF)
    assert path.exists()
    assert (path/"softgel").exists()
    assert (path/"capsule").exists()
