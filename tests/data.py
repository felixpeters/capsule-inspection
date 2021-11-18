from pathlib import Path

from src.data.download import download_from_url
from src.config import URLs


def test_download_data():
    path = download_from_url(URLs.SENSUM_SODF, force_download=True)
    assert path.exists()
    assert (path/"softgel").exists()
    assert (path/"capsule").exists()
