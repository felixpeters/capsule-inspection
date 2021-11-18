from pathlib import Path

from src.data.download import URLDownloader
from src.config import URLs


def test_url_downloader():
    downloader = URLDownloader(URLs.SENSUM_SODF, force_download=True)
    path = downloader.download()
    assert path.exists()
    assert (path/"softgel").exists()
    assert (path/"capsule").exists()
