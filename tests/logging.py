import pytest

from src.logging import create_data_table
from src.data.download import URLDownloader
from src.config import URLs


@pytest.fixture
def data_path():
    dl = URLDownloader(URLs.SENSUM_SODF)
    path = dl.download()
    return path


def test_create_data_table(data_path):
    table = create_data_table(data_path/"capsule")
    assert len(table.columns) == 5
    assert len(table.data) == 989
