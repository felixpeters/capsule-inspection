import pytest

from src.logging import create_data_table

from .data.download import data_path


def test_create_data_table(data_path):
    table = create_data_table(data_path/"capsule")
    assert len(table.columns) == 5
    assert len(table.data) == 989
