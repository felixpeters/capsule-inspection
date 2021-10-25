"""This module contains all functionality required for handling data."""
from pathlib import Path
import shutil

from py7zr import pack_7zarchive, unpack_7zarchive
from fastai.data.external import untar_data

shutil.register_archive_format(
    '7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def download_data(url: str, force_download: bool = False) -> Path:
    """Downloads and extracts dataset from given URL.

    Args:
        url (str): Archive URL
        force_download (bool): Download data even if directory already exists.

    Returns:
        Path: Path to where data was extracted
    """

    path = untar_data(url, force_download=force_download)
    return path
