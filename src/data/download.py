"""This module contains all functionality required for handling data."""
from pathlib import Path
import shutil
from abc import ABC, abstractmethod

from py7zr import pack_7zarchive, unpack_7zarchive
from fastai.data.external import untar_data

shutil.register_archive_format(
    '7zip', pack_7zarchive, description='7zip archive')
shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


class DataDownloader(ABC):
    """Abstract base class for data downloaders."""

    @abstractmethod
    def download(self):
        """Downloads data from given source."""

    def get_data_dir(self) -> Path:
        """Returns path to downloaded data."""


class URLDownloader(DataDownloader):
    """Retrieves data from given URL."""

    def __init__(self, url: str, force_download: bool = False):
        """Initialize downloader.

        Args:
            url (str): URL to retrieve data from.
            force_download (bool, optional): Override existing download. Defaults to False.
        """
        self.url = url
        self.force_download = force_download
        self.data_dir = None

    def download(self) -> Path:
        """Download and unpack data.

        Returns:
            Path: Path where data was extracted to.
        """
        self.data_dir = untar_data(
            self.url, force_download=self.force_download)

    def get_data_dir(self) -> Path:
        """Return path to data.

        Returns:
            Path: Path where data was extracted to.
        """
        return self.data_dir
