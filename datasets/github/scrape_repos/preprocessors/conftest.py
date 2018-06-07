"""Test fixtures for //datasets/github/scrape_repos/preprocessors."""
import pathlib
import tempfile

import pytest


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  """Test fixture for an empty temporary directory."""
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)
