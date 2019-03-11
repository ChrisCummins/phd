"""Unit tests for //datasets/linux."""

import pytest

from datasets.linux import linux
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.fixture(scope='session')
def dataset() -> linux.LinuxSourcesDataset:
  yield linux.LinuxSourcesDataset()


# The following tests are all hardcoded to the current version of @linux_srcs
# that is defined the WORKSPACE file. Changing the linux version will require
# updating these tests.


def test_version(dataset: linux.LinuxSourcesDataset):
  assert dataset.version == '4.19'


def test_known_file_locations(dataset: linux.LinuxSourcesDataset):
  """Test that known files exist."""
  assert (dataset.src_tree_root / 'kernel' / 'kexec.c').is_file()
  assert (dataset.src_tree_root / 'kernel' / 'smpboot.h').is_file()


def test_all_srcs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel sources."""
  # FIXME(cec): This value does not appear to stable across platforms, but it
  # should be.
  assert abs(len(dataset.all_srcs) - 26091) < 1000


def test_all_srcs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.all_srcs:
    assert path.is_file()


def test_all_hdrs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel headers."""
  # FIXME(cec): This value does not appear to stable across platforms, but it
  # should be.
  assert abs(len(dataset.all_hdrs) - 19459) < 1000


def test_all_hdrs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.all_hdrs:
    assert path.is_file()


def test_kernel_srcs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel sources."""
  # FIXME(cec): This value does not appear to stable across platforms, but it
  # should be.
  assert abs(len(dataset.kernel_srcs) - 310) < 10


def test_kernel_srcs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.kernel_srcs:
    assert path.is_file()


def test_kernel_hdrs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel headers."""
  # FIXME(cec): This value does not appear to stable across platforms, but it
  # should be.
  assert abs(len(dataset.kernel_hdrs) - 64) < 10


def test_kernel_hdrs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.kernel_hdrs:
    assert path.is_file()


def test_include_headers_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of include headers."""
  # FIXME(cec): This value does not appear to stable across platforms, but it
  # should be.
  assert abs(len(dataset.ListFiles('include', '*.h')) - 4890) < 100


if __name__ == '__main__':
  test.Main()
