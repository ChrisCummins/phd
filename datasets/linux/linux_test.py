"""Unit tests for //datasets/linux."""
import sys

import pytest
from absl import app
from absl import flags

from datasets.linux import linux


FLAGS = flags.FLAGS


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
  assert len(dataset.all_srcs) == 26091


def test_all_srcs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.all_srcs:
    assert path.is_file()


def test_all_hdrs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel headers."""
  assert len(dataset.all_hdrs) == 19459


def test_all_hdrs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.all_hdrs:
    assert path.is_file()


def test_kernel_srcs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel sources."""
  assert len(dataset.kernel_srcs) == 310


def test_kernel_srcs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.kernel_srcs:
    assert path.is_file()


def test_kernel_hdrs_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of kernel headers."""
  assert len(dataset.kernel_hdrs) == 64


def test_kernel_hdrs_are_files(dataset: linux.LinuxSourcesDataset):
  """Test that files exist."""
  for path in dataset.kernel_hdrs:
    assert path.is_file()


def test_include_headers_count(dataset: linux.LinuxSourcesDataset):
  """Test the number of include headers."""
  assert len(dataset.ListFiles('include', '*.h')) == 4890


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
