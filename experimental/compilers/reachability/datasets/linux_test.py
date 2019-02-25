"""Unit tests for //experimental/compilers/reachability/datasets:linux.py."""

import pytest
from absl import flags

from datasets.linux import linux
from experimental.compilers.reachability.datasets import linux
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def linux_dataset() -> linux.LinuxSourcesDataset():
  yield linux.LinuxSourcesDataset()


def test_BytecodeFromLinuxSrc_known_file():
  """Test that a known file produces bytecode."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  bytecode = linux.BytecodeFromLinuxSrc(path, '-O0')
  assert bytecode


def test_TryToCreateControlFlowGraphsFromLinuxSrc_known_file():
  """Test that a known file produces graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  # TODO(cec): No stable value.
  assert len(linux.TryToCreateControlFlowGraphsFromLinuxSrc(path)) < 20


def test_TryToCreateControlFlowGraphsFromLinuxSrc_graphs_are_valid():
  """Test that a known file produces valid graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  for graph in linux.TryToCreateControlFlowGraphsFromLinuxSrc(path):
    assert graph.IsValidControlFlowGraph(strict=False)


def test_LinuxSourcesDataset_df_count(linux_dataset: linux.LinuxSourcesDataset):
  """Test that dataset has expected number of rows."""
  # TODO(cec): This doesn't seem to be deterministic.
  assert len(linux_dataset.cfgs_df) >= 1600


if __name__ == '__main__':
  test.Main()
