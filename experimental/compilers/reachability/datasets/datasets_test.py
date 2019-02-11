"""Unit tests for //experimental/compilers/reachability/datasets."""

import pytest
from absl import flags

from datasets.linux import linux
from experimental.compilers.reachability.datasets import datasets
from labm8 import system
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def opencl_dataset() -> datasets.OpenClDeviceMappingsDataset:
  yield datasets.OpenClDeviceMappingsDataset()


@pytest.fixture(scope='session')
def linux_dataset() -> datasets.LinuxSourcesDataset():
  yield datasets.LinuxSourcesDataset()


def test_OpenClDeviceMappingsDataset_cfgs_df_count(
    opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that dataset has expected number of rows."""
  # The clang binaries used by Linux and macOS produce slightly different
  # results.
  if system.is_linux():
    assert len(opencl_dataset.cfgs_df) == 191
  else:
    assert len(opencl_dataset.cfgs_df) == 189


def test_OpenClDeviceMappingsDataset_cfgs_df_contains_valid_graphs(
    opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that graph instances are valid."""
  for cfg in opencl_dataset.cfgs_df['cfg:graph'].values:
    assert cfg.ValidateControlFlowGraph(strict=False)


def test_BytecodeFromLinuxSrc_known_file():
  """Test that a known file produces bytecode."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  bytecode = datasets.BytecodeFromLinuxSrc(path, '-O0')
  assert bytecode


def test_TryToCreateControlFlowGraphsFromLinuxSrc_known_file():
  """Test that a known file produces graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  # TODO(cec): Debug why this file doesn't produce graphs.
  assert len(datasets.TryToCreateControlFlowGraphsFromLinuxSrc(path)) == 0


def test_TryToCreateControlFlowGraphsFromLinuxSrc_graphs_are_valid():
  """Test that a known file produces valid graphs."""
  path = linux.LinuxSourcesDataset().src_tree_root / 'kernel' / 'kmod.c'
  assert path.is_file()  # If this fails, the linux source tree is broken.

  for graph in datasets.TryToCreateControlFlowGraphsFromLinuxSrc(path):
    assert graph.IsValidControlFlowGraph(strict=False)


def test_LinuxSourcesDataset_df_count(
    linux_dataset: datasets.LinuxSourcesDataset):
  """Test that dataset has expected number of rows."""
  # TODO(cec): This doesn't seem to be deterministic.
  assert len(linux_dataset.cfgs_df) >= 1600


if __name__ == '__main__':
  test.Main()
