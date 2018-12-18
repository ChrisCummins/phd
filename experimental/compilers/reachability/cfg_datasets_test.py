"""Unit tests for //experimental/compilers/reachability:datasets."""
import sys

import pytest
from absl import app
from absl import flags

from datasets.linux import linux
from experimental.compilers.reachability import cfg_datasets as datasets
from labm8 import system


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

  bytecode = datasets.BytecodeFromLinuxSrc(path)
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
