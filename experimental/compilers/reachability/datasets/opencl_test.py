"""Unit tests for //experimental/compilers/reachability/datasets:opencl.py."""
import pytest
from absl import flags

from experimental.compilers.reachability.datasets import opencl
from labm8 import system
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def opencl_dataset() -> opencl.OpenClDeviceMappingsDataset:
  yield opencl.OpenClDeviceMappingsDataset()


def test_OpenClDeviceMappingsDataset_cfgs_df_count(
    opencl_dataset: opencl.OpenClDeviceMappingsDataset):
  """Test that dataset has expected number of rows."""
  # The clang binaries used by Linux and macOS produce slightly different
  # results.
  if system.is_linux():
    assert len(opencl_dataset.cfgs_df) == 191
  else:
    assert len(opencl_dataset.cfgs_df) == 189


def test_OpenClDeviceMappingsDataset_cfgs_df_contains_valid_graphs(
    opencl_dataset: opencl.OpenClDeviceMappingsDataset):
  """Test that graph instances are valid."""
  for cfg in opencl_dataset.cfgs_df['cfg:graph'].values:
    assert cfg.ValidateControlFlowGraph(strict=False)


if __name__ == '__main__':
  test.Main()
