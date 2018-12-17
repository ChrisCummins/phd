"""Unit tests for //experimental/compilers/reachability:datasets."""
import sys

import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import cfg_datasets as datasets


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def opencl_dataset() -> datasets.OpenClDeviceMappingsDataset:
  yield datasets.OpenClDeviceMappingsDataset()


def test_OpenClDeviceMappingsDataset_cfgs_df_count(
    opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that dataset has expected number of rows."""
  assert len(opencl_dataset.cfgs_df) == 189


def test_OpenClDeviceMappingsDataset_cfgs_df_contains_valid_graphs(
    opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that graph instances are valid."""
  for cfg in opencl_dataset.cfgs_df['cfg:graph'].values:
    assert cfg.ValidateControlFlowGraph(strict=False)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
