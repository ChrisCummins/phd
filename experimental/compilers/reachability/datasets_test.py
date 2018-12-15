"""Unit tests for //experimental/compilers/reachability:datasets."""
import sys

import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import datasets


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def opencl_dataset() -> datasets.OpenClDeviceMappingsDataset:
  yield datasets.OpenClDeviceMappingsDataset()


def test_TODO(opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Short summary of test."""
  df = opencl_dataset.cfgs_df


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
