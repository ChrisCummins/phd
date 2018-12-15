"""Unit tests for :opencl_device_mapping_dataset."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from datasets.opencl.device_mapping import \
  opencl_device_mapping_dataset as ocl_dataset


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def dataset() -> ocl_dataset.OpenClDeviceMappingsDataset:
  yield ocl_dataset.OpenClDeviceMappingsDataset()


def test_programs_df_row_count(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Short summary of test."""
  # TODO
  assert len(dataset.programs_df) == 1


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
