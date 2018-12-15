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
  """Test that the dataset has 256 rows."""
  # There are 256 unique OpenCL kernels in the dataset.
  assert len(dataset.programs_df) == 256


def test_programs_df_index_names(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the name of index columns."""
  assert list(level.name for level in dataset.programs_df.index.levels) == [
    'program:benchmark_suite_name',
    'program:benchmark_name',
    'program:opencl_kernel_name',
  ]


def test_programs_df_column_names(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the name of columns."""
  assert list(dataset.programs_df.columns.values) == [
    'program:opencl_src',
  ]


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
