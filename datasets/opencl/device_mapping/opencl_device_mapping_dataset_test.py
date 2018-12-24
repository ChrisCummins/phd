"""Unit tests for :opencl_device_mapping_dataset."""
import sys
import typing

import numpy as np
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


def test_grewe_features_df_row_count(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the number of rows."""
  assert len(dataset.grewe_features_df) == 680


def test_grewe_features_df_columns(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the column names."""
  np.testing.assert_array_equal(dataset.grewe_features_df.columns.values, [
    'feature:grewe1',
    'feature:grewe2',
    'feature:grewe3',
    'feature:grewe4',
  ])


def test_df_row_count(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the number of rows."""
  assert len(dataset.df) == 680


def test_df_columns(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the column names."""
  for i, col in enumerate(dataset.df.columns.values):
    print('col', i, col, dataset.df[col].values[0])
  np.testing.assert_array_equal(dataset.df.columns.values, [
    'program:benchmark_suite_name',
    'program:benchmark_name',
    'program:opencl_kernel_name',
    'program:opencl_src',
    'data:dataset_name',
    'param:wgsize',
    'feature:mem',
    'feature:comp',
    'feature:localmem',
    'feature:coalesced',
    'feature:transfer',
    'feature:atomic',
    'feature:rational',
    'runtime:intel_core_i7_3820',
    'runtime:amd_tahiti_7970',
    'runtime:nvidia_gtx_960',
  ])


@pytest.mark.parametrize('property_name', ('df', 'grewe_features_df'))
def test_df_nan(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset, property_name: str):
  """Test that tables have no NaNs."""
  df = getattr(dataset, property_name)
  assert not df.isnull().values.any()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
