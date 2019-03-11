"""Unit tests for :opencl_device_mapping_dataset."""
import typing

import numpy as np
import pandas as pd
import pytest

from datasets.opencl.device_mapping import \
  opencl_device_mapping_dataset as ocl_dataset
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


# Use the session scope so that cached properties on the instance are shared
# across tests.
@pytest.fixture(scope='session')
def dataset() -> ocl_dataset.OpenClDeviceMappingsDataset:
  """Test fixture which yields the dataset."""
  yield ocl_dataset.OpenClDeviceMappingsDataset()


@pytest.fixture(scope='session')
def mini_df(dataset: ocl_dataset.OpenClDeviceMappingsDataset) -> pd.DataFrame:
  """Test fixture which yields a miniature version of the full dataframe."""
  return dataset.df[:10].copy()


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


def test_ComputeGreweFeaturesForGpu_row_count(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the number of rows."""
  assert len(dataset.ComputeGreweFeaturesForGpu('amd_tahiti_7970')) == 680
  assert len(dataset.ComputeGreweFeaturesForGpu('nvidia_gtx_960')) == 680


def test_ComputeGreweFeaturesForGpu_unknown_device(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test that error is raised for unknown device."""
  with pytest.raises(KeyError):
    dataset.ComputeGreweFeaturesForGpu('not a device')


def test_grewe_features_df_columns(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the column names."""
  df = dataset.ComputeGreweFeaturesForGpu('amd_tahiti_7970')
  np.testing.assert_array_equal(df.columns.values, [
      'feature:grewe1',
      'feature:grewe2',
      'feature:grewe3',
      'feature:grewe4',
  ])


def test_df_row_count(dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the number of rows."""
  assert len(dataset.df) == 680


def test_df_columns(dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test the column names."""
  for i, col in enumerate(dataset.df.columns.values):
    print('col', i, col, dataset.df[col].values[0])
  np.testing.assert_array_equal(dataset.df.columns.values, [
      'program:benchmark_suite_name',
      'program:benchmark_name',
      'program:opencl_kernel_name',
      'program:opencl_src',
      'data:dataset_name',
      'param:amd_tahiti_7970:wgsize',
      'param:nvidia_gtx_960:wgsize',
      'feature:mem',
      'feature:comp',
      'feature:localmem',
      'feature:coalesced',
      'feature:atomic',
      'feature:rational',
      'feature:amd_tahiti_7970:transfer',
      'feature:nvidia_gtx_960:transfer',
      'runtime:intel_core_i7_3820',
      'runtime:amd_tahiti_7970',
      'runtime:nvidia_gtx_960',
  ])


def test_df_gpu_runtimes_not_equal(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset):
  """Test that the two GPU runtime columns are not equal."""
  assert not all(x == pytest.approx(y) for x, y in dataset.df[
      ['runtime:amd_tahiti_7970', 'runtime:nvidia_gtx_960']].values)


@pytest.mark.parametrize('table_getter', (
    lambda x: x.df,
    lambda x: x.ComputeGreweFeaturesForGpu('amd_tahiti_7970'),
    lambda x: x.ComputeGreweFeaturesForGpu('nvidia_gtx_960'),
    lambda x: x.AugmentWithDeadcodeMutations(
        np.random.RandomState(0), df=x.df[:10].copy()),
))
def test_df_nan(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset, table_getter: typing.
    Callable[[ocl_dataset.OpenClDeviceMappingsDataset], pd.DataFrame]):
  """Test that tables have no NaNs."""
  df = table_getter(dataset)
  assert not df.isnull().values.any()


@pytest.mark.slow(
    reason=
    "AugementWithDeadcodeMutations is slow, should switch to a smaller dataset")
def test_AugmentWithDeadcodeMutations_num_output_rows(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset, mini_df: pd.DataFrame):
  """Test the number of rows in generated table."""
  df = dataset.AugmentWithDeadcodeMutations(
      np.random.RandomState(0), num_permutations_of_kernel=3, df=mini_df)
  # the original kernel + 3 mutations
  assert len(df) == len(dataset.df) * (3 + 1)


@pytest.mark.slow(
    reason=
    "AugementWithDeadcodeMutations is slow, should switch to a smaller dataset")
def test_AugmentWithDeadcodeMutations_identical_columns(
    dataset: ocl_dataset.OpenClDeviceMappingsDataset, mini_df: pd.DataFrame):
  """Test the number of unique values in columns."""
  df = dataset.AugmentWithDeadcodeMutations(
      np.random.RandomState(0), num_permutations_of_kernel=3, df=mini_df)
  # Iterate through dataset columns, not the new dataframe's columns. The new
  # dataframe has a 'program:is_mutation' column.
  for column in dataset.df.columns.values:
    if column == 'program:opencl_src':
      # There should be more unique OpenCL kernels than we started with.
      assert len(set(df[column])) > len(set(dataset.df[column]))
    else:
      # There should be exactly the same unique cols as we started with.
      assert len(set(df[column])) == len(set(dataset.df[column]))


if __name__ == '__main__':
  test.Main()
