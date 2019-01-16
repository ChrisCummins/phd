"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:utils."""

import pandas as pd
import pytest
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8 import test


FLAGS = flags.FLAGS


def test_GetAtomizerFromOpenClSources_abc():
  """Test 'abc' corpus."""
  atomizer = utils.GetAtomizerFromOpenClSources(['a', 'b', 'c'])
  assert atomizer.vocab_size == 4  # a, b, c, \n


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_columns(
    full_df: pd.DataFrame, gpu_name: str):
  """Test that expected columns are added to dataframe."""
  full_df = utils.AddClassificationTargetToDataFrame(full_df, gpu_name)
  assert 'y' in full_df.columns.values
  assert 'y_1hot' in full_df.columns.values


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_1hot(
    full_df: pd.DataFrame, gpu_name: str):
  """Test that only a single value in the one hot array is set."""
  full_df = utils.AddClassificationTargetToDataFrame(full_df, gpu_name)
  for onehot in full_df['y_1hot'].values:
    assert sum(onehot) == 1


def test_TrainTestSplitGenerator_num_splits(full_df: pd.DataFrame):
  """Test that train/test splitter returns 2*10 fold splits."""
  assert len(list(utils.TrainTestSplitGenerator(full_df, 0))) == 20


def test_TrainTestSplitGenerator_dataframe_types(full_df: pd.DataFrame):
  """Test that train/test splitter returns data frames."""
  for split in utils.TrainTestSplitGenerator(full_df, 0):
    assert isinstance(split.train_df, pd.DataFrame)
    assert isinstance(split.test_df, pd.DataFrame)


def test_TrainTestSplitGenerator_custom_split_count(full_df: pd.DataFrame):
  """Test that 2 * split_count splits is returned."""
  # The reason that twice as many splits are returned as requested is because
  # there are two devices.
  splits = utils.TrainTestSplitGenerator(full_df, 0, split_count=5)
  assert len(list(splits)) == 10


if __name__ == '__main__':
  test.Main()
