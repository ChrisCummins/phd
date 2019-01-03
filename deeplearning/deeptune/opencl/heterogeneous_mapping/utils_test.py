"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:utils."""

import pandas as pd
import pytest
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def df() -> pd.DataFrame:
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  yield dataset.df


def test_GetAtomizerFromOpenClSources_abc():
  """Test 'abc' corpus."""
  atomizer = utils.GetAtomizerFromOpenClSources(['a', 'b', 'c'])
  assert atomizer.vocab_size == 4  # a, b, c, \n


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_columns(
    df: pd.DataFrame, gpu_name: str):
  """Test that expected columns are added to dataframe."""
  df = utils.AddClassificationTargetToDataFrame(df, gpu_name)
  assert 'y' in df.columns.values
  assert 'y_1hot' in df.columns.values


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_1hot(
    df: pd.DataFrame, gpu_name: str):
  """Test that only a single value in the one hot array is set."""
  df = utils.AddClassificationTargetToDataFrame(df, gpu_name)
  for onehot in df['y_1hot'].values:
    assert sum(onehot) == 1


def test_TrainTestSplitGenerator_num_splits(df: pd.DataFrame):
  """Test that train/test splitter returns 2*10 fold splits."""
  assert len(list(utils.TrainTestSplitGenerator(df, 0))) == 20


if __name__ == '__main__':
  test.Main()
