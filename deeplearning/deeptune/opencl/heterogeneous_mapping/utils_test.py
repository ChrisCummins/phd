"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:utils."""

import pytest
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8 import test


FLAGS = flags.FLAGS


def test_GetAtomizerFromOpenClSources_abc():
  """Test 'abc' corpus."""
  atomizer = utils.GetAtomizerFromOpenClSources(['a', 'b', 'c'])
  assert atomizer.vocab_size == 4  # a, b, c, \n


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_columns(gpu_name: str):
  """Test that expected columns are added to dataframe."""
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  df = utils.AddClassificationTargetToDataFrame(dataset.df, gpu_name)
  assert 'y' in df.columns.values
  assert 'y_1hot' in df.columns.values


@pytest.mark.parametrize('gpu_name', ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_1hot(gpu_name: str):
  """Test that only a single value in the one hot array is set."""
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  df = utils.AddClassificationTargetToDataFrame(dataset.df, gpu_name)
  for onehot in df['y_1hot'].values:
    assert sum(onehot) == 1


if __name__ == '__main__':
  test.Main()
