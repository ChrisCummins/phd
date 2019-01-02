"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:utils."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils


FLAGS = flags.FLAGS


def test_GetAtomizerFromOpenClSources_abc():
  """Test 'abc' corpus."""
  atomizer = utils.GetAtomizerFromOpenClSources(['a', 'b', 'c'])
  assert atomizer.vocab_size == 4  # a, b, c, \n


def test_MakeModelInputDataFrame_ocl_dataset_columns():
  """Test that expected columns are added to dataframe."""
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  df = utils.MakeModelInputDataFrame(dataset.df, "amd_tahiti_7970")
  assert 'y' in df.columns.values
  assert 'y_1hot' in df.columns.values


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
