# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:utils."""
import numpy as np
import pandas as pd
import pytest

from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_GetAtomizerFromOpenClSources_abc():
  """Test 'abc' corpus."""
  atomizer = utils.GetAtomizerFromOpenClSources(["a", "b", "c"])
  assert atomizer.vocab_size == 4  # a, b, c, \n


@test.Parametrize("gpu_name", ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_columns(
  full_df: pd.DataFrame, gpu_name: str
):
  """Test that expected columns are added to dataframe."""
  full_df = utils.AddClassificationTargetToDataFrame(full_df, gpu_name)
  assert "target_gpu_name" in full_df.columns.values
  assert "y" in full_df.columns.values
  assert "y_1hot" in full_df.columns.values


@test.Parametrize("gpu_name", ("amd_tahiti_7970", "nvidia_gtx_960",))
def test_AddClassificationTargetToDataFrame_ocl_dataset_1hot(
  full_df: pd.DataFrame, gpu_name: str
):
  """Test that only a single value in the one hot array is set."""
  full_df = utils.AddClassificationTargetToDataFrame(full_df, gpu_name)
  for onehot in full_df["y_1hot"].values:
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


def test_TrainValidationTestSplits_num_splits(full_df: pd.DataFrame):
  """Train/val/test splitter returns 2 splits."""
  assert (
    len(
      list(utils.TrainValidationTestSplits(full_df, np.random.RandomState(0)))
    )
    == 2
  )


def test_TrainValidationTestSplits_table_sizes(full_df: pd.DataFrame):
  """Train/va/test splits have expected element counts."""
  splits = list(
    utils.TrainValidationTestSplits(full_df, np.random.RandomState(0))
  )
  assert len(splits[0].train_df) == 408
  assert len(splits[0].valid_df) == 202
  assert len(splits[0].test_df) == 70

  assert len(splits[1].train_df) == 407
  assert len(splits[1].valid_df) == 204
  assert len(splits[1].test_df) == 69


if __name__ == "__main__":
  test.Main()
