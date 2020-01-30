# Copyright (c) 2017-2020 Chris Cummins.
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
"""Test fixtures for heterogeneous mapping."""
import pandas as pd

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8.py import test


@test.Fixture(scope="function")
def full_df() -> pd.DataFrame:
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  yield dataset.df


@test.Fixture(scope="function")
def tiny_atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which yields an atomizer."""
  yield atomizers.AsciiCharacterAtomizer.FromText("Hello, world!")


@test.Fixture(scope="function")
def classify_df(full_df: pd.DataFrame) -> pd.DataFrame:
  """A test fixture which yields a tiny dataset for training and prediction."""
  # Use the first 10 rows, and set classification target.
  yield utils.AddClassificationTargetToDataFrame(
    full_df.iloc[range(10), :].copy(), "amd_tahiti_7970"
  )


@test.Fixture(scope="function")
def classify_df_atomizer(classify_df: pd.DataFrame) -> pd.DataFrame:
  """A test fixture which yields an atomizer for the entire dataset."""
  yield atomizers.AsciiCharacterAtomizer.FromText(
    "\n".join(classify_df["program:opencl_src"].values)
  )


@test.Fixture(scope="function")
def single_program_df(classify_df: pd.DataFrame) -> pd.DataFrame:
  """Test fixture which returns a single program dataframe."""
  return classify_df.iloc[[0], :].copy()
