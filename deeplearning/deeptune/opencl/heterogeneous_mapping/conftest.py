"""Test fixtures for heterogeneous mapping."""
import pandas as pd
import pytest

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils


@pytest.fixture(scope='function')
def full_df() -> pd.DataFrame:
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  yield dataset.df


@pytest.fixture(scope='function')
def tiny_atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which yields an atomizer."""
  yield atomizers.AsciiCharacterAtomizer.FromText("Hello, world!")


@pytest.fixture(scope='function')
def classify_df(full_df: pd.DataFrame) -> pd.DataFrame:
  """A test fixture which yields a tiny dataset for training and prediction."""
  # Use the first 10 rows, and set classification target.
  yield utils.AddClassificationTargetToDataFrame(
      full_df.iloc[range(10), :].copy(), 'amd_tahiti_7970')


@pytest.fixture(scope='function')
def classify_df_atomizer(classify_df: pd.DataFrame) -> pd.DataFrame:
  """A test fixture which yields an atomizer for the entire dataset."""
  yield atomizers.AsciiCharacterAtomizer.FromText('\n'.join(
      classify_df['program:opencl_src'].values))


@pytest.fixture(scope='function')
def single_program_df(classify_df: pd.DataFrame) -> pd.DataFrame:
  """Test fixture which returns a single program dataframe."""
  return classify_df.iloc[[0], :].copy()
