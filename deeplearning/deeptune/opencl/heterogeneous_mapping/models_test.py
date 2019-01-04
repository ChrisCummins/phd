"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:models."""
import pathlib
import typing

import numpy as np
import pandas as pd
import pytest

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping import models
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from labm8 import test


@pytest.fixture(scope='function')
def atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which yields an atomizer."""
  yield atomizers.AsciiCharacterAtomizer.FromText("Hello, world!")


@pytest.fixture(scope='session')
def df() -> pd.DataFrame:
  """A test fixture which yields a tiny dataset for training and prediction."""
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  # Use the first 10 rows, and set classification target.
  yield utils.AddClassificationTargetToDataFrame(
      dataset.df.iloc[range(10), :].copy(), 'amd_tahiti_7970')


@pytest.fixture(scope='session')
def df_atomizer(df: pd.DataFrame) -> pd.DataFrame:
  yield atomizers.AsciiCharacterAtomizer.FromText(
      '\n'.join(df['program:opencl_src'].values))


def _InstantiateModelWithTestOptions(
    model_cls: typing.Type) -> models.HeterogeneousMappingModel:
  """Instantiate a model with arguments set for testing, i.e. tiny params."""
  init_opts = {
    models.DeepTune: {
      'lstm_layer_size': 8,
      'dense_layer_size': 4,
      'num_epochs': 2,
      'batch_size': 4,
      'input_shape': (10,),
    },
    models.DeepTuneInst2Vec: {
      # Same as DeepTune, but with embedding matrix.
      'embedding_matrix': np.ones((10, 10)),
      'lstm_layer_size': 8,
      'dense_layer_size': 4,
      'num_epochs': 2,
      'batch_size': 4,
      'input_shape': (10,),
    },
  }.get(model_cls, {})

  return model_cls(**init_opts)


def test_num_models():
  """Test that the number of models. This will change"""
  assert len(models.ALL_MODELS) == 4


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_init(
    atomizer: atomizers.AsciiCharacterAtomizer, model_cls: typing.Type):
  """Test that init() can be called without error."""
  model = _InstantiateModelWithTestOptions(model_cls)
  model.init(0, atomizer)


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_save_restore(
    atomizer: atomizers.AsciiCharacterAtomizer, tempdir: pathlib.Path,
    model_cls: typing.Type):
  """Test that models can be saved and restored from file."""
  model_to_file = _InstantiateModelWithTestOptions(model_cls)
  model_to_file.init(0, atomizer)
  model_to_file.save(tempdir / 'model')

  model_from_file = _InstantiateModelWithTestOptions(model_cls)
  model_from_file.restore(tempdir / 'model')
  # We can't test that restoring the model from file actually does anything,
  # since we don't have __eq__ operator implemented for models.


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_train_predict(
    df: pd.DataFrame, df_atomizer: atomizers.AsciiCharacterAtomizer,
    model_cls: typing.Type):
  """Test that models can be trained, and used to make predictions."""
  model = _InstantiateModelWithTestOptions(model_cls)
  model.init(0, df_atomizer)
  model.train(df, 'amd_tahiti_7970')
  model.predict(df, 'amd_tahiti_7970')


if __name__ == '__main__':
  test.Main()
