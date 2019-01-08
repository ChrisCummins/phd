"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:models."""
import pathlib
import typing

import pandas as pd
import pytest

from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import deeptune
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from labm8 import test


def _InstantiateModelWithTestOptions(
    model_cls: typing.Type) -> base.HeterogeneousMappingModel:
  """Instantiate a model with arguments set for testing, i.e. tiny params."""
  init_opts = {
    deeptune.DeepTune: {
      'lstm_layer_size': 8,
      'dense_layer_size': 4,
      'num_epochs': 2,
      'batch_size': 4,
      'input_shape': (10,),
    },
    ncc.DeepTuneInst2Vec: {
      # Same as DeepTune.
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
  assert len(models.ALL_MODELS) == 5


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_init(
    tiny_atomizer: atomizers.AsciiCharacterAtomizer, model_cls: typing.Type):
  """Test that init() can be called without error."""
  model = _InstantiateModelWithTestOptions(model_cls)
  model.init(0, tiny_atomizer)


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_save_restore(
    tiny_atomizer: atomizers.AsciiCharacterAtomizer, tempdir: pathlib.Path,
    model_cls: typing.Type):
  """Test that models can be saved and restored from file."""
  model_to_file = _InstantiateModelWithTestOptions(model_cls)
  model_to_file.init(0, tiny_atomizer)
  model_to_file.save(tempdir / 'model')

  model_from_file = _InstantiateModelWithTestOptions(model_cls)
  model_from_file.restore(tempdir / 'model')
  # We can't test that restoring the model from file actually does anything,
  # since we don't have __eq__ operator implemented for models.


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_train_predict(
    classify_df: pd.DataFrame,
    classify_df_atomizer: atomizers.AsciiCharacterAtomizer,
    model_cls: typing.Type):
  """Test that models can be trained, and used to make predictions."""
  model = _InstantiateModelWithTestOptions(model_cls)
  model.init(0, classify_df_atomizer)
  model.train(classify_df, 'amd_tahiti_7970')
  model.predict(classify_df, 'amd_tahiti_7970')


if __name__ == '__main__':
  test.Main()
