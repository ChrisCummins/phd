"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping:models."""
import pathlib
import typing

import pytest

from deeplearning.clgen.corpuses import atomizers
from deeplearning.deeptune.opencl.heterogeneous_mapping import models
from labm8 import test


@pytest.fixture(scope='function')
def atomizer() -> atomizers.AsciiCharacterAtomizer:
  """A test fixture which yields an atomizer."""
  yield atomizers.AsciiCharacterAtomizer.FromText("Hello, world!")


def test_num_models():
  """Test that the number of models. This will change"""
  assert len(models.ALL_MODELS) == 4


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_init(
    atomizer: atomizers.AsciiCharacterAtomizer, model_cls: typing.Type):
  """Test that init() can be called without error."""
  model = model_cls()
  model.init(0, atomizer)


@pytest.mark.parametrize('model_cls', models.ALL_MODELS)
def test_HeterogeneousMappingModel_save_restore(
    atomizer: atomizers.AsciiCharacterAtomizer, tempdir: pathlib.Path,
    model_cls: typing.Type):
  """Test that models can be saved and restored from file."""
  model_to_file = model_cls()
  model_to_file.init(0, atomizer)
  model_to_file.save(tempdir / 'model')

  model_from_file = model_cls()
  model_from_file.restore(tempdir / 'model')
  # We can't test that restoring the model from file actually does anything,
  # since we don't have __eq__ operator implemented for models.


if __name__ == '__main__':
  test.Main()
