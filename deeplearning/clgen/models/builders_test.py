"""Unit tests for //deeplearning/clgen/models/builders.py."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.models import builders


FLAGS = flags.FLAGS


# AssertIsBuildable() tests.

def test_AssertIsBuildable_returns_config(abc_model_config):
  """Test that the original config is returned."""
  assert abc_model_config == builders.AssertIsBuildable(abc_model_config)


def test_AssertIsBuildable_no_corpus(abc_model_config):
  """Test that UserError is raised if corpus field not set."""
  abc_model_config.ClearField('corpus')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.corpus'" == str(e_info.value)


def test_AssertIsBuildable_no_architecture(abc_model_config):
  """Test that UserError is raised if architecture field not set."""
  abc_model_config.ClearField('architecture')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.architecture'" == str(e_info.value)


def test_AssertIsBuildable_no_training(abc_model_config):
  """Test that UserError is raised if training field not set."""
  abc_model_config.ClearField('training')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.training'" == str(e_info.value)


def test_AssertIsBuildable_architecture_neuron_type(abc_model_config):
  """UserError is raised if architecture.neuron_type field not set."""
  abc_model_config.architecture.ClearField('neuron_type')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'NetworkArchitecture.neuron_type'" == str(e_info.value)


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
