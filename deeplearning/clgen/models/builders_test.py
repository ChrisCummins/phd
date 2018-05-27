"""Unit tests for //deeplearning/clgen/models/builders.py."""
import pytest
import sys
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.models import builders


FLAGS = flags.FLAGS


def test_AssertIsBuildable_returns_config(abc_model_config):
  """Test that the original config is returned."""
  assert abc_model_config == builders.AssertBuildable(abc_model_config)


def test_AssertIsBuildable_missing_neuron_type_field(abc_model_config):
  """Test that a UserError is raided if neuron_type field not set."""
  abc_model_config.architecture.ClearField('neuron_type')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertBuildable(abc_model_config)
  assert str(e_info).endswith('Model.architecture.neuron_type field not set')


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
