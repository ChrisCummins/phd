"""Unit tests for //deeplearning/clgen/models/builders.py."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.models import builders
from deeplearning.clgen.proto import model_pb2


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


def test_AssertIsBuildable_architecture_neurons_per_layer(abc_model_config):
  """UserError is raised if architecture.neurons_per_layer field invalid."""
  abc_model_config.architecture.ClearField('neurons_per_layer')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.neurons_per_layer must be > 0" == str(
      e_info.value)
  abc_model_config.architecture.neurons_per_layer = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.neurons_per_layer must be > 0" == str(
      e_info.value)


def test_AssertIsBuildable_architecture_num_layers(abc_model_config):
  """UserError is raised if architecture.num_layers field invalid."""
  abc_model_config.architecture.ClearField('num_layers')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.num_layers must be > 0" == str(e_info.value)
  abc_model_config.architecture.num_layers = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.num_layers must be > 0" == str(e_info.value)


def test_AssertIsBuildable_training_num_epochs(abc_model_config):
  """UserError is raised if training.num_epochs field invalid."""
  abc_model_config.training.ClearField('num_epochs')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "TrainingOptions.num_epochs must be > 0" == str(e_info.value)
  abc_model_config.training.num_epochs = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "TrainingOptions.num_epochs must be > 0" == str(e_info.value)


def test_AssertIsBuildable_training_shuffle_corpus_contentfiles_between_epochs(
    abc_model_config):
  """UserError if field not set."""
  abc_model_config.training.ClearField(
      'shuffle_corpus_contentfiles_between_epochs')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert ("Field not set: 'TrainingOptions."
          "shuffle_corpus_contentfiles_between_epochs'") == str(e_info.value)
  abc_model_config.training.shuffle_corpus_contentfiles_between_epochs = -1


def test_AssertIsBuildable_adam_optimizer_initial_learning_rate_micros(
    abc_model_config):
  """UserError is raised if initial_learning_rate_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
      'initial_learning_rate_micros')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.initial_learning_rate_micros must be >= 0" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.initial_learning_rate_micros = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.initial_learning_rate_micros must be >= 0" == str(
      e_info.value)


def test_AssertIsBuildable_adam_optimizer_learning_rate_decay_per_epoch_micros(
    abc_model_config):
  """UserError if learning_rate_decay_per_epoch_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
      'learning_rate_decay_per_epoch_micros')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert ("AdamOptimizer.learning_rate_decay_per_epoch_micros "
          "must be >= 0") == str(e_info.value)
  abc_model_config.training.adam_optimizer.learning_rate_decay_per_epoch_micros = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert ("AdamOptimizer.learning_rate_decay_per_epoch_micros "
          "must be >= 0") == str(e_info.value)


def test_AssertIsBuildable_adam_optimizer_beta_1_micros(
    abc_model_config):
  """UserError if beta_1_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
      'beta_1_micros')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.beta_1_micros = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.beta_1_micros = 1000001
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
      e_info.value)


def test_AssertIsBuildable_adam_optimizer_beta_2_micros(
    abc_model_config):
  """UserError if beta_2_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
      'beta_2_micros')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.beta_2_micros = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.beta_2_micros = 1000001
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
      e_info.value)


def test_AssertIsBuildable_adam_optimizer_normalized_gradient_clip_micros(
    abc_model_config):
  """UserError if normalized_gradient_clip_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
      'normalized_gradient_clip_micros')
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.normalized_gradient_clip_micros must be >= 0" == str(
      e_info.value)
  abc_model_config.training.adam_optimizer.normalized_gradient_clip_micros = -1
  with pytest.raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.normalized_gradient_clip_micros must be >= 0" == str(
      e_info.value)


# BuildOptimizer() tests.

def test_BuildOptimizer_adam():
  """Test AdamOptimizer proto value conversion to Keras config."""
  config = model_pb2.Model()
  config.training.ClearField('optimizer')
  config.training.adam_optimizer.initial_learning_rate_micros = 2000
  config.training.adam_optimizer.learning_rate_decay_per_epoch_micros = 5000
  config.training.adam_optimizer.beta_1_micros = 900000
  config.training.adam_optimizer.beta_2_micros = 999000
  config.training.adam_optimizer.normalized_gradient_clip_micros = 5000000
  optimizer = builders.BuildOptimizer(config)
  adam = optimizer.get_config()
  assert pytest.approx(adam['lr']) == 0.002
  assert pytest.approx(adam['decay']) == 0.005
  assert pytest.approx(adam['beta_1']) == 0.9
  assert pytest.approx(adam['beta_2']) == 0.999
  assert pytest.approx(adam['clipnorm']) == 5.0


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
