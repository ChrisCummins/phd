# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/clgen/models/builders.py."""
import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.models import builders
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.clgen.tests.fixtures"]

# AssertIsBuildable() tests.


def test_AssertIsBuildable_returns_config(abc_model_config):
  """Test that the original config is returned."""
  assert abc_model_config == builders.AssertIsBuildable(abc_model_config)


def test_AssertIsBuildable_no_corpus(abc_model_config):
  """Test that UserError is raised if corpus field not set."""
  abc_model_config.ClearField("corpus")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.corpus'" == str(e_info.value)


def test_AssertIsBuildable_no_architecture(abc_model_config):
  """Test that UserError is raised if architecture field not set."""
  abc_model_config.ClearField("architecture")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.architecture'" == str(e_info.value)


def test_AssertIsBuildable_no_training(abc_model_config):
  """Test that UserError is raised if training field not set."""
  abc_model_config.ClearField("training")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'Model.training'" == str(e_info.value)


def test_AssertIsBuildable_architecture_embedding_size(abc_model_config):
  """UserError is raised if architecture.embedding_size field invalid."""
  # embedding_size is ignored unless backend == KERAS.
  abc_model_config.architecture.backend = model_pb2.NetworkArchitecture.KERAS
  abc_model_config.architecture.ClearField("embedding_size")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.embedding_size must be > 0" == str(e_info.value)
  abc_model_config.architecture.embedding_size = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.embedding_size must be > 0" == str(e_info.value)


def test_AssertIsBuildable_architecture_neuron_type(abc_model_config):
  """UserError is raised if architecture.neuron_type field not set."""
  abc_model_config.architecture.ClearField("neuron_type")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "Field not set: 'NetworkArchitecture.neuron_type'" == str(e_info.value)


def test_AssertIsBuildable_architecture_neurons_per_layer(abc_model_config):
  """UserError is raised if architecture.neurons_per_layer field invalid."""
  abc_model_config.architecture.ClearField("neurons_per_layer")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.neurons_per_layer must be > 0" == str(
    e_info.value
  )
  abc_model_config.architecture.neurons_per_layer = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.neurons_per_layer must be > 0" == str(
    e_info.value
  )


def test_AssertIsBuildable_architecture_num_layers(abc_model_config):
  """UserError is raised if architecture.num_layers field invalid."""
  abc_model_config.architecture.ClearField("num_layers")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.num_layers must be > 0" == str(e_info.value)
  abc_model_config.architecture.num_layers = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "NetworkArchitecture.num_layers must be > 0" == str(e_info.value)


def test_AssertIsBuildable_architecture_post_layer_dropout_micros(
  abc_model_config,
):
  """UserError raised for invalid architecture.post_layer_dropout_micros."""
  abc_model_config.architecture.ClearField("post_layer_dropout_micros")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "NetworkArchitecture.post_layer_dropout_micros must be "
    ">= 0 and <= 1000000"
  ) == str(e_info.value)
  abc_model_config.architecture.post_layer_dropout_micros = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "NetworkArchitecture.post_layer_dropout_micros must be "
    ">= 0 and <= 1000000"
  ) == str(e_info.value)
  abc_model_config.architecture.post_layer_dropout_micros = 1000001
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "NetworkArchitecture.post_layer_dropout_micros must be "
    ">= 0 and <= 1000000"
  ) == str(e_info.value)


def test_AssertIsBuildable_training_num_epochs(abc_model_config):
  """UserError is raised if training.num_epochs field invalid."""
  abc_model_config.training.ClearField("num_epochs")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "TrainingOptions.num_epochs must be > 0" == str(e_info.value)
  abc_model_config.training.num_epochs = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "TrainingOptions.num_epochs must be > 0" == str(e_info.value)


def test_AssertIsBuildable_training_shuffle_corpus_contentfiles_between_epochs(
  abc_model_config,
):
  """UserError if field not set."""
  abc_model_config.training.ClearField(
    "shuffle_corpus_contentfiles_between_epochs"
  )
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "Field not set: 'TrainingOptions."
    "shuffle_corpus_contentfiles_between_epochs'"
  ) == str(e_info.value)
  abc_model_config.training.shuffle_corpus_contentfiles_between_epochs = -1


def test_AssertIsBuildable_adam_optimizer_initial_learning_rate_micros(
  abc_model_config,
):
  """UserError is raised if initial_learning_rate_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
    "initial_learning_rate_micros"
  )
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.initial_learning_rate_micros must be >= 0" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.initial_learning_rate_micros = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.initial_learning_rate_micros must be >= 0" == str(
    e_info.value
  )


def test_AssertIsBuildable_adam_optimizer_learning_rate_decay_per_epoch_micros(
  abc_model_config,
):
  """UserError if learning_rate_decay_per_epoch_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
    "learning_rate_decay_per_epoch_micros"
  )
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "AdamOptimizer.learning_rate_decay_per_epoch_micros " "must be >= 0"
  ) == str(e_info.value)
  abc_model_config.training.adam_optimizer.learning_rate_decay_per_epoch_micros = (
    -1
  )
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert (
    "AdamOptimizer.learning_rate_decay_per_epoch_micros " "must be >= 0"
  ) == str(e_info.value)


def test_AssertIsBuildable_adam_optimizer_beta_1_micros(abc_model_config):
  """UserError if beta_1_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField("beta_1_micros")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.beta_1_micros = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.beta_1_micros = 1000001
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )


def test_AssertIsBuildable_adam_optimizer_beta_2_micros(abc_model_config):
  """UserError if beta_2_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField("beta_2_micros")
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.beta_2_micros = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.beta_2_micros = 1000001
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000" == str(
    e_info.value
  )


def test_AssertIsBuildable_adam_optimizer_normalized_gradient_clip_micros(
  abc_model_config,
):
  """UserError if normalized_gradient_clip_micros field is invalid."""
  abc_model_config.training.adam_optimizer.ClearField(
    "normalized_gradient_clip_micros"
  )
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.normalized_gradient_clip_micros must be >= 0" == str(
    e_info.value
  )
  abc_model_config.training.adam_optimizer.normalized_gradient_clip_micros = -1
  with test.Raises(errors.UserError) as e_info:
    builders.AssertIsBuildable(abc_model_config)
  assert "AdamOptimizer.normalized_gradient_clip_micros must be >= 0" == str(
    e_info.value
  )


# BuildOptimizer() tests.


def test_BuildOptimizer_adam():
  """Test AdamOptimizer proto value conversion to Keras config."""
  config = model_pb2.Model()
  config.training.ClearField("optimizer")
  config.training.adam_optimizer.initial_learning_rate_micros = 2000
  config.training.adam_optimizer.learning_rate_decay_per_epoch_micros = 5000
  config.training.adam_optimizer.beta_1_micros = 900000
  config.training.adam_optimizer.beta_2_micros = 999000
  config.training.adam_optimizer.normalized_gradient_clip_micros = 5000000
  optimizer = builders.BuildOptimizer(config)
  optimizer_config = optimizer.get_config()
  assert pytest.approx(optimizer_config["decay"]) == 0.005
  assert pytest.approx(optimizer_config["beta_1"]) == 0.9
  assert pytest.approx(optimizer_config["beta_2"]) == 0.999
  assert pytest.approx(optimizer_config["clipnorm"]) == 5.0


def test_BuildOptimizer_rmsprop():
  """Test RmsOptimizer proto value conversion to Keras config."""
  config = model_pb2.Model()
  config.training.ClearField("optimizer")
  config.training.rmsprop_optimizer.initial_learning_rate_micros = 1000
  config.training.rmsprop_optimizer.learning_rate_decay_per_epoch_micros = 1000
  optimizer = builders.BuildOptimizer(config)
  optimizer_config = optimizer.get_config()
  assert pytest.approx(optimizer_config["decay"]) == 0.001
  assert pytest.approx(optimizer_config["rho"]) == 0.9


if __name__ == "__main__":
  test.Main()
