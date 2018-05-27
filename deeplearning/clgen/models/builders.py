"""This file builds Keras models from CLgen Model config protos."""
from absl import flags
from keras import layers
from keras import models
from keras import optimizers

from deeplearning.clgen import errors
from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS


def AssertBuildable(config: model_pb2.Model) -> model_pb2.Model:
  """Assert that a model configuration is buildable.

  Args:
    config: A model proto.

  Returns:
    The input model proto, unmodified.

  Rases:
    UserError: If the model is not buildable.
  """
  if not config.architecture.HasField('neuron_type'):
    raise errors.UserError('Model.architecture.neuron_type field not set')
  if not config.training.HasField('optimizer'):
    raise errors.UserError('Model.training.optimizer field not set')
  return config


def BuildKerasOptimizer(config: model_pb2.Model) -> optimizers.Optimizer:
  # TODO(cec): Re-implement learning rate, decay rate, and gradient clip.
  # learning_rate = config.training.initial_learning_rate_micros / 10e6
  # decay_rate = config.training.percent_learning_rate_decay_per_epoch
  return 'adam'


def BuildKerasModel(config: model_pb2.Model,
                    sequence_length: int,
                    vocabulary_size: int) -> models.Sequential:
  """Build a Keras model from a Model proto.

  Args:
    config: A Model proto instance.
    sequence_length: The length of sequences.
    vocabulary_size: The number of tokens in the vocabulary.

  Returns:
    A Sequential model instance.
  """
  model = models.Sequential()
  layer = {
    model_pb2.NetworkArchitecture.LSTM: layers.LSTM,
    model_pb2.NetworkArchitecture.RNN: layers.RNN,
    model_pb2.NetworkArchitecture.GRU: layers.GRU,
  }[config.architecture.neuron_type]
  model.add(layer(config.architecture.neurons_per_layer,
                  input_shape=(
                    sequence_length,
                    vocabulary_size),
                  return_sequences=config.architecture.neurons_per_layer > 1))
  for _ in range(1, config.architecture.num_layers - 1):
    model.add(layer(config.architecture.neurons_per_layer,
                    return_sequences=True))
  model.add(layer(config.architecture.neurons_per_layer))
  model.add(layers.Dense(vocabulary_size))
  model.add(layers.Activation('softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer=BuildKerasOptimizer(config))
  return model
