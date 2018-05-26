"""This file builds Keras models from CLgen Model config protos."""
from absl import flags
from keras import models, layers

from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS


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
  return_sequences = config.architecture.neurons_per_layer > 1
  model.add(layer(config.architecture.neurons_per_layer,
                  input_shape=(
                    sequence_length,
                    vocabulary_size),
                  return_sequences=return_sequences))
  for _ in range(1, config.architecture.num_layers - 1):
    model.add(layer(config.architecture.neurons_per_layer,
                    return_sequences=True))
  model.add(layer(config.architecture.neurons_per_layer))
  model.add(layers.Dense(vocabulary_size))
  model.add(layers.Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model
