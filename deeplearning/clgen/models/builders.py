# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
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
"""This file builds Keras models from CLgen Model config protos."""
from deeplearning.clgen import errors
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import pbutil

FLAGS = app.FLAGS


def AssertIsBuildable(config: model_pb2.Model) -> model_pb2.Model:
  """Assert that a model configuration is buildable.

  Args:
    config: A model proto.

  Returns:
    The input model proto, unmodified.

  Raises:
    UserError: If the model is not buildable.
    InternalError: If the value of the training.optimizer field is not
      understood.
  """
  # Any change to the Model proto schema will require a change to this function.
  try:
    pbutil.AssertFieldIsSet(config, 'corpus')
    pbutil.AssertFieldIsSet(config, 'architecture')
    pbutil.AssertFieldIsSet(config, 'training')
    pbutil.AssertFieldIsSet(config.architecture, 'backend')
    pbutil.AssertFieldIsSet(config.architecture, 'neuron_type')
    if config.architecture.backend == model_pb2.NetworkArchitecture.KERAS:
      pbutil.AssertFieldConstraint(
          config.architecture, 'embedding_size', lambda x: 0 < x,
          'NetworkArchitecture.embedding_size must be > 0')
    pbutil.AssertFieldConstraint(
        config.architecture, 'neurons_per_layer', lambda x: 0 < x,
        'NetworkArchitecture.neurons_per_layer must be > 0')
    pbutil.AssertFieldConstraint(config.architecture, 'num_layers',
                                 lambda x: 0 < x,
                                 'NetworkArchitecture.num_layers must be > 0')
    pbutil.AssertFieldConstraint(
        config.architecture, 'post_layer_dropout_micros',
        lambda x: 0 <= x <= 1000000,
        'NetworkArchitecture.post_layer_dropout_micros '
        'must be >= 0 and <= 1000000')
    pbutil.AssertFieldConstraint(config.training, 'num_epochs', lambda x: 0 < x,
                                 'TrainingOptions.num_epochs must be > 0')
    pbutil.AssertFieldIsSet(config.training,
                            'shuffle_corpus_contentfiles_between_epochs')
    pbutil.AssertFieldConstraint(config.training, 'batch_size', lambda x: 0 < x,
                                 'TrainingOptions.batch_size must be > 0')
    pbutil.AssertFieldIsSet(config.training, 'optimizer')
    if config.training.HasField('adam_optimizer'):
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer, 'initial_learning_rate_micros',
          lambda x: 0 <= x,
          'AdamOptimizer.initial_learning_rate_micros must be >= 0')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          'learning_rate_decay_per_epoch_micros', lambda x: 0 <= x,
          'AdamOptimizer.learning_rate_decay_per_epoch_micros must be >= 0')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer, 'beta_1_micros',
          lambda x: 0 <= x <= 1000000,
          'AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer, 'beta_2_micros',
          lambda x: 0 <= x <= 1000000,
          'AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer, 'normalized_gradient_clip_micros',
          lambda x: 0 <= x,
          'AdamOptimizer.normalized_gradient_clip_micros must be >= 0')
    elif config.training.HasField('rmsprop_optimizer'):
      pbutil.AssertFieldConstraint(
          config.training.rmsprop_optimizer, 'initial_learning_rate_micros',
          lambda x: 0 <= x,
          'RmsPropOptimizer.initial_learning_rate_micros must be >= 0')
      pbutil.AssertFieldConstraint(
          config.training.rmsprop_optimizer,
          'learning_rate_decay_per_epoch_micros', lambda x: 0 <= x,
          'RmsPropOptimizer.learning_rate_decay_per_epoch_micros must be >= 0')
    else:
      raise errors.InternalError(
          "Unrecognized value: 'TrainingOptions.optimizer'")
  except pbutil.ProtoValueError as e:
    raise errors.UserError(str(e))
  return config


def BuildOptimizer(config: model_pb2.Model) -> 'keras.optimizers.Optimizer':
  """Construct the training optimizer from config.

  Args:
    config: A Model config proto.

  Raises:
    InternalError: If the value of the optimizer field is not understood.
  """
  # Deferred importing of Keras so that we don't have to activate the
  # TensorFlow backend every time we import this module.
  import keras

  # We do not use *any* default values for arguments, in case for whatever
  # reason the Keras API changes a default arg.
  if config.training.HasField('adam_optimizer'):
    opts = {}
    opt = config.training.adam_optimizer
    if opt.normalized_gradient_clip_micros:
      opts['clipnorm'] = opt.normalized_gradient_clip_micros / 1e6
    return keras.optimizers.Adam(
        lr=opt.initial_learning_rate_micros / 1e6,
        beta_1=opt.beta_1_micros / 1e6,
        beta_2=opt.beta_2_micros / 1e6,
        epsilon=None,
        decay=opt.learning_rate_decay_per_epoch_micros / 1e6,
        amsgrad=False,
        **opts,
    )
  elif config.training.HasField('rmsprop_optimizer'):
    opt = config.training.rmsprop_optimizer
    return keras.optimizers.RMSprop(
        lr=opt.initial_learning_rate_micros / 1e6,
        decay=opt.initial_learning_rate_micros / 1e6,
        rho=0.9,
        epsilon=None,
    )
  else:
    raise errors.InternalError(
        "Unrecognized value: 'TrainingOptions.optimizer'")


def BuildKerasModel(config: model_pb2.Model,
                    vocabulary_size: int) -> 'keras.models.Sequential':
  """Build a Keras model from a Model proto.

  Args:
    config: A Model proto instance.
    vocabulary_size: The number of tokens in the vocabulary.

  Returns:
    A Sequential model instance.
  """
  # Deferred importing of Keras so that we don't have to activate the
  # TensorFlow backend every time we import this module.
  import keras

  dropout = (config.architecture.post_layer_dropout_micros or 0) / 1e6
  model = keras.models.Sequential()
  layer = {
      model_pb2.NetworkArchitecture.LSTM: keras.layers.LSTM,
      model_pb2.NetworkArchitecture.RNN: keras.layers.RNN,
      model_pb2.NetworkArchitecture.GRU: keras.layers.GRU,
  }[config.architecture.neuron_type]

  # The input layer.
  model.add(
      keras.layers.Embedding(
          vocabulary_size,
          config.architecture.embedding_size,
          batch_input_shape=(config.training.batch_size,
                             config.training.sequence_length)))
  model.add(keras.layers.Dropout(dropout))
  # The recurrent network layers.
  for _ in range(config.architecture.num_layers):
    model.add(
        layer(config.architecture.neurons_per_layer,
              return_sequences=True,
              stateful=True))
    model.add(keras.layers.Dropout(dropout))
  # The output layer.
  model.add(
      keras.layers.TimeDistributed(
          keras.layers.Dense(vocabulary_size, activation='softmax')))
  return model
