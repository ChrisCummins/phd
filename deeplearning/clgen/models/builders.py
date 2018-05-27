"""This file builds Keras models from CLgen Model config protos."""
from absl import flags
from keras import layers
from keras import models
from keras import optimizers

from deeplearning.clgen import errors
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


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
    pbutil.AssertFieldIsSet(config.architecture, 'neuron_type')
    pbutil.AssertFieldConstraint(
        config.architecture, 'neurons_per_layer', lambda x: 0 < x,
        'NetworkArchitecture.neurons_per_layer must be > 0')
    pbutil.AssertFieldConstraint(
        config.architecture, 'num_layers', lambda x: 0 < x,
        'NetworkArchitecture.num_layers must be > 0')
    pbutil.AssertFieldConstraint(
        config.training, 'num_epochs', lambda x: 0 < x,
        'TrainingOptions.num_epochs must be > 0')
    pbutil.AssertFieldIsSet(
        config.training, 'shuffle_corpus_contentfiles_between_epochs')
    pbutil.AssertFieldConstraint(
        config.training, 'batch_size', lambda x: 0 < x,
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
          config.training.adam_optimizer,
          'beta_1_micros', lambda x: 0 <= x <= 1000000,
          'AdamOptimizer.beta_1_micros must be >= 0 and <= 1000000')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          'beta_2_micros', lambda x: 0 <= x <= 1000000,
          'AdamOptimizer.beta_2_micros must be >= 0 and <= 1000000')
      pbutil.AssertFieldConstraint(
          config.training.adam_optimizer,
          'normalized_gradient_clip_micros', lambda x: 0 <= x,
          'AdamOptimizer.normalized_gradient_clip_micros must be >= 0')
    else:
      raise errors.InternalError(
          "Unrecognized value: 'TrainingOptions.optimizer'")
  except pbutil.ProtoValueError as e:
    raise errors.UserError(str(e))
  return config


def BuildOptimizer(config: model_pb2.Model) -> optimizers.Optimizer:
  """Construct the training optimizer from config.

  Args:
    config: A Model config proto.

  Raises:
    InternalError: If the value of the optimizer field is not understood.
  """
  # We do not use *any* default values for arguments, in case for whatever
  # reason the Keras API changes a default arg.
  if config.training.HasField('adam_optimizer'):
    adam = config.training.adam_optimizer
    return optimizers.Adam(
        lr=adam.initial_learning_rate_micros / 1e6,
        beta_1=adam.beta_1_micros / 1e6,
        beta_2=adam.beta_2_micros / 1e6,
        epsilon=None,
        decay=adam.learning_rate_decay_per_epoch_micros / 1e6,
        amsgrad=False,
        clipnorm=adam.normalized_gradient_clip_micros / 1e6,
    )
  else:
    raise errors.InternalError(
        "Unrecognized value: 'TrainingOptions.optimizer'")


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
  return model
