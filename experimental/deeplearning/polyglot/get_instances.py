"""Get the CLgen instances under test."""
import itertools
import typing

from absl import flags

from deeplearning.clgen import clgen
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import bazelutil
from lib.labm8 import pbutil


flags.DEFINE_string(
    'working_dir', '/mnt/cc/data/experimental/deeplearning/polyglot/clgen',
    'Path to CLgen working directory')

FLAGS = flags.FLAGS

# Paths to protos within //experimental/polyglot/baselines.
LANGUAGES = {
  'opencl': {
    'corpuses': ['opencl-char', 'opencl-tok'],
    'samplers': ['opencl-1.0', 'opencl-0.5'],
  },
  'java': {
    'corpuses': ['java-char', 'java-tok'],
    'samplers': ['java-1.0', 'java-0.5'],
  }
}

# The CLgen model to base all permutations off, and the permutation options.
NUM_NEURONS = [512, 1024]
BASE_MODEL = """
# File: //deeplearning/clgen/proto/model.proto
# Proto: clgen.Model
architecture {
  embedding_size: 32
  neuron_type: LSTM
  neurons_per_layer: 512
  num_layers: 2
  post_layer_dropout_micros: 0
}
training {
  num_epochs: 50
  sequence_length: 64
  batch_size: 64
  shuffle_corpus_contentfiles_between_epochs: true
  adam_optimizer {
    initial_learning_rate_micros: 2000
    learning_rate_decay_per_epoch_micros: 50000
    beta_1_micros: 900000
    beta_2_micros: 999000
    normalized_gradient_clip_micros: 5000000
  }
}
"""


def EnumerateModels() -> typing.List[model_pb2.Model]:
  """Enumerate the model configurations."""
  models = []
  base_model = pbutil.FromString(BASE_MODEL, model_pb2.Model())
  for num_neurons, in itertools.product(NUM_NEURONS):
    model = model_pb2.Model()
    model.CopyFrom(base_model)
    model.architecture.neurons_per_layer = num_neurons
    models.append(model)
  return models


def EnumerateLanguageInstances(
    language: typing.Dict[str, typing.List[str]]) -> typing.List[
  clgen.Instance]:
  """Enumerate the options for a language."""
  instances = []
  for corpus, model, sampler in itertools.product(
      language['corpuses'], EnumerateModels(), language['samplers']):
    instance_config = clgen_pb2.Instance()
    instance_config.working_dir = FLAGS.working_dir
    instance_config.model.CopyFrom(model)
    instance_config.model.corpus.CopyFrom(pbutil.FromFile(bazelutil.DataPath(
        f'phd/experimental/deeplearning/polyglot/corpuses/{corpus}.pbtxt'),
        corpus_pb2.Corpus()))
    instance_config.sampler.CopyFrom(pbutil.FromFile(bazelutil.DataPath(
        f'phd/experimental/deeplearning/polyglot/samplers/{sampler}.pbtxt'),
        sampler_pb2.Sampler()))
    instance = clgen.Instance(instance_config)
    # Swap the model for a TensorFlow model.
    instance.model = tensorflow_backend.TensorFlowModel(instance_config.model)
    instances.append(instance)
  return instances


def GetInstances() -> typing.List[clgen.Instance]:
  """Get the list of CLgen instances to test."""
  instances = []
  for _, config in LANGUAGES.items():
    instances += EnumerateLanguageInstances(config)
  return instances
