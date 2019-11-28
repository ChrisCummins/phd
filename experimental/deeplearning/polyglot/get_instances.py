"""Get the CLgen instances under test."""
import itertools
import typing

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
  "working_dir",
  "/mnt/cc/data/experimental/deeplearning/polyglot/clgen",
  "Path to CLgen working directory",
)

# Paths to protos within //experimental/polyglot/baselines.
LANGUAGES = {
  "opencl": {
    "corpuses": ["opencl-char", "opencl-tok"],
    "samplers": ["opencl-1.0", "opencl-0.5"],
  },
  "java": {
    "corpuses": ["java-char", "java-tok"],
    "samplers": ["java-1.0", "java-0.5"],
  },
}

# The CLgen model to base all permutations off, and the permutation options.
NUM_NEURONS = [256, 512, 1024]
NUM_LAYERS = [2, 3]
BASE_MODEL = """
# File: //deeplearning/clgen/proto/model.proto
# Proto: clgen.Model
architecture {
  backend: TENSORFLOW
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
  for num_neurons, num_layers in itertools.product(NUM_NEURONS, NUM_LAYERS):
    model = model_pb2.Model()
    model.CopyFrom(base_model)
    model.architecture.neurons_per_layer = num_neurons
    model.architecture.num_layers = num_layers
    models.append(model)
  return models


def EnumerateLanguageInstanceConfigs(
  language: typing.Dict[str, typing.List[str]]
) -> typing.List[clgen_pb2.Instance]:
  """Enumerate the options for a language."""
  configs = []
  for corpus, model, sampler in itertools.product(
    language["corpuses"], EnumerateModels(), language["samplers"]
  ):
    instance_config = clgen_pb2.Instance()
    instance_config.working_dir = FLAGS.working_dir
    instance_config.model.CopyFrom(model)
    instance_config.model.corpus.CopyFrom(
      pbutil.FromFile(
        bazelutil.DataPath(
          f"phd/experimental/deeplearning/polyglot/corpuses/{corpus}.pbtxt"
        ),
        corpus_pb2.Corpus(),
      )
    )
    instance_config.sampler.CopyFrom(
      pbutil.FromFile(
        bazelutil.DataPath(
          f"phd/experimental/deeplearning/polyglot/samplers/{sampler}.pbtxt"
        ),
        sampler_pb2.Sampler(),
      )
    )
    configs.append(instance_config)
  return configs


def GetInstanceConfigs() -> clgen_pb2.Instances:
  """Get the list of CLgen instance configs to test."""
  instances = clgen_pb2.Instances()
  for _, config in LANGUAGES.items():
    instances.instance.extend(EnumerateLanguageInstanceConfigs(config))
  return instances


def GetInstances() -> typing.List[clgen.Instance]:
  """Get the list of CLgen instances to test."""
  return [clgen.Instance(c) for c in GetInstanceConfigs().instance]


def RewriteContentIds(instances: typing.List[clgen.Instance]):
  for instance in instances:
    instance.model.config.corpus.ClearField("contentfiles")
    with instance.Session():
      instance.model.corpus.Create()
    instance.model.config.corpus.content_id = instance.model.corpus.content_id


def GetInstancesConfig(
  instances: typing.List[clgen.Instance],
) -> clgen_pb2.Instances:
  """Get an Instances proto from a list of instances."""
  config = clgen_pb2.Instances()
  config.instance.extend(instance.ToProto() for instance in instances)
  return config


def main(argv):
  """Main entry point."""
  del argv
  instances = GetInstances()
  RewriteContentIds(instances)
  print(GetInstancesConfig(instances))


if __name__ == "__main__":
  app.RunWithArgs(main)
