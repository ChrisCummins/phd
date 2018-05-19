import collections
import time
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen.proto import clgen_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.services import generator
from deeplearning.deepsmith.services import services

FLAGS = flags.FLAGS

ClgenJsonConfigs = collections.namedtuple(
  'ClgenJsonConfigs', ['model', 'sampler'])


def ProtosToClgenJson(corpus: clgen_pb2.Corpus,
                      model: clgen_pb2.Model,
                      sampler: clgen_pb2.Sampler) -> ClgenJsonConfigs:
  json = ClgenJsonConfigs({}, {})
  # Convert Corpus proto to JSON.
  json.model['corpus'] = {}
  json.model['corpus']['language'] = corpus.language
  json.model['corpus']['path'] = corpus.path
  if corpus.HasField('ascii_character_tokenizer'):
    json.model['corpus']['vocabulary'] = 'char'
  elif corpus.HasField('greedy_multichar_tokenizer'):
    json.model['corpus']['vocabulary'] = 'greedy'
  else:
    raise ValueError()
  json.model['corpus']['seq_length'] = corpus.sequence_length
  json.model['corpus']['eof'] = corpus.contentfile_separator != '\n\n'
  # Convert Model proto to JSON.
  json.model['architecture'] = {}
  if model.architecture.neuron_type == clgen_pb2.NetworkArchitecture.LSTM:
    json.model['architecture']['model_type'] = 'lstm'
  elif model.architecture.neuron_type == clgen_pb2.NetworkArchitecture.RNN:
    json.model['architecture']['model_type'] = 'rnn'
  elif model.architecture.neuron_type == clgen_pb2.NetworkArchitecture.GRU:
    json.model['architecture']['model_type'] = 'gru'
  else:
    raise ValueError()
  json.model['architecture']['rnn_size'] = model.architecture.neurons_per_layer
  json.model['architecture']['num_layers'] = model.architecture.num_layers
  json.model['train_opts'] = {}
  json.model['train_opts']['epochs'] = model.training.num_epochs
  json.model['corpus'][
    'preserve_order'] = model.training.shuffle_corpus_contentfiles_between_epochs
  json.model['corpus']['batch_size'] = model.training.batch_size
  json.model['train_opts']['grad_clip'] = model.training.gradient_clip
  json.model['train_opts'][
    'learning_rate'] = model.training.initial_learning_rate
  json.model['train_opts'][
    'lr_decay_rate'] = model.training.percent_learning_rate_decay_per_epoch
  json.model['train_opts'][
    'intermediate_checkpoints'] = model.training.save_intermediate_checkpoints
  # Convert Sampler proto to JSON.
  json.sampler['kernels'] = {}
  json.sampler['kernels']['language'] = corpus.language
  json.sampler['kernels']['start_text'] = corpus.start_text
  json.sampler['kernels']['seed'] = sampler.seed
  for t in sampler.termination_criteria:
    if t.HasField('maxlen'):
      json.sampler['kernels']['max_length'] = t.maxlen.maximum_tokens_in_sample
  json.sampler['sampler'] = {}
  json.sampler['sampler']['static_checker'] = False
  json.sampler['sampler']['gpuverify'] = False
  return json


class ClgenGenerator(generator.GeneratorBase,
                     generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    self.config = config

  def GetGeneratorCapabilities(
      self, request: generator_pb2.GetGeneratorCapabilitiesRequest,
      context) -> generator_pb2.GetGeneratorCapabilitiesResponse:
    del context
    logging.info('GetGeneratorCapabilities() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
      generator_pb2.GetGeneratorCapabilitiesRequest)
    # TODO(cec): Implement!
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    logging.info('GenerateTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
      generator_pb2.GenerateTestcasesResponse)
    # TODO(cec): Implement!
    return response


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Unrecognized arguments')
  generator_config = services.ServiceConfigFromFlag(
    'generator_config', generator_pb2.ClgenGenerator())
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  services.AssertLocalServiceHostname(generator_config.service)
  service = ClgenGenerator(generator_config)
  generator_pb2_grpc.add_GeneratorServiceServicer_to_server(service, server)
  server.add_insecure_port(f'[::]:{generator_config.service.port}')
  logging.info('%s listening on %s:%s', type(service).__name__,
               generator_config.service.hostname,
               generator_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
