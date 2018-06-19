import contextlib
import os
import pathlib
import time
import typing
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

import deeplearning.clgen.models.keras_backend
from deeplearning.clgen import samplers
from deeplearning.clgen.proto import model_pb2
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import generator_pb2
from deeplearning.deepsmith.proto import generator_pb2_grpc
from deeplearning.deepsmith.services import generator
from deeplearning.deepsmith.services import services


FLAGS = flags.FLAGS


def ClgenConfigToGenerator(
    m: deeplearning.clgen.models.keras_backend.KerasBackend,
    s: samplers.Sampler) -> deepsmith_pb2.Generator:
  """Convert a CLgen model+sampler pair to a DeepSmith generator proto."""
  # TODO(cec): Update for new options and add unit tests.
  g = deepsmith_pb2.Generator()
  g.name = f'clgen_model:{m.hash}_sampler:{s.hash}'
  g.opts['contentfiles_id'] = m.corpus.content_id
  g.opts['corpus_id'] = m.corpus.hash
  g.opts['neuron_type'] = model_pb2.NetworkArchitecture.NeuronType.Name(
      m.config.architecture.neuron_type)
  g.opts['neurons_per_layer'] = str(m.config.architecture.neurons_per_layer)
  g.opts['num_layers'] = str(m.config.architecture.num_layers)
  g.opts[
    'shuffle_corpus_contentfiles_between_epochs'] = 'true' if \
    m.config.training.shuffle_corpus_contentfiles_between_epochs else 'false'
  g.opts['training_batch_size'] = str(m.config.training.batch_size)
  # g.opts['initial_learning_rate'] = str(m.config.training.initial_learning_rate)
  # g.opts['percent_learning_rate_decay_per_epoch'] = str(
  #     m.config.training.percent_learning_rate_decay_per_epoch)
  g.opts['start_text'] = s.config.start_text
  g.opts['sampler_batch_size'] = str(s.config.batch_size)
  for criterion in s.config.termination_criteria:
    if criterion.HasField('maxlen'):
      g.opts['max_tokens_in_sample'] = str(
          criterion.maxlen.maximum_tokens_in_sample)
    elif criterion.HasField('symtok'):
      g.opts['depth_left_token'] = criterion.symtok.depth_increase_token
      g.opts['depth_right_token'] = criterion.symtok.depth_decrease_token
  return g


class ClgenGenerator(generator.GeneratorBase,
                     generator_pb2_grpc.GeneratorServiceServicer):

  def __init__(self, config: generator_pb2.ClgenGenerator):
    super(ClgenGenerator, self).__init__(config)
    with self.ClgenWorkingDir():
      m = deeplearning.clgen.models.keras_backend.KerasBackend(
          self.config.model)
      s = samplers.Sampler(self.config.sampler)
      self.generator = ClgenConfigToGenerator(m, s)
      for t in self.config.testcase_skeleton:
        t.generator.CopyFrom(self.generator)
      logging.info('Generator\n%s', self.generator)
      m.Train()

  @contextlib.contextmanager
  def ClgenWorkingDir(self) -> None:
    """Temporarily set CLGEN_CACHE variable to working directory."""
    # Expand shell variables.
    path = os.path.expandvars(self.config.clgen_working_dir)
    path = pathlib.Path(path).expanduser().absolute()
    path.mkdir(parents=True, exist_ok=True)
    previous_value = os.environ.get('CLGEN_CACHE', '')
    os.environ['CLGEN_CACHE'] = str(path)
    try:
      yield
    finally:
      os.environ['CLGEN_CACHE'] = previous_value

  def GetGeneratorCapabilities(self,
                               request:
                               generator_pb2.GetGeneratorCapabilitiesRequest,
                               context) -> \
      generator_pb2.GetGeneratorCapabilitiesResponse:
    del context
    logging.info('GetGeneratorCapabilities() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
        generator_pb2.GetGeneratorCapabilitiesRequest)
    response.toolchain = self.config.model.corpus.language
    response.generator = self.generator
    return response

  def GenerateTestcases(self, request: generator_pb2.GenerateTestcasesRequest,
                        context) -> generator_pb2.GenerateTestcasesResponse:
    del context
    logging.info('GenerateTestcases() client=%s', request.status.client)
    response = services.BuildDefaultResponse(
        generator_pb2.GenerateTestcasesResponse)
    os.environ['CLGEN_CACHE'] = self.config.clgen_working_dir
    with self.ClgenWorkingDir():
      m = deeplearning.clgen.models.keras_backend.KerasBackend(
          self.config.model)
      s = samplers.Sampler(self.config.sampler)
      for sample in m.Sample(s):
        response.testcases.extend(self.SampleToTestcases(sample))
    return response

  def SampleToTestcases(self, sample: model_pb2.Sample) -> typing.List[
    deepsmith_pb2.Testcase]:
    """Convert a CLgen sample to a list of DeepSmith testcase protos."""
    testcases = []
    for skeleton in self.config.testcase_skeleton:
      t = deepsmith_pb2.Testcase()
      t.CopyFrom(skeleton)
      p = t.profiling_events.add()
      p.type = "generation"
      p.duration_ms = sample.sample_time_ms
      p.event_start_epoch_ms = sample.sample_start_epoch_ms_utc
      t.inputs['src'] = sample.text
      testcases.append(t)
    return testcases


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
               generator_config.service.hostname, generator_config.service.port)
  server.start()
  try:
    while True:
      time.sleep(3600 * 24)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
