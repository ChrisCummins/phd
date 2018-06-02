"""Run a baseline."""
import pathlib
import typing

from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen import clgen
from deeplearning.clgen import samplers
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import crypto
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('corpus', None, 'Path to corpus config.')
flags.DEFINE_string('model', None, 'Path to model config.')
flags.DEFINE_string('sampler', None, 'Path to sampler config.')
flags.DEFINE_string('working_dir', '/var/phd/clgen/baseline',
                    'Path to CLgen working directory')
flags.DEFINE_integer('output_corpus_size', 5000,
                     'The minimum number of samples to generate in the output'
                     'corpus.')


def ExtractAllSubsamples(text: str, atomizer: atomizers.AtomizerBase,
                         sampler: samplers.Sampler) -> typing.List[str]:
  """Extract all subsamples from text.

  Find all substrings in text which begin with start_text and have a symmetrical
  balance of left and right chars.
  """
  out = []
  encoded_text = atomizer.TokenizeString(text)
  start_index = encoded_text.find(sampler.start_text)
  while start_index > 0:
    j = start_index
    for j in range(start_index, len(encoded_text)):
      if sampler.SampleIsComplete(encoded_text[start_index:j + 1]):
        break
    out.append(''.join(encoded_text[start_index:j + 1]))
    start_index = encoded_text.find(sampler.start_text, start_index + 1)
  return out


def SampleModel(instance: clgen.Instance) -> None:
  """Make --output_corpus_size samples from model."""
  logging.info('Training and sampling the CLgen model ...')
  target_samples = FLAGS.output_corpus_size
  num_samples = 0
  sample_dir = instance.model.SamplerCache(instance.sampler)
  if sample_dir.is_dir():
    num_samples = len(list(sample_dir.iterdir()))
  logging.info('Need to generate %d samples', target_samples - num_samples)
  while num_samples < target_samples:
    instance.Sample(min_num_samples=target_samples - num_samples)
    num_samples = len(list(sample_dir.iterdir()))


def CreateOutputCorpus(instance: clgen.Instance) -> corpuses.Corpus:
  """Create a CLgen corpus from the samples we just made."""
  out_dir = pathlib.Path(
      str(instance.model.SamplerCache(instance.sampler)) + '.postprocessed')
  logging.info('Creating output corpus at %s', out_dir)

  out_dir.mkdir(exist_ok=True)
  sample_dir = instance.model.SamplerCache(instance.sampler)
  for sample_path in sample_dir.iterdir():
    sample = pbutil.FromFile(sample_dir / sample_path, model_pb2.Sample())
    out_samples = ExtractAllSubsamples(
        sample.text, instance.model.corpus.atomizer, instance.sampler)
    for out_sample in out_samples:
      sha256 = crypto.sha256_str(out_sample)
      with open(out_dir / (sha256 + '.txt'), 'w') as f:
        f.write(out_sample)
  output_corpus_config = corpus_pb2.Corpus()
  output_corpus_config.CopyFrom(instance.model.corpus.config)
  output_corpus_config.local_directory = str(out_dir)
  output_corpus = corpuses.Corpus(output_corpus_config)
  output_corpus.Create()
  return output_corpus


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  corpus = pbutil.FromFile(pathlib.Path(FLAGS.corpus), corpus_pb2.Corpus())
  model = pbutil.FromFile(pathlib.Path(FLAGS.model), model_pb2.Model())
  model.corpus.CopyFrom(corpus)
  sampler = pbutil.FromFile(pathlib.Path(FLAGS.sampler), sampler_pb2.Sampler())

  config = clgen_pb2.Instance(model=model, sampler=sampler)
  config.working_dir = FLAGS.working_dir
  instance = clgen.Instance(config)

  with instance.Session():
    SampleModel(instance)
    CreateOutputCorpus(instance)


if __name__ == '__main__':
  app.run(main)
