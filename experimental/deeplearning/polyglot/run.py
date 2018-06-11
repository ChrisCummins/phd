"""Run a baseline."""
import collections
import pathlib
import random
import time

import humanize
from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen import clgen
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from experimental.deeplearning.polyglot import get_instances
from lib.labm8 import crypto
from lib.labm8 import lockfile
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_integer('output_corpus_size', 10000,
                     'The minimum number of samples to generate in the output'
                     'corpus.')

# A mapping from language name to a list of CLgen pre-processor functions.
# These pre-processors are used as rejection samplers on the sample corpuses.
POSTPROCESSORS = {
    'opencl': ['deeplearning.clgen.preprocessors.opencl:Compile'],
    'java': ['deeplearning.clgen.preprocessors.java:Compile'],
}

def IsEligible(instance: clgen.Instance) -> bool:
  """Return whether an instance is eligible for training or sampling."""
  if instance.model.corpus.is_locked:
    logging.info('Corpus is locked')
    return False
  if instance.model.training_lock.islocked:
    logging.info('Model is locked')
    return False
  sample_dir = instance.model.SamplerCache(instance.sampler)
  sample_lock = lockfile.LockFile(sample_dir / 'LOCK')
  if sample_lock.islocked:
    logging.info('Sampler is locked')
    return False
  return True


def SampleModel(instance: clgen.Instance) -> None:
  """Take --output_corpus_size samples from model."""
  logging.info('Training and sampling the CLgen model ...')
  target_samples = FLAGS.output_corpus_size
  sample_dir = instance.model.SamplerCache(instance.sampler)
  sample_dir.mkdir(exist_ok=True)
  num_samples = len(list(sample_dir.iterdir()))
  logging.info('Need to generate %d samples in %s',
               max(target_samples - num_samples, 0), sample_dir)
  if num_samples < target_samples:
    sample_lock = lockfile.LockFile(sample_dir / 'LOCK')
    with sample_lock.acquire(replace_stale=True, block=True):
      num_samples = len(list(sample_dir.iterdir()))
      while num_samples < target_samples:
        samples = instance.model.SampleFast(
            instance.sampler, target_samples - num_samples)
        for sample in samples:
          sample_id = crypto.sha256_str(sample.text)
          pbutil.ToFile(sample, sample_dir / f'{sample_id}.pbtxt')
        num_samples = len(list(sample_dir.iterdir()))


def PostprocessSampleCorpus(instance: clgen.Instance):
  """Create a corpus from the model samples and pre-process."""
  sample_dir = instance.model.SamplerCache(instance.sampler)

  # Read the sample protos and write them to a directory of content files.
  contentfiles_dir = pathlib.Path(str(sample_dir) + '.contentfiles')
  contentfiles_dir.mkdir(exist_ok=True)
  logging.info('Writing output contentfiles to %s', contentfiles_dir)
  if len(list(contentfiles_dir.iterdir())) != len(list(sample_dir.iterdir())):
    for proto_path in sample_dir.iterdir():
      sample = pbutil.FromFile(proto_path, model_pb2.Sample())
      with open(contentfiles_dir / proto_path.name, 'w') as f:
        f.write(sample.text)

  logging.info('Creating output corpus')
  output_corpus_config = corpus_pb2.Corpus()
  output_corpus_config.CopyFrom(instance.model.corpus.config)
  output_corpus_config.local_directory = str(contentfiles_dir)
  # We derive the programming language name from the input corpus directory.
  # This depends on corpuses being in directories named after their language,
  # e.g. ~/corpuses/opencl, or ~/corpuses/java.
  language = pathlib.Path(instance.model.corpus.config.local_directory).name
  output_corpus_config.preprocessor[:] = POSTPROCESSORS[language]
  output_corpus = corpuses.Corpus(output_corpus_config)
  output_corpus.Create()
  return output_corpus


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  start_time = time.time()
  instances = get_instances.GetInstances()
  random.shuffle(instances)
  candidate_instances = collections.deque(instances)
  logging.info('Loaded %d instances in %s ms', len(candidate_instances),
               humanize.intcomma(int((time.time() - start_time) * 1000)))

  while candidate_instances:
    instance = candidate_instances.popleft()
    with instance.Session():
      if IsEligible(instance):
        logging.info('Found an eligible candidate to work on')
        SampleModel(instance)
        PostprocessSampleCorpus(instance)
      else:
        logging.info('Candidate is ineligible')
        candidate_instances.append(instance)
        time.sleep(1)

  logging.info('Done.')


if __name__ == '__main__':
  app.run(main)
