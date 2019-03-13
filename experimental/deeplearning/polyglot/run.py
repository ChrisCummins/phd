"""Run a baseline."""
import collections
import pathlib
import random
import time

from deeplearning.clgen import clgen
from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from labm8 import app
from labm8 import crypto
from labm8 import humanize
from labm8 import lockfile
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_integer(
    'output_corpus_size', 10000,
    'The minimum number of samples to generate in the output'
    'corpus.')
app.DEFINE_string('instances', None, 'Path to a clgen.Instances proto')

# A mapping from language name to a list of CLgen pre-processor functions.
# These pre-processors are used as rejection samplers on the sample corpuses.
POSTPROCESSORS = {
    'opencl': ['deeplearning.clgen.preprocessors.opencl:Compile'],
    'java': ['deeplearning.clgen.preprocessors.java:Compile'],
}


def IsEligible(instance: clgen.Instance) -> bool:
  """Return whether an instance is eligible for training or sampling."""
  if instance.model.corpus.is_locked:
    app.Log(1, 'Corpus is locked')
    return False
  if instance.model.training_lock.islocked:
    app.Log(1, 'Model is locked')
    return False
  sample_dir = instance.model.SamplerCache(instance.sampler)
  sample_lock = lockfile.LockFile(sample_dir / 'LOCK')
  if sample_lock.islocked:
    app.Log(1, 'Sampler is locked')
    return False
  return True


def SampleModel(instance: clgen.Instance) -> None:
  """Take --output_corpus_size samples from model."""
  app.Log(1, 'Training and sampling the CLgen model ...')
  target_samples = FLAGS.output_corpus_size
  sample_dir = instance.model.SamplerCache(instance.sampler)
  sample_dir.mkdir(exist_ok=True)
  num_samples = len(list(sample_dir.iterdir()))
  app.Log(1, 'Need to generate %d samples in %s',
          max(target_samples - num_samples, 0), sample_dir)
  if num_samples < target_samples:
    sample_lock = lockfile.LockFile(sample_dir / 'LOCK')
    with sample_lock.acquire(replace_stale=True, block=True):
      num_samples = len(list(sample_dir.iterdir()))
      while num_samples < target_samples:
        samples = instance.model.Sample(
            instance.sampler,
            target_samples - num_samples,
            cache_samples=False,
            print_samples=False)
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
  app.Log(1, 'Writing output contentfiles to %s', contentfiles_dir)
  if len(list(contentfiles_dir.iterdir())) != len(list(sample_dir.iterdir())):
    for proto_path in sample_dir.iterdir():
      sample = pbutil.FromFile(proto_path, model_pb2.Sample())
      with open(contentfiles_dir / proto_path.name, 'w') as f:
        f.write(sample.text)

  app.Log(1, 'Creating output corpus')
  output_corpus_config = corpus_pb2.Corpus()
  output_corpus_config.CopyFrom(instance.model.corpus.config)
  output_corpus_config.local_directory = str(contentfiles_dir)
  # We derive the programming language name from the input corpus directory.
  # This depends on corpuses being in directories named after their language,
  # e.g. ~/corpuses/opencl, or ~/corpuses/java.A
  preprocessed_dir = instance.model.corpus.preprocessed.url[len('sqlite:///'
                                                               ):].parent
  language = (preprocessed_dir / 'contentfiles').resolve().name
  output_corpus_config.preprocessor[:] = POSTPROCESSORS[language]
  output_corpus = corpuses.Corpus(output_corpus_config)
  try:
    output_corpus.Create()
  except errors.EmptyCorpusException:
    pass
  return output_corpus


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  start_time = time.time()
  instances = [
      clgen.Instance(p) for p in pbutil.FromFile(
          pathlib.Path(FLAGS.instances), clgen_pb2.Instances()).instance
  ]
  random.shuffle(instances)
  candidate_instances = collections.deque(instances)
  app.Log(1, 'Loaded %d instances in %s ms', len(candidate_instances),
          humanize.Commas(int((time.time() - start_time) * 1000)))

  while candidate_instances:
    instance = candidate_instances.popleft()
    with instance.Session():
      if IsEligible(instance):
        app.Log(1, 'Found an eligible candidate to work on')
        SampleModel(instance)
        PostprocessSampleCorpus(instance)
      else:
        app.Log(1, 'Candidate is ineligible')
        candidate_instances.append(instance)
        time.sleep(1)

  app.Log(1, 'Done.')


if __name__ == '__main__':
  app.RunWithArgs(main)
