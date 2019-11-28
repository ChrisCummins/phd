"""Sample and import from CLgen."""
import multiprocessing
import pathlib
import tempfile
import typing

from deeplearning.clgen import clgen
from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db
from labm8.py import app
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')
app.DEFINE_integer('batch_size', 512, 'Number of samples to make per batch.')
app.DEFINE_string('origin', 'clgen',
                  'Name of the origin of the kernels, e.g. "clgen".')
app.DEFINE_string('clgen_dir', '~/.cache/clgen',
                  'Name of the origin of the kernels, e.g. "clgen".')
app.DEFINE_string('clgen_corpus_dir',
                  "/mnt/cc/data/datasets/github/corpuses/opencl",
                  "WHere the corpus is stored.")
app.DEFINE_string(
    'profile_dir',
    '/tmp/phd/experimental/deeplearning/clgen/closeness_to_grewe_features/clgen_profiles',
    'Path to a directory to store profiling data in.')


def CreateTempFileFromSample(tempdir: pathlib.Path, sample: model_pb2.Sample,
                             number: int) -> pathlib.Path:
  """Write testcase to a file in directory."""
  path = tempdir / f'{number}.cl'
  with open(path, 'w') as f:
    f.write(sample.text)
  return path


class SampleObserver(sample_observers_lib.SampleObserver):

  def __init__(self):
    self.samples = []

  def OnSample(self, sample):
    self.samples.append(sample)
    return len(self.samples) < FLAGS.batch_size


def Sample(instance: clgen.Instance, db: grewe_features_db.Database,
           profiler: prof.AutoCsvProfiler, pool: multiprocessing.Pool):
  observer = SampleObserver()
  with profiler.Profile(f'Create {FLAGS.batch_size} samples'):
    samples = instance.model.Sample(instance.sampler, [observer])
  prefix = 'phd_experimental_deeplearning_'
  with tempfile.TemporaryDirectory(prefix=prefix) as d:
    d = pathlib.Path(d)
    with profiler.Profile(f'Create {FLAGS.batch_size} tempfiles'):
      paths_to_import = [
          CreateTempFileFromSample(d, sample, i)
          for i, sample in enumerate(observer.samples)
      ]
    with profiler.Profile() as p:
      num_successes = db.ImportStaticFeaturesFromPaths(paths_to_import,
                                                       FLAGS.origin)
      p.name = f'Import {num_successes} / {FLAGS.batch_size} samples'


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  instance = clgen.Instance(
      clgen_pb2.Instance(
          working_dir=FLAGS.clgen_dir,
          model=model_pb2.Model(
              corpus=corpus_pb2.Corpus(
                  local_directory=FLAGS.clgen_corpus_dir,
                  ascii_character_atomizer=True,
                  preprocessor=[
                      "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim",
                      "deeplearning.clgen.preprocessors.opencl:Compile",
                      "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers",
                      "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
                      "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines",
                      "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype",
                      "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace",
                      "deeplearning.clgen.preprocessors.opencl:ClangFormat",
                      "deeplearning.clgen.preprocessors.common:MinimumLineCount3",
                      "deeplearning.clgen.preprocessors.opencl:Compile",
                  ],
                  contentfile_separator='\n\n',
              ),
              architecture=model_pb2.NetworkArchitecture(
                  backend=model_pb2.NetworkArchitecture.TENSORFLOW,
                  neuron_type=model_pb2.NetworkArchitecture.LSTM,
                  neurons_per_layer=512,
                  num_layers=2,
                  post_layer_dropout_micros=0,
              ),
              training=model_pb2.TrainingOptions(
                  num_epochs=50,
                  sequence_length=64,
                  batch_size=64,
                  shuffle_corpus_contentfiles_between_epochs=True,
                  adam_optimizer=model_pb2.AdamOptimizer(
                      initial_learning_rate_micros=2000,
                      learning_rate_decay_per_epoch_micros=50000,
                      beta_1_micros=900000,
                      beta_2_micros=999000,
                      normalized_gradient_clip_micros=5000000,
                  ),
              ),
          ),
          sampler=sampler_pb2.Sampler(
              start_text="kernel void ",
              batch_size=64,
              sequence_length=1024,
              temperature_micros=1000000,  # = 1.0 real value
              termination_criteria=[
                  sampler_pb2.SampleTerminationCriterion(
                      symtok=sampler_pb2.SymmetricalTokenDepth(
                          depth_increase_token="{",
                          depth_decrease_token="}",
                      )),
                  sampler_pb2.SampleTerminationCriterion(
                      maxlen=sampler_pb2.MaxTokenLength(
                          maximum_tokens_in_sample=20000,)),
              ],
          ),
      ),)
  db = grewe_features_db.Database(FLAGS.db)
  profile_dir = pathlib.Path(FLAGS.profile_dir)
  profile_dir.mkdir(parents=True, exist_ok=True)
  profiler = prof.AutoCsvProfiler(profile_dir)

  with instance.Session(), multiprocessing.Pool() as pool:
    while True:
      Sample(instance, db, profiler, pool)


if __name__ == '__main__':
  app.RunWithArgs(main)
