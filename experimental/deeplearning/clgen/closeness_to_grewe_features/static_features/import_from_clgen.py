"""Sample and import from CLgen."""
import pathlib
import tempfile
import typing

from absl import app
from absl import flags

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from experimental.deeplearning.clgen.closeness_to_grewe_features import \
  grewe_features_db


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'db',
    'sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db',
    'URL of the database to import OpenCL kernels to.')
flags.DEFINE_integer('batch_size', 512, 'Number of samples to make per batch.')
flags.DEFINE_string(
    'origin', 'clgen',
    'Name of the origin of the kernels, e.g. "clgen".')
flags.DEFINE_string(
    'clgen_dir', '~/.cache/clgen',
    'Name of the origin of the kernels, e.g. "clgen".')
flags.DEFINE_string(
    'clgen_corpus_dir', "/mnt/cc/data/datasets/github/corpuses/opencl",
    "WHere the corpus is stored.")


def CreateTempFileFromSample(
    tempdir: pathlib.Path, sample: model_pb2.Sample,
    number: int) -> pathlib.Path:
  """Write testcase to a file in directory."""
  path = tempdir / f'{number}.cl'
  with open(path, 'w') as f:
    f.write(sample.text)
  return path


def Sample(instance: clgen.Instance, db: grewe_features_db.Database):
  samples = instance.model.SampleFast(
      instance.sampler, min_num_samples=FLAGS.batch_size)
  prefix = 'phd_experimental_deeplearning_'
  with tempfile.TemporaryDirectory(prefix=prefix) as d:
    d = pathlib.Path(d)
    paths_to_import = [
      CreateTempFileFromSample(d, sample, i)
      for i, sample in enumerate(samples)
    ]
    db.ImportStaticFeaturesFromPaths(paths_to_import, FLAGS.origin,
                                     multiprocess=False)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  instance = clgen.Instance(clgen_pb2.Instance(
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
              )
          ),
      ),
      sampler=sampler_pb2.Sampler(
          start_text="kernel void ",
          batch_size=1,
          temperature_micros=1000000,  # = 1.0 real value
          termination_criteria=[
            sampler_pb2.SampleTerminationCriterion(
                symtok=sampler_pb2.SymmetricalTokenDepth(
                    depth_increase_token="{",
                    depth_decrease_token="}",
                )),
            sampler_pb2.SampleTerminationCriterion(
                maxlen=sampler_pb2.MaxTokenLength(
                    maximum_tokens_in_sample=10000,
                )),
          ],
      )))
  db = grewe_features_db.Database(FLAGS.db)

  with instance.Session():
    while True:
      Sample(instance, db)


if __name__ == '__main__':
  app.run(main)
