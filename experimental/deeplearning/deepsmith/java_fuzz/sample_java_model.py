"""Train and sample a Java model."""
import pathlib

from deeplearning.clgen import clgen
from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8 import app
from labm8 import crypto
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_output_path(
    'java_clgen_working_dir',
    '/var/phd/experimental/deeplearning/deepsmith/java_fuzz/clgen_cache',
    'Path to store CLgen cache files.')
app.DEFINE_output_path(
    'java_sample_dir',
    '/var/phd/experimental/deeplearning/deepsmith/java_fuzz/samples',
    'Path to store CLgen samples.')
app.DEFINE_database(
    'java_encoded_contentfiles', encoded.EncodedContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/encoded.db',
    'URL of the database of encoded Java methods.')
app.DEFINE_integer('java_training_epochs', 50,
                   'The number of epochs to train for.')
app.DEFINE_string('java_seed_text', 'public ',
                  'The text to initialize sampling with.')


def MakeClgenInstance(
    working_dir: pathlib.Path, encoded_db: encoded.EncodedContentFiles,
    num_training_epochs: int, seed_text: str) -> clgen.Instance:
  """Construct a CLgen instance.

  Args:
    working_dir: The directory to cache CLgen working files in.
    encoded_db: The directory of encoded content files.
    num_training_epochs: The number of epochs to train for.
    seed_text: The text to initiate sampling with.
  """
  return clgen.Instance(
      clgen_pb2.Instance(
          working_dir=str(working_dir),
          model=model_pb2.Model(
              corpus=corpus_pb2.Corpus(pre_encoded_corpus_url=encoded_db.url,),
              architecture=model_pb2.NetworkArchitecture(
                  backend=model_pb2.NetworkArchitecture.TENSORFLOW,
                  neuron_type=model_pb2.NetworkArchitecture.LSTM,
                  neurons_per_layer=1024,
                  num_layers=2,
                  post_layer_dropout_micros=0,
              ),
              training=model_pb2.TrainingOptions(
                  num_epochs=num_training_epochs,
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
              start_text=seed_text,
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
      ))


class JavaSampleObserver(sample_observers_lib.SampleObserver):
  """Sample observer that writes samples to the given directory."""

  def __init__(self, outdir: pathlib.Path):
    self.outdir = outdir
    self.outdir.mkdir(parents=True, exist_ok=True)

  def OnSample(self, sample: model_pb2.Sample):
    checksum = crypto.sha1_str(sample.text)
    outpath = self.outdir / f'{checksum}.txt'
    pbutil.ToFile(sample, outpath)


def main():
  """Main entry point."""
  instance = MakeClgenInstance(FLAGS.java_clgen_working_dir,
                               FLAGS.java_encoded_contentfiles(),
                               FLAGS.java_training_epochs, FLAGS.java_seed_text)
  observer = JavaSampleObserver(FLAGS.java_sample_dir)

  with instance.Session():
    app.Log(1, 'Beginning model training')
    instance.model.Train()

    app.Log(1, 'Beginning infinite sampling loop.')
    instance.model.Sample(instance.sampler, [observer])


if __name__ == '__main__':
  app.Run(main)
