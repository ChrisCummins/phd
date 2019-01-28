"""Train LDA model.

Usage:

  bazel run //docs/wip_graph/notebooks:04_train_model -- --v=1
    --df=/var/phd/shared/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl
    --outdir=/var/phd/shared/docs/wip_graph/model_files
"""
import contextlib
import pathlib
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  heterogeneous_mapping
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import lda
from experimental.compilers.reachability import graph_model
from labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'df', '/tmp/phd/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl',
    'Path of the dataframe to load')
flags.DEFINE_string(
    'outdir', '/tmp/phd/docs/wip_graph/model_files',
    'Path of directory to generate files')
flags.DEFINE_bool(
    'profile_tensorflow', False,
    'Enable profiling of tensorflow.')


class NoOpProfileContext():
  """A profiling context which does no profiling.

  This is used a return value of ProfileContext() which allows the
  unconditional execution of profiling code, irrespective of profiling being
  enabled.
  """

  def add_auto_profiling(self, *args, **kwargs):
    """No-op."""
    pass


@contextlib.contextmanager
def GetProfileContext(outdir: pathlib.Path, profile: bool = True):
  """Return a profile context."""
  if profile:
    yield tf.contrib.tfprof.ProfileContext(str(outdir))
  else:
    yield NoOpProfileContext()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Hello!')
  # tf.logging.set_verbosity(tf.logging.DEBUG)

  # Load graphs from file.
  df_path = pathlib.Path(FLAGS.df)
  assert df_path.is_file()

  assert FLAGS.outdir
  outdir = pathlib.Path(FLAGS.outdir)
  # Make the output directories.
  (outdir / 'values').mkdir(exist_ok=True, parents=True)

  tensorboard_outdir = outdir / 'tensorboard'

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)

  logging.info('Loaded %s dataframe from %s', df.shape, df_path)

  # Reset Session.

  builder = tf.profiler.ProfileOptionBuilder
  opts = builder(builder.time_and_memory()).order_by('micros').build()
  opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
  profile_context = GetProfileContext(
      outdir / 'profile', FLAGS.profile_tensorflow)

  model = graph_model.CompilerGraphNeuralNetwork(
      df, outdir, graph_model.LossOps.GlobalsSoftmaxCrossEntropy,
      graph_model.AccuracyEvaluators.OneHotGlobals)

  with profile_context as pctx, tf.Session() as sess:
    # Set up profiling. Note that if not FLAGS.profile_tensorflow, these
    # methods are no-ops.
    pctx.add_auto_profiling('op', opts, [15, 50, 100])
    pctx.add_auto_profiling('scope', opts2, [14, 49, 99])

    cached_predictions_path = outdir / f'values/test_{i}.pkl'
    if cached_predictions_path.is_file():
      # Read the predictions made a previously trained model.
      with open(cached_predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    else:
      outputs = model.TrainAndEvaluate(sess)
      predictions = np.array([
        np.argmax(d['globals']) for d in outputs
      ])
      with open(outdir / f'values/test_{i}.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    eval_data = utils.EvaluatePredictions(lda.Lda(), df, predictions)
  # End of TensorFlow session scope.

  df = utils.PredictionEvaluationsToTable(eval_data)
  with open(outdir / 'results.pkl', 'wb') as f:
    pickle.dump(df, f)
    logging.info("Results written to %s", outdir / 'results.pkl')

  heterogeneous_mapping.HeterogeneousMappingExperiment.PrintResultsSummary(df)

  gpu_predicted_count = sum(df['Predicted Mapping'])
  cpu_predicted_count = len(df) - gpu_predicted_count

  logging.info("Final predictions count: cpu=%d, gpu=%d", cpu_predicted_count,
               gpu_predicted_count)
  logging.info("Connect to this session with Tensorboard using:\n"
               "    python -m tensorboard.main --logdir='%s'",
               tensorboard_outdir)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
