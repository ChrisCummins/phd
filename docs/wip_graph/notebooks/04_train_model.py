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
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
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

# Experimental flags.

flags.DEFINE_integer('experimental_maximum_split_count', 0,
                     'If > 0, sets the number of splits before stopping.')


def GraphsTupleToStr(graph_tuple):
  """Format and print a GraphTuple"""
  return '\n'.join(
      [f'    {k:10s} {v}' for k, v in graph_tuple._asdict().items()])


def CreateVariableSummaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean', mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
  tf.summary.scalar('stddev', stddev)
  tf.summary.scalar('max', tf.reduce_max(var))
  tf.summary.scalar('min', tf.reduce_min(var))
  tf.summary.histogram('histogram', var)


def GetLearningRate(epoch_num: int) -> float:
  """Compute the learning rate.

  Args:
    epoch_num: The (zero-based) epoch number.

  Returns:
     A learning rate, in range (0,inf).
  """
  return FLAGS.initial_learning_rate * FLAGS.learning_rate_decay ** epoch_num


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
  (outdir / 'json_logs').mkdir(exist_ok=True, parents=True)

  tensorboard_outdir = outdir / 'tensorboard'

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)

  logging.info('Loaded %s dataframe from %s', df.shape, df_path)

  # Graph Model.
  lda = models.Lda()

  # Reset Session.
  seed = np.random.RandomState(FLAGS.seed)

  builder = tf.profiler.ProfileOptionBuilder
  opts = builder(builder.time_and_memory()).order_by('micros').build()
  opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
  profile_context = GetProfileContext(
      outdir / 'profile', FLAGS.profile_tensorflow)

  with profile_context as pctx, tf.Session() as sess:
    # Set up profiling. Note that if not FLAGS.profile_tensorflow, these
    # methods are no-ops.
    pctx.add_auto_profiling('op', opts, [15, 50, 100])
    pctx.add_auto_profiling('scope', opts2, [14, 49, 99])

    # Assemble model components into a tuple.
    model = Model(
        placeholders=InputTargetValue(input=input_ph, target=target_ph),
        loss_summary_op=TrainTestValue(
            train=loss_summary_op_tr, test=loss_summary_op_ge),
        step_op=step_op,
        loss_op=TrainTestValue(train=loss_op_tr, test=loss_op_ge),
        output_op=TrainTestValue(train=output_op_tr, test=output_op_ge),
        learning_rate=learning_rate, num_processing_steps=num_processing_steps,
    )

    # Split the data into independent train/test splits.
    splits = utils.TrainValidationTestSplits(
        df, rand=np.random.RandomState(FLAGS.seed))

    eval_data = []
    for i, split in enumerate(splits):
      # Experimental early exit using a flag. Use this for quickly running
      # reduced-size experiments. This will be removed later.
      if (FLAGS.experimental_maximum_split_count and
          i >= FLAGS.experimental_maximum_split_count):
        logging.warning("Terminating early because "
                        "--experimental_maximum_split_count=%d reached",
                        FLAGS.experimental_maximum_split_count)
        break

      cached_predictions_path = outdir / f'values/test_{i}.pkl'
      if cached_predictions_path.is_file():
        # Read the predictions made a previously trained model.
        with open(cached_predictions_path, 'rb') as f:
          predictions = pickle.load(f)
      else:
        # Create a new set of predictions.

        # Create the summary writers.
        writer_base_path = f'{tensorboard_outdir}/{split.gpu_name}'
        summary_writers = TrainTestValue(
            train=tf.summary.FileWriter(
                f'{writer_base_path}_train', sess.graph),
            test=tf.summary.FileWriter(f'{writer_base_path}_test', sess.graph))

        # Reset TensorFlow seed at every split, since we train and test each
        # split independently.
        tf.set_random_seed(FLAGS.seed + i)
        predictions = TrainAndEvaluate(sess, df, model, seed, summary_writers,
                                       outdir / 'json_logs')
        with open(outdir / f'values/test_{i}.pkl', 'wb') as f:
          pickle.dump(predictions, f)

      eval_data += utils.EvaluatePredictions(lda, split, predictions)
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
