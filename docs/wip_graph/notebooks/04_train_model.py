"""Train LDA model.

Usage:

  bazel run //docs/wip_graph/notebooks:04_train_model -- --v=1
    --df=/var/phd/shared/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl
    --outdir=/var/phd/shared/docs/wip_graph/model_files
"""
import pathlib
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from graph_nets import utils_np as graph_net_utils_np
from graph_nets.demos import models as gn_models

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
flags.DEFINE_integer(
    'seed', 0, 'Seed to use for reproducible results')
flags.DEFINE_integer(
    'num_epochs', 3,
    'The number of epochs to train for.')
flags.DEFINE_integer(
    'batch_size', 32,
    'Batch size.')
flags.DEFINE_integer(
    'num_splits', 10,
    'The number of train/test splits per device. There are two devices, so '
    'the total number of splits evaluated will be 2 * num_splits.')

# Experimental flags.

flags.DEFINE_integer('experimental_force_num_processing_steps', 0,
                     'If > 0, sets the number of processing steps.')


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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Hello!')
  tf.logging.set_verbosity(tf.logging.DEBUG)

  # Load graphs from file.
  df_path = pathlib.Path(FLAGS.df)
  assert df_path.is_file()

  assert FLAGS.outdir
  outdir = pathlib.Path(FLAGS.outdir)
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / 'values').mkdir(exist_ok=True, parents=True)

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)

  logging.info('Loaded %s dataframe from %s', df.shape, df_path)

  # Graph Model.
  lda = models.Lda()

  # Get the number of message passing steps.
  num_processing_steps = lda.GetNumberOfMessagePassingSteps(
      df['lda:input_graph'], df['lda:target_graph'])
  if FLAGS.experimental_force_num_processing_steps:
    num_processing_steps = FLAGS.experimental_force_num_processing_steps

  logging.info('Number of processing steps: %d', num_processing_steps)

  tf.reset_default_graph()

  with prof.Profile('create placeholders'):
    input_ph, target_ph = lda.CreatePlaceholdersFromGraphs(
        df['lda:input_graph'], df['lda:target_graph'])

  logging.info("Input placeholders:\n%s", GraphsTupleToStr(input_ph))
  logging.info("Target placeholders:\n%s", GraphsTupleToStr(target_ph))

  # Connect the data to the model.
  # Instantiate the model.
  model = gn_models.EncodeProcessDecode(global_output_size=2)
  # A list of outputs, one per processing step.
  with prof.Profile('create train model'), tf.name_scope('training_model'):
    output_ops_tr = model(input_ph, num_processing_steps)
  with prof.Profile('create test model'), tf.name_scope('test_model'):
    output_ops_ge = model(input_ph, num_processing_steps)

  # Create loss ops.
  with prof.Profile('training loss'), tf.name_scope('training_loss'):
    loss_ops_tr = lda.CreateLossOps(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps
    CreateVariableSummaries(loss_ops_tr)

  with prof.Profile('test loss'), tf.name_scope('test_loss'):
    loss_ops_ge = lda.CreateLossOps(target_ph, output_ops_ge)
    # Loss from final processing step.
    loss_op_ge = loss_ops_ge[-1]
    CreateVariableSummaries(loss_op_ge)

  # Optimizer and training step.
  with prof.Profile('optimizer'):
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

  # Lets an iterable of TF graphs be output from a session as NP graphs.
  with prof.Profile('runnable'):
    input_ph, target_ph = lda.MakeRunnableInSession(input_ph, target_ph)

  # Reset Session.
  tf.set_random_seed(FLAGS.seed)
  sess = tf.Session()

  # Log writers.
  merged = tf.summary.merge_all()
  logging.info("Connect to this session with Tensorboard using:\n"
               "    python -m tensorboard.main --logdir='%s/tf_logs'", outdir)
  writer_tr = tf.summary.FileWriter(str(outdir / 'tf_logs/train'), sess.graph)
  writer_ge = tf.summary.FileWriter(str(outdir / 'tf_logs/test'), sess.graph)

  splits = utils.TrainTestSplitGenerator(
      df, seed=FLAGS.seed, split_count=FLAGS.num_splits)

  eval_data = []
  for i, split in enumerate(splits):
    # Each split must be evaluated independently of other splits since they
    # contain overlapping information. Reset the model at the start of each
    # split.
    sess.run(tf.global_variables_initializer())
    logging.info("Split %d / %d with %d train graphs, %d test graphs",
                 i + 1, 2 * FLAGS.num_splits, len(split.train_df),
                 len(split.test_df))

    with prof.Profile('train split'):
      batches = list(range(0, len(split.train_df), FLAGS.batch_size))
      with prof.Profile('train epoch'):
        for e in range(FLAGS.num_epochs):
          for j, b in enumerate(batches):
            feed_dict = lda.CreateFeedDict(
                split.train_df['lda:input_graph'].iloc[b:b + FLAGS.batch_size],
                split.train_df['lda:target_graph'].iloc[b:b + FLAGS.batch_size],
                input_ph, target_ph)
            train_values = sess.run({
              "summary": merged,
              "step": step_op,
              "target": target_ph,
              "loss": loss_op_tr,
              "outputs": output_ops_tr
            }, feed_dict=feed_dict)
            writer_tr.add_summary(train_values['summary'], i)
            logging.info('Split %d / %d, epoch %d / %d, batch %d / %d, '
                         'training loss: %.4f', i + 1, 2 * FLAGS.num_splits,
                         e + 1, FLAGS.num_epochs, j + 1, len(batches),
                         train_values['loss'])

        # Shuffle the training data at the end of each epoch.
        split.train_df = split.train_df.sample(frac=1).reset_index(drop=True)

    with prof.Profile('test split'):
      predictions = []
      for j, b in enumerate(range(0, len(split.test_df), FLAGS.batch_size)):
        feed_dict = lda.CreateFeedDict(
            split.test_df['lda:input_graph'].iloc[b:b + FLAGS.batch_size],
            split.test_df['lda:target_graph'].iloc[b:b + FLAGS.batch_size],
            input_ph, target_ph)
        test_values = sess.run({
          "summary": merged,
          "target": target_ph,
          "loss": loss_op_ge,
          "outputs": output_ops_ge
        }, feed_dict=feed_dict)
        writer_ge.add_summary(test_values['summary'], i)
        logging.info('Step %d, batch %d, test loss: %.4f', i, j + 1,
                     test_values['loss'])

        predictions += [
          np.argmax(d['globals']) for d in
          graph_net_utils_np.graphs_tuple_to_data_dicts(
              test_values["outputs"][-1])
        ]

      eval_data += utils.EvaluatePredictions(lda, split, predictions)

    with open(outdir / f'values/test_{i}.pkl', 'wb') as f:
      pickle.dump(predictions, f)

  df = utils.PredictionEvaluationsToTable(eval_data)
  with open(outdir / 'results.pkl', 'wb') as f:
    pickle.dump(df, f)
    logging.info("Results written to %s", outdir / 'results.pkl')

  heterogeneous_mapping.HeterogeneousMappingExperiment.PrintResultsSummary(df)

  logging.info("Connect to this session with Tensorboard using:\n"
               "    python -m tensorboard.main --logdir='%s/tf_logs'", outdir)
  logging.info('done')


if __name__ == '__main__':
  app.run(main)
