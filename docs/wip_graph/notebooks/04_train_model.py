"""Train LDA model.

Usage:

  bazel run //docs/wip_graph/notebooks:04_train_model -- --v=1
    --df=/var/phd/shared/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl
    --outdir=/var/phd/shared/docs/wip_graph/model_files
"""
import pathlib
import pickle

import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from graph_nets.demos import models as gn_models

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
flags.DEFINE_integer('batch_size', 32, 'Batch size')
flags.DEFINE_integer('num_splits', 10, 'The number of train splits')


def GraphsTupleToStr(graph_tuple):
  """Format and print a GraphTuple"""
  return '\n'.join(
      [f'    {k:10s} {v}' for k, v in graph_tuple._asdict().items()])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  logging.info('Hello!')
  tf.logging.set_verbosity(tf.logging.DEBUG)

  # Load graphs from file
  df_path = pathlib.Path(FLAGS.df)
  assert df_path.is_file()

  assert FLAGS.outdir
  outdir = pathlib.Path(FLAGS.outdir)
  outdir.mkdir(parents=True, exist_ok=True)

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)

  logging.info('Loaded %s dataframe from %s', df.shape, df_path)

  # Graph Model.
  lda = models.Lda()

  # Get the number of message passing steps.
  num_processing_steps = lda.GetNumberOfMessagePassingSteps(
      df['lda:input_graph'], df['lda:target_graph'])

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
  with prof.Profile('create train model'):
    output_ops_tr = model(input_ph, num_processing_steps)
  # with prof.Profile('create test model'):
  #   output_ops_ge = model(input_ph, num_processing_steps)

  # Create loss ops.
  with prof.Profile('training loss'):
    loss_ops_tr = lda.CreateLossOps(target_ph, output_ops_tr)
    # Loss across processing steps.
    loss_op_tr = sum(loss_ops_tr) / num_processing_steps

  # with prof.Profile('test loss'):
  #   loss_ops_ge = lda.CreateLossOps(target_ph, output_ops_ge)
  #   loss_op_ge = loss_ops_ge[-1]  # Loss from final processing step.

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

  sess.run(tf.global_variables_initializer())

  # Training step stats.
  step_stats = []

  splits = utils.TrainTestSplitGenerator(
      df, seed=FLAGS.seed, split_count=FLAGS.num_splits)

  for split in splits:
    logging.info("Split %d / %d with %d train graphs, %d test graphs",
                 split.i, 2 * FLAGS.num_splits, len(split.train_df),
                 len(split.test_df))

    with prof.Profile('train step'):
      feed_dict = lda.CreateFeedDict(
          split.train_df['lda:input_graph'], split.train_df['lda:target_graph'],
          input_ph, target_ph)
      train_values = sess.run({
        "step": step_op,
        "target": target_ph,
        "loss": loss_op_tr,
        "outputs": output_ops_tr
      }, feed_dict=feed_dict)

    with prof.Profile('save file'):
      path = outdir / f'train_{split.i}.pkl'
      logging.info('Writing %s', path)
      with open(path, 'wb') as f:
        pickle.dump(train_values, f)

    # with prof.Profile('test step'):
    #   feed_dict = lda.CreateFeedDict(
    #       split.test_df['lda:input_graph'], split.test_df['lda:target_graph'],
    #       input_ph, target_ph)
    #   test_values = sess.run({
    #     "step": step_op,
    #     "target": target_ph,
    #     "loss": loss_op_ge,
    #     "outputs": output_ops_tr
    #   }, feed_dict=feed_dict)

    # with prof.Profile('save file'):
    #   path = outdir / f'test_{split.i}.pkl'
    #   logging.info('Writing %s', path)
    #   with open(path, 'wb') as f:
    #     pickle.dump(test_values, f)

    # TODO(cec): Continue from here.
    break

  logging.info('done (for now)')
  #     eval_dicts_tr.append(train_results)
  #     eval_dicts_ge.append(test_results)
  #     print(f'#{split.i}  Train loss {train_values["loss"]:.4f}  '
  #           f'Test loss {test_values["loss"]:.4f}')


if __name__ == '__main__':
  app.run(main)
