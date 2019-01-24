"""Train LDA model.

Usage:

  bazel run //docs/wip_graph/notebooks:04_train_model -- --v=1
    --df=/var/phd/shared/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl
    --outdir=/var/phd/shared/docs/wip_graph/model_files
"""
import collections
import pathlib
import pickle

import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

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
    'batch_size', 64,
    'Batch size.')
flags.DEFINE_integer(
    'num_splits', 10,
    'The number of train/test splits per device. There are two devices, so '
    'the total number of splits evaluated will be 2 * num_splits.')

# Experimental flags.

flags.DEFINE_integer('experimental_force_num_processing_steps', 0,
                     'If > 0, sets the number of processing steps.')
flags.DEFINE_integer('experimental_maximum_split_count', 0,
                     'If > 0, sets the number of splits before stopping.')
flags.DEFINE_integer('experimental_mlp_model_latent_size', 16,
                     'Latent layer size in edge/node/global models.')
flags.DEFINE_integer('experimental_mlp_model_layer_count', 2,
                     'Number of layers in edge/node/global models.')


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


# A value which has different values for training and testing.
TrainTestValue = collections.namedtuple('TrainTestValue', ['train', 'test'])

InputTargetValue = collections.namedtuple('InputTargetValue',
                                          ['input', 'target'])

Model = collections.namedtuple('Model', [
  'placeholders',
  'loss_summary_op',
  'step_op',
  'loss_op',
  'output_op',
])


def TrainAndEvaluateSplit(sess: tf.Session, split: utils.TrainTestSplit,
                          model: Model, seed: np.random.RandomState,
                          summary_writers: TrainTestValue):
  # Reset the model at the start of each split - each split must be evaluated
  # independently of other splits since they contain overlapping information.
  sess.run(tf.global_variables_initializer())
  logging.info("Split %d / %d with %d train graphs, %d test graphs",
               split.global_step, 2 * FLAGS.num_splits, len(split.train_df),
               len(split.test_df))

  with prof.Profile('train split'):
    batches = list(range(0, len(split.train_df), FLAGS.batch_size))

    # A counter of the global training step.
    tensorboard_step = 0

    # Make a copy of the training data that we can shuffle.
    train_df = split.train_df.copy()

    for e in range(FLAGS.num_epochs):
      with prof.Profile('train epoch'):
        # Iterate over all training data in batches.
        for j, b in enumerate(batches):
          # Run a training set.
          feed_dict = models.Lda.CreateFeedDict(
              train_df['lda:input_graph'].iloc[b:b + FLAGS.batch_size],
              train_df['lda:target_graph'].iloc[b:b + FLAGS.batch_size],
              model.placeholders.input, model.placeholders.target)
          train_values = sess.run({
            "summary": model.loss_summary_op.train,
            "step": model.step_op,
            "target": model.placeholders.target,
            "loss": model.loss_op.train,
            "output": model.output_op.train,
          }, feed_dict=feed_dict)
          summary_writers.train.add_summary(
              train_values['summary'], tensorboard_step)

          if not j % 10:
            # Feed a single batch of the testing set through so that we can log
            # testing loss to tensorboard. These aren't the values we will be
            # using to return the actual predictions, we do that at the end of
            # training.
            feed_dict = models.Lda.CreateFeedDict(
                split.test_df['lda:input_graph'].iloc[:FLAGS.batch_size],
                split.test_df['lda:target_graph'].iloc[:FLAGS.batch_size],
                model.placeholders.input, model.placeholders.target)
            test_values = sess.run({
              "summary": model.loss_summary_op.test,
              "target": model.placeholders.target,
              "loss": model.loss_op.test,
              "output": model.output_op.test,
            }, feed_dict=feed_dict)
            summary_writers.test.add_summary(
              test_values['summary'], tensorboard_step)

          logging.info('Step %d / %d, epoch %d / %d, batch %d / %d, '
                       'training loss: %.4f, test loss: %.4f',
                       split.global_step, 2 * FLAGS.num_splits, e + 1,
                       FLAGS.num_epochs, j + 1, len(batches),
                       train_values['loss'], test_values['loss'])
          tensorboard_step += 1


      # Shuffle the training data at the end of each epoch.
      with prof.Profile('shuffle training data'):
        train_df = train_df.sample(
            frac=1, random_state=seed).reset_index(drop=True)

  predictions = []
  with prof.Profile('test split'):
    for j, b in enumerate(range(0, len(split.test_df), FLAGS.batch_size)):
      feed_dict = models.Lda.CreateFeedDict(
          split.test_df['lda:input_graph'].iloc[b:b + FLAGS.batch_size],
          split.test_df['lda:target_graph'].iloc[b:b + FLAGS.batch_size],
          model.placeholders.input, model.placeholders.target)
      test_values = sess.run({
        # "summary": model.summary_op,
        "target": model.placeholders.target,
        "loss": model.loss_op.test,
        "output": model.output_op.test,
      }, feed_dict=feed_dict)
      # FIXME(cec): Temporarily disabling test summaries.
      # summary_writers.test.add_summary(test_values['summary'], training_step)
      logging.info('Step %d, batch %d, test loss: %.4f', split.global_step,
                   j + 1, test_values['loss'])

      predictions += [
        np.argmax(d['globals']) for d in
        utils_np.graphs_tuple_to_data_dicts(test_values["output"])
      ]

  return predictions


def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.
  The parameters of each new MLP are not shared with others generated by
  this function.
  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([FLAGS.experimental_mlp_model_latent_size] *
        FLAGS.experimental_mlp_model_layer_count,
        activate_final=True),
      snt.LayerNorm()
  ])


class MLPGraphIndependent(snt.AbstractModule):
  """GraphIndependent with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphIndependent(
          edge_model_fn=make_mlp_model,
          node_model_fn=make_mlp_model,
          global_model_fn=make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(make_mlp_model, make_mlp_model,
                                           make_mlp_model)

  def _build(self, inputs):
    return self._network(inputs)


class EncodeProcessDecode(snt.AbstractModule):
  """Full encode-process-decode model.

  The model includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t"
    is the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on the final
    message-passing step.
                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(
          edge_fn, node_fn, global_fn)

  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)

    # We differ here from the demo graph net model in that we include only an
    # output for the final step of message passing, rather than an output for
    # each step of message passing.
    decoded_op = self._decoder(latent)
    output_op = self._output_transform(decoded_op)
    return output_op


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

  tensorboard_outdir = outdir / 'tensorboard'

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

  # Instantiate the model.
  with tf.name_scope('model'):
    model = EncodeProcessDecode(global_output_size=2)

  # Create loss ops.
  with tf.name_scope('train'):
    with prof.Profile('create train model'):
      # A list of outputs, one per processing step.
      output_op_tr = model(input_ph, num_processing_steps)
    with prof.Profile('create training loss'):
      loss_op_tr = tf.losses.softmax_cross_entropy(
        target_ph.globals, output_op_tr.globals)
      loss_summary_op_tr = tf.summary.scalar("loss", loss_op_tr)

  with tf.name_scope('test'):
    with prof.Profile('create test model'):
      output_op_ge = model(input_ph, num_processing_steps)
    with prof.Profile('create test loss'):
      loss_op_ge = tf.losses.softmax_cross_entropy(
        target_ph.globals, output_op_ge.globals)
      loss_summary_op_ge = tf.summary.scalar("loss", loss_op_ge)

  # Optimizer and training step.
  with prof.Profile('create optimizer'), tf.name_scope('optimizer'):
    learning_rate = 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

  # Lets an iterable of TF graphs be output from a session as NP graphs.
  with prof.Profile('runnable'):
    input_ph, target_ph = lda.MakeRunnableInSession(input_ph, target_ph)

  # Reset Session.
  seed = np.random.RandomState(FLAGS.seed)

  with tf.Session() as sess:
    # Log writers.
    logging.info("Connect to this session with Tensorboard using:\n"
                 "    python -m tensorboard.main --logdir='%s'",
                 tensorboard_outdir)

    # Assemble model components into a tuple.
    model = Model(
        placeholders=InputTargetValue(input=input_ph, target=target_ph),
        loss_summary_op=TrainTestValue(
            train=loss_summary_op_tr, test=loss_summary_op_ge),
        step_op=step_op,
        loss_op=TrainTestValue(train=loss_op_tr, test=loss_op_ge),
        output_op=TrainTestValue(train=output_op_tr, test=output_op_ge),
    )

    # Split the data into independent train/test splits.
    splits = utils.TrainTestSplitGenerator(
        df, seed=FLAGS.seed, split_count=FLAGS.num_splits)

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
        writer_base_path = f'{tensorboard_outdir}/{split.gpu_name}_{split.i}'
        summary_writers = TrainTestValue(
            train=tf.summary.FileWriter(f'{writer_base_path}_train',
                                        sess.graph),
            test=tf.summary.FileWriter(f'{writer_base_path}_test', sess.graph))

        # Reset TensorFlow seed at every split, since we train and test each
        # split independently.
        tf.set_random_seed(FLAGS.seed + i)
        predictions = TrainAndEvaluateSplit(sess, split, model, seed,
                                            summary_writers)
        with open(outdir / f'values/test_{i}.pkl', 'wb') as f:
          pickle.dump(predictions, f)

      eval_data += utils.EvaluatePredictions(lda, split, predictions)

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
