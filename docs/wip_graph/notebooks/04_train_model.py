"""Train LDA model.

Usage:

  bazel run //docs/wip_graph/notebooks:04_train_model -- --v=1
    --df=/var/phd/shared/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl
    --outdir=/var/phd/shared/docs/wip_graph/model_files
"""
import collections
import contextlib
import json
import pathlib
import pickle
import time

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
flags.DEFINE_bool(
    'profile_tensorflow', False,
    'Enable profiling of tensorflow.')
flags.DEFINE_float(
    'initial_learning_rate', 1e-3,
    'The initial Adam learning rate.')
flags.DEFINE_float(
    'learning_rate_decay', 0.95,
    'The rate at which learning decays. If 1.0, the learning rate does not '
    'decay.')

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
  'learning_rate',
])


def GetLearningRate(epoch_num: int) -> float:
  """Compute the learning rate.

  Args:
    epoch_num: The (zero-based) epoch number.

  Returns:
     A learning rate, in range (0,inf).
  """
  return FLAGS.initial_learning_rate * FLAGS.learning_rate_decay ** epoch_num


def TrainAndEvaluateSplit(sess: tf.Session, split: utils.TrainTestSplit,
                          model: Model, seed: np.random.RandomState,
                          summary_writers: TrainTestValue,
                          outdir: pathlib.Path):
  # Reset the model at the start of each split - each split must be evaluated
  # independently of other splits since they contain overlapping information.
  sess.run(tf.global_variables_initializer())
  logging.info("%d split with %d train graphs, %d validation graphs, "
               "%d test graphs", split.gpu_name, len(split.train_df),
               len(split.valid_df), len(split.test_df))

  with prof.Profile('train split'):
    batches = list(range(0, len(split.train_df), FLAGS.batch_size))

    # A counter of the global training step.
    tensorboard_step = 0

    # Make a copy of the training data that we can shuffle.
    train_df = split.train_df.copy()

    for e in range(FLAGS.num_epochs):
      with prof.Profile('train epoch'):
        # Per-epoch log to be written to file.
        log = {
          'epoch': e + 1,
          'gpu_name': split.gpu_name,
          'batch_runtime_ms': [],
          'validation_runtime_ms': 0,
          'test_runtime_ms': 0,
          'training_graphs_per_second': 0,
          'validation_graphs_per_second': 0,
          'test_graphs_per_second': 0,
          'training_graph_count': len(split.train_df),
          'validation_graph_count': len(split.valid_df),
          'test_graph_count': len(split.test_df),
          'training_losses': [],
          # Average loss over all batches in validation and test sets.
          'validation_loss': 0,
          'test_loss': 0,
          # Each model output on the test set.
          'test_outputs': [],
          'validation_accuracy': 0,
          'test_accuracy': 0,
          'learning_rate': GetLearningRate(e),
        }

        # Set the new learning rate.
        sess.run(tf.assign(model.learning_rate, GetLearningRate(e)))

        # Iterate over all training data in batches.
        graphs_per_seconds = []
        for j, b in enumerate(batches):
          batch_start_time = time.time()

          # Run a training set.
          input_graphs = train_df['lda:input_graph'].iloc[
                         b:b + FLAGS.batch_size]
          target_graphs = train_df['lda:target_graph'].iloc[
                          b:b + FLAGS.batch_size]
          num_graphs_processed = len(input_graphs)
          feed_dict = models.Lda.CreateFeedDict(
              input_graphs, target_graphs,
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

          batch_runtime = time.time() - batch_start_time
          graphs_per_second = num_graphs_processed / batch_runtime
          graphs_per_seconds.append(graphs_per_second)

          log['batch_runtime_ms'].append(int(batch_runtime * 1000))
          log['training_losses'].append(float(train_values['loss']))

          logging.info('Split %s in %.3fs (%02d graphs/sec), '
                       'epoch %02d / %02d, batch %02d / %02d, '
                       'training loss: %.4f',
                       split.gpu_name, batch_runtime, int(graphs_per_second),
                       e + 1, FLAGS.num_epochs, j + 1, len(batches),
                       train_values['loss'])
          tensorboard_step += 1

      log['training_graphs_per_second'] = sum(graphs_per_seconds) / len(
          graphs_per_seconds)

      # End of epoch. Get predictions for the validation and test sets.

      # Validation set first.
      outputs, ground_truth, losses = [], [], []
      validation_start_time = time.time()
      validation_runtime = 0
      num_graphs_processed = 0

      for j, b in enumerate(range(0, len(split.valid_df), FLAGS.batch_size)):
        input_graphs = split.valid_df['lda:input_graph'].iloc[
                       b:b + FLAGS.batch_size]
        target_graphs = split.valid_df['lda:target_graph'].iloc[
                        b:b + FLAGS.batch_size]
        num_graphs_processed += len(input_graphs)
        feed_dict = models.Lda.CreateFeedDict(
            input_graphs, target_graphs,
            model.placeholders.input, model.placeholders.target)
        validation_values = sess.run({
          "target": model.placeholders.target,
          "loss": model.loss_op.test,
          "output": model.output_op.test,
        }, feed_dict=feed_dict)
        losses.append(float(validation_values['loss']))

        # Exclude data wrangling from prediction time - we're only interested
        # in measuring inference rate, not the rate of the python util code.
        validation_runtime += time.time() - validation_start_time
        ground_truth += [
          int(np.argmax(d.graph['features'])) for d in target_graphs
        ]
        outputs += [
          d['globals'] for d in
          utils_np.graphs_tuple_to_data_dicts(validation_values["output"])
        ]
        validation_start_time = time.time()
      graphs_per_second = num_graphs_processed / validation_runtime
      log['validation_graphs_per_second'] = graphs_per_second

      predictions = [int(np.argmax(output)) for output in outputs]
      accuracy = sum(
          np.array(ground_truth) == np.array(predictions)) / len(predictions)
      log['validation_accuracy'] = accuracy
      log['validation_loss'] = sum(losses) / len(losses)

      logging.info('validation set in %.3f seconds (%02d graphs/sec), '
                   '%.3f%% accuracy', validation_runtime, graphs_per_second,
                   accuracy * 100)

      # Now the test set.
      outputs, ground_truth, losses = [], [], []
      test_start_time = time.time()
      test_runtime = 0
      num_graphs_processed = 0

      for j, b in enumerate(range(0, len(split.test_df), FLAGS.batch_size)):
        input_graphs = split.test_df['lda:input_graph'].iloc[
                       b:b + FLAGS.batch_size]
        target_graphs = split.test_df['lda:target_graph'].iloc[
                        b:b + FLAGS.batch_size]
        num_graphs_processed += len(input_graphs)
        feed_dict = models.Lda.CreateFeedDict(
            input_graphs, target_graphs,
            model.placeholders.input, model.placeholders.target)
        test_values = sess.run({
          "summary": model.loss_summary_op.test,
          "target": model.placeholders.target,
          "loss": model.loss_op.test,
          "output": model.output_op.test,
        }, feed_dict=feed_dict)
        summary_writers.test.add_summary(
            test_values['summary'], tensorboard_step)
        losses.append(float(validation_values['loss']))

        # Exclude data wrangling from prediction time - we're only interested
        # in measuring inference rate, not the rate of the python util code.
        test_runtime += time.time() - test_start_time
        ground_truth += [
          int(np.argmax(d.graph['features'])) for d in target_graphs
        ]
        outputs += [
          # Convert from np.array of float32 to a list of floats for JSON
          # serialization.
          d['globals'].astype(float).tolist() for d in
          utils_np.graphs_tuple_to_data_dicts(test_values["output"])
        ]
        test_start_time = time.time()
      graphs_per_second = num_graphs_processed / test_runtime
      log['test_graphs_per_second'] = graphs_per_second

      predictions = [int(np.argmax(output)) for output in outputs]

      # Record the raw model output predictions, and the accuracy of those
      # predictions.
      log['test_outputs'] = outputs
      accuracy = sum(
          np.array(ground_truth) == np.array(predictions)) / len(predictions)
      log['test_accuracy'] = accuracy
      log['test_loss'] = sum(losses) / len(losses)

      logging.info('test split in %.3f seconds (%02d graphs/sec), '
                   '%.3f%% accuracy', test_runtime, graphs_per_second,
                   accuracy * 100)

      # Dump epoch log to file.
      log_name = f'{split.gpu_name}.epoch_{e+1:03d}.json'
      with open(outdir / log_name, 'w') as f:
        json.dump(log, f)

      # Shuffle the training data at the end of each epoch.
      train_df = train_df.sample(
          frac=1, random_state=seed).reset_index(drop=True)

  # Return the predictions from the final epoch.
  return predictions


def MakeMlpModel():
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
          edge_model_fn=MakeMlpModel,
          node_model_fn=MakeMlpModel,
          global_model_fn=MakeMlpModel)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(
          edge_model_fn=MakeMlpModel,
          node_model_fn=MakeMlpModel,
          global_model_fn=MakeMlpModel)

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
      self._encoder = MLPGraphIndependent(name="encoder")
      self._core = MLPGraphNetwork(name="core")
      self._decoder = MLPGraphIndependent(name="decoder")
      self._output_transform = modules.GraphIndependent(
          edge_fn, node_fn, global_fn, name="output_transform")

  def _build(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)

    # We differ here from the demo graph net model in that we include only an
    # output for the final step of message passing, rather than an output for
    # each step of message passing.
    decoded_op = self._decoder(latent)
    output_op = self._output_transform(decoded_op)
    return output_op


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

  logging.debug("Input placeholders:\n%s", GraphsTupleToStr(input_ph))
  logging.debug("Target placeholders:\n%s", GraphsTupleToStr(target_ph))

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
    # Learning rate is a variable so that we can adjust it during training.
    learning_rate = tf.Variable(0.0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    step_op = optimizer.minimize(loss_op_tr)

  # Lets an iterable of TF graphs be output from a session as NP graphs.
  with prof.Profile('runnable'):
    input_ph, target_ph = lda.MakeRunnableInSession(input_ph, target_ph)

  # Reset Session.
  seed = np.random.RandomState(FLAGS.seed)

  params = {
    'num_processing_steps': num_processing_steps,
  }

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
        learning_rate=learning_rate,
    )

    # Split the data into independent train/test splits.
    splits = utils.TrainValidationTestSplits(
        df, seed=np.random.RandomState(FLAGS.seed))

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
        predictions = TrainAndEvaluateSplit(
            sess, split, model, seed, summary_writers, outdir / 'json_logs')
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
