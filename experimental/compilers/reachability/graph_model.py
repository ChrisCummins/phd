"""Implementation of graph networks for compilers."""
import collections
import json
import pathlib
import time

import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from absl import flags
from absl import logging
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'seed', 0, 'Seed to use for reproducible results')
flags.DEFINE_integer(
    'num_epochs', 3,
    'The number of epochs to train for.')
flags.DEFINE_integer(
    'batch_size', 64,
    'Batch size.')
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
flags.DEFINE_integer('experimental_mlp_model_latent_size', 16,
                     'Latent layer size in edge/node/global models.')
flags.DEFINE_integer('experimental_mlp_model_layer_count', 2,
                     'Number of layers in edge/node/global models.')

# A value which has different values for training and testing.
TrainTestValue = collections.namedtuple('TrainTestValue', ['train', 'test'])

InputTargetValue = collections.namedtuple('InputTargetValue',
                                          ['input', 'target'])


def GetNumberOfMessagePassingSteps(df: pd.DataFrame) -> int:
  """Get the number of message passing steps for a dataframe of graphs.

  For now, we use the number of edges in the graph as the number of
  Experimentally, we find this to return a value in the O(thousands) range for
  "real" LLVM full flow graphs.

  In the future, we may want to compute this on a per-graph basis. OR we may
  want to determine this dynamically, by continually passing messages around the
  graph until we reach a fixed point.
  """
  if FLAGS.experimental_force_num_processing_steps:
    return FLAGS.experimental_force_num_processing_steps
  return max([g.number_of_edges() for g in df['lda:input_graph']])


def CreatePlaceholdersFromGraphs(input_graphs: typing.List[nx.DiGraph],
                                 target_graphs: typing.List[nx.DiGraph]):
  """Creates placeholders for the model training and evaluation.

  Args:
    input_graphs: A list of input graphs.
    target_graphs: A list of input graphs.

  Returns:
    A tuple of the input graph's and target graph's placeholders, as a
    graph namedtuple.
  """
  input_ph = utils_tf.placeholders_from_networkxs(
      input_graphs, force_dynamic_num_graphs=True, name="input_ph")
  target_ph = utils_tf.placeholders_from_networkxs(
      target_graphs, force_dynamic_num_graphs=True, name="target_ph")
  return InputTargetValue(input_ph, target_ph)


class CompilerGraphNeuralNetwork(object):
  def __init__(self, placeholders, loss_summary_op, step_op, loss_op, output_op,
               learning_rate, num_processing_steps):
    self.placeholders = placeholders
    self.loss_summary_op = loss_summary_op
    self.step_op = step_op
    self.loss_op = loss_op
    self.output_op = output_op
    self.learning_rate = learning_rate
    self.num_processing_steps = num_processing_steps

  @classmethod
  def FromDataFrame(cls, sess: tf.Session, df: pd.DataFrame):
    # Get the number of message passing steps.
    num_processing_steps = GetNumberOfMessagePassingSteps(df)
    logging.info('Number of processing steps: %d', num_processing_steps)

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

    # Assemble model components into a tuple.
    return cls(
        placeholders=InputTargetValue(input=input_ph, target=target_ph),
        loss_summary_op=TrainTestValue(
            train=loss_summary_op_tr, test=loss_summary_op_ge),
        step_op=step_op,
        loss_op=TrainTestValue(train=loss_op_tr, test=loss_op_ge),
        output_op=TrainTestValue(train=output_op_tr, test=output_op_ge),
        learning_rate=learning_rate, num_processing_steps=num_processing_steps,
    )


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


def TrainAndEvaluate(sess: tf.Session, df: pd.DataFrame,
                     model: Model, seed: np.random.RandomState,
                     summary_writers: TrainTestValue,
                     logdir: pathlib.Path) -> None:
  """Train and evaluate a model.

  DataFrame columns:
    split:type (str): The type of the data. One of "training", "validation",
      "test".
    lda:input_graph (nx.DiGraph):
    lda:target_graph (nx.DiGraph):

  Args:
    sess: A TensorFlow session.
    df: The DataFrame, see above for description of dataframe.
    model:
    seed:
    summary_writers:
    logdir: A directory to write log files to.
  """
  # Reset the model at the start of each split - each split must be evaluated
  # independently of other splits since they contain overlapping information.
  sess.run(tf.global_variables_initializer())

  # Split the dataframe into training, validation, and test data. Make a copy of
  # the training data that we can shuffle it.
  train_df = df[df['split'] == 'training'].copy()
  validation_df = df[df['split'] == 'validation']
  test_df = df[df['split'] == 'test']

  logging.info("%d train graphs, %d validation graphs, %d test graphs",
               len(train_df), len(validation_df), len(test_df))

  with prof.Profile('train split'):
    batches = list(range(0, len(train_df), FLAGS.batch_size))

    # A counter of the global training step.
    tensorboard_step = 0

    for e in range(FLAGS.num_epochs):
      with prof.Profile('train epoch'):
        # Per-epoch log to be written to file.
        log = {
          # Model attributes. These are constant across epochs.
          'batch_size': FLAGS.batch_size,
          'num_processing_steps': model.num_processing_steps,
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'learning_rate_decay': FLAGS.learning_rate_decay,
          'dataframe': FLAGS.df,
          # Dataset attributes. These are constant across epochs.
          'training_graph_count': len(train_df),
          'validation_graph_count': len(validation_df),
          'test_graph_count': len(test_df),
          # Per-epoch attributes.
          'epoch': e + 1,
          'learning_rate': GetLearningRate(e),
          # Runtime metrics.
          'batch_runtime_ms': [],
          'validation_runtime_ms': 0,
          'test_runtime_ms': 0,
          # Throughput metrics.
          'training_graphs_per_second': 0,
          'validation_graphs_per_second': 0,
          'test_graphs_per_second': 0,
          # Model losses. Record all training losses. Average validation and
          # test losses across batches.
          'training_losses': [],
          'validation_loss': 0,
          'test_loss': 0,
          # Accuracies of model on training / validation / test data.
          'training_accuracy': 0,
          'validation_accuracy': 0,
          'test_accuracy': 0,
          # Each model output on the test set.
          'test_outputs': [],
        }

        # Set the learning rate based on the epoch number.
        sess.run(tf.assign(model.learning_rate, GetLearningRate(e)))

        # Iterate over all training data in batches.
        graphs_per_seconds = []
        correct = []
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

          ground_truth = np.array([
            np.argmax(d.graph['features']) for d in target_graphs
          ])
          predictions = np.array([
            np.argmax(d['globals']) for d in
            utils_np.graphs_tuple_to_data_dicts(train_values["output"])
          ])
          correct += (ground_truth == predictions).tolist()

          logging.info('Batch %d in %.3fs (%02d graphs/sec), '
                       'epoch %02d / %02d, batch %02d / %02d, '
                       'training loss: %.4f',
                       tensorboard_step, batch_runtime, int(graphs_per_second),
                       e + 1, FLAGS.num_epochs, j + 1, len(batches),
                       train_values['loss'])
          tensorboard_step += 1

      log['training_accuracy'] = sum(correct) / len(correct)
      log['training_graphs_per_second'] = sum(graphs_per_seconds) / len(
          graphs_per_seconds)

      # End of epoch. Get predictions for the validation and test sets.

      # Validation set first.
      correct, losses = [], []
      validation_start_time = time.time()
      validation_runtime = 0
      num_graphs_processed = 0

      for j, b in enumerate(range(0, len(validation_df), FLAGS.batch_size)):
        input_graphs = validation_df['lda:input_graph'].iloc[
                       b:b + FLAGS.batch_size]
        target_graphs = validation_df['lda:target_graph'].iloc[
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
        ground_truth = np.array([
          np.argmax(d.graph['features']) for d in target_graphs
        ])
        predictions = np.array([
          np.argmax(d['globals']) for d in
          utils_np.graphs_tuple_to_data_dicts(validation_values["output"])
        ])
        correct += (ground_truth == predictions).tolist()
        validation_start_time = time.time()
      graphs_per_second = num_graphs_processed / validation_runtime
      log['validation_graphs_per_second'] = graphs_per_second

      accuracy = sum(correct) / len(correct)
      log['validation_accuracy'] = accuracy
      log['validation_loss'] = sum(losses) / len(losses)
      log['validation_runtime_ms'] = validation_runtime * 1000

      logging.info('validation set in %.3f seconds (%02d graphs/sec), '
                   '%.3f%% accuracy', validation_runtime, graphs_per_second,
                   accuracy * 100)

      # Now the test set.
      outputs, ground_truth, losses = [], [], []
      test_start_time = time.time()
      test_runtime = 0
      num_graphs_processed = 0

      for j, b in enumerate(range(0, len(test_df), FLAGS.batch_size)):
        input_graphs = test_df['lda:input_graph'].iloc[b:b + FLAGS.batch_size]
        target_graphs = test_df['lda:target_graph'].iloc[b:b + FLAGS.batch_size]
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
      log['test_runtime_ms'] = test_runtime * 1000

      logging.info('test split in %.3f seconds (%02d graphs/sec), '
                   '%.3f%% accuracy', test_runtime, graphs_per_second,
                   accuracy * 100)

      # Dump epoch log to file.
      log_name = f'epoch_{e+1:03d}.json'
      with open(logdir / log_name, 'w') as f:
        json.dump(log, f)

      # Shuffle the training data at the end of each epoch.
      train_df = train_df.sample(
          frac=1, random_state=seed).reset_index(drop=True)

  # Return the predictions from the final epoch.
  return predictions
