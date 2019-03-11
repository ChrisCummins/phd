"""Implementation of graph networks for compilers."""
import collections
import contextlib
import json
import pathlib
import pickle
import time
import typing

import networkx as nx
import numpy as np
import pandas as pd
import sonnet as snt
import tensorflow as tf
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

from labm8 import app
from labm8 import labdate
from labm8 import prof

FLAGS = app.FLAGS

app.DEFINE_string(
    'df', '/tmp/phd/docs/wip_graph/lda_opencl_device_mapping_dataset.pkl',
    'Path of the dataframe to load')
app.DEFINE_string('outdir', '/tmp/phd/docs/wip_graph/model_files',
                  'Path of directory to generate files')
app.DEFINE_boolean('profile_tensorflow', False,
                   'Enable profiling of tensorflow.')
app.DEFINE_integer('model_seed', 0, 'Seed to use for reproducible results')
app.DEFINE_integer('num_epochs', 3, 'The number of epochs to train for.')
app.DEFINE_integer('batch_size', 64, 'Batch size.')
app.DEFINE_float('initial_learning_rate', 1e-3,
                 'The initial Adam learning rate.')
app.DEFINE_float(
    'learning_rate_exponential_decay', 0.1,
    'The rate at which learning decays. If 1.0, the learning rate does not '
    'decay.')

# Experimental flags.

app.DEFINE_integer('experimental_force_num_processing_steps', 0,
                   'If > 0, sets the number of processing steps.')
app.DEFINE_boolean(
    'experimental_graph_diameter_processing_steps', True,
    'Use the undirected CFG diameter to derive the number of '
    'processing steps.')
app.DEFINE_integer('experimental_mlp_model_latent_size', 16,
                   'Latent layer size in edge/node/global models.')
app.DEFINE_integer('experimental_mlp_model_layer_count', 2,
                   'Number of layers in edge/node/global models.')
app.DEFINE_boolean(
    'experimental_use_encode_process_decode_with_loop', False,
    'Use an experimental encode-process-decode model which '
    'uses a TensorFlow while loop rather than being unrolled '
    'for each time step')
app.DEFINE_integer(
    'experimental_while_loop_sequence_length', 10,
    'The number of unrolled steps inside while loop '
    'sequences. Only matters is '
    '--experimental_use_encode_process_decode_with_loop is '
    'set')

# A value which has different values for training, validation, and testing.
TrainingValidationTestValue = collections.namedtuple(
    'TrainingValidationTestValue', ['training', 'validation', 'test'])

# A value which has an input and target pair.
InputTargetValue = collections.namedtuple('InputTargetValue',
                                          ['input', 'target'])

InputLatentTargetValue = collections.namedtuple('InputLatentTargetValue',
                                                ['input', 'latent', 'target'])


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

  # Consider only the training graphs when determining step count.
  train_df = df[df['split:type'] == 'training']

  if FLAGS.experimental_graph_diameter_processing_steps:
    return int(train_df['cfg:diameter'].max())

  return int(train_df['cfg:block_count'].max())


def CreatePlaceholdersFromGraphs(
    input_graphs: typing.List[nx.DiGraph],
    target_graphs: typing.List[nx.DiGraph]) -> InputLatentTargetValue:
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


def MakeRunnableInSession(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [utils_tf.make_runnable_in_session(a) for a in args]


def GraphTupleToString(graph_tuple):
  """Format and a GraphTuple to string"""
  return '\n'.join(
      [f'    {k:10s} {v}' for k, v in graph_tuple._asdict().items()])


def CreateFeedDict(input_graphs, target_graphs, placeholders):
  """Creates placeholders for the model training and evaluation.

  Args:
      graphs: A list of graphs that will be inspected for vector sizes.

  Returns:
      The feed `dict` of input and target placeholders and data.
  """
  input_graphs = utils_np.networkxs_to_graphs_tuple(input_graphs)
  target_graphs = utils_np.networkxs_to_graphs_tuple(target_graphs)
  feed_dict = {
      placeholders.input: input_graphs,
      placeholders.target: target_graphs
  }
  return feed_dict


def GetLearningRate(epoch_num: int) -> float:
  """Compute the learning rate.

  Args:
    epoch_num: The (zero-based) epoch number.

  Returns:
     A learning rate, in range (0,inf).
  """
  return FLAGS.initial_learning_rate / (
      1 + FLAGS.learning_rate_exponential_decay * epoch_num)


def AssertDataFrameIsValidOrDie(df: pd.DataFrame) -> None:
  """Assert that the dataframe is valid else die."""
  for column in [
      'networkx:input_graph',
      'networkx:target_graph',
      'split:type',
      'graphnet:loss_op',
      'graphnet:accuracy_evaluator',
  ]:
    assert column in df.columns.values
  assert set(df['split:type']) == {'training', 'validation', 'test'}

  assert len(set(df['graphnet:loss_op'])) == 1
  assert len(set(df['graphnet:accuracy_evaluator'])) == 1


def MakeMultilayerPerceptron() -> snt.Sequential:
  """Instantiates a new MLP, followed by layer normalization.

  The parameters of each new MLP are not shared with others generated by
  this function.

  References:
    Layer normalization: https://arxiv.org/abs/1607.06450

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP(
          [FLAGS.experimental_mlp_model_latent_size] *
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
          edge_model_fn=MakeMultilayerPerceptron,
          node_model_fn=MakeMultilayerPerceptron,
          global_model_fn=MakeMultilayerPerceptron)

  def _build(self, inputs):
    return self._network(inputs)


class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
      self._network = modules.GraphNetwork(
          edge_model_fn=MakeMultilayerPerceptron,
          node_model_fn=MakeMultilayerPerceptron,
          global_model_fn=MakeMultilayerPerceptron,
          # TODO(cec): To disable redundant feature values:
          # edge_block_opt={
          #   'use_edges': False,
          #   'use_globals': False,
          # },
          # node_block_opt={
          #   'use_globals': False,
          # },
          # global_block_opt={
          #   'use_globals': False,
          # }
      )

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


class EncodeProcessDecodeUsingLoop(EncodeProcessDecode):
  """An implementation of an encode-process-decode architecture that uses
  a while loop to update the core recurrently, rather than the fully unrolled
  core of the example implementation.

  Based on @alvarosg's suggestion in:
      https://github.com/deepmind/graph_nets/issues/44
  """

  def _build(self, input_op, num_processing_steps):
    LoopVars = collections.namedtuple('LoopVars', ['i', 'latent', 'latent0'])

    imax = tf.constant(num_processing_steps)

    def Condition(v: LoopVars):
      """The while loop condition."""
      return tf.less(v.i, imax)

    def Body(v: LoopVars):
      """The while loop body."""
      # Partially unrolled core units.
      latent = v.latent
      for i in range(FLAGS.experimental_while_loop_sequence_length):
        core_input = utils_tf.concat([v.latent0, latent], axis=1)
        latent = self._core(core_input)

      return [
          LoopVars(
              i=tf.add(v.i, FLAGS.experimental_while_loop_sequence_length),
              latent=latent,
              latent0=v.latent0)
      ]

    init_latent = self._encoder(input_op)

    init_vars = LoopVars(i=0, latent=init_latent, latent0=init_latent)
    final_vars, = tf.while_loop(Condition, Body, [
        init_vars,
    ])

    decoded_op = self._decoder(final_vars.latent)
    output_op = self._output_transform(decoded_op)
    return output_op


def DataDictsToJsonSerializable(
    data_dicts: typing.List[typing.Dict[str, typing.Any]]
) -> typing.List[typing.Dict[str, typing.Any]]:
  """Make data dicts JSON serializable."""

  def RewriteNumpyTypes(data: typing.Dict[str, typing.Any]):
    """Rewrite numpy dict keys."""
    for key, val in data.items():
      if isinstance(val, np.ndarray):
        # Rewrite arrays as lists.
        data[key] = val.tolist()
      elif isinstance(val, np.int32):
        # Rewrite int32s as ints.
        data[key] = int(val)
      elif isinstance(val, np.int64):
        # Rewrite int64s as ints.
        data[key] = int(val)
      elif isinstance(val, dict):
        RewriteNumpyTypes(val)

  for data_dict in data_dicts:
    RewriteNumpyTypes(data_dict)

  return data_dicts


class LossOps(object):
  """Operations for producing model loss."""
  Type = typing.Callable[[graphs.GraphsTuple, graphs.GraphsTuple], tf.Tensor]

  @staticmethod
  def GlobalsSoftmaxCrossEntropy(target_ph: graphs.GraphsTuple,
                                 output_op: graphs.GraphsTuple) -> tf.Tensor:
    """Softmax cross entropy loss of graph globals."""
    return tf.losses.softmax_cross_entropy(target_ph.globals, output_op.globals)

  @staticmethod
  def NodesSoftmaxCrossEntropy(target_ph: graphs.GraphsTuple,
                               output_op: graphs.GraphsTuple) -> tf.Tensor:
    """Softmax cross entropy loss of graph nodes."""
    return tf.losses.softmax_cross_entropy(target_ph.nodes, output_op.nodes)


EvaluationResult = collections.namedtuple('EvaluationResult',
                                          ['accuracy', 'solved'])


class AccuracyEvaluators(object):
  """Methods to compute the accuracy of a model's outputs."""
  Type = typing.Callable[
      [typing.List[nx.DiGraph], typing.
       List[typing.Dict[str, typing.Any]]], EvaluationResult]

  @staticmethod
  def OneHotGlobals(target_graphs: typing.List[nx.DiGraph],
                    output_graphs: typing.List[typing.Dict[str, typing.Any]]
                   ) -> EvaluationResult:
    """Return accuracy of one-hot globals features."""
    targets = np.array([np.argmax(d.graph['features']) for d in target_graphs])
    outputs = np.array([np.argmax(d['globals']) for d in output_graphs])
    accuracy = (targets == outputs).mean()
    return EvaluationResult(float(accuracy), float(accuracy))

  @staticmethod
  def OneHotNodes(target_graphs: typing.List[nx.DiGraph],
                  output_graphs: typing.List[typing.Dict[str, typing.Any]]
                 ) -> EvaluationResult:
    """Return accuracy of one-hot globals features."""
    cs = []
    ss = []

    predicted_reachabilities = [
        np.argmax(d['nodes'], axis=-1) for d in output_graphs
    ]

    for target, predicted in zip(target_graphs, predicted_reachabilities):
      reachables = np.array(
          [np.argmax(n['features']) for _, n in target.nodes(data=True)])
      c = [reachables == predicted]
      c = np.concatenate(c, axis=0)
      s = np.all(c)
      cs.append(c)
      ss.append(s)

    accuracy = np.concatenate(cs, axis=0).mean()
    solved = np.stack(ss).mean()
    return EvaluationResult(float(accuracy), float(solved))


def GuessOutputSizesFromTargetGraph(graph: nx.DiGraph) -> typing.Dict[str, int]:
  global_features = graph.graph['features']
  for _, node in graph.nodes(data=True):
    node_features = node['features']
    break
  for _, _, edge in graph.edges(data=True):
    edge_features = edge['features']
    break

  def GuessOutputSize(features: np.ndarray) -> typing.Optional[int]:
    return len(features) if len(features) > 1 else None

  return {
      'global_output_size': GuessOutputSize(global_features),
      'node_output_size': GuessOutputSize(node_features),
      'edge_output_size': GuessOutputSize(edge_features),
  }


class CompilerGraphNeuralNetwork(object):

  def __init__(self, df: pd.DataFrame, outdir: pathlib.Path):
    """Instantiate a compiler graph neural network for the given dataset.

    DataFrame columns:
      split:type (str): The type of the data. One of "training", "validation",
        "test".
      networkx:input_graph (nx.DiGraph):
      networkx:target_graph (nx.DiGraph):

    Args:
      df: A DataFrame. See above for schema.
      outdir: A directory to write output files to.
      make_loss_op: A function that generates the loss op.
    """
    AssertDataFrameIsValidOrDie(df)

    # Lookup the loss op and evaluator functions from the table.
    make_loss_op = getattr(LossOps, df['graphnet:loss_op'].values[0])
    app.Info('Using loss op %s', make_loss_op.__name__)
    evaluate_outputs = getattr(AccuracyEvaluators,
                               df['graphnet:accuracy_evaluator'].values[0])
    app.Info('Using evaluator %s', evaluate_outputs.__name__)

    # Create output directories.
    (outdir / 'telemetry').mkdir(exist_ok=True, parents=True)
    (outdir / 'tensorboard').mkdir(exist_ok=True, parents=True)
    (outdir / 'test_outputs').mkdir(exist_ok=True, parents=True)

    # Get the number of message passing steps.
    num_processing_steps = GetNumberOfMessagePassingSteps(df)
    app.Info('Number of processing steps: %d', num_processing_steps)

    with prof.Profile('create placeholders'):
      input_ph, target_ph = CreatePlaceholdersFromGraphs(
          df['networkx:input_graph'], df['networkx:target_graph'])

    app.Debug("Input placeholders:\n%s", GraphTupleToString(input_ph))
    app.Debug("Target placeholders:\n%s", GraphTupleToString(target_ph))

    # Instantiate the model.
    with tf.name_scope('model'):
      init_args = GuessOutputSizesFromTargetGraph(
          df['networkx:target_graph'].values[0])

      # Enable support for experimental while-loop model architecture.
      if FLAGS.experimental_use_encode_process_decode_with_loop:
        model = EncodeProcessDecodeUsingLoop(**init_args)
      else:
        model = EncodeProcessDecode(**init_args)

      # Create loss ops.
      with prof.Profile('create output op'):
        output_op = model(input_ph, num_processing_steps)
      with prof.Profile('create training loss'):
        loss_op = make_loss_op(target_ph, output_op)
        loss_summary_op = tf.summary.scalar("loss", loss_op)

    # Optimizer and training step.
    with prof.Profile('create optimizer'), tf.name_scope('optimizer'):
      # Learning rate is a variable so that we can adjust it during training.
      learning_rate = tf.Variable(0.0, trainable=False)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      step_op = optimizer.minimize(loss_op)

    # Lets an iterable of TF graphs be output from a session as NP graphs.
    input_ph, target_ph = MakeRunnableInSession(input_ph, target_ph)

    self.df = df
    self.placeholders = InputTargetValue(input=input_ph, target=target_ph)
    self.loss_summary_op = loss_summary_op
    self.step_op = step_op
    self.loss_op = loss_op
    self.output_op = output_op
    self.learning_rate = learning_rate
    self.num_processing_steps = num_processing_steps
    self.outdir = outdir
    self.evaluate_outputs = evaluate_outputs

  def TrainAndEvaluate(self, sess: tf.Session) -> typing.List[nx.DiGraph]:
    """Train and evaluate a model.

    Args:
      sess: A TensorFlow session.

    Returns:
      A list of output graphs, one for each graph in the test set.
    """
    summary_writers = TrainingValidationTestValue(
        training=tf.summary.FileWriter(f'{self.outdir}/tensorboard/training',
                                       sess.graph),
        validation=tf.summary.FileWriter(
            f'{self.outdir}/tensorboard/validation', sess.graph),
        test=tf.summary.FileWriter(f'{self.outdir}/tensorboard/test',
                                   sess.graph))

    random_state = np.random.RandomState(FLAGS.model_seed)

    # Seed tensorflow and initialize model.
    tf.set_random_seed(FLAGS.model_seed)
    sess.run(tf.global_variables_initializer())

    # Split the dataframe into training, validation, and test data. Make a copy
    # of the training data so that we can shuffle it.
    train_df = self.df[self.df['split:type'] == 'training'].copy()
    validation_df = self.df[self.df['split:type'] == 'validation']
    test_df = self.df[self.df['split:type'] == 'test']

    app.Info("%d train graphs, %d validation graphs, %d test graphs",
             len(train_df), len(validation_df), len(test_df))

    with prof.Profile('train split'):
      batches = list(range(0, len(train_df), FLAGS.batch_size))

      # A counter of the global training step.
      tensorboard_step = 0

      for epoch_num in range(FLAGS.num_epochs):
        with prof.Profile('train epoch'):
          # Per-epoch log to be written to file.
          log = {
              # Model attributes. These are constant across epochs.
              'batch_size': FLAGS.batch_size,
              'num_processing_steps': self.num_processing_steps,
              'initial_learning_rate': FLAGS.initial_learning_rate,
              'learning_rate_exponential_decay':
              FLAGS.learning_rate_exponential_decay,
              # Dataset attributes. These are constant across epochs.
              'training_graph_count': len(train_df),
              'validation_graph_count': len(validation_df),
              'test_graph_count': len(test_df),
              # Per-epoch attributes.
              'epoch': epoch_num + 1,
              'learning_rate': GetLearningRate(epoch_num),
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
              'training_solved': 0,
              'validation_accuracy': 0,
              'validation_solved': 0,
              'test_accuracy': 0,
              'test_solved': 0,
          }

          # Set the learning rate based on the epoch number.
          sess.run(tf.assign(self.learning_rate, GetLearningRate(epoch_num)))

          # Iterate over all training data in batches.
          graphs_per_seconds = []
          output_graphs = []
          for j, b in enumerate(batches):
            batch_start_time = time.time()

            # Run a training set.
            input_graphs = train_df['networkx:input_graph'].iloc[b:b + FLAGS.
                                                                 batch_size]
            target_graphs = train_df['networkx:target_graph'].iloc[b:b + FLAGS.
                                                                   batch_size]
            num_graphs_processed = len(input_graphs)

            for s in range(0, self.num_processing_steps):
              feed_dict = CreateFeedDict(input_graphs, target_graphs,
                                         self.placeholders)
              train_values = sess.run({
                  "summary": self.loss_summary_op,
                  "step": self.step_op,
                  "target": self.placeholders.target,
                  "loss": self.loss_op,
                  "output": self.output_op,
              },
                                      feed_dict=feed_dict)

            output_graphs += utils_np.graphs_tuple_to_data_dicts(
                train_values["output"])
            summary_writers.training.add_summary(train_values['summary'],
                                                 tensorboard_step)

            batch_runtime = time.time() - batch_start_time
            graphs_per_second = num_graphs_processed / batch_runtime
            graphs_per_seconds.append(graphs_per_second)

            log['batch_runtime_ms'].append(int(batch_runtime * 1000))
            log['training_losses'].append(float(train_values['loss']))

            app.Info(
                'Epoch %02d / %02d, batch %02d / %02d in %.3fs (%02d graphs/sec), '
                'training loss: %.4f', epoch_num + 1, FLAGS.num_epochs, j + 1,
                len(batches), batch_runtime, int(graphs_per_second),
                train_values['loss'])
            tensorboard_step += 1

        log['training_accuracy'], log['training_solved'] = (
            self.evaluate_outputs(train_df['networkx:target_graph'],
                                  output_graphs))
        log['training_graphs_per_second'] = sum(graphs_per_seconds) / len(
            graphs_per_seconds)

        # End of epoch. Evaluate model on the validation and test set.

        # Validation set first.
        output_graphs, losses = [], []
        validation_start_time = time.time()
        num_graphs_processed = 0

        for j, b in enumerate(range(0, len(validation_df), FLAGS.batch_size)):
          input_graphs = validation_df['networkx:input_graph'].iloc[b:b + FLAGS.
                                                                    batch_size]
          target_graphs = validation_df['networkx:target_graph'].iloc[
              b:b + FLAGS.batch_size]
          num_graphs_processed += len(input_graphs)
          feed_dict = CreateFeedDict(input_graphs, target_graphs,
                                     self.placeholders)
          validation_values = sess.run({
              "summary": self.loss_summary_op,
              "target": self.placeholders.target,
              "loss": self.loss_op,
              "output": self.output_op,
          },
                                       feed_dict=feed_dict)
          output_graphs += utils_np.graphs_tuple_to_data_dicts(
              validation_values["output"])
          summary_writers.validation.add_summary(validation_values['summary'],
                                                 tensorboard_step)
          losses.append(float(validation_values['loss']))

        validation_runtime = time.time() - validation_start_time
        graphs_per_second = num_graphs_processed / validation_runtime
        log['validation_graphs_per_second'] = graphs_per_second

        eval_result = self.evaluate_outputs(
            validation_df['networkx:target_graph'], output_graphs)
        log['validation_accuracy'], log['validation_solved'] = eval_result
        log['validation_loss'] = sum(losses) / len(losses)
        log['validation_runtime_ms'] = validation_runtime * 1000

        app.Info(
            'Validation set in %.3f seconds (%02d graphs/sec), '
            'loss: %.4f, %.2f%% accuracy, %.2f%% solved', validation_runtime,
            graphs_per_second, log['validation_loss'],
            eval_result.accuracy * 100, eval_result.solved * 100)

        # Now the test set.
        output_graphs, losses = [], []
        test_start_time = time.time()
        num_graphs_processed = 0

        for j, b in enumerate(range(0, len(test_df), FLAGS.batch_size)):
          input_graphs = test_df['networkx:input_graph'].iloc[b:b +
                                                              FLAGS.batch_size]
          target_graphs = test_df['networkx:target_graph'].iloc[b:b + FLAGS.
                                                                batch_size]
          num_graphs_processed += len(input_graphs)
          feed_dict = CreateFeedDict(input_graphs, target_graphs,
                                     self.placeholders)
          test_values = sess.run({
              "summary": self.loss_summary_op,
              "target": self.placeholders.target,
              "loss": self.loss_op,
              "output": self.output_op,
          },
                                 feed_dict=feed_dict)
          output_graphs += utils_np.graphs_tuple_to_data_dicts(
              test_values["output"])
          summary_writers.test.add_summary(test_values['summary'],
                                           tensorboard_step)
          losses.append(float(test_values['loss']))

        test_runtime = time.time() - test_start_time
        graphs_per_second = num_graphs_processed / test_runtime
        log['test_graphs_per_second'] = graphs_per_second

        # Record the output graphs as JSON, and the accuracy of those outputs.
        eval_result = self.evaluate_outputs(test_df['networkx:target_graph'],
                                            output_graphs)
        log['test_accuracy'], log['test_solved'] = eval_result
        log['test_loss'] = sum(losses) / len(losses)
        log['test_runtime_ms'] = test_runtime * 1000

        app.Info(
            'Test set in %.3f seconds (%02d graphs/sec), '
            'loss: %.4f, %.2f%% accuracy, %.2f%% solved', test_runtime,
            graphs_per_second, log['test_loss'], eval_result.accuracy * 100,
            eval_result.solved * 100)

        # Dump epoch log to file.
        logpath = (f'{self.outdir}/telemetry/epoch_{epoch_num+1:03d}.'
                   f'T{labdate.MillisecondsTimestamp()}.json')
        with open(logpath, 'w') as f:
          json.dump(log, f)
        app.Info("Wrote %s", logpath)

        test_outputs_path = (
            f'{self.outdir}/test_outputs/epoch_{epoch_num+1:03d}.'
            f'T{labdate.MillisecondsTimestamp()}.pkl')
        with open(test_outputs_path, 'wb') as f:
          pickle.dump(output_graphs, f)

        # Shuffle the training data at the end of each epoch.
        train_df = train_df.sample(
            frac=1, random_state=random_state).reset_index(drop=True)

    # Return the output graphs from the final epoch.
    return output_graphs


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

  app.Info('Starting evaluating graph model')

  # Load graphs from file.
  df_path = pathlib.Path(FLAGS.df)
  assert df_path.is_file()

  assert FLAGS.outdir
  outdir = pathlib.Path(FLAGS.outdir)
  # Make the output directories.
  (outdir / 'values').mkdir(exist_ok=True, parents=True)

  with prof.Profile('load dataframe'):
    df = pd.read_pickle(df_path)
  app.Info('Loaded %s dataframe from %s', df.shape, df_path)

  # Prepare TensorFlow profiler.
  builder = tf.profiler.ProfileOptionBuilder
  opts = builder(builder.time_and_memory()).order_by('micros').build()
  opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
  profile_context = GetProfileContext(outdir / 'profile',
                                      FLAGS.profile_tensorflow)

  # Create the model.
  model = CompilerGraphNeuralNetwork(df, outdir)

  with profile_context as pctx, tf.Session() as sess:
    # Set up profiling. Note that if not FLAGS.profile_tensorflow, these
    # methods are no-ops.
    pctx.add_auto_profiling('op', opts, [15, 50, 100])
    pctx.add_auto_profiling('scope', opts2, [14, 49, 99])

    outputs_path = outdir / 'outputs.pkl'
    if not outputs_path.is_file():
      outputs = model.TrainAndEvaluate(sess)
      with open(outdir / 'outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == '__main__':
  app.RunWithArgs(main)
