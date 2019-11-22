"""Base class for implementing gated graph neural networks."""
import math
import typing

import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.models import base_utils
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8 import app
from labm8 import humanize
from labm8 import prof

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
# Some of these flags define parameters which must be equal when restoring from
# file, such as the hidden layer sizes. Other parameters may change between
# runs of the same model, such as the input data batch size. To accomodate for
# this, a ClassifierBase.GetModelFlagNames() method returns the list of flags
# which must be consistent between runs of the same model.
#
# For the sake of readability, these important model flags are saved into a
# global set classifier_base.MODEL_FLAGS here, so that the declaration of model
# flags is local to the declaration of the flag.
app.DEFINE_list('layer_timesteps', ['2', '2', '2'],
                'A list of layers, and the number of steps for each layer.')
# Note that although layer_timesteps is a model flag, there is special handling
# that permits the number of steps in each layer to differ when loading models.
# This is to permit testing a model with a larger number of timesteps than it
# was trained for.
classifier_base.MODEL_FLAGS.add("layer_timesteps")

app.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")
classifier_base.MODEL_FLAGS.add("learning_rate")

# TODO(cec): Poorly understood:
app.DEFINE_float("clamp_gradient_norm", 1.0, "Clip gradients to L-2 norm.")
classifier_base.MODEL_FLAGS.add("clamp_gradient_norm")

app.DEFINE_integer("hidden_size", 202, "The size of hidden layer(s).")
classifier_base.MODEL_FLAGS.add("hidden_size")

app.DEFINE_string(
    "inst2vec_embeddings", "random",
    "The type of per-node inst2vec embeddings to use. One of: "
    "{constant,constant_zero,constant_random,finetune,random}.")
classifier_base.MODEL_FLAGS.add("inst2vec_embeddings")

app.DEFINE_boolean(
    "tensorboard_logging", True,
    "If true, write tensorboard logs to '<working_dir>/tensorboard'.")

app.DEFINE_string(
    "unroll_strategy", "none", "The unroll strategy to use. One of: "
    "{none, constant, edge_count, data_flow_max_steps, label_convergence} "
    "constant: Unroll by a constant number of steps. The total number of steps is "
    "(unroll_factor * message_passing_step_count).")

app.DEFINE_float(
    "unroll_factor", 0,
    "Determine the number of dynamic model unrolls to perform. If "
    "--unroll_strategy=constant, this number of unrolls - each of size sum(layer_timesteps) are performed. "
    "So one unroll adds sum(layer_timesteps) many steps to the network. If "
    "--unroll_strategy=edge_counts, max_edge_count * --unroll_factor timesteps "
    "are performed. (rounded up to the next multiple of sum(layer_timesteps))")

# We assume that position_embeddings exist in every dataset.
# the flag now only controls whether they are used or not.
# This could be nice for ablating our model and also debugging with and without.
app.DEFINE_string(
    "position_embeddings", "fancy",
    "Whether to use position embeddings as signals for edge order."
    "Options: initial, every, fancy, off"
    "initial takes A (h + pos) at first timestep, every does the same at every timestep"
    "fancy learns another weight matrix B, s.th. propagation is A h + B pos"
    "We expect them to be part of the ds anyway, but you can toggle off their effect."
)
classifier_base.MODEL_FLAGS.add("position_embeddings")

# TODO(cec): Poorly understood.
app.DEFINE_boolean("freeze_graph_model", False, "???")
#
##### End of flag declarations.

# Type alias for the feed_dict argument of tf.compat.v1.Session.run().
FeedDict = typing.Dict[str, typing.Any]


class GgnnBaseModel(classifier_base.ClassifierBase):
  """Abstract base class for implementing gated graph neural networks.

  Subclasses must provide implementations of
  MakeLossAndAccuracyAndPredictionOps().
  """

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    raise NotImplementedError("abstract class")

  def __init__(self, *args):
    """Constructor."""
    super(GgnnBaseModel, self).__init__(*args)

    # Instantiate Tensorflow model.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.graph = tf.Graph()
    self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

    with self.graph.as_default():
      tf.set_random_seed(FLAGS.random_seed)
      with prof.Profile('Made model'):
        self.placeholders = utils.MakePlaceholders(self.stats)

        self.ops = {}
        with tf.compat.v1.variable_scope("graph_model"):
          (self.ops["loss"], self.ops["accuracies"], self.ops["accuracy"],
           self.ops["predictions"]) = (
               self.MakeLossAndAccuracyAndPredictionOps())

        if FLAGS.unroll_strategy != "none":
          with tf.compat.v1.variable_scope("modular_graph_model"):
            (self.ops["modular_loss"], self.ops["modular_accuracies"],
             self.ops["modular_accuracy"],
             self.ops["modular_predictions"]) = (self.MakeModularGraphOps())

            with tf.compat.v1.variable_scope("TransformAndUpdate"):
              self.ops[
                  "raw_node_output_features"] = self.MakeTransformAndUpdateOps(
                      self.placeholders['raw_node_input_features'])

          # Modular Tensorboard summaries
          self.ops["modular_summary_loss"] = tf.summary.scalar(
              "modular_loss", self.ops["modular_loss"], family='loss')
          self.ops["modular_summary_accuracy"] = tf.summary.scalar(
              "modular_accuracy",
              self.ops["modular_accuracy"],
              family='accuracy')

        # Tensorboard summaries.
        self.ops["summary_loss"] = tf.summary.scalar(
            "loss", self.ops["loss"], family='loss')
        self.ops["summary_accuracy"] = tf.summary.scalar(
            "accuracy", self.ops["accuracy"], family='accuracy')

        if not FLAGS.test_only:
          with prof.Profile('Make training step'):
            with tf.compat.v1.variable_scope("train_step"):
              self.ops["train_step"] = self._MakeTrainStep()

    # Tensorboard logging.
    if FLAGS.tensorboard_logging:
      tensorboard_dir = self.working_dir / 'tensorboard' / self.run_id
      app.Log(1, f"Writing tensorboard logs to: `{tensorboard_dir}`")
      tensorboard_dir.mkdir(parents=True, exist_ok=True)
      self.summary_writers = {
          "train":
          tf.compat.v1.summary.FileWriter(tensorboard_dir / "train",
                                          self.sess.graph),
          "val":
          tf.compat.v1.summary.FileWriter(tensorboard_dir / "val",
                                          self.sess.graph),
          "test":
          tf.compat.v1.summary.FileWriter(tensorboard_dir / "test",
                                          self.sess.graph),
      }

  def _GetPositionEmbeddingsAsTensorflowVariable(self) -> tf.Tensor:
    """It's probably a good memory/compute trade-off to have this additional embedding table instead of computing it on the fly."""
    embeddings = base_utils.pos_emb(
        positions=range(self.stats.max_edge_positions),
        demb=FLAGS.hidden_size - 2)  # hard coded
    pos_emb = tf.Variable(
        initial_value=embeddings, trainable=False, dtype=tf.float32)
    return pos_emb

  def _GetEmbeddingsAsTensorflowVariables(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """Read the embeddings table and return as a tensorflow variable."""
    # TODO(github.com/ChrisCummins/ml4pl/issues/12): In the future we may want
    # to be more flexible in supporting multiple types of embeddings tables, but
    # for now I have hardcoded this to always return a tuple
    # <inst2vec_embeddings, selector_embeddings>, where inst2vec_embeddings
    # is the augmented table of pre-trained statement embeddings (the
    # augmentation adds !MAGIC, !IMMEDIATE, and !IDENTIFIER vocabulary
    # elements). selector_embeddings is a 2x2 1-hot embedding table:
    # [[1, 0], [0, 1]. The selector_embeddings table is always constant, the
    # inst2vec_embeddings table can be made trainable or re-initialized with
    # random values using the --inst2vec_embeddings flag.
    embeddings = list(self.graph_db.embeddings_tables)
    if FLAGS.inst2vec_embeddings == 'constant':
      app.Log(1,
              "Using pre-trained inst2vec embeddings without further training")
      trainable = False
    elif FLAGS.inst2vec_embeddings == 'constant_zero':
      embeddings[0] = np.zeros(embeddings[0].shape)
      trainable = False
    elif FLAGS.inst2vec_embeddings == 'constant_random':
      embeddings[0] = np.random.rand(*embeddings[0].shape)
      trainable = False
    elif FLAGS.inst2vec_embeddings == 'finetune':
      app.Log(1, "Fine-tuning inst2vec embeddings")
      trainable = True
    elif FLAGS.inst2vec_embeddings == 'random':
      app.Log(1, "Initializing with random embeddings")
      embeddings[0] = np.random.rand(*embeddings[0].shape)
      trainable = True
    else:
      raise app.UsageError(
          f"--inst2vec_embeddings=`{FLAGS.inst2vec_embeddings}` "
          "unrecognized. Must be one of "
          "{constant,constant_zero,finetune,random}")
    inst2vec_embeddings = tf.Variable(
        initial_value=embeddings[0], trainable=trainable, dtype=tf.float32)
    selector_embeddings = tf.Variable(
        initial_value=embeddings[1] * 50, trainable=False, dtype=tf.float32)
    return inst2vec_embeddings, selector_embeddings

  @property
  def message_passing_step_count(self) -> int:
    return self.layer_timesteps.sum()

  @property
  def layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.layer_timesteps])

  def ModularlyRunWithFetchDict(
      self,
      log: log_database.Database,
      fetch_dict: typing.Dict[str, tf.Tensor],
      feed_dict: typing.Dict[tf.Tensor, typing.Any],
      unroll_factor: int,
      print_context: typing.Any = None,
  ) -> typing.Dict[str, tf.Tensor]:

    # cec: Temporarily disabling all of this debugging printout:
    # print("#" * 30 + "fetch dict keys" + "#" * 30)
    # for k in fetch_dict.keys():
    #   print(k, "   --   ", fetch_dict[k])
    #
    # print("#" * 30 + "feed dict keys" + "#" * 30)
    # for k in feed_dict.keys():
    #   print(k)

    # leaving debug comments for the next problem with another unrolling mode...
    # print("#"*30 + "fetch dict complete" + "#"*30)
    # print(fetch_dict)
    # print('\n')
    # print("#"*30 + "feed dict complete" + "#"*30)
    # print(feed_dict)
    # print('\n')
    # assert False

    # first get input_nodes_states manually
    # depends on placeholder['node_x']
    input_node_states = utils.RunWithFetchDict(
        self.sess, {'in': self.encoded_node_x}, feed_dict)['in']
    # now we should be independent of node_x, o/w we cannot guarantee that
    # no fetch_dict op won't use self.encoded_node_x which it is not allowed to under
    # modular unrolling!
    feed_dict.pop(self.placeholders['node_x'])

    # now get first raw_node_output_states manually
    # depends on placeholder['raw_node_input_features']
    loop_fetch = {
        "raw_node_output_features": self.ops["raw_node_output_features"]
    }

    loop_feed = {
        self.placeholders['raw_node_input_features']: input_node_states
    }
    feed_dict.update(loop_feed)

    _node_states = utils.RunWithFetchDict(self.sess, loop_fetch,
                                          feed_dict)["raw_node_output_features"]

    # add the loop_feed to the feed_dict
    feed_dict.update(
        {self.placeholders["raw_node_output_features"]: _node_states})

    # now get first predictions manually (for convergence tests)
    pred_fetch = {"modular_predictions": self.ops["modular_predictions"]}
    _new_predictions = utils.RunWithFetchDict(self.sess, pred_fetch,
                                              feed_dict)["modular_predictions"]

    # now always fetch modular_predictions w/ old _node_states and
    # simulateously generate new _node_states from old _node_states
    loop_fetch.update(pred_fetch)

    if unroll_factor < 1:
      stop_once_converged = True
      # We still provide *some* value to stop iterating.
      unroll_factor = 100
    else:
      stop_once_converged = True

    log.model_converged = False
    for iteration_count in range(1, unroll_factor):
      # First compute the current model labels.
      previous_labels = np.argmax(_new_predictions, axis=1)

      # we use the same value to simultaneously get
      # the next state update and the predictions from
      # that same state update.
      feed_dict.update({
          self.placeholders["raw_node_input_features"]:
          _node_states,
          self.placeholders["raw_node_output_features"]:
          _node_states,
      })
      _results = utils.RunWithFetchDict(self.sess, loop_fetch, feed_dict)
      _node_states = _results["raw_node_output_features"]

      # Compute the current model labels.
      current_labels = np.argmax(_results['modular_predictions'], axis=1)

      # Compare the labels before and after running to see if the model has
      # converged.
      converged = (previous_labels == current_labels).all()
      log.model_converged |= converged

      app.Log(
          4,
          'Completed dynamic unrolling loop step %s. Converged? %s',
          iteration_count,
          converged,
          print_context=print_context)
      if stop_once_converged and converged:
        break

    log.model_converged = converged
    log.iteration_count = iteration_count

    if log.model_converged:
      app.Log(2,
              "Model outputs converged after %s iterations",
              iteration_count,
              print_context=print_context)
    else:
      app.Log(2,
              "Model outputs failed to converge after %s iterations",
              iteration_count,
              print_context=print_context)

    # finally compute everything from the original fetch_dict
    # using our unrolled states.
    # we have to pop self.placeholders['node_x']
    # just to make sure that no output depends on self.encoded_node_x
    # implicitly, as whatever that is should use raw_node_input now!

    # we pop the globally speaking "intermediate node features"
    feed_dict.pop(self.placeholders['raw_node_input_features'])

    feed_dict.update({
        # and add the actual input features from above
        self.placeholders['raw_node_input_features']:
        input_node_states,
        self.placeholders["raw_node_output_features"]:
        _node_states,
    })
    fetch_dict = utils.RunWithFetchDict(self.sess, fetch_dict, feed_dict)
    return fetch_dict

  def GetUnrollFactor(self, unroll_strategy: str, unroll_factor: float,
                      log: log_database.BatchLogMeta) -> int:
    """Determine the unroll factor from the --unroll_strategy and --unroll_factor
    flags, and the batch log.
    """
    # Determine the unrolling strategy.
    if unroll_strategy == "none" or log.type == "train":
      # Perform no unrolling. The inputs are processed for a single run of
      # message_passing_step_count. This is required during training to
      # propagate gradients.
      return 1
    elif unroll_strategy == "constant":
      # Unroll by a constant number of steps. The total number of steps is
      # (unroll_factor * message_passing_step_count).
      return int(unroll_factor)
    elif unroll_strategy == "data_flow_max_steps":
      max_data_flow_steps = log._transient_data['data_flow_max_steps_required']
      unroll_factor = math.ceil(
          max_data_flow_steps / self.message_passing_step_count)
      app.Log(2, 'Determined unroll factor %d from max data flow steps %d',
              unroll_factor, max_data_flow_steps)
      return unroll_factor
    elif unroll_strategy == "edge_count":
      max_edge_count = log._transient_data['max_edge_count']
      unroll_factor = math.ceil(
          (max_edge_count * unroll_factor) / self.message_passing_step_count)
      app.Log(2, 'Determined unroll factor %d from max edge count %d',
              unroll_factor, self.message_passing_step_count)
      return unroll_factor
    elif unroll_strategy == "label_convergence":
      return 0
    else:
      raise app.UsageError(f"Unknown unroll strategy '{unroll_strategy}'")

  def RunMinibatch(self,
                   log: log_database.BatchLogMeta,
                   feed_dict: typing.Any,
                   print_context: typing.Any = None
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    unroll_factor = self.GetUnrollFactor(FLAGS.unroll_strategy,
                                         FLAGS.unroll_factor, log)

    if unroll_factor == 1:
      fetch_dict = {
          "loss": self.ops["loss"],
          "accuracies": self.ops["accuracies"],
          "accuracy": self.ops["accuracy"],
          "predictions": self.ops["predictions"],
          "summary_loss": self.ops["summary_loss"],
          "summary_accuracy": self.ops["summary_accuracy"],
      }
      if log.type == "train":
        fetch_dict["train_step"] = self.ops["train_step"]
      fetch_dict = utils.RunWithFetchDict(self.sess, fetch_dict, feed_dict)
    else:
      fetch_dict = {
          "loss": self.ops["modular_loss"],
          "accuracies": self.ops["modular_accuracies"],
          "accuracy": self.ops["modular_accuracy"],
          "predictions": self.ops["modular_predictions"],
          # "summary_loss": self.ops["modular_summary_loss"],
          # "summary_accuracy": self.ops["modular_summary_accuracy"],
      }
      if log.type == "train":
        fetch_dict["train_step"] = self.ops["train_step"]
      fetch_dict = self.ModularlyRunWithFetchDict(log, fetch_dict, feed_dict,
                                                  unroll_factor)

    log.loss = float(fetch_dict['loss'])

    if 'node_y' in self.placeholders:
      targets = feed_dict[self.placeholders['node_y']]
    elif 'graph_y' in self.placeholders:
      targets = feed_dict[self.placeholders['graph_y']]
    else:
      raise TypeError("Neither node_y or graph_y in placeholders dict!")

    return self.MinibatchResults(
        y_true_1hot=targets, y_pred_1hot=fetch_dict['predictions'])

  def InitializeModel(self) -> None:
    super(GgnnBaseModel, self).InitializeModel()
    with self.graph.as_default():
      self.sess.run(
          tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer()))

  def ModelDataToSave(self) -> typing.Any:
    with self.graph.as_default():
      weights_to_save = {}
      for variable in self.sess.graph.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES):
        assert variable.name not in weights_to_save
        weights_to_save[variable.name] = self.sess.run(variable)
    return weights_to_save

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    with self.graph.as_default():
      variables_to_initialize = []
      with tf.name_scope("restore"):
        restore_ops = []
        used_vars = set()
        for variable in self.sess.graph.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES):
          used_vars.add(variable.name)
          if variable.name in data_to_load:
            restore_ops.append(variable.assign(data_to_load[variable.name]))
          else:
            app.Log(
                1, "Freshly initializing %s since no saved value "
                "was found.", variable.name)
            variables_to_initialize.append(variable)
        for var_name in data_to_load:
          if var_name not in used_vars:
            app.Log(1, "Saved weights for %s not used by model.", var_name)
        restore_ops.append(
            tf.compat.v1.variables_initializer(variables_to_initialize))
        self.sess.run(restore_ops)

  def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
    # Special handling for layer_timesteps: We permit a different number of
    # steps per layer, but require that the number of layers be the same.
    num_layers = len(flags['layer_timesteps'])
    saved_num_layers = len(saved_flags['layer_timesteps'])
    if num_layers != saved_num_layers:
      raise EnvironmentError(
          "Saved model has "
          f"{humanize.Plural(saved_num_layers, 'layer')} but flags has "
          f"incompatible {humanize.Plural(num_layers, 'layer')}")

    # Use regular comparison method for the other flags.
    del flags['layer_timesteps']
    del saved_flags['layer_timesteps']
    super(GgnnBaseModel, self).CheckThatModelFlagsAreEquivalent(
        flags, saved_flags)

  def _MakeTrainStep(self) -> tf.Tensor:
    """Helper function."""
    trainable_vars = self.sess.graph.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)
    if FLAGS.freeze_graph_model:
      graph_vars = set(
          self.sess.graph.get_collection(
              tf.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"))
      filtered_vars = []
      for var in trainable_vars:
        if var not in graph_vars:
          filtered_vars.append(var)
        else:
          app.Log(1, "Freezing weights of variable `%s`.", var.name)
      trainable_vars = filtered_vars
    optimizer = tf.compat.v1.train.AdamOptimizer(
        FLAGS.learning_rate * self.placeholders['learning_rate_multiple'])
    grads_and_vars = optimizer.compute_gradients(
        self.ops["loss"], var_list=trainable_vars)
    clipped_grads = []
    for grad, var in grads_and_vars:
      if grad is not None:
        clipped_grads.append((tf.clip_by_norm(grad, FLAGS.clamp_gradient_norm),
                              var))
      else:
        clipped_grads.append((grad, var))
    train_step = optimizer.apply_gradients(clipped_grads)

    # Also run batch_norm update ops, if any.
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = tf.group([train_step, update_ops])

    # Initialize newly-introduced variables:
    self.sess.run(tf.compat.v1.local_variables_initializer())

    return train_step
