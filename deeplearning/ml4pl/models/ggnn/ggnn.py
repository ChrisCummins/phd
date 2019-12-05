"""Train and evaluate a model for node classification."""
import warnings
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Tuple

import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.models import base_utils
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_base as ggnn
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof
from labm8.py import progress

FLAGS = app.FLAGS


app.DEFINE_string(
  "graph_rnn_cell",
  "GRU",
  "The RNN cell type. One of {GRU,CudnnCompatibleGRUCell,RNN}",
)
app.DEFINE_string(
  "graph_rnn_activation", "tanh", "The RNN activation type. One of {tanh,ReLU}."
)
app.DEFINE_boolean(
  "ggnn_use_edge_msg_avg_aggregation",
  True,
  "If true, normalize incoming messages by the number of incoming messages.",
)
app.DEFINE_float(
  "graph_state_dropout_keep_prob",
  1.0,
  "Graph state dropout keep probability (rate = 1 - keep_prob)",
)
app.DEFINE_float(
  "edge_weight_dropout_keep_prob",
  1.0,
  "Edge weight dropout keep probability (rate = 1 - keep_prob)",
)
app.DEFINE_float(
  "output_layer_dropout_keep_prob",
  1.0,
  "Dropout keep probability on the output layer. In range 0 < x <= 1.",
)
app.DEFINE_float(
  "ggnn_intermediate_loss_discount_factor",
  0.2,
  "The actual loss is computed as loss + factor * intermediate_loss",
)
app.DEFINE_integer(
  "auxiliary_inputs_dense_layer_size",
  32,
  "Size for MLP that combines graph_x and GGNN output features",
)
app.DEFINE_boolean(
  "use_dsc_loss",
  False,
  "Whether to use the DSC loss instead of Cross Entropy. DSC loss help with "
  "class imbalances. See <https://arxiv.org/pdf/1911.02855.pdf>",
)
app.DEFINE_list(
  "ggnn_layer_timesteps",
  ["2", "2", "2"],
  "A list of layers, and the number of steps for each layer.",
)
app.DEFINE_float("clamp_gradient_norm", 1.0, "Clip gradients to L-2 norm.")
# TODO(github.com/ChrisCummins/ProGraML/issues/16): Refactor to remove this.
app.DEFINE_integer("hidden_size", 202, "The size of hidden layer(s).")
# TODO(github.com/ChrisCummins/ProGraML/issues/24): Replace
# --ggnn_unroll_strategy with an enum, and move unroll logic into a
# dynamic_unroll.py module.
app.DEFINE_string(
  "ggnn_unroll_strategy",
  "none",
  "The unroll strategy to use. One of: "
  "{none, constant, edge_count, data_flow_max_steps, label_convergence} "
  "constant: Unroll by a constant number of steps. The total number of steps is "
  "(ggnn_unroll_factor * message_passing_step_count).",
)
app.DEFINE_float(
  "ggnn_unroll_factor",
  0,
  "Determine the number of dynamic model unrolls to perform. If "
  "--ggnn_unroll_strategy=constant, this number of unrolls - each of size "
  "sum(ggnn_layer_timesteps) are performed. So one unroll adds "
  "sum(ggnn_layer_timesteps) many steps to the network. If "
  "--ggnn_unroll_strategy=edge_counts, max_edge_count * --ggnn_unroll_factor "
  "timesteps are performed. (rounded up to the next multiple of "
  "sum(ggnn_layer_timesteps))",
)
app.DEFINE_float(
  "ggnn_unroll_convergence_threshold",
  0.99,
  "The ratio of labels which must have converged when "
  "--ggnn_unroll_strategy=label_convergence for dynamic unrolling to cease.",
)
# TODO(github.com/ChrisCummins/ProGraML/issues/24): Rename --inst2vec_embeddings
# to --statement_embeddings.
app.DEFINE_string(
  "inst2vec_embeddings",
  "random",
  "The type of per-node inst2vec embeddings to use. One of: "
  "{constant,constant_zero,constant_random,finetune,random}.",
)
# We assume that position_embeddings exist in every dataset.
# the flag now only controls whether they are used or not.
# This could be nice for ablating our model and also debugging with and without.
app.DEFINE_string(
  "position_embeddings",
  "fancy",
  "Whether to use position embeddings as signals for edge order."
  "Options: initial, every, fancy, off"
  "initial takes A (h + pos) at first timestep, every does the same at every timestep"
  "fancy learns another weight matrix B, s.th. propagation is A h + B pos"
  "We expect them to be part of the ds anyway, but you can toggle off their effect.",
)
app.DEFINE_boolean(
  "tensorboard_logging",
  False,
  "If true, write tensorboard logs to '<working_dir>/tensorboard'.",
)
# TODO(github.com/ChrisCummins/ProGraML/issues/18): Poorly understood.
app.DEFINE_boolean("freeze_graph_model", False, "???")


# Type alias for the feed_dict argument of tf.compat.v1.Session.run().
FeedDict = Dict[str, Any]


class GgnnWeights(NamedTuple):
  """The weights of a GGNN model."""

  edge_weights: List[tf.Tensor]
  rnn_cells: List[tf.Tensor]

  def Create(self):
    """Create an empty set of weights."""
    return GgnnWeights([], [])


class Ggnn(ggnn.Ggnn):
  """Gated graph neural network for node-level or graph-level classification."""

  def __init__(self, *args):
    """Constructor."""
    super(Ggnn, self).__init__(*args)

    # Instantiate the Tensorflow graph and session.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.graph = tf.Graph()
    self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

    with self.graph.as_default():
      with prof.Profile("Made model"):
        self.placeholders = utils.MakePlaceholders(self.stats)

        self.ops = {}
        with tf.compat.v1.variable_scope("graph_model"):
          (
            self.ops["loss"],
            self.ops["accuracies"],
            self.ops["accuracy"],
            self.ops["predictions"],
          ) = self.MakeLossAndAccuracyAndPredictionOps()

        if FLAGS.ggnn_unroll_strategy != "none":
          with tf.compat.v1.variable_scope("modular_graph_model"):
            (
              self.ops["modular_loss"],
              self.ops["modular_accuracies"],
              self.ops["modular_accuracy"],
              self.ops["modular_predictions"],
            ) = self.MakeModularGraphOps()

            with tf.compat.v1.variable_scope("TransformAndUpdate"):
              self.ops[
                "raw_node_output_features"
              ] = self.MakeTransformAndUpdateOps(
                self.placeholders["raw_node_input_features"]
              )

          # TODO(github.com/ChrisCummins/ProGraML/issues/21): Re-implement.
          # Modular Tensorboard summaries
          # self.ops["modular_summary_loss"] = tf.summary.scalar(
          #   "modular_loss", self.ops["modular_loss"], family="loss"
          # )
          # self.ops["modular_summary_accuracy"] = tf.summary.scalar(
          #   "modular_accuracy", self.ops["modular_accuracy"], family="accuracy"
          # )

        # TODO(github.com/ChrisCummins/ProGraML/issues/21): Re-implement.
        # Tensorboard summaries.
        # self.ops["summary_loss"] = tf.summary.scalar(
        #   "loss", self.ops["loss"], family="loss"
        # )
        # self.ops["summary_accuracy"] = tf.summary.scalar(
        #   "accuracy", self.ops["accuracy"], family="accuracy"
        # )

        if not FLAGS.test_only:
          with prof.Profile("Make training step"):
            with tf.compat.v1.variable_scope("train_step"):
              self.ops["train_step"] = self.MakeTrainStep()

    # Tensorboard logging.
    # TODO(github.com/ChrisCummins/ProGraML/issues/21): Re-implement.
    # if FLAGS.tensorboard_logging:
    #   tensorboard_dir = self.working_dir / "tensorboard" / self.run_id
    #   app.Log(1, f"Writing tensorboard logs to: `{tensorboard_dir}`")
    #   tensorboard_dir.mkdir(parents=True, exist_ok=True)
    #   self.summary_writers = {
    #     "train": tf.compat.v1.summary.FileWriter(
    #       tensorboard_dir / "train", self.sess.graph
    #     ),
    #     "val": tf.compat.v1.summary.FileWriter(
    #       tensorboard_dir / "val", self.sess.graph
    #     ),
    #     "test": tf.compat.v1.summary.FileWriter(
    #       tensorboard_dir / "test", self.sess.graph
    #     ),
    #   }

  def GetPositionEmbeddingsAsTensorflowVariable(self) -> tf.Tensor:
    """It's probably a good memory/compute trade-off to have this additional
    embedding table instead of computing it on the fly.
    """
    # hard coded
    embeddings = base_utils.pos_emb(
      positions=range(self.stats.max_edge_positions), demb=FLAGS.hidden_size - 2
    )
    pos_emb = tf.Variable(
      initial_value=embeddings, trainable=False, dtype=tf.float32
    )
    return pos_emb

  def _GetEmbeddingsAsTensorflowVariables(self,) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read the embeddings table and return as a tensorflow variable."""
    # TODO(github.com/ChrisCummins/ProGraML/issues/12): In the future we may want
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
    if FLAGS.inst2vec_embeddings == "constant":
      app.Log(
        1, "Using pre-trained inst2vec embeddings without further training"
      )
      trainable = False
    elif FLAGS.inst2vec_embeddings == "constant_zero":
      embeddings[0] = np.zeros(embeddings[0].shape)
      trainable = False
    elif FLAGS.inst2vec_embeddings == "constant_random":
      embeddings[0] = np.random.rand(*embeddings[0].shape)
      trainable = False
    elif FLAGS.inst2vec_embeddings == "finetune":
      app.Log(1, "Fine-tuning inst2vec embeddings")
      trainable = True
    elif FLAGS.inst2vec_embeddings == "random":
      app.Log(1, "Initializing with random embeddings")
      embeddings[0] = np.random.rand(*embeddings[0].shape)
      trainable = True
    else:
      raise app.UsageError(
        f"--inst2vec_embeddings=`{FLAGS.inst2vec_embeddings}` "
        "unrecognized. Must be one of "
        "{constant,constant_zero,finetune,random}"
      )
    inst2vec_embeddings = tf.Variable(
      initial_value=embeddings[0], trainable=trainable, dtype=tf.float32
    )
    selector_embeddings = tf.Variable(
      initial_value=embeddings[1] * 50, trainable=False, dtype=tf.float32
    )
    return inst2vec_embeddings, selector_embeddings

  @property
  def message_passing_step_count(self) -> int:
    return self.ggnn_layer_timesteps.sum()

  @property
  def ggnn_layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.ggnn_layer_timesteps])

  def ModularlyRunWithFetchDict(
    self,
    log: log_database.Database,
    fetch_dict: Dict[str, tf.Tensor],
    feed_dict: Dict[tf.Tensor, Any],
    ggnn_unroll_factor: int,
    print_context: Any = None,
  ) -> Dict[str, tf.Tensor]:
    app.Log(1, "-> MODULARLY RUN", print_context=print_context)

    # First we compute the input_nodes_states placeholder['node_x'] to produce
    # the initial encoded inputs.
    initial_node_states = utils.RunWithFetchDict(
      self.sess, {"in": self.encoded_node_x}, feed_dict
    )["in"]

    # Now we are independent of node_x, otherwise we cannot guarantee that a
    # fetch_dict op won't use self.encoded_node_x which it is not allowed to
    # under modular unrolling.
    feed_dict.pop(self.placeholders["node_x"])

    # Now we compute the first raw_node_output_states manually using
    # placeholder['raw_node_input_features'].
    loop_feed = {
      self.placeholders["raw_node_input_features"]: initial_node_states
    }
    loop_fetch = {
      "raw_node_output_features": self.ops["raw_node_output_features"]
    }

    # TODO(github.com/ChrisCummins/ProGraML/issues/18): Investigate this.
    feed_dict.update(loop_feed)

    node_states = utils.RunWithFetchDict(self.sess, loop_fetch, feed_dict)[
      "raw_node_output_features"
    ]

    # Add the loop_feed to the feed_dict.
    feed_dict[self.placeholders["raw_node_output_features"]] = node_states

    # now get first predictions manually (for convergence tests)
    pred_fetch = {"modular_predictions": self.ops["modular_predictions"]}
    current_predictions = utils.RunWithFetchDict(
      self.sess, pred_fetch, feed_dict
    )["modular_predictions"]
    current_labels = np.argmax(current_predictions, axis=1)

    # now always fetch modular_predictions w/ old node_states and
    # simulateously generate new node_states from old node_states
    loop_fetch.update(pred_fetch)

    if ggnn_unroll_factor < 1:
      stop_once_converged = True
      # We still provide *some* value to stop iterating.
      ggnn_unroll_factor = 100
    else:
      stop_once_converged = True

    log.model_converged = False
    iteration_count = 1
    for iteration_count in range(1, ggnn_unroll_factor):
      app.Log(1, "--> DYNAMIC UNROLL LOOP", print_context=print_context)
      # First compute the current model labels.
      previous_labels = current_labels

      # we use the same value to simultaneously get
      # the next state update and the predictions from
      # that same state update.
      feed_dict.update(
        {
          self.placeholders["raw_node_input_features"]: node_states,
          self.placeholders["raw_node_output_features"]: node_states,
        }
      )
      _results = utils.RunWithFetchDict(self.sess, loop_fetch, feed_dict)
      node_states = _results["raw_node_output_features"]

      # Compute the current model labels.
      current_labels = np.argmax(_results["modular_predictions"], axis=1)

      # Compare the labels before and after running to see if the model has
      # converged.
      converged_labels = (previous_labels == current_labels).mean()
      log.model_converged |= (
        converged_labels >= FLAGS.ggnn_unroll_convergence_threshold
      )

      app.Log(
        4,
        "Completed dynamic unrolling loop step %s. Converged labels: %s",
        iteration_count,
        converged_labels,
        print_context=print_context,
      )
      if stop_once_converged and log.model_converged:
        break

    log.iteration_count = iteration_count

    if log.model_converged:
      app.Log(
        2,
        "Model outputs converged after %s iterations",
        iteration_count,
        print_context=print_context,
      )
    else:
      app.Log(
        2,
        "Model outputs failed to converge after %s iterations",
        iteration_count,
        print_context=print_context,
      )

    # finally compute everything from the original fetch_dict
    # using our unrolled states.
    # we have to pop self.placeholders['node_x']
    # just to make sure that no output depends on self.encoded_node_x
    # implicitly, as whatever that is should use raw_node_input now!

    # we pop the globally speaking "intermediate node features"
    feed_dict.pop(self.placeholders["raw_node_input_features"])

    feed_dict.update(
      {
        # Add the actual input features that we computed above.
        # TODO(github.com/ChrisCummins/ProGraML/issues/18): Why??
        self.placeholders["raw_node_input_features"]: initial_node_states,
        self.placeholders["raw_node_output_features"]: node_states,
      }
    )
    fetch_dict = utils.RunWithFetchDict(self.sess, fetch_dict, feed_dict)
    return fetch_dict

  def GetUnrollFactor(
    self,
    ggnn_unroll_strategy: str,
    ggnn_unroll_factor: float,
    log: log_database.BatchLogMeta,
  ) -> int:
    """Determine the unroll factor from the --ggnn_unroll_strategy and --ggnn_unroll_factor
    flags, and the batch log.
    """
    # Determine the unrolling strategy.
    if ggnn_unroll_strategy == "none" or log.type == "train":
      # Perform no unrolling. The inputs are processed for a single run of
      # message_passing_step_count. This is required during training to
      # propagate gradients.
      return 1
    elif ggnn_unroll_strategy == "constant":
      # Unroll by a constant number of steps. The total number of steps is
      # (ggnn_unroll_factor * message_passing_step_count).
      return int(ggnn_unroll_factor)
    elif ggnn_unroll_strategy == "data_flow_max_steps":
      max_data_flow_steps = log._transient_data["data_flow_max_steps_required"]
      ggnn_unroll_factor = math.ceil(
        max_data_flow_steps / self.message_passing_step_count
      )
      app.Log(
        2,
        "Determined unroll factor %d from max data flow steps %d",
        ggnn_unroll_factor,
        max_data_flow_steps,
      )
      return ggnn_unroll_factor
    elif ggnn_unroll_strategy == "edge_count":
      max_edge_count = log._transient_data["max_edge_count"]
      ggnn_unroll_factor = math.ceil(
        (max_edge_count * ggnn_unroll_factor) / self.message_passing_step_count
      )
      app.Log(
        2,
        "Determined unroll factor %d from max edge count %d",
        ggnn_unroll_factor,
        self.message_passing_step_count,
      )
      return ggnn_unroll_factor
    elif ggnn_unroll_strategy == "label_convergence":
      return 0
    else:
      raise app.UsageError(f"Unknown unroll strategy '{ggnn_unroll_strategy}'")

  def RunMinibatch(
    self,
    log: log_database.BatchLogMeta,
    feed_dict: Any,
    print_context: Any = None,
  ) -> classifier_base.ClassifierBase.MinibatchResults:
    ggnn_unroll_factor = self.GetUnrollFactor(
      FLAGS.ggnn_unroll_strategy, FLAGS.ggnn_unroll_factor, log
    )

    if ggnn_unroll_factor == 1:
      fetch_dict = {
        "loss": self.ops["loss"],
        "accuracies": self.ops["accuracies"],
        "accuracy": self.ops["accuracy"],
        "predictions": self.ops["predictions"],
        # TODO(github.com/ChrisCummins/ProGraML/issues/21): Re-implement.
        # "summary_loss": self.ops["summary_loss"],
        # "summary_accuracy": self.ops["summary_accuracy"],
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
        # TODO(github.com/ChrisCummins/ProGraML/issues/21): Re-implement.
        # "summary_loss": self.ops["modular_summary_loss"],
        # "summary_accuracy": self.ops["modular_summary_accuracy"],
      }
      if log.type == "train":
        fetch_dict["train_step"] = self.ops["train_step"]
      fetch_dict = self.ModularlyRunWithFetchDict(
        log, fetch_dict, feed_dict, ggnn_unroll_factor
      )

    log.loss = float(fetch_dict["loss"])

    if "node_y" in self.placeholders:
      targets = feed_dict[self.placeholders["node_y"]]
    elif "graph_y" in self.placeholders:
      targets = feed_dict[self.placeholders["graph_y"]]
    else:
      raise TypeError("Neither node_y or graph_y in placeholders dict!")

    return self.MinibatchResults(
      y_true_1hot=targets, y_pred_1hot=fetch_dict["predictions"]
    )

  def InitializeModel(self) -> None:
    super(Ggnn, self).InitializeModel()
    with self.graph.as_default():
      self.sess.run(
        tf.group(
          tf.compat.v1.global_variables_initializer(),
          tf.compat.v1.local_variables_initializer(),
        )
      )

  def ModelDataToSave(self) -> Any:
    with self.graph.as_default():
      weights_to_save = {}
      for variable in self.sess.graph.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
      ):
        assert variable.name not in weights_to_save
        weights_to_save[variable.name] = self.sess.run(variable)
    return weights_to_save

  def LoadModelData(self, data_to_load: Any) -> None:
    with self.graph.as_default():
      variables_to_initialize = []
      with tf.name_scope("restore"):
        restore_ops = []
        used_vars = set()
        for variable in self.sess.graph.get_collection(
          tf.compat.v1.GraphKeys.GLOBAL_VARIABLES
        ):
          used_vars.add(variable.name)
          if variable.name in data_to_load:
            restore_ops.append(variable.assign(data_to_load[variable.name]))
          else:
            app.Log(
              1,
              "Freshly initializing %s since no saved value " "was found.",
              variable.name,
            )
            variables_to_initialize.append(variable)
        for var_name in data_to_load:
          if var_name not in used_vars:
            app.Log(1, "Saved weights for %s not used by model.", var_name)
        restore_ops.append(
          tf.compat.v1.variables_initializer(variables_to_initialize)
        )
        self.sess.run(restore_ops)

  def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
    # Special handling for ggnn_layer_timesteps: We permit a different number of
    # steps per layer, but require that the number of layers be the same.
    num_layers = len(flags["ggnn_layer_timesteps"])
    saved_num_layers = len(saved_flags["ggnn_layer_timesteps"])
    if num_layers != saved_num_layers:
      raise EnvironmentError(
        "Saved model has "
        f"{humanize.Plural(saved_num_layers, 'layer')} but flags has "
        f"incompatible {humanize.Plural(num_layers, 'layer')}"
      )

    # Use regular comparison method for the other flags.
    del flags["ggnn_layer_timesteps"]
    del saved_flags["ggnn_layer_timesteps"]
    super(Ggnn, self).CheckThatModelFlagsAreEquivalent(flags, saved_flags)

  def MakeTrainStep(self) -> tf.Tensor:
    """Helper function."""
    trainable_vars = self.sess.graph.get_collection(
      tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES
    )
    if FLAGS.freeze_graph_model:
      graph_vars = set(
        self.sess.graph.get_collection(
          tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="graph_model"
        )
      )
      filtered_vars = []
      for var in trainable_vars:
        if var not in graph_vars:
          filtered_vars.append(var)
        else:
          app.Log(1, "Freezing weights of variable `%s`.", var.name)
      trainable_vars = filtered_vars
    optimizer = tf.compat.v1.train.AdamOptimizer(
      self.placeholders["learning_rate"]
    )
    grads_and_vars = optimizer.compute_gradients(
      self.ops["loss"], var_list=trainable_vars
    )
    clipped_grads = []
    for grad, var in grads_and_vars:
      if grad is not None:
        clipped_grads.append(
          (tf.clip_by_norm(grad, FLAGS.clamp_gradient_norm), var)
        )
      else:
        clipped_grads.append((grad, var))
    train_step = optimizer.apply_gradients(clipped_grads)

    # Also run batch_norm update ops, if any.
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_step = tf.group([train_step, update_ops])

    # Initialize newly-introduced variables:
    self.sess.run(tf.compat.v1.local_variables_initializer())

    return train_step

  def MakeTransformAndUpdateOps(
    self, raw_node_input_features: tf.Tensor
  ) -> tf.Tensor:
    """
    Takes node input states (raw vectors) and returns the transformed and
    updated final raw node states.

    Depends on many attributes existing:
        self.placeholders: dict,
        self.ggnn_layer_timesteps,
        self.ggnn_weights,
        self.position_embeddings,
    """
    # Initial node states and then one entry per layer
    # (final state of that layer), shape: number of nodes
    # in batch v x D.
    node_states_per_layer = [raw_node_input_features]
    # Number of nodes in batch.
    num_nodes_in_batch = self.placeholders["node_count"]

    message_targets = []  # List of tensors of message targets of shape [E]
    message_edge_types = []  # List of tensors of edge type of shape [E]
    for edge_type, adjacency_list in enumerate(
      self.placeholders["adjacency_lists"]
    ):
      edge_targets = adjacency_list[:, 1]
      message_targets.append(edge_targets)
      message_edge_types.append(
        tf.ones_like(edge_targets, dtype=tf.int32) * edge_type
      )

    message_targets = tf.concat(
      message_targets, axis=0, name="message_targets"
    )  # Shape [M]
    # TODO(github.com/ChrisCummins/ProGraML/issues/24): This variable is unused.
    # Why?
    message_edge_types = tf.concat(
      message_edge_types, axis=0, name="message_edge_types"
    )  # Shape [M]

    for (layer_idx, num_timesteps) in enumerate(self.ggnn_layer_timesteps):
      with tf.compat.v1.variable_scope(f"gnn_layer_{layer_idx}"):
        # Used shape abbreviations:
        #   V ~ number of nodes
        #   D ~ state dimension
        #   E ~ number of edges of current type
        #   M ~ number of messages (sum of all E)

        # Record new states for this layer. Initialised to last state, but will
        # be updated below:
        node_states_per_layer.append(node_states_per_layer[-1])
        for step in range(num_timesteps):
          with tf.compat.v1.variable_scope(f"timestep_{step}"):
            # list of tensors of messages of shape [E, D]
            messages = []
            # list of tensors of edge source states of shape [E, D]
            message_source_states = []

            # Collect incoming messages per edge type
            for edge_type, (adjacency_list, edge_positions) in enumerate(
              zip(
                self.placeholders["adjacency_lists"],
                self.placeholders["edge_positions"],
              )
            ):
              edge_sources = adjacency_list[:, 0]

              edge_source_states = tf.nn.embedding_lookup(
                params=node_states_per_layer[-1], ids=edge_sources
              )  # Shape [E, D]

              if FLAGS.position_embeddings != "off":
                edge_pos_embedding = tf.nn.embedding_lookup(
                  self.position_embeddings, ids=edge_positions
                )  # shape [E, D]

              # one among: {initial, every, fancy, off}
              # maybe add position to edge_source_states here
              if FLAGS.position_embeddings == "every" or (
                step and FLAGS.position_embeddings == "initial"
              ):
                edge_source_states = tf.add(
                  edge_source_states,
                  edge_pos_embedding,
                  name="edge_source_states_with_position",
                )

              # Message propagation.
              # Term: A * h
              all_messages_for_edge_type = tf.matmul(
                edge_source_states,
                self.gnn_weights.edge_weights[layer_idx][edge_type],
              )  # Shape [E, D]

              # maybe add term B * pos
              if FLAGS.position_embeddings == "fancy":
                all_messages_for_edge_type += tf.matmul(
                  edge_pos_embedding,
                  # last edge_type corresponds to fancy_position_weights B
                  self.gnn_weights.edge_weights[layer_idx][-1],
                )  # Shape [E, D]

              messages.append(all_messages_for_edge_type)
              message_source_states.append(edge_source_states)

            messages = tf.concat(messages, axis=0)  # Shape [M, D]

            incoming_messages = tf.math.unsorted_segment_sum(
              data=messages,
              segment_ids=message_targets,
              num_segments=num_nodes_in_batch,
            )  # Shape [V, D]

            if FLAGS.ggnn_use_edge_msg_avg_aggregation:
              num_incoming_edges = tf.reduce_sum(
                self.placeholders["incoming_edge_counts"],
                keepdims=True,
                axis=-1,
              )  # Shape [V, 1]
              incoming_messages /= num_incoming_edges + utils.SMALL_NUMBER

            # Shape [V, D]
            incoming_information = tf.concat([incoming_messages], axis=-1)

            # pass updated vertex features into RNN cell, shape [V, D].
            node_states_per_layer[-1] = self.gnn_weights.rnn_cells[layer_idx](
              incoming_information, node_states_per_layer[-1]
            )[1]

    return node_states_per_layer[-1]

  def MakeLossAndAccuracyAndPredictionOps(
    self, ctx: progress.ProgressContext = progress.NullContext
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    ggnn_layer_timesteps = np.array(
      [int(x) for x in FLAGS.ggnn_layer_timesteps]
    )
    ctx.Log(
      1,
      "Using layer timesteps: %s for a total of %s message passing " "steps",
      ggnn_layer_timesteps,
      self.message_passing_step_count,
    )

    # Generate per-layer values for edge weights, biases and gated units:
    self.weights = {}  # Used by super-class to place generic things
    self.gnn_weights = GgnnWeights.Create()

    for layer_index in range(len(self.ggnn_layer_timesteps)):
      with tf.compat.v1.variable_scope(f"gnn_layer_{layer_index}"):
        # position propagation matrices are treated like another edge type
        if FLAGS.position_embeddings == "fancy":
          type_count_with_fancy = 1 + self.stats.edge_type_count
        else:
          type_count_with_fancy = self.stats.edge_type_count

        edge_weights = tf.reshape(
          tf.Variable(
            utils.glorot_init(
              [type_count_with_fancy * FLAGS.hidden_size, FLAGS.hidden_size]
            ),
            name=f"gnn_edge_weights_{layer_index}",
          ),
          [type_count_with_fancy, FLAGS.hidden_size, FLAGS.hidden_size],
        )

        # Add dropout as required.
        if FLAGS.edge_weight_dropout_keep_prob < 1.0:
          edge_weights = tf.nn.dropout(
            edge_weights,
            rate=1 - self.placeholders["edge_weight_dropout_keep_prob"],
          )
        self.gnn_weights.edge_weights.append(edge_weights)

        cell = utils.BuildRnnCell(
          FLAGS.graph_rnn_cell,
          FLAGS.graph_rnn_activation,
          FLAGS.hidden_size,
          name=f"cell_layer_{layer_index}",
        )
        # Apply dropout as required.
        if FLAGS.graph_state_dropout_keep_prob < 1:
          cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            cell,
            state_keep_prob=self.placeholders["graph_state_dropout_keep_prob"],
          )
        self.gnn_weights.rnn_cells.append(cell)
      # end of variable scope f"gnn_layer_{layer_index}"

    with tf.compat.v1.variable_scope("embeddings"):
      # maybe generate table with position embs up to pos 512.
      if FLAGS.position_embeddings != "off":
        self.position_embeddings = (
          self.GetPositionEmbeddingsAsTensorflowVariable()
        )

      # Lookup each node embedding table and concatenate the result.
      embeddings = self._GetEmbeddingsAsTensorflowVariables()
      for i in range(len(embeddings)):
        self.weights[f"node_embeddings_{i}"] = embeddings[i]

      self.encoded_node_x = tf.compat.v1.concat(
        [
          tf.nn.embedding_lookup(
            self.weights[f"node_embeddings_{i}"],
            ids=self.placeholders["node_x"][:, i],
          )
          for i in range(len(embeddings))
        ],
        axis=1,
        name="embeddings_concat",
      )

    ###########################################################################
    #  Dynamic unrolling.
    ###########################################################################

    with tf.compat.v1.variable_scope("TransformAndUpdate"):
      self.ops["final_node_x"] = self.MakeTransformAndUpdateOps(
        self.encoded_node_x,
      )

    ###########################################################################
    # Readout function.
    ###########################################################################

    if FLAGS.output_layer_dropout_keep_prob < 1:
      out_layer_dropout = self.placeholders["output_layer_dropout_keep_prob"]
    else:
      out_layer_dropout = None

    labels_dimensionality = (
      self.stats.node_labels_dimensionality
      or self.stats.graph_labels_dimensionality
    )
    predictions, regression_gate, regression_transform = utils.MakeOutputLayer(
      initial_node_state=self.encoded_node_x,
      final_node_state=self.ops["final_node_x"],
      hidden_size=FLAGS.hidden_size,
      labels_dimensionality=labels_dimensionality,
      dropout_keep_prob_placeholder=out_layer_dropout,
    )
    self.weights["regression_gate"] = regression_gate
    self.weights["regression_transform"] = regression_transform

    if self.stats.graph_features_dimensionality:
      # Sum node representations across graph (per graph).
      computed_graph_only_values = tf.math.unsorted_segment_sum(
        predictions,
        segment_ids=self.placeholders["graph_nodes_list"],
        num_segments=self.placeholders["graph_count"],
        name="computed_graph_only_values",
      )  # [g, c]

      # Add global features to the graph readout.
      x = tf.concat(
        [
          computed_graph_only_values,
          tf.cast(self.placeholders["graph_x"], tf.float32),
        ],
        axis=-1,
      )
      x = tf.layers.batch_normalization(
        x, training=self.placeholders["is_training"]
      )
      x = tf.layers.dense(
        x, FLAGS.auxiliary_inputs_dense_layer_size, activation=tf.nn.relu
      )
      x = tf.layers.dropout(
        x,
        rate=1 - self.placeholders["output_layer_dropout_keep_prob"],
        training=self.placeholders["is_training"],
      )
      predictions = tf.layers.dense(x, 2)

    if self.stats.graph_labels_dimensionality:
      targets = tf.argmax(
        self.placeholders["graph_y"],
        axis=1,
        output_type=tf.int32,
        name="targets",
      )
    elif self.stats.node_labels_dimensionality:
      targets = tf.argmax(
        self.placeholders["node_y"], axis=1, output_type=tf.int32
      )
    else:
      raise ValueError("No graph labels and no node labels!")

    argmaxed_predictions = tf.argmax(predictions, axis=1, output_type=tf.int32)
    accuracies = tf.equal(argmaxed_predictions, targets)

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    if self.stats.graph_labels_dimensionality:
      graph_only_loss = tf.compat.v1.losses.softmax_cross_entropy(
        self.placeholders["graph_y"], computed_graph_only_values
      )
      _loss = tf.compat.v1.losses.softmax_cross_entropy(
        self.placeholders["graph_y"], predictions
      )
      loss = (
        _loss + FLAGS.ggnn_intermediate_loss_discount_factor * graph_only_loss
      )
    elif self.stats.node_labels_dimensionality:
      if FLAGS.use_dsc_loss:
        # self.placeholders['node_y'] have shape (num_nodes_in_batch, 2)
        p1 = tf.nn.softmax(predictions[:, 0])
        y1 = tf.cast(self.placeholders["node_y"][:, 0], tf.float32)

        # we fix class 2 bc here 0 is the dominant mode!
        p2 = 1.0 - tf.nn.softmax(predictions[:, 1])
        y2 = 1.0 - tf.cast(self.placeholders["node_y"][:, 1], tf.float32)

        loss = (self.make_dsc_loss(p1, y1) + self.make_dsc_loss(p2, y2)) / 2.0
      else:
        loss = tf.compat.v1.losses.softmax_cross_entropy(
          self.placeholders["node_y"], predictions
        )
    else:
      raise ValueError("No graph labels and no node labels!")

    return loss, accuracies, accuracy, predictions

  def make_dsc_loss(self, p1: tf.Tensor, y1: tf.Tensor):
    normalization = tf.cast(self.placeholders["node_count"], tf.float32)
    numerator = (1.0 - p1) * p1 * y1
    denominator = ((1.0 - p1) * p1 + y1) * normalization
    neg_loss = tf.reduce_sum(numerator / (denominator + utils.SMALL_NUMBER))
    return 1.0 - neg_loss

  def MakeMinibatchIterator(
    self, epoch_type: str, groups: List[str], print_context: Any = None,
  ) -> Iterable[Tuple[log_database.BatchLogMeta, ggnn.FeedDict]]:
    """Create mini-batches by flattening adjacency matrices into a single
    adjacency matrix with multiple disconnected components."""
    options = graph_batcher.GraphBatchOptions(
      max_nodes=0 if FLAGS.batch_by_graph else FLAGS.batch_size,
      max_graphs=FLAGS.batch_size if FLAGS.batch_by_graph else 0,
      groups=groups,
      data_flow_max_steps_required=(
        None if epoch_type == "test" else self.message_passing_step_count
      ),
    )
    max_instance_count = (
      FLAGS.max_train_per_epoch
      if epoch_type == "train"
      else FLAGS.max_val_per_epoch
      if epoch_type == "val"
      else None
    )
    for batch in self.batcher.MakeGraphBatchIterator(
      options, max_instance_count, print_context=print_context
    ):
      feed_dict = utils.BatchDictToFeedDict(batch, self.placeholders)

      if epoch_type == "train":
        feed_dict.update(
          {
            self.placeholders[
              "graph_state_dropout_keep_prob"
            ]: FLAGS.graph_state_dropout_keep_prob,
            self.placeholders[
              "edge_weight_dropout_keep_prob"
            ]: FLAGS.edge_weight_dropout_keep_prob,
            self.placeholders[
              "output_layer_dropout_keep_prob"
            ]: FLAGS.output_layer_dropout_keep_prob,
            self.placeholders["is_training"]: True,
            self.placeholders["learning_rate"]: base_utils.GetLearningRate(
              self.epoch_num, FLAGS.epoch_count
            ),
          }
        )
      else:
        feed_dict.update(
          {
            self.placeholders["graph_state_dropout_keep_prob"]: 1.0,
            self.placeholders["edge_weight_dropout_keep_prob"]: 1.0,
            self.placeholders["output_layer_dropout_keep_prob"]: 1.0,
            self.placeholders["is_training"]: False,
          }
        )
      yield batch.log, feed_dict

  def MakeModularGraphOps(self):
    """ Maps from
            self.placeholders['raw_node_input_features'] and
            self.placeholders['raw_node_output_features']
        to modular
            loss, accuracies, accuracy and predictions
        ops.

        Depends on the usual attributes existing.
    """
    # get this out of the way:
    if (
      not (
        self.weights["regression_gate"] and self.weights["regression_transform"]
      )
      and self.encoded_node_x
      and self.placeholders["raw_node_input_features"]
      and self.placeholders["raw_node_output_features"]
    ):
      raise TypeError(
        "MakeModularGraphOps() call before "
        "MakeLossAndAccuracyAndPredictionOps() is not working!"
      )

    # map from placeholders for raw features to predictions
    predictions = utils.MakeModularOutputLayer(
      self.placeholders["raw_node_input_features"],
      self.placeholders["raw_node_output_features"],
      self.weights["regression_gate"],
      self.weights["regression_transform"],
    )

    targets = tf.argmax(
      self.placeholders["node_y"], axis=1, output_type=tf.int32
    )

    accuracies = tf.equal(
      tf.argmax(predictions, axis=1, output_type=tf.int32), targets
    )

    accuracy = tf.reduce_mean(tf.cast(accuracies, tf.float32))

    loss = tf.compat.v1.losses.softmax_cross_entropy(
      self.placeholders["node_y"], predictions
    )

    return loss, accuracies, accuracy, predictions


def main():
  """Main entry point."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/13): Only filter https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html
  warnings.filterwarnings("ignore")

  # TODO(github.com/ChrisCummins/ProGraML/issues/24): Replace with an Enum flag.
  if FLAGS.position_embeddings not in ["initial", "every", "fancy", "off"]:
    app.FatalWithoutStackTrace(
      "--position_embeddings has to be one of <initial, every, fancy, off>"
    )

  classifier_base.Run(Ggnn)


if __name__ == "__main__":
  app.Run(main)
