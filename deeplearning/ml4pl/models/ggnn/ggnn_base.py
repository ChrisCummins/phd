"""Base class for implementing gated graph neural networks."""
import typing

import numpy as np
import tensorflow as tf
from labm8 import app
from labm8 import humanize
from labm8 import prof

from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models.ggnn import ggnn_utils as utils

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

app.DEFINE_integer(
    "patience", 300,
    "The number of epochs to train for without any improvement in validation "
    "accuracy before stopping.")

app.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")
classifier_base.MODEL_FLAGS.add("learning_rate")

# TODO(cec): Poorly understood:
app.DEFINE_float("clamp_gradient_norm", 1.0, "Clip gradients to L-2 norm.")
classifier_base.MODEL_FLAGS.add("clamp_gradient_norm")

app.DEFINE_integer("hidden_size", 200, "The size of hidden layer(s).")
classifier_base.MODEL_FLAGS.add("hidden_size")

app.DEFINE_string(
    "embeddings", "inst2vec",
    "The type of embeddings to use. One of: {inst2vec,finetune,random}.")
classifier_base.MODEL_FLAGS.add("embeddings")

app.DEFINE_boolean(
    "tensorboard_logging", True,
    "If true, write tensorboard logs to '<working_dir>/tensorboard'.")

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
        self.weights = {
            "embedding_table": self._GetEmbeddingsTable(),
        }

        self.placeholders = utils.MakePlaceholders(self.stats)

        self.ops = {}
        with tf.compat.v1.variable_scope("graph_model"):
          (self.ops["loss"], self.ops["accuracies"], self.ops["accuracy"],
           self.ops["predictions"]) = (
               self.MakeLossAndAccuracyAndPredictionOps())

        # Tensorboard summaries.
        self.ops["summary_loss"] = tf.summary.scalar("loss", self.ops["loss"])
        self.ops["summary_accuracy"] = tf.summary.scalar(
            "accuracy", self.ops["accuracy"])
        # TODO(cec): More tensorboard telemetry: input class distributions,
        # predicted class distributions, etc.

        with prof.Profile('Make training step'), tf.compat.v1.variable_scope(
            "train_step"):
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

  @property
  def message_passing_step_count(self) -> int:
    return self.layer_timesteps.sum()

  @property
  def layer_timesteps(self) -> np.array:
    return np.array([int(x) for x in FLAGS.layer_timesteps])

  def RunMinibatch(self, epoch_type: str, feed_dict: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    fetch_dict = {
        "loss": self.ops["loss"],
        "accuracies": self.ops["accuracies"],
        "accuracy": self.ops["accuracy"],
        "predictions": self.ops["predictions"],
        "summary_loss": self.ops["summary_loss"],
        "summary_accuracy": self.ops["summary_accuracy"],
    }

    if epoch_type == "train":
      fetch_dict["train_step"] = self.ops["train_step"]

    fetch_dict = utils.RunWithFetchDict(self.sess, fetch_dict, feed_dict)

    if FLAGS.tensorboard_logging:
      self.summary_writers[epoch_type].add_summary(fetch_dict["summary_loss"],
                                                   self.global_training_step)
      self.summary_writers[epoch_type].add_summary(
          fetch_dict["summary_accuracy"], self.global_training_step)

    # TODO(cec): Add support for edge labels.
    targets = (feed_dict[self.placeholders['node_y']]
               if self.placeholders['node_y'] in feed_dict else
               feed_dict[self.placeholders['graph_y']])

    return self.MinibatchResults(
        loss=float(fetch_dict['loss']),
        y_true_1hot=targets,
        y_pred_1hot=fetch_dict['predictions'])

  def InitializeModel(self) -> None:
    super(GgnnBaseModel, self).InitializeModel()
    with self.graph.as_default():
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      self.sess.run(init_op)

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
        restore_ops.append(tf.variables_initializer(variables_to_initialize))
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
    optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS.learning_rate)
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
    self.sess.run(tf.local_variables_initializer())

    return train_step
