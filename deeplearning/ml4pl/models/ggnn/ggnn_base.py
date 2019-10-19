"""Base class for implementing gated graph neural networks."""
import os
import time

import sys
import numpy as np
import pathlib
import pickle
import random
import tensorflow as tf
import typing

import build_info
from deeplearning.ml4pl.models import batch_logger
from deeplearning.ml4pl.models import log_writer
from deeplearning.ml4pl.models.ggnn import ggnn_utils
from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import jsonutil
from labm8 import pbutil
from labm8 import prof
from labm8 import system

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
# Some of these flags define parameters which must be equal when restoring from
# file, such as the hidden layer sizes. Other parameters may change between
# runs of the same model, such as the input data batch size. To accomodate for
# this, models have a GetModelFlagNames() method which returns the list of flags
# which must be consistent between runs of the same model.
#
# For the sake of readability, these important model flags are saved into a
# global set here, so that the declaration of model flags is local to the
# declaration of the flag.
MODEL_FLAGS = set()

app.DEFINE_output_path(
    'working_dir',
    '/tmp/deeplearning/ml4pl/models/ggnn/',
    'The directory to write files to.',
    is_dir=True)

app.DEFINE_integer("random_seed", 42, "A random seed value.")

app.DEFINE_integer("num_epochs", 300, "The number of epochs to train for.")

app.DEFINE_integer(
    "patience", 300,
    "The number of epochs to train for without any improvement in validation "
    "accuracy before stopping.")

app.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")
MODEL_FLAGS.add("learning_rate")

# TODO(cec): Poorly understood:
app.DEFINE_float("clamp_gradient_norm", 1.0, "Clip gradients to L-2 norm.")
MODEL_FLAGS.add("clamp_gradient_norm")

app.DEFINE_float(
    "out_layer_dropout_keep_prob", 1.0,
    "Dropout keep probability on the output layer. In range 0 < x <= 1.")
MODEL_FLAGS.add("out_layer_dropout_keep_prob")

app.DEFINE_integer("hidden_size", 200, "The size of hidden layer(s).")
MODEL_FLAGS.add("hidden_size")

app.DEFINE_string(
    "embeddings", "inst2vec",
    "The type of embeddings to use. One of: {inst2vec,finetune,random}.")
MODEL_FLAGS.add("embeddings")

app.DEFINE_input_path(
    "embedding_path",
    bazelutil.DataPath('phd/deeplearning/ncc/published_results/emb.p'),
    "The path of the embeddings file to use.")

app.DEFINE_boolean(
    "tensorboard_logging", True,
    "If true, write tensorboard logs to '<working_dir>/tensorboard'.")

app.DEFINE_boolean(
    "test_on_improvement", True,
    "If true, test model accuracy on test data when the validation accuracy "
    "improves.")

app.DEFINE_input_path("restore_model", None,
                      "An optional file to restore the model from.")

# TODO(cec): Poorly understood.
app.DEFINE_boolean("freeze_graph_model", False, "???")

##### End of flag declarations.

# Type alias for the feed_dict argument of tf.Session.run().
FeedDict = typing.Dict[str, typing.Any]


class GgnnBaseModel(object):
  """Abstract base class for implementing gated graph neural networks.

  Subclasses must provide implementations of
  MakeLossAndAccuracyAndPredictionOps() and MakeMinibatchIterator().
  """

  def MakeLossAndAccuracyAndPredictionOps(
      self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    raise NotImplementedError("abstract class")

  def MakeMinibatchIterator(
      self, epoch_type: str) -> typing.Iterable[typing.Tuple[int, FeedDict]]:
    """Create and return an iterator over mini-batches of data.

    Args:
      epoch_type: The type of epoch to return mini-batches for.

    Returns:
      An iterator of mini-batches, where each mini-batch is a tuple of the batch
      size (as an int), and a feed dict to be fed to tf.Session.run().
    """
    raise NotImplementedError("abstract class")

  def GetModelFlagNames(self) -> typing.Iterable[str]:
    """Subclasses may extend this method to mark additional flags as important."""
    return MODEL_FLAGS

  def __init__(self):
    """Constructor.

    All subclasses that define constructors must call this first!
    """
    self.run_id: str = (f"{time.strftime('%Y.%m.%dT%H:%M:%S')}."
                        f"{system.HOSTNAME}.{os.getpid()}")

    self.working_dir = FLAGS.working_dir
    self.logger = log_writer.FormattedJsonLogWriter(
        self.working_dir / 'logs' / self.run_id)
    self.best_model_file = self.working_dir / f'{self.run_id}.best_model.pickle'
    self.working_dir.mkdir(exist_ok=True, parents=True)

    # Write app.Log() calls to file. To also log to stderr, use flag
    # --alsologtostderr.
    app.Log(
        1, 'Writing logs to `%s`. Unless --alsologtostderr flag is set, '
        'this is the last message you will see.', self.working_dir)
    app.LogToDirectory(self.working_dir, 'model')

    app.Log(1, "Build information: %s",
            jsonutil.format_json(pbutil.ToJson(build_info.GetBuildInfo())))

    app.Log(1, "Model flags: %s",
            jsonutil.format_json(self._ModelFlagsToDict()))

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=config)

    with self.graph.as_default():
      tf.set_random_seed(FLAGS.random_seed)
      with prof.Profile('Made model'):
        self.placeholders = {}
        self.weights = {}
        self.ops = {}
        self._MakeModel()
      with prof.Profile('Make training step'), tf.variable_scope("train_step"):
        self.ops["train_step"] = self._MakeTrainStep()

      # Restor or initialize the model:
      if FLAGS.restore_model:
        with prof.Profile('Restored model'):
          self.LoadModel(FLAGS.restore_model)
      else:
        with prof.Profile('Initialized model'):
          self.InitializeModel()

    # Progress counters. These are saved and restored from file.
    self.global_training_step = 0
    self.best_epoch_validation_accuracy = 0
    self.best_epoch_num = 0

    # Tensorboard logging.
    if FLAGS.tensorboard_logging:
      tensorboard_dir = self.working_dir / 'tensorboard' / self.run_id
      app.Log(1, f"Writing tensorboard logs to: `{tensorboard_dir}`")
      tensorboard_dir.mkdir(parents=True, exist_ok=True)
      self.summary_writers = {
          "train": tf.summary.FileWriter(tensorboard_dir / "train",
                                         self.sess.graph),
          "val": tf.summary.FileWriter(tensorboard_dir / "val",
                                       self.sess.graph),
          "test": tf.summary.FileWriter(tensorboard_dir / "test",
                                        self.sess.graph),
      }

  def RunEpoch(self, epoch_name: str,
               epoch_type: str) -> batch_logger.InMemoryBatchLogger:
    assert epoch_type in {"train", "val", "test"}
    logger = batch_logger.InMemoryBatchLogger(epoch_name)

    batch_iterator = ggnn_utils.ThreadedIterator(
        self.MakeMinibatchIterator(epoch_type), max_queue_size=5)
    for step, (batch_size, batch) in enumerate(batch_iterator):
      if not batch_size:
        raise ValueError("Mini-batch with zero graphs generated")

      self.global_training_step += 1

      fetch_list = [
          self.ops["loss"], self.ops["accuracy"], self.ops["summary_loss"]
      ]

      batch[self.placeholders["num_graphs"]] = batch_size

      if epoch_type == "train":
        batch[self.placeholders["out_layer_dropout_keep_prob"]] = (
            FLAGS.out_layer_dropout_keep_prob)
        fetch_list.append(self.ops["train_step"])
      else:
        batch[self.placeholders["out_layer_dropout_keep_prob"]] = 1.0

      batch_loss, batch_accuracy, loss_summary, *_ = self.sess.run(
          fetch_list, feed_dict=batch)

      if FLAGS.tensorboard_logging:
        self.summary_writers[epoch_type].add_summary(loss_summary,
                                                     self.global_training_step)
      logger.Log(
          batch_size=batch_size, loss=batch_loss, accuracy=batch_accuracy)
      sys.stdout.write(f'\r{logger}')
      sys.stdout.flush()

    logger.StopTheClock()
    sys.stdout.write('\r\n')
    app.Log(1, "%s: loss: %.5f | acc: %.3f%% | instances/sec: %.2f", epoch_name,
            logger.average_loss, logger.average_accuracy * 100,
            logger.instances_per_second)
    return logger

  def Train(self):
    with self.graph.as_default():
      for epoch_num in range(1, FLAGS.num_epochs + 1):
        epoch_start_time = time.time()
        print(f"== Epoch {epoch_num}")
        train = self.RunEpoch(f"Epoch {epoch_num} train", "train")
        valid = self.RunEpoch(f"Epoch {epoch_num}   val", "val")
        app.Log(
            1, "Epoch %s completed. Trained on %s instances and validated on "
            "%s instances in %s", epoch_num,
            humanize.Commas(train.instance_count),
            humanize.Commas(valid.instance_count),
            humanize.Duration(time.time() - epoch_start_time))
        log_file = self.logger.Log({
            "epoch": epoch_num,
            "time": time.time() - epoch_start_time,
            "train_results": train.ToJson(),
            "valid_results": valid.ToJson(),
        })

        if valid.average_accuracy > self.best_epoch_validation_accuracy:
          self.SaveModel(self.best_model_file)
          app.Log(
              1, "Best epoch so far, validation accuracy increased from "
              "%.5f from %.5f (+%.3f%%). Saving to '%s'",
              self.best_epoch_validation_accuracy, valid.average_accuracy,
              ((valid.average_accuracy / self.best_epoch_validation_accuracy) -
               1) * 100, self.best_model_file)
          self.best_epoch_validation_accuracy = valid.average_accuracy
          self.best_epoch_num = epoch_num

          # Run on test set.
          if FLAGS.test_on_improvement:
            test = self.RunEpoch(f"Epoch {epoch_num} (training)", "test")

            # Add the test results to the logfile.
            log = jsonutil.read_file(log_file)
            log["time"] = time.time() - start_time
            log['test_reults'] = (test.average_loss, test.average_accuracy,
                                  test.instances_per_second)
            jsonutil.write_file(log_file, log)
        elif epoch_num - self.best_epoch_num >= FLAGS.patience:
          app.Log(
              1, "Stopping training after %i epochs without "
              "improvement on validation accuracy", FLAGS.patience)
          break

  def Test(self):
    start_time = time.time()
    with self.graph.as_default():
      valid = self.RunEpoch("Test mode (validation)", "val")
      test = self.RunEpoch("Test mode (test)", "test")
      self.logger.Log({
          "time": time.time() - start_time,
          "valid_results": valid.ToJson(),
          "test_results": test.ToJson(),
      })

  def InitializeModel(self) -> None:
    with self.graph.as_default():
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      self.sess.run(init_op)

  def SaveModel(self, path: pathlib.Path) -> None:
    with self.graph.as_default():
      weights_to_save = {}
      for variable in self.sess.graph.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES):
        assert variable.name not in weights_to_save
        weights_to_save[variable.name] = self.sess.run(variable)
    # Save all of the flags and their values, as well as
    data_to_save = {
        "flags": app.FlagsToDict(json_safe=True),
        "model_flags": self._ModelFlagsToDict(),
        "weights": weights_to_save,
        "build_info": pbutil.ToJson(build_info.GetBuildInfo()),
        "global_training_step": self.global_training_step,
        "best_epoch_validation_accuracy": self.best_epoch_validation_accuracy,
        "best_epoch_num": self.best_epoch_num,
    }
    with open(path, "wb") as out_file:
      pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

  def LoadModel(self, path: pathlib.Path) -> None:
    """Load and restore the model from the given model file.

    Args:
      path: The path of the file to restore from, as created by SaveModel().

    Raises:
      EnvironmentError: If the flags in the saved model do not match the current
        model flags.
    """
    with prof.Profile(f"Read pickled model from `{path}`"):
      with open(path, "rb") as in_file:
        data_to_load = pickle.load(in_file)

    # Restore progress counters.
    self.global_training_step = data_to_load.get("global_training_step", 0)
    self.best_epoch_validation_accuracy = data_to_load.get(
        "best_epoch_validation_accuracy", 0)
    self.best_epoch_num = data_to_load.get("best_epoch_num", 0)

    # Assert that we got the same model configuration.
    # Flag values found in the saved file but not present currently are ignored.
    flags = self._ModelFlagsToDict()
    saved_flags = data_to_load["model_flags"]
    flag_names = set(flags.keys())
    saved_flag_names = set(saved_flags.keys())
    if flag_names != saved_flag_names:
      raise EnvironmentError(
          "Saved flags do not match current flags. "
          f"Flags not found in saved flags: '{flag_names - saved_flag_names}'."
          f"Saved flags not present now: '{saved_flag_names - flag_names}'")
    for flag, flag_value in flags.items():
      if flag_value != saved_flags[flag]:
        raise EnvironmentError(
            f"Saved flag {flag} value does not match current value:"
            f"'{saved_flags[flag]}' != '{flag_value}'")

    with self.graph.as_default():
      variables_to_initialize = []
      with tf.name_scope("restore"):
        restore_ops = []
        used_vars = set()
        for variable in self.sess.graph.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES):
          used_vars.add(variable.name)
          if variable.name in data_to_load["weights"]:
            restore_ops.append(
                variable.assign(data_to_load["weights"][variable.name]))
          else:
            app.Log(
                1, "Freshly initializing %s since no saved value "
                "was found.", variable.name)
            variables_to_initialize.append(variable)
        for var_name in data_to_load["weights"]:
          if var_name not in used_vars:
            app.Log(1, "Saved weights for %s not used by model.", var_name)
        restore_ops.append(tf.variables_initializer(variables_to_initialize))
        self.sess.run(restore_ops)

  def _ModelFlagsToDict(self) -> typing.Dict[str, typing.Any]:
    """Return the flags which are """
    return {
        flag: jsonutil.JsonSerializable(getattr(FLAGS, flag))
        for flag in sorted(set(self.GetModelFlagNames()))
    }

  def _MakeModel(self):
    self.placeholders["num_graphs"] = tf.placeholder(
        tf.int32, [], name="num_graphs")
    self.placeholders["out_layer_dropout_keep_prob"] = tf.placeholder(
        tf.float32, [], name="out_layer_dropout_keep_prob")

    self.weights["embedding_table"] = self._MakeEmbeddingsTable()

    with tf.variable_scope("graph_model"):
      self.ops["loss"], self.ops["accuracy"], self.ops["predictions"] = (
          self.MakeLossAndAccuracyAndPredictionOps())

    # Tensorboard summaries.
    self.ops["summary_loss"] = tf.summary.scalar("loss", self.ops["loss"])
    # TODO(cec): More tensorboard telemetry: input class distributions,
    # predicted class distributions, etc.

  def _MakeEmbeddingsTable(self) -> tf.Tensor:
    with prof.Profile(f"Read embeddings table `{FLAGS.embedding_path}`"):
      with open(FLAGS.embedding_path, 'rb') as f:
        embedding_table = pickle.load(f)

    if FLAGS.embeddings == "inst2vec":
      return tf.constant(
          embedding_table, dtype=tf.float32, name="embedding_table")
    elif FLAGS.embeddings == "finetune":
      return tf.Variable(
          embedding_table,
          dtype=tf.float32,
          name="embedding_table",
          trainable=True)
    elif FLAGS.embeddings == "random":
      return tf.Variable(
          ggnn_utils.uniform_init(np.shape(embedding_table)),
          dtype=tf.float32,
          name="embedding_table",
          trainable=True)
    else:
      raise ValueError("Invalid value for --embeding. Supported values are "
                       "{inst2vec,finetune,random}")

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
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
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

  # TODO(cec):
  # def predict(self, data):
  #   loss = 0
  #   accuracies, predictions = [], []
  #   start_time = time.time()
  #   processed_graphs = 0
  #   batch_iterator = ggnn_utils.ThreadedIterator(
  #       self.MakeMinibatchIterator(data, is_training=False), max_queue_size=5
  #   )
  #
  #   for step, batch in enumerate(batch_iterator):
  #     num_graphs = batch[self.placeholders["num_graphs"]]
  #     processed_graphs += num_graphs
  #
  #     batch[self.placeholders["out_layer_dropout_keep_prob"]] = 1.0
  #     fetch_list = [self.ops["loss"], self.ops["accuracy"], self.ops["predictions"]]
  #
  #     batch_loss, batch_accuracy, _preds, *_ = self.sess.run(
  #         fetch_list, feed_dict=batch
  #     )
  #     loss += batch_loss * num_graphs
  #     accuracies.append(np.array(batch_accuracy) * num_graphs)
  #     predictions.extend(_preds)
  #
  #     print(
  #         "Running prediction, batch %i (has %i graphs). Loss so far: %.4f"
  #         % (step, num_graphs, loss / processed_graphs),
  #         end="\r",
  #         )
  #
  #   accuracy = np.sum(accuracies, axis=0) / processed_graphs
  #   loss = loss / processed_graphs
  #   instance_per_sec = processed_graphs / (time.time() - start_time)
  #   return predictions, loss, accuracy, instance_per_sec
