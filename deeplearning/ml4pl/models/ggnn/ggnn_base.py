"""Base class for implementing gated graph neural networks."""
import json
import os
import time

import numpy as np
import pickle
import random
import tensorflow as tf
import typing

from deeplearning.ml4pl.models.ggnn import ggnn_utils
from labm8 import app
from labm8 import bazelutil
from labm8 import jsonutil
from labm8 import prof
from labm8 import system


FLAGS = app.FLAGS

app.DEFINE_output_path(
    'working_dir', '/tmp/deeplearning/ml4pl/models/ggnn/',
    'The directory to write files to.', is_dir=True)
app.DEFINE_integer(
    "num_epochs", 300,
    "The number of epochs to train for.")
app.DEFINE_integer(
    "patience", 300,
    "The number of epochs to train for without any improvement in validation "
    "accuracy before stopping.")
app.DEFINE_float(
    "learning_rate", 0.001,
    "The initial learning rate.")
# TODO(cec): Poorly understood:
app.DEFINE_float(
    "clamp_gradient_norm", 1.0,
    "Clip gradients to L-2 norm.")
app.DEFINE_float(
    "out_layer_dropout_keep_prob", 1.0,
    "Dropout keep probability on the output layer. In range 0 < x <= 1.")
app.DEFINE_integer(
    "hidden_size", 200,
    "The size of hidden layer(s).")
# TODO(cec): Poorly understood:
app.DEFINE_boolean(
    "use_graph", True,
    "???")
app.DEFINE_boolean(
    "tie_fwd_bkwd", True,
    "If true, add backward edges.")
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
app.DEFINE_input_path(
    "restore_model", None,
    "An optional file to restore the model from.")
# TODO(cec): Poorly understood.
app.DEFINE_boolean(
    "freeze_graph_model", False,
    "???")

def FlagsToDict():
  """Return the flags """
  flags_dict = app.FlagsToDict()
  # Strip only the flags that belong in this package.
  model_flags_dict = {k: v for k, v in flags_dict.items()
                      if 'deeplearning.m4lpl' in k}
  return model_flags_dict

def ModelFlagsToDict():
  model_flags = FlagsToDict()
  return model_flags

class GGNNBaseModel(object):

  def __init__(self):
    """Constructor. All subclasses must call this first!"""
    self.run_id: str = (f"{time.strftime('%Y.%m.%dT%H:%M:%S')}."
                        f"{system.HOSTNAME}.{os.getpid()}")

    self.working_dir = FLAGS.working_dir
    self.log_dir = self.working_dir / 'logs' / self.run_id
    self.best_model_file = self.working_dir / 'best_mode.pickle'
    self.working_dir.mkdir(exist_ok=True, parents=True)

    # Log to file. To also log to stderr, use flag --alsologtostderr.
    app.Log(1, 'Writing files to `%s`', self.working_dir)
    app.LogToDirectory(self.working_dir, 'model')

    app.Log(1, "Using the following parameters:\n%s",
            jsonutil.format_json(FlagsToDict()))

    with prof.Profile(f"Read embeddings table `{FLAGS.embedding_path}`"):
      with open(FLAGS.embedding_path, 'rb') as f:
        self.embedding_table = pickle.load(f)

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    # Build the actual model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=config)

    with self.graph.as_default():
      tf.set_random_seed(FLAGS.random_seed)
      self.placeholders = {}
      self.constants = {}
      self.weights = {}
      self.ops = {}
      with prof.Profile('Made model'):
        self._MakeModel()
      with prof.Profile('Make training step'), tf.variable_scope("train_step"):
          self.ops["train_step"] = self._MakeTrainStep()

      # Restor or initialize the model:
      if FLAGS.restore_model:
        with prof.Profile('Restored model'):
          self._LoadModel(FLAGS.restore_model)
      else:
        with prof.Profile('Initialized model'):
          self._InitializeModel()

    # Tensorboard logging.
    self.global_training_step = 0
    if FLAGS.tensorboard_logging:
      tensorboard_dir = self.working_dir / 'tensorboard' / self.run_id
      app.Log(1, f"Writing tensorboard logs to: `{tensorboard_dir}`")
      tensorboard_dir.mkdir(parents=True, exist_ok=True)
      self.summary_writers = {
          "train": tf.summary.FileWriter(
              tensorboard_dir / "train", self.sess.graph),
          "val": tf.summary.FileWriter(
              tensorboard_dir / "val", self.sess.graph),
          "test": tf.summary.FileWriter(
              tensorboard_dir / "test", self.sess.graph),
      }

  def GetNodeFeaturesDimensionality(self) -> int:
    raise NotImplementedError("abstract class")

  def GetNumberOfClasses(self) -> int:
    raise NotImplementedError("abstract class")

  def GetNumberOfEdgeTypes(self) -> int:
    raise NotImplementedError("abstract class")

  def _MakeModel(self):
    self.placeholders["target_values"] = self.MakeTargetValuesPlaceholder()
    self.placeholders["num_graphs"] = tf.placeholder(
        tf.int32, [], name="num_graphs")
    self.placeholders["out_layer_dropout_keep_prob"] = tf.placeholder(
        tf.float32, [], name="out_layer_dropout_keep_prob")

    if self.params["embeddings"] == "inst2vec":
      self.constants["embedding_table"] = tf.constant(self.embedding_table,
                                                     dtype=tf.float32,
                                                     name="embedding_table")
    elif self.params["embeddings"] == "finetune":
      self.constants["embedding_table"] = tf.Variable(self.embedding_table,
                                                     dtype=tf.float32,
                                                     name="embedding_table",
                                                     trainable=True)
    elif self.params["embeddings"] == "random":
      self.constants["embedding_table"] = tf.Variable(
          ggnn_utils.uniform_init(np.shape(self.embedding_table)),
                                                     dtype=tf.float32,
                                                     name="embedding_table",
                                                     trainable=True)

    with tf.variable_scope("graph_model"):
      self.PrepareSpecificGraphModel()
      # This does the actual graph work:
      if FLAGS.use_graph:
        self.ops["final_node_representations"] = (
          self.ComputeFinalNodeRepresentations())
      else:
        self.ops["final_node_representations"] = tf.zeros(shape=[
            self.placeholders["number_of_nodes"],
            FLAGS.hidden_size,
        ])

    with tf.variable_scope("out_layer"):
      self.ops["loss"], self.ops["accuracy"], self.ops["predictions"] = (self.MakeLossAndAccuracyAndPredictionOps())

    # Tensorboard summaries.
    self.ops["summary_loss"] = tf.summary.scalar("loss", self.ops["loss"])

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
    grads_and_vars = optimizer.compute_gradients(self.ops["loss"],
                                                 var_list=trainable_vars)
    clipped_grads = []
    for grad, var in grads_and_vars:
      if grad is not None:
        clipped_grads.append((tf.clip_by_norm(
            grad, FLAGS.clamp_gradient_norm), var))
      else:
        clipped_grads.append((grad, var))
    train_step = optimizer.apply_gradients(clipped_grads)

    # Also run batch_norm update ops, if any.
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_step = tf.group([train_step, update_ops])

    # Initialize newly-introduced variables:
    self.sess.run(tf.local_variables_initializer())

    return train_step

  def MakeTargetValuesPlaceholder(self) -> tf.Tensor:
    raise NotImplementedError("abstract class")

  def MakeLossAndAccuracyAndPredictionOps(self) -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    raise NotImplementedError("abstract class")

  def PrepareSpecificGraphModel(self) -> None:
    raise NotImplementedError("abstract class")

  def ComputeFinalNodeRepresentations(self) -> tf.Tensor:
    raise NotImplementedError("abstract class")

  def MakeMinibatchIterator(self, epoch_type: str):
    raise NotImplementedError("abstract class")

  def RunEpoch(self, epoch_name: str, epoch_type: str):
    assert epoch_type in {"train", "val", "test"}

    loss = 0
    accuracies = []
    accuracy_op = self.ops["accuracy"]
    start_time = time.time()
    processed_graphs = 0
    batch_iterator = ggnn_utils.ThreadedIterator(
        self.MakeMinibatchIterator(epoch_type), max_queue_size=5)
    for step, batch_data in enumerate(batch_iterator):
      self.global_training_step += 1
      num_graphs = batch_data[self.placeholders["num_graphs"]]
      processed_graphs += num_graphs
      fetch_list = [self.ops["loss"], accuracy_op, self.ops["summary_loss"]]
      if epoch_type == "train":
        batch_data[self.placeholders["out_layer_dropout_keep_prob"]] = (
            FLAGS.out_layer_dropout_keep_prob)
        fetch_list.append(self.ops["train_step"])
      else:
        batch_data[self.placeholders["out_layer_dropout_keep_prob"]] = 1.0

      batch_loss, batch_accuracy, loss_summary, *_ = self.sess.run(
          fetch_list, feed_dict=batch_data)
      loss += batch_loss * num_graphs
      accuracies.append(np.array(batch_accuracy) * num_graphs)

      if FLAGS.tensorboard_logging:
        self.summary_writers[epoch_type].add_summary(
            loss_summary, self.global_training_step)

      print(
          "Running %s, batch %i (has %i graphs). Loss so far: %.4f" %
          (epoch_name, step, num_graphs, loss / processed_graphs),
          end="\r",
      )

    accuracy = np.sum(accuracies, axis=0) / processed_graphs
    loss /= processed_graphs
    instance_per_sec = processed_graphs / (time.time() - start_time)
    return loss, accuracy, instance_per_sec


  def Train(self):
    total_time_start = time.time()
    with self.graph.as_default():
      if FLAGS.restore_model:
        _, valid_acc, _, _ = self.RunEpoch("Resumed (validation)", "val")
        best_val_acc = np.sum(valid_acc)
        best_val_acc_epoch = 0
        print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" %
              best_val_acc)
        app.Log(1, "Resumed operation, initial cum. val. acc: %.5f",
                best_val_acc)
      else:
        (best_val_acc, best_val_acc_epoch) = (0.0, 0)
      for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"== Epoch {epoch}")

        train_loss, train_acc, train_speed = self.RunEpoch(
            "epoch %i (training)" % epoch, "train")
        print("\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f" %
              (train_loss, f"{train_acc:.5f}", train_speed))
        app.Log(
            1, "Epoch %s training: loss: %.5f | acc: %s | "
            "instances/sec: %.2f", epoch, train_loss, f"{train_acc:.5f}",
            train_speed)

        valid_loss, valid_acc, valid_speed = self.RunEpoch(
            "epoch %i (validation)" % epoch, "val")
        print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" %
              (valid_loss, f"{valid_acc:.5f}", valid_speed))
        app.Log(
            1, "Epoch %d validation: loss: %.5f | acc: %s | "
            "instances/sec: %.2f", epoch, valid_loss, f"{valid_acc:.5f}",
            valid_speed)

        epoch_time = time.time() - total_time_start
        log_file = self._WriteLog({
          "epoch": epoch,
          "time": epoch_time,
          "train_results": (train_loss, train_acc.tolist(), train_speed),
          "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
        })

        # TODO: sum seems redundant if only one task is trained.
        val_acc = np.sum(valid_acc)  # type: float
        if val_acc > best_val_acc:
          self._SaveModel(self.best_model_file)
          print("  (Best epoch so far, cum. val. acc increased to "
                "%.5f from %.5f. Saving to '%s')" %
                (val_acc, best_val_acc, self.best_model_file))
          app.Log(
              1, "Best epoch so far, cum. val. acc increased to "
              "%.5f from %.5f. Saving to '%s'", val_acc, best_val_acc,
              self.best_model_file)
          best_val_acc = val_acc
          best_val_acc_epoch = epoch

          # Run on test set.
          if FLAGS.test_on_improvement:
            test_loss, test_acc, test_speed = self.RunEpoch(
                f"epoch {epoch} (training)", "test")
            print("\r\x1b[K Epoch %d Test: loss: %.5f | acc: %s | "
                  "instances/sec: %.2f" %
                  (test_loss, f"{test_acc:.5f}", test_speed))
            app.Log(
                1, "Epoch %d Test: loss: %.5f | acc: %s | "
                "instances/sec: %.2f", test_loss, f"{test_acc:.5f}", test_speed)

            # Add the test results to the logfile.
            log = jsonutil.read_file(log_file)
            log["time"] = time.time() - total_time_start
            log['test_reults'] = (test_loss, test_acc.tolist(), test_speed)
            jsonutil.write_file(log_file, log)
        elif epoch - best_val_acc_epoch >= FLAGS.patience:
          app.Log(
              1, "Stopping training after %i epochs without "
              "improvement on validation accuracy", FLAGS.patience)
          break

  def Test(self):
    total_time_start = time.time()

    with self.graph.as_default():
      valid_loss, valid_acc, valid_speed = self.RunEpoch(
          "Test mode (validation)", "val")
      print("\r\x1b[K Validation: loss: %.5f | acc: %s | instances/sec: %.2f" %
            (valid_loss, f"{valid_acc:.5f}", valid_speed))
      app.Log(1, "Validation: loss: %.5f | acc: %s | instances/sec: %.2f",
              valid_loss, f"{valid_acc:.5f}", valid_speed)

      test_loss, test_acc, test_speed = self.RunEpoch("Test mode (test)",
                                                       "test")
      print("\r\x1b[K Test: loss: %.5f | acc: %s | instances/sec: %.2f" %
            (test_loss, f"{test_acc:.5f}", test_speed))
      app.Log(1, "Test: loss: %.5f | acc: %s | instances/sec: %.2f", test_loss,
              f"{test_acc:.5f}", test_speed)

      epoch_time = time.time() - total_time_start
      self._WriteLog({
        "time": epoch_time,
        "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
        "test_results": (test_loss, test_acc.tolist(), test_speed),
      })

  def _WriteLog(self, log: typing.Dict[str, typing.Any]) -> pathlib.Path:
    log_file = self.log_dir / f'{time.strftime('%Y.%m.%dT%H:%M:%S')}.json'
    jsonutil.write_file(log_file, log)
    return log_file

  def _InitializeModel(self) -> None:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)

  def _SaveModel(self, path: pathlib.Path) -> None:
    weights_to_save = {}
    for variable in self.sess.graph.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES):
      assert variable.name not in weights_to_save
      weights_to_save[variable.name] = self.sess.run(variable)
    data_to_save = {
      "params": ModelFlagsToDict(),
      "weights": weights_to_save,
      "global_training_step": self.global_training_step
    }
    with open(path, "wb") as out_file:
      pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

  def _LoadModel(self, path: pathlib.Path) -> None:
    with prof.Profile(f"Read pickled model from `{path}`"):
      with open(path, "rb") as in_file:
        data_to_load = pickle.load(in_file)

    self.global_training_step = data_to_load.get("global_training_step", 0)

    # Assert that we got the same model configuration
    params = ModelFlagsToDict()
    if len(params) != len(data_to_load["params"]):
      raise EnvironmentError('unmatched params')
    for (par, par_value) in params.items():
      # Fine to have different task_ids:
      assert par_value == data_to_load["params"][par], f"Failed at {par}: {par_value} vs. data_to_load: {data_to_load['params'][par]}"

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
          app.Log(1, "Freshly initializing %s since no saved value "
                  "was found.", variable.name)
          variables_to_initialize.append(variable)
      for var_name in data_to_load["weights"]:
        if var_name not in used_vars:
          app.Log(1, "Saved weights for %s not used by model.", var_name)
      restore_ops.append(tf.variables_initializer(variables_to_initialize))
      self.sess.run(restore_ops)

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
  #   for step, batch_data in enumerate(batch_iterator):
  #     num_graphs = batch_data[self.placeholders["num_graphs"]]
  #     processed_graphs += num_graphs
  #
  #     batch_data[self.placeholders["out_layer_dropout_keep_prob"]] = 1.0
  #     fetch_list = [self.ops["loss"], self.ops["accuracy"], self.ops["predictions"]]
  #
  #     batch_loss, batch_accuracy, _preds, *_ = self.sess.run(
  #         fetch_list, feed_dict=batch_data
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
