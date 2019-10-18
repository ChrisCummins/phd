"""Base class for implementing gated graph neural networks."""
import json
import os
import time

import numpy as np
import pathlib
import pickle
import random
import tensorflow as tf

from deeplearning.ml4pl.models.ggnn import ggnn_utils
from labm8 import app
from labm8 import prof

FLAGS = app.FLAGS


class GGNNBaseModel(object):

  @classmethod
  def default_params(cls):
    return {
        "num_epochs": 300,
        "patience": 300,
        "learning_rate": 0.001,
        "clamp_gradient_norm": 1.0,
        "out_layer_dropout_keep_prob": 1.0,
        "hidden_size": 200,
        "num_timesteps": 8,  # TODO: this is unused!
        "use_graph": True,
        "tie_fwd_bkwd": True,
        # "task_ids": [0],
        "random_seed": 42,
        "train_file": "train.json",
        "valid_file": "val.json",
        "test_file": "test.json",
        "emb_file": "emb.p",
        "num_classes": 104,
        "tensorboard_logging": False,
        "test_on_improvement": False,
    }

  def __init__(self, args):
    self.args = args

    # Collect argument things:
    data_dir = ""
    if "--data_dir" in args and args["--data_dir"] is not None:
      data_dir = args["--data_dir"]
    self.data_dir = data_dir

    self.run_id = "_".join(
        [time.strftime("%Y-%m-%d-%H-%M-%S"),
         str(os.getpid())])
    log_dir = args.get("--log_dir") or "."
    self.log_file = os.path.join(log_dir, "%s_log.json" % self.run_id)
    # Log to file. To also log to stderr, use flag --alsologtostderr.
    app.Log(1, 'Writing logs to %s', pathlib.Path(log_dir) / self.run_id)
    app.LogToDirectory(pathlib.Path(log_dir) / self.run_id, 'model')
    self.test_log_file = os.path.join(log_dir, f"{self.run_id}_test_log.json")
    self.best_model_file = os.path.join(log_dir,
                                        "%s_model_best.pickle" % self.run_id)

    # Collect parameters:
    params = self.default_params()
    config_file = args.get("--config-file")
    if config_file is not None:
      with open(config_file, "r") as f:
        params.update(json.load(f))
    config = args.get("--config")
    if config is not None:
      params.update(json.loads(config))
    self.params = params
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    with open(os.path.join(log_dir, "%s_params.json" % self.run_id), "w") as f:
      json.dump(params, f)
    app.Log(1, "Run %s starting with following parameters:\n%s", self.run_id,
            json.dumps(self.params))
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    # Load data:
    self.max_num_vertices = 0
    self.num_edge_types = 0
    self.annotation_size = 0
    with prof.Profile("Loaded datasets"):
      self.load_datasets()

    self.emb_table_ndarray = self.load_emb(params["emb_file"])

    app.Log(1, "building the actual model...")
    # Build the actual model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph, config=config)
    with self.graph.as_default():
      tf.set_random_seed(params["random_seed"])
      self.placeholders = {}
      self.constants = {}
      self.weights = {}
      self.ops = {}
      with prof.Profile('Make model'):
        self.make_model()
      with prof.Profile('Make training step'):
        with tf.variable_scope("train_step"):
          self.make_train_step()

      # Restore/initialize variables:
      restore_file = args.get("--restore")
      if restore_file is not None:
        with prof.Profile('Restore model'):
          self.restore_model(restore_file)
      else:
        with prof.Profile('Initialize model'):
          self.initialize_model()

    self.global_training_step = 0

    if self.params["tensorboard_logging"]:
      tensorboard_dir = (pathlib.Path(self.data_dir) / 'tensorboard' /
                         self.run_id)
      app.Log(1, f"Writing tensorboard logs to: {tensorboard_dir}")
      tensorboard_dir.mkdir(parents=True, exist_ok=True)
      self.summary_writers = {
          "train": tf.summary.FileWriter(tensorboard_dir / "train",
                                         self.sess.graph),
          "val": tf.summary.FileWriter(tensorboard_dir / "val",
                                       self.sess.graph),
          "test": tf.summary.FileWriter(tensorboard_dir / "test",
                                        self.sess.graph),
      }

  def load_emb(self, emb_file_name):
    full_path = os.path.join(self.data_dir, emb_file_name)
    with prof.Profile(f"loaded embedding file from `{full_path}`"):
      with open(full_path, "rb") as f:
        emb = pickle.load(f)
    return emb

  def load_data(self, file_name, is_training_data: bool):
    full_path = os.path.join(self.data_dir, file_name)
    with prof.Profile(f"loaded data `{full_path}`"):
      with open(full_path, "r") as f:
        data = json.load(f)

    restrict = self.args.get("--restrict_data")
    if restrict is not None and restrict > 0:
      data = data[:restrict]

    # Get some common data out:
    ss = time.time()
    app.Log(1, "Getting some common data from loaded json...")
    num_fwd_edge_types = 0
    for g in data:
      self.max_num_vertices = max(
          self.max_num_vertices,
          g['number_of_nodes'],
          # max([v for e in g["graph"] for v in [e[0], e[2]]]),
      )
      num_fwd_edge_types = max(num_fwd_edge_types,
                               max([e[1] for e in g["graph"]]))
    self.num_edge_types = max(
        self.num_edge_types,
        (num_fwd_edge_types + 1) * (1 if self.params["tie_fwd_bkwd"] else 2),
    )  # +1 because edges count from 0.

    # Dataset is created with 'none' type on some edges! hotfix / TODO
    if self.params["map_none_type_to_ctrl"]:
      self.num_edge_types -= 1

    self.annotation_size = max(self.annotation_size,
                               len(data[0].get("node_features", [[0]])[0]))
    app.Log(
        1,
        f"Getting some common data from loaded json... took {time.time() - ss} seconds."
    )

    return self.process_raw_graphs(data, is_training_data)

  @staticmethod
  def graph_string_to_array(graph_string: str) -> List[List[int]]:
    return [[int(v) for v in s.split(" ")] for s in graph_string.split("\n")]

  def process_raw_graphs(self, raw_data: Sequence[Any],
                         is_training_data: bool) -> Any:
    raise Exception("Models have to implement process_raw_graphs!")

  def make_model(self):
    self.placeholders["target_values"] = self.make_target_values_placeholder()
    self.placeholders["num_graphs"] = tf.placeholder(tf.int32, [],
                                                     name="num_graphs")
    self.placeholders["out_layer_dropout_keep_prob"] = tf.placeholder(
        tf.float32, [], name="out_layer_dropout_keep_prob")

    if self.params["embeddings"] == "inst2vec":
      self.constants["edge_emb_table"] = tf.constant(self.emb_table_ndarray,
                                                     dtype=tf.float32,
                                                     name="edge_emb_table")
    elif self.params["embeddings"] == "finetune":
      self.constants["edge_emb_table"] = tf.Variable(self.emb_table_ndarray,
                                                     dtype=tf.float32,
                                                     name="edge_emb_table",
                                                     trainable=True)
    elif self.params["embeddings"] == "random":
      self.constants["edge_emb_table"] = tf.Variable(ggnn_utils.uniform_init(
          np.shape(self.emb_table_ndarray)),
                                                     dtype=tf.float32,
                                                     name="edge_emb_table",
                                                     trainable=True)

    with tf.variable_scope("graph_model"):
      self.prepare_specific_graph_model()
      # This does the actual graph work:
      if self.params["use_graph"]:
        self.ops[
            "final_node_representations"] = self.compute_final_node_representations(
            )
      else:
        self.ops["final_node_representations"] = tf.zeros(shape=[
            self.placeholders["number_of_nodes"],
            self.params["hidden_size"],
        ])

    with tf.variable_scope("out_layer"):
      self.ops["loss"], self.ops["accuracy"] = self.make_loss_and_accuracy_ops()

    self.ops["summary_loss"] = tf.summary.scalar("loss", self.ops["loss"])

  def make_train_step(self):
    trainable_vars = self.sess.graph.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES)
    if self.args.get("--freeze-graph-model"):
      graph_vars = set(
          self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope="graph_model"))
      filtered_vars = []
      for var in trainable_vars:
        if var not in graph_vars:
          filtered_vars.append(var)
        else:
          app.Log(1, "Freezing weights of variable %s." % var.name)
      trainable_vars = filtered_vars
    optimizer = tf.train.AdamOptimizer(self.params["learning_rate"])
    grads_and_vars = optimizer.compute_gradients(self.ops["loss"],
                                                 var_list=trainable_vars)
    clipped_grads = []
    for grad, var in grads_and_vars:
      if grad is not None:
        clipped_grads.append((tf.clip_by_norm(
            grad, self.params["clamp_gradient_norm"]), var))
      else:
        clipped_grads.append((grad, var))
    self.ops["train_step"] = optimizer.apply_gradients(clipped_grads)
    # Initialize newly-introduced variables:
    self.sess.run(tf.local_variables_initializer())

  def make_target_values_placeholder(self) -> tf.Tensor:
    raise Exception("Models have to implement make_target_values_placeholder!")

  def make_loss_and_accuracy_ops(self) -> Tuple[tf.Tensor, tf.Tensor]:
    raise Exception("Models have to implement make_loss_and_accuracy_ops!")

  def prepare_specific_graph_model(self) -> None:
    raise Exception("Models have to implement prepare_specific_graph_model!")

  def compute_final_node_representations(self) -> tf.Tensor:
    raise Exception(
        "Models have to implement compute_final_node_representations!")

  def make_minibatch_iterator(self, epoch_type: str):
    raise Exception("Models have to implement make_minibatch_iterator!")

  def run_epoch(self, epoch_name: str, epoch_type: str):
    assert epoch_type in {"train", "val", "test"}

    loss = 0
    accuracies = []
    accuracy_op = self.ops["accuracy"]
    start_time = time.time()
    processed_graphs = 0
    batch_iterator = ggnn_utils.ThreadedIterator(
        self.make_minibatch_iterator(epoch_type), max_queue_size=5)
    for step, batch_data in enumerate(batch_iterator):
      self.global_training_step += 1
      num_graphs = batch_data[self.placeholders["num_graphs"]]
      if num_graphs == 1:
        app.Log(1, "Only 1 graph in batch. This is where it would fail!!!")
      processed_graphs += num_graphs
      fetch_list = [self.ops["loss"], accuracy_op, self.ops["summary_loss"]]
      if epoch_type == "train":
        batch_data[
            self.placeholders["out_layer_dropout_keep_prob"]] = self.params[
                "out_layer_dropout_keep_prob"]
        fetch_list.append(self.ops["train_step"])
      else:
        batch_data[self.placeholders["out_layer_dropout_keep_prob"]] = 1.0

      batch_loss, batch_accuracy, loss_summary, *_ = self.sess.run(
          fetch_list, feed_dict=batch_data)
      loss += batch_loss * num_graphs
      accuracies.append(np.array(batch_accuracy) * num_graphs)

      if self.params["tensorboard_logging"]:
        self.summary_writers[epoch_type].add_summary(loss_summary,
                                                     self.global_training_step)

      print(
          "Running %s, batch %i (has %i graphs). Loss so far: %.4f" %
          (epoch_name, step, num_graphs, loss / processed_graphs),
          end="\r",
      )

    accuracy = np.sum(accuracies, axis=0) / processed_graphs
    loss = loss / processed_graphs
    instance_per_sec = processed_graphs / (time.time() - start_time)
    return loss, accuracy, instance_per_sec

  def train(self):
    log_to_save = []
    total_time_start = time.time()
    with self.graph.as_default():
      if self.args.get("--restore") is not None:
        _, valid_acc, _, _ = self.run_epoch("Resumed (validation)", "val")
        best_val_acc = np.sum(valid_acc)
        best_val_acc_epoch = 0
        print("\r\x1b[KResumed operation, initial cum. val. acc: %.5f" %
              best_val_acc)
        app.Log(1, "Resumed operation, initial cum. val. acc: %.5f",
                best_val_acc)
      else:
        (best_val_acc, best_val_acc_epoch) = (0.0, 0)
      for epoch in range(1, self.params["num_epochs"] + 1):
        print(f"== Epoch {epoch}")

        train_loss, train_acc, train_speed = self.run_epoch(
            "epoch %i (training)" % epoch, "train")
        print("\r\x1b[K Train: loss: %.5f | acc: %s | instances/sec: %.2f" %
              (train_loss, f"{train_acc:.5f}", train_speed))
        app.Log(
            1, "Epoch %s training: loss: %.5f | acc: %s | "
            "instances/sec: %.2f", epoch, train_loss, f"{train_acc:.5f}",
            train_speed)

        valid_loss, valid_acc, valid_speed = self.run_epoch(
            "epoch %i (validation)" % epoch, "val")
        print("\r\x1b[K Valid: loss: %.5f | acc: %s | instances/sec: %.2f" %
              (valid_loss, f"{valid_acc:.5f}", valid_speed))
        app.Log(
            1, "Epoch %d validation: loss: %.5f | acc: %s | "
            "instances/sec: %.2f", epoch, valid_loss, f"{valid_acc:.5f}",
            valid_speed)

        epoch_time = time.time() - total_time_start
        log_entry = {
            "epoch": epoch,
            "time": epoch_time,
            "train_results": (train_loss, train_acc.tolist(), train_speed),
            "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
        }
        log_to_save.append(log_entry)
        with open(self.log_file, "w") as f:
          json.dump(log_to_save, f, indent=4)

        # TODO: sum seems redundant if only one task is trained.
        val_acc = np.sum(valid_acc)  # type: float
        if val_acc > best_val_acc:
          self.save_model(self.best_model_file)
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
          if self.params["test_on_improvement"]:
            test_loss, test_acc, test_speed = self.run_epoch(
                f"epoch {epoch} (training)", "test")
            print("\r\x1b[K Epoch %d Test: loss: %.5f | acc: %s | "
                  "instances/sec: %.2f" %
                  (test_loss, f"{test_acc:.5f}", test_speed))
            app.Log(
                1, "Epoch %d Test: loss: %.5f | acc: %s | "
                "instances/sec: %.2f", test_loss, f"{test_acc:.5f}", test_speed)

            # Add the test results to the logfile.
            log_to_save[-1]["time"] = time.time() - total_time_start
            log_to_save[-1]['test_reults']: (test_loss, test_acc.tolist(),
                                             test_speed)
            with open(self.log_file, "w") as f:
              json.dump(log_to_save, f, indent=4)

        elif epoch - best_val_acc_epoch >= self.params["patience"]:
          app.Log(
              1, "Stopping training after %i epochs without "
              "improvement on validation accuracy", self.params["patience"])
          break

  def test(self):
    log_to_save = []
    total_time_start = time.time()

    with self.graph.as_default():
      valid_loss, valid_acc, valid_speed = self.run_epoch(
          "Test mode (validation)", "val")
      print("\r\x1b[K Validation: loss: %.5f | acc: %s | instances/sec: %.2f" %
            (valid_loss, f"{valid_acc:.5f}", valid_speed))
      app.Log(1, "Validation: loss: %.5f | acc: %s | instances/sec: %.2f",
              valid_loss, f"{valid_acc:.5f}", valid_speed)

      test_loss, test_acc, test_speed = self.run_epoch("Test mode (test)",
                                                       "test")
      print("\r\x1b[K Test: loss: %.5f | acc: %s | instances/sec: %.2f" %
            (test_loss, f"{test_acc:.5f}", test_speed))
      app.Log(1, "Test: loss: %.5f | acc: %s | instances/sec: %.2f", test_loss,
              f"{test_acc:.5f}", test_speed)

      epoch_time = time.time() - total_time_start
      log_entry = {
          "time": epoch_time,
          "valid_results": (valid_loss, valid_acc.tolist(), valid_speed),
          "test_results": (test_loss, test_acc.tolist(), test_speed),
      }
      log_to_save.append(log_entry)
      with open(self.test_log_file, "w") as f:
        json.dump(log_to_save, f, indent=4)

  def save_model(self, path: str) -> None:
    weights_to_save = {}
    for variable in self.sess.graph.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES):
      assert variable.name not in weights_to_save
      weights_to_save[variable.name] = self.sess.run(variable)

    data_to_save = {
        "params": self.params,
        "weights": weights_to_save,
        "global_training_step": self.global_training_step
    }

    with open(path, "wb") as out_file:
      pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

  def initialize_model(self) -> None:
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)

  def restore_model(self, path: str) -> None:
    with prof.Profile(f"read weights from file {path}"):
      with open(path, "rb") as in_file:
        data_to_load = pickle.load(in_file)
        # hotfix test_file
        if "test_file" not in data_to_load["params"]:
          data_to_load["params"]["test_file"] = self.params["test_file"]

    self.global_training_step = data_to_load.get("global_training_step", 0)

    # Assert that we got the same model configuration
    assert len(self.params) == len(data_to_load["params"]), [
        p for p in self.params if p not in data_to_load["params"]
    ]  # (self.params, data_to_load["params"])
    for (par, par_value) in self.params.items():
      # Fine to have different task_ids:
      if par not in ["task_ids", "num_epochs", "num_timesteps"]:
        assert par_value == data_to_load["params"][
            par], f"Failed at {par}: {par_value} vs. data_to_load: {data_to_load['params'][par]}"

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
