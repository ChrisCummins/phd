"""Base class for implementing classifier models."""
import collections
import pathlib
import pickle
import random
import time
import typing

import numpy as np
import sklearn.metrics
from labm8 import app
from labm8 import bazelutil
from labm8 import decorators
from labm8 import humanize
from labm8 import jsonutil
from labm8 import pbutil
from labm8 import ppar
from labm8 import prof
from labm8 import system

import build_info
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_batcher
from deeplearning.ml4pl.models import log_database

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
# global set MODEL_FLAGS here, so that the declaration of model flags is local
# to the declaration of the flag.
MODEL_FLAGS = set()

app.DEFINE_output_path('working_dir',
                       '/tmp/deeplearning/ml4pl/models/',
                       'The directory to write files to.',
                       is_dir=True)

app.DEFINE_database('graph_db',
                    graph_database.Database,
                    None,
                    'The database to read graph data from.',
                    must_exist=True)

app.DEFINE_database('log_db', log_database.Database, None,
                    'The database to write logs to.')

app.DEFINE_integer("num_epochs", 300, "The number of epochs to train for.")

app.DEFINE_integer("random_seed", 42, "A random seed value.")

app.DEFINE_input_path(
    "embedding_path",
    bazelutil.DataPath('phd/deeplearning/ncc/published_results/emb.p'),
    "The path of the embeddings file to use.")

app.DEFINE_string(
    'batch_scores_averaging_method', 'weighted',
    'Selects the averaging method to use when computing recall/precision/F1 '
    'scores. See <https://scikit-learn.org/stable/modules/generated/sklearn'
    '.metrics.f1_score.html>')
MODEL_FLAGS.add("batch_scores_averaging_method")

app.DEFINE_boolean(
    "test_on_improvement", True,
    "If true, test model accuracy on test data when the validation accuracy "
    "improves.")

app.DEFINE_input_path("restore_model", None,
                      "An optional file to restore the model from.")

app.DEFINE_boolean(
    "test_only", False,
    "If this flag is set, only a single pass of the test set is ran.")

#
##### End of flag declarations.

SMALL_NUMBER = 1e-7


class ClassifierBase(object):
  """Abstract base class for implementing classification models."""

  def MakeMinibatchIterator(
      self, epoch_type: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLog, typing.Any]]:
    """Create and return an iterator over mini-batches of data.

    Args:
      epoch_type: The type of epoch to return mini-batches for.

    Returns:
      An iterator of mini-batches and batch logs, where each
      mini-batch will be passed as an argument to RunMinibatch().
    """
    raise NotImplementedError("abstract class")

  MinibatchResults = collections.namedtuple('MinibatchResults',
                                            ['y_true_1hot', 'y_pred_1hot'])

  def RunMinibatch(self, epoch_type: str,
                   batch: typing.Any) -> MinibatchResults:
    raise NotImplementedError("abstract class")

  def GetModelFlagNames(self) -> typing.Iterable[str]:
    """Subclasses may extend this method to mark additional flags as important."""
    return MODEL_FLAGS

  def __init__(self, db: graph_database.Database,
               log_db: log_database.Database):
    """Constructor."""
    self.run_id: str = (f"{time.strftime('%Y%m%dT%H%M%S')}@"
                        f"{system.HOSTNAME}")
    app.Log(1, "Run ID: %s", self.run_id)

    self.batcher = graph_batcher.GraphBatcher(
        db, message_passing_step_count=self.message_passing_step_count)
    self.stats = self.batcher.stats
    app.Log(1, "%s", self.stats)

    self.working_dir = FLAGS.working_dir
    self.best_model_file = self.working_dir / f'{self.run_id}.best_model.pickle'
    self.working_dir.mkdir(exist_ok=True, parents=True)

    # Write app.Log() calls to file. To also log to stderr, use flag
    # --alsologtostderr.
    app.Log(
        1, 'Writing logs to `%s`. Unless --alsologtostderr flag is set, '
        'this is the last message you will see', self.working_dir)
    app.LogToDirectory(self.working_dir, self.run_id)

    self.log_db = log_db
    app.Log(1, 'Writing batch logs to `%s`', self.log_db.url)
    self._CreateExperimentalParameters()

    app.Log(1, "Build information: %s",
            jsonutil.format_json(pbutil.ToJson(build_info.GetBuildInfo())))

    app.Log(1, "Model flags: %s",
            jsonutil.format_json(self._ModelFlagsToDict()))

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    # Progress counters. These are saved and restored from file.
    self.epoch_num = 1
    self.global_training_step = 0
    self.best_epoch_validation_accuracy = 0
    self.best_epoch_num = 0

  @property
  def message_passing_step_count(self) -> int:
    return 0

  @decorators.memoized_property
  def labels_dimensionality(self) -> int:
    return (self.stats.node_labels_dimensionality +
            self.stats.edge_labels_dimensionality +
            self.stats.graph_labels_dimensionality)

  @decorators.memoized_property
  def labels(self):
    return np.arange(self.labels_dimensionality, dtype=np.int32)

  def RunEpoch(self, epoch_type: str) -> float:
    assert epoch_type in {"train", "val", "test"}
    epoch_accuracies = []

    batch_type = typing.Tuple[log_database.BatchLog, typing.Dict[str, typing.
                                                                 Any]]
    batch_iterator: typing.Iterable[batch_type] = ppar.ThreadedIterator(
        self.MakeMinibatchIterator(epoch_type), max_queue_size=5)

    for step, (log, feed_dict) in enumerate(batch_iterator):
      if not log.graph_count:
        raise ValueError("Mini-batch with zero graphs generated")

      batch_start_time = time.time()
      self.global_training_step += 1
      log.epoch = self.epoch_num
      log.batch = step + 1
      log.global_step = self.global_training_step
      log.run_id = self.run_id

      assert log.group in {"train", "val", "test"}
      targets, predictions = self.RunMinibatch(log, feed_dict)

      # Compute statistics.
      y_true = np.argmax(targets, axis=1)
      y_pred = np.argmax(predictions, axis=1)
      accuracies = y_true == y_pred

      log.accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
      log.precision = sklearn.metrics.precision_score(
          y_true,
          y_pred,
          labels=self.labels,
          average=FLAGS.batch_scores_averaging_method)
      log.recall = sklearn.metrics.recall_score(
          y_true,
          y_pred,
          labels=self.labels,
          average=FLAGS.batch_scores_averaging_method)
      log.f1 = sklearn.metrics.f1_score(
          y_true,
          y_pred,
          labels=self.labels,
          average=FLAGS.batch_scores_averaging_method)

      log.accuracy = accuracies.mean()
      log.accuracies = accuracies
      log.predictions = predictions

      epoch_accuracies.append(log.accuracy)

      log.elapsed_time_seconds = time.time() - batch_start_time

      app.Log(1, "%s", log)
      # Create a new database session for every batch because we don't want to
      # leave the connection lying around for a long time (they can drop out)
      # and epochs may take O(hours). Alternatively we could store all of the
      # logs for an epoch in-memory and write them in a single shot, but this
      # might consume a lot of memory (when the predictions arrays are large).
      with self.log_db.Session(commit=True) as session:
        session.add(log)

    return np.mean(epoch_accuracies)

  def Train(self):
    for epoch_num in range(self.epoch_num, FLAGS.num_epochs + 1):
      self.epoch_num = epoch_num
      epoch_start_time = time.time()
      self.RunEpoch("train")
      val_acc = self.RunEpoch("val")
      app.Log(1, "Epoch %s completed in %s. Validation "
              "accuracy: %.2f%%", epoch_num,
              humanize.Duration(time.time() - epoch_start_time), val_acc * 100)

      if val_acc > self.best_epoch_validation_accuracy:
        self.SaveModel(self.best_model_file)
        # Compute the ratio of the new best validation accuracy against the
        # old best validation accuracy.
        if self.best_epoch_validation_accuracy:
          accuracy_ratio = (
              val_acc / max(self.best_epoch_validation_accuracy, SMALL_NUMBER))
          relative_increase = f", (+{accuracy_ratio - 1:.3%} relative)"
        else:
          relative_increase = ''
        app.Log(
            1, "Best epoch so far, validation accuracy increased "
            "+%.3f%%%s. Saving to '%s'",
            (val_acc - self.best_epoch_validation_accuracy) * 100,
            relative_increase, self.best_model_file)
        self.best_epoch_validation_accuracy = val_acc
        self.best_epoch_num = epoch_num

        # Run on test set.
        if FLAGS.test_on_improvement:
          test_acc = self.RunEpoch("test")
          app.Log(1, "Test accuracy at epoch %s: %.3f%%", epoch_num,
                  test_acc * 100)
      elif epoch_num - self.best_epoch_num >= FLAGS.patience:
        app.Log(
            1, "Stopping training after %i epochs without "
            "improvement on validation accuracy", FLAGS.patience)
        break

  def InitializeModel(self) -> None:
    """Initialize a new model state."""
    pass

  def ModelDataToSave(self) -> None:
    return None

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    return None

  def SaveModel(self, path: pathlib.Path) -> None:
    data_to_save = {
        "flags": app.FlagsToDict(json_safe=True),
        "model_flags": self._ModelFlagsToDict(),
        "model_data": self.ModelDataToSave(),
        "build_info": pbutil.ToJson(build_info.GetBuildInfo()),
        "epoch_num": self.epoch_num,
        "global_training_step": self.global_training_step,
        "best_epoch_validation_accuracy": self.best_epoch_validation_accuracy,
        "best_epoch_num": self.best_epoch_num,
    }
    with open(path, "wb") as out_file:
      pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

  def _CreateExperimentalParameters(self):
    """Private helper method to populate parameters table."""

    def ToParams(type, key_value_dict):
      return [
          log_database.Parameter(
              run_id=self.run_id,
              type=type,
              parameter=str(key),
              value=str(value),
          ) for key, value in key_value_dict.items()
      ]

    with self.log_db.Session(commit=True) as session:
      session.add_all(
          ToParams('flags', app.FlagsToDict()) +
          ToParams('modeL_flags', self._ModelFlagsToDict()) +
          ToParams('build_info', pbutil.ToJson(build_info.GetBuildInfo())))

  def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
    for flag, flag_value in flags.items():
      if flag_value != saved_flags[flag]:
        raise EnvironmentError(
            f"Saved flag {flag} value does not match current value:"
            f"'{saved_flags[flag]}' != '{flag_value}'")

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
    self.epoch_num = data_to_load.get('epoch_num', 0)
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
    self.CheckThatModelFlagsAreEquivalent(flags, saved_flags)

    if 'model_data' in data_to_load:
      model_data = data_to_load['model_data']
    elif 'modeL_data' in data_to_load:
      # Workaround for a typo in an earlier version of this script.
      model_data = data_to_load['modeL_data']
    else:
      raise OSError("Model data not found in restore file with keys: "
                    f"`{list(data_to_load.keys())}`")
    self.LoadModelData(model_data)

  def _ModelFlagsToDict(self) -> typing.Dict[str, typing.Any]:
    """Return the flags which are """
    return {
        flag: jsonutil.JsonSerializable(getattr(FLAGS, flag))
        for flag in sorted(set(self.GetModelFlagNames()))
    }

  def _GetEmbeddingsTable(self) -> np.array:
    """Reading embeddings table"""
    with prof.Profile(f"Read embeddings table `{FLAGS.embedding_path}`"):
      with open(FLAGS.embedding_path, 'rb') as f:
        return pickle.load(f)


def Run(model_class):
  graph_db = FLAGS.graph_db()
  log_db = FLAGS.log_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, 'Using working dir %s', working_dir)

  model = model_class(graph_db, log_db)

  # Restore or initialize the model:
  if FLAGS.restore_model:
    with prof.Profile('Restored model'):
      model.LoadModel(FLAGS.restore_model)
  else:
    with prof.Profile('Initialized model'):
      model.InitializeModel()

  if FLAGS.test_only:
    test_acc = model.RunEpoch("test")
    app.Log(1, "Test accuracy %.4f%%", test_acc * 100)
  else:
    model.Train()
