"""Base class for implementing classifier models."""
import pickle
import random
import time
import typing

import numpy as np
import sklearn.metrics
import sqlalchemy as sql
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
from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
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
    bazelutil.DataPath('phd/deeplearning/ml4pl/graphs/unlabelled/cdfg/'
                       'node_embeddings/inst2vec_augmented_embeddings.pickle'),
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

app.DEFINE_integer(
    "batch_size", 15000,
    "The maximum number of nodes to include in each graph batch.")

app.DEFINE_integer(
    'max_train_per_epoch', None,
    'Use this flag to limit the maximum number of instances used in a single '
    'training epoch. For k-fold cross-validation, each of the k folds will '
    'train on a maximum of this many graphs.')

app.DEFINE_integer(
    'max_val_per_epoch', None,
    'Use this flag to limit the maximum number of instances used in a single '
    'validation epoch.')

app.DEFINE_input_path("restore_model", None,
                      "An optional file to restore the model from.")

app.DEFINE_boolean(
    "test_only", False,
    "If this flag is set, only a single pass of the test set is ran.")

app.DEFINE_string(
    "val_group", "val",
    "The name of the group to be used for validating model performance. All "
    "groups except --val_group and --test_group will be used for training.")

app.DEFINE_string(
    "test_group", "test",
    "The name of the hold-out group to be used for testing. All groups "
    "except --val_group and --test_group will be used for training.")

app.DEFINE_integer(
    "patience", 300,
    "The number of epochs to train for without any improvement in validation "
    "accuracy before stopping.")

#
##### End of flag declarations.

SMALL_NUMBER = 1e-7


class ClassifierBase(object):
  """Abstract base class for implementing classification models.

  Subclasses must implement the following methods:
    MakeMinibatchIterator()
    RunMinibatch()

  And may optionally wish to implement these additional methods:
    InitializeModel()
    ModelDataToSave()
    LoadModelData()
    CheckThatModelFlagsAreEquivalent()
  """

  def MakeMinibatchIterator(
      self, epoch_type: str, group: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Create and return an iterator over mini-batches of data.

    Args:
      epoch_type: The type of mini-batches to generate. One of {train,val,test}.
        For some models, different data may be produced for training vs testing.
      group: The dataset group to return mini-batches for.

    Returns:
      An iterator of mini-batches and batch logs, where each
      mini-batch will be passed as an argument to RunMinibatch().
    """
    raise NotImplementedError("abstract class")

  # The result of running a minibatch. Return 1-hot target values and the raw
  # 1-hot outputs of the model. These are used to compute evaluation metrics.
  class MinibatchResults(typing.NamedTuple):
    y_true_1hot: np.array  # Shape [num_labels,num_classes]
    y_pred_1hot: np.array  # Shape [num_labels,num_classes]

  def RunMinibatch(self, log: log_database.BatchLogMeta,
                   batch: typing.Any) -> MinibatchResults:
    """Process a mini-batch of data using the model.

    Args:
      log: The mini-batch log returned by MakeMinibatchIterator().
      batch: The batch data returned by MakeMinibatchIterator().

    Returns:
      The target values for the batch, and the predicted values.
    """
    raise NotImplementedError("abstract class")

  def GetModelFlagNames(self) -> typing.Iterable[str]:
    """Return the 'model flags', a subset of all flags which are used to
    describe the model architecture. These flags must be consistent across runs
    of the same model. Subclasses may extend this method to mark additional
    flags as important.
    """
    return MODEL_FLAGS

  def __init__(self, db: graph_database.Database,
               log_db: log_database.Database):
    """Constructor. Subclasses should call this first."""
    self._initialized = False  # Set by LoadModel() or InitializeModel()
    self.run_id: str = (f"{time.strftime('%Y%m%dT%H%M%S')}@"
                        f"{system.HOSTNAME}")
    app.Log(1, "Run ID: %s", self.run_id)

    self.batcher = graph_batcher.GraphBatcher(db)
    self.stats = self.batcher.stats

    self.working_dir = FLAGS.working_dir
    self.working_dir.mkdir(exist_ok=True, parents=True)

    # Write app.Log() calls to file.
    FLAGS.alsologtostderr = True
    app.Log(1, 'Writing logs to `%s`', self.working_dir)
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
    self.best_epoch_num = 0

  @decorators.memoized_property
  def labels_dimensionality(self) -> int:
    """Return the dimensionality of the node/graph labels."""
    return (self.stats.node_labels_dimensionality +
            self.stats.graph_labels_dimensionality)

  @decorators.memoized_property
  def labels(self):
    """Return a dense array of integer label values."""
    return np.arange(self.labels_dimensionality, dtype=np.int32)

  def RunEpoch(self, epoch_type: str,
               group: typing.Optional[str] = None) -> float:
    """Run the model with the given epoch.

    Args:
      epoch_type: The type of epoch. One of {train,val,test}.
      group: The dataset group to use. This is the GraphMeta.group column that
        is loaded. Defaults to `epoch_type`.

    Returns:
      The average accuracy of the model over all mini-batches.
    """
    if not self._initialized:
      raise TypeError("RunEpoch() before model initialized")

    if epoch_type not in {"train", "val", "test"}:
      raise TypeError(f"Unknown epoch type `{type}`. Expected one of "
                      "{train,val,test}")
    group = group or epoch_type

    epoch_accuracies = []

    batch_type = typing.Tuple[log_database.BatchLogMeta, typing.Any]
    batch_generator: typing.Iterable[batch_type] = ppar.ThreadedIterator(
        self.MakeMinibatchIterator(epoch_type, group), max_queue_size=5)

    for step, (log, batch_data) in enumerate(batch_generator):
      if not log.graph_count:
        raise ValueError("Mini-batch with zero graphs generated")

      batch_start_time = time.time()
      self.global_training_step += 1
      log.type = epoch_type
      log.epoch = self.epoch_num
      log.batch = step + 1
      log.global_step = self.global_training_step
      log.run_id = self.run_id

      targets, predictions = self.RunMinibatch(log, batch_data)

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

    if not epoch_accuracies:
      raise ValueError("Batch generator produced no batches!")

    return np.mean(epoch_accuracies)

  def Train(self,
            num_epochs: int,
            val_group: str = "val",
            test_group: str = "test") -> float:
    """Train and evaluate the model.

    Args:
      num_epoch: The number of epochs to run for training and validation.
      val_group: The name of the dataset group to use for validating model
        performance.
      test_group: The name of the dataset group to use as holdout test data.
        This group is only used during training if the --test_on_improvement
        flag is set, in which case test group performance is evaluated every
        time that validation accuracy improves.

    Returns:
      The best validation accuracy of the model.
    """
    # We train on everything except the validation and test data.
    train_groups = set(self.batcher.stats.groups) - {val_group, test_group}

    for epoch_num in range(self.epoch_num, num_epochs + 1):
      self.epoch_num = epoch_num
      epoch_start_time = time.time()

      # Switch up the training group order between epochs.
      random.shuffle(train_groups)

      # Train on the training data.
      [self.RunEpoch("train", train_group) for train_group in train_groups]

      # Validate.
      val_acc = self.RunEpoch("val", val_group)
      app.Log(1, "Epoch %s completed in %s. Validation "
              "accuracy: %.2f%%", epoch_num,
              humanize.Duration(time.time() - epoch_start_time), val_acc * 100)

      # Get the current best validation accurcy now before saving the model,
      # as that may become the new best.
      previous_best_val_acc = self.best_epoch_validation_accuracy
      self.SaveModel(validation_accuracy=val_acc)

      # Save the model when validation accuracy improves.
      if val_acc > previous_best_val_acc:
        # Compute the ratio of the new best validation accuracy against the
        # old best validation accuracy.
        if previous_best_val_acc:
          accuracy_ratio = (
              val_acc / max(self.best_epoch_validation_accuracy, SMALL_NUMBER))
          relative_increase = f", (+{accuracy_ratio - 1:.3%} relative)"
        else:
          relative_increase = ''
        app.Log(1, "Best epoch so far, validation accuracy increased "
                "+%.3f%%%s",
                (val_acc - self.best_epoch_validation_accuracy) * 100,
                relative_increase)
        self.best_epoch_num = epoch_num

        # Run on test set if we haven't already.
        if FLAGS.test_on_improvement:
          test_acc = self.RunEpoch("test", test_group)
          app.Log(1, "Test accuracy at epoch %s: %.3f%%", epoch_num,
                  test_acc * 100)
      elif epoch_num - self.best_epoch_num >= FLAGS.patience:
        app.Log(
            1, "Stopping training after %i epochs without "
            "improvement on validation accuracy", FLAGS.patience)
        break

    return self.best_epoch_validation_accuracy

  @property
  def best_epoch_validation_accuracy(self) -> float:
    with self.log_db.Session() as session:
      q = session.query(log_database.ModelCheckpointMeta.validation_accuracy)
      q = q.filter(log_database.ModelCheckpointMeta.run_id == self.run_id)
      q = q.order_by(
          log_database.ModelCheckpointMeta.validation_accuracy.desc())
      q = q.limit(1)
      best = q.first()
      if best:
        return best.validation_accuracy
      else:
        return 0

  def InitializeModel(self) -> None:
    """Initialize a new model state."""
    self._initialized = True

  def ModelDataToSave(self) -> None:
    return None

  def LoadModelData(self, data_to_load: typing.Any) -> None:
    return None

  def SaveModel(self, validation_accuracy: float) -> None:
    # Compute the data to save first.
    data_to_save = self.ModelDataToSave()

    with self.log_db.Session(commit=True) as session:
      # Check for an existing model with this state.
      existing = session.query(log_database.ModelCheckpointMeta.id) \
        .filter(log_database.ModelCheckpointMeta.run_id == self.run_id) \
        .filter(log_database.ModelCheckpointMeta.epoch == self.epoch_num) \
        .first()

      # Delete any existing model checkpoint with this state.
      if existing:
        app.Log(1, "Replacing existing model checkpoint")
        delete = sql.delete(log_database.ModelCheckpointMeta) \
          .where(log_database.ModelCheckpointMeta.id == existing.id)
        self.log_db.connection.execute(delete)

      # Add the new checkpoint.
      session.add(
          log_database.ModelCheckpointMeta(
              run_id=self.run_id,
              epoch=self.epoch_num,
              global_step=self.global_training_step,
              validation_accuracy=validation_accuracy,
              model_checkpoint=log_database.ModelCheckpoint(
                  pickled_data=pickle.dumps(data_to_save))))

  def LoadModel(self, run_id: str, epoch_num: int) -> None:
    """Load and restore the model from the given model file.

    Args:
      run_id: The run ID of the model to checkpoint to restore.
      epoch_num: The epoch number of the checkpoint to restore.

    Raises:
      LookupError: If no corresponding entry in the checkpoint table exists.
      EnvironmentError: If the flags in the saved model do not match the current
        model flags.
    """
    with self.log_db.Session() as session:
      # Fetch the corresponding checkpoint from the database.
      q = session.query(log_database.ModelCheckpointMeta)
      q = q.filter(log_database.ModelCheckpointMeta.run_id == run_id)
      q = q.filter(log_database.ModelCheckpointMeta.epoch == epoch_num)
      q = q.options(
          sql.orm.joinedload(log_database.ModelCheckpointMeta.model_checkpoint))
      checkpoint: typing.Optional[log_database.ModelCheckpointMeta] = q.first()

      if not checkpoint:
        raise LookupError(f"No checkpoint found with run id `{run_id}` at "
                          f"epoch {epoch_num}")

      # Assert that we got the same model configuration.
      # Flag values found in the saved file but not present currently are ignored.
      flags = self._ModelFlagsToDict()
      saved_flags = self.log_db.ModelFlagsToDict(run_id)
      if not saved_flags:
        raise LookupError("Unable to load model flags for run id `{run_id}`, "
                          "but found a model checkpoint. This means that your "
                          "log database is probably corrupt :-( "
                          "sorry aboot that")
      flag_names = set(flags.keys())
      saved_flag_names = set(saved_flags.keys())
      if flag_names != saved_flag_names:
        raise EnvironmentError(
            "Saved flags do not match current flags. "
            f"Flags not found in saved flags: {flag_names - saved_flag_names}."
            f"Saved flags not present now: {saved_flag_names - flag_names}")
      self.CheckThatModelFlagsAreEquivalent(flags, saved_flags)

      # Restore state from checkpoint.
      self.epoch_num = checkpoint.epoch
      # We assume that the model we are loading has a higher validation accuracy
      # than current. Since best_epoch_num is used only for computing epoch
      # patience, I think this is okay.
      self.best_epoch_num = checkpoint.epoch
      self.global_training_step = checkpoint.global_step
      self.LoadModelData(checkpoint.data)

    self._initialized = True

  def _CreateExperimentalParameters(self):
    """Private helper method to populate parameters table."""

    def ToParams(type_: log_database.ParameterType, key_value_dict):
      return [
          log_database.Parameter(
              run_id=self.run_id,
              type=type_,
              parameter=str(key),
              pickled_value=pickle.dumps(value),
          ) for key, value in key_value_dict.items()
      ]

    with self.log_db.Session(commit=True) as session:
      session.add_all(
          ToParams(log_database.ParameterType.FLAG, app.FlagsToDict()) +
          ToParams(log_database.ParameterType.MODEL_FLAG,
                   self._ModelFlagsToDict()) +
          ToParams(log_database.ParameterType.BUILD_INFO,
                   pbutil.ToJson(build_info.GetBuildInfo())))

  def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
    for flag, flag_value in flags.items():
      if flag_value != saved_flags[flag]:
        raise EnvironmentError(
            f"Saved flag {flag} value does not match current value:"
            f"'{saved_flags[flag]}' != '{flag_value}'")

  def _ModelFlagsToDict(self) -> typing.Dict[str, typing.Any]:
    """Return the flags which describe the model."""
    model_flags = {
        flag: getattr(FLAGS, flag)
        for flag in sorted(set(self.GetModelFlagNames()))
    }
    # TODO(cec): Set database URL for input graphs.
    return model_flags


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
    model.Train(num_epochs=FLAGS.num_epochs,
                val_group=FLAGS.val_group,
                test_group=FLAGS.test_group)
