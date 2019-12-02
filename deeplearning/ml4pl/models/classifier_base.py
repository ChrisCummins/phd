"""Base class for implementing classifier models."""
import enum
import pickle
import random
import time
from typing import Any
from typing import Iterable

import numpy as np
import sklearn.metrics
import sqlalchemy as sql
import tqdm

import build_info
from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import graph_batcher
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import base_utils as utils
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger
from labm8.py import app
from labm8.py import decorators
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import pbutil
from labm8.py import ppar
from labm8.py import prof
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_database(
  "log_db", log_database.Database, None, "The database to write logs to."
)
app.DEFINE_string(
  "restore_model",
  None,
  "Select a model checkpoint to restore the model state from. The checkpoint "
  "is identified by a run ID and optionally an epoch number, in the format "
  "--restore_model=<run_id>[:<epoch_num>]. If no epoch number is specified, "
  "the most recent epoch is used. Model checkpoints are loaded from "
  "the log database.",
)
app.DEFINE_boolean(
  "test_only",
  False,
  "If this flag is set, only a single pass of the test set is ran.",
)


class ClassifierBase(object):
  """Abstract base class for implementing classification models.

  Subclasses must implement the following methods:
    MakeBatch()
    RunBatch()

  And may optionally wish to implement these additional methods:
    InitializeModel()
    ModelDataToSave()
    LoadModelData()
    CheckThatModelFlagsAreEquivalent()
  """

  def MakeBatch(self, graphs: Iterable[graph_tuple_database.GraphTuple]) -> Any:
    """Create and return an iterator over mini-batches of data.

    Args:
      epoch_type: The type of mini-batches to generate. One of {train,val,test}.
        For some models, different data may be produced for training vs testing.
      groups: The dataset groups to return mini-batches for.

    Returns:
      An iterator of mini-batches and batch logs, where each
      mini-batch will be passed as an argument to RunBatch().
    """
    raise NotImplementedError("abstract class")

  def RunBatch(self, batch: Any) -> logger.MinibatchResults:
    """Process a mini-batch of data using the model.

    Args:
      log: The mini-batch log returned by MakeBatch().
      batch: The batch data returned by MakeBatch().

    Returns:
      The target values for the batch, and the predicted values.
    """
    raise NotImplementedError("abstract class")

  def __init__(self, graph_db: graph_tuple_database.Database):
    """Constructor. Subclasses should call this first."""
    # Set by LoadModel() or InitializeModel().
    self._initialized = False

    self.graph_db = graph_db

    # Progress counters. These are saved and restored from file.
    self.epoch_num = 0
    self.best_epoch_num = 0
    self.global_step = 0

  @decorators.memoized_property
  def labels_dimensionality(self) -> int:
    """Return the dimensionality of the node/graph labels."""
    return (
      self.stats.node_labels_dimensionality
      + self.stats.graph_labels_dimensionality
    )

  @decorators.memoized_property
  def labels(self):
    """Return a dense array of integer label values."""
    return np.arange(self.labels_dimensionality, dtype=np.int32)

  def RunEpoch(
    self, epoch_type: str, groups: typing.Optional[typing.List[str]] = None
  ) -> float:
    """Run the model with the given epoch.

    Args:
      epoch_type: The type of epoch. One of {train,val,test}.
      groups: The dataset groups to use. This is a list of GraphMeta.group
        column values that are loaded. Defaults to [epoch_type].

    Returns:
      The average accuracy of the model over all mini-batches.
    """
    if not self._initialized:
      raise TypeError("RunEpoch() before model initialized")

    if epoch_type not in {"train", "val", "test"}:
      raise TypeError(
        f"Unknown epoch type `{type}`. Expected one of " "{train,val,test}"
      )

    groups = groups or [epoch_type]

    epoch_accuracies = []

    # FANCY PROGRESS BAR
    # TODO(cec): method still not working.
    # epoch_size = self.batcher.GetGraphsInGroupCount(groups)
    epoch_size = 2 ** 30
    if "devmap" in self.graph_db.url:
      epoch_size = 544 if epoch_type == "train" else 68
    elif "alias" in self.graph_db.url:
      epoch_size = 269000 if epoch_type == "train" else 31500
    # TODO(cec): please have a look at the preceding line, i think the method is rotten:
    #  q = s.query(sql.func.count(graph_database.GraphMeta)) \
    # "Object %r is not legal as a SQL literal value" % value
    # sqlalchemy.exc.ArgumentError: Object <class 'deeplearning.ml4pl.graphs.graph_database.GraphMeta'> is not legal as a SQL literal value
    if FLAGS.max_train_per_epoch and epoch_type == "train":
      epoch_size = min(epoch_size, FLAGS.max_train_per_epoch)
    elif FLAGS.max_val_per_epoch and epoch_type == "val":
      epoch_size = min(epoch_size, FLAGS.max_val_per_epoch)
    else:
      # guestimate for test set size on the full dataset
      epoch_size = min(epoch_size, 206000)

    bar = tqdm.tqdm(
      total=epoch_size,
      desc=epoch_type + f" epoch {self.epoch_num}/{FLAGS.epoch_count}",
      unit="graphs",
      position=1,
    )
    loss_sum = acc_sum = prec_sum = rec_sum = 0.0

    # Whether to record per-instance batch logs.
    record_batch_logs = epoch_type in FLAGS.batch_log_types

    batch_type = typing.Tuple[log_database.BatchLogMeta, typing.Any]
    batch_generator: typing.Iterable[batch_type] = ppar.ThreadedIterator(
      self.MakeBatch(epoch_type, groups, print_context=bar.external_write_mode),
      max_queue_size=5,
    )

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

      # at this point we are pretty sure that batch_data has in fact at least one sequence.
      targets, predictions = self.RunBatch(log, batch_data)

      if targets.shape != predictions.shape:
        raise TypeError(
          "Expected model to produce targets with shape "
          f"{targets.shape} but instead received predictions "
          f"with shape {predictions.shape}"
        )

      # Compute statistics.
      y_true = np.argmax(targets, axis=1)
      y_pred = np.argmax(predictions, axis=1)

      if app.GetVerbosity() >= 4:
        app.Log(
          4,
          "Bincount y_true: %s, pred_y: %s",
          np.bincount(y_true),
          np.bincount(y_pred),
          print_context=bar.external_write_mode,
        )

      accuracies = y_true == y_pred

      log.accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
      log.precision = sklearn.metrics.precision_score(
        y_true,
        y_pred,
        labels=self.labels,
        average=FLAGS.batch_scores_averaging_method,
      )
      log.recall = sklearn.metrics.recall_score(
        y_true,
        y_pred,
        labels=self.labels,
        average=FLAGS.batch_scores_averaging_method,
      )
      log.f1 = sklearn.metrics.f1_score(
        y_true,
        y_pred,
        labels=self.labels,
        average=FLAGS.batch_scores_averaging_method,
      )

      # Only create a batch log for test runs.
      if record_batch_logs:
        log.batch_log = log_database.BatchLog()
        log.graph_indices = log._transient_data["graph_indices"]
        log.accuracies = accuracies
        log.predictions = predictions

      epoch_accuracies.append(log.accuracy)

      log.elapsed_time_seconds = time.time() - batch_start_time

      # now only for debugging:
      app.Log(6, "%s", log, print_context=bar.external_write_mode)

      # update epoch-so-far avgs for printing in bar
      loss_sum += log.loss
      acc_sum += log.accuracy
      prec_sum += log.precision
      rec_sum += log.recall
      bar.set_postfix(
        loss=loss_sum / (step + 1),
        acc=acc_sum / (step + 1),
        prec=prec_sum / (step + 1),
        rec=rec_sum / (step + 1),
      )
      bar.update(log.graph_count)

      # Create a new database session for every batch because we don't want to
      # leave the connection lying around for a long time (they can drop out)
      # and epochs may take O(hours). Alternatively we could store all of the
      # logs for an epoch in-memory and write them in a single shot, but this
      # might consume a lot of memory (when the predictions arrays are large).
      with prof.Profile(
        "Wrote log to database",
        print_to=lambda msg: app.Log(
          5, msg, print_context=bar.external_write_mode
        ),
      ):
        with self.log_db.Session(commit=True) as session:
          session.add(log)

    bar.close()
    if not len(epoch_accuracies):
      raise ValueError("Batch generator produced no batches!")

    return np.mean(epoch_accuracies)

  def Train(
    self, epoch_count: int, val_split: str = "val", test_split: str = "test"
  ) -> float:
    """Train and evaluate the model.

    Args:
      num_epoch: The number of epochs to run for training and validation.
      val_split: The name of the dataset group to use for validating model
        performance.
      test_split: The name of the dataset group to use as holdout test data.
        This group is only used during training if the --test_on_improvement
        flag is set, in which case test group performance is evaluated every
        time that validation accuracy improves.

    Returns:
      The best validation accuracy of the model.
    """
    if val_split == test_split:
      raise app.UsageError(
        f"val_split `{val_split}` == test_split `{test_split}`!"
      )
    # We train on everything except the validation and test data.
    train_groups = list(
      set(self.batcher.stats.groups) - {val_split, test_split}
    )

    for epoch_num in tqdm.tqdm(
      range(self.epoch_num, self.epoch_num + epoch_count),
      unit="ep",
      initial=self.epoch_num,
      total=self.epoch_num + epoch_count,
      position=0,
      desc=f"(Grps:{test_split}|{val_split}) {self.run_id}",
    ):
      self.epoch_num = epoch_num + 1
      epoch_start_time = time.time()

      # Switch up the training group order between epochs.
      random.shuffle(train_groups)

      # Train on the training data.
      # [self.RunEpoch("train", train_group) for train_group in train_groups]

      # groups as list supported!
      train_acc = self.RunEpoch("train", train_groups)
      app.Log(
        1,
        "Epoch %s/%s completed in %s. Train " "accuracy: %.2f%%",
        self.epoch_num,
        FLAGS.epoch_count,
        humanize.Duration(time.time() - epoch_start_time),
        train_acc * 100,
      )
      if FLAGS.use_lr_schedule:
        app.Log(
          1,
          "learning_rate_multiple for next epoch is: %s",
          utils.WarmUpAndFinetuneLearningRateSchedule(
            self.epoch_num, FLAGS.epoch_count
          ),
        )

      # Get the current best validation accuracy so that we can compare against.
      previous_best_val_acc = self.best_epoch_validation_accuracy

      # Validate.
      val_acc = self.RunEpoch("val", [val_split])
      app.Log(
        1,
        "Epoch %s completed in %s. Validation "
        "accuracy: %.2f%% (Previous best: %.2f%% @ epoch %s)",
        self.epoch_num,
        humanize.Duration(time.time() - epoch_start_time),
        val_acc * 100,
        previous_best_val_acc * 100,
        self.best_epoch_num,
      )

      # To minimize the size of the log database we only store model checkpoints
      # and detailed batch logs when the validation accuracy improves, and we
      # only store a single checkpoint / set of logs.
      if val_acc > previous_best_val_acc:
        # Compute the ratio of the new best validation accuracy against the
        # old best validation accuracy.
        if previous_best_val_acc:
          accuracy_ratio = val_acc / max(
            previous_best_val_acc, utils.SMALL_NUMBER
          )
          relative_increase = f", (+{accuracy_ratio - 1:.3%} relative)"
        else:
          relative_increase = ""
        app.Log(
          1,
          "Best epoch so far, validation accuracy increased " "+%.3f%%%s",
          (val_acc - previous_best_val_acc) * 100,
          relative_increase,
        )
        self.best_epoch_num = self.epoch_num

        self.SaveModel(validation_accuracy=val_acc)

        # We only store a single model checkpoint, so delete any old ones.
        with self.log_db.Session() as session:
          query = session.query(log_database.ModelCheckpointMeta.id)
          query = query.filter(
            log_database.ModelCheckpointMeta.epoch != self.epoch_num,
            log_database.ModelCheckpointMeta.run_id == self.run_id,
          )
          model_checkpoints_to_delete = [row.id for row in query]
        if model_checkpoints_to_delete:
          app.Log(
            2,
            "Deleting %s",
            humanize.Plural(
              len(model_checkpoints_to_delete), "old model checkpoint"
            ),
          )
          # Cascade delete is broken, we have to first delete the checkpoint
          # data followed by the checkpoint meta entry.
          delete = sql.delete(log_database.ModelCheckpoint)
          delete = delete.where(
            log_database.ModelCheckpoint.id.in_(model_checkpoints_to_delete)
          )
          self.log_db.engine.execute(delete)
          delete = sql.delete(log_database.ModelCheckpointMeta)
          delete = delete.where(
            log_database.ModelCheckpointMeta.id.in_(model_checkpoints_to_delete)
          )
          self.log_db.engine.execute(delete)

        # Run on test set if we haven't already.
        if FLAGS.test_on_improvement:
          test_acc = self.RunEpoch("test", [test_split])
          app.Log(
            1,
            "Test accuracy at epoch %s: %.3f%%",
            self.epoch_num,
            test_acc * 100,
          )

        # Now that we have performed a new validation / test run, we delete the
        # detailed logs of any old val/test runs.
        with self.log_db.Session() as session:
          query = session.query(log_database.BatchLogMeta.id)
          query = query.filter(log_database.BatchLogMeta.run_id == self.run_id)
          query = query.filter(
            log_database.BatchLogMeta.epoch != self.epoch_num
          )
          batch_logs_to_delete = [row.id for row in query]
        if batch_logs_to_delete:
          app.Log(
            2,
            "Deleting %s old detailed batch logs",
            humanize.Commas(len(batch_logs_to_delete)),
          )
          delete = sql.delete(log_database.BatchLog)
          delete = delete.where(
            log_database.BatchLog.id.in_(batch_logs_to_delete)
          )
          self.log_db.engine.execute(delete)
      else:
        # No improvement over the previous epoch, so delete the detailed
        # validation logs.
        with self.log_db.Session() as session:
          query = session.query(log_database.BatchLogMeta.id)
          query = query.filter(log_database.BatchLogMeta.run_id == self.run_id)
          query = query.filter(
            log_database.BatchLogMeta.epoch == self.epoch_num
          )
          batch_logs_to_delete = [row.id for row in query]
        assert batch_logs_to_delete
        app.Log(
          2,
          "Deleting %s detailed batch logs",
          humanize.Commas(len(batch_logs_to_delete)),
        )
        delete = sql.delete(log_database.BatchLog)
        delete = delete.where(
          log_database.BatchLog.id.in_(batch_logs_to_delete)
        )
        self.log_db.engine.execute(delete)

      if self.epoch_num - self.best_epoch_num >= FLAGS.patience:
        app.Log(
          1,
          "Stopping training after %i epochs without "
          "improvement on validation accuracy",
          FLAGS.patience,
        )
        break

    return self.best_epoch_validation_accuracy

  @property
  def best_epoch_validation_accuracy(self) -> float:
    with self.log_db.Session() as session:
      q = session.query(log_database.ModelCheckpointMeta.validation_accuracy)
      q = q.filter(log_database.ModelCheckpointMeta.run_id == self.run_id)
      q = q.order_by(
        log_database.ModelCheckpointMeta.validation_accuracy.desc()
      )
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
      existing = (
        session.query(log_database.ModelCheckpointMeta.id)
        .filter(log_database.ModelCheckpointMeta.run_id == self.run_id)
        .filter(log_database.ModelCheckpointMeta.epoch == self.epoch_num)
        .first()
      )

      # Delete any existing model checkpoint with this state.
      if existing:
        app.Log(2, "Replacing existing model checkpoint")
        delete = sql.delete(log_database.ModelCheckpoint).where(
          log_database.ModelCheckpoint.id == existing.id
        )
        self.log_db.engine.execute(delete)
        delete = sql.delete(log_database.ModelCheckpointMeta).where(
          log_database.ModelCheckpointMeta.id == existing.id
        )
        self.log_db.engine.execute(delete)

      # Add the new checkpoint.
      session.add(
        log_database.ModelCheckpointMeta.Create(
          run_id=self.run_id,
          epoch=self.epoch_num,
          global_step=self.global_training_step,
          validation_accuracy=validation_accuracy,
          data=data_to_save,
        )
      )

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
        sql.orm.joinedload(log_database.ModelCheckpointMeta.model_checkpoint)
      )
      checkpoint: typing.Optional[log_database.ModelCheckpointMeta] = q.first()

      if not checkpoint:
        raise LookupError(
          f"No checkpoint found with run id `{run_id}` at " f"epoch {epoch_num}"
        )

      # Assert that we got the same model configuration.
      # Flag values found in the saved file but not present currently are ignored.
      flags = self.ModelFlagsToDict()
      saved_flags = self.log_db.ModelFlagsToDict(run_id)
      if not saved_flags:
        raise LookupError(
          "Unable to load model flags for run id `{run_id}`, "
          "but found a model checkpoint. This means that your "
          "log database is probably corrupt :-( "
          "sorry aboot that"
        )
      flag_names = set(flags.keys())
      saved_flag_names = set(saved_flags.keys())
      if flag_names != saved_flag_names:
        raise EnvironmentError(
          "Saved flags do not match current flags. "
          f"Flags not found in saved flags: {flag_names - saved_flag_names}."
          f"Saved flags not present now: {saved_flag_names - flag_names}"
        )
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
        )
        for key, value in key_value_dict.items()
      ]

    with self.log_db.Session(commit=True) as session:
      session.add_all(
        ToParams(log_database.ParameterType.FLAG, app.FlagsToDict())
        + ToParams(
          log_database.ParameterType.MODEL_FLAG, self.ModelFlagsToDict()
        )
        + ToParams(
          log_database.ParameterType.BUILD_INFO,
          pbutil.ToJson(build_info.GetBuildInfo()),
        )
      )

  def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
    flags = dict(flags)  # shallow copy
    flags.pop("restore_model", None)
    for flag, flag_value in flags.items():
      if flag_value != saved_flags[flag]:
        raise EnvironmentError(
          f"Saved flag '{flag}' value does not match current value:"
          f"'{saved_flags[flag]}' != '{flag_value}'"
        )

  def ModelFlagsToDict(self) -> typing.Dict[str, typing.Any]:
    """Return the flags which describe the model."""
    model_flags = {
      flag: getattr(FLAGS, flag)
      for flag in sorted(set(self.GetModelFlagNames()))
    }
    model_flags["model"] = type(self).__name__
    return model_flags


def Run(model_class):
  if FLAGS.graph_db:
    graph_db = FLAGS.graph_db()
  else:
    raise app.UsageError("--graph_db must be set")
  log_db = FLAGS.log_db()
  working_dir = FLAGS.working_dir
  if not working_dir:
    raise app.UsageError("--working_dir is required")

  app.Log(1, "Using working dir %s", working_dir)

  model = model_class(graph_db, log_db)

  # Restore or initialize the model:
  if FLAGS.restore_model:
    if ":" in FLAGS.restore_model:
      # Restore from a specific epoch number:
      try:
        run_id, epoch_num = FLAGS.restore_model.split(":")
        epoch_num = int(epoch_num)
      except Exception as e:
        raise app.UsageError(
          f"Invalid --restore_model=`{FLAGS.restore_model}`. "
          "Must be in the form <run_id>[:<epoch_num>]."
        )
    else:
      # No epoch num specified, so use the most recent checkpoint.
      run_id = FLAGS.restore_model
      with log_db.Session() as session:
        query = session.query(log_database.ModelCheckpointMeta.epoch)
        query = query.filter(log_database.ModelCheckpointMeta.run_id == run_id)
        query = query.order_by(
          log_database.ModelCheckpointMeta.date_added.desc()
        )
        result = query.first()
      if not result:
        raise app.UsageError(
          f"No checkpoints found for model {FLAGS.restore_model}"
        )
      epoch_num = result.epoch
    with prof.Profile(f"Restored run {run_id} at epoch {epoch_num}"):
      model.LoadModel(run_id=run_id, epoch_num=epoch_num)
  else:
    with prof.Profile("Initialized model"):
      model.InitializeModel()

  if FLAGS.test_only:
    test_acc = model.RunEpoch("test", [FLAGS.test_split])
    app.Log(1, "Test accuracy %.4f%%", test_acc * 100)
  else:
    model.Train(
      epoch_count=FLAGS.epoch_count,
      val_split=FLAGS.val_split,
      test_split=FLAGS.test_split,
    )
