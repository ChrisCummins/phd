"""Base class for implementing classifier models."""
import enum
import pickle
import random
import time
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set

import numpy as np
import sklearn.metrics
import sqlalchemy as sql
import tqdm

import build_info
from deeplearning.ml4pl import run_id
from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import base_utils as utils
from deeplearning.ml4pl.models import batch as batchs
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import logger as logging
from labm8.py import app
from labm8.py import decorators
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import pbutil
from labm8.py import ppar
from labm8.py import prof
from labm8.py import progress


FLAGS = app.FLAGS


class ClassifierBase(object):
  """Abstract base class for implementing classification models.

  Subclasses must implement the following methods:
    MakeBatch()
    RunBatch()

  And may optionally wish to implement these additional methods:
    Initialize()
    ModelDataToSave()
    LoadModelData()
    CheckThatModelFlagsAreEquivalent()
  """

  def __init__(self, graph_db: graph_tuple_database.Database):
    """Constructor."""
    # Set by LoadModel() or Initialize().
    self._initialized = False

    self.run_id: run_id.RunId = run_id.RunId.GenerateUnique(type(self).__name__)

    self.graph_db = graph_db

    # Progress counters. These are saved and restored from file.
    self.epoch_num = 0
    self.best_val_results = epoch.Results.NullResults()
    self.global_step = 0

  def Initialize(self) -> None:
    """Initialize a new model state."""
    self._initialized = True

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Data:
    """Create a mini-batch of data from the input graphs.

    Returns:
      TODO.
    """
    graph_ids = []
    while len(graph_ids) < 10:
      try:
        graph = next(graphs)
      except StopIteration:
        break
      graph_ids.append(graph.id)
    return batchs.Data(graph_ids=graph_ids, data=None)
    # TODO: raise NotImplementedError("abstract class")

  def RunBatch(
    self,
    batch: batchs.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batchs.Results:
    """Process a mini-batch of data using the model.

    Args:
      log: The mini-batch log returned by MakeBatch().
      batch: The batch data returned by MakeBatch().

    Returns:
      The target values for the batch, and the predicted values.
    """
    return batchs.Results.NullResults()
    # TODO: raise NotImplementedError("abstract class")

  def __call__(
    self,
    epoch_type: epoch.Type,
    graph_reader: graph_database_reader.BufferedGraphReader,
    logger: logging.Logger,
  ) -> epoch.Results:
    """Run the model for the given epoch type.

    Args:
      epoch_type: The type of epoch to run.
      graph_reader: An input iterable of graphs to process.

    Returns:
      The average accuracy of the model over all batches.
    """
    if not self._initialized:
      raise TypeError("RunEpoch() before model initialized")

    thread = Epoch(self, epoch_type, graph_reader, logger)
    progress.Run(thread)
    return thread.results

  def ModelDataToSave(self) -> None:
    return None

  def LoadModelData(self, data_to_load: Any) -> None:
    # TODO: Figure out how to restore self.epoch_num, self.best_val_results,
    # and self.global_step = 0
    return None


#   def SaveModel(self, validation_accuracy: float) -> None:
#     # Compute the data to save first.
#     data_to_save = self.ModelDataToSave()
#
#     with self.log_db.Session(commit=True) as session:
#       # Check for an existing model with this state.
#       existing = (
#         session.query(log_database.ModelCheckpointMeta.id)
#         .filter(log_database.ModelCheckpointMeta.run_id == self.run_id)
#         .filter(log_database.ModelCheckpointMeta.epoch == self.epoch_num)
#         .first()
#       )
#
#       # Delete any existing model checkpoint with this state.
#       if existing:
#         app.Log(2, "Replacing existing model checkpoint")
#         delete = sql.delete(log_database.ModelCheckpoint).where(
#           log_database.ModelCheckpoint.id == existing.id
#         )
#         self.log_db.engine.execute(delete)
#         delete = sql.delete(log_database.ModelCheckpointMeta).where(
#           log_database.ModelCheckpointMeta.id == existing.id
#         )
#         self.log_db.engine.execute(delete)
#
#       # Add the new checkpoint.
#       session.add(
#         log_database.ModelCheckpointMeta.Create(
#           run_id=self.run_id,
#           epoch=self.epoch_num,
#           global_step=self.global_training_step,
#           validation_accuracy=validation_accuracy,
#           data=data_to_save,
#         )
#       )
#
#   def LoadModel(self, run_id: str, epoch_num: int) -> None:
#     """Load and restore the model from the given model file.
#
#     Args:
#       run_id: The run ID of the model to checkpoint to restore.
#       epoch_num: The epoch number of the checkpoint to restore.
#
#     Raises:
#       LookupError: If no corresponding entry in the checkpoint table exists.
#       EnvironmentError: If the flags in the saved model do not match the current
#         model flags.
#     """
#     with self.log_db.Session() as session:
#       # Fetch the corresponding checkpoint from the database.
#       q = session.query(log_database.ModelCheckpointMeta)
#       q = q.filter(log_database.ModelCheckpointMeta.run_id == run_id)
#       q = q.filter(log_database.ModelCheckpointMeta.epoch == epoch_num)
#       q = q.options(
#         sql.orm.joinedload(log_database.ModelCheckpointMeta.model_checkpoint)
#       )
#       checkpoint: typing.Optional[log_database.ModelCheckpointMeta] = q.first()
#
#       if not checkpoint:
#         raise LookupError(
#           f"No checkpoint found with run id `{run_id}` at " f"epoch {epoch_num}"
#         )
#
#       # Assert that we got the same model configuration.
#       # Flag values found in the saved file but not present currently are ignored.
#       flags = self.ModelFlagsToDict()
#       saved_flags = self.log_db.ModelFlagsToDict(run_id)
#       if not saved_flags:
#         raise LookupError(
#           "Unable to load model flags for run id `{run_id}`, "
#           "but found a model checkpoint. This means that your "
#           "log database is probably corrupt :-( "
#           "sorry aboot that"
#         )
#       flag_names = set(flags.keys())
#       saved_flag_names = set(saved_flags.keys())
#       if flag_names != saved_flag_names:
#         raise EnvironmentError(
#           "Saved flags do not match current flags. "
#           f"Flags not found in saved flags: {flag_names - saved_flag_names}."
#           f"Saved flags not present now: {saved_flag_names - flag_names}"
#         )
#       self.CheckThatModelFlagsAreEquivalent(flags, saved_flags)
#
#       # Restore state from checkpoint.
#       self.epoch_num = checkpoint.epoch
#       # We assume that the model we are loading has a higher validation accuracy
#       # than current. Since best_epoch_num is used only for computing epoch
#       # patience, I think this is okay.
#       self.best_epoch_num = checkpoint.epoch
#       self.global_training_step = checkpoint.global_step
#       self.LoadModelData(checkpoint.data)
#
#     self._initialized = True
#
#   def _CreateExperimentalParameters(self):
#     """Private helper method to populate parameters table."""
#
#     def ToParams(type_: log_database.ParameterType, key_value_dict):
#       return [
#         log_database.Parameter(
#           run_id=self.run_id,
#           type=type_,
#           parameter=str(key),
#           pickled_value=pickle.dumps(value),
#         )
#         for key, value in key_value_dict.items()
#       ]
#
#     with self.log_db.Session(commit=True) as session:
#       session.add_all(
#         ToParams(log_database.ParameterType.FLAG, app.FlagsToDict())
#         + ToParams(
#           log_database.ParameterType.MODEL_FLAG, self.ModelFlagsToDict()
#         )
#         + ToParams(
#           log_database.ParameterType.BUILD_INFO,
#           pbutil.ToJson(build_info.GetBuildInfo()),
#         )
#       )
#
#   def CheckThatModelFlagsAreEquivalent(self, flags, saved_flags) -> None:
#     flags = dict(flags)  # shallow copy
#     flags.pop("restore_model", None)
#     for flag, flag_value in flags.items():
#       if flag_value != saved_flags[flag]:
#         raise EnvironmentError(
#           f"Saved flag '{flag}' value does not match current value:"
#           f"'{saved_flags[flag]}' != '{flag_value}'"
#         )
#
#   def ModelFlagsToDict(self) -> typing.Dict[str, typing.Any]:
#     """Return the flags which describe the model."""
#     model_flags = {
#       flag: getattr(FLAGS, flag)
#       for flag in sorted(set(self.GetModelFlagNames()))
#     }
#     model_flags["model"] = type(self).__name__
#     return model_flags


class Epoch(progress.Progress):
  def __init__(
    self,
    model: ClassifierBase,
    epoch_type: epoch.Type,
    graph_reader: graph_database_reader.BufferedGraphReader,
    logger: logging.Logger,
  ):
    self.model = model
    self.epoch_type: epoch_type.Type
    self.graph_reader = graph_reader
    self.logger = logger

    # Set at the end of Run().
    self.results: epoch.Results = epoch.Results.NullResults()

    super(Epoch, self).__init__(
      f"{epoch_type.name.capitalize()} epoch {model.epoch_num}",
      0,
      graph_reader.n,
      unit="graph",
      vertical_position=0,
      leave=False,
    )

  def Run(self):
    """Run the epoch."""
    batch = self.model.MakeBatch(self.graph_reader)
    self.ctx.i += len(batch.graph_ids)
    if not batch.graph_ids:
      raise OSError("No batches generated!")

    batch_count = 0
    while batch.graph_ids:
      batch_count += 1
      batch_results = self.model.RunBatch(batch)
      batch = self.model.MakeBatch(self.graph_reader)
      self.ctx.i += len(batch.graph_ids)

    self.results = epoch.Results(
      accuracy=random.random(), precision=0, recall=0, batch_count=batch_count,
    )


#     loss_sum = acc_sum = prec_sum = rec_sum = 0.0
#
#     # Whether to record per-instance batch logs.
#     record_batch_logs = epoch_type in FLAGS.batch_log_types
#
#     batch_type = typing.Tuple[log_database.BatchLogMeta, typing.Any]
#     batch_generator: typing.Iterable[batch_type] = ppar.ThreadedIterator(
#       self.MakeBatch(epoch_type, groups, print_context=bar.external_write_mode),
#       max_queue_size=5,
#     )
#
#     for step, (log, batch_data) in enumerate(batch_generator):
#       if not log.graph_count:
#         raise ValueError("Mini-batch with zero graphs generated")
#
#       batch_start_time = time.time()
#       self.global_training_step += 1
#       log.type = epoch_type
#       log.epoch = self.epoch_num
#       log.batch = step + 1
#       log.global_step = self.global_training_step
#       log.run_id = self.run_id
#
#       # at this point we are pretty sure that batch_data has in fact at least one sequence.
#       targets, predictions = self.RunBatch(log, batch_data)
#
#       if targets.shape != predictions.shape:
#         raise TypeError(
#           "Expected model to produce targets with shape "
#           f"{targets.shape} but instead received predictions "
#           f"with shape {predictions.shape}"
#         )
#
#       # Compute statistics.
#       y_true = np.argmax(targets, axis=1)
#       y_pred = np.argmax(predictions, axis=1)
#
#       if app.GetVerbosity() >= 4:
#         app.Log(
#           4,
#           "Bincount y_true: %s, pred_y: %s",
#           np.bincount(y_true),
#           np.bincount(y_pred),
#           print_context=bar.external_write_mode,
#         )
#
#       accuracies = y_true == y_pred
#
#       log.accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
#       log.precision = sklearn.metrics.precision_score(
#         y_true,
#         y_pred,
#         labels=self.labels,
#         average=FLAGS.batch_scores_averaging_method,
#       )
#       log.recall = sklearn.metrics.recall_score(
#         y_true,
#         y_pred,
#         labels=self.labels,
#         average=FLAGS.batch_scores_averaging_method,
#       )
#       log.f1 = sklearn.metrics.f1_score(
#         y_true,
#         y_pred,
#         labels=self.labels,
#         average=FLAGS.batch_scores_averaging_method,
#       )
#
#       # Only create a batch log for test runs.
#       if record_batch_logs:
#         log.batch_log = log_database.BatchLog()
#         log.graph_indices = log._transient_data["graph_indices"]
#         log.accuracies = accuracies
#         log.predictions = predictions
#
#       batch_accuracies.append(log.accuracy)
#
#       log.elapsed_time_seconds = time.time() - batch_start_time
#
#       # now only for debugging:
#       app.Log(6, "%s", log, print_context=bar.external_write_mode)
#
#       # update epoch-so-far avgs for printing in bar
#       loss_sum += log.loss
#       acc_sum += log.accuracy
#       prec_sum += log.precision
#       rec_sum += log.recall
#       bar.set_postfix(
#         loss=loss_sum / (step + 1),
#         acc=acc_sum / (step + 1),
#         prec=prec_sum / (step + 1),
#         rec=rec_sum / (step + 1),
#       )
#       bar.update(log.graph_count)
#
#       # Create a new database session for every batch because we don't want to
#       # leave the connection lying around for a long time (they can drop out)
#       # and epochs may take O(hours). Alternatively we could store all of the
#       # logs for an epoch in-memory and write them in a single shot, but this
#       # might consume a lot of memory (when the predictions arrays are large).
#       with prof.Profile(
#         "Wrote log to database",
#         print_to=lambda msg: app.Log(
#           5, msg, print_context=bar.external_write_mode
#         ),
#       ):
#         with self.log_db.Session(commit=True) as session:
#           session.add(log)
#
#     bar.close()
#     if not len(batch_accuracies):
#       raise ValueError("Batch generator produced no batches!")
#
#     return np.mean(batch_accuracies)
#
