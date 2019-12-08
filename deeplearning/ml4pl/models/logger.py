"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Optional

import sqlalchemy as sql

import build_info
from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import schedules
from labm8.py import app
from labm8.py import pbutil
from labm8.py import prof
from labm8.py import progress
from labm8.py import sqlutil


FLAGS = app.FLAGS

app.DEFINE_enum(
  "keep_checkpoints",
  schedules.KeepCheckpoints,
  schedules.KeepCheckpoints.ALL,
  "How many checkpoints to keep.",
)
app.DEFINE_enum(
  "keep_detailed_batches",
  schedules.KeepDetailedBatches,
  schedules.KeepDetailedBatches.ALL,
  "The type of detailed batches to keep.",
)
app.DEFINE_string(
  "logger_tag",
  "",
  "An arbitrary tag which will be stored as a lag in the parameters table. Use "
  "this to group multiple runs of a model with a meaningful name, e.g. for "
  "grouping the 'k' run IDs of a k-fold dataset.",
)
app.DEFINE_integer(
  "logger_buffer_size_mb",
  32,
  "Tuning parameter. The maximum size of the log buffer, in megabytes.",
)
app.DEFINE_integer(
  "logger_buffer_length",
  1024,
  "Tuning parameter. The maximum length of the log buffer.",
)
app.DEFINE_integer(
  "logger_flush_seconds",
  10,
  "Tuning parameter. The maximum number of seconds between flushes.",
)
app.DEFINE_boolean(
  "fail_on_logger_error", True, "Raise an error if log writing fails."
)


class Logger(object):
  """An database-backed logger with asynchronous writes.

  This class exposes callbacks for recording logging events during the execution
  of a model.
  """

  def __init__(
    self,
    db: log_database.Database,
    max_buffer_size: Optional[int] = None,
    max_buffer_length: Optional[int] = None,
    max_seconds_since_flush: Optional[float] = None,
    log_level: int = 2,
    ctx: progress.ProgressContext = progress.NullContext,
  ):
    self.db = db
    self.ctx = ctx
    self._writer = sqlutil.BufferedDatabaseWriter(
      db,
      ctx=ctx,
      max_buffer_size=max_buffer_size,
      max_buffer_length=max_buffer_length,
      max_seconds_since_flush=max_seconds_since_flush,
      log_level=log_level,
    )

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    del exc_type
    del exc_val
    del exc_tb
    self._writer.Close()
    self.CheckForError()

  def CheckForError(self) -> None:
    """Check for errors in log writing.

    Raises:
      OSError: If log writing has failed.
    """
    if self._writer.error_count and FLAGS.fail_on_logger_error:
      raise OSError(
        f"Stopping now because since the last time I checked there have been "
        f"been {self._writer.error_count} log writing failures. "
        "Disable these checks using --fail_on_logger_error=false"
      )

  #############################################################################
  # Event callbacks.
  #############################################################################

  def OnStartRun(
    self, run_id: run_id_lib.RunId, graph_db: graph_tuple_database.Database
  ) -> None:
    """Register the creation of a new run ID.

    This records the experimental parameters of a run.
    """
    # Record the run ID and experimental parameters.
    flags = {k.split(".")[-1]: v for k, v in app.FlagsToDict().items()}
    self._writer.AddMany(
      # Record run ID.
      [log_database.RunId(run_id=run_id)]
      +
      # Record flag values.
      log_database.Parameter.CreateManyFromDict(
        run_id, log_database.ParameterType.FLAG, flags
      )
      +
      # Record graph database stats.
      log_database.Parameter.CreateManyFromDict(
        run_id,
        log_database.ParameterType.INPUT_GRAPHS_INFO,
        graph_db.stats_json,
      )
      +
      # Record build info.
      log_database.Parameter.CreateManyFromDict(
        run_id,
        log_database.ParameterType.BUILD_INFO,
        pbutil.ToJson(build_info.GetBuildInfo()),
      )
    )

  def OnBatchEnd(
    self,
    run_id: run_id_lib.RunId,
    epoch_type: epoch.Type,
    epoch_num: int,
    batch_num: int,
    timer: prof.ProfileTimer,
    data: batch.Data,
    results: batch.Results,
  ):
    self._writer.AddOne(
      log_database.Batch.Create(
        run_id=run_id,
        epoch_type=epoch_type,
        epoch_num=epoch_num,
        batch_num=batch_num,
        timer=timer,
        data=data,
        results=results,
        details=log_database.BatchDetails.Create(data=data, results=results),
      )
    )

  def OnEpochEnd(
    self,
    run_id: run_id_lib.RunId,
    epoch_type: epoch.Type,
    epoch_num: epoch.Type,
    results: epoch.Results,
  ):
    del epoch_type
    del results

    schedule = FLAGS.keep_detailed_batches()

    if schedule == schedules.KeepDetailedBatches.NONE:
      pass
    elif schedule == schedules.KeepDetailedBatches.ALL:
      pass
    elif schedule == schedules.KeepDetailedBatches.LAST_EPOCH:

      def DeleteOldDetailedBatchLogs(session):
        """Delete old detailed batch logs."""
        detailed_batches_to_delete = [
          row.id
          for row in session.query(log_database.Batch.id).filter(
            log_database.Batch.run_id == run_id,
            log_database.Batch.epoch_num != epoch_num,
          )
        ]
        if detailed_batches_to_delete:
          session.query(log_database.BatchDetails).filter(
            log_database.BatchDetails.id.in_(detailed_batches_to_delete)
          ).delete(synchronize_session=False)
          self.ctx.Log(
            2,
            "Deleted %s old batch log details",
            len(detailed_batches_to_delete),
          )

      self._writer.AddLambdaOp(DeleteOldDetailedBatchLogs)
      self.CheckForError()

  #############################################################################
  # Save and restore checkpoints.
  #############################################################################

  def Save(self, checkpoint: checkpoints.Checkpoint) -> None:
    """Save a checkpoint.

    Args:
      checkpoint: A model checkpoint, as generated by model.GetCheckpoint().
    Returns:
      pass
    """
    keep_schedule = FLAGS.keep_checkpoints()

    checkpoint = log_database.Checkpoint.Create(checkpoint)

    # Delete old checkpoints if required.
    if keep_schedule == schedules.KeepCheckpoints.ALL:
      pass
    elif keep_schedule == schedules.KeepCheckpoints.LAST:
      self._writer.AddLambdaOp(
        lambda session: session.query(log_database.Checkpoint)
        .filter(log_database.Checkpoint.run_id == checkpoint.run_id)
        .delete()
      )
    else:
      raise NotImplementedError("unreachable")

    self._writer.AddOne(checkpoint)

  def Load(
    self, checkpoint_ref: checkpoints.CheckpointReference
  ) -> checkpoints.Checkpoint:
    """Load model data.

    Args:
      run_id: The run ID of the model data to load.
      epoch_num: An optional epoch number to restore model data from. If None,
        the most recent epoch is used.

    Returns:
      A checkpoint instance.

    Raises:
      ValueError: If no corresponding entry in the checkpoint table exists.
    """
    # A previous Save() call from this logger might still be buffered. Flush the
    # buffer before loading from the database.
    self._writer.Flush()

    with self.db.Session() as session:
      checkpoint_entry = (
        session.query(log_database.Checkpoint)
        .filter(
          log_database.Checkpoint.run_id == checkpoint_ref.run_id,
          log_database.Checkpoint.epoch_num == checkpoint_ref.epoch_num,
        )
        .options(sql.orm.joinedload(log_database.Checkpoint.data))
        .first()
      )
      # Check that the requested checkpoint exists.
      if not checkpoint_entry:
        raise ValueError(f"Checkpoint not found: {checkpoint_ref}")

      checkpoint = checkpoints.Checkpoint(
        run_id=run_id_lib.RunId.FromString(checkpoint_entry.run_id),
        epoch_num=checkpoint_entry.epoch_num,
        best_results=self.db.GetBestResults(
          run_id=checkpoint_ref.run_id, session=session
        ),
        model_data=checkpoint_entry.model_data,
      )

    return checkpoint

  @classmethod
  def FromFlags(cls, ctx: progress.ProgressContext = progress.NullContext):
    if not FLAGS.log_db:
      raise app.UsageError("--log_db not set")

    return cls(
      FLAGS.log_db(),
      ctx=ctx,
      max_buffer_size=FLAGS.logger_buffer_size_mb * 1024 * 1024,
      max_buffer_length=FLAGS.logger_buffer_length,
      max_seconds_since_flush=FLAGS.logger_flush_seconds,
    )
