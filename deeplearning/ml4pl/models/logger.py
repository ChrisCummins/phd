"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import enum
from typing import Optional

import sqlalchemy as sql

import build_info
from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import batch
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import pbutil
from labm8.py import prof
from labm8.py import progress
from labm8.py import sqlutil


FLAGS = app.FLAGS


class KeepCheckpoints(enum.Enum):
  NONE = 0
  ALL = 1
  LAST_EPOCH = 2


class KeepDetailedBatches(enum.Enum):
  NONE = 0
  ALL = 1
  LAST_EPOCH = 2


app.DEFINE_enum(
  "keep_checkpoints",
  KeepCheckpoints,
  KeepCheckpoints.ALL,
  "The type of checkpoints to keep.",
)
app.DEFINE_enum(
  "keep_detailed_batches",
  KeepDetailedBatches,
  KeepDetailedBatches.ALL,
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
    option = FLAGS.keep_detailed_batches()

    if option == KeepDetailedBatches.NONE:
      pass
    elif option == KeepDetailedBatches.ALL:
      pass
    elif option == KeepDetailedBatches.LAST_EPOCH:

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

  #############################################################################
  # Save and restore checkpoints.
  #############################################################################

  def Save(self, checkpoint: checkpoints.Checkpoint) -> None:
    option = FLAGS.keep_checkpoints()

    if option == KeepCheckpoints.NONE:
      pass
    elif option == KeepCheckpoints.ALL:
      self._writer.AddOne(log_database.Checkpoint.Create(checkpoint))
    elif option == KeepCheckpoints.LAST_EPOCH:
      self._writer.AddLambdaOp(
        lambda session: session.query(log_database.Checkpoint)
        .filter(log_database.Checkpoint.run_id == checkpoint.run_id)
        .delete()
      )
      self._writer.AddOne(log_database.Checkpoint.Create(checkpoint))
    else:
      raise NotImplementedError("unreachable")

  def Load(
    self, run_id: run_id_lib.RunId, epoch_num: Optional[int] = None
  ) -> checkpoints.Checkpoint:
    """Load model data.

    Args:
      run_id: The run ID of the model data to load.
      epoch_num: An optional epoch number to restore model data from. If None,
        the most recent epoch is used.

    Returns:
      A checkpoint instance.

    Raises:
      KeyError: If no corresponding entry in the checkpoint table exists.
    """
    # The results of previous Save() call might still be buffered. Flush the
    # buffer before loading from the database.
    self._writer.Flush()

    # Check that the requested run ID exists.
    with self.db.Session() as session:
      checkpoint = (
        session.query(log_database.Checkpoint)
        .filter(
          log_database.Checkpoint.run_id == run_id,
          log_database.Checkpoint.epoch_num == epoch_num,
        )
        .options(sql.orm.joinedload(log_database.Checkpoint.data))
        .first()
      )

      if not checkpoint:
        raise KeyError(
          f"No checkpoint exists for '"
          f"{checkpoints.RunIdAndEpochNumToString(run_id, epoch_num)}'"
        )

      return checkpoint.ToCheckpoint(session)

    # TODO(github.com/ChrisCummins/ProGraML/issues/24): Port old code:
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
