"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import Optional

import build_info
from deeplearning.ml4pl import run_id as run_id_lib
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


app.DEFINE_integer(
  "logger_buffer_size_mb",
  32,
  "The maximum size of the log buffer, in megabytes.",
)
app.DEFINE_integer(
  "logger_buffer_length", 1024, "The maximum length of the log buffer."
)
app.DEFINE_integer(
  "logger_flush_seconds", 10, "The maximum number of seconds between flushes."
)


class Logger(object):
  def __init__(
    self,
    db: log_database.Database,
    max_buffer_size: int,
    max_buffer_length: int,
    max_seconds_since_flush: float,
    log_level: int = 3,
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

  def OnStartRun(self, run_id: run_id_lib.RunId) -> None:
    """Record model parameters."""
    flags = {k.split(".")[-1]: v for k, v in app.FlagsToDict().items()}
    self._writer.AddMany(
      log_database.Parameter.CreateManyFromDict(
        run_id, log_database.ParameterType.FLAG, flags
      )
    )
    self._writer.AddMany(
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
    batch = log_database.Batch.Create(
      run_id=run_id,
      epoch_type=epoch_type,
      epoch_num=epoch_num,
      batch_num=batch_num,
      timer=timer,
      data=data,
      results=results,
      details=log_database.BatchDetails.Create(data=data, results=results),
    )
    self._writer.AddOne(batch)

  def OnEpochEnd(
    self,
    run_id: run_id_lib.RunId,
    epoch_type: epoch.Type,
    epoch_num: epoch.Type,
    results: epoch.Results,
  ):
    pass

  #############################################################################
  # Save and restore checkpoints.
  #############################################################################

  def Save(self, checkpoint: checkpoints.Checkpoint) -> None:
    self.ctx.Log(1, "Save")
    # TODO(github.com/ChrisCummins/ProGraML/issues/24): Port old code:
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

  def Load(
    self, run_id: run_id_lib.RunId, epoch_num: Optional[int] = None
  ) -> Any:
    """Load model data.

    Args:
      run_id: The run ID of the model data to load.
      epoch_num: An optional epoch number to restore model data from. If None,
        the most recent epoch is used.

    Raises:
      LookupError: If no corresponding entry in the checkpoint table exists.
      EnvironmentError: If the flags in the saved model do not match the current
        model flags.
    """
    self.ctx.Log(1, "Load")
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
