"""Database backend for model logs."""
import codecs
import datetime
import enum
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import sqlalchemy as sql
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import humanize
from labm8.py import labdate
from labm8.py import pdutil
from labm8.py import prof
from labm8.py import sqlutil

FLAGS = app.FLAGS
# Note that log_db flag is declared at the bottom of this file, after Database
# class is defined.

Base = declarative.declarative_base()


###############################################################################
# Parameters.
###############################################################################


class ParameterType(enum.Enum):
  """The parameter type."""

  FLAG = 1
  BUILD_INFO = 2


class Parameter(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description of an experimental parameter."""

  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = run_id_lib.RunId.SqlStringColumn(default=None)

  type: ParameterType = sql.Column(
    sql.Enum(ParameterType), nullable=False, index=True
  )

  # The name of the parameter.
  name: str = sql.Column(sql.String(1024), nullable=False)
  # The value for the parameter.
  binary_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  @property
  def value(self) -> Any:
    return pickle.loads(self.binary_value)

  @value.setter
  def value(self, data: Any) -> None:
    self.binary_value = pickle.dumps(data)

  __table_args__ = (
    sql.UniqueConstraint("run_id", "type", "name", name="unique_parameter"),
  )

  @classmethod
  def Create(cls, run_id: run_id_lib.RunId, type: str, name: str, value: Any):
    return cls(
      run_id=run_id,
      type=type,
      name=str(name),
      binary_value=pickle.dumps(value),
    )

  @classmethod
  def CreateManyFromDict(
    cls,
    run_id: run_id_lib.RunId,
    type: ParameterType,
    parameters: Dict[str, Any],
  ):
    return [
      Parameter.Create(run_id=run_id, type=type, name=name, value=value,)
      for name, value in parameters.items()
    ]


###############################################################################
# Batches.
###############################################################################


class Batch(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description running a batch of graphs through a model."""

  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = run_id_lib.RunId.SqlStringColumn(default=None)

  # The epoch number, >= 1.
  epoch_num: int = sql.Column(sql.Integer, nullable=False, index=True)

  epoch_type: int = sql.Column(sql.Enum(epoch.Type), nullable=False, index=True)

  # The batch number within the epoch, in range [1, epoch_batch_count].
  batch_num: int = sql.Column(sql.Integer, nullable=False)

  # The elapsed time of the batch.
  elapsed_time_ms: int = sql.Column(sql.Integer, nullable=False)

  # The number of graphs in the batch.
  graph_count: int = sql.Column(sql.Integer, nullable=False)

  # Batch-level average performance metrics.
  iteration_count: int = sql.Column(sql.Integer, nullable=False)
  model_converged: bool = sql.Column(sql.Boolean, nullable=False)
  loss: float = sql.Column(sql.Float, nullable=True)
  accuracy: float = sql.Column(sql.Float, nullable=False)
  precision: float = sql.Column(sql.Float, nullable=False)
  recall: float = sql.Column(sql.Float, nullable=False)
  f1: float = sql.Column(sql.Float, nullable=False)

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # Create the one-to-one relationship from batch to details.
  details: "BatchDetails" = sql.orm.relationship(
    "BatchDetails", uselist=False, cascade="all, delete-orphan"
  )

  # Unique batch results.
  __table_args__ = (
    sql.UniqueConstraint(
      "run_id", "epoch_num", "epoch_type", "batch_num", name="unique_batch"
    ),
  )

  @property
  def elapsed_time(self) -> float:
    return self.elapsed_time_ms * 1000

  @property
  def graphs_per_second(self) -> float:
    if self.elapsed_time:
      return self.graph_count / self.elapsed_time
    else:
      return 0

  @property
  def nodes_per_second(self) -> float:
    if self.elapsed_time:
      return self.graph_count / self.elapsed_time
    else:
      return 0

  @classmethod
  def Create(
    cls,
    run_id: run_id_lib.RunId,
    epoch_type: epoch.Type,
    epoch_num: int,
    batch_num: int,
    timer: prof.ProfileTimer,
    data: batches.Data,
    results: batches.Results,
    details: Optional["BatchDetails"] = None,
  ):
    return cls(
      run_id=str(run_id),
      epoch_type=epoch_type,
      epoch_num=epoch_num,
      batch_num=batch_num,
      elapsed_time_ms=timer.elapsed_ms,
      graph_count=data.graph_count,
      iteration_count=results.iteration_count,
      model_converged=results.model_converged,
      loss=results.loss,
      accuracy=results.accuracy,
      precision=results.precision,
      recall=results.recall,
      f1=results.f1,
      timestamp=timer.start,
      details=details,
    )

  #############################################################################
  # Batch details.
  # Convenience properties to set and get columns from the joined BathDetails
  # table. If accessing many of these properties, consider using
  # sql.orm.joinedload(Batch.details) to eagerly load the joined table when
  # querying batches.
  #############################################################################

  @property
  def has_details(self) -> bool:
    return self.details is not None

  @property
  def graph_ids(self) -> List[int]:
    return pickle.loads(self.details.binary_graph_ids)

  @property
  def true_y(self) -> Any:
    return pickle.loads(codecs.decode(self.details.binary_true_y, "zlib"))

  @property
  def predictions(self) -> Any:
    return pickle.loads(self.details.binary_predictions)


class BatchDetails(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """The per-instance results of a batch."""

  id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey("batches.id", onupdate="CASCADE", ondelete="CASCADE"),
    primary_key=True,
  )

  # A pickled array of
  # deeplearning.ml4pl.graphs.labelled.graph_tuple_database.GraphTuple.id
  # values.
  binary_graph_ids: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  # A pickled array of labels, of shape (target_count), dtype int32. The number
  # of targets per instance will depend on the type of classification problem.
  # For graph-level classification, there is graph_count values.
  # For node-level classification, there are
  # sum(graph.node_count for graph in batch.data) targets.
  binary_true_y: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  # A pickled array of 1-hot model predictions, of shape
  # (target_count, y_dimensionality), dtype float32. See binary_true_y for a
  # description of target_count.
  binary_predictions: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  @classmethod
  def Create(cls, data: batches.Data, results: batches.Results):
    return cls(
      binary_graph_ids=pickle.dumps(data.graph_ids),
      binary_true_y=codecs.encode(
        pickle.dumps(np.argmax(results.targets, axis=1)), "zlib"
      ),
      binary_predictions=pickle.dumps(results.predictions),
    )


###############################################################################
# Epochs.
###############################################################################


###############################################################################
# Checkpoints.
###############################################################################


class Checkpoint(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = run_id_lib.RunId.SqlStringColumn(default=None, index=True)

  # The epoch number, >= 1.
  epoch_num: int = sql.Column(sql.Integer, nullable=False, index=True)

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  data: "CheckpointModelData" = sql.orm.relationship(
    "CheckpointModelData", uselist=False, cascade="all, delete-orphan",
  )

  # Unique checkpoint.
  __table_args__ = (
    sql.UniqueConstraint("run_id", "epoch_num", name="unique_checkpoint"),
  )

  @property
  def model_data(self) -> Any:
    # Checkpoints are stored with zlib compression.
    return pickle.loads(codecs.decode(self.data.binary_model_data, "zlib"))

  @classmethod
  def Create(
    cls, checkpoint: checkpoints.Checkpoint,
  ):
    """Instantiate a model checkpoint. Use this convenience method rather than
    constructing objects directly to ensure that fields are encoded correctly.
    """
    # Note that best_results is not stored, as we can re-compute it from the
    # batch logs table.
    return Checkpoint(
      run_id=str(checkpoint.run_id),
      epoch_num=checkpoint.epoch_num,
      data=CheckpointModelData(
        binary_data=codecs.encode(pickle.dumps(checkpoint.model_data), "zlib")
      ),
    )


class CheckpointModelData(
  Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin
):
  """The sister table of Checkpoint, which stores the actual data of a model,
  as returned by classifier_base.GetModelData().
  """

  id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey("checkpoints.id", onupdate="CASCADE", ondelete="CASCADE"),
    primary_key=True,
  )

  binary_data: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )


###############################################################################
# Database.
###############################################################################


class Database(sqlutil.Database):
  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)

  def BatchLogsToDataFrame(self, run_id: str, per_global_step: bool = False):
    """Return a table of log stats for the given run_id.

    Args:
      run_id: The run ID to return the logs for.
      per_global_step: If true, return the raw stats for each global step. Else,
        aggregate stats across each epoch.

    Returns:
      A pandas dataframe.
    """
    with self.Session() as session:
      q = session.query(
        BatchLogMeta.epoch,
        BatchLogMeta.type,
        sql.func.count(BatchLogMeta.epoch).label("num_batches"),
        sql.func.min(BatchLogMeta.date_added).label("timestamp"),
        sql.func.min(BatchLogMeta.global_step).label("global_step"),
        sql.func.avg(BatchLogMeta.loss).label("loss"),
        sql.func.avg(BatchLogMeta.iteration_count).label("iteration_count"),
        sql.func.avg(BatchLogMeta.model_converged).label("converged"),
        sql.func.avg(BatchLogMeta.accuracy * 100).label("accuracy"),
        sql.func.avg(BatchLogMeta.precision).label("precision"),
        sql.func.avg(BatchLogMeta.recall).label("recall"),
        sql.func.avg(BatchLogMeta.f1).label("f1"),
        sql.func.sum(BatchLogMeta.elapsed_time_seconds).label(
          "elapsed_time_seconds"
        ),
        sql.sql.expression.cast(
          sql.func.sum(BatchLogMeta.graph_count), sql.Integer
        ).label("graph_count"),
        sql.sql.expression.cast(
          sql.func.sum(BatchLogMeta.node_count), sql.Integer
        ).label("node_count"),
      )

      q = q.filter(BatchLogMeta.run_id == run_id)

      q = q.group_by(BatchLogMeta.epoch, BatchLogMeta.type)

      # Group each individual step. Since there is only one log per step,
      # this means return all rows without grouping.
      if per_global_step:
        q = q.group_by(BatchLogMeta.global_step).order_by(
          BatchLogMeta.global_step
        )

      q = q.order_by(BatchLogMeta.epoch, BatchLogMeta.type)

      df = pdutil.QueryToDataFrame(session, q)

      df["reltime"] = [t - df["timestamp"][0] for t in df["timestamp"]]

      return df

  # TODO(github.com/ChrisCummins/ProGraML/issues/14): Add a
  # DeleteLogsForRunIds() method which accepts a list of run ids, allowing us
  # to perform a single delete query for all.
  def DeleteLogsForRunId(self, run_id: str) -> None:
    """Delete the logs for this run.

    This deletes the batch logs, model checkpoints, and model parameters.

    Args:
      run_id: The ID of the run to delete.
    """
    # Because the cascaded delete is broken, we first delete the BatchLog child
    # rows, then the parents.
    with self.Session() as session:
      query = session.query(BatchLogMeta.id).filter(
        BatchLogMeta.run_id == run_id
      )
      ids_to_delete = [row.id for row in query]

    if ids_to_delete:
      app.Log(
        1,
        "Deleting %s batch logs for run %s",
        humanize.Commas(len(ids_to_delete)),
        run_id,
      )
      delete = sql.delete(BatchLog).where(BatchLog.id.in_(ids_to_delete))
      self.engine.execute(delete)

      delete = sql.delete(BatchLogMeta).where(BatchLogMeta.run_id == run_id)
      self.engine.execute(delete)

    # Because the cascaded delete is broken, we first delete the checkpoint
    # child rows, then the parents.
    with self.Session() as session:
      query = session.query(ModelCheckpointMeta.id).filter(
        ModelCheckpointMeta.run_id == run_id
      )
      ids_to_delete = [row.id for row in query]

    if ids_to_delete:
      app.Log(
        1,
        "Deleting %s model checkpoints for run %s",
        humanize.Commas(len(ids_to_delete)),
        run_id,
      )
      delete = sql.delete(ModelCheckpoint).where(
        ModelCheckpoint.id.in_(ids_to_delete)
      )
      self.engine.execute(delete)

      delete = sql.delete(ModelCheckpointMeta).where(
        ModelCheckpointMeta.run_id == run_id
      )
      self.engine.execute(delete)

    # Delete the parameters for this Run ID.
    app.Log(1, "Deleting model parameters for run %s", run_id)
    delete = sql.delete(Parameter).where(Parameter.run_id == run_id)
    self.engine.execute(delete)

  def ModelFlagsToDict(self, run_id: str):
    """Load the model flags for the given run ID to a {flag: value} dict."""
    with self.Session() as session:
      q = session.query(Parameter)
      q = q.filter(Parameter.run_id == run_id)
      q = q.filter(Parameter.type == ParameterType.MODEL_FLAG)
      return {param.parameter: param.value for param in q.all()}

  def ParametersToDataFrame(self, run_id: str, parameter_type: str):
    """Return a table of parameters of the given type for the specified run.

    Args:
      run_id: The run ID to return the parameters of.
      parameter_type: The type of parameter to return.

    Returns:
      A pandas dataframe.
    """
    with self.Session() as session:
      query = session.query(
        Parameter.parameter, Parameter.binary_value.label("value")
      )
      query = query.filter(Parameter.run_id == run_id)
      query = query.filter(sql.func.lower(Parameter.type) == parameter_type)
      query = query.order_by(Parameter.parameter)
      df = pdutil.QueryToDataFrame(session, query)
      # Strip the prefix, 'foo.bar' -> 'foo':
      pdutil.RewriteColumn(df, "parameter", lambda x: x.split(".")[-1])
      # Un-pickle the parameter values:
      pdutil.RewriteColumn(df, "value", lambda x: pickle.loads(x))
      return df.set_index("parameter")

  @property
  def run_ids(self) -> List[str]:
    """Get a list of all run IDs in the databse."""
    with self.Session() as session:
      query = session.query(Parameter.run_id.distinct().label("run_id"))
      return [row.run_id for row in query]


app.DEFINE_database(
  "log_db", Database, None, "The database to write model logs to."
)
