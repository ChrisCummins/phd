"""Database backend for model logs."""
import codecs
import datetime
import enum
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import pdutil
from labm8.py import prof
from labm8.py import sqlutil

FLAGS = app.FLAGS
# Note that log_db flag is declared at the bottom of this file, after Database
# class is defined.

Base = declarative.declarative_base()

###############################################################################
# Run IDs.
###############################################################################


class RunId(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A run ID."""

  run_id: str = sql.Column(
    run_id_lib.RunId.SqlStringColumnType(),
    default=None,
    primary_key=True,
    nullable=False,
  )

  # Relationships to data.
  parameters: "Parameter" = sql.orm.relationship(
    "Parameter",
    back_populates="run_id_relationship",
    cascade="all, delete-orphan",
  )
  batches: "Batch" = sql.orm.relationship(
    "Batch", back_populates="run_id_relationship", cascade="all, delete-orphan"
  )
  checkpoints: "Checkpoint" = sql.orm.relationship(
    "Checkpoint",
    back_populates="run_id_relationship",
    cascade="all, delete-orphan",
  )

  def __repr__(self):
    return str(self.run_id)

  def __eq__(self, rhs: "RunId"):
    return self.run_id == rhs.run_id


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

  # The run ID.
  run_id: str = sql.Column(
    run_id_lib.RunId.SqlStringColumnType(),
    sql.ForeignKey("run_ids.run_id", onupdate="CASCADE", ondelete="CASCADE"),
    default=None,
    index=True,
    nullable=False,
  )
  run_id_relationship: RunId = sql.orm.relationship(
    "RunId", back_populates="parameters", uselist=False
  )

  # The numeric value of the ParameterType num. Use type property to access enum
  # value.
  type_num: int = sql.Column(sql.Integer, nullable=False, index=True)

  # The name of the parameter.
  name: str = sql.Column(sql.String(128), nullable=False, index=True)

  # The value for the parameter.
  binary_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  @property
  def type(self) -> ParameterType:
    return ParameterType(self.type_num)

  @type.setter
  def type(self, value: ParameterType) -> None:
    self.type_num = value.value

  @property
  def value(self) -> Any:
    return pickle.loads(self.binary_value)

  @value.setter
  def value(self, data: Any) -> None:
    self.binary_value = pickle.dumps(data)

  __table_args__ = (
    sql.UniqueConstraint("run_id", "type_num", "name", name="unique_parameter"),
  )

  @classmethod
  def Create(
    cls, run_id: run_id_lib.RunId, type: ParameterType, name: str, value: Any
  ):
    return cls(
      run_id=str(run_id),
      type_num=type.value,
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


# ###############################################################################
# # Batches.
# ###############################################################################
#
#
class Batch(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description running a batch of graphs through a model."""

  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: int = sql.Column(
    run_id_lib.RunId.SqlStringColumnType(),
    sql.ForeignKey("run_ids.run_id", onupdate="CASCADE", ondelete="CASCADE"),
    default=None,
    index=True,
    nullable=False,
  )
  run_id_relationship: RunId = sql.orm.relationship(
    "RunId", back_populates="batches", uselist=False,
  )

  # The epoch number, >= 1.
  epoch_num: int = sql.Column(sql.Integer, nullable=False, index=True)

  # The numeric value of the epoch type, use the epoch_type property to access
  # the enum value.
  epoch_type_num: int = sql.Column(sql.Integer, nullable=False, index=True)

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
      "run_id", "epoch_num", "epoch_type_num", "batch_num", name="unique_batch",
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

  @property
  def epoch_type(self) -> epoch.Type:
    return epoch.Type(self.epoch_type_num)

  @epoch_type.setter
  def epoch_type(self, value: epoch.Type) -> None:
    self.epoch_type_num = value.value

  @classmethod
  def Create(
    cls,
    run_id: RunId,
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
      epoch_type_num=epoch_type.value,
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
  # Convenience properties to get columns from the joined BathDetails table.
  # If accessing many of these properties, consider using
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
# Checkpoints.
###############################################################################


class Checkpoint(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey("run_ids.run_id", onupdate="CASCADE", ondelete="CASCADE"),
    default=None,
    index=True,
    nullable=False,
  )
  run_id_relationship: RunId = sql.orm.relationship(
    "RunId", back_populates="checkpoints", uselist=False,
  )

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


class CheckpointModelData(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
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
# RunLogs collection.
###############################################################################


class RunLogs(NamedTuple):
  """All of the logs for a single run."""

  run_id: RunId
  parameters: List[Parameter]
  batches: List[Batch]
  checkpoints: List[Checkpoint]

  @property
  def all(self) -> List[Union[Parameter, Batch, Checkpoint]]:
    """Return all mapped database entries."""
    return [self.run_id] + self.parameters + self.batches + self.checkpoints


###############################################################################
# Database.
###############################################################################


# A registry of database statics, where each entry is a <name, property> tuple.
database_statistics_registry: List[Tuple[str, Callable[["Database"], Any]]] = []


def database_statistic(func):
  """A decorator to mark a method on a Database as a database static.

  Database statistics can be accessed using Database.stats_json property to
  retrieve a <name, vale> dictionary.
  """
  global database_statistics_registry
  database_statistics_registry.append((func.__name__, func))
  return property(func)


class Database(sqlutil.Database):
  """A database of model logs."""

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)

  @database_statistic
  def run_ids(self):
    with self.Session() as session:
      return [
        run_id_lib.RunId.FromString(row.run_id)
        for row in session.query(
          sql.func.distinct(Parameter.run_id).label("run_id")
        )
      ]

  def GetRunLogs(
    self,
    run_id=run_id_lib.RunId,
    eager_batch_details: bool = True,
    eager_checkpoint_data: bool = True,
    session: Optional[sqlutil.Database.SessionType] = None,
  ):
    with self.Session(session=session) as session:
      parameters = session.query(Parameter).filter(Parameter.run_id == run_id)
      batches = session.query(Batch).filter(Batch.run_id == run_id)
      if eager_batch_details:
        batches = batches.options(sql.orm.joinedload(Batch.details))

      checkpoints = (
        session.query(Checkpoint).filter(Checkpoint.run_id == run_id).all()
      )
      if eager_checkpoint_data:
        checkpoints = checkpoints.options(sql.orm.joinedload(Checkpoint.data))

      return RunLogs(
        run_id=run_id,
        parameters=parameters.all(),
        batches=batches.all(),
        checkpoints=checkpoints.all(),
      )

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


app.DEFINE_database(
  "log_db", Database, None, "The database to write model logs to."
)


def Main():
  """Main entry point."""
  log_db = FLAGS.log_db()
  print(jsonutil.format_json(log_db.stats_json))


if __name__ == "__main__":
  app.Run(Main)
