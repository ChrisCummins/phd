"""This module provides a database backend for storing model logs."""
import codecs
import datetime
import enum
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import checkpoints
from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import crypto
from labm8.py import humanize
from labm8.py import jsonutil
from labm8.py import pdutil
from labm8.py import prof
from labm8.py import sqlutil


FLAGS = app.FLAGS
# Note that log_db flag is declared at the bottom of this file, after Database
# class is defined.
app.DEFINE_boolean(
  "prune_logs",
  False,
  "When //deeplearning/ml4pl/models:log_database is executed as a script, "
  "using this flag will prune any runs that do not have a checkpoint.",
)
app.DEFINE_list(
  "rm",
  [],
  "When //deeplearning/ml4pl/models:log_database is executed as a script, "
  "pass a list of run IDs to this argument to delete them.",
)
app.DEFINE_list(
  "rm_tag",
  [],
  "When //deeplearning/ml4pl/models:log_database is executed as a script, "
  "this specifies a list of --tag strings to remove the runs for.",
)

Base = declarative.declarative_base()

###############################################################################
# Run IDs.
###############################################################################


class RunId(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A run ID. This single-column table enables one-to-many foreign key
  relationships to {parameters,batches,checkpoints}.

  Deleting a run ID then cascades to all other tables.

  Similarly, adding a batch / parameter / checkpoint that does not have a
  corresponding RunId entry will raise an integrity error.
  """

  run_id: str = sql.Column(
    run_id_lib.RunId.SqlStringColumnType(),
    default=None,
    primary_key=True,
    nullable=False,
  )

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  # Relationships to logs.
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
  """The type of a parameter."""

  # Parameters that are the values of command line flags. E.g. a parameter of
  # this type with value 'foo' is the value of the '--foo' flag.
  FLAG = 1
  # Parameters that are statistics describing the graph database that was used
  # to train/val/test a model.
  INPUT_GRAPHS_INFO = 2
  # Build information such as the current repo commit, hostname, etc.
  BUILD_INFO = 3


class Parameter(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """An experimental parameter. These describe the environment and configuration
  options used to run a model.

  Don't instantiate these objects directly, use Parameter.Create().
  """

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

  # The value for the parameter. These are stored as pickled bytes to preserve
  # the type.
  binary_value: bytes = sql.Column(
    sqlutil.ColumnTypes.LargeBinary(), nullable=False
  )
  # The sha1sum of the 'binary_value' column. Use this for querying and
  # grouping by value.
  binary_value_sha1: str = sql.Column(
    sql.String(40), nullable=False, index=True
  )

  timestamp: datetime.datetime = sqlutil.ColumnFactory.MillisecondDatetime()

  @property
  def type(self) -> ParameterType:
    """Returns the ParameterType enum value."""
    return ParameterType(self.type_num)

  @property
  def value(self) -> Any:
    """Returns the pickled value."""
    return pickle.loads(self.binary_value)

  __table_args__ = (
    sql.UniqueConstraint("run_id", "type_num", "name", name="unique_parameter"),
  )

  @classmethod
  def Create(
    cls, run_id: run_id_lib.RunId, type: ParameterType, name: str, value: Any
  ):
    """Construct an experimental parameter.

    Args:
      run_id: The run ID.
      type: The parameter type.
      name: The name of the parameter.
      value: The value to of the parameter.

    Returns:
      A Parameter instance.
    """
    binary_value = pickle.dumps(value)
    return cls(
      run_id=str(run_id),
      type_num=type.value,
      name=str(name),
      binary_value=binary_value,
      binary_value_sha1=crypto.sha1(binary_value),
    )

  @classmethod
  def CreateManyFromDict(
    cls,
    run_id: run_id_lib.RunId,
    type: ParameterType,
    parameters: Dict[str, Any],
  ):
    """Construct a list of parameters from a <name,value> dictionary.

    Args:
      run_id: The run ID.
      type: The parameter type.
      parameters: A <name, value> dictionary.

    Returns:
      A list of parameters.
    """
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

  # The number of prediction targets in the batch. For graph-level inference,
  # this is equal to graph_count. For node-level inference, this is the sum
  # node_count for all graphs in the batch.
  target_count: int = sql.Column(sql.Integer, nullable=False)

  # Batch-level average performance metrics.
  iteration_count: int = sql.Column(sql.Integer, nullable=False)
  model_converged: bool = sql.Column(sql.Boolean, nullable=False)
  learning_rate: float = sql.Column(sql.Float, nullable=True)
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
    return self.elapsed_time_ms / 1000

  @property
  def graphs_per_second(self) -> float:
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
      target_count=results.targets.shape[0],
      graph_count=data.graph_count,
      iteration_count=results.iteration_count,
      model_converged=results.model_converged,
      learning_rate=results.learning_rate,
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
    run_id_lib.RunId.SqlStringColumnType(),
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
    return pickle.loads(codecs.decode(self.data.binary_data, "zlib"))

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

  def GetRunLogs(
    self,
    run_id: run_id_lib.RunId,
    eager_batch_details: bool = True,
    eager_checkpoint_data: bool = True,
    session: Optional[sqlutil.Database.SessionType] = None,
  ):
    with self.Session(session=session) as session:
      if not session.query(RunId).filter(RunId.run_id == str(run_id)).scalar():
        raise ValueError(f"Run not found: {run_id}")

      parameters = session.query(Parameter).filter(
        Parameter.run_id == str(run_id)
      )
      batches = session.query(Batch).filter(Batch.run_id == str(run_id))
      if eager_batch_details:
        batches = batches.options(sql.orm.joinedload(Batch.details))

      checkpoints = session.query(Checkpoint).filter(
        Checkpoint.run_id == str(run_id)
      )
      if eager_checkpoint_data:
        checkpoints = checkpoints.options(sql.orm.joinedload(Checkpoint.data))

      return RunLogs(
        run_id=run_id,
        parameters=parameters.all(),
        batches=batches.all(),
        checkpoints=checkpoints.all(),
      )

  def GetRunParameters(
    self,
    run_id: run_id_lib.RunId,
    session: Optional[sqlutil.Database.SessionType] = None,
  ):
    """Return a table of parameters for the given run.

    Args:
      run_id: The run ID.
      session: A session object to re-use.

    Returns:
      A dataframe with timestamp, name, value, type columns.
    """
    with self.Session(session=session) as session:
      query = (
        session.query(
          Parameter.timestamp,
          Parameter.name,
          Parameter.binary_value.label("value"),
          Parameter.type_num.label("type"),
        )
        .filter(Parameter.run_id == str(run_id))
        .order_by(Parameter.run_id, Parameter.type_num, Parameter.name)
      )

      df = pdutil.QueryToDataFrame(session, query)
      if not len(df):
        raise ValueError(f"Run not found: {run_id}")

      pdutil.RewriteColumn(df, "type", lambda x: ParameterType(x).name.lower())
      pdutil.RewriteColumn(df, "value", lambda x: pickle.loads(x))
      return df

  def GetBestResults(
    self,
    run_id: run_id_lib.RunId,
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> Dict[epoch.Type, epoch.BestResults]:
    """Get the best results for a given run.

    Returns:
      A mapping from <epoch_type, epoch.Results> for the best accuracy on each
      of the epoch types.
    """
    with self.Session(session=session) as session:
      # Check that the run exists:
      if not session.query(RunId).filter(RunId.run_id == str(run_id)).scalar():
        raise ValueError(f"Run not found: {run_id}")

      best_results: Dict[epoch.Type, epoch.BestResults] = {}
      for epoch_type in list(epoch.Type):
        accuracy_to_epoch_num = {
          row.accuracy: row.epoch_num
          for row in session.query(
            Batch.epoch_num, sql.func.avg(Batch.accuracy).label("accuracy")
          )
          .filter(
            Batch.run_id == str(run_id),
            Batch.epoch_type_num == epoch_type.value,
          )
          .group_by(Batch.epoch_num)
        }
        if accuracy_to_epoch_num:
          epoch_num = accuracy_to_epoch_num[max(accuracy_to_epoch_num.keys())]
          epoch_results = self.GetEpochResults(
            run_id=run_id, epoch_num=epoch_num, epoch_type=epoch_type
          )
          best_results_for_type = epoch.BestResults(
            epoch_num=epoch_num, results=epoch_results
          )
        else:
          best_results_for_type = epoch.BestResults()
        best_results[epoch_type] = best_results_for_type
    return best_results

  def GetEpochResults(
    self,
    run_id: run_id_lib.RunId,
    epoch_num: int,
    epoch_type: epoch.Type,
    weight: sql.Column = Batch.target_count,
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> epoch.Results:
    """Get the aggregate results for a single epoch.

    Args:
      run_id: The run ID.
      epoch_num: The epoch num.
      epoch_type: The epoch type.
      weight: The batch column to use for weighting batch metrics.
      session: A session instance.

    Returns:
      An epoch.Results tuple.
    """
    with self.Session(session=session) as session:
      # Check that the epoch exists:
      if not session.query(RunId).filter(RunId.run_id == str(run_id)).scalar():
        raise ValueError(f"Run not found: {run_id}")

      df = self.GetWeightedEpochStats(
        batch_filters=[
          lambda: Batch.run_id == str(run_id),
          lambda: Batch.epoch_num == epoch_num,
          lambda: Batch.epoch_type_num == epoch_type.value,
        ],
        weight=weight,
        session=session,
      )
      # Check that a single match was made.
      if not len(df):
        raise ValueError(
          f"Epoch not found: {epoch_type.name.lower()} {run_id}@{epoch_num}"
        )
      elif len(df) > 1:
        raise ValueError(
          "Multiple rows found for: "
          f"{epoch_type.name.lower()} {run_id}@{epoch_num}"
        )

    epoch_results = df.loc[0]

    return epoch.Results(
      batch_count=epoch_results.batch_count,
      iteration_count=epoch_results.iteration_count,
      model_converged=epoch_results.model_converged,
      learning_rate=epoch_results.learning_rate,
      loss=epoch_results.loss,
      accuracy=epoch_results.accuracy,
      precision=epoch_results.precision,
      recall=epoch_results.recall,
      f1=epoch_results.f1,
    )

  def GetWeightedEpochStats(
    self,
    batch_filters: List[Callable[[], bool]] = None,
    weight: sql.Column = Batch.target_count,
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> pd.DataFrame:
    """Compute a table of per-epoch results.

    The weighting method used here is the same as used in
    deeplearning.ml4pl.models.batch.RollingResults.

    Use this method to aggregate over the batches table with a consistent
    weighting strategy, don't roll your own implementation.

    Args:
      batch_filters: An optional list of callbacks which return filters on the
        Batch table.
      weight: The weighting strategy. By default, weight by target count.
      session: An optional database session to re-use.

    Returns:
      A data frame consisting of per-epoch metrics.
    """
    batch_filters = batch_filters or []
    with self.Session(session=session) as session:
      # Compute per-epoch weighted metrics.
      left = session.query(
        Batch.run_id,
        Batch.epoch_num,
        Batch.epoch_type_num.label("epoch_type"),
        sql.func.min(Batch.timestamp).label("timestamp"),
        sql.func.count(Batch.run_id).label("batch_count"),
        sql.func.sum(Batch.graph_count).label("graph_count"),
        sql.func.sum(Batch.target_count).label("target_count"),
        sql.func.avg(Batch.iteration_count * weight).label(
          "weighted_iteration_count"
        ),
        sql.func.avg(Batch.model_converged * weight).label(
          "weighted_model_converged"
        ),
        sql.func.avg(Batch.learning_rate * weight).label(
          "weighted_learning_rate"
        ),
        sql.func.avg(Batch.loss * weight).label("weighted_loss"),
        sql.func.sum(Batch.accuracy * weight).label("weighted_accuracy"),
        sql.func.sum(Batch.precision * weight).label("weighted_precision"),
        sql.func.sum(Batch.recall * weight).label("weighted_recall"),
        sql.func.sum(Batch.f1 * weight).label("weighted_f1"),
        sql.func.sum(Batch.elapsed_time_ms).label("runtime"),
        sql.func.sum(Batch.elapsed_time_ms * weight).label("weighted_runtime"),
      ).group_by(Batch.run_id, Batch.epoch_num, Batch.epoch_type_num)
      for filter in batch_filters:
        left = left.filter(filter())
      left = left.subquery()

      # Compute per-epoch weight sums.
      right = session.query(
        Batch.run_id,
        Batch.epoch_num,
        Batch.epoch_type_num.label("epoch_type"),
        sql.func.sum(weight).label("weight"),
      ).group_by(Batch.run_id, Batch.epoch_num, Batch.epoch_type_num)
      for filter in batch_filters:
        right = right.filter(filter())
      right = right.subquery()

      # Normalize the metrics by their weight.
      query = session.query(
        left.c.run_id,
        left.c.epoch_num,
        left.c.epoch_type,
        left.c.timestamp,
        left.c.batch_count,
        left.c.graph_count,
        left.c.target_count,
        (left.c.weighted_iteration_count / right.c.weight).label(
          "iteration_count"
        ),
        (left.c.weighted_model_converged / right.c.weight).label(
          "model_converged"
        ),
        (left.c.weighted_learning_rate / right.c.weight).label("learning_rate"),
        (left.c.weighted_loss / right.c.weight).label("loss"),
        (left.c.weighted_accuracy / right.c.weight).label("accuracy"),
        (left.c.weighted_precision / right.c.weight).label("precision"),
        (left.c.weighted_recall / right.c.weight).label("recall"),
        (left.c.weighted_f1 / right.c.weight).label("f1"),
        left.c.runtime,
        left.c.weighted_runtime,
      ).join(
        right,
        sql.and_(
          left.c.run_id == right.c.run_id,
          left.c.epoch_num == right.c.epoch_num,
          left.c.epoch_type == right.c.epoch_type,
        ),
      )

      df = pdutil.QueryToDataFrame(session, query)

    # Rewrite the epoch_type column to use the native enum type.
    pdutil.RewriteColumn(df, "epoch_type", lambda x: epoch.Type(x))

    # Convert ints to floats. We can't do this in the query because MySQL
    # does not support casting to FLOAT.
    df["runtime"] = df["runtime"].values.astype(np.float32) / 1000
    df["weighted_runtime"] = (
      df["weighted_runtime"].values.astype(np.float32) / 1000
    )
    # Compute a per-graph throughput column.
    df["throughput"] = df["graph_count"] / df["runtime"]

    return df

  ############################################################################
  # Properties.
  ############################################################################

  @database_statistic
  def last_batch(self) -> Optional[datetime.datetime]:
    """Returns the timestamp of the most recent batch."""
    with self.Session() as session:
      return (
        session.query(Batch.timestamp)
        .order_by(Batch.timestamp.desc())
        .limit(1)
        .scalar()
      )

  @database_statistic
  def run_count(self) -> int:
    """Returns the number of unique runs in the database."""
    with self.Session() as session:
      return session.query(sql.func.count(RunId.run_id)).scalar()

  @database_statistic
  def parameters_count(self) -> int:
    """Returns the number of parameters in the database."""
    with self.Session() as session:
      return session.query(sql.func.count(Parameter.id)).scalar()

  @database_statistic
  def batch_count(self) -> int:
    """Returns the number of batches in the database."""
    with self.Session() as session:
      return session.query(sql.func.count(Batch.id)).scalar()

  @database_statistic
  def batch_details_count(self) -> int:
    """Returns the number of batch details in the database."""
    with self.Session() as session:
      return session.query(sql.func.count(BatchDetails.id)).scalar()

  @database_statistic
  def checkpoint_count(self) -> int:
    """Returns the number of checkpoints in the database."""
    with self.Session() as session:
      return session.query(sql.func.count(Checkpoint.id)).scalar()

  @database_statistic
  def run_ids(self) -> List[run_id_lib.RunId]:
    """Returns the list of run IDs."""
    with self.Session() as session:
      return [row.run_id for row in session.query(RunId.run_id)]

  @database_statistic
  def tags(self) -> List[str]:
    """Returns the list of run IDs."""
    with self.Session() as session:
      return sorted(
        [
          pickle.loads(row.binary_value)
          for row in session.query(
            sql.func.distinct(Parameter.binary_value).label("binary_value")
          ).filter(
            Parameter.type_num == ParameterType.FLAG.value,
            Parameter.name == "tag",
          )
        ]
      )

  @database_statistic
  def tag_run_count(self) -> Dict[str, int]:
    """Returns a mapping from tag to the number of runs with that count."""
    with self.Session() as session:
      return {
        pickle.loads(row.binary_value): row.count
        for row in session.query(
          Parameter.binary_value,
          sql.func.count(Parameter.binary_value).label("count"),
        )
        .filter(
          Parameter.type_num == ParameterType.FLAG.value,
          Parameter.name == "tag",
        )
        .group_by(Parameter.binary_value)
      }

  @property
  def stats_json(self) -> Dict[str, Any]:
    """Returns the database statics as a JSON dictionary."""
    return {
      name: function(self) for name, function in database_statistics_registry
    }

  ############################################################################
  # Export.
  ############################################################################

  def CopyRunLogs(
    self,
    output_db: "Database",
    run_ids: List[run_id_lib.RunId],
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> int:
    """Copy the logs for a given runs. This handles copying all of the tables.

    Args:
      output_db: The destination database to copy to.
      run_ids: A list of run IDs to copy.

    Returns:
      The total number of rows that were copied.

    Raises:
      ValueError: If any of the runs are not found, or if they already exists
        in the destination database.
    """

    def Copy(
      query, session: sqlutil.Database.SessionType, batch_size: int = 512
    ):
      """Copy the results of the query to the destination session."""
      row_count = 0
      for chunk in sqlutil.OffsetLimitBatchedQuery(
        query, batch_size=batch_size
      ):
        for row in chunk.rows:
          row_count += 1
          session.merge(row)
      return row_count

    run_id_strings = set(str(run_id) for run_id in run_ids)

    with self.Session(session=session) as src, output_db.Session(
      commit=True
    ) as dst:
      # The queries of rows to copy.
      src_run_ids = src.query(RunId).filter(RunId.run_id.in_(run_id_strings))
      src_params = src.query(Parameter).filter(
        Parameter.run_id.in_(run_id_strings)
      )
      src_batches = (
        src.query(Batch)
        .filter(Batch.run_id.in_(run_id_strings))
        .options(sql.orm.joinedload(Batch.details))
      )
      src_checkpoints = (
        src.query(Checkpoint)
        .filter(Checkpoint.run_id.in_(run_id_strings))
        .options(sql.orm.joinedload(Checkpoint.data))
      )

      # Check for any runs that do not exist.
      missing_runs = run_id_strings - set(row.run_id for row in src_run_ids)
      if missing_runs:
        raise ValueError(f"Runs not found: {missing_runs}")

      # Check that the runs don't exist in the destination.
      already_exists = [
        row.run_id
        for row in dst.query(RunId).filter(RunId.run_id.in_(run_id_strings))
      ]
      if already_exists:
        raise ValueError(
          f"Destination database {output_db.url} already has "
          f"runs: {already_exists}"
        )

      # Copy the tables.
      row_count = 0
      row_count += Copy(src_run_ids, dst)
      row_count += Copy(src_params, dst)
      row_count += Copy(src_batches, dst)
      row_count += Copy(src_checkpoints, dst)

    return row_count

  def GetParametersJson(
    self,
    type: ParameterType,
    name: str,
    session: sqlutil.Database.SessionType = None,
  ) -> Dict[str, Any]:
    """Returns a <run_id, value> dictionary of a parameter value.

    Args:
      type: The type of parameter.
      name: The name of the parameter.
      session: An existing session object to re-use.

    Returns:
      A <run_id, value> dictionary of all values for this parameter.
    """
    with self.Session(session=session):
      return {
        row.run_id: pickle.loads(row.binary_value)
        for row in session.query(
          Parameter.run_id, Parameter.binary_value
        ).filter(
          Parameter.type_num == type.value, Parameter.name == name
        )
      }

  def SelectRunIds(
    self,
    run_ids: Optional[Iterable[Union[run_id_lib.RunId, str]]] = None,
    tags: Optional[Iterable[str]] = None,
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> Set[str]:
    """Select a list of run ID strings using the given filters.

    Args:
      run_ids: A list of candidate runs. Only those that exist are returned.
      tags: A list of tags. All runs belonging to those tags are returned.
      session: A session to re-use.

    Returns:
      A (possible empty) set of run ID strings.
    """
    selected_run_ids = set()

    with self.Session(session=session) as session:
      # Select runs by run ID.
      if run_ids:
        selected_run_ids = selected_run_ids.union(
          {
            row.run_id
            for row in session.query(RunId).filter(
              RunId.run_id.in_({str(s) for s in run_ids})
            )
          }
        )

      # Select runs by tag name.
      if tags:
        binary_value_sha1 = [crypto.sha1(pickle.dumps(tag)) for tag in tags]
        selected_run_ids = selected_run_ids.union(
          {
            row.run_id
            for row in session.query(
              sql.func.distinct(Parameter.run_id).label("run_id")
            ).filter(
              Parameter.type_num == ParameterType.FLAG.value,
              Parameter.name == "tag",
              Parameter.binary_value_sha1.in_(binary_value_sha1),
            )
          }
        )

    return selected_run_ids

  def GetTables(
    self,
    run_ids: List[run_id_lib.RunId] = None,
    extra_flags: List[str] = None,
    session: Optional[sqlutil.Database.SessionType] = None,
  ) -> Iterable[Tuple[str, pd.DataFrame]]:
    """Compute tables of database statisics.

    Args:
      run_id: An optional list of run IDs to generate the tables for. If not
        provided, all runs are used.
      extra_flags: A list of additional flag parameters to dump.
      session: An optional session to re-use.

    Returns:
      An iterator over <name, dataframe> tuples.
    """
    extra_flag_names = ["tag", "graph_db"] + (extra_flags or [])

    with self.Session(session=session) as session:

      #########################################################################
      # Parameters.
      #########################################################################

      # A map from parameter name to values.
      extra_flags: Dict[str, Dict[str, Any]] = {
        flag: self.GetParametersJson(ParameterType.FLAG, flag, session=session)
        for flag in extra_flag_names
      }

      # Table of parameters.
      query = session.query(
        Parameter.timestamp,
        Parameter.run_id,
        Parameter.name,
        Parameter.binary_value.label("value"),
        Parameter.type_num.label("type"),
      ).order_by(Parameter.run_id, Parameter.type_num, Parameter.name)
      if run_ids:
        query = query.filter(
          Parameter.run_id.in_([str(run_id) for run_id in run_ids])
        )

      df = pdutil.QueryToDataFrame(session, query)
      pdutil.RewriteColumn(df, "type", lambda x: ParameterType(x).name.lower())
      pdutil.RewriteColumn(df, "value", lambda x: pickle.loads(x))
      yield "parameters", df

      #########################################################################
      # Per-epoch stats.
      #########################################################################

      if run_ids:
        batch_filters = [
          lambda: Batch.run_id.in_([str(run_id) for run_id in run_ids])
        ]
      else:
        batch_filters = None
      per_epoch_df = self.GetWeightedEpochStats(batch_filters=batch_filters)

      # Flatten the {train,val,test} rows into an array of columns.
      rows = []
      epoch_type_columns = [
        "batch_count",
        "graph_count",
        "iteration_count",
        "model_converged",
        "learning_rate",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "runtime",
        "throughput",
      ]

      for run_id in set(per_epoch_df["run_id"].values):
        run_df = per_epoch_df[per_epoch_df["run_id"] == run_id]
        for epoch_num in set(run_df["epoch_num"].values):
          row = {"run_id": run_id, "epoch_num": epoch_num}
          for flag in extra_flag_names:
            row[flag] = extra_flags[flag].get(run_id)

          for i, epoch_type in enumerate(list(epoch.Type)):
            epoch_df = run_df[
              (run_df["epoch_num"] == epoch_num)
              & (run_df["epoch_type"] == epoch_type)
            ]

            if not i:
              row["timestamp"] = min(epoch_df["timestamp"].values)

            # Sanity check that we have exctly one epoch.
            if len(epoch_df) == 1:
              for column in epoch_type_columns:
                row[f"{epoch_type.name.lower()}_{column}"] = epoch_df.iloc[0][
                  column
                ]
            elif len(epoch_df) > 1:
              raise ValueError

          rows.append(row)

      # Build the column name list.
      columns = ["run_id", "timestamp", "epoch_num"] + extra_flag_names
      for epoch_type in list(epoch.Type):
        columns += [
          f"{epoch_type.name.lower()}_{column}" for column in epoch_type_columns
        ]

      # Put it into a dataframe.
      per_epoch_df = pd.DataFrame(rows, columns=columns)
      if len(per_epoch_df):
        per_epoch_df.sort_values(
          ["run_id", "timestamp", "epoch_num"], inplace=True
        )

      yield "epochs", per_epoch_df

      #########################################################################
      # Per-run stats.
      # This table contains a single row for each run where there is a best
      # validation accuracy.
      #########################################################################

      rows = []
      epoch_counts: List[int] = []
      for run_id in set(per_epoch_df.run_id.values):

        run_df = per_epoch_df[per_epoch_df["run_id"] == run_id]
        # Safely hand 'null' values.
        run_df = run_df[run_df["val_accuracy"].notnull()]
        if len(run_df):
          epoch_counts.append(len(run_df))
          rows.append(run_df.loc[run_df["val_accuracy"].idxmax()])

      per_run_df = pd.DataFrame(rows, columns=per_epoch_df.columns.values)
      per_run_df["epoch_count"] = epoch_counts
      if len(rows):
        per_run_df.sort_values(["run_id", "timestamp"], inplace=True)
      per_run_df.rename(columns={"epoch_num": "best_epoch"}, inplace=True)

      # Rejig the columns so that epoch_count comes after best_epoch.
      columns = per_run_df.columns.tolist()
      i = columns.index("best_epoch")
      columns = columns[:i] + [columns[-1]] + columns[i:-1]
      per_run_df = per_run_df[columns]

      yield "runs", per_run_df

  def Prune(self, session: Optional[sqlutil.Database.SessionType] = None):
    """Remove any runs that do not have a model checkpoint.

    This is useful for "spring cleaning" a database which has a bunch of
    test/failed runs, although note that any job that is currently running but
    hasn't yet reached the end of epoch 1 will be tidied up!
    """
    with self.Session(session=session, commit=True) as session:
      runs_with_checkpoints = {
        row.run_id
        for row in session.query(
          sql.func.distinct(Checkpoint.run_id).label("run_id")
        )
      }

      runs_with_no_checkpoints = set(self.run_ids) - runs_with_checkpoints
      app.Log(
        1,
        "Pruning %s runs: %s",
        len(runs_with_no_checkpoints),
        runs_with_no_checkpoints,
      )
      session.query(RunId).filter(
        RunId.run_id.in_(runs_with_no_checkpoints)
      ).delete(synchronize_session=False)


app.DEFINE_database(
  "log_db", Database, None, "The database to write model logs to."
)


def DatetimeHandler(object):
  if isinstance(object, datetime.datetime):
    return str(object)


def Main():
  """Main entry point."""
  log_db = FLAGS.log_db()

  # Delete runs without checkpoints.
  if FLAGS.prune_logs:
    log_db.Prune()

  # Delete logs as requested.
  with log_db.Session() as session:
    run_ids_to_remove = log_db.SelectRunIds(
      run_ids=FLAGS.rm, tags=FLAGS.rm_tag, session=session
    )
    if run_ids_to_remove:
      app.Log(
        1,
        "Removing %s: %s",
        humanize.Plural(len(run_ids_to_remove), "run"),
        run_ids_to_remove,
      )
      session.query(RunId).filter(RunId.run_id.in_(run_ids_to_remove)).delete(
        synchronize_session=False
      )

  print(jsonutil.format_json(log_db.stats_json, default=DatetimeHandler))


if __name__ == "__main__":
  app.Run(Main)
