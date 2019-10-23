"""Database backend for model logs."""
import datetime
import pickle
import sqlalchemy as sql
import typing
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import humanize
from labm8 import labdate
from labm8 import pdutil
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)


class BatchLog(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description of a batch of graphs and the results of running it through a
  model."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = sql.Column(sql.String(64), nullable=False)

  # The epoch number, >= 1.
  epoch: int = sql.Column(sql.Integer, nullable=False)
  # The batch number within the epoch, >= 1.
  batch: int = sql.Column(sql.Integer, nullable=False)
  # The batch number across all epochs.
  global_step: int = sql.Column(sql.Integer, nullable=False)

  timestamp: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)

  elapsed_time_seconds: float = sql.Column(sql.Float, nullable=False)

  # The number of graphs in the batch.
  graph_count: int = sql.Column(sql.Integer, nullable=False)

  # The number of nodes in the batch.
  node_count: int = sql.Column(sql.Integer, nullable=False)

  # The model loss on the batch.
  loss: float = sql.Column(sql.Float, nullable=False)

  # The model accuracy on the batch.
  accuracy: float = sql.Column(sql.Float, nullable=False)

  # The GraphMeta.group column that this batch of graphs came from.
  group: str = sql.Column(sql.String(32), nullable=False)

  # A pickled array of GraphMeta.id values.
  pickled_graph_indices: bytes = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False)

  # A pickled array of accuracies, with the same shape as pickled_predictions.
  pickled_accuracies: bytes = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False)

  # A pickled array of model predictions, one for each graph in the batch.
  pickled_predictions: bytes = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False)

  @property
  def graph_indices(self) -> typing.List[int]:
    return pickle.loads(self.pickled_graph_indices)

  @property
  def accuracies(self) -> typing.Any:
    return pickle.loads(self.pickled_accuracies)

  @property
  def predictions(self) -> typing.Any:
    return pickle.loads(self.pickled_predictions)

  @property
  def graphs_per_second(self):
    if self.elapsed_time_seconds:
      return max(self.graph_count, 1) / self.elapsed_time_seconds
    else:
      return 0

  @property
  def nodes_per_second(self):
    if self.elapsed_time_seconds:
      return max(self.node_count, 1) / self.elapsed_time_seconds
    else:
      return 0

  def __repr__(self) -> str:
    return (
        f"{self.run_id} Epoch {humanize.Commas(self.epoch)} {self.group}: "
        f"graphs/sec={humanize.DecimalPrefix(self.graphs_per_second, '')} | "
        f"loss={self.loss:.4f} | "
        f"acc={self.accuracy:.2%}")


class Parameter(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """TODO."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = sql.Column(sql.String(64), nullable=False)

  # One of: {model_flags,flags,build_info}
  type: str = sql.Column(sql.String(256), nullable=False)

  # The name of the parameter.
  parameter: str = sql.Column(sql.String(256), nullable=False)
  # The value for the parameter.
  value: str = sql.Column(
      sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False)


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)

  def BatchLogsToDataFrame(self, run_id: str, per_global_step: bool = False):
    with self.Session() as session:
      q = session.query(
          BatchLog.epoch,
          BatchLog.group,
          sql.func.min(BatchLog.global_step).label("global_step"),
          sql.func.avg(BatchLog.loss).label("loss"),
          sql.func.avg(BatchLog.accuracy * 100).label("accuracy"),
          sql.func.sum(
              BatchLog.elapsed_time_seconds).label("elapsed_time_seconds"),
          sql.sql.expression.cast(sql.func.sum(BatchLog.graph_count), sql.Integer).label("graph_count"),
          sql.sql.expression.cast(sql.func.sum(BatchLog.node_count), sql.Integer).label("node_count"),
      )

      q = q.filter(BatchLog.run_id == run_id)

      q = q.group_by(BatchLog.epoch, BatchLog.group)

      # Group each individual step. Since there is only one log per step,
      # this means return all rows without grouping.
      if per_global_step:
        q = q.group_by(BatchLog.global_step) \
          .order_by(BatchLog.global_step)

      q = q.order_by(BatchLog.epoch, BatchLog.group)

      return pdutil.QueryToDataFrame(session, q)

  def ParametersToDataFrame(self, run_id: str, type: str):
    with self.Session() as session:
      q = session.query(Parameter.parameter, Parameter.value)
      q = q.filter(Parameter.run_id == run_id)
      q = q.filter(Parameter.type == type)
      q = q.order_by(Parameter.parameter)
      return pdutil.QueryToDataFrame(session, q).set_index('parameter')
