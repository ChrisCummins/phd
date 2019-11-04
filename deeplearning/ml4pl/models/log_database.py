"""Database backend for model logs."""
import datetime
import pickle
import typing

import sqlalchemy as sql
from labm8 import app
from labm8 import humanize
from labm8 import labdate
from labm8 import pdutil
from labm8 import sqlutil
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                          nullable=False)


class BatchLogMeta(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description running a batch of graphs through a model."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run. An experiment run
  # is a timestamp and a hostname.
  run_id: str = sql.Column(sql.String(64), nullable=False, index=True)

  # The epoch number, >= 1.
  epoch: int = sql.Column(sql.Integer, nullable=False, index=True)

  # The batch number within the epoch, >= 1.
  batch: int = sql.Column(sql.Integer, nullable=False)

  # The type of batch. One of {train,test,val}
  type: str = sql.Column(sql.String(32), nullable=False)

  # The GraphMeta.group column that this batch of graphs came from.
  group: str = sql.Column(sql.String(32), nullable=False)

  # The batch number across all epochs.
  global_step: int = sql.Column(sql.Integer, nullable=False)

  # The duration of the batch.
  elapsed_time_seconds: float = sql.Column(sql.Float, nullable=False)

  # The number of model iterations to compute the final results. This is used
  # by iterative models such as message passing networks.
  iteration_count: int = sql.Column(sql.Integer, nullable=False, default=1)
  # For iterative models, this indicates whether the state of the model at
  # iteration_count had converged on a solution.
  model_converged: bool = sql.Column(sql.Boolean, nullable=True)

  # The number of graphs in the batch.
  graph_count: int = sql.Column(sql.Integer, nullable=False)

  # The number of nodes in the batch.
  node_count: int = sql.Column(sql.Integer, nullable=False)

  # Metrics describing model performance.
  loss: float = sql.Column(sql.Float, nullable=True)
  accuracy: float = sql.Column(sql.Float, nullable=False)
  precision: float = sql.Column(sql.Float, nullable=False)
  recall: float = sql.Column(sql.Float, nullable=False)
  f1: float = sql.Column(sql.Float, nullable=False)

  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)

  batch_log: 'BatchLog' = sql.orm.relationship('BatchLog',
                                               uselist=False,
                                               cascade="all",
                                               back_populates="meta")

  # Convenience properties to access data in the joined 'BatchLog' table. If
  # accessing many of these properties, consider using
  # sql.orm.joinedload(BatchLogMeta.batch) to eagerly load the joined table.

  @property
  def graph_indices(self) -> typing.List[int]:
    return pickle.loads(self.batch_log.pickled_graph_indices)

  @graph_indices.setter
  def graph_indices(self, data) -> None:
    self.batch_log.pickled_graph_indices = pickle.dumps(data)

  @property
  def accuracies(self) -> typing.Any:
    return pickle.loads(self.batch_log.pickled_accuracies)

  @accuracies.setter
  def accuracies(self, data) -> None:
    self.batch_log.pickled_accuracies = pickle.dumps(data)

  @property
  def predictions(self) -> typing.Any:
    return pickle.loads(self.batch_log.pickled_predictions)

  @predictions.setter
  def predictions(self, data) -> None:
    self.batch_log.pickled_predictions = pickle.dumps(data)

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
        f"{self.run_id} Epoch {humanize.Commas(self.epoch)} {self.type}: "
        f"graphs/sec={humanize.DecimalPrefix(self.graphs_per_second, '')} | "
        f"loss={self.loss:.4f} | "
        f"acc={self.accuracy:.2%}")


class BatchLog(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The per-instance results of a batch log.

  In practise, this table will grow large.
  """
  id: int = sql.Column(sql.Integer,
                       sql.ForeignKey('batch_log_metas.id'),
                       primary_key=True)

  # A pickled array of GraphMeta.id values.
  pickled_graph_indices: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                            nullable=False)

  # A pickled array of accuracies, of shape
  # [num_instances * num_targets_per_instance]. The number of targets per
  # instance will depend on the type of classification problem. For graph-level
  # classification, there is one target per instance. For node-level
  # classification, there are num_nodes targets for each graph.
  pickled_accuracies: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                         nullable=False)

  # A pickled array of sparse model predictions, of shape
  # [num_instances * num_targets_per_instance, num_classes]. See
  # pickled_accuracies for a description of row count.
  pickled_predictions: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                          nullable=False)

  meta: BatchLogMeta = sql.orm.relationship(BatchLogMeta,
                                            back_populates="batch_log")


class Parameter(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A description of an experimental parameter."""
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string to uniquely identify the given experiment run.
  run_id: str = sql.Column(sql.String(64), nullable=False, index=True)

  # One of: {model_flags,flags,build_info}
  type: str = sql.Column(sql.String(1024), nullable=False, index=True)

  # The name of the parameter.
  parameter: str = sql.Column(sql.String(1024), nullable=False)
  # The value for the parameter.
  value: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                          nullable=False)


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
          sql.func.min(BatchLogMeta.timestamp).label('timestamp'),
          sql.func.min(BatchLogMeta.global_step).label("global_step"),
          sql.func.avg(BatchLogMeta.loss).label("loss"),
          sql.func.avg(BatchLogMeta.iteration_count).label("iteration_count"),
          sql.func.avg(BatchLogMeta.model_converged).label("converged"),
          sql.func.avg(BatchLogMeta.accuracy * 100).label("accuracy"),
          sql.func.avg(BatchLogMeta.precision).label("precision"),
          sql.func.avg(BatchLogMeta.recall).label("recall"),
          sql.func.avg(BatchLogMeta.f1).label("f1"),
          sql.func.sum(
              BatchLogMeta.elapsed_time_seconds).label("elapsed_time_seconds"),
          sql.sql.expression.cast(sql.func.sum(BatchLogMeta.graph_count),
                                  sql.Integer).label("graph_count"),
          sql.sql.expression.cast(sql.func.sum(BatchLogMeta.node_count),
                                  sql.Integer).label("node_count"),
      )

      q = q.filter(BatchLogMeta.run_id == run_id)

      q = q.group_by(BatchLogMeta.epoch, BatchLogMeta.type)

      # Group each individual step. Since there is only one log per step,
      # this means return all rows without grouping.
      if per_global_step:
        q = q.group_by(BatchLogMeta.global_step) \
          .order_by(BatchLogMeta.global_step)

      q = q.order_by(BatchLogMeta.epoch, BatchLogMeta.type)

      df = pdutil.QueryToDataFrame(session, q)

      df['reltime'] = [t - df['timestamp'][0] for t in df['timestamp']]

      return df

  def ParametersToDataFrame(self, run_id: str, type: str):
    """Return a table of parameters of the given type for the specified run.

    Args:
      run_id: The run ID to return the parameters of.
      type: The type of parameter to return.

    Returns:
      A pandas dataframe.
    """
    with self.Session() as session:
      q = session.query(Parameter.parameter, Parameter.value)
      q = q.filter(Parameter.run_id == run_id)
      q = q.filter(Parameter.type == type)
      q = q.order_by(Parameter.parameter)
      return pdutil.QueryToDataFrame(session, q).set_index('parameter')
