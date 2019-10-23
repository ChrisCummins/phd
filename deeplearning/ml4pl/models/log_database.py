"""Database backend for model logs."""
import datetime
import pickle
import sqlalchemy as sql
import typing
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import labdate
from labm8 import sqlutil
from labm8 import humanize

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

  # The epoch number, starting at zero.
  epoch: int = sql.Column(sql.Integer, nullable=False)
  # The batch number within the epoch, starting at zero.
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

  # A pickled array of model predictions, one for each graph in the batch.
  pickled_predictions: bytes = sql.Column(
      sqlutil.ColumnTypes.LargeBinary(), nullable=False)

  @property
  def graph_indices(self) -> typing.List[int]:
    return pickle.loads(self.pickled_graph_indices)

  @property
  def predictions(self) -> typing.Any:
    return pickle.loads(self.pickled_predictions)

  @property
  def graphs_per_second(self):
    return max(self.graph_count, 1) / self.elapsed_time_seconds

  @property
  def nodes_per_second(self):
    return max(self.node_count, 1) / self.elapsed_time_seconds

  def __repr__(self) -> str:
    return (f"Epoch {humanize.Commas(self.epoch)} {self.group}. "
            f"batch={humanize.Commas(self.batch)} | "
            f"graphs/sec={self.graphs_per_second:.2f} | "
            f"loss={self.loss:.4f} | "
            f"acc={self.accuracy:.2%}")


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
