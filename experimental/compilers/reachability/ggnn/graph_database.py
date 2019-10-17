"""Database backend for GGNN graphs."""
import datetime
import sqlalchemy as sql
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from labm8 import app
from labm8 import labdate
from labm8 import sqlutil


FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sql.String(64), nullable=False)


class GraphMeta(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A table of graph metadata.

  For every GraphMeta, there should be a corresponding Graph row containing the
  actual data.
  """
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string name to group graphs into discrete buckets, e.g. "train", "test",
  # or "1", "2", ... k for k-fold classification.
  group: str = sql.Column(sql.String(32), nullable=False)

  # No foreign key relationship because they are separate databases.
  bytecode_id: int = sql.Column(sql.Integer, nullable=False)

  # The source of the graph. This is duplicates of information stored in the
  # LlvmBytecode table.
  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False)

  node_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)
  node_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)
  edge_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)

  node_features_dimensionality: int = sql.Column(sql.Integer,default=0, nullable=False)
  edge_features_dimensionality: int = sql.Column(sql.Integer, default=0, nullable=False)
  graph_features_dimensionality: int = sql.Column(sql.Integer, default=0, nullable=False)

  node_labels_dimensionality: int = sql.Column(sql.Integer, default=0, nullable=False)
  edge_labels_dimensionality: int = sql.Column(sql.Integer, default=0, nullable=False)
  graph_labels_dimensionality: int = sql.Column(sql.Integer, default=0, nullable=False)

  # The minimum number of message passing steps that are be required to produce
  # the labels from the features. E.g. for graph flooding problems, this value
  # will be the diameter of the graph.
  data_flow_max_steps_required: int = sql.Column(
      sql.Integer, default=0, nullable=False)

  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)

  graph: 'Graph' = sql.orm.relationship(
      'Graph', uselist=False, back_populates="meta")


class Graph(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for a graph.

  This is an opaque byte array that can be used as needed, e.g. for pickled
  dictionaries, networkx graphs, etc.
  """
  id: int = sql.Column(sql.Integer, sql.ForeignKey('graph_metas.id'),
                       primary_key=True)
  data: bytes = sql.Column(
      sql.LargeBinary().with_variant(sql.LargeBinary(2**31), 'mysql'),
      nullable=False)
  meta: GraphMeta = sql.orm.relationship('GraphMeta', back_populates="graph")


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)


def BufferedGraphReader(db: Database, filter_cb = None,
                        order_by_random: bool = False,
                        eager_graph_loading: bool = True,
                        buffer_size: int=256):
  """An iterator over the graphs in a database.

  Args:
    db: The database to iterate over the graphs of.
    filter_db: An optional callback which returns a filter condition on the
      graph table.
    order_by_random: If true, return the graphs of the database in a random
      order.
    eager_graph_loading: If true, load the contents of the Graph table eagerly,
      preventing the need for subsequent SQL queries to access the graph data.
    buffer_size: The number of graphs to query from the database at a time. A
      larger number reduces the number of queries, but increases the memory
      requirement.
  """
  with db.Session() as s:
    # Load both the graph data and the graph eagerly.
    q = s.query(GraphMeta)

    if eager_graph_loading:
      q = q.options(sql.orm.joinedload(GraphMeta.graph))

    if filter_cb:
      q = q.filter(filter_cb())

    if order_by_random:
      q = q.order_by(db.Random())

    for batch in sqlutil.OffsetLimitBatchedQuery(q, batch_size=buffer_size):
      for row in batch.rows:
        yield row
