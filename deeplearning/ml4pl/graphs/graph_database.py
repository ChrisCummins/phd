"""Database backend for GGNN graphs."""
import datetime
import networkx as nx
import pickle
import sqlalchemy as sql
import typing
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_dict
from labm8 import app
from labm8 import labdate
from labm8 import sqlutil

FLAGS = app.FLAGS

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                          nullable=False)


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

  node_features_dimensionality: int = sql.Column(sql.Integer,
                                                 default=0,
                                                 nullable=False)
  edge_features_dimensionality: int = sql.Column(sql.Integer,
                                                 default=0,
                                                 nullable=False)
  graph_features_dimensionality: int = sql.Column(sql.Integer,
                                                  default=0,
                                                  nullable=False)

  node_labels_dimensionality: int = sql.Column(sql.Integer,
                                               default=0,
                                               nullable=False)
  edge_labels_dimensionality: int = sql.Column(sql.Integer,
                                               default=0,
                                               nullable=False)
  graph_labels_dimensionality: int = sql.Column(sql.Integer,
                                                default=0,
                                                nullable=False)

  # The minimum number of message passing steps that are be required to produce
  # the labels from the features. E.g. for graph flooding problems, this value
  # will be the diameter of the graph.
  data_flow_max_steps_required: int = sql.Column(sql.Integer,
                                                 default=0,
                                                 nullable=False)

  date_added: datetime.datetime = sql.Column(
      sql.DateTime().with_variant(mysql.DATETIME(fsp=3), 'mysql'),
      nullable=False,
      default=labdate.GetUtcMillisecondsNow)

  graph: 'Graph' = sql.orm.relationship('Graph',
                                        uselist=False,
                                        back_populates="meta")

  @property
  def pickled_data(self) -> typing.Any:
    return pickle.loads(self.graph.data)

  @classmethod
  def CreateWithGraphDict(cls, g: nx.MultiDiGraph, edge_types: typing.Set[str],
                          **graph_dict_opts):
    """Create a GraphMeta with a corresponding Graph containing a graph dict.

    Args:
      g: The graph to convert to a GraphMeta. Must have the following attributes
       set: bytecode_id, source_name, relpath, language.
      edge_types: The set of edge flow types, e.g. {"control", "flow"}, etc.
      graph_dict_opts: Keyword argument to be passed to ToGraphDict().

    Returns:
      A fully-populated GraphMeta instance.
    """

    # TODO(cec): This implementation only supports a single node type.

    def _FirstFromListOfLists(list_of_lists):
      """Return the first element in a list of lists."""
      for lst in list_of_lists:
        for element in lst:
          return element

    gd = graph_dict.ToGraphDict(g, edge_types, **graph_dict_opts)
    node_features_dimensionality = 0
    if 'node_x' in gd:
      node_features_dimensionality = len(gd['node_x'][0])
    node_labels_dimensionality = 0
    if 'node_y' in gd:
      node_labels_dimensionality = len(gd['node_y'][0])
    edge_features_dimensionality = 0
    if 'edge_x' in gd:
      edge_features_dimensionality = len(_FirstFromListOfLists(gd['edge_x']))
    edge_labels_dimensionality = 0
    if 'edge_y' in gd:
      edge_labels_dimensionality = len(_FirstFromListOfLists(gd['edge_y']))
    graph_features_dimensionality = 0
    if 'graph_x' in gd:
      graph_features_dimensionality = len(gd['graph_x'])
    graph_labels_dimensionality = 0
    if 'graph_y' in gd:
      graph_labels_dimensionality = len(gd['graph_y'])
    data_flow_max_steps_required = getattr(g, 'data_flow_max_steps_required', 0)
    return GraphMeta(
        bytecode_id=g.bytecode_id,
        source_name=g.source_name,
        relpath=g.relpath,
        language=g.language,
        node_count=g.number_of_nodes(),
        edge_count=g.number_of_edges(),
        # Get the number of edge types *after* graph_dict has inserted the
        # backward edges.
        edge_type_count=len(gd['adjacency_lists']),
        node_features_dimensionality=node_features_dimensionality,
        node_labels_dimensionality=node_labels_dimensionality,
        edge_features_dimensionality=edge_features_dimensionality,
        edge_labels_dimensionality=edge_labels_dimensionality,
        graph_features_dimensionality=graph_features_dimensionality,
        graph_labels_dimensionality=graph_labels_dimensionality,
        data_flow_max_steps_required=data_flow_max_steps_required,
        graph=Graph.CreatePickled(gd))


class Graph(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for a graph.

  This is an opaque byte array that can be used as needed, e.g. for pickled
  dictionaries, networkx graphs, etc.
  """
  id: int = sql.Column(sql.Integer,
                       sql.ForeignKey('graph_metas.id'),
                       primary_key=True)
  data: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(), nullable=False)
  meta: GraphMeta = sql.orm.relationship('GraphMeta', back_populates="graph")

  @property
  def pickled_data(self) -> typing.Any:
    return pickle.loads(self.data)

  @classmethod
  def CreatePickled(cls, data: typing.Any) -> 'Graph':
    return Graph(data=pickle.dumps(data))


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)
