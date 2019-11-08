"""Database backend for GGNN graphs."""
import datetime
import pickle
import typing

import networkx as nx
import numpy as np
import sqlalchemy as sql
from labm8 import app
from labm8 import bazelutil
from labm8 import decorators
from labm8 import labdate
from labm8 import sqlutil
from sqlalchemy.dialects import mysql
from sqlalchemy.ext import declarative

from deeplearning.ml4pl.graphs.labelled.graph_tuple import \
  graph_tuple as graph_tuples

FLAGS = app.FLAGS

EMBEDDINGS = bazelutil.DataPath(
    'phd/deeplearning/ml4pl/graphs/unlabelled/cdfg/node_embeddings/inst2vec_augmented_embeddings.pickle'
)

Base = declarative.declarative_base()


class Meta(Base, sqlutil.TablenameFromClassNameMixin):
  """Key-value database metadata store."""
  key: str = sql.Column(sql.String(64), primary_key=True)
  value: str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(),
                          nullable=False)


class GraphMeta(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """A table of graph metadata.

  For every GraphMeta, there should be a corresponding Graph row containing the
  actual data as pickled blob.
  """
  id: int = sql.Column(sql.Integer, primary_key=True)

  # A string name to group graphs into discrete buckets, e.g. "train", "test",
  # or "1", "2", ... k for k-fold classification.
  group: str = sql.Column(sql.String(32), nullable=False, index=True)

  # No foreign key relationship because they are separate databases.
  bytecode_id: int = sql.Column(sql.Integer, nullable=False, index=True)

  # The source of the graph. This is duplicates of information stored in the
  # LlvmBytecode table.
  source_name: str = sql.Column(sql.String(256), nullable=False)
  relpath: str = sql.Column(sql.String(256), nullable=False)
  language: str = sql.Column(sql.String(16), nullable=False)

  node_count: int = sql.Column(sql.Integer, nullable=False)
  edge_count: int = sql.Column(sql.Integer, nullable=False)
  node_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)
  edge_type_count: int = sql.Column(sql.Integer, default=1, nullable=False)

  # The maximum value of the 'position' attribute of edges.
  edge_position_max: int = sql.Column(sql.Integer, nullable=False)

  # The number of distinct embeddings for each node.
  node_embeddings_count: int = sql.Column(sql.Integer,
                                          default=0,
                                          nullable=False)
  node_labels_dimensionality: int = sql.Column(sql.Integer,
                                               default=0,
                                               nullable=False)
  graph_features_dimensionality: int = sql.Column(sql.Integer,
                                                  default=0,
                                                  nullable=False)
  graph_labels_dimensionality: int = sql.Column(sql.Integer,
                                                default=0,
                                                nullable=False)

  # The loop connectedness (loop depth) of the graph. This is the largest number
  # of back edges found in any cycle-free path of the full flow graph.
  loop_connectedness: int = sql.Column(sql.Integer, nullable=False)

  # The diameter of the graph is the maximum eccentricity, where eccentricity
  # is the maximum distance from one node to all other nodes.
  undirected_diameter: int = sql.Column(sql.Integer, nullable=False)

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
                                        back_populates="meta",
                                        cascade="all")

  @property
  def data(self) -> typing.Any:
    """Load the pickled data."""
    return self.graph.data

  @classmethod
  def CreateFromNetworkX(cls, g: nx.MultiDiGraph, **graph_tuple_opts):
    """Create a GraphMeta with a corresponding Graph containing a graph tuple.

    Args:
      g: The graph to convert to a GraphMeta. Must have the following attributes
       set: bytecode_id, source_name, relpath, language.
      graph_tuple_opts: Keyword argument to be passed to CreateFromNetworkX().

    Returns:
      A fully-populated GraphMeta instance.
    """
    graph_tuple = graph_tuples.GraphTuple.CreateFromNetworkX(
        g, **graph_tuple_opts)
    node_embeddings_count = len(graph_tuple.node_x_indices[0])
    node_labels_dimensionality = (len(graph_tuple.node_y[0])
                                  if graph_tuple.has_node_y else 0)
    graph_features_dimensionality = (len(graph_tuple.graph_x)
                                     if graph_tuple.has_graph_x else 0)
    graph_labels_dimensionality = (len(graph_tuple.graph_y)
                                   if graph_tuple.has_graph_y else 0)

    data_flow_max_steps_required = getattr(g, 'data_flow_max_steps_required', 0)

    edge_position_max = 0
    for src, dst, position in g.edges(data='position', default=0):
      edge_position_max = max(edge_position_max, position)

    return GraphMeta(
        group=getattr(g, 'group', None),
        bytecode_id=g.bytecode_id,
        source_name=g.source_name,
        relpath=g.relpath,
        language=g.language,
        node_count=len(graph_tuple.node_x_indices),
        # Get the edge stats *after* graph_tuple has inserted the
        # backward edges.
        edge_count=sum([len(a) for a in graph_tuple.adjacency_lists]),
        edge_type_count=len(graph_tuple.adjacency_lists),
        edge_position_max=edge_position_max,
        node_embeddings_count=node_embeddings_count,
        node_labels_dimensionality=node_labels_dimensionality,
        graph_features_dimensionality=graph_features_dimensionality,
        graph_labels_dimensionality=graph_labels_dimensionality,
        # TODO(github.com/ChrisCummins/ml4pl/issues/5): Compute loop stats.
        loop_connectedness=0,
        undirected_diameter=0,
        # loop_connectedness=query.LoopConnectedness(g),
        # undirected_diameter=nx.diameter(g.to_undirected()),
        data_flow_max_steps_required=data_flow_max_steps_required,
        graph=Graph.CreateFromPickled(graph_tuple))

  @classmethod
  def CreateWithNetworkXGraph(cls, g: nx.MultiDiGraph):
    """Create a GraphMeta with a corresponding Graph containing a graph tuple.

    Args:
      g: The graph to convert to a GraphMeta. Must have the following attributes
       set: bytecode_id, source_name, relpath, language.

    Returns:
      A fully-populated GraphMeta instance.
    """
    for node in g.nodes():
      break

    edge_types = set()
    edge_position_max = 0
    for src, dst, data in g.edges(data=True):
      edge_position_max = max(edge_position_max, data.get('position', 0))
      edge_types.add(data.get('flow', 'control'))

    return GraphMeta(
        group=getattr(g, 'group', None),
        bytecode_id=g.bytecode_id,
        source_name=g.source_name,
        relpath=g.relpath,
        language=g.language,
        node_count=g.number_of_nodes(),
        edge_count=g.number_of_edges(),
        edge_type_count=len(edge_types),
        edge_position_max=edge_position_max,
        node_embeddings_count=len(g.nodes[node]['x']),
        node_labels_dimensionality=(len(g.nodes[node]['y'])
                                    if 'y' in g.nodes[node] else 0),
        graph_features_dimensionality=getattr(g, 'x', 0),
        graph_labels_dimensionality=getattr(g, 'y', 0),
        # TODO(github.com/ChrisCummins/ml4pl/issues/5): Compute loop stats.
        loop_connectedness=0,
        undirected_diameter=0,
        # loop_connectedness=query.LoopConnectedness(g),
        # undirected_diameter=nx.diameter(g.to_undirected()),
        data_flow_max_steps_required=getattr(g, 'data_flow_max_steps_required',
                                             0),
        graph=Graph.CreateFromPickled(g))


class Graph(Base, sqlutil.PluralTablenameFromCamelCapsClassNameMixin):
  """The data for a graph.

  This is an opaque byte array that can be used as needed, e.g. for pickled
  graph tuples, networkx graphs, etc.
  """
  id: int = sql.Column(sql.Integer,
                       sql.ForeignKey('graph_metas.id'),
                       primary_key=True)
  pickled_data: bytes = sql.Column(sqlutil.ColumnTypes.LargeBinary(),
                                   nullable=False)
  meta: GraphMeta = sql.orm.relationship('GraphMeta',
                                         back_populates="graph",
                                         uselist=False,
                                         cascade="all")

  @property
  def data(self) -> typing.Any:
    return pickle.loads(self.pickled_data)

  @classmethod
  def CreateFromPickled(cls, data: typing.Any) -> 'Graph':
    return Graph(pickled_data=pickle.dumps(data))


class Database(sqlutil.Database):

  def __init__(self, url: str, must_exist: bool = False):
    super(Database, self).__init__(url, Base, must_exist=must_exist)

  @decorators.memoized_property
  def embeddings_tables(self) -> typing.List[np.array]:
    """Return the embeddings tables."""
    # TODO(github.com/ChrisCummins/ml4pl/issues/12): In the future we may want
    # to add support for different numbers of embeddings tables, or embeddings
    # tables with different types. This is hardcoded to support only two
    # embeddings tables: our augmented inst2vec statement embeddings, and
    # a binary 'selector' table which can be used to select one or more nodes
    # of interests in graphs, e.g. for setting the starting point for computing
    # iterative data flow analyses.
    with open(EMBEDDINGS, 'rb') as f:
      augmented_inst2vec_embeddings = pickle.load(f)

    node_selector = np.vstack([
        [1, 0],
        [0, 1],
    ]).astype(np.float64)

    return augmented_inst2vec_embeddings, node_selector

  def DeleteGraphs(self, graph_ids: typing.List[int]) -> None:
    """Delete the logs for this run.

    This deletes the batch logs, model checkpoints, and model parameters.

    Args:
      run_id: The ID of the run to delete.
    """
    graph_ids = list(graph_ids)
    # Because the cascaded delete is broken, we first delete the Graph rows,
    # then the GraphMetas.
    with self.Session() as session:
      query = session.query(GraphMeta.id) \
        .filter(GraphMeta.id.in_(graph_ids))
      ids_to_delete = [row.id for row in query]

    app.Log(1, "Deleting %s graphs", humanize.Commas(len(ids_to_delete)))
    delete = sql.delete(Graph) \
      .where(Graph.id.in_(ids_to_delete))
    self.engine.execute(delete)

    delete = sql.delete(GraphMeta) \
      .where(GraphMeta.id.in_(ids_to_delete))
    self.engine.execute(delete)
