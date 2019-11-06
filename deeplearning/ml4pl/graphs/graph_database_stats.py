"""A module for obtaining stats from graph databases."""
import typing

import numpy as np
import sqlalchemy as sql
from labm8 import app
from labm8 import decorators
from labm8 import humanize
from labm8 import prof

from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS


class GraphDatabaseStats(object):
  """Efficient aggregation of graph stats.

  This is a generic class for aggregating stats from GraphMeta tables. If you
  are using pickled graph_tuples in the Graph table, use GraphTupleDatabaseStats
  to obtain additional stats.
  """

  def __init__(
      self,
      db: graph_database.Database,
      filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None):
    self.db = db
    self._filters = filters or []

  @property
  def graph_count(self) -> int:
    return self._stats.graph_count

  @property
  def edge_type_count(self) -> int:
    return self._stats.edge_type_count

  @property
  def max_node_count(self) -> int:
    return self._stats.max_node_count

  @property
  def max_edge_count(self) -> int:
    return self._stats.max_edge_count

  @property
  def node_embeddings_shapes(self) -> int:
    return self._node_embeddings_stats['node_embeddings_shapes']

  @property
  def node_embeddings_dtype(self) -> np.dtype:
    return self._node_embeddings_stats['node_embeddings_dtype']

  @property
  def graph_features_dimensionality(self) -> int:
    return self._stats.graph_features_dimensionality

  @property
  def node_labels_dimensionality(self) -> int:
    return self._stats.node_labels_dimensionality

  @property
  def graph_labels_dimensionality(self) -> int:
    return self._stats.graph_labels_dimensionality

  @property
  def data_flow_max_steps_required(self) -> int:
    return self._stats.data_flow_max_steps_required

  @decorators.memoized_property
  def groups(self) -> typing.List[str]:
    """Fetch a list of distinct group names."""
    with self.db.Session() as session:
      query = session.query(graph_database.GraphMeta.group).distinct()
      return list(sorted([row.group for row in query]))

  def __repr__(self):
    embeddings_shapes = ', '.join(
        [f'{shape[0]}x{shape[1]}' for shape in self.node_embeddings_shapes])
    summaries = [
        f"Graphs database: {humanize.Plural(self.graph_count, 'instance', commas=True)}",
        humanize.Plural(self.edge_type_count, 'edge type'),
        f'({embeddings_shapes}) {self.node_embeddings_dtype} node embeddings',
    ]
    if self.graph_features_dimensionality:
      summaries.append(f"{self.graph_features_dimensionality}-d graph features")
    if self.node_labels_dimensionality:
      summaries.append(f"{self.node_labels_dimensionality}-d node labels")
    if self.graph_labels_dimensionality:
      summaries.append(f"{self.graph_labels_dimensionality}-d graph labels")
    summaries += [
        f"max {humanize.Plural(self.max_node_count, 'node', commas=True)}",
        f"max {humanize.Plural(self.max_edge_count, 'edge', commas=True)}",
    ]
    if self.data_flow_max_steps_required:
      summaries.append(
          f"max {humanize.Plural(self.data_flow_max_steps_required, 'data flow step')}"
      )
    return ", ".join(summaries)

  @decorators.memoized_property
  def _node_embeddings_stats(self):
    # Fetch all embeddings dtypes and assert that they are of the same type.
    embedding_dtypes = [table.dtype for table in self.db.embeddings_tables]
    if len(set(embedding_dtypes)) != 1:
      raise ValueError("Embeddings tables must all have the same dtype. "
                       f"Found {embedding_dtypes}")
    embedding_shapes = [table.shape for table in self.db.embeddings_tables]
    return {
        'node_embeddings_count': len(embedding_shapes),
        'node_embeddings_shapes': embedding_shapes,
        'node_embeddings_dtype': embedding_dtypes[0],
    }

  @decorators.memoized_property
  def _stats(self):
    """Private helper function to compute whole-table stats."""
    graph_count = 0
    label = lambda t: ("Computed stats over "
                       f"{humanize.Commas(graph_count)} instances")
    with prof.Profile(label), self.db.Session() as s:
      q = s.query(
          sql.func.count(graph_database.GraphMeta.id).label("graph_count"),
          sql.func.max(graph_database.GraphMeta.edge_type_count).label(
              "edge_type_count"),
          sql.func.max(
              graph_database.GraphMeta.node_count).label("max_node_count"),
          sql.func.max(
              graph_database.GraphMeta.edge_count).label("max_edge_count"),
          sql.func.max(
              graph_database.GraphMeta.graph_features_dimensionality).label(
                  "graph_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.node_labels_dimensionality).label(
                  "node_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.graph_labels_dimensionality).label(
                  "graph_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.data_flow_max_steps_required).label(
                  "data_flow_max_steps_required"))

      for filter_cb in self._filters:
        q = q.filter(filter_cb())

      stats = q.one()
      graph_count = stats.graph_count

    del graph_count  # Used in prof.Profile() callback.
    return stats


class GraphTupleDatabaseStats(GraphDatabaseStats):
  """Aggregation of stats of databases of graph tuples.

  This stats object is specialized to databases which store pickled graph_tuple
  dictionaries as their Graph.data column. See
  //deeplearning/ml4pl/graphs/labelled/graph_tuple for the schema of graph_tuple
  dictionaries.
  """

  @decorators.memoized_property
  def node_labels_dtype(self) -> np.dtype:
    """Return the numpy dtype of node labels."""
    if not self.node_labels_dimensionality:
      raise ValueError("Trying to access dtype when no node labels")
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.node_y[0].dtype

  @decorators.memoized_property
  def graph_features_dtype(self) -> np.dtype:
    """Return the numpy dtype of graph features."""
    if not self.graph_features_dimensionality:
      raise ValueError("Trying to access dtype when no graph feature")
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.graph_x.dtype

  @decorators.memoized_property
  def graph_labels_dtype(self) -> np.dtype:
    """Return the numpy dtype of graph labels."""
    if not self.graph_labels_dimensionality:
      raise ValueError("Trying to access dtype when no graph labels")
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.graph_y.dtype

  def __repr__(self):
    embeddings_shapes = ', '.join(
        [f'{shape[0]}x{shape[1]}' for shape in self.node_embeddings_shapes])
    summaries = [
        f"Graphs database: {humanize.Plural(self.graph_count, 'instance', commas=True)}",
        humanize.Plural(self.edge_type_count, 'edge type'),
        f'({embeddings_shapes}) {self.node_embeddings_dtype} node embeddings',
    ]
    if self.graph_features_dimensionality:
      summaries.append(
          f"{self.graph_features_dimensionality}-d {self.graph_features_dtype} "
          "graph features")
    if self.node_labels_dimensionality:
      summaries.append(
          f"{self.node_labels_dimensionality}-d {self.node_labels_dtype} "
          "node labels")
    if self.graph_labels_dimensionality:
      summaries.append(
          f"{self.graph_labels_dimensionality}-d {self.graph_labels_dtype} "
          "graph labels")
    if self.data_flow_max_steps_required:
      summaries.append(
          humanize.Plural(self.data_flow_max_steps_required, 'data flow step'))
    summaries += [
        f"max {humanize.Plural(self.max_node_count, 'node')}",
        f"max {humanize.Plural(self.max_edge_count, 'edge')}",
    ]
    return ", ".join(summaries)
