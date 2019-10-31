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
  def node_embedding_count(self) -> int:
    return self._embedding_stats.node_embedding_count

  @property
  def node_embedding_dimensionality(self) -> int:
    return self._embedding_stats.node_embedding_dimensionality

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

  def __repr__(self):
    summaries = [
        f"Graphs database: {humanize.Plural(self.graph_count, 'instance', commas=True)}",
        humanize.Plural(self.edge_type_count, 'edge type'),
        (f"{self.node_embedding_count}x{self.node_embedding_dimensionality} "
         f"{self.node_embedding_dtype} node embeddings")
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
  def node_embedding_dtype(self) -> np.dtype:
    """Return the numpy dtype of node embeddings."""
    with self.db.Session() as s:
      embedding_table = s.query(graph_database.EmbeddingTable).one()
    return embedding_table.embedding_table.dtype

  @decorators.memoized_property
  def _embedding_stats(self):
    with self.db.Session() as s:
      q = s.query(
          graph_database.EmbeddingTable.embedding_count.label(
              'node_embedding_count'),
          graph_database.EmbeddingTable.embedding_dimensionality.label(
              'node_embedding_dimensionality'))
      if q.count() != 1:
        raise ValueError(
            f"Expected a single embedding table, found {q.count()}")
      return q.one()

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
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.node_y[0].dtype

  @decorators.memoized_property
  def graph_features_dtype(self) -> np.dtype:
    """Return the numpy dtype of graph features."""
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.graph_x.dtype

  @decorators.memoized_property
  def graph_labels_dtype(self) -> np.dtype:
    """Return the numpy dtype of graph labels."""
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_tuple = q.data
    return graph_tuple.graph_y.dtype

  def __repr__(self):
    summaries = [
        f"Graphs database: {humanize.Plural(self.graph_count, 'instance')}",
        humanize.Plural(self.edge_type_count, 'edge type'),
        (f"{self.node_embedding_count}x{self.node_embedding_dimensionality} "
         f"{self.node_embedding_dtype} node embeddings")
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
