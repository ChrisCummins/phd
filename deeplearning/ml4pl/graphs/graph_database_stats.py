"""A module for obtaining stats from graph databases."""
import numpy as np
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import decorators
from labm8 import humanize
from labm8 import prof


FLAGS = app.FLAGS


class GraphDatabaseStats(object):
  """Efficient aggregation of graph stats."""

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
  def node_features_dimensionality(self) -> int:
    return self._stats.node_features_dimensionality

  @property
  def edge_features_dimensionality(self) -> int:
    return self._stats.edge_features_dimensionality

  @property
  def graph_features_dimensionality(self) -> int:
    return self._stats.graph_features_dimensionality

  @property
  def node_labels_dimensionality(self) -> int:
    return self._stats.node_labels_dimensionality

  @property
  def edge_labels_dimensionality(self) -> int:
    return self._stats.edge_labels_dimensionality

  @property
  def graph_labels_dimensionality(self) -> int:
    return self._stats.graph_labels_dimensionality

  @property
  def data_flow_max_steps_required(self) -> int:
    return self._stats.data_flow_max_steps_required

  def __repr__(self):
    summaries = [
        f"Graphs database: {humanize.Plural(self.graph_count, 'instance')}",
        humanize.Plural(self.edge_type_count, 'edge type'),
    ]
    if self.node_features_dimensionality:
      summaries.append(
          f"{self.node_features_dimensionality}-d {self.node_features_dtype} "
          "node features")
    if self.edge_features_dimensionality:
      summaries.append(
          f"{self.edge_features_dimensionality}-d {self.edge_features_dtype} "
          "edge features")
    if self.graph_features_dimensionality:
      summaries.append(
          f"{self.graph_features_dimensionality}-d {self.graph_features_dtype} "
          "graph features")
    if self.node_labels_dimensionality:
      summaries.append(
          f"{self.node_labels_dimensionality}-d {self.node_labels_dtype} "
          "node labels")
    if self.edge_labels_dimensionality:
      summaries.append(
          f"{self.edge_labels_dimensionality}-d {self.edge_labels_dtype} "
          "edge labels")
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

  @decorators.memoized_property
  def node_features_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
    return graph_dict['node_x'][0].dtype

  @decorators.memoized_property
  def node_labels_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
    return graph_dict['node_y'][0].dtype

  @decorators.memoized_property
  def edge_features_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
      for features_list in graph_dict['edge_x']:
        if features_list:
          return features_list[0].dtype
    raise ValueError("Unable to determine edge features")

  @decorators.memoized_property
  def edge_labels_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
      for labels_list in graph_dict['edge_y']:
        if labels_list:
          return labels_list[0].dtype
    raise ValueError("Unable to determine edge labels")

  @decorators.memoized_property
  def graph_features_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
    return graph_dict['graph_x'].dtype

  @decorators.memoized_property
  def graph_labels_dtype(self) -> np.dtype:
    with self.db.Session() as s:
      q = s.query(graph_database.Graph).first()
      graph_dict = q.pickled_data
    return graph_dict['graph_y'].dtype

  @decorators.memoized_property
  def _stats(self):
    """Private helper function to compute whole-table stats."""
    graph_count = 0
    label = lambda t: f"Computed stats over {humanize.Commas(graph_count)} instances"
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
              graph_database.GraphMeta.node_features_dimensionality).label(
                  "node_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.edge_features_dimensionality).label(
                  "edge_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.graph_features_dimensionality).label(
                  "graph_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.node_labels_dimensionality).label(
                  "node_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.edge_labels_dimensionality).label(
                  "edge_labels_dimensionality"),
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

    return stats
