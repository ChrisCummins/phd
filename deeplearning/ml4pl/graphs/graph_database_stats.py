"""A module for obtaining stats from graph databases."""
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import decorators
from labm8 import prof

FLAGS = app.FLAGS


class GraphDatabaseStats(object):
  """Efficient aggregation of graph stats."""

  def __init__(self, db: graph_database.Database):
    self.db = db
    self._edge_type_count = 0
    self._node_features_dimensionality = 0
    self._data_flow_max_steps_required = 0

  @decorators.memoized_property
  def graph_count(self) -> int:
    self._ComputeStats()
    return self._graph_count

  @decorators.memoized_property
  def edge_type_count(self) -> int:
    self._ComputeStats()
    return self._edge_type_count

  @decorators.memoized_property
  def node_features_dimensionality(self) -> int:
    self._ComputeStats()
    return self._node_features_dimensionality

  @decorators.memoized_property
  def node_labels_dimensionality(self) -> int:
    self._ComputeStats()
    return self._node_labels_dimensionality

  @decorators.memoized_property
  def data_flow_max_steps_required(self) -> int:
    self._ComputeStats()
    return self._data_flow_max_steps_required

  def _ComputeStats(self) -> None:
    with prof.Profile("Computed database stats"), self.db.Session() as s:
      q = s.query(
          sql.func.count(graph_database.GraphMeta.id).label("graph_count"),
          sql.func.max(graph_database.GraphMeta.edge_type_count).label(
              "edge_type_count"),
          sql.func.max(
              graph_database.GraphMeta.node_features_dimensionality).label(
                  "node_features_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.node_labels_dimensionality).label(
              "node_labels_dimensionality"),
          sql.func.max(
              graph_database.GraphMeta.data_flow_max_steps_required).label(
                  "data_flow_max_steps_required")).one()

      self._graph_count = q.graph_count
      self._edge_type_count = q.edge_type_count
      self._node_features_dimensionality = q.node_features_dimensionality
      self._node_labels_dimensionality = q.node_labels_dimensionality
      self._data_flow_max_steps_required = q.data_flow_max_steps_required
