"""A module for batching graph dictionaries."""
import time

import numpy as np
import typing

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as graph_readers
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.graphs.labelled.graph_dict import \
  graph_dict as graph_dicts
from labm8 import app
from labm8 import humanize


FLAGS = app.FLAGS

app.DEFINE_integer(
    "batch_size", 8000,
    "The maximum number of nodes to include in each graph batch.")
app.DEFINE_boolean(
    'limit_data_flow_max_steps_required_to_message_passing_steps', True,
    'If true, limit the graphs loaded to those that require fewere data flow '
    'steps than the number of layer timesteps. E.g. with '
    '--layer_timesteps=2,2,2, only graphs where data_flow_max_steps_required '
    '<= 6 will be used.')

class GraphBatcher(object):
  """A generalised graph batcher which flattens adjacency matrics into a single
  adjacency matric with multiple disconnected components. Supports all feature
  and label types of graph dicts.
  """

  def __init__(self, db: graph_database.Database,
               message_passing_step_count: int):
    self.db = db
    self.message_passing_step_count = message_passing_step_count
    self.stats = graph_stats.GraphDatabaseStats(
        self.db, filters=self._GetFilters())
    app.Log(1, "%s", self.stats)

  def MakeGroupBatchIterator(
      self, group: str
  ) -> typing.Iterable[typing.Dict[str, typing.Union[int, np.array]]]:
    """Make a batch iterator over the given group."""
    filters = self._GetFilters()
    filters.append(lambda: graph_database.GraphMeta.group == group)

    graph_reader = graph_readers.BufferedGraphReader(
        self.db,
        filters=filters,
        order_by_random=True,
        eager_graph_loading=True,
        buffer_size=min(512, FLAGS.batch_size // 10),
        limit=FLAGS.max_instance_count)

    # Batch creation outer-loop.
    while True:
      start_time = time.time()
      batch = self._CreateBatchDict(graph_reader)
      if batch:
        elapsed_time = time.time() - start_time
        app.Log(1, "Created batch of %s graphs in %s (%.2f graphs / second)",
                humanize.Commas(batch['graph_count']),
                humanize.Duration(elapsed_time),
                batch['graph_count'] / elapsed_time)
        yield batch
      else:
        return

  def _GetFilters(self):
    filters = []
    # Optionally limit the number of steps which are required to compute the
    # graphs.
    if FLAGS.limit_data_flow_max_steps_required_to_message_passing_steps:
      filters.append(
          lambda: (graph_database.GraphMeta.data_flow_max_steps_required
                   <= self.message_passing_step_count))
    return filters

  def _CreateBatchDict(
      self, graphs: typing.Iterable[graph_database.GraphMeta]
  ) -> typing.Optional[typing.Dict[str, typing.Union[int, np.array]]]:
    """Construct a single batch dictionary.

    Args:
      graphs: An iterator of graphs to construct the batch from.

    Returns:
      An batch dictionary, unless there are no graphs to batch, in which case
      it returns None.
    """
    try:
      graph = next(graphs)
    except StopIteration:  # We have run out of graphs.
      return None

    edge_type_count = self.stats.edge_type_count

    # Create the empty batch dictionary.
    batch = {
      "adjacency_lists": [[] for _ in range(edge_type_count)],
      "incoming_edge_counts": [],
      "graph_nodes_list": [],
      "graph_count": 0,
      "node_count": 0,
    }

    if self.stats.node_features_dimensionality:
      batch['node_x'] = []

    if self.stats.node_labels_dimensionality:
      batch['node_y'] = []

    if self.stats.edge_features_dimensionality:
      batch['edge_x'] = [[] for _ in range(edge_type_count)]

    if self.stats.edge_labels_dimensionality:
      batch['edge_y'] = [[] for _ in range(edge_type_count)]

    if self.stats.graph_features_dimensionality:
      batch['graph_x'] = []

    if self.stats.graph_labels_dimensionality:
      batch['graph_y'] = []

    # Pack until we cannot fit more graphs in the batch.
    while graph and batch['node_count'] + graph.node_count < FLAGS.batch_size:
      # De-serialize pickled data in database and process.
      graph_dict = graph.pickled_data

      batch['graph_nodes_list'].append(
          np.full(
              shape=[graph.node_count],
              fill_value=batch['graph_count'],
              dtype=np.int32,
          ))

      # Offset the adjacency list node indices.
      for i, adjacency_list in enumerate(graph_dict['adjacency_lists']):
        if adjacency_list.size:
          batch['adjacency_lists'][i].append(
              adjacency_list + np.array((batch['node_count'], batch['node_count']),
                                        dtype=np.int32))

      # Turn counters for incoming edges into a dense array:
      batch['incoming_edge_counts'].append(
          graph_dicts.IncomingEdgeCountsToDense(
              graph_dict["incoming_edge_counts"],
              node_count=graph.node_count,
              edge_type_count=self.stats.edge_type_count))

      # Add features and labels.

      if 'node_x' in batch:
        # TODO(cec): Specialized to GGNN.
        #
        # Pad node feature vector of size <= hidden_size up to hidden_size so
        # that the size matches embedding dimensionality.
        padded_features = np.pad(
            graph_dict["node_x"],
            ((0, 0),
             (0, FLAGS.hidden_size - self.stats.node_features_dimensionality)),
            "constant",
        )
        # Shape: [graph.node_count, node_features_dimensionality]
        batch['node_x'].extend(padded_features)

      if 'node_y' in batch:
        # Shape: [graph.node_count, node_labels_dimensionality]
        batch['node_y'].extend(graph_dict['node_y'])

      if 'edge_x' in batch:
        for i, feature_list in enumerate(graph_dict['edge_x']):
          if feature_list.size:
            batch['edge_x'][i].append(feature_list)

      if 'edge_y' in batch:
        for i, label_list in enumerate(graph_dict['edge_y']):
          if label_list.size:
            batch['edge_y'][i].append(label_list)

      if 'graph_x' in batch:
        batch['graph_x'].append(graph_dict['graph_x'])

      if 'graph_y' in batch:
        batch['graph_y'].append(graph_dict['graph_y'])

      # Update batch counters.
      batch['graph_count'] += 1
      batch['node_count'] += graph.node_count

      try:
        graph = next(graphs)
      except StopIteration:  # Nothing left to read from the database.
        break

    # Concatenate and convert lists to numpy arrays.

    for i in range(self.stats.edge_type_count):
      if len(batch['adjacency_lists'][i]):
        batch['adjacency_lists'][i] = np.concatenate(batch['adjacency_lists'][i])
      else:
        batch['adjacency_lists'][i] = np.zeros((0, 2), dtype=np.int32)

    batch['incoming_edge_counts'] = np.concatenate(
        batch['incoming_edge_counts'], axis=0)

    batch['graph_nodes_list'] = np.concatenate(batch['graph_nodes_list'])

    if 'node_x' in batch:
      batch['node_x'] = np.array(batch['node_x'])

    if 'node_y' in batch:
      batch['node_y'] = np.array(batch['node_y'])

    if 'edge_x' in batch:
      for i in range(self.stats.edge_type_count):
        if len(batch['edge_x'][i]):
          batch['edge_x'][i] = np.concatenate(batch['edge_x'][i])
        else:
          batch['edge_x'][i] = np.zeros(
              (0, self.stats.edge_features_dimensionality), dtype=np.int32)

    if 'edge_y' in batch:
      for i in range(self.stats.edge_type_count):
        if len(batch['edge_y'][i]):
          batch['edge_y'][i] = np.concatenate(batch['edge_y'][i])
        else:
          batch['edge_y'][i] = np.zeros(
              (0, self.stats.edge_features_dimensionality), dtype=np.int32)

    if 'graph_x' in batch:
      batch['graph_x'] = np.array(batch['graph_x'])

    if 'graph_y' in batch:
      batch['graph_y'] = np.array(batch['graph_y'])

    return batch
