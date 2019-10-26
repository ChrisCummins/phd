"""A module for batching graph dictionaries."""
import time

import networkx as nx
import numpy as np
import pickle
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as graph_readers
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.graphs.labelled.graph_dict import \
  graph_dict as graph_dicts
from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS

app.DEFINE_integer(
    "batch_size", 8000,
    "The maximum number of nodes to include in each graph batch.")
app.DEFINE_integer(
    'max_instance_count', None,
    'A debugging option. Use this to set the maximum number of instances used '
    'from training/validation/test files. Note this still requires reading '
    'the entirety of the file contents into memory.')
app.DEFINE_boolean(
    'limit_data_flow_max_steps_required_to_message_passing_steps', True,
    'If true, limit the graphs loaded to those that require fewer data flow '
    'steps than the number of message passing steps. E.g. with '
    '--layer_timesteps=2,2,2, only graphs where data_flow_max_steps_required '
    '<= 6 will be used.')
app.DEFINE_boolean(
    'match_data_flow_max_steps_required_to_message_passing_steps', False,
    'If true, limit the graphs loaded to those that require exactly the '
    'same number of data flow steps as there are message passing steps. E.g. '
    'with --layer_timesteps=2,2,2, only graphs where '
    'data_flow_max_steps_required == 6 will be used.')


class GraphBatcher(object):
  """A generalised graph batcher which flattens adjacency matrices into a single
  adjacency matrix with multiple disconnected components. Supports all feature
  and label types of graph dicts.
  """

  def __init__(self,
               db: graph_database.Database,
               message_passing_step_count: typing.Optional[int] = None):
    """Constructor.

    Args:
      db: The database to read and batch graphs from.
      message_passing_step_count: The number of message passing steps in the
        model that this batcher is feeding. This value is used when the
        --{limit,match}_data_flow_max_steps_required_to_message_passing_steps
        flags are set to limit the graphs which are used to construct batches.
    """
    if ((FLAGS.limit_data_flow_max_steps_required_to_message_passing_steps or
         FLAGS.match_data_flow_max_steps_required_to_message_passing_steps) and
        not message_passing_step_count):
      raise ValueError(
          "message_passing_step_count argument must be provied when "
          "--limit_data_flow_max_steps_required_to_message_passing_steps "
          "or --match_data_flow_max_steps_required_to_message_passing_steps "
          "flags are set")
    self.db = db
    self.message_passing_step_count = message_passing_step_count
    self.stats = graph_stats.GraphDictDatabaseStats(self.db,
                                                    filters=self._GetFilters())
    app.Log(1, "%s", self.stats)

  def GetGraphsInGroupCount(self, group: str) -> int:
    """Get the number of graphs in the given group."""
    with self.db.Session() as s:
      q = s.query(sql.func.count(graph_database.GraphMeta)) \
        .filter(graph_database.GraphMeta.group == group)
      for filter_cb in self._GetFilters():
        q = q.filter(filter_cb())
      num_rows = q.one()[0]
    return num_rows

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
      batch = self.CreateBatchDict(graph_reader)
      if batch:
        elapsed_time = time.time() - start_time
        app.Log(
            1, "Created batch of %s graphs (%s nodes) in %s "
            "(%s graphs/sec)", humanize.Commas(batch['log'].graph_count),
            humanize.Commas(batch['log'].node_count),
            humanize.Duration(elapsed_time),
            humanize.Commas(batch['log'].graph_count / elapsed_time))
        yield batch
      else:
        return

  def _GetFilters(self):
    """Private helper function to return GraphMeta table filters."""
    filters = []
    # Optionally limit the number of steps which are required to compute the
    # graphs.
    if FLAGS.limit_data_flow_max_steps_required_to_message_passing_steps:
      filters.append(
          lambda: (graph_database.GraphMeta.data_flow_max_steps_required <= self
                   .message_passing_step_count))
    if FLAGS.match_data_flow_max_steps_required_to_message_passing_steps:
      filters.append(
          lambda: (graph_database.GraphMeta.data_flow_max_steps_required == self
                   .message_passing_step_count))
    return filters

  def CreateBatchDict(self, graphs: typing.Iterable[graph_database.GraphMeta]
                     ) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Construct a single batch dictionary.

    Args:
      graphs: An iterator of graphs to construct the batch from.

    Returns:
      The batch dictionary. If there are no graphs to batch then None is
      returned.
    """
    graph_ids: typing.List[int] = []

    try:
      graph = next(graphs)
    except StopIteration:  # We have run out of graphs.
      return None

    edge_type_count = self.stats.edge_type_count

    # The batch log contains properties describing the batch (such as the list
    # of graphs used).
    log = log_database.BatchLog(graph_count=0, node_count=0, group=graph.group)

    # Create the empty batch dictionary.
    batch = {
        "adjacency_lists": [[] for _ in range(edge_type_count)],
        "incoming_edge_counts": [],
        "graph_nodes_list": [],
        "log": log,
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
    while graph and log.node_count + graph.node_count < FLAGS.batch_size:
      graph_ids.append(graph.id)

      # De-serialize pickled data in database and process.
      graph_dict = graph.pickled_data

      batch['graph_nodes_list'].append(
          np.full(
              shape=[graph.node_count],
              fill_value=log.graph_count,
              dtype=np.int32,
          ))

      # Offset the adjacency list node indices.
      for edge_type, adjacency_list in enumerate(graph_dict['adjacency_lists']):
        if adjacency_list.size:
          offset = np.array((log.node_count, log.node_count), dtype=np.int32)
          batch['adjacency_lists'][edge_type].append(adjacency_list + offset)

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
      log.graph_count += 1
      log.node_count += graph.node_count

      try:
        graph = next(graphs)
      except StopIteration:  # Nothing left to read from the database.
        break

    # Concatenate and convert lists to numpy arrays.

    for i in range(self.stats.edge_type_count):
      if len(batch['adjacency_lists'][i]):
        batch['adjacency_lists'][i] = np.concatenate(
            batch['adjacency_lists'][i])
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

    batch['graph_count'] = log.graph_count
    batch['node_count'] = log.node_count

    # Record the graphs that we used in this batch.
    log.pickled_graph_indices = pickle.dumps(graph_ids)

    return batch

  @staticmethod
  def BatchDictToGraphs(batch_dict: typing.Dict[str, typing.Any]
                       ) -> typing.Iterable[nx.MultiDiGraph]:
    """Perform the inverse transformation from batch_dict to list of graphs.

    Args:
      batch_dict: The batch dictionary to construct graphs from.

    Returns:
      A generator of graph instances.
    """
    node_count = 0
    for graph_count in range(batch_dict['graph_count']):
      g = nx.MultiDiGraph()
      # Mask the nodes from the node list to determine how many nodes are in
      # the graph.
      nodes = batch_dict['graph_nodes_list'][batch_dict['graph_nodes_list'] ==
                                             graph_count]
      graph_node_count = len(nodes)

      # Make a list of all the adj
      adjacency_lists_indices = []

      for edge_type, adjacency_list in enumerate(batch_dict['adjacency_lists']):
        adjacency_list_indices = []
        adjacency_lists_indices.append(adjacency_list_indices)

        # No edges of this type.
        if not adjacency_list.size:
          continue

        # The adjacency list contains the adjacencies for all graphs. Determine
        # those that are in this graph by selecting only those with a source
        # node in the list of this graph's nodes.
        srcs = adjacency_list[:, 0]
        adjacency_list_indices.extend(
            np.where(
                np.logical_and(srcs >= node_count,
                               srcs < node_count + graph_node_count)))
        adjacency_list = adjacency_list[tuple(adjacency_list_indices)]

        # Negate the positive offset into adjacency lists.
        offset = np.array((node_count, node_count), dtype=np.int32)
        adjacency_list -= offset

        # Add the edges to the graph.
        for src, dst in adjacency_list:
          g.add_edge(src, dst, flow=edge_type)

      if 'node_x' in batch_dict:
        node_x = batch_dict['node_x'][node_count:node_count + graph_node_count]
        if len(node_x) != g.number_of_nodes():
          raise ValueError(f"Graph has {g.number_of_nodes()} nodes but "
                           f"expected {len(node_x)}")
        for i, values in enumerate(node_x):
          g.nodes[i]['x'] = values

      if 'node_y' in batch_dict:
        node_y = batch_dict['node_y'][node_count:node_count + graph_node_count]
        if len(node_y) != g.number_of_nodes():
          raise ValueError(f"Graph has {g.number_of_nodes()} nodes but "
                           f"expected {len(node_y)}")
        for i, values in enumerate(node_y):
          g.nodes[i]['y'] = values

      if 'edge_x' in batch_dict:
        for edge_type, adjacency_list_indices in enumerate(
            adjacency_lists_indices):
          values = batch_dict['edge_x'][edge_type][adjacency_list_indices]
          for (_, _, data), value in zip(g.edges(data=True), values):
            data['x'] = value

      if 'edge_y' in batch_dict:
        for edge_type, adjacency_list_indices in enumerate(
            adjacency_lists_indices):
          values = batch_dict['edge_y'][edge_type][adjacency_list_indices]
          for (_, _, data), value in zip(g.edges(data=True), values):
            data['y'] = value

      if 'graph_x' in batch_dict:
        g.x = batch_dict['graph_x'][graph_count]

      if 'graph_y' in batch_dict:
        g.y = batch_dict['graph_y'][graph_count]

      yield g

      node_count += graph_node_count
