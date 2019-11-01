"""A module for batching graph tuples."""
import time

import networkx as nx
import numpy as np
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_reader as graph_readers
from deeplearning.ml4pl.graphs import graph_database_stats as graph_stats
from deeplearning.ml4pl.graphs.labelled.graph_tuple import \
  graph_tuple as graph_tuples
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
    'limit_data_flow_max_steps_required_to_message_passing_steps', False,
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

GraphBatch = typing.Dict[str, typing.Union[int, np.array]]


class GraphBatch(graph_tuples.GraphTuple):
  """An extension to GraphTuple to support multiple disconnected graphs."""
  # A list of shape [node_count] which segments the nodes by graph.
  graph_nodes_list: np.array

  # The number of disconnected graphs in the batch.
  graph_count: int

  # A batch log.
  log: log_database.BatchLog

  def CreateFromGraphMetas(self,
                           graphs: typing.Iterable[graph_database.GraphMeta],
                           stats: graph_stats.GraphTupleDatabaseStats
                          ) -> typing.Optional[graph_tuples.GraphTuple]:
    """Construct a graph batch.

    Args:
      graphs: An iterator of graphs to construct the batch from.

    Returns:
      The graph batch. If there are no graphs to batch then None is returned.
    """
    try:
      graph = next(graphs)
    except StopIteration:  # We have run out of graphs.
      return None

    edge_type_count = stats.edge_type_count

    # The batch log contains properties describing the batch (such as the list
    # of graphs used).
    log = log_database.BatchLog(graph_count=0,
                                node_count=0,
                                group=graph.group,
                                instances=log_database.Instances())

    graph_ids: typing.List[int] = []
    adjacency_lists = [[] for _ in range(edge_type_count)]
    position_lists = [[] for _ in range(edge_type_count)]
    incoming_edge_counts = []
    graph_nodes_list = []
    node_x_indices = []

    has_node_labels = stats.node_labels_dimensionality > 0
    has_graph_features = stats.graph_features_dimensionality > 0
    has_graph_labels = stats.graph_labels_dimensionality > 0

    if has_node_labels:
      node_y = []
    else:
      node_y = None

    if has_graph_features:
      graph_x = []
    else:
      graph_x = None

    if has_graph_labels:
      graph_y = []
    else:
      graph_y = None

    # Pack until we cannot fit more graphs in the batch.
    while graph and log.node_count + graph.node_count < FLAGS.batch_size:
      graph_ids.append(graph.id)

      # De-serialize pickled data in database and process.
      graph_tuple = graph.data

      graph_nodes_list.append(
          np.full(
              shape=[graph.node_count],
              fill_value=log.graph_count,
              dtype=np.int32,
          ))

      # Offset the adjacency list node indices.
      for edge_type, (adjacency_list, position_list) in enumerate(
          zip(graph_tuple.adjacency_lists, graph_tuple.edge_positions)):
        if adjacency_list.size:
          offset = np.array((log.node_count, log.node_count), dtype=np.int32)
          adjacency_lists[edge_type].append(adjacency_list + offset)
          position_lists[edge_type].append(position_list + offset)

      incoming_edge_counts.append(graph_tuple.dense_incoming_edge_counts)

      # Add features and labels.

      # Shape: [graph.node_count, node_features_dimensionality]
      node_x_indices.extend(graph_tuple.node_x_indices)

      if has_node_labels:
        # Shape: [graph_tuple.node_count, node_labels_dimensionality]
        node_y.extend(graph_tuple.node_y)

      if has_graph_features:
        graph_x.append(graph_tuple.graph_x)

      if has_graph_labels:
        graph_y.append(graph_tuple.graph_y)

      # Update batch counters.
      log.graph_count += 1
      log.node_count += graph.node_count

      try:
        graph = next(graphs)
      except StopIteration:  # Nothing left to read from the database.
        break

    # Concatenate and convert lists to numpy arrays.

    for i in range(stats.edge_type_count):
      if len(adjacency_lists[i]):
        adjacency_lists[i] = np.concatenate(adjacency_lists[i])
      else:
        adjacency_lists[i] = np.zeros((0, 2), dtype=np.int32)

      if len(position_lists[i]):
        position_lists[i] = np.concatenate(position_lists[i])
      else:
        position_lists[i] = np.array([], dtype=np.int32)

    incoming_edge_counts = np.concatenate(incoming_edge_counts, axis=0)
    graph_nodes_list = np.concatenate(graph_nodes_list)
    node_x_indices = np.array(node_x_indices)
    if has_node_labels:
      node_y = np.array(node_y)
    if has_graph_features:
      graph_x = np.array(graph_x)
    if has_graph_labels:
      graph_y = np.array(graph_y)

    # Record the graphs that we used in this batch.
    log.graph_indices = graph_ids

    return GraphBatch(
        adjacency_lists=adjacency_lists,
        edge_positions=position_lists,
        incoming_edge_counts=incoming_edge_counts,
        node_x_indices=node_x_indices,
        node_y=node_y,
        graph_x=graph_x,
        graph_y=graph_y,
        graph_nodes_list=graph_nodes_list,
        graph_count=log.graph_count,
        log=log,
    )

  def ToNetworkXGraphs(self) -> typing.Iterable[nx.MultiDiGraph]:
    """Perform the inverse transformation from batch_dict to list of graphs.

    Args:
      batch_dict: The batch dictionary to construct graphs from.

    Returns:
      A generator of graph instances.
    """
    node_count = 0
    for graph_count in range(self.graph_count):
      g = nx.MultiDiGraph()
      # Mask the nodes from the node list to determine how many nodes are in
      # the graph.
      nodes = self.graph_nodes_list[self.graph_nodes_list == graph_count]
      graph_node_count = len(nodes)

      # Make a list of all the adj
      adjacency_lists_indices = []

      for edge_type, (adjacency_list, position_list) in enumerate(
          self.adjacency_lists, self.edge_positions):
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
        position_list = position_list[tuple(adjacency_lists_indices)]

        # Negate the positive offset into adjacency lists.
        offset = np.array((node_count, node_count), dtype=np.int32)
        adjacency_list -= offset

        # Add the edges to the graph.
        for (src, dst), position in zip(adjacency_list, position_list):
          g.add_edge(src, dst, flow=edge_type, position=position)

      node_x = self.node_x_indices[node_count:node_count + graph_node_count]
      if len(node_x) != g.number_of_nodes():
        raise ValueError(f"Graph has {g.number_of_nodes()} nodes but "
                         f"expected {len(node_x)}")
      for i, values in enumerate(node_x):
        g.nodes[i]['x'] = values

      if self.has_node_y:
        node_y = self.node_y[node_count:node_count + graph_node_count]
        if len(node_y) != g.number_of_nodes():
          raise ValueError(f"Graph has {g.number_of_nodes()} nodes but "
                           f"expected {len(node_y)}")
        for i, values in enumerate(node_y):
          g.nodes[i]['y'] = values

      if self.has_graph_x:
        g.x = self.graph_x[graph_count]

      if self.has_graph_y:
        g.y = self.graph_y[graph_count]

      yield g

      node_count += graph_node_count


class GraphBatcher(object):
  """A generalised graph batcher which flattens adjacency matrices into a single
  adjacency matrix with multiple disconnected components. Supports all feature
  and label types of graph tuples.
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
          "`message_passing_step_count` argument must be provied when "
          "--limit_data_flow_max_steps_required_to_message_passing_steps "
          "or --match_data_flow_max_steps_required_to_message_passing_steps "
          "flags are set")
    self.db = db
    self.message_passing_step_count = message_passing_step_count
    self.stats = graph_stats.GraphTupleDatabaseStats(self.db,
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

  def MakeGaphBatchIterator(self, group: str) -> typing.Iterable[GraphBatch]:
    """Make a batch iterator over the given group."""
    filters = self._GetFilters()
    filters.append(lambda: graph_database.GraphMeta.group == group)

    graph_reader = graph_readers.BufferedGraphReader(
        self.db,
        filters=filters,
        order_by_random=True,
        eager_graph_loading=True,
        buffer_size=max(min(64, FLAGS.batch_size // 100), 16),
        limit=FLAGS.max_instance_count)

    # Batch creation outer-loop.
    while True:
      start_time = time.time()
      batch = GraphBatch.CreateFromGraphMetas(graph_reader, self.stats)
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
