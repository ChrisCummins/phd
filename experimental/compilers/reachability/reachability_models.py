"""Models for learning reachability analysis."""

import collections
import typing

import numpy as np
from graph_nets import utils_tf as graph_net_utils_tf

from experimental.compilers.reachability import control_flow_graph as cfg
from labm8 import app

FLAGS = app.FLAGS

TargetGraphSpec = collections.namedtuple('TargetGraphSpec',
                                         ['graph', 'target_node_index'])

# Functions to generate feature vectors. Features vectors are np.arrays of
# floating point values.


def InputGraphNodeFeatures(spec: TargetGraphSpec, node_index: int) -> np.array:
  """Extract node features for an input graph."""
  # If the node is the target node, the features are [0, 1]. Else, the features
  # are [1, 0].
  return np.array([
      0 if node_index == spec.target_node_index else 1,
      1 if node_index == spec.target_node_index else 0,
  ],
                  dtype=float)


def InputGraphEdgeFeatures(spec: TargetGraphSpec,
                           edge_index: typing.Tuple[int, int]):
  """Extract edge features for an input graph."""
  del spec
  del edge_index
  return np.ones(1, dtype=float)


def TargetGraphNodeFeatures(spec: TargetGraphSpec, node_index: int):
  """Extract node features for a target graph."""
  reachable = spec.graph.IsReachable(spec.target_node_index, node_index)
  # If the node is reachable, the features are [0, 1]. Else, the features
  # are [1, 0].
  return np.array([
      0 if reachable else 1,
      1 if reachable else 0,
  ], dtype=float)


def TargetGraphEdgeFeatures(spec: TargetGraphSpec,
                            edge_index: typing.Tuple[int, int]):
  """Extract edge features for a target graph."""
  del spec
  del edge_index
  return np.ones(1, dtype=float)


def GraphToInputTarget(spec: TargetGraphSpec):
  """Produce two graphs with input and target feature vectors for training.

  A 'features' attributes is added node and edge data, which is a numpy array
  of features describing the node or edge. The shape of arrays is consistent
  across input nodes, input edges, target nodes, and target edges.
  """
  input_graph = spec.graph.copy()
  target_graph = spec.graph.copy()

  # Set node features.
  for node_index in input_graph.nodes():
    input_graph.add_node(
        node_index, features=InputGraphNodeFeatures(spec, node_index))

  for node_index in target_graph.nodes():
    target_graph.add_node(
        node_index, features=TargetGraphNodeFeatures(spec, node_index))

  # Set edge features.
  for edge_index in input_graph.edges():
    input_graph.add_edge(
        *edge_index, features=InputGraphEdgeFeatures(spec, edge_index))

  for edge_index in target_graph.edges():
    target_graph.add_edge(
        *edge_index, features=TargetGraphEdgeFeatures(spec, edge_index))

  # Set global (graph) features.
  input_graph.graph['features'] = np.array([0.0])
  target_graph.graph['features'] = np.array([0.0])

  return input_graph, target_graph


def CreatePlaceholdersFromGraphs(graphs: typing.List[cfg.ControlFlowGraph],
                                 batch_size: int):
  """Creates placeholders for the model training and evaluation.

  Args:
      graphs: A list of graphs that will be inspected for vector sizes.
      batch_size: Total number of graphs per batch.

  Returns:
      A tuple of the input graph's and target graph's placeholders, as a
      graph namedtuple.
  """
  input_graphs, target_graphs = zip(*[GraphToInputTarget(g) for g in graphs])

  input_ph = graph_net_utils_tf.placeholders_from_networkxs(
      input_graphs, force_dynamic_num_graphs=True)
  target_ph = graph_net_utils_tf.placeholders_from_networkxs(
      target_graphs, force_dynamic_num_graphs=True)
  return input_ph, target_ph


def PrintGraphTuple(gt):
  print('nodes', gt.nodes)
  print('edges', gt.edges)
  print('receivers', gt.receivers)
  print('senders', gt.senders)
  print('globals', gt.globals)
  print('n_node', gt.n_node)
  print('n_edge', gt.n_edge)


def MakeRunnableInSession(*args):
  """Lets an iterable of TF graphs be output from a session as NP graphs."""
  return [graph_net_utils_tf.make_runnable_in_session(a) for a in args]


class SpecGenerator(object):

  def __init__(self, graphs: typing.Iterator[cfg.ControlFlowGraph]):
    self._graphs = graphs

  def Generate(self, n: int = 0):
    """Generate specs.

    Args:
        n: The maximum number of spec to generatte. If zero, enumerate
            all graphs.
    """
    i = 0
    for g in self._graphs:
      for nn in range(g.number_of_nodes()):
        i += 1
        if n and i > n:
          return
        yield TargetGraphSpec(graph=g, target_node_index=nn)


class ReachabilityModelBase(object):

  def Fit(self, training_graphs: typing.Iterator[cfg.ControlFlowGraph],
          validation_graphs: typing.Iterator[cfg.ControlFlowGraph]):
    raise NotImplementedError

  def Predict(self, testing_graphs: typing.Iterator[cfg.ControlFlowGraph]):
    raise NotImplementedError


class GraphNetReachabilityModel(ReachabilityModelBase):
  pass
