"""Graph utilities for the shortest path model."""
import collections

import networkx as nx
import numpy as np
from absl import flags
from scipy import spatial

from labm8 import labtypes


FLAGS = flags.FLAGS


def GenerateGraph(
    rand: np.random.RandomState,
    num_nodes_min_max, dimensions=2,
    theta=1000.0, rate=1.0, weight_name: str = "distance") -> nx.Graph:
  """Creates a connected graph.

  The graphs are geographic threshold graphs, but with added edges via a
  minimum spanning tree algorithm, to ensure all nodes are connected.

  Args:
    rand: A random seed for the graph generator.
    num_nodes_min_max: A sequence [lower, upper) number of nodes per graph.
    dimensions: (optional) An `int` number of dimensions for the positions.
      Default= 2.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Large values (1000+) make mostly trees. Try
      20-60 for good non-trees. Default=1000.0.
    rate: (optional) A rate parameter for the node weight exponential sampling
      distribution. Default= 1.0.
    weight_name: The name for the weight edge attribute.

  Returns:
    The graph.
  """
  # Sample num_nodes.
  num_nodes = rand.randint(*num_nodes_min_max)

  # Create geographic threshold graph.
  pos_array = rand.uniform(size=(num_nodes, dimensions))
  pos = dict(enumerate(pos_array))
  weight = dict(enumerate(rand.exponential(rate, size=num_nodes)))
  geo_graph = nx.geographical_threshold_graph(
      num_nodes, theta, pos=pos, weight=weight)

  # Create minimum spanning tree across geo_graph's nodes.
  distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
  i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
  weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
  mst_graph = nx.Graph()
  mst_graph.add_weighted_edges_from(weighted_edges, weight=weight_name)
  mst_graph = nx.minimum_spanning_tree(mst_graph, weight=weight_name)
  # Put geo_graph's node attributes into the mst_graph.
  for i in mst_graph.nodes():
    mst_graph.node[i].update(geo_graph.node[i])

  # Compose the graphs.
  combined_graph = nx.compose_all((mst_graph, geo_graph.copy()))
  # Put all distance weights into edge attributes.
  for i, j in combined_graph.edges():
    combined_graph.get_edge_data(i, j).setdefault(weight_name,
                                                  distances[i, j])
  return combined_graph


def AddShortestPath(rand, graph, min_length=1, weight_name: str = "distance"):
  """Samples a shortest path from A to B and adds attributes to indicate it.

  Args:
    rand: A random seed for the graph generator. Default= None.
    graph: A `nx.Graph`.
    min_length: (optional) An `int` minimum number of edges in the shortest
      path. Default= 1.

  Returns:
    The `nx.DiGraph` with the shortest path added.

  Raises:
    ValueError: All shortest paths are below the minimum length
  """
  # Map from node pairs to the length of their shortest path.
  pair_to_length_dict = {}
  lengths = list(nx.all_pairs_shortest_path_length(graph))
  for x, yy in lengths:
    for y, l in yy.items():
      if l >= min_length:
        pair_to_length_dict[x, y] = l
  if not pair_to_length_dict:
    raise ValueError("All shortest paths are below the minimum length")
  # The node pairs which exceed the minimum length.
  node_pairs = list(pair_to_length_dict)

  # Computes probabilities per pair, to enforce uniform sampling of each
  # shortest path lengths.
  # The counts of pairs per length.
  counts = collections.Counter(pair_to_length_dict.values())
  prob_per_length = 1.0 / len(counts)
  probabilities = [
    prob_per_length / counts[pair_to_length_dict[x]] for x in node_pairs
  ]

  # Choose the start and end points.
  i = rand.choice(len(node_pairs), p=probabilities)
  start, end = node_pairs[i]
  path = nx.shortest_path(
      graph, source=start, target=end, weight=weight_name)
  print(path)

  # Creates a directed graph, to store the directed path from start to end.
  directed_graph = graph.to_directed()

  # Add the "start", "end", and "solution" attributes to the nodes.
  directed_graph.add_node(start, start=True)
  directed_graph.add_node(end, end=True)
  directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), [start])), start=False)
  directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), [end])), end=False)
  directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), path)), solution=False)
  directed_graph.add_nodes_from(path, solution=True)

  # Now do the same for the edges.
  path_edges = list(labtypes.PairwiseIterator(path))
  directed_graph.add_edges_from(
      list(labtypes.SetDiff(directed_graph.edges(), path_edges)),
      solution=False)
  directed_graph.add_edges_from(path_edges, solution=True)

  return directed_graph
