"""Graph utilities for the shortest path model."""
import collections
import typing

import networkx as nx
import numpy as np
from scipy import spatial

from labm8.py import app
from labm8.py import labtypes

FLAGS = app.FLAGS


class GraphGeneratorOptions(object):
  """Options for graph generator.

  Attributes:
    num_nodes_min_max (Tuple[int, int]): A 2-tuple range [lower, upper) for the
      number of nodes per graph.
    min_shortest_path_length (int): The minimum number of edges in the shortest
      path.
    dimensions (int): The number of dimensions for graph positions.
    theta (float): A threshold parameter for the geographic threshold graph's
      threshold. Large values (1000+) make mostly trees. Try 20-60 for good
      non-trees.
    rate (float): A rate parameter for the node weight exponential sampling
      distribution.
    weight_name (str): The name for the weight edge attribute.
  """

  num_nodes_min_max = [10, 12]
  min_shortest_path_length = 1
  dimensions = 2
  theta = 20
  rate = 1
  weight_name = "distance"


class GraphGenerator(object):
  def __init__(
    self, rand: np.random.RandomState, options: GraphGeneratorOptions
  ):
    self._rand = rand
    self._options = options

  @property
  def rand(self) -> np.random.RandomState:
    return self._rand

  @rand.setter
  def rand(self, rand: np.random.RandomState) -> None:
    """Set a new random state."""
    self._rand = rand

  @property
  def options(self) -> GraphGeneratorOptions:
    return self._options

  def GenerateGraph(self) -> nx.Graph:
    """Creates a connected graph.

    The graphs are geographic threshold graphs, but with added edges via a
    minimum spanning tree algorithm, to ensure all nodes are connected.

    Returns:
      The graph.
    """
    # Sample num_nodes.
    num_nodes = self.rand.randint(*self.options.num_nodes_min_max)

    # Create geographic threshold graph.
    pos_array = self.rand.uniform(size=(num_nodes, self.options.dimensions))
    pos = dict(enumerate(pos_array))
    weight = dict(
      enumerate(self.rand.exponential(self.options.rate, size=num_nodes))
    )
    geo_graph = nx.geographical_threshold_graph(
      num_nodes, self.options.theta, pos=pos, weight=weight
    )

    # Create minimum spanning tree across geo_graph's nodes.
    distances = spatial.distance.squareform(spatial.distance.pdist(pos_array))
    i_, j_ = np.meshgrid(range(num_nodes), range(num_nodes), indexing="ij")
    weighted_edges = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    mst_graph = nx.Graph()
    mst_graph.add_weighted_edges_from(
      weighted_edges, weight=self.options.weight_name
    )
    mst_graph = nx.minimum_spanning_tree(
      mst_graph, weight=self.options.weight_name
    )
    # Put geo_graph's node attributes into the mst_graph.
    for i in mst_graph.nodes():
      mst_graph.node[i].update(geo_graph.node[i])

    # Compose the graphs.
    combined_graph = nx.compose_all([mst_graph, geo_graph.copy()])
    # Put all distance weights into edge attributes.
    for i, j in combined_graph.edges():
      combined_graph.get_edge_data(i, j).setdefault(
        self.options.weight_name, distances[i, j]
      )
    return combined_graph

  def AddShortestPath(self, graph: nx.Graph) -> nx.DiGraph:
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
        if l >= self.options.min_shortest_path_length:
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
    i = self.rand.choice(len(node_pairs), p=probabilities)
    start, end = node_pairs[i]
    path = nx.shortest_path(graph, source=start, target=end, weight=weight_name)

    # Creates a directed graph, to store the directed path from start to end.
    directed_graph = graph.to_directed()

    # Add the "start", "end", and "solution" attributes to the nodes.
    directed_graph.add_node(start, start=True)
    directed_graph.add_node(end, end=True)
    directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), [start])), start=False
    )
    directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), [end])), end=False
    )
    directed_graph.add_nodes_from(
      list(labtypes.SetDiff(directed_graph.nodes(), path)), solution=False
    )
    directed_graph.add_nodes_from(path, solution=True)

    # Now do the same for the edges.
    path_edges = list(labtypes.PairwiseIterator(path))
    directed_graph.add_edges_from(
      list(labtypes.SetDiff(directed_graph.edges(), path_edges)), solution=False
    )
    directed_graph.add_edges_from(path_edges, solution=True)

    return directed_graph

  @staticmethod
  def CreateFeature(
    data_dict: typing.Dict[str, typing.Any], feature_names: typing.List[str]
  ):
    return np.hstack(
      [np.array(data_dict[feature], dtype=float) for feature in feature_names]
    )

  @staticmethod
  def ToOneHot(indices: typing.Iterator[int], max_value: int, axis: int = -1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
      one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot

  @classmethod
  def GraphToInputTarget(
    cls, graph: nx.DiGraph
  ) -> typing.Tuple[nx.DiGraph, nx.DiGraph]:
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
      graph: An `nx.DiGraph` instance.

    Returns:
      The input `nx.DiGraph` instance.
      The target `nx.DiGraph` instance.

    Raises:
      ValueError: unknown node type
    """
    input_node_fields = ("pos", "weight", "start", "end")
    input_edge_fields = ("distance",)
    target_node_fields = ("solution",)
    target_edge_fields = ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    solution_length = 0
    # Set node features.
    for node_index, node_feature in graph.nodes(data=True):
      input_graph.add_node(
        node_index, features=cls.CreateFeature(node_feature, input_node_fields)
      )
      target_node = cls.ToOneHot(
        cls.CreateFeature(node_feature, target_node_fields).astype(int), 2
      )[0]
      target_graph.add_node(node_index, features=target_node)
      solution_length += int(node_feature["solution"])
    solution_length /= graph.number_of_nodes()

    # Set edge features.
    for receiver, sender, features in graph.edges(data=True):
      input_graph.add_edge(
        sender,
        receiver,
        features=cls.CreateFeature(features, input_edge_fields),
      )
      target_edge = cls.ToOneHot(
        cls.CreateFeature(features, target_edge_fields).astype(int), 2
      )[0]
      target_graph.add_edge(sender, receiver, features=target_edge)

    # Set graph features.
    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([solution_length], dtype=float)

    return input_graph, target_graph

  def GenerateGraphs(
    self, n: int
  ) -> typing.Tuple[
    typing.List[nx.DiGraph], typing.List[nx.DiGraph], typing.List[nx.DiGraph]
  ]:
    """Generate graphs for training.

    Args:
      n: Total number of graphs to generate.

    Returns:
      A tuple with three elements: the list of input graphs, the list of output
      graphs, and the list of generated graphs.
    """
    input_graphs: typing.List[nx.DiGraph] = []
    target_graphs: typing.List[nx.DiGraph] = []
    graphs: typing.List[nx.DiGraph] = []
    for _ in range(n):
      graph = self.GenerateGraph()
      graph = self.AddShortestPath(graph)
      input_graph, target_graph = self.GraphToInputTarget(graph)

      graphs.append(graph)
      input_graphs.append(input_graph)
      target_graphs.append(target_graph)

    return input_graphs, target_graphs, graphs


def GenerateGraph(
  rand: np.random.RandomState,
  num_nodes_min_max,
  dimensions: int = 2,
  theta: float = 1000.0,
  rate: float = 1.0,
  weight_name: str = "distance",
) -> nx.Graph:
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
    num_nodes, theta, pos=pos, weight=weight
  )

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
    combined_graph.get_edge_data(i, j).setdefault(weight_name, distances[i, j])
  return combined_graph


def AddShortestPath(
  rand: np.random.RandomState,
  graph: nx.Graph,
  min_length: int = 1,
  weight_name: str = "distance",
) -> nx.DiGraph:
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
  path = nx.shortest_path(graph, source=start, target=end, weight=weight_name)

  # Creates a directed graph, to store the directed path from start to end.
  directed_graph = graph.to_directed()

  # Add the "start", "end", and "solution" attributes to the nodes.
  directed_graph.add_node(start, start=True)
  directed_graph.add_node(end, end=True)
  directed_graph.add_nodes_from(
    list(labtypes.SetDiff(directed_graph.nodes(), [start])), start=False
  )
  directed_graph.add_nodes_from(
    list(labtypes.SetDiff(directed_graph.nodes(), [end])), end=False
  )
  directed_graph.add_nodes_from(
    list(labtypes.SetDiff(directed_graph.nodes(), path)), solution=False
  )
  directed_graph.add_nodes_from(path, solution=True)

  # Now do the same for the edges.
  path_edges = list(labtypes.PairwiseIterator(path))
  directed_graph.add_edges_from(
    list(labtypes.SetDiff(directed_graph.edges(), path_edges)), solution=False
  )
  directed_graph.add_edges_from(path_edges, solution=True)

  return directed_graph


def GraphToInputTarget(
  graph: nx.DiGraph,
) -> typing.Tuple[nx.DiGraph, nx.DiGraph]:
  """Returns 2 graphs with input and target feature vectors for training.

  Args:
    graph: An `nx.DiGraph` instance.

  Returns:
    The input `nx.DiGraph` instance.
    The target `nx.DiGraph` instance.

  Raises:
    ValueError: unknown node type
  """

  def CreateFeature(
    data_dict: typing.Dict[str, typing.Any], feature_names: typing.List[str]
  ):
    return np.hstack(
      [np.array(data_dict[feature], dtype=float) for feature in feature_names]
    )

  def ToOneHot(indices: typing.Iterator[int], max_value: int, axis: int = -1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
      one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot

  input_node_fields = ("pos", "weight", "start", "end")
  input_edge_fields = ("distance",)
  target_node_fields = ("solution",)
  target_edge_fields = ("solution",)

  input_graph = graph.copy()
  target_graph = graph.copy()

  solution_length = 0
  # Set node features.
  for node_index, node_feature in graph.nodes(data=True):
    input_graph.add_node(
      node_index, features=CreateFeature(node_feature, input_node_fields)
    )
    target_node = ToOneHot(
      CreateFeature(node_feature, target_node_fields).astype(int), 2
    )[0]
    target_graph.add_node(node_index, features=target_node)
    solution_length += int(node_feature["solution"])
  solution_length /= graph.number_of_nodes()

  # Set edge features.
  for receiver, sender, features in graph.edges(data=True):
    input_graph.add_edge(
      sender, receiver, features=CreateFeature(features, input_edge_fields)
    )
    target_edge = ToOneHot(
      CreateFeature(features, target_edge_fields).astype(int), 2
    )[0]
    target_graph.add_edge(sender, receiver, features=target_edge)

  # Set graph features.
  input_graph.graph["features"] = np.array([0.0])
  target_graph.graph["features"] = np.array([solution_length], dtype=float)

  return input_graph, target_graph


def GenerateGraphs(
  rand: np.random.RandomState,
  n: int,
  num_nodes_min_max: typing.Tuple[int, int],
  theta: float,
) -> typing.Tuple[
  typing.List[nx.DiGraph], typing.List[nx.DiGraph], typing.List[nx.DiGraph]
]:
  """Generate graphs for training.

  Args:
    rand: A random seed (np.RandomState instance).
    n: Total number of graphs to generate.
    num_nodes_min_max: A 2-tuple with the [lower, upper) number of nodes per
      graph. The number of nodes for a graph is uniformly sampled within this
      range.
    theta: (optional) A `float` threshold parameters for the geographic
      threshold graph's threshold. Default= the number of nodes.

  Returns:
    input_graphs: The list of input graphs.
    target_graphs: The list of output graphs.
    graphs: The list of generated graphs.
  """
  input_graphs: typing.List[nx.DiGraph] = []
  target_graphs: typing.List[nx.DiGraph] = []
  graphs: typing.List[nx.DiGraph] = []
  for _ in range(n):
    graph = GenerateGraph(rand, num_nodes_min_max, theta=theta)
    graph = AddShortestPath(rand, graph)
    input_graph, target_graph = GraphToInputTarget(graph)

    graphs.append(graph)
    input_graphs.append(input_graph)
    target_graphs.append(target_graph)

  return input_graphs, target_graphs, graphs
