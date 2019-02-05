"""A generator for control flow graphs."""
import typing

import networkx as nx
import numpy as np
from absl import flags

from experimental.compilers.reachability import control_flow_graph as cfg


FLAGS = flags.FLAGS


class UniqueNameSequence(object):
  """A unique name sequence generator.

  Generates name sequences from a base characeter.
  E.g. 'a', 'b', 'c', ... 'aa', 'ab', ...
  """

  def __init__(self, base_char: str, prefix: str = '', suffix: str = ''):
    """Instantiate a unique name sequence.

    Args:
      base_char: The first character in the sequence. Must be 'a' or 'A'.
      prefix: An optional prefix to include in sequence names.
      suffix: An optional suffix to include in sequence names.

    Raises:
      ValueError: If base_char is not 'a' or 'A'.
    """
    if base_char not in {'a', 'A'}:
      raise ValueError(f"Invalid base_char '{base_char}'")
    self._base_ord = ord(base_char)
    self._prefix = prefix
    self._suffix = suffix
    self._i = 0

  def StringInSequence(self, i: int) -> str:
    """Return the i-th string in the sequence.

    Args:
      i: The index into the name sequence.

    Returns:
      The i-th name in the sequence.

    Raises:
      ValueError: If i is out of range (negative).
    """
    if i < 0:
      raise ValueError
    s = [self._prefix]

    while (i > 25):
      k = i // 26
      i %= 26
      s.append(chr(self._base_ord - 1 + k))
    s.append(chr(self._base_ord + i))

    s.append(self._suffix)

    return ''.join(s)

  def __iter__(self):
    return self

  def __next__(self) -> str:
    """Generate the next name in the sequence."""
    s = self.StringInSequence(self._i)
    self._i += 1
    return s


class ControlFlowGraphGenerator(object):
  """A generator for control flow graphs."""

  def __init__(self, rand: np.random.RandomState,
               num_nodes_min_max: typing.Tuple[int, int],
               edge_density: float, strict: bool):
    """Instantiate a control flow graph generator.

    Args:
      rand: A random state instance.
      num_nodes: The number of CFG nodes.
      edge_density: The edge edge_density, in range (0,1], where 1.0 will produce fully
        connected graphs, and lower numbers will produce more sparsely connected
        graphs.
      strict: If True, generate strictly valid control flow graphs. A valid
        control flow graph has no unreachable nodes.
    """
    # Validate inputs.
    if num_nodes_min_max[0] > num_nodes_min_max[1]:
      raise ValueError("Upper bound of num nodes must be >= lower bound")
    if num_nodes_min_max[0] < 2:
      raise ValueError("Lower bound for num nodes must be >= 2")
    if not 0 < edge_density <= 1:
      raise ValueError('Edge density must be in range (0,1]')

    self._rand = rand
    self._num_nodes_min_max = num_nodes_min_max
    self._edge_density = edge_density
    self._graph_name_sequence = UniqueNameSequence('A', prefix='cfg_')
    self._strict = strict

  def __iter__(self):
    return self

  def __next__(self) -> cfg.ControlFlowGraph:
    return self.GenerateOne()

  def GenerateOne(self) -> cfg.ControlFlowGraph:
    """Create a random CFG.

    Returns:
      A ControlFlowGraph instance.
    """
    # TODO(cec): Several issues here:
    #   * Non-terminating while loops when we can't match the expected edge
    #     density.
    #   * Parallel edges being created.
    #   * In "real" CFGs, does outdegree(n) ever exceed 2? If not, this should
    #     be reflected here. Possible sources for increased outdegree > 2 could
    #     be exceptions and indirect jumps.

    # Sample the number of nodes to put in the graph, unless min == max.
    if self._num_nodes_min_max[0] == self._num_nodes_min_max[1]:
      num_nodes = self._num_nodes_min_max[0]
    else:
      num_nodes = self._rand.randint(*self._num_nodes_min_max)

    # Generate the graph and create the named nodes.
    graph = cfg.ControlFlowGraph(name=next(self._graph_name_sequence))
    node_name_sequence = UniqueNameSequence('A')
    [graph.add_node(i, name=next(node_name_sequence)) for i in range(num_nodes)]

    # Set the entry and exit blocks.
    entry_block = 0
    exit_block = num_nodes - 1
    graph.nodes[entry_block]['entry'] = True
    graph.nodes[exit_block]['exit'] = True

    # Generate an adjacency matrix of random binary values.
    adjacency_matrix = self._rand.choice([False, True], size=(num_nodes, num_nodes), p=(0.9, 0.1))

    # Helper methods.

    def NotSelfLoop(i, j):
      """Self loops are not allowed in CFGs."""
      return i != j

    def NotExitNodeOutput(i, j):
      """Outputs are not allowed from the exit block."""
      del j
      return i != exit_block

    def IsEdge(i, j):
      """Return whether edge is set in adjacency matrix."""
      return adjacency_matrix[i, j]

    def AddRandomEdge(src):
      """Add an outgoing edge from src to a random destination."""
      dst = src
      while dst == src:
        dst = self._rand.randint(0, num_nodes)
      graph.add_edge(src, dst)

    def AddRandomIncomingEdge(dst):
      """Add an incoming edge to dst from a random source."""
      src = dst
      while src == dst or src == exit_block:
        src = self._rand.randint(0, num_nodes)
      graph.add_edge(src, dst)

    # Add the edges to the graph, subject to the constraints of CFGs.
    for i, j in np.argwhere(adjacency_matrix):
      # CFG nodes cannot be connected to themselves.
      if NotSelfLoop(i, j) and NotExitNodeOutput(i, j):
        graph.add_edge(i, j)

    # Make sure that every node has one output. This ensures that the graph is
    # fully connected, but does not ensure that each node has an incoming
    # edge (i.e. is unreachable).
    modified = True
    while modified:
      for node in graph.nodes:
        if NotExitNodeOutput(node, 0) and not graph.out_degree(node):
          AddRandomEdge(node)
          break
      else:
        # We iterated through all nodes without making any modifications: we're
        # done.
        modified = False

    if self._strict:
      # Make sure that every node is reachable, and no edge can be fused.
      modified = True
      while modified:
        for src, dst in graph.edges:
          if (not (graph.out_degree(src) > 1 or graph.in_degree(dst) > 1) and
              NotExitNodeOutput(src, dst)):
            AddRandomEdge(src)
            break
        else:
          # We iterated through all nodes without making any modifications:
          # we're done.
          modified = False

    # Make sure the exit block has at least one incoming edge.
    if not graph.in_degree(exit_block):
      AddRandomIncomingEdge(exit_block)

    # Ensure that there is a path from the entry to exit nodes. If not, chose a
    # random successor of the entry block and connect the exit block to it. If
    # there are no successors, connect the entry and exit blocks directly.
    if not nx.has_path(graph, entry_block, exit_block):
      successors = list(graph.successors(entry_block))
      dst = self._rand.choice(successors) if successors else entry_block
      graph.add_edge(dst, exit_block)

    # Continue adding random edges until we reach the target edge density.
    while graph.edge_density < self._edge_density:
      AddRandomEdge(self._rand.randint(0, exit_block))

    return graph.ValidateControlFlowGraph(strict=self._strict)

  def Generate(self, n: int) -> typing.Iterator[cfg.ControlFlowGraph]:
    """Generate a sequence of graphs.

    Args:
      n: The number of graphs to generate.

    Returns:
      An iterator of graphs.
    """
    return (self.GenerateOne() for _ in range(n))

  def GenerateUnique(self, n: int) -> typing.Iterator[cfg.ControlFlowGraph]:
    """Generate a sequence of unique graphs.

    Args:
      n: The number of unique graphs to generate.

    Returns:
      An iterator of unique graphs, where g0 != g1 != ... != gn.
    """
    graph_hashes = set()
    while len(graph_hashes) < n:
      graph = self.GenerateOne()
      graph_hash = hash(graph)
      if graph_hash not in graph_hashes:
        graph_hashes.add(graph_hash)
        yield graph
