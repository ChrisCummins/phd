"""A generator for control flow graphs."""
import typing

import numpy as np
from absl import flags

from experimental.compilers.reachability import control_flow_graph as cfg


FLAGS = flags.FLAGS


class UniqueNameSequence(object):

  def __init__(self, base_char: str, prefix: str = ''):
    if base_char not in {'a', 'A'}:
      raise ValueError(f"Invalid base_char '{base_char}'")
    self._base_ord = ord(base_char)
    self._prefix = prefix
    self._i = 0

  def StringInSequence(self, i: int) -> str:
    if i < 0:
      raise ValueError
    s = [self._prefix]

    while (i > 25):
      k = i // 26
      i %= 26
      s.append(chr(self._base_ord - 1 + k))
    s.append(chr(self._base_ord + i))

    return ''.join(s)

  def __next__(self):
    s = self.StringInSequence(self._i)
    self._i += 1
    return s


class ControlFlowGraphGenerator(object):
  """A generator for control flow graphs."""

  def __init__(self, rand: np.random.RandomState,
               num_nodes_min_max: typing.Tuple[int, int],
               connections_scaling_param: float):
    """Instantiate a control flow graph generator.

    Args:
      rand: A random state instance.
      num_nodes: The number of CFG nodes.
      connections_scaling_param: Scaling parameter to use to determine the
        likelihood of edges between CFG nodes.
    """
    self._rand = rand
    self._num_nodes_min_max = num_nodes_min_max
    self._connections_scaling_param = connections_scaling_param

  @property
  def rand(self) -> np.random.RandomState:
    return self._rand

  @property
  def num_nodes_min_max(self) -> typing.Tuple[int, int]:
    return self._num_nodes_min_max

  @property
  def connections_scaling_param(self) -> float:
    return self._connections_scaling_param

  def GenerateOne(self) -> 'ControlFlowGraph':
    """Create a random CFG.

    Returns:
      A ControlFlowGraph instance.
    """
    num_nodes = self.rand.randint(*self.num_nodes_min_max)

    nodes = [cfg.ControlFlowGraph(NumberToLetters(i)) for i in range(num_nodes)]
    for node in nodes:
      node.all_nodes = nodes
    adjacency_matrix = (
        self.rand.rand(num_nodes, num_nodes) * self.connections_scaling_param)
    adjacency_matrix = np.clip(adjacency_matrix, 0, 1)
    adjacency_matrix = np.rint(adjacency_matrix)
    # CFG nodes cannot be connected to self.
    for i in range(len(adjacency_matrix)):
      adjacency_matrix[i][i] = 0
    for j, row in enumerate(adjacency_matrix):
      for i, col in enumerate(row):
        if col:
          nodes[j].children.add(nodes[i])
    for i, node in enumerate(nodes):
      if not node.children:
        j = i
        while j == i:
          j = self.rand.randint(0, len(nodes) - 1)
        node.children.add(nodes[j])
    return nodes[0]
