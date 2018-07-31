"""A generator for reachability graphs."""
import numpy as np
import pathlib
import random
import typing
from absl import app
from absl import flags
from phd.lib.labm8 import fmt
from phd.lib.labm8 import fs
from phd.lib.labm8 import graph as libgraph

from experimental.compilers.reachability.proto import reachability_pb2


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'reachability_num_nodes', 6,
    'The number of CFG nodes to generate.')
flags.DEFINE_integer(
    'reachability_seed', None,
    'Random number to seed numpy RNG with.')
flags.DEFINE_float(
    'reachability_scaling_param', 1.0,
    'Scaling parameter to use to determine the likelihood of edges between CFG '
    'vertices. A larger number makes for more densely connected CFGs.')
flags.DEFINE_string(
    'reachability_dot_path',
    '/tmp/phd/experimental/compilers/reachability/reachability.dot',
    'Path to dot file to generate.')


def NumberToLetters(num: int) -> str:
  if num >= 26:
    raise ValueError
  return chr(ord('A') + num)


class ControlFlowGraph(libgraph.Graph):

  def __init__(self, name: str):
    super(ControlFlowGraph, self).__init__(name)
    self.all_nodes: typing.List['ControlFlowGraph'] = None

  def _IsReachable(self, node: 'ControlFlowGraph', visited) -> bool:
    visited.add(self)
    if node == self:
      return True
    else:
      for child in self.children:
        if child not in visited:
          if child._IsReachable(node, visited):
            return True
    return False

  def IsReachable(self, node: 'ControlFlowGraph') -> bool:
    """Return whether a node is reachable."""
    return self._IsReachable(node, set())

  @classmethod
  def GenerateRandom(cls, num_nodes: int,
                     connections_scaling_param: float = 1.0,
                     seed: typing.Optional[int] = None) -> 'ControlFlowGraph':
    if isinstance(seed, int):
      np.random.seed(seed)
      random.seed(seed)
    nodes = [cls(NumberToLetters(i)) for i in range(num_nodes)]
    for node in nodes:
      node.all_nodes = nodes
    adjacency_matrix = (
        np.random.rand(num_nodes, num_nodes) * connections_scaling_param)
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
          j = random.randint(0, len(nodes) - 1)
        node.children.add(nodes[j])
    return nodes[0]

  def _ToSucessorsList(self, successors, visited) -> None:
    if self in visited:
      return
    visited.add(self)
    s = ' '.join(child.name for child in self.children)
    successors.append(f'{self.name}: {s}')
    for child in self.children:
      child._ToSucessorsList(successors, visited)

  def ToSuccessorsList(self) -> typing.List[str]:
    successors = []
    self._ToSucessorsList(successors, set())
    return list(sorted(successors))

  def ToDot(self) -> str:
    strings = []
    for node in self.all_nodes:
      strings += [f'{node.name} -> {child}' for child in node.children]
    dot = fmt.IndentList(2, strings)
    return f"digraph graphname {{\n  {dot}\n}}"

  def Reachables(self) -> typing.List[bool]:
    src = self.all_nodes[0]
    return [src.IsReachable(node) for node in self.all_nodes]

  def SetReachabilityTrainingDataPointProto(
      self, proto: reachability_pb2.ReachabilityTrainingDataPoint) -> None:
    for node in self.all_nodes:
      proto_node = proto.graph.node.add()
      proto_node.name = node.name
      proto_node.child.extend([c.name for c in node.children])
    proto.reachable.extend(self.Reachables())

  def ToReachabilityTrainingDataPointProto(
      self) -> reachability_pb2.ReachabilityTrainingDataPoint:
    data = reachability_pb2.ReachabilityTrainingDataPoint()
    self.SetReachabilityTrainingDataPointProto(data)
    return data


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  graph = ControlFlowGraph.GenerateRandom(
      FLAGS.reachability_num_nodes, seed=FLAGS.reachability_seed,
      connections_scaling_param=FLAGS.reachability_scaling_param)

  fs.mkdir(pathlib.Path(FLAGS.reachability_dot_path).parent)
  with open(FLAGS.reachability_dot_path, 'w') as f:
    f.write(graph.ToDot())

  for node in graph.all_nodes:
    s = ' '.join(child.name for child in node.children)
    print(f'{node.name}: {s}')


if __name__ == '__main__':
  app.run(main)
