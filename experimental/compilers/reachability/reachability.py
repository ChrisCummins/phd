"""A generator for Control Flow Graphs."""
import pathlib

from absl import app
from absl import flags

from experimental.compilers.reachability import control_flow_graph
from labm8 import fs


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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  graph = control_flow_graph.ControlFlowGraph.GenerateRandom(
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
