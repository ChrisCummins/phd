"""Generate multiple reachability graphs."""
from absl import app
from absl import flags

from experimental.compilers.reachability import reachability
from experimental.compilers.reachability.proto import reachability_pb2


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'reachability_num_training_graphs', 10000,
    'The number of training protos to generate.')


def MakeReachabilityTrainingDataProto(
    num_data_points: int) -> reachability_pb2.ReachabilityTrainingData:
  """Generate a training data proto."""
  data = reachability_pb2.ReachabilityTrainingData()
  for i in range(num_data_points):
    graph = reachability.ControlFlowGraph.GenerateRandom(
        FLAGS.reachability_num_nodes, seed=i,
        connections_scaling_param=FLAGS.reachability_scaling_param)
    proto = data.entry.add()
    graph.SetReachabilityTrainingDataPointProto(proto)
    proto.seed = i
  return data


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  print(MakeReachabilityTrainingDataProto(
      FLAGS.reachability_num_training_graphs))


if __name__ == '__main__':
  app.run(main)
