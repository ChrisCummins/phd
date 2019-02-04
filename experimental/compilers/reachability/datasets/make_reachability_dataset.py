"""Construct datasets for learning CFG reachability."""
import collections
import pathlib
import typing

import numpy as np
import pandas as pd
from absl import app
from absl import flags

from experimental.compilers.reachability import cfg_datasets as datasets
from experimental.compilers.reachability import control_flow_graph as cfg
from experimental.compilers.reachability import \
  control_flow_graph_generator as cfg_generator
from labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_outdir', '/tmp/phd/docs/wip_graph/datasets/reachability',
                    'The directory to write the output dataframes to.')
flags.DEFINE_integer(
    'synthetic_generator_seed', 0xCEC,
    'Random seed for synthetic training graph generator.')
flags.DEFINE_integer(
    'num_synthetic_training_graphs', 2000,
    'The number of training graphs to generate.')
flags.DEFINE_integer(
    'num_synthetic_validation_graphs', 850,
    'The number of validation graphs to generate.')

# Synthetic graph properties.
num_nodes_min_max_tr = (5, 20)
num_nodes_min_max_ge = (15, 25)
edge_density_tr = .01
edge_density_ge = .01

TargetGraphSpec = collections.namedtuple(
    'TargetGraphSpec', ['graph', 'target_node_index'])


class SpecGenerator(object):

  def __init__(self, graphs: typing.Iterator[cfg.ControlFlowGraph]):
    self._graphs = graphs

  def Generate(self, n: int = 0):
    """Generate specs.

    Args:
        n: The maximum number of spec to generatte. If zero, enumerate
            all graphs.
    """
    yield_count = 0
    for g in self._graphs:
      # Only yield graphs which contain at least one edge.
      if not g.edges:
        continue

      for node in g.node:
        # Stop if we have produced enough graphs.
        yield_count += 1
        if n and yield_count > n:
          return

        yield TargetGraphSpec(graph=g, target_node_index=int(node))


# Functions to generate feature vectors. Features vectors are np.arrays of
# floating point values.

def InputGraphNodeFeatures(spec: TargetGraphSpec, node_index: int) -> np.array:
  """Extract node features for an input graph."""
  # If the node is the target node, the features are [0, 1]. Else, the features
  # are [1, 0].
  return np.array([
    0 if node_index == spec.target_node_index else 1,
    1 if node_index == spec.target_node_index else 0,
  ], dtype=np.float32)


def InputGraphEdgeFeatures(spec: TargetGraphSpec,
                           edge_index: typing.Tuple[int, int]):
  """Extract edge features for an input graph."""
  del spec
  del edge_index
  return np.ones(1, dtype=np.float32)


def TargetGraphNodeFeatures(spec: TargetGraphSpec, node_index: int):
  """Extract node features for a target graph."""
  reachable = spec.graph.IsReachable(spec.target_node_index, node_index)
  # If the node is reachable, the features are [0, 1]. Else, the features are
  # [1, 0].
  return np.array([
    0 if reachable else 1,
    1 if reachable else 0,
  ], dtype=np.float32)


def TargetGraphEdgeFeatures(spec: TargetGraphSpec,
                            edge_index: typing.Tuple[int, int]):
  """Extract edge features for a target graph."""
  del spec
  del edge_index
  return np.ones(1, dtype=np.float32)


def SpecToInputTarget(spec: TargetGraphSpec):
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
  input_graph.graph['features'] = np.array([0.0], dtype=np.float32)
  target_graph.graph['features'] = np.array([0.0], dtype=np.float32)

  return input_graph, target_graph


def SpecsToDataFrame(specs: typing.Iterator[TargetGraphSpec], split_type):
  """Return a data frame from specs."""

  def SpecToRow(spec: TargetGraphSpec):
    """Compute a single row in dataframe as a dict."""
    input_graph, target_graph = SpecToInputTarget(spec)
    return {
      'program:source': 'Synthetic',
      'program:name': 'Synthetic',
      'reachability:target_node_index': spec.target_node_index,
      'cfg:graph': spec.graph,
      'cfg:block_count': spec.graph.number_of_nodes(),
      'cfg:edge_count': spec.graph.number_of_edges(),
      'cfg:edge_density': spec.graph.edge_density,
      'cfg:is_valid': spec.graph.IsValidControlFlowGraph(strict=False),
      'cfg:is_strict_valid': spec.graph.IsValidControlFlowGraph(strict=True),
      'networkx:input_graph': input_graph,
      'networkx:target_graph': target_graph,
      'split:type': split_type,
      'graphnet:loss_op': 'NodesSoftmaxCrossEntropy',
      'graphnet:accuracy_evaluator': 'OneHotNodes',
    }

  return pd.DataFrame([SpecToRow(s) for s in specs])


def PickleDataFrame(df: pd.DataFrame, path: pathlib.Path):
  """Pickle a dataframe."""
  with prof.Profile(f"pickled {path}"):
    df.to_pickle(str(path))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  outdir = pathlib.Path(FLAGS.dataset_outdir)
  outdir.mkdir(exist_ok=True, parents=True)

  random_state = np.random.RandomState(FLAGS.synthetic_generator_seed)

  with prof.Profile('synthetic training graphs'):
    training_graph_generator = SpecGenerator(
        cfg_generator.ControlFlowGraphGenerator(
            random_state, num_nodes_min_max_tr, edge_density_tr, strict=False))
    train_df = SpecsToDataFrame(
        training_graph_generator.Generate(FLAGS.num_synthetic_training_graphs),
        'training')

  with prof.Profile('synthetic validation graphs'):
    validation_graph_generator = SpecGenerator(
        cfg_generator.ControlFlowGraphGenerator(
            random_state, num_nodes_min_max_ge, edge_density_ge, strict=False))
    valid_df = SpecsToDataFrame(
        validation_graph_generator.Generate(
            FLAGS.num_synthetic_validation_graphs),
        'validation')

  synthetic_df = pd.concat((train_df, valid_df))
  PickleDataFrame(synthetic_df, outdir / 'synthetic.pkl')

  # OpenCL dataset.
  with prof.Profile('opencl dataset'):
    ocl_dataset = datasets.OpenClDeviceMappingsDataset().cfgs_df.reset_index()

    # Set the program names on the networkx graph instances.
    for _, row in ocl_dataset.iterrows():
      row['cfg:graph'].graph['name'] = ':'.join([
        row['program:benchmark_suite_name'],
        row['program:benchmark_name'],
        row['program:opencl_kernel_name'],
      ])

    ocl_df = SpecsToDataFrame(
        SpecGenerator(ocl_dataset['cfg:graph'].values).Generate(), 'test')
    del ocl_dataset

    # Set the program name column.
    ocl_df['program:source'] = 'OpenCL'
    ocl_df['program:name'] = [
      r['cfg:graph'].graph['name'] for _, r in ocl_df.iterrows()
    ]

  PickleDataFrame(ocl_df, outdir / 'ocl.pkl')
  PickleDataFrame(pd.concat((synthetic_df, ocl_df)),
                  outdir / 'synthetic_ocl.pkl')

  # Linux dataset.
  with prof.Profile('linux dataset'):
    linux_dataset = datasets.LinuxSourcesDataset().cfgs_df.reset_index()

    # Set the program names on the networkx graph instances.
    for _, row in linux_dataset.iterrows():
      row['cfg:graph'].graph['name'] = row['program:src_relpath'].replace(
          '/', '.')

    linux_df = SpecsToDataFrame(
        SpecGenerator(linux_dataset['cfg:graph'].values).Generate(),
        'test')
    del linux_dataset

    # Set the program name column.
    linux_df['program:source'] = 'Linux'
    linux_df['program:name'] = [
      r['cfg:graph'].graph['name'] for _, r in linux_df.iterrows()
    ]

  PickleDataFrame(linux_df, outdir / 'linux.pkl')
  PickleDataFrame(pd.concat((synthetic_df, linux_df)),
                  outdir / 'synthetic_linux.pkl')

  PickleDataFrame(pd.concat((synthetic_df, ocl_df, linux_df)),
                  outdir / 'synthetic_linux_ocl.pkl')


if __name__ == '__main__':
  app.run(main)
