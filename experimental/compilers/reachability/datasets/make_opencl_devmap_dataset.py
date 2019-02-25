"""Generate the CFG datasets for OpenCL device mapping."""
import pathlib

import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping import utils
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import models
from labm8 import prof

FLAGS = flags.FLAGS

flags.DEFINE_string('outdir', '/tmp/phd/docs/wip_graph/datasets/opencl_devmap',
                    'The directory to write the output dataframes to.')


def SplitToDataFrame(split):
  split.train_df['split:type'] = ['training'] * len(split.train_df)
  split.valid_df['split:type'] = ['validation'] * len(split.valid_df)
  split.test_df['split:type'] = ['test'] * len(split.test_df)
  return pd.concat((split.train_df, split.valid_df, split.test_df))


def MakeOpenClDevmapDataset(df: pd.DataFrame, outdir: pathlib.Path):
  """Make a device mapping dataset from the given dataframe."""
  df = utils.AddClassificationTargetToDataFrame(df, 'amd_tahiti_7970')

  outdir.mkdir(exist_ok=True, parents=True)

  model = models.Lda()
  model.init(seed=0, atomizer=None)

  with prof.Profile('extracted graphs'):
    extracted_graphs = list(model.ExtractGraphs(lda.SetNormalizedColumns(df)))

  with prof.Profile('encoded graphs'):
    encoded_graphs = list(model.EncodeGraphs(extracted_graphs))

  unknowns = np.array([
      g.graph['num_unknown_statements'] / g.number_of_nodes()
      for _, g in encoded_graphs
  ])

  logging.info(f'{unknowns.mean():.1%} of statements are unknown '
               f'(min={unknowns.min():.1%}, max={unknowns.max():.1%})')

  # Set CFG properties columns.
  graphs = [x[1] for x in encoded_graphs]
  df['cfg:graph'] = graphs
  df['cfg:block_count'] = [graph.number_of_nodes() for graph in graphs]
  df['cfg:edge_count'] = [graph.number_of_edges() for graph in graphs]
  df['cfg:edge_density'] = [graph.edge_density for graph in graphs]
  df['cfg:diameter'] = [graph.undirected_diameter for graph in graphs]
  df['cfg:is_valid'] = [
      graph.IsValidControlFlowGraph(strict=False) for graph in graphs
  ]
  df['cfg:is_strict_valid'] = [
      graph.IsValidControlFlowGraph(strict=True) for graph in graphs
  ]

  with prof.Profile('input target graphs'):
    input_graphs, target_graphs = zip(
        *list(model.GraphsToInputTargets(encoded_graphs)))

  # Add the graph representations to the dataframe they were extracted from.
  df['networkx:graph'] = [x[1] for x in encoded_graphs]
  df['networkx:input_graph'] = input_graphs
  df['networkx:target_graph'] = target_graphs

  # Add the dynamic functions.
  df['graphnet:loss_op'] = 'GlobalsSoftmaxCrossEntropy'
  df['graphnet:accuracy_evaluator'] = 'OneHotGlobals'

  # Split into train/test/validation datasets:
  splits = list(
      utils.TrainValidationTestSplits(df, np.random.RandomState(0xCEC)))
  assert len(splits) == 2

  # Re-combine into dataframes.
  amd_split, nvidia_split = splits

  assert amd_split.gpu_name == "amd_tahiti_7970"
  amd_df = SplitToDataFrame(amd_split)

  assert nvidia_split.gpu_name == "nvidia_gtx_960"
  nvidia_df = SplitToDataFrame(nvidia_split)

  # Sanity check
  assert len(input_graphs) == len(target_graphs) == len(encoded_graphs) == len(
      amd_df) == len(nvidia_df)

  logging.info("Writing %s", outdir / 'amd.pkl')
  amd_df.to_pickle(str(outdir / 'amd.pkl'))
  logging.info("Writing %s", outdir / 'nvidia.pkl')
  nvidia_df.to_pickle(str(outdir / 'nvidia.pkl'))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  outdir = pathlib.Path(FLAGS.outdir)

  with prof.Profile('read dataset'):
    dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()

  MakeOpenClDevmapDataset(dataset.df, outdir)


if __name__ == '__main__':
  app.run(main)
