"""Unit tests for //experimental/compilers/reachability:datasets."""
import sys

import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import control_flow_graph as cfg
from experimental.compilers.reachability import reachability_pb2
from experimental.compilers.reachability import cfg_datasets as datasets


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def opencl_dataset() -> datasets.OpenClDeviceMappingsDataset:
  yield datasets.OpenClDeviceMappingsDataset()


def test_OpenClDeviceMappingsDataset_cfgs_df_count(
      opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that dataset has expected number of rows."""
  assert len(opencl_dataset.cfgs_df) == 195


def test_OpenClDeviceMappingsDataset_cfgs_df_contains_valid_protos(
      opencl_dataset: datasets.OpenClDeviceMappingsDataset):
  """Test that proto columns can be parsed."""
  for cfg_proto in opencl_dataset.cfgs_df['program:cfg_proto'].values:
    proto = reachability_pb2.ControlFlowGraph.FromString(cfg_proto)
    cfg.ControlFlowGraph.FromProto(proto)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
