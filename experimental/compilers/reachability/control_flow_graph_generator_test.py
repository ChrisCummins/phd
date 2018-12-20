"""Unit tests for :control_flow_graph_generator."""
import sys
import typing

import numpy as np
import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import control_flow_graph_generator


FLAGS = flags.FLAGS


def test_UniqueNameSequence_next():
  """Test iterator interface."""
  g = control_flow_graph_generator.UniqueNameSequence('a')
  assert next(g) == 'a'
  assert next(g) == 'b'
  assert next(g) == 'c'


def test_UniqueNameSequence_StringInSequence_single_char():
  """Test single character sequence output."""
  g = control_flow_graph_generator.UniqueNameSequence('a')
  assert g.StringInSequence(0) == 'a'
  assert g.StringInSequence(1) == 'b'
  assert g.StringInSequence(2) == 'c'
  assert g.StringInSequence(25) == 'z'


def test_UniqueNameSequence_StringInSequence_multi_char():
  """Test multi character sequence output."""
  g = control_flow_graph_generator.UniqueNameSequence('a')
  assert g.StringInSequence(26) == 'aa'
  assert g.StringInSequence(27) == 'ab'
  assert g.StringInSequence(28) == 'ac'


def test_UniqueNameSequence_StringInSequence_prefix():
  """Test prefix."""
  g = control_flow_graph_generator.UniqueNameSequence('a', prefix='prefix_')
  assert g.StringInSequence(0) == 'prefix_a'


def test_UniqueNameSequence_StringInSequence_prefix():
  """Test suffix."""
  g = control_flow_graph_generator.UniqueNameSequence('a', suffix='_suffix')
  assert g.StringInSequence(0) == 'a_suffix'


def test_UniqueNameSequence_StringInSequence_base_char():
  """Test different base char."""
  g = control_flow_graph_generator.UniqueNameSequence('A')
  assert g.StringInSequence(0) == 'A'


def test_UniqueNameSequence_StringInSequence_invalid_base_char():
  """Test that invalid base char raises error."""
  with pytest.raises(ValueError):
    control_flow_graph_generator.UniqueNameSequence('AA')


def test_ControlFlowGraphGenerator_invalid_edge_density():
  """Test that invalid edge densities raise error."""
  with pytest.raises(ValueError):
    control_flow_graph_generator.ControlFlowGraphGenerator(
        np.random.RandomState(1), (10, 10), 0, strict=True)
  with pytest.raises(ValueError):
    control_flow_graph_generator.ControlFlowGraphGenerator(
        np.random.RandomState(1), (10, 10), -1, strict=True)
  with pytest.raises(ValueError):
    control_flow_graph_generator.ControlFlowGraphGenerator(
        np.random.RandomState(1), (10, 10), 1.1, strict=True)


def test_ControlFlowGraphGenerator_generates_valid_graphs():
  """Test that generator produces valid graphs."""
  generator = control_flow_graph_generator.ControlFlowGraphGenerator(
      np.random.RandomState(1), (10, 10), 0.5, strict=True)
  g = next(generator)
  assert g.ValidateControlFlowGraph() == g
  g = next(generator)
  assert g.ValidateControlFlowGraph() == g
  g = next(generator)
  assert g.ValidateControlFlowGraph() == g


def test_ControlFlowGraphGenerator_num_nodes():
  """Test that generator produces graphs with expected number of nodes."""
  generator = control_flow_graph_generator.ControlFlowGraphGenerator(
      np.random.RandomState(1), (10, 10), 0.5, strict=True)
  assert next(generator).number_of_nodes() == 10

  generator = control_flow_graph_generator.ControlFlowGraphGenerator(
      np.random.RandomState(1), (5, 5), 0.5, strict=True)
  assert next(generator).number_of_nodes() == 5


def test_ControlFlowGraphGenerator_generates_unique_graphs():
  """Test that generator produces unique graphs."""
  generator = control_flow_graph_generator.ControlFlowGraphGenerator(
      np.random.RandomState(1), (10, 10), 0.5, strict=True)
  # Flaky test, but unlikely to fail in practise.
  graphs = [g for g, _ in zip(generator, range(50))]
  edge_sets = {'.'.join(f'{s}-{d}' for s, d in g.edges) for g in graphs}
  assert len(edge_sets) > 1


def test_ControlFlowGraphGenerator_GenerateUnique():
  """Test that unique graphs are generated."""
  generator = control_flow_graph_generator.ControlFlowGraphGenerator(
      np.random.RandomState(1), (10, 10), 0.5, strict=True)
  uniq_graphs = list(generator.GenerateUnique(100))
  assert len(uniq_graphs) == 100
  assert len(set(uniq_graphs)) == 100


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
