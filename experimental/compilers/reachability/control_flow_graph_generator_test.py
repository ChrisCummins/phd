"""Unit tests for :control_flow_graph_generator."""
import sys
import typing

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


def test_UniqueNameSequence_StringInSequence_base_char():
  """Test different base char."""
  g = control_flow_graph_generator.UniqueNameSequence('A')
  assert g.StringInSequence(0) == 'A'


def test_UniqueNameSequence_StringInSequence_invalid_base_char():
  """Test that invalid base char raises error."""
  with pytest.raises(ValueError):
    control_flow_graph_generator.UniqueNameSequence('AA')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
