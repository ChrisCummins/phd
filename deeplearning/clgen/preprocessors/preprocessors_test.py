"""Unit tests for //deeplearning/clgen/preprocessors/preprocessors.py."""
import sys

import pytest
from absl import app
from absl import logging

import deeplearning.clgen.errors
import deeplearning.clgen.preprocessors.clang
import deeplearning.clgen.preprocessors.opencl


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_rewriter_good_code():
  """Test that OpenCL rewriter renames variables and functions."""
  rewritten = deeplearning.clgen.preprocessors.opencl.rewrite_cl("""\
__kernel void FOOBAR(__global int * b) {
    if (  b < *b) {
          *b *= 2;
    }
}\
""")
  assert rewritten == """\
__kernel void A(__global int * a) {
    if (  a < *a) {
          *a *= 2;
    }
}\
"""


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
