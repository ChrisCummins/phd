"""Unit tests for //compilers/clsmith/clsmith.py."""
import pytest
import sys
from absl import app
from absl import flags

from compilers.clsmith import clsmith


FLAGS = flags.FLAGS


def test_Exec_no_args():
  """Test that CLSmith returns a generated file."""
  src = clsmith.Exec()
  # Check the basic structure of the generated file.
  assert src.startswith('// -g ')
  assert '__kernel void ' in src


def test_Exec_invalid_argument():
  """Test that CLSmithError is raised if invalid args passed to CLSmith."""
  with pytest.raises(clsmith.CLSmithError) as e_ctx:
    clsmith.Exec('--invalid_opt')
  assert '' == str(e_ctx.value)
  assert e_ctx.value.returncode == 255


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
