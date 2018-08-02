"""Unit tests for //configure."""
import os
import pathlib
import pytest
import sys
import tempfile
from absl import app
from absl import flags

from lib.labm8 import bazelutil


FLAGS = flags.FLAGS

# The path of the configure script.
CONFIG_SCRIPT = bazelutil.DataPath('phd/configure')


@pytest.fixture(scope='module')
def configure():
  """A test fixture which yields the configure script as an imported module."""
  with tempfile.TemporaryDirectory() as d:
    temp_module = pathlib.Path(d)
    with open(temp_module / '__init__.py', 'w') as f:
      f.write('')
    with open(CONFIG_SCRIPT) as f:
      src = f.read()
    with open(temp_module / 'configure.py', 'w') as f:
      f.write(src)
    sys.path.append(d)
    import configure
    yield configure


def test_Mkdir(configure):
  """Test making a directory."""
  with tempfile.TemporaryDirectory() as d:
    path = pathlib.Path(d) / 'a' / 'b' / 'c'
    configure.Mkdir(os.path.join(d, 'a', 'b', 'c'))
    assert path.is_dir()
    # Test that you can call Mkdir() with an already existing directory.
    configure.Mkdir(os.path.join(d, 'a', 'b', 'c'))
    assert path.is_dir()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
