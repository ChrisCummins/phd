"""Test absl logging to file."""
import pathlib
import sys
import tempfile

import pytest
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir():
  with tempfile.TemporaryDirectory() as d:
    yield d


def test_log_to_file(tempdir):
  """Benchmark instantiation of a one layer LSTM network without compiling."""
  logging.set_verbosity(logging.DEBUG)
  logging.set_stderrthreshold(logging.INFO)
  logdir = pathlib.Path(tempdir)
  old_logtostderr = FLAGS.logtostderr
  logging.get_absl_handler().start_logging_to_file('foo', logdir)
  # start_logging_to_file() sets logtostderr to False. Re-enable whatever
  # value it was before the call.
  FLAGS.logtostderr = old_logtostderr
  logging.info('Hello, info!')
  logging.debug('Hello, debug!')
  logging.warning('Hello, warning!')
  logging.error('Hello, error!')
  logging.info('Hello, ...\nmultiline!')
  logging.flush()

  assert (logdir / 'foo.INFO').is_file()

  with open(logdir / 'foo.INFO') as f:
    c = f.read()
  lines = c.split('\n')
  assert len(lines) == 7
  assert lines[0][0] == 'I'
  assert lines[0].endswith('Hello, info!')
  assert lines[1][0] == 'I'
  assert lines[1].endswith('Hello, debug!')
  assert lines[2][0] == 'W'
  assert lines[2].endswith('Hello, warning!')
  assert lines[3][0] == 'E'
  assert lines[3].endswith('Hello, error!')
  assert lines[4][0] == 'I'
  assert lines[4].endswith('Hello, ...')
  assert lines[5] == 'multiline!'


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
