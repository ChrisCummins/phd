"""Test absl logging to file."""
import pathlib

from labm8 import app
from labm8 import logutil
from labm8 import test

FLAGS = app.FLAGS


def test_log_to_file(tempdir: pathlib.Path):
  """Benchmark instantiation of a one layer LSTM network without compiling."""
  app.SetLogLevel(app.DEBUG)
  logutil.StartTeeLogsToFile('foo', logdir)
  app.Log(1, 'Hello, info!')
  app.Log(2, 'Hello, debug!')
  app.Warning('Hello, warning!')
  app.Error('Hello, error!')
  app.Log(1, 'Hello, ...\nmultiline!')
  app.FlushLogs()

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


if __name__ == '__main__':
  test.Main()
