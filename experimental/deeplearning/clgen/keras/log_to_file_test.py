"""Test absl logging to file."""
import pathlib

from labm8.py import app
from labm8.py import logutil
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


def test_log_to_file(tempdir: pathlib.Path):
  """Benchmark instantiation of a one layer LSTM network without compiling."""
  app.SetLogLevel(1)
  logutil.StartTeeLogsToFile("foo", tempdir)
  app.Log(1, "Hello, info!")
  app.Log(2, "Hello, debug!")
  app.Warning("Hello, warning!")
  app.Error("Hello, error!")
  app.Log(1, "Hello, ...\nmultiline!")
  app.FlushLogs()

  assert (tempdir / "foo.INFO").is_file()

  with open(tempdir / "foo.INFO") as f:
    c = f.read()
  lines = c.split("\n")
  assert len(lines) == 7
  assert lines[0][0] == "I"
  assert "Hello, info!" in lines[0]
  assert lines[1][0] == "I"
  assert "Hello, debug!" in lines[1]
  assert lines[2][0] == "W"
  assert "Hello, warning!" in lines[2]
  assert lines[3][0] == "E"
  assert "Hello, error!" in lines[3]
  assert lines[4][0] == "I"
  assert "Hello, ..." in lines[4]
  assert "multiline!" in lines[5]


if __name__ == "__main__":
  test.Main()
