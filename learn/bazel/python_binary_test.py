"""A test python binary which reads a data file and runs a data binary."""
import os
import subprocess
import sys

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import test

FLAGS = app.FLAGS

DATA_FILE = bazelutil.DataPath("phd/learn/bazel/data_file.txt")
DATA_BINARY = bazelutil.DataPath("phd/learn/bazel/data_binary")

MODULE_UNDER_TEST = None  # No coverage.


def test_main():
  """Main entry point."""
  print("Hello from python", sys.executable)
  print("File location:", __file__)
  print("Current working directory:", os.getcwd())
  with open(DATA_FILE) as f:
    print("Data file:", f.read().rstrip())
  p = subprocess.Popen(
    [DATA_BINARY], stdout=subprocess.PIPE, universal_newlines=True
  )
  stdout, _ = p.communicate()
  print("Data binary:", stdout.rstrip())


if __name__ == "__main__":
  test.Main()
