"""Unit tests for //labm8/py/internal:workspace_status."""
import subprocess

from labm8.py import bazelutil
from labm8.py import test

PRINT_WORKSPACE_STATUS = bazelutil.DataPath(
  "phd/labm8/py/internal/test/print_workspace_status.par"
)

FLAGS = test.FLAGS


def test_print_workspace_status():
  """Test for the presence of bazel builtin workspace status variables."""
  assert subprocess.check_output([str(PRINT_WORKSPACE_STATUS)])


if __name__ == "__main__":
  test.Main()
