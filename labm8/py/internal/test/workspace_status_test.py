"""Unit tests for //labm8/py/internal:workspace_status."""
from labm8.py import test
from labm8.py.internal import workspace_status

FLAGS = test.FLAGS


def test_bazel_builtins_are_set():
  """Test for the presence of bazel builtin workspace status variables."""
  assert workspace_status.BUILD_HOST
  assert workspace_status.BUILD_USER
  assert workspace_status.BUILD_TIMESTAMP


def test_expected_constants_are_set():
  """Test for custom workspace status variables."""
  assert workspace_status.STABLE_ARCH
  assert workspace_status.STABLE_GIT_URL
  assert workspace_status.STABLE_GIT_COMMIT
  assert workspace_status.STABLE_GIT_DIRTY
  assert workspace_status.STABLE_VERSION


if __name__ == "__main__":
  test.Main()
