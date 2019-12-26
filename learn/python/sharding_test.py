"""Using test sharding to divide a test file.

This script contains a single test which is executed using a parameterized
fixture. When executed with sharding, the fixture parameters are divided amongst
the shards and executed. Note that each shard executes in its own session,
so that the fixture setup and teardown is performed shard_count times, even
for session-level fixtures.
"""
from labm8.py import test


FLAGS = test.FLAGS

MODULE_UNDER_TEST = None


@test.Fixture(scope="session", params=list(range(1, 101)))
def a(request) -> int:
  """A test fixture which returns a number."""
  return request.param


def test_foo(a: int):
  """Test running a graph classifier."""
  assert a


if __name__ == "__main__":
  test.Main()
