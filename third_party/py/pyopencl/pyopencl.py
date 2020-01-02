"""pyopencl module shim.

This is a drop-in replacement for pyopencl that enables support for execution
inside of bazel sandboxing.
"""
import os

# Switch out the default value for ~/.cache on Linux when executing in
# bazel sandbox. See:
# https://docs.bazel.build/versions/master/test-encyclopedia.html#initial-conditions
_BAZEL_TEST_TMPDIR = os.environ.get("TEST_TMPDIR")
if _BAZEL_TEST_TMPDIR:
  os.environ["XDG_CACHE_HOME"] = _BAZEL_TEST_TMPDIR

from pyopencl import *
