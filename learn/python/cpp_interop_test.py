"""Test of Python calling into C++ library.

NOTE: this code has been librified and moved into //labm8:ppar.
"""
import subprocess

from labm8 import app
from labm8 import bazelutil
from labm8 import test
from learn.python import cpp_interop_pb2

FLAGS = app.FLAGS

# Paths of the C++ binaries to be tested.
CPP_INTEROP_BIN = bazelutil.DataPath('phd/learn/python/cpp_interop_bin')
CPP_INTEROP_IN_PLACE_BIN = bazelutil.DataPath(
    'phd/learn/python/cpp_interop_inplace_bin')

MODULE_UNDER_TEST = None  # No coverage.


def test_AddXandY():
  """Test running a native binary with proto input and output."""
  # Create the proto containing the inputs for the C++ binary.
  input_proto = cpp_interop_pb2.AddXandY(x=2, y=3)

  # Run the C++ binary, passing the proto as input and capturing it's output.
  process = subprocess.Popen([str(CPP_INTEROP_BIN)],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  stdout, _ = process.communicate(input_proto.SerializeToString())
  assert not process.returncode

  # Decode the C++ binary output.
  output_proto = cpp_interop_pb2.AddXandY()
  output_proto.ParseFromString(stdout)

  # Check the output proto values, set by the binary.
  assert input_proto.x == 2
  assert input_proto.y == 3
  assert not input_proto.result

  assert not output_proto.x
  assert not output_proto.y
  assert output_proto.result == 2 + 3


def test_AddXandY_in_place():
  """Test running a native binary with proto input and output."""
  # Create the proto containing the inputs for the C++ binary.
  proto = cpp_interop_pb2.AddXandY(x=2, y=3)

  # Run the C++ binary, passing the proto as input and capturing it's output.
  process = subprocess.Popen([str(CPP_INTEROP_IN_PLACE_BIN)],
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  stdout, _ = process.communicate(proto.SerializeToString())
  assert not process.returncode

  # Decode the C++ binary output.
  proto.ParseFromString(stdout)

  # Check the proto values produced by the binary.
  assert proto.x == 2
  assert proto.y == 3
  assert proto.result == 2 + 3


if __name__ == '__main__':
  test.Main()
