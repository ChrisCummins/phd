"""Test python protocol buffers."""
# This file does not use the phd repo's test infrastructure because it is
# currently broken.

from learn.bazel.protocol_buffers import another_pb2
# The name of the python file to import is the name of the proto file with the
# '.proto' suffix replaced with '_pb2'.
from learn.bazel.protocol_buffers import example_pb2


if __name__ == '__main__':
  message = example_pb2.Example(message="Hello, world!")
  assert message.message == "Hello, world!"
  
  message = another_pb2.Another(answer=42)
  assert message.answer == 42

  print('all tests passed')
