"""This module contains utility functions for parallelism in python.

The goal of the module is to provide easy to use implementations of typical
parallel workloads, such as data parallel map operations.
"""
import multiprocessing

import subprocess
import typing
from absl import flags

from lib.labm8 import bazelutil
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


class MapWorkerError(EnvironmentError):
  """Resulting error from a _MapWorker that fails."""

  def __init__(self, returncode: int):
    """Create a _MapWorker error.

    Args:
      returncode: The process return code.
    """
    self._returncode = returncode

  def __repr__(self) -> str:
    return f"Command exited with code {self.returncode}"

  @property
  def returncode(self) -> int:
    """Get the return code of the process."""
    return self._returncode


class _MapWorker(object):
  """A work unit for a data parallel workload.

  A _MapWorker executes a command as a subprocess, passes it a protocol buffer,
  and decodes a protocol buffer output.

  This is a helper class created by MapNativeProtoProcessingBinary() and
  returned to the user. It is not to be used by user code.
  """

  def __init__(self, id: int, cmd: typing.List[str],
               input_proto: pbutil.ProtocolBuffer):
    """Create a map worker.

    Args:
      id: The numeric ID of the map worker.
      cmd: The command to execute, as a list of arguments to subprocess.Popen().
      input_proto: The protocol buffer to pass to the command.
    """
    self._id = id
    self._cmd = cmd
    # We store the input proto in wire format (as a serialized string) rather
    # than as a class object as pickle can get confused by the types.
    # See: https://stackoverflow.com/a/1413299
    self._input_proto: typing.Optional[pbutil.ProtocolBuffer] = None
    self._input_proto_string = input_proto.SerializeToString()
    self._output_proto_string: typing.Optional[str] = None
    self._output_proto: typing.Optional[pbutil.ProtocolBuffer] = None
    self._output_proto_decoded = False
    self._returncode: typing.Optional[int] = None
    self._error_message_binary: typing.Optional[str] = None
    self._done = False

  def Run(self) -> None:
    """Execute the process and store the output.

    If the process fails, no exception is raised. The error can be accessed
    using the error() method. After calling this method, SetProtos() *must* be
    called.
    """
    assert not self._done

    # Run the C++ worker process, capturing it's output.
    process = subprocess.Popen(
        self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    # Send the input proto to the C++ worker process.
    # TODO: Add timeout.
    stdout, _ = process.communicate(self._input_proto_string)
    self._returncode = process.returncode
    del self._input_proto_string

    if not process.returncode:
      # Store the C++ binary output in wire format.
      self._output_proto_string = stdout

  def SetProtos(self, input_proto: pbutil.ProtocolBuffer,
                output_proto_class: typing.Type) -> None:
    """Set the input protocol buffer, and decode the output protocol buffer.

    This is performed by the SetProtos() method (rather than during Run()) so
    that when pickled, this class contains only basic types, not protocol buffer
    instances.

    Args:
      input_proto: The input protocol buffer message.
      output_proto_class: The protocol buffer class of the output message.
    """
    assert not self._done
    self._done = True
    self._input_proto = input_proto
    # Only parse the output if the worker completed successfully.
    if not self._returncode:
      # Be careful that errors during protocol buffer decoding (e.g.
      # unrecognized fields, conflicting field type/tags) are silently ignored
      # here.
      self._output_proto = output_proto_class.FromString(
          self._output_proto_string)
    # No need to hand onto the string message any more.
    del self._output_proto_string

  @property
  def id(self):
    """Return the numeric ID of the map worker."""
    return self._id

  def input(self) -> pbutil.ProtocolBuffer:
    """Get the input protocol buffer."""
    assert self._done
    return self._input_proto

  def output(self) -> typing.Optional[pbutil.ProtocolBuffer]:
    """Get the protocol buffer decoded from stdout of the executed binary.

    If the process failed (e.g. not _MapWorker.ok()), None is returned.
    """
    assert self._done
    return self._output_proto

  def error(self) -> typing.Optional[MapWorkerError]:
    """Get the error generated by a failed binary execution.

    If the process succeeded (e.g. _MapWorker.ok()), None is returned.
    """
    if self._returncode:
      return MapWorkerError(self._returncode, self._error_message_binary)

  def ok(self) -> bool:
    """Return whether binary execution succeeded."""
    return not self._returncode


def _RunNativeProtoProcessingWorker(map_worker: _MapWorker) -> _MapWorker:
  """Private helper message to execute Run() method of _MapWorker.

  This is passed to Pool.imap_unordered() as the function to execute for every
  work unit. This is needed because only module-level functions can be pickled.
  """
  map_worker.Run()
  return map_worker


def MapNativeProtoProcessingBinary(
    binary_data_path: str, input_protos: typing.List[pbutil.ProtocolBuffer],
    output_proto_class: typing.Type,
    binary_args: typing.Optional[typing.List[str]] = None,
    pool: typing.Optional[multiprocessing.Pool] = None,
    num_processes: typing.Optional[int] = None) -> typing.Iterator[_MapWorker]:
  """Run a protocol buffer processing binary over a set of inputs.

  Args:
    binary_data_path: The path of the binary to execute, as provied to
      bazelutil.DataPath().
    binary_args:
  """
  binary_path = bazelutil.DataPath(binary_data_path)
  binary_args = binary_args or []
  cmd = [str(binary_path)] + binary_args

  # Read all inputs to a list. We need the inputs in a list so that we can
  # map an inputs position in the list to a _MapWorker.id.
  input_protos = list(input_protos)

  # Create the multiprocessing pool to use, if not provided.
  pool = pool or multiprocessing.Pool(processes=num_processes)

  map_worker_iterator = (
    _MapWorker(i, cmd, input_proto) for
    i, input_proto in enumerate(input_protos))

  for map_worker in pool.imap_unordered(_RunNativeProtoProcessingWorker,
                                        map_worker_iterator):
    map_worker.SetProtos(input_protos[map_worker.id], output_proto_class)
    yield map_worker
