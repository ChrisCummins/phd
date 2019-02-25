import collections
import pickle
import re
import sys
import typing
from contextlib import suppress
from signal import Signals
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile

import numpy as np
from absl import app
from absl import flags

from gpu.cldrive.legacy import args as _args
from gpu.cldrive.legacy import env as _env
from gpu.cldrive.proto import cldrive_pb2
from gpu.oclgrind import oclgrind
from labm8 import bazelutil
from labm8 import err
from labm8 import pbutil

FLAGS = flags.FLAGS

ArgTuple = collections.namedtuple('ArgTuple', ['hostdata', 'devdata'])

_NATIVE_DRIVER = bazelutil.DataPath('phd/gpu/cldrive/native_driver')


class TimeoutError(RuntimeError):
  """Thrown if kernel executions fails to complete within time budget."""

  def __init__(self, timeout: int):
    self.timeout = timeout

  def __repr__(self) -> str:
    return f"Execution failed to complete with {self.timeout} seconds"


class PorcelainError(RuntimeError):
  """Raised if porcelain subprocess exits with non-zero return code."""

  def __init__(self, status: typing.Union[int, str]):
    self.status = status

  def __repr__(self) -> str:
    return f"Porcelain subprocess exited with return code {self.status}"


class NDRange(collections.namedtuple('NDRange', ['x', 'y', 'z'])):
  """A 3 dimensional NDRange tuple. Has components x,y,z.

  Attributes:
    x: x component.
    y: y component.
    z: z component.

  Examples:
    >>> NDRange(1, 2, 3)
    [1, 2, 3]

    >>> NDRange(1, 2, 3).product
    6

    >>> NDRange(10, 10, 10) > NDRange(10, 9, 10)
    True
  """
  __slots__ = ()

  def __repr__(self) -> str:
    return f"[{self.x}, {self.y}, {self.z}]"

  @property
  def product(self) -> int:
    """Get the Linear product (x * y * z)."""
    return self.x * self.y * self.z

  def __eq__(self, rhs: 'NDRange') -> bool:
    return self.x == rhs.x and self.y == rhs.y and self.z == rhs.z

  def __gt__(self, rhs: 'NDRange') -> bool:
    return (self.product > rhs.product and self.x >= rhs.x and
            self.y >= rhs.y and self.z >= rhs.z)

  def __ge__(self, rhs: 'NDRange') -> bool:
    return self == rhs or self > rhs

  def ToString(self) -> str:
    return f'{self.x},{self.y},{self.x}'

  @staticmethod
  def FromString(string: str) -> 'NDRange':
    """Parse an NDRange from a string of format 'x,y,z'.

    Args:
      string: Comma separated NDRange values.

    Returns:
      Parsed NDRange.

    Raises:
      ValueError: If the string does not contain three comma separated integers.

    Examples:
      >>> NDRange.FromString('10,11,3')
      [10, 11, 3]

      >>> NDRange.FromString('10,11,3,1')  # doctest: +IGNORE_EXCEPTION_DETAIL
      Traceback (most recent call last):
      ...
      ValueError
    """
    components = string.split(',')
    if not len(components) == 3:
      raise ValueError(f"invalid NDRange '{string}'")
    x, y, z = int(components[0]), int(components[1]), int(components[2])
    return NDRange(x, y, z)


def DriveKernel(env: _env.OpenCLEnvironment,
                src: str,
                inputs: np.array,
                gsize: typing.Union[typing.Tuple[int, int, int], NDRange],
                lsize: typing.Union[typing.Tuple[int, int, int], NDRange],
                timeout: int = -1,
                optimizations: bool = True,
                profiling: bool = False,
                debug: bool = False) -> np.array:
  """Drive an OpenCL kernel.

  Executes an OpenCL kernel on the given environment, over the given inputs.
  Execution is performed in a subprocess.

  Args:
    env: The OpenCL environment to run the kernel in.
    src: The OpenCL kernel source.
    inputs: The input data to the kernel.
    optimizations: Whether to enable or disbale OpenCL compiler optimizations.
    profiling: If true, print OpenCLevent times for data transfers and kernel
      executions to stderr.
    timeout: Cancel execution if it has not completed after this many seconds.
      A value <= 0 means never time out.
    debug: If true, silence the OpenCL compiler.

  Returns:
    A numpy array of the same shape as the inputs, with the values after
    running the OpenCL kernel.

  Raises:
  ValueError: If input types are incorrect.
  TypeError: If an input is of an incorrect type.
  LogicError: If the input types do not match OpenCL kernel types.
  PorcelainError: If the OpenCL subprocess exits with non-zero return code.
  RuntimeError: If OpenCL program fails to build or run.

  Examples:
    A simple kernel which doubles its inputs:
    >>> src = "kernel void A(global int* a) { a[get_global_id(0)] *= 2; }"
    >>> inputs = [[1, 2, 3, 4, 5]]
    >>> DriveKernel(env, src, inputs, gsize=(5,1,1), lsize=(1,1,1)) # doctest: +SKIP
    array([[ 2,  4,  6,  8, 10]], dtype=int32)
  """

  def Log(*args, **kwargs):
    """Log a message to stderr."""
    if debug:
      print("[cldrive] ", end="", file=sys.stderr)
      print(*args, **kwargs, file=sys.stderr)

  # Assert input types.
  err.assert_or_raise(
      isinstance(env, _env.OpenCLEnvironment), ValueError,
      "env argument is of incorrect type")
  err.assert_or_raise(
      isinstance(src, str), ValueError, "source is not a string")

  # Validate global and local sizes.
  err.assert_or_raise(len(gsize) == 3, TypeError)
  err.assert_or_raise(len(lsize) == 3, TypeError)
  gsize, lsize = NDRange(*gsize), NDRange(*lsize)

  err.assert_or_raise(gsize.product >= 1, ValueError,
                      f"Scalar global size {gsize.product} must be >= 1")
  err.assert_or_raise(lsize.product >= 1, ValueError,
                      f"Scalar local size {lsize.product} must be >= 1")
  err.assert_or_raise(
      gsize >= lsize, ValueError,
      f"Global size {gsize} must be larger than local size {lsize}")

  # Parse args in this process since we want to preserve the sueful exception
  # type.
  args = _args.GetKernelArguments(src)

  # Check that the number of inputs is correct.
  args_with_inputs = [
      i for i, arg in enumerate(args) if not arg.address_space == 'local'
  ]
  err.assert_or_raise(
      len(args_with_inputs) == len(inputs), ValueError,
      "Kernel expects {} inputs, but {} were provided".format(
          len(args_with_inputs), len(inputs)))

  # All inputs must have some length.
  for i, x in enumerate(inputs):
    err.assert_or_raise(len(x), ValueError, f"Input {i} has size zero")

  # Copy inputs into the expected data types.
  data = np.array(
      [np.array(d).astype(a.numpy_type) for d, a in zip(inputs, args)])

  job = {
      "env": env,
      "src": src,
      "args": args,
      "data": data,
      "gsize": gsize,
      "lsize": lsize,
      "optimizations": optimizations,
      "profiling": profiling
  }

  with NamedTemporaryFile('rb+', prefix='cldrive-', suffix='.job') as tmp_file:
    porcelain_job_file = tmp_file.name

    # Write job file.
    pickle.dump(job, tmp_file)
    tmp_file.flush()

    # Enforce timeout using sigkill.
    if timeout > 0:
      cli = ["timeout", "--signal=9", str(int(timeout))]
    else:
      cli = []
    cli += [sys.executable, __file__, porcelain_job_file]

    cli_str = " ".join(cli)
    Log("Porcelain invocation:", cli_str)

    # Fork and run.
    process = Popen(cli, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    status = process.returncode

    if debug:
      print(stdout.decode('utf-8').strip(), file=sys.stderr)
      print(stderr.decode('utf-8').strip(), file=sys.stderr)
    elif profiling:
      # Print profiling output when not in debug mode.
      for line in stderr.decode('utf-8').split('\n'):
        if re.match(r'\[cldrive\] .+ time: [0-9]+\.[0-9]+ ms', line):
          print(line, file=sys.stderr)
    Log(f"Porcelain return code: {status}")

    # Test for non-zero exit codes. The porcelain subprocess catches exceptions
    # and completes gracefully, so a non-zero return code is indicative of a
    # more serious problem.
    #
    # FIXME: I'm seeing a number of SIGABRT return codes which I can't explain.
    # However, ignoring them seems to not cause a problem ...
    if status != 0 and status != -Signals['SIGABRT'].value:
      # A negative return code means a signal. Try and convert the value into a
      # signal name.
      with suppress(ValueError):
        status = Signals(-status).name

      if status == "SIGKILL":
        raise TimeoutError(timeout)
      else:
        raise PorcelainError(status)

    # Read result.
    tmp_file.seek(0)
    return_value = pickle.load(tmp_file)
    outputs = return_value["outputs"]
    error = return_value["err"]
    if error:  # Porcelain raised an exception, re-raise it.
      raise error
    else:
      return outputs


def DriveInstance(
    instance: cldrive_pb2.CldriveInstance) -> cldrive_pb2.CldriveInstance:
  if instance.device.name == _env.OclgrindOpenCLEnvironment().name:
    command = [str(oclgrind.OCLGRIND_PATH), str(_NATIVE_DRIVER)]
  else:
    command = [str(_NATIVE_DRIVER)]

  pbutil.RunProcessMessageInPlace(command, instance)
  return instance


def main(argv):
  assert not argv[1:]
  # TODO(cec): Temporary hacky code for testing.
  print(
      DriveInstance(
          cldrive_pb2.CldriveInstance(
              device=_env.OclgrindOpenCLEnvironment().proto,
              opencl_src="""
kernel void A(global int* a, global float* b, const int c) {
if (get_global_id(0) < c) { 
  a[get_global_id(0)] = get_global_id(0);
  b[get_global_id(0)] *= 2.0;
}
}""",
              min_runs_per_kernel=10,
              dynamic_params=[
                  cldrive_pb2.DynamicParams(
                      global_size_x=16,
                      local_size_x=16,
                  ),
                  cldrive_pb2.DynamicParams(
                      global_size_x=1024,
                      local_size_x=64,
                  ),
                  cldrive_pb2.DynamicParams(
                      global_size_x=128,
                      local_size_x=64,
                  ),
              ],
          )))


if __name__ == '__main__':
  app.run(main)
