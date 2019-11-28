# This file is part of libcecl.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.
"""Runtime utility code for libcecl binaries."""
import os
import time
import typing

from gpu.cldrive.legacy import env as cldrive_env
from gpu.libcecl import libcecl_compile
from gpu.libcecl.proto import libcecl_pb2
from labm8.py import app
from labm8.py import labdate

FLAGS = app.FLAGS


def KernelInvocationsFromCeclLog(
    cecl_log: typing.List[str],
    expected_devtype: typing.Optional[str] = None,
    expected_device_name: typing.Optional[str] = None,
    nanosecond_identity: typing.Callable[[str], int] = int
) -> typing.List[libcecl_pb2.OpenClKernelInvocation]:
  """Interpret and parse the output of a libcecl instrumented application.

  This is an updated and adapted implementation of
  kernel_invocations_from_cecl_log() from:
    //docs/2017_02_cgo/code/benchmarks:cecl2features

  Args:
    cecl_log: The lines of output printed by libcecl, with the '[CECL] ' prefix
      stripped. As a list of lines.
    expected_devtype: The expected device type, e.g. "GPU" or "CPU".
    expected_device_name: The expected OpenCL device name.
    nanosecond_identity: A callable which is invoked for every string runtime
      in the log.

  Raises:
    ValueError: If the device type or name reported in the cecl_log does not
      match the expected value.
  """
  # Per-benchmark data transfer size and time.
  total_transferred_bytes = 0
  total_transfer_time = 0

  kernel_invocations = []

  # Iterate over each line in the cec log.
  app.Log(2, 'Processing %d lines of libcecl logs', len(cecl_log))
  for line in cecl_log:
    # Split line based on ; delimiter into opcode and operands.
    components = [x.strip() for x in line.strip().split(';')]
    opcode, operands = components[0], components[1:]

    # Skip empty lines.
    if not opcode:
      continue

    if opcode == "clCreateCommandQueue":
      devtype, devname = operands

      # If we don't know the device type, don't check it. This isn't a problem -
      # not all drivers report device type correctly, e.g. POCL returns a
      # non-standard device type value.
      if (expected_devtype and devtype != 'UNKNOWN' and
          devtype != expected_devtype):
        raise ValueError(
            f"Expected device type {expected_devtype} does not match actual "
            f"device type {devtype}")

      if expected_device_name and devname != expected_device_name:
        raise ValueError(
            f"Expected device name '{expected_device_name}' does not match "
            f"actual device name '{devname}'")

    elif opcode == "clEnqueueNDRangeKernel":
      kernel_name, global_size, local_size, elapsed = operands
      kernel_invocations.append(
          libcecl_pb2.OpenClKernelInvocation(
              kernel_name=kernel_name,
              global_size=int(global_size),
              local_size=int(local_size),
              kernel_time_ns=nanosecond_identity(elapsed)))
      app.Log(2, 'Extracted clEnqueueNDRangeKernel from log')
    elif opcode == "clEnqueueTask":
      kernel_name, elapsed = operands
      kernel_invocations.append(
          libcecl_pb2.OpenClKernelInvocation(
              kernel_name=kernel_name,
              global_size=1,
              local_size=1,
              kernel_time_ns=nanosecond_identity(elapsed)))
      app.Log(2, 'Extracted clEnqueueTask from log')
    elif opcode == "clCreateBuffer":
      size, _, flags = operands
      size = int(size)
      flags = flags.split("|")
      if "CL_MEM_COPY_HOST_PTR" in flags and "CL_MEM_READ_ONLY" not in flags:
        # Device <-> host.
        total_transferred_bytes += size * 2
      else:
        # Host -> Device, or Device -> host.
        total_transferred_bytes += size
      app.Log(2, 'Extracted clCreateBuffer from log')
    elif (opcode == "clEnqueueReadBuffer" or opcode == "clEnqueueWriteBuffer" or
          opcode == "clEnqueueMapBuffer"):
      _, size, elapsed = operands
      total_transfer_time += nanosecond_identity(elapsed)
    else:
      # Not a line that we're interested in.
      pass

  # Defer transfer overhead until we have computed it.
  for ki in kernel_invocations:
    ki.transferred_bytes = total_transferred_bytes
    ki.transfer_time_ns = total_transfer_time

  return kernel_invocations


class StderrComponents(typing.NamedTuple):
  stderr: str
  cecl_log: str
  program_sources: typing.List[str]


def SplitStderrComponents(stderr: str) -> StderrComponents:
  """Split stderr output into components."""
  program_sources = []
  current_program_source: typing.Optional[typing.List[str]] = None

  # Split libcecl logs out from stderr.
  cecl_lines, stderr_lines = [], []
  in_program_source = False
  for line in stderr.split('\n'):
    if line == '[CECL] BEGIN PROGRAM SOURCE':
      assert not in_program_source
      in_program_source = True
      current_program_source = []
    elif line == '[CECL] END PROGRAM SOURCE':
      assert in_program_source
      in_program_source = False
      program_sources.append('\n'.join(current_program_source).strip())
      current_program_source = None
    elif line.startswith('[CECL] '):
      stripped_line = line[len('[CECL] '):].strip()
      if in_program_source:
        # Right strip program sources only, don't left strip since that would
        # lose indentation.
        current_program_source.append(line[len('[CECL] '):].rstrip())
      elif stripped_line:
        cecl_lines.append(stripped_line)
    elif line.strip():
      stderr_lines.append(line.strip())

  return StderrComponents(stderr_lines, cecl_lines, program_sources)


def RunEnv(cldrive_environment: cldrive_env.OpenCLEnvironment,
           os_env: typing.Optional[typing.Dict[str, str]] = None
          ) -> typing.Dict[str, str]:
  """Return an execution environment for a libcecl benchmark."""
  env = (os_env or os.environ).copy()
  env.update(libcecl_compile.LibCeclExecutableEnvironmentVariables())
  env['LIBCECL_DEVICE'] = cldrive_environment.device_name
  env['LIBCECL_PLATFORM'] = cldrive_environment.platform_name
  return env


def RunLibceclExecutable(
    command: typing.List[str],
    env: cldrive_env.OpenCLEnvironment,
    os_env: typing.Optional[typing.Dict[str, str]] = None,
    record_outputs: bool = True) -> libcecl_pb2.LibceclExecutableRun:
  """Run executable using libcecl and log output."""
  timestamp = labdate.MillisecondsTimestamp()

  os_env = RunEnv(env, os_env)
  start_time = time.time()
  process = env.Exec(command, env=os_env)
  elapsed = time.time() - start_time

  stderr_lines, cecl_lines, program_sources = SplitStderrComponents(
      process.stderr)

  return libcecl_pb2.LibceclExecutableRun(
      ms_since_unix_epoch=timestamp,
      returncode=process.returncode,
      stdout=process.stdout if record_outputs else '',
      stderr='\n'.join(stderr_lines) if record_outputs else '',
      cecl_log='\n'.join(cecl_lines) if record_outputs else '',
      device=env.proto,
      kernel_invocation=KernelInvocationsFromCeclLog(
          cecl_lines,
          expected_devtype=env.device_type,
          expected_device_name=env.device_name),
      elapsed_time_ns=int(elapsed * 1e9),
      opencl_program_source=program_sources)
