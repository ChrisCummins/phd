"""A python wrapper around cl_launcher, the CLSmith program driver.

CLSmith is developed by Chris Lidbury <christopher.lidbury10imperial.ac.uk>.
See https://github.com/ChrisLidbury/CLSmith.

This file can be executed as a binary in order to invoke CLSmith. Note you must
use '--' to prevent this script from attempting to parse the args, and a second
'--' if invoked using bazel, to prevent bazel from parsing the args.

Usage:

  bazel run //compilers/clsmith:cl_launcher [-- -- <args>]
"""
import os
import subprocess
import sys
import typing

from absl import app
from absl import flags

from gpu.cldrive.legacy import driver
from gpu.cldrive.legacy import env
from labm8 import bazelutil
from labm8 import fs
from labm8 import system

FLAGS = flags.FLAGS

# The path to cl_launcher.
CL_LAUNCHER = bazelutil.DataPath('CLSmith/cl_launcher')

# The header files required by generated CLProg.c files.
CL_LAUNCHER_RUN_FILES = [
    bazelutil.DataPath('CLSmith/cl_safe_math_macros.h'),
    bazelutil.DataPath('CLSmith/safe_math_macros.h'),
    bazelutil.DataPath('CLSmith/runtime/CLSmith.h'),
    bazelutil.DataPath('CLSmith/runtime/csmith.h'),
    bazelutil.DataPath('CLSmith/runtime/csmith_minimal.h'),
    bazelutil.DataPath('CLSmith/runtime/custom_limits.h'),
    bazelutil.DataPath('CLSmith/runtime/custom_stdint_x86.h'),
    bazelutil.DataPath('CLSmith/runtime/platform_avr.h'),
    bazelutil.DataPath('CLSmith/runtime/platform_generic.h'),
    bazelutil.DataPath('CLSmith/runtime/platform_msp430.h'),
    bazelutil.DataPath('CLSmith/runtime/random_inc.h'),
    bazelutil.DataPath('CLSmith/runtime/safe_abbrev.h'),
    bazelutil.DataPath('CLSmith/runtime/stdint_avr.h'),
    bazelutil.DataPath('CLSmith/runtime/stdint_ia32.h'),
    bazelutil.DataPath('CLSmith/runtime/stdint_ia64.h'),
    bazelutil.DataPath('CLSmith/runtime/stdint_msp430.h'),
    bazelutil.DataPath('CLSmith/runtime/volatile_runtime.h')
]

# The path to libOpenCL.so, needed on Linux.
if system.is_linux():
  LIBOPENCL_SO = bazelutil.DataPath('libopencl/libOpenCL.so')


def PrependToPath(path_to_prepend: str, path: typing.Optional[str]) -> str:
  """Prepend a path to a path environment variable.

  This uses ':' to separate path components, as used in system $PATH variables.

  Args:
    path_to_prepend: The path to prepend.
    path: The environment variable.

  Returns:
    A concatenation of the path to prepend and the existing path.
  """
  if path:
    return f'{path_to_prepend}:{path}'
  else:
    return path_to_prepend


def Exec(opencl_environment: env.OpenCLEnvironment,
         *opts,
         timeout_seconds: int = 60,
         env: typing.Dict[str, str] = None) -> subprocess.Popen:
  """Execute cl_launcher.

  This creates a Popen process, executes it, and sets the stdout and stderr
  attributes to the process output.

  Args:
    opencl_environment: The OpenCL environment to execute cl_launcher with.
    opts: A list of arguments to pass to the cl_launcher binary.
    timeout_seconds: The maximum number of seconds to execute cl_launcher for.
    env: An optional environment to run cl_launcher under.

  Returns:
    A Popen instance, with string stdout and stderr attributes set.
  """
  if system.is_linux():
    env = env or {}
    env['LD_PRELOAD'] = PrependToPath(str(LIBOPENCL_SO), env.get('LD_PRELOAD'))
  with fs.TemporaryWorkingDir(prefix='cl_launcher_') as d:
    for src in CL_LAUNCHER_RUN_FILES:
      os.symlink(src, str(d / src.name))
    cmd = [
        'timeout', '-s9',
        str(timeout_seconds),
        str(CL_LAUNCHER), '-i',
        str(d)
    ] + list(opts)
    process = opencl_environment.Exec(cmd, env=env)
  return process


def ExecClsmithSource(opencl_environment: env.OpenCLEnvironment,
                      src: str,
                      gsize: driver.NDRange,
                      lsize: driver.NDRange,
                      *opts,
                      timeout_seconds: int = 60,
                      env: typing.Dict[str, str] = None) -> subprocess.Popen:
  """Execute a CLsmith source program using cl_launcher.

  This creates a Popen process, executes it, and sets the stdout and stderr
  attributes to the process output.

  Args:
    opencl_environment: The OpenCL environment to execute.
    src: The CLSmith program src.
    gsize: Kernel global size.
    lsize: Kernel local size.
    opts: A list of optional command line flags.
    timeout_seconds: The maximum number of seconds to execute cl_launcher for.
    env: An optional environment to run the program under.

  Returns:
    A Popen instance, with string stdout and stderr attributes set.
  """
  platform_id, device_id = opencl_environment.ids()
  with fs.TemporaryWorkingDir(prefix='cl_launcher_') as d:
    with open(d / 'CLProg.c', 'w') as f:
      f.write(src)
    proc = Exec(
        opencl_environment,
        '-f',
        str(d / 'CLProg.c'),
        '-g',
        gsize.ToString(),
        '-l',
        lsize.ToString(),
        '-p',
        str(platform_id),
        '-d',
        str(device_id),
        *opts,
        timeout_seconds=timeout_seconds,
        env=env)
  return proc


def main(argv):
  """Main entry point."""
  proc = Exec(*argv[1:])
  print(proc.stdout)
  print(proc.stderr, file=sys.stderr)
  sys.exit(proc.returncode)


if __name__ == '__main__':
  app.run(main)
