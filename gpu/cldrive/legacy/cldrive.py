# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Drive arbitrary OpenCL kernels.

Reads an OpenCL kernel from stdin, generates data for it, executes it on a
suitable device, and prints the outputs.
"""
import io
import pickle
import sys

import numpy as np

from gpu.cldrive.legacy import args
from gpu.cldrive.legacy import cgen
from gpu.cldrive.legacy import data
from gpu.cldrive.legacy import driver
from gpu.cldrive.legacy import env
from labm8 import app

FLAGS = app.FLAGS
app.DEFINE_boolean(
    'ls_env', False,
    'If set, list the names and details of available OpenCL environments, and '
    'exit.')
app.DEFINE_boolean('emit_c', False, 'Generate standalone C code.')
app.DEFINE_boolean(
    'compile_only', False,
    'If --emit_c, generate standalone code C code which only compiles kernel.')
app.DEFINE_boolean(
    'with_kernel', False,
    'If --compile_only, this creates kernel object after compilation.')
app.DEFINE_string(
    'platform', None,
    "Specify the OpenCL platform name to use, e.g. 'NVIDIA CUDA'")
app.DEFINE_string(
    'device', None,
    "Specify the OpenCL device name to use, e.g. 'GeForce GTX 1080'")
app.DEFINE_string('devtype', "all",
                  "Use any OpenCL device of type: {all,cpu,gpu}.")
app.DEFINE_integer('size', 64, 'Size of the input arrays to generate.')
app.DEFINE_string(
    'generator', 'arange',
    'The input generator to use, one of: {rand,arange,zeros,ones}.')
app.DEFINE_float('scalar_val', None, 'Values to assign to scalar inputs.')
app.DEFINE_string("gsize", "64,1,1", "Comma separated NDRange for global size.")
app.DEFINE_string("lsize", "32,1,1",
                  "Comma separated NDRange for local (workgroup) size.")
app.DEFINE_integer(
    "timeout", -1,
    "Error if execution has not completed after this many seconds.")
app.DEFINE_boolean("cl_opt", True, "Whether OpenCL optimizations are enabled.")
app.DEFINE_boolean("profiling", False, "Enable kernel and transfer profiling.")
app.DEFINE_boolean("debug", False,
                   "Enable more verbose OpenCL compilation and execution.")
app.DEFINE_boolean("binary", False,
                   "Print outputs as a pickled binary numpy array.")


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    app.Warning("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  if FLAGS.ls_env:
    env.PrintOpenClEnvironments()

  # Read kernel source.
  src = sys.stdin.read()

  # Parse inputs from strings.
  gsize = driver.NDRange.FromString(FLAGS.gsize)
  lsize = driver.NDRange.FromString(FLAGS.lsize)
  data_generator = data.Generator.FromString(FLAGS.generator)
  env_ = env.make_env(devtype=FLAGS.devtype,
                      platform=FLAGS.platform,
                      device=FLAGS.device)

  if FLAGS.compile_only:
    inputs = []
  else:
    inputs = data.MakeData(src=src,
                           size=FLAGS.size,
                           data_generator=data_generator,
                           scalar_val=FLAGS.scalar_val)

  drive_args = {
      "src": src,
      "inputs": inputs,
      "gsize": gsize,
      "lsize": lsize,
      "optimizations": not FLAGS.cl_opt,
      "profiling": FLAGS.profiling,
      "debug": FLAGS.debug,
      "timeout": FLAGS.timeout,
  }

  if FLAGS.emit_c:
    emit_c_args = {
        "compile_only": FLAGS.compile_only,
        "create_kernel": FLAGS.with_kernel,
    }

    print(cgen.emit_c(**drive_args, **emit_c_args))
  else:
    outputs = driver.DriveKernel(**drive_args, env=env_)

    # Print result.
    if FLAGS.binary:
      d = pickle.dumps(outputs)
      sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='latin-1')
      print(d.decode('latin-1'), end='', flush=True)
    else:
      np.set_printoptions(threshold=np.nan)
      args_ = [
          arg for arg in args.GetKernelArguments(src)
          if not arg.address_space == 'local'
      ]
      assert (len(args_) == len(outputs))
      for arr, arg in zip(outputs, args_):
        print(f"{arg.name}: {arr}")


if __name__ == '__main__':
  app.RunWithArgs(main)
