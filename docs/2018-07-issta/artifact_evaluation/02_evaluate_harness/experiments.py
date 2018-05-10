"""This file defines TODO:

TODO: Detailed explanation of this file.
"""

from absl import app
from absl import flags

from deeplearning.deepsmith.proto import deepsmith_pb2
from gpu import cldrive
from lib.labm8 import fmt
from lib.labm8 import system

FLAGS = flags.FLAGS

flags.DEFINE_string('opencl_device', '1+',
                    'The OpenCL device to run the experiments on.')
flags.DEFINE_list('testcase_dirs',
                  ['./02_evaluate_harness/data/testcases'],
                  'Directories to read testcases from.')
flags.DEFINE_string('output_dir',
                    './02_evaluate_harness/run/results',
                    'Directory to write results to.')


def CldriveEnvToTestbed(env: cldrive.OpenCLEnvironment,
                        optimizations: bool) -> deepsmith_pb2.Testbed:
  testbed = deepsmith_pb2.Testbed()

  def _Escape(x): return '.'.join(x.lower().split())

  testbed.name = '_'.join([
    _Escape(env.platform_name), _Escape(env.device_type),
    _Escape(env.device_name), _Escape(env.driver_version)
  ])
  testbed.toolchain = 'opencl'
  testbed.opts['driver_version'] = env.driver_version
  testbed.opts['host'] = system.HOSTNAME
  testbed.opts['opencl_device'] = env.device_name
  testbed.opts['opencl_devtype'] = env.device_type
  testbed.opts['opencl_opt'] = 'true' if optimizations else 'false'
  testbed.opts['opencl_platform'] = env.platform_name
  testbed.opts['opencl_version'] = env.opencl_version
  return testbed


def GetTestbed(arg: str) -> deepsmith_pb2.Testbed:
  """Lookup and return the requested testbed."""
  _num, _opt = arg[0], arg[1]
  num = int(_num)
  if _opt == '+':
    optimizations = True
  elif _opt == '-':
    optimizations = False
  else:
    raise Exception
  env = list(cldrive.all_envs())[num - 1]
  return CldriveEnvToTestbed(env, optimizations)


def main(argv):
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f"Unknown arguments {unknown_args}")

  try:
    testbed = GetTestbed(FLAGS.opencl_device)
  except:
    raise app.UsageError(f"Unknown OpenCL device '{FLAGS.opencl_device}''")

  print("Running experiments on OpenCL device:")
  print(fmt.Indent(2, testbed))


if __name__ == '__main__':
  app.run(main)
