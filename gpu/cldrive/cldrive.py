"""Drive arbitrary OpenCL kernels."""
import pathlib
import sys

from absl import app
from absl import flags
from absl import logging

from gpu.cldrive import api
from gpu.cldrive.legacy import env
from gpu.cldrive.proto import cldrive_pb2

FLAGS = flags.FLAGS
flags.DEFINE_string('src', None, 'Path to a file containing OpenCL kernels.')
flags.DEFINE_string(
    'env',
    env.OclgrindOpenCLEnvironment().name,
    "Specify the OpenCL device to use. Run `bazel run //gpu/clinfo` to see a "
    "list of available devices. Defaults to using the builtin CPU simulator.")
flags.DEFINE_integer("gsize", 64, "The global size to use.")
flags.DEFINE_integer("lsize", 32, "The local (workgroup) size.")
flags.DEFINE_integer("timeout", 60,
                     "Terminate execution after this many seconds.")
flags.DEFINE_boolean("cl_opt", True,
                     "Whether OpenCL optimizations are enabled.")
flags.DEFINE_integer("num_runs", 10, "The number of runs per kernel.")


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    logging.warning("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  try:
    opencl_environment = env.OpenCLEnvironment.FromName(FLAGS.env)
  except KeyError as e:
    print(e, file=sys.stderr)
    sys.exit(1)

  src_path = pathlib.Path(FLAGS.src)
  if not src_path.is_file():
    print("File not found:", src_path, file=sys.stderr)
    sys.exit(1)
  with open(src_path) as f:
    opencl_kernel = f.read()

  instance = cldrive_pb2.CldriveInstance(
      device=opencl_environment.proto,
      opencl_src=opencl_kernel,
      min_runs_per_kernel=FLAGS.num_runs,
      dynamic_params=[
          cldrive_pb2.DynamicParams(
              global_size_x=FLAGS.gsize, local_size_x=FLAGS.lsize)
      ])

  print(api.Drive(instance))


if __name__ == '__main__':
  app.run(main)
