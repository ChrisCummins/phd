"""OpenCL feature extraction.

Example usage:

    $ bazel run //gpu/portable_mapping_of_data_parallel_programs_to_opencl:feature_extractor
        -- --feature_extractor_opencl_src_path=/tmp/foo.cl
"""
import collections
import os
import pathlib
import subprocess
import typing

from absl import app
from absl import flags
from absl import logging

from labm8 import bazelutil


FEATURE_EXTRACTOR_BINARY = bazelutil.DataPath(
    'phd/gpu/portable_mapping_of_data_parallel_programs_to_opencl/'
    'feature_extractor_binary')

INLINED_OPENCL_HEADER = bazelutil.DataPath(
    'phd/third_party/opencl/inlined/include/cl.h')

# On Linux we must preload the LLVM shared libraries.
FEATURE_EXTRACTOR_ENV = os.environ.copy()
_LIBCLANG_SO = bazelutil.DataPath(
    'llvm_linux/lib/libclang.so', must_exist=False)
_LIBLTO_SO = bazelutil.DataPath(
    'llvm_linux/lib/libLTO.so', must_exist=False)
if _LIBCLANG_SO.is_file() and _LIBLTO_SO.is_file():
  FEATURE_EXTRACTOR_ENV['LD_PRELOAD'] = f'{_LIBCLANG_SO}:{_LIBLTO_SO}'

FLAGS = flags.FLAGS

flags.DEFINE_string('feature_extractor_opencl_src_path', None,
                    'Path of OpenCL file to extract features of.')

# The definition of features extracted by the feature extractor.
GreweEtAlFeatures = collections.namedtuple('GreweEtAlFeatures', [
  'file',  # str: the base name of the file path.
  'kernel_name',  # str: name of the kernel, e.g. `kernel void A() {}` -> `A`.
  'compute_operation_count',  # int: compute operation count
  'rational_operation_count',  # int: rational operation count
  'global_memory_access_count',  # int: accesses to global memory
  'local_memory_access_count',  # int: accesses to local memory
  'coalesced_memory_access_count',  # int: coalesced memory accesses
  'atomic_operation_count',  # int: atomic operations
  'coalesced_memory_access_ratio',  # float: derived feature
  'compute_to_memory_access_ratio',  # float: derived feature
])


def _GreweEtAlFeatures_from_binary_output(
    file, kernel_name, compute_operation_count, rational_operation_count,
    global_memory_access_count, local_memory_access_count,
    coalesced_memory_access_count, atomic_operation_count,
    coalesced_memory_access_ratio,
    compute_to_memory_access_ratio) -> GreweEtAlFeatures:
  """Tuple constructor which converts types."""
  return GreweEtAlFeatures(
      str(file), str(kernel_name), int(compute_operation_count),
      int(rational_operation_count), int(global_memory_access_count),
      int(local_memory_access_count), int(coalesced_memory_access_count),
      int(atomic_operation_count), float(coalesced_memory_access_ratio),
      float(compute_to_memory_access_ratio))


GreweEtAlFeatures._from_binary_output = _GreweEtAlFeatures_from_binary_output


class FeatureExtractionError(ValueError):
  """ Thrown in case feature extraction fails """
  pass


# TODO(polyglot): Add support for multiple languages.
def ExtractFeaturesFromPath(
    path: pathlib.Path,
    extra_args: typing.Optional[typing.List[str]] = None
) -> typing.Iterator[GreweEtAlFeatures]:
  """Print CSV format features of file.

  Args:
    path: Path of OpenCL kernel file.
    extra_args: Additional args to pass to the compiler.

  Returns:
    An iterator of GreweEtAlFeatures tuples.

  Raises:
    FileNotFoundError: If path does not exist.
    FeatureExtractionError: If feature extraction fails.
  """
  extra_args = extra_args or []

  if not path.is_file():
    raise FileNotFoundError(f"File not found: {path}")

  cmd = [str(FEATURE_EXTRACTOR_BINARY), str(INLINED_OPENCL_HEADER),
         str(path)] + [f'-extra_arg={arg}' for arg in extra_args]
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      env=FEATURE_EXTRACTOR_ENV, universal_newlines=True)
  stdout, stderr = process.communicate()

  lines = [line.split(',') for line in stdout.split('\n')]

  if process.returncode:
    raise FeatureExtractionError(
        f"Feature extracted exited with return code {process.returncode}: "
        f"{stderr}")

  if "error: " in stderr:
    raise FeatureExtractionError(
        f"Failed to extract features from {path}: {stderr}")

  return (GreweEtAlFeatures._from_binary_output(*line) for line in lines[:-1])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError(f"Unknown args: {argv[1:]}")

  try:
    features = list(
        ExtractFeaturesFromPath(
            pathlib.Path(FLAGS.feature_extractor_opencl_src_path)))

    if features:
      print(*features[0]._fields, sep=',')
    for line in features:
      print(*line, sep=',')
  except FeatureExtractionError as e:
    logging.fatal(e)


if __name__ == '__main__':
  app.run(main)
