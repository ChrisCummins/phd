"""A python wrapper around bzip2 benchmark."""
import subprocess
import sys
import typing

from absl import app
from absl import flags
from absl import logging

from datasets.benchmarks.proto import benchmarks_pb2
from labm8 import bazelutil

FLAGS = flags.FLAGS

flags.DEFINE_integer('bzip2_timeout_seconds', 60,
                     'The maximum number of seconds to allow process to run.')

# Path to bzip2.
BZIP2 = bazelutil.DataPath('bzip2/bzip2')

# Source files for bzip2.
BZIP2_SRCS = [
    bazelutil.DataPath('bzip2/blocksort.c'),
    bazelutil.DataPath('bzip2/bzlib.c'),
    bazelutil.DataPath('bzip2/compress.c'),
    bazelutil.DataPath('bzip2/crctable.c'),
    bazelutil.DataPath('bzip2/decompress.c'),
    bazelutil.DataPath('bzip2/huffman.c'),
    bazelutil.DataPath('bzip2/randtable.c'),
    bazelutil.DataPath('bzip2/bzip2.c'),
]

# Header files for bzip2.
BZIP2_HEADERS = [
    bazelutil.DataPath('bzip2/bzlib.h'),
    bazelutil.DataPath('bzip2/bzlib_private.h'),
]


class Bzip2Timeout(EnvironmentError):
  """Eror raised in case of time out."""
  pass


def Exec(data: str, args: typing.List[str],
         timeout_seconds: int = 60) -> subprocess.Popen:
  """Run bzip2.

  Args:
    args: A list of arguments to pass to binary.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    A Popen instance with stdout and stderr set to strings.
  """
  cmd = ['timeout', '-s9', str(timeout_seconds), str(BZIP2)] + args
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      stdin=subprocess.PIPE)
  stdout, stderr = process.communicate(data)
  if process.returncode == 9:
    raise Bzip2Timeout(f'bzip2 timed out after {timeout_seconds}s')
  process.stdout = stdout
  process.stderr = stderr
  return process


Bzip2 = benchmarks_pb2.Benchmark(
    name='bzip2',
    binary=str(BZIP2),
    srcs=[str(x) for x in BZIP2_SRCS],
    hdrs=[str(x) for x in BZIP2_HEADERS],
)

# A list of all benchmarks in this file.
BENCHMARKS = [
    Bzip2,
]


def main(argv):
  """Main entry point."""
  try:
    proc = Exec(argv[1:], timeout_seconds=FLAGS.opt_timeout_seconds)
    if proc.stdout:
      print(proc.stdout)
    if proc.stderr:
      print(proc.stderr, file=sys.stderr)
    sys.exit(proc.returncode)
  except Bzip2Timeout as e:
    print(e, file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
