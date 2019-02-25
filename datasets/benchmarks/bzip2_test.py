"""Unit tests for //datasets/benchmarks/bzip2.py."""

from absl import flags

from datasets.benchmarks import bzip2
from labm8 import test

FLAGS = flags.FLAGS


def test_Exec_compress_decompress():
  """Test compressing and de-compressing a string."""
  compress = bzip2.Exec('Hello, bzip2!'.encode('utf-8'), ['-z'])
  assert not compress.stderr
  assert not compress.returncode
  assert compress.stdout != 'Hello, bzip2!'.encode('utf-8')

  decompress = bzip2.Exec(compress.stdout, ['-d'])
  assert not decompress.stderr
  assert not decompress.returncode
  assert decompress.stdout == 'Hello, bzip2!'.encode('utf-8')


def test_BZIP2_SRCS():
  """Test source files for bzip2."""
  for path in bzip2.BZIP2_SRCS:
    assert path.is_file()


if __name__ == '__main__':
  test.Main()
