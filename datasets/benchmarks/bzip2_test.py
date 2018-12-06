"""Unit tests for //datasets/benchmarks/bzip2.py."""
import sys
import typing

import pytest
from absl import app
from absl import flags

from datasets.benchmarks import bzip2


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


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
