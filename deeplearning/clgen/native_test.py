"""Unit tests for //deeplearning/clgen/native.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import native
from lib.labm8 import fs


BINARIES = [native.CLANG, native.CLANG_FORMAT, native.CLGEN_REWRITER,
            native.OPT]

FILES = [fs.path(native.LIBCLC, "clc", "clc.h"), native.SHIMFILE, ]


def test_binaries_exist():
  for binary in BINARIES:
    assert fs.isexe(binary)


def test_files_exist():
  for file in FILES:
    assert fs.isfile(file)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
