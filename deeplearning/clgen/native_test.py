#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
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
