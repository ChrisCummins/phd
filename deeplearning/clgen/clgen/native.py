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
"""
Paths to native files.

Attributes
----------
CLANG : str
    Path to clang binary.
CLANG_FORMAT : str
    Path to clang-format binary.
CLGEN_FEATURES : str
    Path to clgen-features binary.
CLGEN_REWRITER : str
    Path to clgen-rewriter binary.
GPUVERIFY : str
    Path to GPUVerify.
LIBCLC : str
    Path to libclc directory.
OCLGRIND : str
     Path to OCLgrind.
OPT : str
    Path to LLVM opt binary.
SHIMFILE : str
    Path to shim headerfile.
"""
from deeplearning.clgen import clgen
from lib.labm8 import fs


CLANG = clgen.data_path(fs.path("bin", "clang"))
CLANG_FORMAT = clgen.data_path(fs.path("bin", "clang-format"))
CLGEN_FEATURES = clgen.data_path(fs.path("bin", "clgen-features"))
CLGEN_REWRITER = clgen.data_path(fs.path("bin", "clgen-rewriter"))
GPUVERIFY = clgen.data_path(fs.path("gpuverify", "gpuverify"))
LIBCLC = clgen.data_path("libclc")
try:
  OCLGRIND = clgen.data_path(fs.path("oclgrind", "bin", "oclgrind"))
except clgen.File404:
  pass  # FIXME: oclgrind is broken on Travis CI.
OPT = clgen.data_path(fs.path("bin", "opt"))
SHIMFILE = clgen.data_path(fs.path("include", "opencl-shim.h"))
