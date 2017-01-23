#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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

Variables:
    * `CLANG` (str): Path to clang binary.
    * `CLANG_FORMAT` (str): Path to clang-format binary.
    * `CLGEN_FEATURES` (str): Path to clgen-features binary.
    * `CLGEN_REWRITER` (str): Path to clgen-rewriter binary.
    * `LIBCLC` (str): Path to libclc directory.
    * `OPT` (str): Path to LLVM opt binary.
    * `SHIMFILE` (str): Path to shim headerfile.
"""
from labm8 import fs

import clgen


CLANG = clgen.data_path(fs.path("bin", "clang"))
CLANG_FORMAT = clgen.data_path(fs.path("bin", "clang-format"))
CLGEN_FEATURES = clgen.data_path(fs.path("bin", "clgen-features"))
CLGEN_REWRITER = clgen.data_path(fs.path("bin", "clgen-rewriter"))
LIBCLC = clgen.data_path("libclc")
OPT = clgen.data_path(fs.path("bin", "opt"))
SHIMFILE = clgen.data_path(fs.path("include", "opencl-shim.h"))
