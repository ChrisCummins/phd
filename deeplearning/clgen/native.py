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

from deeplearning.clgen import package_util


CLANG = package_util.must_exist('../llvm_mac/bin/clang')
CLANG_FORMAT = package_util.must_exist('../llvm_mac/bin/clang-format')
CLGEN_FEATURES = package_util.must_exist(
  'deeplearning/clgen/native/clgen-features')
CLGEN_REWRITER = package_util.must_exist(
  'deeplearning/clgen/native/clgen-features')
# TODO(cec): Add GPUVerify.
GPUVERIFY = 'TODO'
LIBCLC = 'TODO'
OCLGRIND = 'TODO'
OPT = 'TODO'
SHIMFILE = package_util.must_exist(
  'deeplearning/clgen/data/include/opencl-shim.h')
