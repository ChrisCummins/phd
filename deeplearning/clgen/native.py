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

Attributes:
    CLANG: Path to clang binary.
    CLANG_FORMAT: Path to clang-format binary.
    OPT: Path to LLVM opt binary.
    LIBCLANG_SO: Path to LLVM libclang.so, if it exists. On macOS, this does
      not exist, and is set to None.
    CLGEN_FEATURES: Path to clgen-features binary.
    CLGEN_REWRITER: Path to clgen-rewriter binary.
    GPUVERIFY: Path to GPUVerify.
    LIBCLC: Path to libclc directory.
    OCLGRIND:  Path to OCLgrind.
    SHIMFILE: Path to shim headerfile.
"""

from config import getconfig
from deeplearning.clgen import package_util
from lib.labm8 import fs


_config = getconfig.GetGlobalConfig()
CLANG = package_util.must_exist(_config.paths.cc)
CLANG_FORMAT = package_util.must_exist(_config.paths.clang_format)
OPT = package_util.must_exist(_config.paths.opt)
LIBCLANG_SO = None
if _config.paths.libclang_so:
  LIBCLANG_SO = package_util.must_exist(_config.paths.libclang_so)
CLGEN_FEATURES = package_util.must_exist(
  fs.abspath('deeplearning/clgen/native/clgen-features'))
CLGEN_REWRITER = package_util.must_exist(
  fs.abspath('deeplearning/clgen/native/clgen-rewriter'))
LIBCLC = package_util.must_exist(
  fs.abspath('third_party/libclc/generic/include'))
SHIMFILE = package_util.must_exist(
  fs.abspath('deeplearning/clgen/data/include/opencl-shim.h'))
# TODO(cec): Add these remaining files.
GPUVERIFY = 'TODO'
OCLGRIND = 'TODO'
