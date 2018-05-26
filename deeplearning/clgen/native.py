"""Paths to native files.

Attributes:
  CLANG: Path to clang binary.
  CLANG_FORMAT: Path to clang-format binary.
  OPT: Path to LLVM opt binary.
  LIBCLANG_SO: Path to LLVM libclang.so, if it exists. On macOS, this does not
    exist, and is set to None.
  CLGEN_FEATURES: Path to clgen-features binary.
  CLGEN_REWRITER: Path to clgen-rewriter binary.
  GPUVERIFY: Path to GPUVerify.
  LIBCLC: Path to libclc directory.
  OCLGRIND:  Path to OCLgrind.
  SHIMFILE: Path to shim headerfile.
"""

from config import getconfig
from deeplearning.clgen import package_util
from lib.labm8 import bazelutil
from lib.labm8 import fs


_config = getconfig.GetGlobalConfig()
CLANG = package_util.must_exist(_config.paths.cc)
CLANG_FORMAT = package_util.must_exist(_config.paths.clang_format)
OPT = package_util.must_exist(_config.paths.opt)
LIBCLANG_SO = None
if _config.paths.libclang_so:
  LIBCLANG_SO = package_util.must_exist(_config.paths.libclang_so)
CLGEN_FEATURES = package_util.must_exist(
    bazelutil.DataPath('phd/deeplearning/clgen/native/clgen-features'))
CLGEN_REWRITER = package_util.must_exist(
    bazelutil.DataPath('phd/deeplearning/clgen/native/clgen-rewriter'))
LIBCLC = package_util.must_exist(
    bazelutil.DataPath('phd/third_party/libclc/generic/include'))
OPENCL_H = package_util.must_exist(
    bazelutil.DataPath('phd/deeplearning/clgen/data/include/opencl.h'))
SHIMFILE = package_util.must_exist(
    bazelutil.DataPath('phd/deeplearning/clgen/data/include/opencl-shim.h'))
# On Linux, we use @libcxx//:headers. On MacOS we use the homebrew libcxx
# headers since @libcxx//:headers raise compilation errors and appear to be
# incompatible.
if _config.uname == 'darwin':
  LIBCXX_HEADERS = package_util.must_exist(
      fs.abspath(_config.paths.llvm_prefix, 'include/c++/v1'))
else:
  LIBCXX_HEADERS = package_util.must_exist(bazelutil.DataPath('libcxx/include'))
# TODO(cec): Add these remaining files.
GPUVERIFY = 'TODO'
OCLGRIND = 'TODO'
