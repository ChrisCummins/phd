from unittest import main

from labm8.tests.testutil import TestCase
from omnitune import llvm

from labm8 import fs


class TestLLVM(TestCase):
  LLVM_PATH = fs.path("~/src/msc-thesis/skelcl/libraries/llvm/build/bin/")

  # assert_program_exists():
  def test_assert_program_exists(self):
    self._test(None, llvm.assert_program_exists(__file__))

  def test_assert_program_exists_fail(self):
    self.assertRaises(llvm.ProgramNotFoundError, llvm.assert_program_exists,
                      "/not a real path")

  # bitcode()
  def test_bitcode_cl(self):
    self._test(
        self.stencil_gaussian_kernel_bc,
        llvm.bitcode(
            self.stencil_gaussian_kernel, language="cl", path=self.LLVM_PATH))

  def test_bitcode_error_bad_src(self):
    self.assertRaises(
        llvm.ClangError, llvm.bitcode, "<NOT REAL CODE>", path=self.LLVM_PATH)

  def test_bitcode_error_bad_lang(self):
    self.assertRaises(
        llvm.ClangError,
        llvm.bitcode,
        self.stencil_gaussian_kernel,
        language="foobar",
        path=self.LLVM_PATH)

  def test_bitcode_missing_clang(self):
    self.assertRaises(
        llvm.ProgramNotFoundError, llvm.bitcode, "", path="/not a real path")

  # parse_instcounts()
  def test_parse_isntcounts(self):
    self._test(self.stencil_gaussian_kernel_ic_json,
               llvm.parse_instcounts(self.stencil_gaussian_kernel_ic))

  def test_parse_isntcounts_empty(self):
    self._test({}, llvm.parse_instcounts(""))

  # instcounts()
  def test_instcounts_cl(self):
    self._test(
        self.stencil_gaussian_kernel_ic_json,
        llvm.instcounts(self.stencil_gaussian_kernel_bc, path=self.LLVM_PATH))

  def test_instcounds_missing_opt(self):
    self.assertRaises(
        llvm.ProgramNotFoundError, llvm.instcounts, "", path="/not a real path")

  # instcounts2ratios()
  def test_instcounts2ratios(self):
    self._test(self.stencil_gaussian_kernel_ratios_json,
               llvm.instcounts2ratios(self.stencil_gaussian_kernel_ic_json))


if __name__ == '__main__':
  main()
