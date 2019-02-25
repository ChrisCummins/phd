import itertools
from unittest import main

from labm8.tests.testutil import TestCase
from omnitune import skelcl


class TestSkelCL(TestCase):

  # hash_params()
  def test_hash_params(self):
    vals = range(4, 40, 4)
    wgs = itertools.product(vals, vals)
    checksums = [skelcl.hash_params(*wg) for wg in wgs]
    print(checksums)
    self._test(len(checksums), len(set(checksums)))

  # hash_dataset()
  def test_hash_dataset(self):
    vals = [[1024, 1024, "int", "float"], [1024, 2048, "int", "float"],
            [1024, 1024, "float", "float"], [1024, 1024, "int", "int"]]
    checksums = [skelcl.hash_dataset(*val) for val in vals]
    print(checksums)
    self._test(len(checksums), len(set(checksums)))

  # checksum_str()
  def test_checksum_str(self):
    self._test("a9993e364706816aba3e25717850c26c9cd0d89d",
               skelcl.checksum_str("abc"))
    self._test("835fcc99584b3e47546bd1819a157831a4fcf0e2",
               skelcl.checksum_str("a\nc"))
    self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
               skelcl.checksum_str(""))
    self._test("9e97c70ba595f82d52b11d5602567c2410cf9b84",
               skelcl.checksum_str(self.stencil_gaussian_kernel))

  # get_user_source()
  def test_get_user_source(self):
    self._test(self.stencil_gaussian_kernel_user,
               skelcl.get_user_source(self.stencil_gaussian_kernel))


if __name__ == '__main__':
  main()
