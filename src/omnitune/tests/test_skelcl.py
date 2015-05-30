from unittest import main
from tests import TestCase

import omnitune
from omnitune import skelcl

class TestSkelCL(TestCase):


    def __init__(self, *args, **kwargs):
        super(TestSkelCL, self).__init__(*args, **kwargs)

        # Load test database.
        self.db = skelcl.SkelCLDatabase("tests/data/skelcl.db")

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
