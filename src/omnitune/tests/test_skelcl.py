from unittest import main
from tests import TestCase

import omnitune
from omnitune import skelcl

class TestSkelCL(TestCase):

    # checksum_str()
    def test_checksum_str(self):
        self._test("a9993e364706816aba3e25717850c26c9cd0d89d",
                   skelcl.checksum_str("abc"))
        self._test("835fcc99584b3e47546bd1819a157831a4fcf0e2",
                   skelcl.checksum_str("a\nc"))
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   skelcl.checksum_str(""))
        self._test("35b1e342a8662025ddb60a9f7867bbadd8d60ef1",
                   skelcl.checksum_str(self.stencil_gaussian_kernel))


if __name__ == '__main__':
    main()
