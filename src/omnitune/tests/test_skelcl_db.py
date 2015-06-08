from unittest import main
from tests import TestCase

import itertools

import labm8 as lab
from labm8 import fs

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as db

class TestSkelCLDB(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSkelCLDB, self).__init__(*args, **kwargs)

        fs.cp("tests/data/skelcl.db", "/tmp/skelcl.db")
        self.db = db.Database("/tmp/skelcl.db")

    # create_test_db()
    def test_create_test_db(self):
        test = db.create_test_db("/tmp/skelcl.test.db", self.db,
                                 num_runtimes=100)
        # Check table sizes.
        self._test(100, test.num_rows("runtimes"))
        self._test(True, test.num_rows("devices") > 0)
        self._test(True, test.num_rows("devices") <= self.db.num_rows("devices"))
        self._test(True, test.num_rows("kernels") > 0)
        self._test(True, test.num_rows("kernels") <= self.db.num_rows("kernels"))
        self._test(True, test.num_rows("datasets") > 0)
        self._test(True, test.num_rows("datasets") <= self.db.num_rows("datasets"))

        test.close()
        fs.rm(test.path)


if __name__ == '__main__':
    main()
