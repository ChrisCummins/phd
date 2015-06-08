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

        # Copy and load test datasets.
        fs.cp("tests/data/skelcl.db", "/tmp/skelcl.db")
        self.db = db.Database("/tmp/skelcl.db")

        fs.cp("tests/data/skelcl.tiny.db", "/tmp/skelcl.tiny.db")
        self.db_tiny = db.Database("/tmp/skelcl.tiny.db")

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

    def test_mldatabase_init_from_db(self):
        test = db.MLDatabase.init_from_db("/tmp/ml.db", self.db_tiny)

        self._test(self.db_tiny.num_rows("kernels"),
                   test.num_rows("kernel_features"))
        self._test(self.db_tiny.num_rows("devices"),
                   test.num_rows("device_features"))
        self._test(self.db_tiny.num_rows("datasets"),
                   test.num_rows("dataset_features"))
        self._test(True, test.num_rows("runtime_stats") > 0)
        self._test(True, test.num_rows("oracle_params") > 0)


if __name__ == '__main__':
    main()
