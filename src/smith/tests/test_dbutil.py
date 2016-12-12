from unittest import TestCase
import tests

import os

import smith
from smith import dbutil

class TestDbutil(TestCase):
    def test_table_exists(self):
        self.assertTrue(dbutil.table_exists(
            tests.db('empty'), 'ContentFiles'))
        self.assertFalse(dbutil.table_exists(
            tests.db('empty'), 'NotATable'))

    def test_is_github(self):
        self.assertFalse(dbutil.is_github(tests.db('empty')))
        self.assertTrue(dbutil.is_github(tests.db('empty-gh')))


if __name__ == '__main__':
    main()
