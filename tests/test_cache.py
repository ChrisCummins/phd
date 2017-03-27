#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
from unittest import TestCase, skip, skipIf
import tests

from copy import deepcopy
from functools import partial

import os
import sys

import labm8
from labm8 import fs

import clgen
from clgen import cache


class TestCache(TestCase):
    def test_init_and_empty(self):
        c = cache.Cache("__test__")
        self.assertTrue(fs.isdir(fs.path(cache.ROOT, "__test__")))
        c.empty()
        self.assertFalse(fs.isdir(fs.path(cache.ROOT, "__test__")))

    def test_set_and_get(self):
        c = cache.Cache("__test_set_and_get__")

        # create file
        fs.write_file(tests.data_path("tmp", "file.txt", exists=False),
                      "Hello, world!")
        # sanity check
        self.assertEqual(fs.read_file(tests.data_path("tmp", "file.txt")),
                         "Hello, world!")

        # insert file into cache
        c['foobar'] = tests.data_path("tmp", "file.txt")

        # file must be in cache
        self.assertTrue(fs.isfile(fs.path(c.path, "foobar")))
        # file must have been moved
        self.assertFalse(fs.isfile(tests.data_path("file.txt", exists=False)))
        # check file contents
        self.assertTrue(fs.read_file(c['foobar']),
                        "Hello, world!")
        c.empty()

    def test_404(self):
        c = cache.Cache("__test_404__")
        self.assertFalse(c['foobar'])
        with self.assertRaises(cache.Cache404):
            del c['foobar']
        c.empty()

    def test_remove(self):
        c = cache.Cache("__test_remove__")

        # create file
        fs.write_file(tests.data_path("tmp", "file.txt", exists=False),
                      "Hello, world!")
        # sanity check
        self.assertEqual(fs.read_file(tests.data_path("tmp", "file.txt")),
                         "Hello, world!")

        # insert file into cache
        c['foobar'] = tests.data_path("tmp", "file.txt")

        # remove from cache
        del c['foobar']

        self.assertFalse(c['foobar'])
        c.empty()

    def test_path_escape(self):
        c = cache.Cache("__test_path_escape__")

        # create file
        fs.write_file(tests.data_path("tmp", "file.txt", exists=False),
                      "Hello, world!")
        # sanity check
        self.assertEqual(fs.read_file(tests.data_path("tmp", "file.txt")),
                         "Hello, world!")

        # insert file into cache
        key = 'this key/path needs: escaping!.?'
        c[key] = tests.data_path("tmp", "file.txt")

        # check file contents
        self.assertEqual(fs.read_file(c[key]), "Hello, world!")
        c.empty()
