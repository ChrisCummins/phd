#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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

import numpy as np
import os
import sys

import labm8
from labm8 import fs

import clgen
from clgen import corpus

TINY_HASH = tests.data_str("tiny", "corpus.contents.sha1").rstrip()


class TestCorpus(TestCase):
    def test_path(self):
        path = tests.archive("tiny", "corpus")
        c = corpus.Corpus(path)
        self.assertEqual(TINY_HASH, c.hash)

    def test_badpath(self):
        with self.assertRaises(clgen.CLgenError):
            corpus.Corpus("notarealpath")

    def test_unpack_archive(self):
        # delete any existing unpacked directory
        fs.rm(tests.data_path("tiny", "corpus"))

        c = corpus.Corpus(tests.data_path("tiny", "corpus", exists=False))
        self.assertEqual(TINY_HASH, c.hash)

    def test_path_is_archive(self):
        # delete any existing unpacked directory
        fs.rm(tests.data_path("tiny", "corpus"))

        c = corpus.Corpus(tests.data_path("tiny", "corpus.tar.bz2"))
        self.assertEqual(TINY_HASH, c.hash)
