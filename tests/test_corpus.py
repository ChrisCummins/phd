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
from unittest import TestCase, skip
import tests

from labm8 import fs

import clgen
from clgen import corpus

TINY_HASH = '1fd1cf76b997b4ae163fd7b2cd3e3d94beb281ff'


class TestCorpus(TestCase):
    def test_path(self):
        path = tests.archive("tiny", "corpus")
        c = corpus.Corpus.from_json({"id": TINY_HASH, "path": path})
        self.assertEqual(TINY_HASH, c.hash)

    def test_badpath(self):
        with self.assertRaises(clgen.CLgenError):
            corpus.Corpus("notarealid", path="notarealpath")

    def test_from_archive(self):
        # delete any existing unpacked directory
        fs.rm(tests.data_path("tiny", "corpus", exists=False))

        c = corpus.Corpus.from_json({
            "path": tests.data_path("tiny", "corpus", exists=False)
        })
        self.assertEqual(TINY_HASH, c.hash)

    def test_from_archive_path(self):
        # delete any existing unpacked directory
        fs.rm(tests.data_path("tiny", "corpus", exists=False))

        c = corpus.Corpus.from_json({
            "path": tests.data_path("tiny", "corpus.tar.bz2")
        })
        self.assertEqual(TINY_HASH, c.hash)

    def test_hash(self):
        c1 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus")
        })

        # same as c1, with explicit default opt:
        c2 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus"),
            "eof": False
        })

        # different opt value:
        c3 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus"),
            "eof": True
        })

        self.assertEqual(c1.hash, c2.hash)
        self.assertNotEqual(c2.hash, c3.hash)

    def test_get_features(self):
        code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}"""
        import numpy as np
        self.assertTrue(np.array_equal(corpus.get_features(code),
                                       [0, 0, 1, 0, 1, 0, 1, 0]))

    def test_get_features_bad_code(self):
        code = """\
__kernel void A(__global float* a) {
  SYNTAX ERROR!
}"""
        with tests.DevNullRedirect():
            with self.assertRaises(corpus.FeaturesError):
                corpus.get_features(code, quiet=True)

    def test_get_features_multiple_kernels(self):
        code = """\
__kernel void A(__global float* a) {}
__kernel void B(__global float* a) {}"""

        with self.assertRaises(corpus.FeaturesError):
            corpus.get_features(code)

    @skip("FIXME: UserError not raised")
    def test_bad_option(self):
        with self.assertRaises(clgen.UserError):
            corpus.Corpus.from_json({
                "path": tests.archive("tiny", "corpus"),
                "not_a_real_option": False
            })

    @skip("FIXME: UserError not raised")
    def test_bad_vocab(self):
        with self.assertRaises(clgen.UserError):
            corpus.Corpus.from_json({
                "path": tests.archive("tiny", "corpus"),
                "vocab": "INVALID_VOCAB"
            })

    @skip("FIXME: UserError not raised")
    def test_bad_encoding(self):
        with self.assertRaises(clgen.UserError):
            corpus.Corpus.from_json({
                "path": tests.archive("tiny", "corpus"),
                "encoding": "INVALID_ENCODING"
            })

    def test_eq(self):
        c1 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus"),
            "eof": False
        })
        c2 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus"),
            "eof": False
        })
        c3 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus"),
            "eof": True
        })

        self.assertEqual(c1, c2)
        self.assertNotEqual(c2, c3)
        self.assertNotEqual(c1, False)
        self.assertNotEqual(c1, 'abcdef')

    def test_preprocessed(self):
        c1 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus")
        })
        self.assertEqual(len(list(c1.preprocessed())), 187)
        self.assertEqual(len(list(c1.preprocessed(1))), 56)
        self.assertEqual(len(list(c1.preprocessed(2))), 7)

    def test_contentfiles(self):
        c1 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus")
        })
        self.assertEqual(len(list(c1.contentfiles())), 250)

    def test_to_json(self):
        c1 = corpus.Corpus.from_json({
            "path": tests.archive("tiny", "corpus")
        })
        c2 = corpus.Corpus.from_json(c1.to_json())
        self.assertEqual(c1, c2)
