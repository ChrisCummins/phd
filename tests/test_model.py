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

import json
import re

from labm8 import fs
from six import string_types
from tempfile import mkdtemp

import clgen
from clgen import cache
from clgen import model


def get_test_model():
    return model.from_json({
        "corpus": {
            "path": tests.data_path("tiny", "corpus"),
        },
        "train_opts": {
            "model_type": "lstm",
            "rnn_size": 8,
            "num_layers": 2,
            "max_epochs": 1
        }
    })


class TestModel(TestCase):
    def test_most_recent_checkpoint_untrained(self):
        m = get_test_model()
        m.cache.empty()  # untrain
        self.assertEqual(m.most_recent_checkpoint, None)

    def test_meta(self):
        m = get_test_model()
        m.train()
        meta = m.meta

        # meta format spec: https://github.com/ChrisCummins/clgen/issues/25

        # version
        self.assertEqual(meta["version"], clgen.version())
        # author
        self.assertTrue(isinstance(meta["author"], string_types))
        self.assertNotEqual(meta["author"], "")
        # date packaged
        self.assertTrue(isinstance(meta["date packaged"], string_types))
        self.assertNotEqual(meta["date packaged"], "")
        # contents
        contents = meta["contents"]

        # compare meta checksums to files
        for file in contents:
            path = fs.path(cache.ROOT, file)
            checksum = clgen.checksum_file(path)
            self.assertEqual(checksum, contents[file])

        # train opts
        self.assertEqual(meta["train_opts"], m.train_opts)

    def test_to_dist(self):
        m = get_test_model()
        m.train()

        tmpdir = mkdtemp(prefix="clgen-")
        outpath = m.to_dist(fs.path(tmpdir, "dist"))
        self.assertTrue(fs.isfile(outpath))
        self.assertEqual(fs.dirname(outpath), tmpdir)
        fs.rm(tmpdir)


class TestDistModel(TestCase):
    def test_import(self):
        m = get_test_model()
        m.train()

        train_opts = m.train_opts

        tmpdir = mkdtemp(prefix="clgen-")
        outpath = m.to_dist(fs.path(tmpdir, "dist"))

        m = model.from_tar(outpath)
        self.assertEqual(type(m), model.DistModel)
        self.assertEqual(m.train_opts, train_opts)
        fs.rm(tmpdir)

    def test_import_bad_path(self):
        with self.assertRaises(clgen.File404):
            model.from_tar("/bad/path")
