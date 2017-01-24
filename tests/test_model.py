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
from unittest import TestCase, skip, skipIf, main
import tests

from io import StringIO
from labm8 import fs
from six import string_types
from tempfile import mkdtemp

import clgen
from clgen import cache
from clgen import model


def get_test_model(vocab="char"):
    return model.from_json({
        "corpus": {
            "path": tests.data_path("tiny", "corpus"),
            "vocabulary": vocab
        },
        "architecture": {
          "rnn_size": 8,
          "num_layers": 2
        },
        "train_opts": {
            "epochs": 2
        }
    })


class TestModel(TestCase):
    def test_hash(self):
        m1 = model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            }
        })

        # same as m1, with explicit default opt:
        m2 = model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": True
            }
        })

        # different opt value:
        m3 = model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": False
            }
        })

        self.assertEqual(m1.hash, m2.hash)
        self.assertNotEqual(m2.hash, m3.hash)

    def test_checkpoint_path_untrained(self):
        m = get_test_model()
        m.cache.empty()  # untrain
        self.assertEqual(m.checkpoint_path, None)

    def test_meta(self):
        m = get_test_model()
        m.train()
        meta = m.meta

        # version
        self.assertEqual(meta["version"], clgen.version())
        # author
        self.assertTrue(isinstance(meta["author"], string_types))
        self.assertNotEqual(meta["author"], "")
        # date packaged
        self.assertTrue(isinstance(meta["date_packaged"], string_types))
        self.assertNotEqual(meta["date_packaged"], "")
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

    def test_sample_seed(self):
        m = get_test_model()
        m.train()

        # sample 20 chars three times
        buf1 = StringIO()
        m.sample(seed_text='__kernel void ', output=buf1, seed=204,
                 max_length=20)
        out1 = buf1.getvalue()
        print("OUT1", out1)
        for _ in range(2):
            buf = StringIO()
            m.sample(seed_text='__kernel void ', output=buf, seed=204,
                     max_length=20)
            out = buf.getvalue()
            print("OUT", out)
            self.assertEqual(out1, out)

    def test_sample_seed_greedy(self):
        m = get_test_model(vocab="greedy")
        m.train()

        buf1 = StringIO()
        m.sample(seed_text='__kernel void ', output=buf1, seed=204,
                 max_length=20)
        out1 = buf1.getvalue()
        print("OUT1", out1)
        for _ in range(2):
            buf = StringIO()
            m.sample(seed_text='__kernel void ', output=buf, seed=204,
                     max_length=20)
            out = buf.getvalue()
            print("OUT", out)
            self.assertEqual(out1, out)


if __name__ == "__main__":
    main()
