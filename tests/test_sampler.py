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

import json
import re

from labm8 import fs
from six import string_types
from tempfile import mkdtemp

import clgen
from clgen import dbutil
from clgen import model
from clgen import sampler


def get_test_model():
    return model.from_json({
        "corpus": {
            "path": tests.data_path("tiny", "corpus"),
        },
        "architecture": {
          "rnn_size": 8,
          "num_layers": 2,
        },
        "train_opts": {
          "epochs": 2
        }
    })


class TestSampler(TestCase):
    def test_sample(self):
        m = get_test_model()
        m.train()

        argspec = [
            '__global float*',
            '__global float*',
            '__global float*',
            'const int'
        ]
        s = sampler.from_json({
            "kernels": {
                "args": argspec,
                "max_length": 300,
            },
            "sampler": {
                "batch_size": 1,
                "max_batches": 1
            }
        })

        s.cache(m).empty()  # clear old samples

        # sample a single kernel:
        s.sample(m)
        nun_contentfiles = dbutil.num_rows_in(s.cache(m)["kernels.db"], "ContentFiles")
        num_preprocessed = dbutil.num_rows_in(s.cache(m)["kernels.db"], "PreProcessedFiles")
        self.assertEqual(nun_contentfiles, 1)
        self.assertEqual(num_preprocessed, 1)

        s.sample(m)
        nun_contentfiles = dbutil.num_rows_in(s.cache(m)["kernels.db"], "ContentFiles")
        num_preprocessed = dbutil.num_rows_in(s.cache(m)["kernels.db"], "PreProcessedFiles")
        # if sample is the same as previous, then there will still only be a
        # single sample in db:
        self.assertTrue(nun_contentfiles >= 1)
        self.assertTrue(num_preprocessed >= 1)


if __name__ == "__main__":
    main()
