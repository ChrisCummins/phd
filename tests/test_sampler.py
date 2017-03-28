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
                "min_samples": 1
            }
        })

        s.cache(m).clear()  # clear old samples

        # sample a single kernel:
        s.sample(m)
        num_contentfiles = dbutil.num_rows_in(s.cache(m)["kernels.db"], "ContentFiles")
        num_preprocessed = dbutil.num_rows_in(s.cache(m)["kernels.db"], "PreProcessedFiles")
        self.assertTrue(num_contentfiles >= 1)
        self.assertTrue(num_preprocessed >= 1)

        s.sample(m)
        num_contentfiles2 = dbutil.num_rows_in(s.cache(m)["kernels.db"], "ContentFiles")
        num_preprocessed2 = dbutil.num_rows_in(s.cache(m)["kernels.db"], "PreProcessedFiles")
        diff = num_contentfiles2 - num_contentfiles
        # if sample is the same as previous, then there will still only be a
        # single sample in db:
        self.assertTrue(diff >= 1)
        self.assertTrue(num_preprocessed2 - num_preprocessed == diff)

    def test_eq(self):
        s1 = sampler.from_json({
            "kernels": {
                "args": [
                    '__global float*',
                    '__global float*',
                    'const int'
                ]
            }
        })
        s2 = sampler.from_json({
            "kernels": {
                "args": [
                    '__global float*',
                    '__global float*',
                    'const int'
                ]
            }
        })
        s3 = sampler.from_json({
            "kernels": {
                "args": [
                    'int'
                ]
            }
        })

        self.assertEqual(s1, s2)
        self.assertNotEqual(s2, s3)
        self.assertNotEqual(s1, False)
        self.assertNotEqual(s1, 'abcdef')

    def test_to_json(self):
        s1 = sampler.from_json({
            "kernels": {
                "args": [
                    '__global float*',
                    '__global float*',
                    'const int'
                ]
            }
        })
        s2 = sampler.from_json(s1.to_json())
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    main()
