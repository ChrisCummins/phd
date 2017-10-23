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
from clgen import test as tests

import clgen
from clgen import dbutil


def _get_test_model():
    return clgen.Model.from_json({
        "corpus": {
            "language": "opencl",
            "path": tests.data_path("tiny", "corpus"),
        },
        "architecture": {
          "rnn_size": 8,
          "num_layers": 2,
        },
        "train_opts": {
          "epochs": 1
        }
    })


def test_sample():
    m = _get_test_model()
    m.train()

    argspec = [
        '__global float*',
        '__global float*',
        '__global float*',
        'const int'
    ]
    s = clgen.Sampler.from_json({
        "kernels": {
            "language": "opencl",
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
    assert num_contentfiles >= 1
    assert num_preprocessed >= 1

    s.sample(m)
    num_contentfiles2 = dbutil.num_rows_in(s.cache(m)["kernels.db"], "ContentFiles")
    num_preprocessed2 = dbutil.num_rows_in(s.cache(m)["kernels.db"], "PreProcessedFiles")
    diff = num_contentfiles2 - num_contentfiles
    # if sample is the same as previous, then there will still only be a
    # single sample in db:
    assert diff >= 1
    assert num_preprocessed2 - num_preprocessed == diff


def test_eq():
    s1 = clgen.Sampler.from_json({
        "kernels": {
            "language": "opencl",
            "args": [
                '__global float*',
                '__global float*',
                'const int'
            ]
        }
    })
    s2 = clgen.Sampler.from_json({
        "kernels": {
            "language": "opencl",
            "args": [
                '__global float*',
                '__global float*',
                'const int'
            ]
        }
    })
    s3 = clgen.Sampler.from_json({
        "kernels": {
            "language": "opencl",
            "args": [
                'int'
            ]
        }
    })

    assert s1 == s2
    assert s2 != s3
    assert s1
    assert s1 != 'abcdef'


def test_to_json():
    s1 = clgen.Sampler.from_json({
        "kernels": {
            "language": "opencl",
            "args": [
                '__global float*',
                '__global float*',
                'const int'
            ]
        }
    })
    s2 = clgen.Sampler.from_json(s1.to_json())
    assert s1 == s2
