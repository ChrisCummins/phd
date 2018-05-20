#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
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
from deeplearning.clgen import dbutil
from deeplearning.clgen import model
from deeplearning.clgen import sampler
from deeplearning.clgen.tests import testlib as tests


def _get_test_model():
  return model.Model.from_json({"corpus": {"language": "opencl",
                                           "path": tests.data_path("tiny",
                                                                   "corpus"), },
                                "architecture": {"rnn_size": 8,
                                                 "num_layers": 2, },
                                "train_opts": {"epochs": 1}})


def test_sample():
  m = _get_test_model()
  m.train()

  argspec = ['__global float*', '__global float*', '__global float*',
             'const int']
  s = sampler.Sampler.from_json(
    {"kernels": {"language": "opencl", "args": argspec, "max_length": 300, },
     "sampler": {"min_samples": 1}})

  s.cache(m).clear()  # clear old samples

  # sample a single kernel:
  s.sample(m)
  num_contentfiles = dbutil.num_rows_in(s.cache(m)["kernels.db"],
                                        "ContentFiles")
  assert num_contentfiles >= 1

  s.sample(m)
  num_contentfiles2 = dbutil.num_rows_in(s.cache(m)["kernels.db"],
                                         "ContentFiles")
  diff = num_contentfiles2 - num_contentfiles
  # if sample is the same as previous, then there will still only be a
  # single sample in db:
  assert diff >= 1


def test_eq():
  s1 = sampler.Sampler.from_json({"kernels": {"language": "opencl",
                                              "args": ['__global float*',
                                                       '__global float*',
                                                       'const int']}})
  s2 = sampler.Sampler.from_json({"kernels": {"language": "opencl",
                                              "args": ['__global float*',
                                                       '__global float*',
                                                       'const int']}})
  s3 = sampler.Sampler.from_json(
    {"kernels": {"language": "opencl", "args": ['int']}})

  assert s1 == s2
  assert s2 != s3
  assert s1
  assert s1 != 'abcdef'


def test_to_json():
  s1 = sampler.Sampler.from_json({"kernels": {"language": "opencl",
                                              "args": ['__global float*',
                                                       '__global float*',
                                                       'const int']}})
  s2 = sampler.Sampler.from_json(s1.to_json())
  assert s1 == s2
