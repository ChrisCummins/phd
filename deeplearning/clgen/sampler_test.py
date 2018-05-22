"""Unit tests for //deeplearning/clgen/sampler.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import dbutil
from deeplearning.clgen import model
from deeplearning.clgen import sampler
from deeplearning.clgen.tests import testlib as tests


def _get_test_model():
  return model.Model.from_json({"corpus": {"language": "opencl",
                                           "path": tests.archive("tiny",
                                                                 "corpus"), },
                                "architecture": {"rnn_size": 8,
                                                 "num_layers": 2, },
                                "train_opts": {"epochs": 1}})


def test_sample(clgen_cache_dir):
  del clgen_cache_dir
  m = _get_test_model()
  m.Train()

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


def test_eq(clgen_cache_dir):
  del clgen_cache_dir
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


def test_to_json(clgen_cache_dir):
  del clgen_cache_dir
  s1 = sampler.Sampler.from_json({"kernels": {"language": "opencl",
                                              "args": ['__global float*',
                                                       '__global float*',
                                                       'const int']}})
  s2 = sampler.Sampler.from_json(s1.to_json())
  assert s1 == s2


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
