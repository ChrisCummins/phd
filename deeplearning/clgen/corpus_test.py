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
import sys

import numpy as np
import pytest
from absl import app

from deeplearning.clgen import corpus
from deeplearning.clgen import errors
from deeplearning.clgen.tests import testlib as tests


TINY_HASH = '88762834496984cbd11355eca15cbd4082902d24'


def test_badpath(clgen_cache_dir):
  """Test that CLgenError is raised when corpus has a non-existent path."""
  del clgen_cache_dir
  with pytest.raises(errors.CLgenError):
    corpus.Corpus("notarealid", path="notarealpath")


def test_corpus_id(clgen_cache_dir):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  c = corpus.Corpus.from_json({"language": "opencl", "id": TINY_HASH,
                               "path": tests.archive("tiny", "corpus"),
                               "preprocess": False})
  assert TINY_HASH == c.hash


def test_archive_corpus_id(clgen_cache_dir):
  """Test that the ID of a known archive corpus matches expected value."""
  del clgen_cache_dir
  c = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.data_path("tiny", "corpus.tar.bz2"),
     "preprocess": False})
  assert TINY_HASH == c.hash


def test_hash(clgen_cache_dir):
  """Test that the corpus ID depends on the corpus config."""
  del clgen_cache_dir
  c1 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus")})
  # same as c1, with explicit default opt:
  c2 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus"),
     "eof": False})
  # different opt value:
  c3 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus"),
     "eof": True})
  assert c1.hash == c2.hash
  assert c2.hash != c3.hash


def test_get_features():
  """Test that features of a known kernel matches expected values."""
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}\
"""
  assert np.array_equal(corpus.get_kernel_features(code),
                        [0, 0, 1, 0, 1, 0, 1, 0])


def test_get_features_bad_code():
  """Test that a FeaturesError is raised if code contains errors."""
  with tests.DevNullRedirect():
    with pytest.raises(errors.FeaturesError):
      corpus.get_kernel_features("SYNTAX ERROR!", quiet=True)


def test_get_features_multiple_kernels():
  """Features cannot be extracted if file contains more than one kernel."""
  kernel_a = "__kernel void A(__global float* a){}\n"
  kernel_b = "__kernel void B(__global float* b){}\n"
  # Note that get_kernel_features returns a numpy array, so we can't simply
  # "assert" it. Instead we check that the sum is 0, since the kernels contain
  # no instructions, the feature vectors will be all zeros.
  assert corpus.get_kernel_features(kernel_a).sum() == 0
  assert corpus.get_kernel_features(kernel_b).sum() == 0
  with pytest.raises(errors.FeaturesError):
    corpus.get_kernel_features(kernel_a + kernel_b)


def test_no_language(clgen_cache_dir):
  """Test that an error is raised if no language is specified."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus.from_json({"path": tests.archive("tiny", "corpus")})


def test_bad_language(clgen_cache_dir):
  """Test that an error is raised if an invalid language is specified."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus.from_json(
      {"language": "NOTALANG", "path": tests.archive("tiny", "corpus")})


def test_bad_option(clgen_cache_dir):
  """Test that an error is raised if an invalid option is given."""
  with pytest.raises(errors.UserError):
    corpus.Corpus.from_json(
      {"language": "opencl", "path": tests.archive("tiny", "corpus"),
       "not_a_real_option": False})


def test_bad_vocab(clgen_cache_dir):
  """Test that invalid vocabulary option raises an error."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus.from_json(
      {"language": "opencl", "path": tests.archive("tiny", "corpus"),
       "vocab": "INVALID_VOCAB"})


@pytest.mark.xfail(reason="FIXME: UserError not raised")
def test_bad_encoding(clgen_cache_dir):
  """Test that an invalid encoding option raises an error."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus.from_json(
      {"language": "opencl", "path": tests.archive("tiny", "corpus"),
       "encoding": "INVALID_ENCODING"})


def test_corpus_equalivancy_checks(clgen_cache_dir):
  """Test that corpuses with the same configs are equal to each other."""
  del clgen_cache_dir
  c1 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus"),
     "eof": False})
  c2 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus"),
     "eof": False})
  c3 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus"),
     "eof": True})
  assert c1 == c2
  assert c2 != c3
  assert c1 != 'abcdef'


def test_preprocessed(clgen_cache_dir):
  """Test the number of preprocessed kernels in a known corpus."""
  del clgen_cache_dir
  c1 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus")})
  assert len(list(c1.preprocessed())) == 197
  assert len(list(c1.preprocessed(1))) == 48
  assert len(list(c1.preprocessed(2))) == 5


def test_contentfiles(clgen_cache_dir):
  """Test the number of contentfiles in a known corpus."""
  del clgen_cache_dir
  c1 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus")})
  assert len(list(c1.contentfiles())) == 250


def test_to_json_equivalency(clgen_cache_dir):
  """Test that from_json() and to_json() are symmetrical."""
  del clgen_cache_dir
  c1 = corpus.Corpus.from_json(
    {"language": "opencl", "path": tests.archive("tiny", "corpus")})
  c2 = corpus.Corpus.from_json(c1.to_json())
  assert c1 == c2


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
