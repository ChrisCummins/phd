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
import tempfile

import numpy as np
import pytest
from absl import app

from deeplearning.clgen import corpus
from deeplearning.clgen import errors
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.tests import testlib as tests


# The Corpus.hash for an OpenCL corpus with asciii_character_atomizer.
TINY_HASH = '0640b32387e34f1e9c5573f940d48227cfb13e08'


def test_GetKernelFeatures():
  """Test that features of a known kernel matches expected values."""
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}\
"""
  assert np.array_equal(corpus.GetKernelFeatures(code),
                        [0, 0, 1, 0, 1, 0, 1, 0])


def test_get_features_bad_code():
  """Test that a FeaturesError is raised if code contains errors."""
  with tests.DevNullRedirect():
    with pytest.raises(errors.FeaturesError):
      corpus.GetKernelFeatures("SYNTAX ERROR!", quiet=True)


def test_Corpus_badpath(clgen_cache_dir):
  """Test that CLgenError is raised when corpus has a non-existent path."""
  del clgen_cache_dir
  with pytest.raises(errors.CLgenError):
    corpus.Corpus(corpus_pb2.Corpus(language="opencl", path="notarealpath"))


def test_Corpus_hash(clgen_cache_dir, abc_corpus):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  c = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                      ascii_character_atomizer=True))
  assert c.hash == TINY_HASH


def test_Corpus_archive_hash(clgen_cache_dir, abc_corpus_archive):
  """Test that the ID of a known archive corpus matches expected value."""
  del clgen_cache_dir
  c = corpus.Corpus(
    corpus_pb2.Corpus(language="opencl", path=abc_corpus_archive,
                      ascii_character_atomizer=True))
  assert c.hash == TINY_HASH


def test_Corpus_config_hash_different_options(clgen_cache_dir, abc_corpus):
  """Test that the corpus ID is changed with a different option value."""
  del clgen_cache_dir
  c1 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       ascii_character_atomizer=True))
  atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=['a'])
  c3 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       greedy_multichar_atomizer=atomizer))
  assert c1.hash != c3.hash


def test_Corpus_empty_directory_raises_error(clgen_cache_dir):
  """Test that a corpus with no data raises an error."""
  del clgen_cache_dir
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(errors.EmptyCorpusException):
      corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=d))


def test_get_features_multiple_kernels():
  """Features cannot be extracted if file contains more than one kernel."""
  kernel_a = "__kernel void A(__global float* a){}\n"
  kernel_b = "__kernel void B(__global float* b){}\n"
  # Note that get_kernel_features returns a numpy array, so we can't simply
  # "assert" it. Instead we check that the sum is 0, since the kernels contain
  # no instructions, the feature vectors will be all zeros.
  assert corpus.GetKernelFeatures(kernel_a).sum() == 0
  assert corpus.GetKernelFeatures(kernel_b).sum() == 0
  with pytest.raises(errors.FeaturesError):
    corpus.GetKernelFeatures(kernel_a + kernel_b)


def test_Corpus_greedy_multichar_atomizer_no_atoms(clgen_cache_dir, abc_corpus):
  """Test that a GreedyMulticharAtomizer raises error if no tokens provided."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=[])
    corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                    greedy_multichar_atomizer=atomizer))


def test_Corpus_no_language_option(clgen_cache_dir, abc_corpus):
  """Test that an error is raised if no language is specified."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus(corpus_pb2.Corpus(path=abc_corpus))


def test_Corpus_bad_language_option(clgen_cache_dir, abc_corpus):
  """Test that an error is raised if an invalid language is specified."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    corpus.Corpus(corpus_pb2.Corpus(language="NOTALANG", path=abc_corpus))


def test_Corpus_equalivancy(clgen_cache_dir, abc_corpus):
  """Test that corpuses with the same configs are equal to each other."""
  del clgen_cache_dir
  atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=['a'])
  c1 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       greedy_multichar_atomizer=atomizer))
  c2 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       greedy_multichar_atomizer=atomizer))
  c3 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       ascii_character_atomizer=True))
  assert c1 == c2
  assert c2 != c3
  assert c1 != 'abcdef'


def test_Corpus_num_contentfiles(clgen_cache_dir, abc_corpus):
  """Test the number of contentfiles in a known corpus."""
  del clgen_cache_dir
  c = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                      ascii_character_atomizer=True))
  assert len(list(c.GetContentFiles())) == 3


def test_Corpus_preprocess_outcomes(clgen_cache_dir, abc_corpus):
  """Test the number of preprocessed kernels in a known corpus."""
  del clgen_cache_dir
  # Add a file containing a "good" OpenCL contentfile.
  with open(abc_corpus + '/cl_good.cl', 'w') as f:
    f.write("""
// Add a 
kernel void foo(global int* a) {
  a[get_global_id(0)] *= 2;
}
""")
  # Add a file containing an "ugly" OpenCL contentfile.
  with open(abc_corpus + '/cl_ugly.cl', 'w') as f:
    f.write("""
kernel void bar(global int* a) {
  // No code, therefore ugly.
}
""")
  # TODO(cec): Update preprocessor list after implementing preprocessor logic.
  c = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                      ascii_character_atomizer=True,
                                      preprocessors=['opencl']))
  assert len(list(c.GetPreprocessedKernels())) == 1
  assert len(list(c.GetPreprocessedKernels(1))) == 3
  assert len(list(c.GetPreprocessedKernels(2))) == 1


def test_Corpus_equivalency(clgen_cache_dir, abc_corpus):
  """Test that two corpuses with identical options are equivalent."""
  del clgen_cache_dir
  c1 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       ascii_character_atomizer=True))
  c2 = corpus.Corpus(corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                                       ascii_character_atomizer=True))
  assert c1 == c2


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
