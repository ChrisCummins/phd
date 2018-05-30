"""Unit tests for //deeplearning/clgen/corpus.py."""
import sys
import tempfile

import numpy as np
import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.tests import testlib as tests


# The Corpus.hash for an OpenCL corpus of the abc_corpus.
ABC_CORPUS_HASH = 'f57ee1466bb4b62fa8e480ec02217dc55909e634'


@pytest.mark.skip(reason='TODO(cec): Fix clgen-features data path.')
# See TODO note in //deeplearning/clgen/native/clgen-features.cpp.
def test_GetKernelFeatures():
  """Test that features of a known kernel matches expected values."""
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}\
"""
  assert np.array_equal(corpuses.GetKernelFeatures(code),
                        [0, 0, 1, 0, 1, 0, 1, 0])


@pytest.mark.skip(reason='TODO(cec): Fix clgen-features data path.')
def test_GetKernelFeatures_multiple_kernels():
  """Features cannot be extracted if file contains more than one kernel."""
  kernel_a = "__kernel void A(__global float* a){}\n"
  kernel_b = "__kernel void B(__global float* b){}\n"
  # Note that get_kernel_features returns a numpy array, so we can't simply
  # "assert" it. Instead we check that the sum is 0, since the kernels contain
  # no instructions, the feature vectors will be all zeros.
  assert corpuses.GetKernelFeatures(kernel_a).sum() == 0
  assert corpuses.GetKernelFeatures(kernel_b).sum() == 0
  with pytest.raises(errors.FeaturesError):
    corpuses.GetKernelFeatures(kernel_a + kernel_b)


@pytest.mark.skip(reason='TODO(cec): Fix clgen-features data path.')
def test_GetKernelFeatures_bad_code():
  """Test that a FeaturesError is raised if code contains errors."""
  with tests.DevNullRedirect():
    with pytest.raises(errors.FeaturesError):
      corpuses.GetKernelFeatures("SYNTAX ERROR!", quiet=True)


def test_Sampler_config_type_error():
  """Test that a TypeError is raised if config is not a Sampler proto."""
  with pytest.raises(TypeError) as e_info:
    corpuses.Corpus(1)
  assert "Config must be a Corpus proto. Received: 'int'" == str(e_info.value)


def test_Corpus_badpath(clgen_cache_dir):
  """Test that CLgenError is raised when corpus has a non-existent path."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError) as e_info:
    corpuses.Corpus(corpus_pb2.Corpus(local_directory="notarealpath"))
  # We resolve the absolute path, so we can't match the whole string.
  assert str(e_info.value).startswith("File not found: '")
  assert str(e_info.value).endswith("notarealpath'")


def test_Corpus_hash(clgen_cache_dir, abc_corpus):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True))
  assert ABC_CORPUS_HASH == c.hash


def test_Corpus_archive_hash(clgen_cache_dir, abc_corpus_archive):
  """Test that the ID of a known archive corpus matches expected value."""
  del clgen_cache_dir
  c = corpuses.Corpus(
      corpus_pb2.Corpus(local_directory=abc_corpus_archive,
                        ascii_character_atomizer=True))
  assert ABC_CORPUS_HASH == c.hash


def test_Corpus_config_hash_different_options(clgen_cache_dir, abc_corpus):
  """Test that the corpus ID is changed with a different option value."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         ascii_character_atomizer=True))
  atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=['a'])
  c3 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         greedy_multichar_atomizer=atomizer))
  assert c1.hash != c3.hash


def test_Corpus_equality(clgen_cache_dir, abc_corpus):
  """Test that two corpuses with identical options are equivalent."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         ascii_character_atomizer=True))
  c2 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         ascii_character_atomizer=True))
  assert c1 == c2


def test_Corpus_inequality(clgen_cache_dir, abc_corpus):
  """Test that two corpuses with different options are not equivalent."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         ascii_character_atomizer=True))
  atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=['a'])
  c2 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         greedy_multichar_atomizer=atomizer))
  assert c1 != c2


def test_Corpus_empty_directory_raises_error(clgen_cache_dir):
  """Test that a corpus with no data raises an error."""
  del clgen_cache_dir
  with tempfile.TemporaryDirectory() as d:
    with pytest.raises(errors.EmptyCorpusException):
      corpuses.Corpus(corpus_pb2.Corpus(local_directory=d,
                                        ascii_character_atomizer=True))


def test_Corpus_greedy_multichar_atomizer_no_atoms(clgen_cache_dir, abc_corpus):
  """Test that a GreedyMulticharAtomizer raises error if no tokens provided."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError):
    atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=[])
    corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                      greedy_multichar_atomizer=atomizer))


def test_Corpus_equalivancy(clgen_cache_dir, abc_corpus):
  """Test that corpuses with the same configs are equal to each other."""
  del clgen_cache_dir
  atomizer = corpus_pb2.GreedyMulticharAtomizer(tokens=['a'])
  c1 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         greedy_multichar_atomizer=atomizer))
  c2 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         greedy_multichar_atomizer=atomizer))
  c3 = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                         ascii_character_atomizer=True))
  assert c1 == c2
  assert c2 != c3
  assert c1 != 'abcdef'


def test_Corpus_num_contentfiles(clgen_cache_dir, abc_corpus):
  """Test the number of contentfiles in a known corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True))
  assert len(list(c.GetContentFiles())) == 3


def test_Corpus_preprocess_outcomes(clgen_cache_dir, abc_corpus):
  """Test the number of preprocessed kernels in a known corpus."""
  del clgen_cache_dir
  # Add a file containing a "good" OpenCL contentfile.
  with open(abc_corpus + '/cl_good.cl', 'w') as f:
    f.write("""
// A good kernel.
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
  config = corpus_pb2.Corpus(local_directory=abc_corpus,
                             ascii_character_atomizer=True,
                             preprocessor=[
                               'deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim',
                               'deeplearning.clgen.preprocessors.opencl:Compile',
                               'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
                               'deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes',
                               'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
                               'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
                               'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
                               'deeplearning.clgen.preprocessors.cxx:ClangFormat',
                               'deeplearning.clgen.preprocessors.common:MinimumLineCount3'])
  c = corpuses.Corpus(config)
  assert len(list(c.GetPreprocessedKernels())) == 1
  assert len(list(c.GetPreprocessedKernels(1))) == 4


def test_Corpus_ConcatenateTextCorpus_no_shuffle(clgen_cache_dir, abc_corpus):
  """Test the concatenation of the abc corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True))
  assert c.ConcatenateTextCorpus(shuffle=False) == """The cat sat on the mat.

Such corpus.
Very wow.

Hello, world!"""


def test_Corpus_ConcatenateTextCorpus_separator(clgen_cache_dir, abc_corpus):
  """Test the concatenation of the abc corpus with a custom separator."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True,
                                        contentfile_separator='\n!!\n'))
  assert c.ConcatenateTextCorpus(shuffle=False) == """The cat sat on the mat.
!!
Such corpus.
Very wow.
!!
Hello, world!"""


def test_Corpus_ConcatenateTextCorpus_random_order(clgen_cache_dir, abc_corpus):
  """Test that random shuffling of contentfiles changes the corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True))
  # Generate five concatenations with a random order. The idea is that it is
  # extremely unlikely that the same ordering would be randomly selected all
  # five times, however, this is not impossible, so consider this test flaky.
  c1 = c.ConcatenateTextCorpus(shuffle=True)
  c2 = c.ConcatenateTextCorpus(shuffle=True)
  c3 = c.ConcatenateTextCorpus(shuffle=True)
  c4 = c.ConcatenateTextCorpus(shuffle=True)
  c5 = c.ConcatenateTextCorpus(shuffle=True)
  assert len(set([c1, c2, c3, c4, c5])) > 1


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
