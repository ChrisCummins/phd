"""Unit tests for //deeplearning/clgen/corpus.py."""
import sys
import tempfile

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import corpus_pb2


# The Corpus.hash for an OpenCL corpus of the abc_corpus.
ABC_CORPUS_HASH = 'cb7c7a23c433a1f628c9b120378759f1723fdf42'


def test_Sampler_config_type_error():
  """Test that a TypeError is raised if config is not a Sampler proto."""
  with pytest.raises(TypeError) as e_info:
    corpuses.Corpus(1)
  assert "Config must be a Corpus proto. Received: 'int'" == str(e_info.value)


def test_Corpus_badpath(clgen_cache_dir, abc_corpus_config):
  """Test that CLgenError is raised when corpus has a non-existent path."""
  del clgen_cache_dir
  abc_corpus_config.local_directory = "notarealpath"
  with pytest.raises(errors.UserError) as e_info:
    corpuses.Corpus(abc_corpus_config)
  # We resolve the absolute path, so we can't match the whole string.
  assert str(e_info.value).startswith("File not found: '")
  assert str(e_info.value).endswith("notarealpath'")


def test_Corpus_hash(clgen_cache_dir, abc_corpus_config):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  c = corpuses.Corpus(abc_corpus_config)
  assert ABC_CORPUS_HASH == c.hash


def test_Corpus_archive_hash(clgen_cache_dir, abc_corpus_config,
                             abc_corpus_archive):
  """Test that the ID of a known archive corpus matches expected value."""
  del clgen_cache_dir
  abc_corpus_config.local_tar_archive = abc_corpus_archive
  c = corpuses.Corpus(abc_corpus_config)
  assert ABC_CORPUS_HASH == c.hash


def test_Corpus_archive_not_found(clgen_cache_dir, abc_corpus_config):
  """Test that UserError is raised if local_tar_archive does not exist."""
  del clgen_cache_dir
  with tempfile.TemporaryDirectory() as d:
    abc_corpus_config.local_tar_archive = f'{d}/missing_archive.tar.bz2'
    with pytest.raises(errors.UserError):
      corpuses.Corpus(abc_corpus_config)


def test_Corpus_config_hash_different_options(clgen_cache_dir,
                                              abc_corpus_config):
  """Test that the corpus ID is changed with a different option value."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(abc_corpus_config)
  abc_corpus_config.greedy_multichar_atomizer.tokens[:] = ['a']
  c3 = corpuses.Corpus(abc_corpus_config)
  assert c1.hash != c3.hash


def test_Corpus_equality(clgen_cache_dir, abc_corpus_config):
  """Test that two corpuses with identical options are equivalent."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(abc_corpus_config)
  c2 = corpuses.Corpus(abc_corpus_config)
  assert c1 == c2


def test_Corpus_inequality(clgen_cache_dir, abc_corpus_config):
  """Test that two corpuses with different options are not equivalent."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(abc_corpus_config)
  abc_corpus_config.greedy_multichar_atomizer.tokens[:] = ['a']
  c2 = corpuses.Corpus(abc_corpus_config)
  assert c1 != c2


def test_Corpus_Create_empty_directory_raises_error(clgen_cache_dir,
                                                    abc_corpus_config):
  """Test that a corpus with no content files raises an error."""
  del clgen_cache_dir
  with tempfile.TemporaryDirectory() as d:
    abc_corpus_config.local_directory = d
    with pytest.raises(errors.EmptyCorpusException) as e_info:
      corpuses.Corpus(abc_corpus_config).Create()
    assert f"Empty content files directory: '{d}'" == str(e_info.value)


def test_Corpus_Create_empty_preprocessed_raises_error(clgen_cache_dir,
                                                       abc_corpus_config):
  """Test that a pre-processed corpus with no data raises an error."""
  del clgen_cache_dir
  c = corpuses.Corpus(abc_corpus_config)
  # Empty the pre-processed database:
  c.preprocessed.Create(abc_corpus_config)
  with c.preprocessed.Session(commit=True) as session:
    session.query(preprocessed.PreprocessedContentFile).delete()
  with pytest.raises(errors.EmptyCorpusException) as e_info:
    c.Create()
  assert ("Pre-processed corpus contains no files: "
          f"'{c.preprocessed.database_path}'") == str(e_info.value)


def test_Corpus_greedy_multichar_atomizer_no_atoms(clgen_cache_dir,
                                                   abc_corpus_config):
  """Test that a GreedyMulticharAtomizer raises error if no tokens provided."""
  del clgen_cache_dir
  abc_corpus_config.greedy_multichar_atomizer.tokens[:] = []
  with pytest.raises(errors.UserError) as e_info:
    corpuses.Corpus(abc_corpus_config)
  assert 'GreedyMulticharAtomizer.tokens is empty' == str(e_info.value)


def test_Corpus_greedy_multichar_atomizer_empty_atoms(clgen_cache_dir,
                                                      abc_corpus_config):
  """Test that a GreedyMulticharAtomizer raises error for zero-length string."""
  del clgen_cache_dir
  with pytest.raises(errors.UserError) as e_info:
    abc_corpus_config.greedy_multichar_atomizer.tokens[:] = ['']
    corpuses.Corpus(abc_corpus_config)
  assert 'Empty string found in GreedyMulticharAtomizer.tokens is empty' == str(
      e_info.value)


def test_Corpus_content_id(clgen_cache_dir, abc_corpus_config):
  """Test that the content_id field resolves to the correct corpus."""
  del clgen_cache_dir
  c1 = corpuses.Corpus(abc_corpus_config)
  content_id = c1.content_id
  # Create an identical corpus but using the content_id field rather than
  # a local_directory.
  abc_corpus_config.ClearField('contentfiles')
  abc_corpus_config.content_id = content_id
  c2 = corpuses.Corpus(abc_corpus_config)
  assert c1.hash == c2.hash


def test_Corpus_invalid_content_id(clgen_cache_dir, abc_corpus_config):
  """Test that UserError is raised if content_id does not resolve to cache."""
  del clgen_cache_dir
  abc_corpus_config.ClearField('contentfiles')
  abc_corpus_config.content_id = '1234invalid'
  with pytest.raises(errors.UserError) as e_ctx:
    corpuses.Corpus(abc_corpus_config)
  assert "Content ID not found: '1234invalid'" == str(e_ctx.value)


def test_Corpus_Create_num_contentfiles(clgen_cache_dir, abc_corpus_config):
  """Test the number of contentfiles in a known corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(abc_corpus_config)
  assert c.GetNumContentFiles() == 0
  c.Create()
  assert c.GetNumContentFiles() == 3


def test_Corpus_Create_preprocess_outcomes(clgen_cache_dir, abc_corpus_config):
  """Test the number of preprocessed kernels in a known corpus."""
  del clgen_cache_dir
  # Add a file containing a "good" OpenCL contentfile.
  with open(abc_corpus_config.local_directory + '/cl_good.cl', 'w') as f:
    f.write("""
// A good kernel.
kernel void foo(global int* a) {
  a[get_global_id(0)] *= 2;
}
""")
  abc_corpus_config.preprocessor[:] = [
    'deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim',
    'deeplearning.clgen.preprocessors.opencl:Compile',
    'deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers',
    'deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes',
    'deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines',
    'deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype',
    'deeplearning.clgen.preprocessors.common:StripTrailingWhitespace',
    'deeplearning.clgen.preprocessors.cxx:ClangFormat',
    'deeplearning.clgen.preprocessors.common:MinimumLineCount3',
  ]
  c = corpuses.Corpus(abc_corpus_config)
  assert c.GetNumContentFiles() == 0
  assert c.GetNumPreprocessedFiles() == 0
  c.Create()
  assert c.GetNumContentFiles() == 4
  assert c.GetNumPreprocessedFiles() == 1


def test_Corpus_GetTextCorpus_no_shuffle(clgen_cache_dir, abc_corpus_config):
  """Test the concatenation of the abc corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(abc_corpus_config)
  assert c.GetTextCorpus(shuffle=False) == ''
  c.Create()
  # We don't know the ordering of the text corpus.
  assert 'The cat sat on the mat.' in c.GetTextCorpus(shuffle=False)
  assert 'Such corpus.\nVery wow.' in c.GetTextCorpus(shuffle=False)
  assert 'Hello, world!' in c.GetTextCorpus(shuffle=False)
  assert c.GetTextCorpus(shuffle=False).count('\n\n') == 2


def test_Corpus_GetTextCorpus_separator(clgen_cache_dir, abc_corpus):
  """Test the concatenation of the abc corpus with a custom separator."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True,
                                        contentfile_separator='\n!!\n'))
  c.Create()
  # We don't know the ordering of the text corpus.
  assert 'The cat sat on the mat.' in c.GetTextCorpus(shuffle=False)
  assert 'Such corpus.\nVery wow.' in c.GetTextCorpus(shuffle=False)
  assert 'Hello, world!' in c.GetTextCorpus(shuffle=False)
  assert c.GetTextCorpus(shuffle=False).count('!!') == 2


def test_Corpus_GetTextCorpus_random_order(clgen_cache_dir, abc_corpus_config):
  """Test that random shuffling of contentfiles changes the corpus."""
  del clgen_cache_dir
  c = corpuses.Corpus(abc_corpus_config)
  assert c.GetTextCorpus(shuffle=True) == ''
  c.Create()
  # Generate five concatenations with a random order. The idea is that it is
  # extremely unlikely that the same ordering would be randomly selected all
  # five times, however, this is not impossible, so consider this test flaky.
  c1 = c.GetTextCorpus(shuffle=True)
  c2 = c.GetTextCorpus(shuffle=True)
  c3 = c.GetTextCorpus(shuffle=True)
  c4 = c.GetTextCorpus(shuffle=True)
  c5 = c.GetTextCorpus(shuffle=True)
  assert len({c1, c2, c3, c4, c5}) > 1


def test_Corpus_GetTrainingData_decode(clgen_cache_dir, abc_corpus):
  """Test the decoded output of GetTrainingData()."""
  del clgen_cache_dir
  c = corpuses.Corpus(corpus_pb2.Corpus(local_directory=abc_corpus,
                                        ascii_character_atomizer=True,
                                        contentfile_separator='\n!!\n'))
  c.Create()
  decoded = c.atomizer.DeatomizeIndices(c.GetTrainingData(shuffle=False))
  # Test that each content file (plus contentfile separator) is in corpus.
  assert '\nSuch corpus.\nVery wow.\n!!\n' in decoded
  assert 'Hello, world!\n!!\n' in decoded
  assert 'The cat sat on the mat.\n!!\n' in decoded
  # Test the total length of the corpus.
  assert len('\nSuch corpus.\nVery wow.\n!!\n' +
             'Hello, world!\n!!\n' +
             'The cat sat on the mat.\n!!\n') == len(decoded)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
