"""Unit tests for //deeplearning/clgen/explore.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import corpus
from deeplearning.clgen import explore
from deeplearning.clgen.tests import testlib as tests
from lib.labm8 import fs


def test_explore(clgen_cache_dir, abc_corpus_config):
  """Test that explore doesn't fail?? This is a shit test."""
  del clgen_cache_dir
  c = corpus.Corpus(abc_corpus_config)
  explore.explore(c.contentfiles_cache["kernels.db"])


def test_explore_gh(clgen_cache_dir):
  """Test that explore doesn't fail?? This is a shit test."""
  del clgen_cache_dir
  db_path = tests.archive("tiny-gh.db")
  assert fs.exists(db_path)
  explore.explore(db_path)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
