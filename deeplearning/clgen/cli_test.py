"""Unit tests for //deeplearning/clgen/cli.py."""
import os
import pathlib
import sys
import tempfile

import pytest
from absl import app

from deeplearning.clgen import cli
from deeplearning.clgen.tests import testlib as tests
from lib.labm8 import fs
from lib.labm8 import pbutil
from lib.labm8 import tar


def test_run(clgen_cache_dir):
  """Test that cli.run() returns correct value for function call."""
  del clgen_cache_dir
  assert cli.run(lambda a, b: a // b, 4, 2) == 2


# def test_run_exception_handler(clgen_cache_dir):
#   del clgen_cache_dir
#   os.environ["DEBUG"] = ""
#   with pytest.raises(SystemExit):
#     cli.run(lambda a, b: a // b, 1, 0)


def test_run_exception_debug(clgen_cache_dir):
  """Test that cli.run() doesn't catch exception when $DEBUG is set."""
  del clgen_cache_dir
  os.environ["DEBUG"] = "1"
  with pytest.raises(ZeroDivisionError):
    cli.run(lambda a, b: a // b, 1, 0)


def test_cli_test_cache_path(clgen_cache_dir):
  del clgen_cache_dir
  with pytest.raises(SystemExit):
    cli.main("test --cache-path".split())


def test_cli_test_coverage_path(clgen_cache_dir):
  del clgen_cache_dir
  with pytest.raises(SystemExit):
    cli.main("test --coverage-path".split())


def test_cli_test_coveragerc_path(clgen_cache_dir):
  del clgen_cache_dir
  with pytest.raises(SystemExit):
    cli.main("test --coveragerc-path".split())


def test_cli(clgen_cache_dir):
  del clgen_cache_dir
  fs.rm("kernels.db")
  cli.main("db init kernels.db".split())
  assert fs.exists("kernels.db")

  corpus_path = tests.archive("tiny", "corpus")
  cli.main("db explore kernels.db".split())
  cli.main(f"fetch fs kernels.db {corpus_path}".split())
  cli.main("preprocess kernels.db".split())
  cli.main("db explore kernels.db".split())

  fs.rm("kernels_out")
  cli.main("db dump kernels.db -d kernels_out".split())
  assert fs.isdir("kernels_out")
  assert len(fs.ls("kernels_out")) >= 1

  fs.rm("kernels.cl")
  cli.main("db dump kernels.db kernels.cl --file-sep --eof --reverse".split())
  assert fs.isfile("kernels.cl")

  fs.rm("kernels_out")
  cli.main("db dump kernels.db --input-samples -d kernels_out".split())
  assert fs.isdir("kernels_out")
  assert len(fs.ls("kernels_out")) == 250

  fs.rm("kernels.db")
  fs.rm("kernels_out")


def test_cli_train(clgen_cache_dir, abc_model_config):
  del clgen_cache_dir
  os.environ["DEBUG"] = "1"
  with tempfile.TemporaryDirectory(prefix="clgen_") as d:
    with tests.chdir(d):
      fs.cp(tests.data_path("pico", "corpus.tar.bz2"), './corpus.tar.bz2')
      tar.unpack_archive('corpus.tar.bz2')
      pbutil.ToFile(abc_model_config, pathlib.Path('model.pbtxt'))
      cli.main("--corpus-dir model.pbtxt".split())
      cli.main("--model-dir model.pbtxt".split())
      cli.main("-v train model.pbtxt".split())


def test_cli_sample(clgen_cache_dir, abc_model_config, abc_sampler_config):
  del clgen_cache_dir
  with tempfile.TemporaryDirectory() as d:
    with tests.chdir(d):
      fs.cp(tests.data_path("pico", "corpus.tar.bz2"), './corpus.tar.bz2')
      pbutil.ToFile(abc_model_config, pathlib.Path('model.pbtxt'))
      pbutil.ToFile(abc_sampler_config, pathlib.Path('sampler.pbtxt'))
      cli.main("--corpus-dir model.pbtxt".split())
      cli.main("--model-dir model.pbtxt".split())
      cli.main("--sampler-dir model.pbtxt sampler.pbtxt".split())
      cli.main("ls files model.pbtxt sampler.pbtxt".split())


def test_cli_ls(clgen_cache_dir):
  del clgen_cache_dir
  cli.main("ls models".split())
  cli.main("ls samplers".split())


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
