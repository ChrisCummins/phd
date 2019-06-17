"""Unit tests for //TODO:deeplearning.clgen.docker/clgen_preprocess_test."""

import sys

import pathlib

import pytest
from labm8 import dockerutil
from labm8 import fs
from labm8 import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = None


@pytest.fixture(scope='function')
def contentfiles(tempdir: pathlib.Path) -> pathlib.Path:
  fs.Write(tempdir / 'a.txt', "int main() {}".encode('utf-8'))
  fs.Write(tempdir / 'b.txt', "invalid syntax".encode('utf-8'))
  yield tempdir


@pytest.fixture(scope='function')
def preprocess() -> dockerutil.BazelPy3Image:
  return dockerutil.BazelPy3Image('deeplearning/clgen/docker/clgen_preprocess')


def test_preprocess_image_smoke_test(preprocess: dockerutil.BazelPy3Image):
  """TODO: Short summary of test."""
  # preprocess = dockerutil.BazelPy3Image(
  #     'deeplearning/clgen/docker/clgen_preprocess')
  #
  # with preprocess.RunContext() as ctx:
  #   ctx.CheckCall(['--version'])


def test_preprocess(contentfiles: pathlib.Path, tempdir2: pathlib.Path,
                    preprocess: dockerutil.BazelPy3Image):
  print("foo")
  sys.stdout.flush()
  sys.stderr.flush()
  # This is functionally the same as the test in
  # //deeplearning/clgen:preprocess_test.
  with preprocess.RunContext() as ctx:
    print("lol")
    sys.stdout.flush()
    sys.stderr.flush()
    ctx.CheckCall(
        [], {
            'contentfiles':
            '/contentfiles',
            'outdir':
            '/preprocessed',
            'preprocessors':
            ("deeplearning.clgen.preprocessors.cxx:Compile,"
             "deeplearning.clgen.preprocessors.cxx:ClangFormat")
        },
        volumes={
            contentfiles: '/contentfiles',
            tempdir2: '/preprocessed'
        },
        timeout=10)
    print("fuck")
    sys.stdout.flush()
    sys.stderr.flush()


#   assert (tempdir2 / 'a.txt').is_file()
#   assert not (tempdir2 / 'b.txt').is_file()
#
#   with open(tempdir2 / 'a.txt') as f:
#     assert f.read() == """int main() {
# }"""

if __name__ == '__main__':
  test.Main()
