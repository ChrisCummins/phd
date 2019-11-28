"""Run lexer on all files in a test stage."""
import pathlib
import subprocess

from compilers.toy.tests import smoke_test_flags
from labm8.py import app
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import test

FLAGS = smoke_test_flags.FLAGS

_TEST_DATA_ROOT = bazelutil.DataPath('phd/compilers/toy/test_data')
# Frustratingly, the go binary does not have the name of the binary.
_lexer_files = list(bazelutil.DataPath('phd/compilers/toy/util/lex').iterdir())
assert len(_lexer_files) == 1
_LEXER = _lexer_files[0] / 'lex'
assert _LEXER.is_file()

MODULE_UNDER_TEST = None  # Disable coverage.


def Lex(path: pathlib.Path):
  """Run lexer on path. Raises an exception is the lexer fails."""
  app.Log(1, "Running lexer on `%s`", path)
  subprocess.check_call([str(_LEXER), str(path)])


def RunStageTests(stage: int):
  """Lex all files for the given stage and check that they all work."""
  assert 1 <= stage <= 10

  valid_files = fs.lsfiles(_TEST_DATA_ROOT / f'stage_{stage}' / 'valid',
                           abspaths=True)
  invalid_files = fs.lsfiles(_TEST_DATA_ROOT / f'stage_{stage}' / 'invalid',
                             abspaths=True)

  for test_file in list(valid_files) + list(invalid_files):
    Lex(test_file)


def test_stage():
  """Run stage tests."""
  RunStageTests(FLAGS.stage)


if __name__ == '__main__':
  test.Main()
