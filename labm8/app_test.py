"""Unit tests for //labm8/app."""
import pathlib

import pytest
from absl import flags as absl_flags

from labm8 import app_test_flags
from labm8 import test

FLAGS = app_test_flags.FLAGS

MODULE_UNDER_TEST = None


def test_string_flag():
  FLAGS.unparse_flags()
  FLAGS(['argv[0]', '--string_flag', 'Hello, world!'])
  assert FLAGS.string_flag == 'Hello, world!'


def test_output_path_flag(tempdir: pathlib.Path):
  FLAGS.unparse_flags()
  FLAGS(['argv[0]', '--output_path_flag', str(tempdir / 'file')])
  assert FLAGS.output_path_flag == pathlib.Path(tempdir / 'file')


def test_int_flag_validator():
  FLAGS.unparse_flags()
  FLAGS(['argv[0]', '--int_flag_with_validator', '2'])
  with pytest.raises(absl_flags.IllegalFlagValueError):
    FLAGS(['argv[0]', '--int_flag_with_validator', '-1'])


def test_database_flag(tempdir: pathlib.Path):
  FLAGS.unparse_flags()
  url = f'sqlite:///{tempdir}/db'
  FLAGS(['argv[0]', '--database_flag', url])
  # The database isn't created until the flag value is called.
  assert not (tempdir / 'db').is_file()
  assert FLAGS.database_flag().url == url
  assert (tempdir / 'db').is_file()


if __name__ == '__main__':
  test.Main()
