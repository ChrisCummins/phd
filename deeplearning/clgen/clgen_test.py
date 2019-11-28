# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/clgen/cli.py."""
import os
import pathlib
import tempfile

import pytest

from deeplearning.clgen import clgen
from deeplearning.clgen import errors
from deeplearning.clgen.proto import clgen_pb2
from labm8.py import app
from labm8.py import pbutil
from labm8.py import test

FLAGS = app.FLAGS

# Instance tests.


def test_Instance_no_working_dir_field(abc_instance_config):
  """Test that working_dir is None when no working_dir field in config."""
  abc_instance_config.ClearField("working_dir")
  instance = clgen.Instance(abc_instance_config)
  assert instance.working_dir is None


def test_Instance_working_dir_shell_variable_expansion(abc_instance_config):
  """Test that shell variables are expanded in working_dir."""
  working_dir = abc_instance_config.working_dir
  os.environ["FOO"] = working_dir
  abc_instance_config.working_dir = "$FOO/"
  instance = clgen.Instance(abc_instance_config)
  assert str(instance.working_dir) == working_dir


def test_Instance_no_model_field(abc_instance_config):
  """Test that UserError is raised when no model field in config."""
  abc_instance_config.ClearField("model_specification")
  with test.Raises(errors.UserError) as e_info:
    clgen.Instance(abc_instance_config)
  assert "Field not set: 'Instance.model_specification'" == str(e_info.value)


def test_Instance_no_sampler_field(abc_instance_config):
  """Test that UserError is raised when no model field in config."""
  abc_instance_config.ClearField("model_specification")
  with test.Raises(errors.UserError) as e_info:
    clgen.Instance(abc_instance_config)
  assert "Field not set: 'Instance.model_specification'" == str(e_info.value)


def test_Instance_Session_clgen_dir(abc_instance_config):
  """Test that $CLEN_CACHE is set to working_dir inside a session."""
  instance = clgen.Instance(abc_instance_config)
  with instance.Session():
    assert os.environ["CLGEN_CACHE"] == abc_instance_config.working_dir


def test_Instance_Session_no_working_dir(
  abc_instance_config, tempdir2: pathlib.Path
):
  """Test that $CLEN_CACHE is not modified config doesn't set working_dir."""
  abc_instance_config.ClearField("working_dir")
  os.environ["CLGEN_CACHE"] = str(tempdir2)
  instance = clgen.Instance(abc_instance_config)
  with instance.Session():
    assert os.environ["CLGEN_CACHE"] == str(tempdir2)


def test_Instance_Session_yield_value(abc_instance_config):
  """Test that Session() yields the instance."""
  instance = clgen.Instance(abc_instance_config)
  with instance.Session() as s:
    assert instance == s


def test_Instance_ToProto_equality(abc_instance_config):
  """Test that ToProto() returns the same as the input config."""
  instance = clgen.Instance(abc_instance_config)
  assert abc_instance_config == instance.ToProto()


# RunWithErrorHandling() tests.


def test_RunWithErrorHandling_return_value(clgen_cache_dir):
  """Test that RunWithErrorHandling() returns correct value for function."""
  del clgen_cache_dir
  assert clgen.RunWithErrorHandling(lambda a, b: a // b, 4, 2) == 2


def test_RunWithErrorHandling_system_exit(clgen_cache_dir):
  """Test that SystemExit is raised on exception."""
  del clgen_cache_dir
  with test.Raises(SystemExit):
    clgen.RunWithErrorHandling(lambda a, b: a // b, 1, 0)


def test_RunWithErrorHandling_exception_debug(clgen_cache_dir):
  """Test that FLAGS.debug disables exception catching."""
  del clgen_cache_dir
  app.FLAGS(["argv[0]", "--clgen_debug"])
  with test.Raises(ZeroDivisionError):
    clgen.RunWithErrorHandling(lambda a, b: a // b, 1, 0)


# main tests.


def test_main_unrecognized_arguments():
  """Test that UsageError is raised if arguments are not recognized."""
  with test.Raises(app.UsageError) as e_info:
    clgen.main(["argv[0]", "--foo", "--bar"])
  assert "Unrecognized command line options: '--foo --bar'" == str(e_info.value)


def test_main_no_config_flag():
  """Test that UsageError is raised if --config flag not set."""
  with test.Raises(app.UsageError) as e_info:
    clgen.main(["argv[0]"])
  assert "CLgen --config file not found: '/clgen/config.pbtxt'" == str(
    e_info.value
  )


def test_main_config_file_not_found():
  """Test that UsageError is raised if --config flag not found."""
  with tempfile.TemporaryDirectory() as d:
    app.FLAGS.unparse_flags()
    app.FLAGS(["argv[0]", "--config", f"{d}/config.pbtxt"])
    with test.Raises(app.UsageError) as e_info:
      clgen.main(["argv[0]"])
    assert f"CLgen --config file not found: '{d}/config.pbtxt'" == str(
      e_info.value
    )


def test_main_print_cache_path_corpus(abc_instance_file, capsys):
  """Test that --print_cache_path=corpus prints directory path."""
  app.FLAGS.unparse_flags()
  app.FLAGS(
    ["argv[0]", "--config", abc_instance_file, "--print_cache_path=corpus"]
  )
  clgen.main([])
  out, err = capsys.readouterr()
  assert "/corpus/" in out
  assert pathlib.Path(out.strip()).is_dir()


def test_main_print_cache_path_model(abc_instance_file, capsys):
  """Test that --print_cache_path=model prints directory path."""
  app.FLAGS.unparse_flags()
  app.FLAGS(
    ["argv[0]", "--config", abc_instance_file, "--print_cache_path=model"]
  )
  clgen.main([])
  out, err = capsys.readouterr()
  assert "/model/" in out
  assert pathlib.Path(out.strip()).is_dir()


def test_main_print_cache_path_sampler(abc_instance_file, capsys):
  """Test that --print_cache_path=sampler prints directory path."""
  app.FLAGS.unparse_flags()
  app.FLAGS(
    ["argv[0]", "--config", abc_instance_file, "--print_cache_path=sampler"]
  )
  clgen.main([])
  out, err = capsys.readouterr()
  assert "/samples/" in out
  # A sampler's cache isn't created until Sample() is called.
  assert not pathlib.Path(out.strip()).is_dir()


def test_main_print_cache_invalid_argument(abc_instance_file):
  """Test that UsageError raised if --print_cache_path arg not valid."""
  app.FLAGS.unparse_flags()
  app.FLAGS(
    ["argv[0]", "--config", abc_instance_file, "--print_cache_path=foo"]
  )
  with test.Raises(app.UsageError) as e_info:
    clgen.main([])
  assert "Invalid --print_cache_path argument: 'foo'" == str(e_info.value)


def test_main_min_samples(abc_instance_file):
  """Test that min_samples samples are produced."""
  app.FLAGS.unparse_flags()
  app.FLAGS(["argv[0]", "--config", abc_instance_file, "--min_samples", "1"])
  clgen.main([])


def test_main_stop_after_corpus(abc_instance_file):
  """Test that --stop_after corpus prevents model training."""
  app.FLAGS.unparse_flags()
  app.FLAGS(
    ["argv[0]", "--config", abc_instance_file, "--stop_after", "corpus"]
  )
  clgen.main([])
  instance = clgen.Instance(
    pbutil.FromFile(pathlib.Path(abc_instance_file), clgen_pb2.Instance())
  )
  assert not instance.model.is_trained


def test_main_stop_after_train(abc_instance_file):
  """Test that --stop_after train trains the model."""
  app.FLAGS.unparse_flags()
  app.FLAGS(["argv[0]", "--config", abc_instance_file, "--stop_after", "train"])
  clgen.main([])
  instance = clgen.Instance(
    pbutil.FromFile(pathlib.Path(abc_instance_file), clgen_pb2.Instance())
  )
  assert instance.model.is_trained


def test_main_stop_after_uncrecognized(abc_instance_file):
  """Test that --stop_after raises an error on unknown."""
  app.FLAGS.unparse_flags()
  app.FLAGS(["argv[0]", "--config", abc_instance_file, "--stop_after", "foo"])
  with test.Raises(app.UsageError):
    clgen.main([])


if __name__ == "__main__":
  test.Main()
