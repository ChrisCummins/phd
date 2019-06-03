# Copyright 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Global configuration of this repository.

This module exposes a single function GetGlobalConfig() which returns the
global configuration of the repository. The configuration schema is defined in
//config.proto, and the instance is created by the ./configure
script.
"""
import os

import config_pb2
import pathlib

from labm8 import pbutil


# The path of the generated config file, which is //config.pbtxt.
GLOBAL_CONFIG_PATH = pathlib.Path(
    pathlib.Path(os.path.dirname(os.path.realpath(__file__))) /
    'config.pbtxt').absolute()


class ConfigNotFound(EnvironmentError):
  """Error thrown in case configuration cannot be found."""

  def __init__(self, path: pathlib.Path):
    """Instantiate the error.

    Args:
      path: The path of the error file which could not be found.
    """
    self.path = path

  def __repr__(self):
    return f'GlobalConfig file {self.path} not found'

  def __str__(self):
    return self.__repr__()


class CorruptConfig(EnvironmentError):
  """Error thrown in case configuration cannot be parsed."""

  def __init__(self, path: pathlib.Path, original_exception):
    """Instantiate the error.

    Args:
      path: The path of the error file which could not be parsed.
      original_exception: The original exception raised during parsing.
    """
    self.path = path
    self.original_exception = original_exception

  def __repr__(self):
    return (f'GlobalConfig file {self.path} could not be parsed.\n'
            f'Original error: {self.original_exception}')

  def __str__(self):
    return self.__repr__()


def GetGlobalConfig(
    path: pathlib.Path = GLOBAL_CONFIG_PATH) -> config_pb2.GlobalConfig:
  """Read and return the GlobalConfig proto.

  Args:
    path: The path of the config to read. The default is the repository config.

  Returns:
    A GlobalConfig instance.

  Raises:
    ConfigNotFound: In case no file exists at the given path.
    CorruptConfig: In case the configuration file could not be parsed.
  """
  if not path.is_file():
    raise ConfigNotFound(path)
  try:
    config = pbutil.FromFile(path, config_pb2.GlobalConfig())
  except pbutil.DecodeError as e:
    raise CorruptConfig(path, e)
  return config
