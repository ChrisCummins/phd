# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
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
"""A utility module for creating GitHub API connections.

If writing code that requires connecting to GitHub, use the
GetGithubConectionFromFlagsOrDie() function defined in this module. Don't write
your own credentials handling code.
"""
import configparser
import pathlib

import github

from datasets.github import github_pb2
from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_string(
    'github_credentials_path', '~/.githubrc',
    'The path to a file containing GitHub login credentials. See '
    '//datasets/github/scrape_repos/README.md for details.')


def ReadGitHubCredentials(path: pathlib.Path) -> github_pb2.GitHubCredentials:
  """Read user GitHub credentials from the local file system.

  Returns:
    A GitHubCredentials instance.
  """
  cfg = configparser.ConfigParser()
  cfg.read(path)
  credentials = github_pb2.GitHubCredentials()
  credentials.username = cfg["User"]["Username"]
  credentials.password = cfg["User"]["Password"]
  return credentials


def GetGithubConectionFromFlagsOrDie() -> github.Github:
  """Get a GitHub API connection or die.

  Returns:
    A PyGithub Github instance.
  """
  try:
    credentials = ReadGitHubCredentials(
        pathlib.Path(FLAGS.github_credentials_path).expanduser())
    return github.Github(credentials.username, credentials.password)
  except Exception as e:  # Deliberately broad catch-all.
    app.Fatal('Failed to create GitHub API connection: %s', e)
