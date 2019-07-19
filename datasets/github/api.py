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
import github
import pathlib
import subprocess

from datasets.github import github_pb2
from labm8 import app


FLAGS = app.FLAGS

app.DEFINE_string(
    'github_access_token', None,
    'Github access token. See <https://github.com/settings/tokens> to '
    'generate an access token.')
app.DEFINE_string('github_access_token_path',
                  '/var/phd/github_access_token.txt',
                  'Path to a file containing a github access token.')
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

  First, it attempts to connect using the --github_access_token flag. If that
  flag is not set, then the contents of --github_access_token_path are used.
  If that file does not exist, --github_credentials_path is read.

  Returns:
    A PyGithub Github instance.
  """
  try:
    if FLAGS.github_access_token:
      return github.Github(FLAGS.github_access_token)
    elif pathlib.Path(FLAGS.github_access_token_path).is_file():
      with open(FLAGS.github_access_token_path) as f:
        access_token = f.read().strip()
      return github.Github(access_token)
    else:
      app.Warning("Using insecure --github_credentials_path to read GitHub "
                  "credentials. Please use token based credentials flags "
                  "--github_access_token or --github_access_token_path.")
      github_credentials_path = pathlib.Path(
          FLAGS.github_credentials_path).expanduser()
      if not github_credentials_path.is_file():
        app.FatalWithoutStackTrace('Github credentials file not found: %s',
                                   github_credentials_path)
      credentials = ReadGitHubCredentials(github_credentials_path.expanduser())
      return github.Github(credentials.username, credentials.password)
  except Exception as e:  # Deliberately broad catch-all.
    app.FatalWithoutStackTrace('Failed to create GitHub API connection: %s', e)


class RepoNotFoundError(ValueError):
  """Error thrown if a github repo is not found."""
  pass


def GetUserRepo(connection: github.Github, repo_name: str) -> github.Repository:
  """Get and return a github repository owned by the user.

  Args:
    connection: A github API connection.
    repo_name: The name of the repo to get.
  """
  try:
    return connection.get_user().get_repo(repo_name)
  except github.UnknownObjectException as e:
    if e.status != 404:
      raise OSError(f"Github API raised error: {e}")
    raise RepoNotFoundError(f"Github repo `{repo_name}` not found")


def GetOrCreateUserRepo(connection: github.Github, repo_name: str,
                        description: str=None, homepage: str=None,
                        has_wiki: bool=True, has_issues: bool=True,
                        private: bool=True) -> github.Repository:
  """Get and return a github repository owned by the user.

  Create it if it doesn't exist.

  Args:
    connection: A github API connection.
    repo_name: The name of the repo to get.
    description: The repo description.
    homepage: The repo homepage.
    has_wiki: Whether the repo has a wiki.
    has_issues: Whether the repo has an issue tracker.
    private: Whether the repo is private.
  """
  try:
    return GetUserRepo(connection, repo_name)
  except RepoNotFoundError:
    app.Log(1, "Creating repo %s", repo_name)
    connection.get_user().create_repo(
        repo_name,
        description=description,
        homepage=homepage,
        has_wiki=has_wiki,
        has_issues=has_issues,
        private=private)
    return GetUserRepo(connection, repo_name)


class RepoCloneFailed(OSError):
  """Error raised if repo fails to clone."""
  pass


def CloneRepoToDestination(repo: github.Repository, destination: pathlib.Path):
  """Clone repo from github."""
  subprocess.check_call(['git', 'clone', repo.ssh_url, str(destination)])
  if not (destination / '.git').is_dir():
    raise RepoCloneFailed(
        f"Cloned repo `{repo.ssh_url}` but `{destination}/.git` not found")
