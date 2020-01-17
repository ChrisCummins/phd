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
"""This module defines a unified method to connect to the Github API.

Authenticating access to the Github API is done exclusively through access
tokens. The function GetDefaultGithubAccessToken() attempts to resolve an access
token using the following order:

(1) If access token paths are provided, first try and read them, if they exist.
(2) If $GITHUB_ACCESS_TOKEN is set, use it.
(3) If $GITHUB_ACCESS_TOKEN_PATH is set and points to a file, attempt to read
    it. If the variable is set but the file does not exist, a warning is
    printed.
(4) If --github_access_token is set, use it.
(5) If --github_credentials_path points to a file, use it.
(6) If we are in a test environment and ~/.github/access_tokens/test.txt is a
    file, read the token from that.

If writing code that requires connecting to Github, use the
GetDefaultGithubConnection() or GetDefaultGithubConnectionOrDie() functions
defined in this module. Don't write your own credentials handling code.
"""
import os
import pathlib
import socket
import subprocess
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Union

import github

from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string(
  "github_access_token",
  None,
  "A Github access token to use for authenticating with Github API. "
  "To generate an access token, visit "
  "<https://github.com/settings/tokens/new>. Please see the documentation for "
  "this app for details on which scopes are required.",
)
app.DEFINE_output_path(
  "github_access_token_path",
  pathlib.Path("~/.github/access_tokens/default.txt").expanduser(),
  "Path to a file containing a Github access token for authenticated with the"
  "Github API. The file should contain a single line with the access token, "
  "and optional comment lines starting with '#' character. To generate an "
  "access token, visit <https://github.com/settings/tokens/new>. Please see "
  "the documentation for this app for details on which scopes are required.",
)

# The path to file containing a Github access token to use for running tests.
# When $TEST_TMPDIR is set (as done by bazel's test environment), calls to
# GetDefaultGithubAccessToken() will attempt to read an access token from this
# path.
#
# When writing tests that require a live Github API connection, be sure to guard
# the test to check for this file, so that the tests will be skipped on systems
# without an access token for testing. E.g.
#
#     from datasets.github.api.testing.requires_access_token import requires_access_token
#
#     @requires_access_token
#     def test_something_on_github():
#       github = api.GetDefaultGithubConnectionOrDie()
#       # go nuts ...
#
# It is assumed that the access token in this file has no scopes.
TEST_ACCESS_TOKEN_PATH = pathlib.Path(
  "~/.github/access_tokens/test.txt"
).expanduser()


class ConnectionFailed(OSError):
  """Error raised if connection to the Github API fails."""

  pass


class BadCredentials(ConnectionFailed):
  """Error raised if authenticating a connection to the Github API fails."""

  pass


def ReadGithubAccessTokenPath(path: pathlib.Path) -> str:
  """Read a Github access token from a file.

  Returns:
    A string access token.

  Raises:
    FileNotFoundError: If the file does not exist.
    BadCredentials: If an access token cannot be read in the given path.
  """
  try:
    with open(path) as f:
      for line in f:
        if not line.startswith("#"):
          return line.strip()
  except PermissionError:
    raise BadCredentials(f"Cannot read file")
  except IsADirectoryError:
    raise BadCredentials(f"File is a directory")

  raise BadCredentials(f"Access token not found in file")


class AccessToken(NamedTuple):
  """The source an access token and the token itself."""

  source: str
  token: str

  def __repr__(self):
    return self.token


def GetDefaultGithubAccessToken(
  extra_access_token_paths: Optional[Iterable[Union[str, pathlib.Path]]] = None
) -> AccessToken:
  """Get a Github access token from environment variables or flags.

  This function provides a uniform means to get a Github access token, resolving
  the access token in the following order:

    (1) If access token paths are provided, first try and read them, if they
        exist.
    (2) If $GITHUB_ACCESS_TOKEN is set, use it.
    (3) If $GITHUB_ACCESS_TOKEN_PATH is set and points to a file, attempt to
        read it. If the variable is set but the file does not exist, a warning
        is printed.
    (4) If --github_access_token is set, use it.
    (5) If --github_credentials_path points to a file, use it.
    (6) If we are in a test environment and ~/.github/access_tokens/test.txt is
        a file, read the token from that.

  Args:
    extra_access_token_paths: A sequence of paths to read Github access tokens
      from. If provided, these paths take precedence over the default locations
      for access tokens.

  Returns:
    An AccessToken tuple.

  Raises:
    BadCredentials: If all of the access token
      resolution methods failed and --github_credentials_path does not exist,
      or if reading an access token from a path fails.
  """

  def _ReadGithubAccessTokenPath(source: str, path: pathlib.Path):
    try:
      return AccessToken(source, ReadGithubAccessTokenPath(path))
    except BadCredentials as e:
      raise BadCredentials(f"Invalid credentials file {source}: {e}")

  extra_access_token_paths = extra_access_token_paths or []
  for extra_access_token_path in extra_access_token_paths:
    extra_access_token_path = pathlib.Path(extra_access_token_path).expanduser()
    if extra_access_token_path.is_file():
      return _ReadGithubAccessTokenPath(
        str(extra_access_token_path), extra_access_token_path
      )

  access_token = os.environ.get("GITHUB_ACCESS_TOKEN")
  if access_token:
    return AccessToken("$GITHUB_ACCESS_TOKEN", access_token)

  access_token_path = os.environ.get("GITHUB_ACCESS_TOKEN_PATH")
  if access_token_path and pathlib.Path(access_token_path).is_file():
    return _ReadGithubAccessTokenPath(
      f"$GITHUB_ACCESS_TOKEN_PATH={access_token_path}",
      pathlib.Path(access_token_path),
    )
  elif access_token_path:
    app.Warning(
      "$GITHUB_ACCESS_TOKEN_PATH set but not found: %s", access_token_path
    )

  if FLAGS.github_access_token:
    return AccessToken("--github_access_token", FLAGS.github_access_token)

  if FLAGS.github_access_token_path.is_file():
    return _ReadGithubAccessTokenPath(
      f"--github_access_token_path={FLAGS.github_access_token_path}",
      FLAGS.github_access_token_path,
    )

  if os.environ.get("TEST_TMPDIR") and TEST_ACCESS_TOKEN_PATH.is_file():
    return _ReadGithubAccessTokenPath(
      f"test_token=TEST_ACCESS_TOKEN_PATH", TEST_ACCESS_TOKEN_PATH
    )

  raise BadCredentials(
    f"--github_access_token_path not found: {FLAGS.github_access_token_path}"
  )


def GetDefaultGithubConnection(
  extra_access_token_paths: Optional[Iterable[Union[str, pathlib.Path]]] = None,
  verify: bool = True,
) -> github.Github:
  """Construct a Github API connection using default access token resolution.

  See GetDefaultGithubAccessToken() for access token resolution.

  Args:
    extra_access_token_paths: A sequence of paths to read Github access tokens
      from. If provided, these paths take precedence over the default locations
      for access tokens.
    verify: Whether to check if the connection established works. If False,
      authentication errors may be deferred until later in program execution
      when a github.BadCredentialsException is raised. Verifying a connection
      requires network access.

  Returns:
    A Github instance.

  Raises:
    ConnectionFailed: If verify is True and the verification API call fails,
      such as due to invalid credentials, or a network error.
  """
  access_token = GetDefaultGithubAccessToken(
    extra_access_token_paths=extra_access_token_paths
  )
  connection = github.Github(str(access_token))
  app.Log(2, "Connecting to Github using %s", access_token.source)

  if verify:
    try:
      connection.get_rate_limit()
      app.Log(3, "Github connection verified")
    except socket.gaierror as e:
      raise ConnectionFailed(f"Failed to connect to Github API: {e}")
    except github.BadCredentialsException:
      raise BadCredentials(
        "Authentication using the Github access token from "
        f"{access_token.source} failed"
      )

  return connection


def GetDefaultGithubConnectionOrDie(
  extra_access_token_paths: Optional[Iterable[Union[str, pathlib.Path]]] = None,
  verify: bool = True,
) -> github.Github:
  """Construct a Github API connection and terminate on failure.

  This is the same as GetDefaultGithubConnection(), except failures result in
  the process terminating rather than raising a BadCredentials exception. This
  is a convenience function for scripts which cannot recover from the inability
  to communicate with Github, use at your own risk.

  Args:
    extra_access_token_paths: A sequence of paths to read Github access tokens
      from. If provided, these paths take precedence over the default locations
      for access tokens.
    verify: Whether to check if the connection established works. If False,
      authentication errors may be deferred until later in program execution
      when a github.BadCredentialsException is raised. Verifying a connection
      requires network access.

  Returns:
    A Github instance.
  """
  try:
    return GetDefaultGithubConnection(
      extra_access_token_paths=extra_access_token_paths, verify=verify
    )
  except ConnectionFailed as e:
    app.FatalWithoutStackTrace("%s", e)


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
  except socket.gaierror as e:
    raise OSError(f"Connection failed with error: {e}")
  except github.UnknownObjectException as e:
    if e.status != 404:
      raise OSError(f"Github API raised error: {e}")
    raise RepoNotFoundError(f"Github repo `{repo_name}` not found")


def GetOrCreateUserRepo(
  connection: github.Github,
  repo_name: str,
  description: str = None,
  homepage: str = None,
  has_wiki: bool = True,
  has_issues: bool = True,
  private: bool = True,
) -> github.Repository:
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
      private=private,
    )
    return GetUserRepo(connection, repo_name)


class RepoCloneFailed(OSError):
  """Error raised if repo fails to clone."""

  pass


def CloneRepoToDestination(repo: github.Repository, destination: pathlib.Path):
  """Clone repo from github."""
  subprocess.check_call(["git", "clone", repo.ssh_url, str(destination)])
  if not (destination / ".git").is_dir():
    raise RepoCloneFailed(
      f"Cloned repo `{repo.ssh_url}` but `{destination}/.git` not found"
    )


def Main():
  """Main entry point."""
  access_token = GetDefaultGithubAccessToken()
  GetDefaultGithubConnectionOrDie()
  print("Authenticated access to Github API using", access_token.source)


if __name__ == "__main__":
  app.Run(Main)
