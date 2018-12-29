"""Authenticating GtiHub API requests."""
import collections
import configparser
import pathlib
import typing

import github
from absl import flags


FLAGS = flags.FLAGS

# FIXME(cec): The default value is not hemetic.
flags.DEFINE_string(
    'github_credentials_path', '~/.githubrc',
    'The path to a file containing GitHub login credentials. See '
    '//datasets/github/scrape_repos/README.md for details.')

GitHubCredentials = collections.namedtuple(
    'GitHubCredentials', ['username', 'password'])


class InvalidGitHubCredentialsFile(ValueError):
  """Raised if credentials are invalid."""
  pass


def GitHubCredentialsFromFile(path: pathlib.Path) -> GitHubCredentials:
  """Read GitHub credentials from a file.

  Returns:
    A GitHubCredentials tuple.

  Raises:
    InvalidGitHubCredentialsFile: If the file is not valid.
  """
  if not path.is_file():
    raise InvalidGitHubCredentialsFile(f"File not found: {path}")

  cfg = configparser.ConfigParser()
  cfg.read(path)
  # TODO(cec): Handle errors when config values not set.
  username = cfg['User']['Username']
  password = cfg['User']['Password']
  return GitHubCredentials(username=username, password=password)


def GitHubCredentialsFromFlag() -> GitHubCredentials:
  """Read GitHub credentials from a flag."""
  return GitHubCredentialsFromFile(FLAGS.github_credentials_path)


def GetGitHubConnection(
    credentials: typing.Optional[GitHubCredentials] = None) -> github.Github:
  """Get a GitHub connection."""
  if credentials is None:
    credentials = GitHubCredentialsFromFlag()

  return github.Github(credentials.username, credentials.password)
