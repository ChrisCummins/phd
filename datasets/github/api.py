"""A utility module for creating GitHub API connections.

If writing code that requires connecting to GitHub, use the
GetGithubConectionFromFlagsOrDie() function defined in this module. Don't write
your own credentials handling code.
"""
import configparser
import pathlib

import github
from absl import flags
from absl import logging

from datasets.github import github_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'github_credentials_path', '~/.githubrc',
    'The path to a file containing GitHub login credentials. See '
    '//datasets/github/scrape_repos/README.md for details.')


def ReadGitHubCredentials(
    path: pathlib.Path) -> github_pb2.GitHubCredentials:
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


def GetGithubConectionFromFlagsOrDie():
  """Get a GitHub API connection or die.

  Returns:
    A PyGithub Github instance.
  """
  try:
    credentials = ReadGitHubCredentials(FLAGS.github_credentials_path)
    return github.Github(credentials.username, credentials.password)
  except Exception as e:  # Deliberately broad catch-all.
    logging.fatal('Failed to create GitHub API connection: %s', e)
