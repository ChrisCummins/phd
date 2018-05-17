"""Tests for //datasets/github/scrape_repos:scraper."""
import pytest
import sys
from absl import app

from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2


def test_ReadGitHubCredentials():
  """Test that GitHub credentials are read from the filesystem."""
  # Note that the function ReadGitHubCredentials() depends on a file
  # (~/.githubrc) which is outside of this repo and not tracked. This will
  # fail on machines where we don't have this file.
  credentials = scraper.ReadGitHubCredentials()
  assert credentials.HasField('username')
  assert credentials.username
  assert credentials.HasField('password')
  assert credentials.password


def test_TODO():
  # TODO(cec): Placeholder for mocking tests of the cloner workers.
  _ = scrape_repos_pb2.GitHubRepo()


def test_main_unrecognized_arguments():
  """Test that main() raises an error when passed arguments."""
  with pytest.raises(app.UsageError):
    scraper.main(['./scraper', '--unrecognized_argument'])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
