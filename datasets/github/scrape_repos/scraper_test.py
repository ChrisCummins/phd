"""Tests for //datasets/github/scrape_repos:scraper."""
import pathlib
import pytest
import sys
import tempfile
from absl import app
from phd.lib.labm8 import labdate

from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2


class MockNamedUser(object):
  """Mock class for github.NamedUser.NamedUser."""
  login = 'login'


class MockRepository(object):
  """Mock class for github.Repository.Repository."""

  def __init__(self):
    self.owner = MockNamedUser()
    self.name = 'name'
    self.watchers_count = 1
    self.forks_count = 2
    self.stargazers_count = 3
    self.clone_url = 'url'


@pytest.fixture(scope='function')
def credentials_file() -> pathlib.Path:
  """A test fixture to yield a GitHub credentials file."""
  with tempfile.TemporaryDirectory() as d:
    with open(pathlib.Path(d) / 'credentials', 'w') as f:
      f.write("""
[User]
Username = foo
Password = bar
""")
    yield pathlib.Path(d) / 'credentials'


def test_ReadGitHubCredentials(credentials_file: pathlib.Path):
  """Test that GitHub credentials are read from the filesystem."""
  credentials = scraper.ReadGitHubCredentials(credentials_file)
  assert credentials.HasField('username')
  assert credentials.username == 'foo'
  assert credentials.HasField('password')
  assert credentials.password == 'bar'


def test_GetRepositoryMetadata():
  repo = MockRepository()
  meta = scraper.GetRepositoryMetadata(repo)
  assert isinstance(meta, scrape_repos_pb2.GitHubRepoMetadata)
  assert meta.owner == repo.owner.login
  assert meta.name == repo.name
  assert meta.num_watchers == repo.watchers_count
  assert meta.num_forks == repo.forks_count
  assert meta.num_stars == repo.stargazers_count
  assert meta.clone_from_url == repo.clone_url


def test_GetRepositoryMetadata_timestamp():
  """Test that the timestamp in metadata is set to (aprox) now."""
  now_ms = labdate.MillisecondsTimestamp(labdate.GetUtcMillisecondsNow())
  meta = scraper.GetRepositoryMetadata(MockRepository())
  assert now_ms - meta.scraped_utc_epoch_ms <= 1000


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
