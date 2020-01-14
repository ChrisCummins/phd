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
"""Tests for //datasets/github/scrape_repos:scraper."""
import pytest

from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8.py import app
from labm8.py import labdate
from labm8.py import test

FLAGS = app.FLAGS


class MockNamedUser(object):
  """Mock class for github.NamedUser.NamedUser."""

  login = "login"


class MockRepository(object):
  """Mock class for github.Repository.Repository."""

  def __init__(self):
    self.owner = MockNamedUser()
    self.name = "name"
    self.watchers_count = 1
    self.forks_count = 2
    self.stargazers_count = 3
    self.clone_url = "url"


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
  assert now_ms - meta.scraped_unix_epoch_ms <= 1000


def test_main_unrecognized_arguments():
  """Test that main() raises an error when passed arguments."""
  with test.Raises(app.UsageError):
    scraper.main(["./scraper", "--unrecognized_argument"])


if __name__ == "__main__":
  test.Main()
