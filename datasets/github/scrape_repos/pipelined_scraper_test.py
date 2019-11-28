"""Unit tests for //datasets/github/scrape_repos:pipelined_scraper."""
import pathlib

import pytest

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import pipelined_scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def query(tempdir: pathlib.Path) -> scrape_repos_pb2.GitHubRepositoryQuery:
  return scrape_repos_pb2.GitHubRepositoryQuery(
    string="test query", max_results=10,
  )


@test.Fixture(scope="function")
def language(
  tempdir: pathlib.Path, query: scrape_repos_pb2.GitHubRepositoryQuery
) -> scrape_repos_pb2.LanguageCloneList:
  return scrape_repos_pb2.LanguageToClone(
    language="java",
    query=[query],
    destination_directory=str(tempdir),
    importer=[
      scrape_repos_pb2.ContentFilesImporterConfig(
        source_code_pattern=".*\\.java"
      )
    ],
  )


class MockQuery(object):
  """A mock GitHub query."""

  def __init__(self, results):
    self._results = results
    self._returned_page = False
    self.totalCount = len(results)

  def get_page(self, num):
    """Return the "results" constructor arg on the first call."""
    del num
    if self._returned_page:
      return []
    else:
      self._returned_page = True
      return self._results


class MockNamedUser(object):
  """Mock class for github.NamedUser.NamedUser."""

  login = "ChrisCummins"


class MockRepository(object):
  """Mock class for github.Repository.Repository."""

  def __init__(self):
    self.owner = MockNamedUser()
    self.name = "empty_repository_for_testing"
    self.watchers_count = 1
    self.forks_count = 0
    self.stargazers_count = 0
    self.html_url = (
      "https://github.com/ChrisCummins/empty_repository_for_testing"
    )
    self.clone_url = (
      "https://github.com/ChrisCummins/empty_repository_for_testing.git"
    )


class MockGitHubConnection(object):
  """Mock class for github connections."""

  def __init__(self):
    self.search_repositories_args = []

  def search_repositories(self, query_string: str):
    self.search_repositories_args.append(query_string)
    return MockQuery([MockRepository()])


@test.Fixture(scope="function")
def connection():
  return MockGitHubConnection()


@test.Fixture(scope="function")
def db(tempdir: pathlib.Path) -> contentfiles.ContentFiles:
  return contentfiles.ContentFiles(f"sqlite:///{tempdir}/db")


def test_PipelinedScraper_runs_query(
  language: scrape_repos_pb2.LanguageCloneList,
  query: scrape_repos_pb2.GitHubRepositoryQuery,
  connection: MockGitHubConnection,
  db: contentfiles.ContentFiles,
):
  """Test that github query is executed."""
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  assert connection.search_repositories_args == [query.string]


def test_PipelinedScraper_temporary_files_are_deleted(
  language: scrape_repos_pb2.LanguageCloneList,
  query: scrape_repos_pb2.GitHubRepositoryQuery,
  connection: MockGitHubConnection,
  db: contentfiles.ContentFiles,
):
  """Test that meta files and cloned directories are deleted."""
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  destination_directory = pathlib.Path(language.destination_directory)
  assert len(list(destination_directory.iterdir())) == 1  # 1 file = database


# The text of the HelloWorld.java file from:
# https://github.com/ChrisCummins/empty_repository_for_testing/
HELLO_WORLD_TEXT = """\
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, World");
  }
}
"""


def test_PipelinedScraper_contentfiles_database_contents(
  language: scrape_repos_pb2.LanguageCloneList,
  query: scrape_repos_pb2.GitHubRepositoryQuery,
  connection: MockGitHubConnection,
  db: contentfiles.ContentFiles,
):
  """Test database contents."""
  # This test will fail if the contents of GitHub repository
  # https://github.com/ChrisCummins/empty_repository_for_testing change.
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  with db.Session() as session:
    assert session.query(contentfiles.ContentFile).count() == 1
    contentfile = session.query(contentfiles.ContentFile).first()

    assert contentfile.clone_from_url == (
      "https://github.com/ChrisCummins/empty_repository_for_testing.git"
    )
    assert contentfile.relpath == "HelloWorld.java"
    assert contentfile.artifact_index == 0
    assert contentfile.text == HELLO_WORLD_TEXT
    assert contentfile.charcount == len(HELLO_WORLD_TEXT)
    assert contentfile.linecount == len(HELLO_WORLD_TEXT.split("\n"))


def test_PipelinedScraper_contentfiles_database_repo_contents(
  language: scrape_repos_pb2.LanguageCloneList,
  query: scrape_repos_pb2.GitHubRepositoryQuery,
  connection: MockGitHubConnection,
  db: contentfiles.ContentFiles,
):
  """Test database contents."""
  # This test will fail if the contents of GitHub repository
  # https://github.com/ChrisCummins/empty_repository_for_testing change.
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  with db.Session() as session:
    assert session.query(contentfiles.GitHubRepository).count() == 1
    repo = session.query(contentfiles.GitHubRepository).first()

    assert repo.clone_from_url == (
      "https://github.com/ChrisCummins/empty_repository_for_testing.git"
    )


def test_PipelinedScraper_contentfiles_database_ignores_duplicates(
  language: scrape_repos_pb2.LanguageCloneList,
  query: scrape_repos_pb2.GitHubRepositoryQuery,
  connection: MockGitHubConnection,
  db: contentfiles.ContentFiles,
):
  """Test database contents."""
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  with db.Session() as session:
    original_contentfile_count = session.query(contentfiles.ContentFile).count()
    assert original_contentfile_count

  # Run the scraper again.
  scraper = pipelined_scraper.PipelinedScraper(language, query, connection, db)
  scraper.start()
  scraper.join()
  with db.Session() as session:
    assert (
      session.query(contentfiles.ContentFile).count()
      == original_contentfile_count
    )


if __name__ == "__main__":
  test.Main()
