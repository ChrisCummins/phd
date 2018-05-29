"""Scrape the 'n' most popular GitHub repositories, by language.

This program reads a LanguageCloneList input, where each LanguageToClone entry
in the LanguageCloneList specifies a programming language on GitHub and a number
of repositories of this language to clone.
"""
import configparser
import pathlib
import sys
import threading
import time
import typing

import github
import humanize
import progressbar
from absl import app
from absl import flags
from absl import logging
from github import Repository

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from lib.labm8 import fs
from lib.labm8 import labdate
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('clone_list', None, 'The path to a LanguageCloneList file.')


def ReadGitHubCredentials() -> scrape_repos_pb2.GitHubCredentials:
  """Read user GitHub credentials from the local file system.

  Returns:
    A GitHubCredentials instance.
  """
  cfg = configparser.ConfigParser()
  cfg.read(pathlib.Path("~/.githubrc").expanduser())
  credentials = scrape_repos_pb2.GitHubCredentials()
  credentials.username = cfg["User"]["Username"]
  credentials.password = cfg["User"]["Password"]
  return credentials


def MakeRepositoryMetas(repos: typing.List[Repository.Repository],
                        destination_directory: pathlib.Path) -> None:
  """Make meta files for a list of repositories.

  Args:
    repos: A list of GitHub Repository instances.
    destination_directory: The directory to clone each repository in to.
  """
  logging.debug('Scraping %s repositories', humanize.intcomma(len(repos)))
  for repo in repos:
    concat_name = '_'.join([repo.owner.login, repo.name])
    clone_dir = destination_directory / concat_name
    meta_path = pathlib.Path(str(clone_dir) + '.pbtxt')
    if not pbutil.ProtoIsReadable(meta_path,
                                  scrape_repos_pb2.GitHubRepoMetadata()):
      meta = GetRepositoryMetadata(repo)
      logging.debug('%s', meta)
      pbutil.ToFile(meta, meta_path)


class LanguageScraper(threading.Thread):
  """Find popular repositories on GitHub and record their metadata.

  An instance of this class is executed as a thread.
  """

  def __init__(self, language: scrape_repos_pb2.LanguageToClone):
    """Instantiate a LanguageScraper.

    Args:
      language: A LanguageToClone instance.
    """
    self.language = language
    self.destination_directory = pathlib.Path(language.destination_directory)
    credentials = ReadGitHubCredentials()
    github_connection = github.Github(credentials.username,
                                      credentials.password)
    # Any access to the query properties can cause the rate limit to be
    # exceeded.
    while True:
      try:
        self.query = github_connection.search_repositories(
            f'language:{language.language} sort:stars fork:false')
        self.total_num_repos_on_github = self.query.totalCount
        break
      except github.RateLimitExceededException:
        logging.debug('Pausing on GitHub rate limit')
        time.sleep(3)
    self.next_page_num = 0
    super(LanguageScraper, self).__init__()

  def GetNumberOfRepoMetas(self) -> int:
    """Get the number of repositories which have been cloned.

    Returns:
      The number of repos which have been cloned.
    """
    if self.destination_directory.is_dir():
      return len([x for x in self.destination_directory.iterdir() if
                  x.suffix == '.pbtxt'])
    else:
      return 0

  def GetNextBatchOfRepositories(self) -> typing.List[Repository.Repository]:
    """Get the next batch of repositories to clone from the query.

    Returns:
      A list of GitHub Repository instances.
    """
    while True:
      try:
        logging.debug('Requesting page %d', self.next_page_num)
        page = list(self.query.get_page(self.next_page_num))
        logging.debug('Page %d contains %d results', self.next_page_num,
                      len(page))
        self.next_page_num += 1
        return page
      except github.RateLimitExceededException:
        logging.info('Pausing on GitHub rate limit')
        time.sleep(3)
      except github.GithubException:
        # One possible cause for this exception is when trying to request
        # a page beyond 1000 results, since GitHub only returns the first
        # 1000 results for a query.
        return []

  def IsDone(self, repos: typing.List[Repository.Repository]) -> bool:
    """Determine if we are done cloning repos.

    Args:
      repos: The next batch of repos to clone.

    Returns:
      True if we should stop cloning repos, else False.
    """
    if not repos:
      # An empty list indicates that we've run out of query results.
      return True
    num_scraped_repos = self.GetNumberOfRepoMetas()
    if num_scraped_repos >= self.language.num_repos_to_clone:
      return True
    if num_scraped_repos >= self.total_num_repos_on_github:
      logging.warning('Ran out of repositories to scrape')
      return True
    return False

  def run(self) -> None:
    """Execute the worker thread."""
    self.destination_directory.mkdir(parents=True, exist_ok=True)
    repos = self.GetNextBatchOfRepositories()
    while not self.IsDone(repos):
      num_remaining = (
          self.language.num_repos_to_clone - self.GetNumberOfRepoMetas())
      repos = repos[:num_remaining]
      MakeRepositoryMetas(repos, self.destination_directory)
      repos = self.GetNextBatchOfRepositories()


def RunScraper(worker: LanguageScraper) -> None:
  """Run a language scraper with an asynchronously updating progress bar.

  Args:
    worker: A LanguageScraper worker instance.
  """
  logging.info('Scraping %s of %s %s repos on GitHub ...',
               humanize.intcomma(worker.language.num_repos_to_clone),
               humanize.intcomma(worker.total_num_repos_on_github),
               worker.language.language)
  bar = progressbar.ProgressBar(max_value=worker.language.num_repos_to_clone,
                                redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.GetNumberOfRepoMetas())
    worker.join(.5)
  bar.update(worker.GetNumberOfRepoMetas())
  sys.stderr.write('\n')
  sys.stderr.flush()


def GetRepositoryMetadata(
    repo: Repository.Repository) -> scrape_repos_pb2.GitHubRepoMetadata():
  """Get metadata about a GitHub repository.

  Args:
    repo: A Repository instance.

  Returns:
    A GitHubRepoMetadata instance.
  """
  meta = scrape_repos_pb2.GitHubRepoMetadata()
  meta.scraped_utc_epoch_ms = labdate.MillisecondsTimestamp(
      labdate.GetUtcMillisecondsNow())
  meta.owner = repo.owner.login
  meta.name = repo.name
  meta.num_watchers = repo.watchers_count
  meta.num_forks = repo.forks_count
  meta.num_stars = repo.stargazers_count
  meta.clone_from_url = repo.clone_url
  return meta


class RepositoryCloneWorker(object):
  """An object which clones a git repository from GitHub."""

  def __init__(self, destination_directory: pathlib.Path):
    """Instantiate a RepositoryCloneWorker.

    Args:
      destination_directory: The root directory to clone to.
    """
    self.destination_directory = destination_directory

  def __call__(self, repo: Repository.Repository) -> None:
    """Clone a git repository from GitHub.

    Args:
      repo: A Repository instance.
    """
    concat_name = '_'.join([repo.owner.login, repo.name])

    clone_dir = self.destination_directory / concat_name
    meta_path = pathlib.Path(str(clone_dir) + '.pbtxt')
    if not pbutil.ProtoIsReadable(meta_path,
                                  scrape_repos_pb2.GitHubRepoMetadata()):
      fs.rm(meta_path)
      meta = GetRepositoryMetadata(repo)
      logging.debug('%s', meta)
      pbutil.ToFile(meta, meta_path)


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')

  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  for language in clone_list.language:
    RunScraper(LanguageScraper(language))


if __name__ == '__main__':
  app.run(main)
