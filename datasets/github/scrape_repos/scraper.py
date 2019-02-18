"""Scrape the 'n' most popular GitHub repositories, by language.

This program reads a LanguageCloneList input, where each LanguageToClone entry
in the LanguageCloneList specifies a programming language on GitHub and a number
of repositories of this language to clone.
"""
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

from datasets.github import api as github_api
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from labm8 import labdate
from labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'clone_list', None,
    'The path to a LanguageCloneList file.')


class QueryScraper(threading.Thread):
  """Scrape repository metadata from the results of GitHub search query.

  An instance of this class is executed as a thread.
  """

  def __init__(self, language: scrape_repos_pb2.LanguageToClone,
               query: scrape_repos_pb2.GitHubRepositoryQuery,
               github_connection: github.Github):
    """Instantiate a QueryScraper.

    Args:
      language: A LanguageToClone instance.
      query: The query to run.
      github_connection: A connection to GitHub API.
    """
    self.language = language
    self.repo_query = query
    self.destination_directory = pathlib.Path(language.destination_directory)
    self.i = 0
    # Any access to the query properties can cause the rate limit to be
    # exceeded.
    while True:
      try:
        self.query = github_connection.search_repositories(
            self.repo_query.string)
        self.total_result_count = self.query.totalCount
        break
      except (github.RateLimitExceededException, github.GithubException) as e:
        logging.debug('Pausing on GitHub error: %s', e)
        time.sleep(3)
    self.next_page_num = 0
    super(QueryScraper, self).__init__()

  def GetNumberOfResultsProcessed(self) -> int:
    """Get the number of results which have been processed."""
    return self.i

  def GetNextBatchOfResults(self) -> typing.List[Repository.Repository]:
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
        logging.debug('Pausing on GitHub rate limit')
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
    else:
      return self.i >= self.repo_query.max_results

  def run(self) -> None:
    """Execute the worker thread."""
    self.destination_directory.mkdir(parents=True, exist_ok=True)
    repos = self.GetNextBatchOfResults()
    while not self.IsDone(repos):
      num_remaining = (self.repo_query.max_results - self.i)
      repos = repos[:num_remaining]
      self.MakeRepositoryMetas(repos)
      repos = self.GetNextBatchOfResults()

  def MakeRepositoryMetas(self,
                          repos: typing.List[Repository.Repository]) -> None:
    """Make meta files for a list of repositories.

    Args:
      repos: A list of GitHub Repository instances.
    """
    logging.debug('Scraping %s repositories', humanize.intcomma(len(repos)))
    for repo in repos:
      self.i += 1
      concat_name = '_'.join([repo.owner.login, repo.name])
      clone_dir = self.destination_directory / concat_name
      meta_path = pathlib.Path(str(clone_dir) + '.pbtxt')
      if not pbutil.ProtoIsReadable(meta_path,
                                    scrape_repos_pb2.GitHubRepoMetadata()):
        meta = GetRepositoryMetadata(repo)
        logging.debug('%s', meta)
        pbutil.ToFile(meta, meta_path)


def RunQuery(worker: QueryScraper) -> None:
  """Run a language scraper with an asynchronously updating progress bar.

  Args:
    worker: A QueryScraper worker instance.
  """
  sys.stderr.flush()
  logging.info("Query '%s' returned %s results. Processing first %s ...",
               worker.repo_query.string,
               humanize.intcomma(worker.total_result_count),
               humanize.intcomma(worker.repo_query.max_results))
  bar = progressbar.ProgressBar(max_value=worker.repo_query.max_results,
                                redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.GetNumberOfResultsProcessed())
    worker.join(.5)
  bar.update(worker.GetNumberOfResultsProcessed())
  bar.finish()
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


def GetNumberOfRepoMetas(language: scrape_repos_pb2.LanguageToClone) -> int:
  """Get the number of repository metafiles for the language.

  Returns:
    The number of repos which have been cloned.
  """
  path = pathlib.Path(language.destination_directory)
  if path.is_dir():
    return len([x for x in path.iterdir() if x.suffix == '.pbtxt'])
  else:
    return 0


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  connection = github_api.GetGithubConectionFromFlagsOrDie()

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')

  clone_list = pbutil.FromFile(clone_list_path,
                               scrape_repos_pb2.LanguageCloneList())

  for language in clone_list.language:
    logging.info('Scraping %s repos using %s queries ...',
                 language.language, humanize.intcomma(len(language.query)))
    for query in language.query:
      RunQuery(QueryScraper(language, query, connection))

  logging.info('Finished scraping. Indexed repository counts:')
  for language in clone_list.language:
    logging.info('  %s: %s', language.language,
                 humanize.intcomma(GetNumberOfRepoMetas(language)))


if __name__ == '__main__':
  app.run(main)
