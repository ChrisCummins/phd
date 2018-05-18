"""Scrape the 'n' most popular GitHub repositories, by language.

This program reads a LanguageCloneList input, where each LanguageToClone entry
in the LanguageCloneList specifies a programming language on GitHub and a number
of repositories of this language to clone.
"""
import configparser
import multiprocessing
import pathlib
import subprocess
import threading
import typing

import github
import progressbar
import sys
import time
from absl import app
from absl import flags
from absl import logging
from github import Repository

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from lib.labm8 import fs
from lib.labm8 import pbutil

FLAGS = flags.FLAGS

flags.DEFINE_string(
  'clone_list', None,
  'The path to a LanguageCloneList file.')
flags.DEFINE_integer(
  'repository_clone_timeout_minutes', 30,
  'The maximum number of minutes to attempt to clone a repository before '
  'quitting and moving on to the next repository.')


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


def CloneRepositories(repos: typing.List[Repository.Repository],
                      destination_directory: pathlib.Path) -> None:
  """Clone a list of repositories using RepositoryCloneWorkers in parallel.

  Args:
    repos: A list of GitHub Repository instances.
    destination_directory: The directory to clone each repository in to.
  """
  logging.debug('Cloning %d repositories', len(repos))
  pool = multiprocessing.Pool()
  cloner = RepositoryCloneWorker(destination_directory)
  pool.imap_unordered(cloner, repos)


class LanguageCloneWorker(threading.Thread):
  """A class which clones repositories of a particular language from GitHub.

  An instance of this class is executed as a thread.
  """

  def __init__(self, language: scrape_repos_pb2.LanguageToClone):
    """Instantiate a LanguageCloneWorker.

    Args:
      language: A LanguageToClone instance.
    """
    self.language = language
    self.destination_directory = pathlib.Path(language.destination_directory)

    credentials = ReadGitHubCredentials()
    github_connection = github.Github(credentials.username, credentials.password)
    # Any access to the query properties can cause the rate limit to be
    # exceeded.
    while True:
      try:
        self.query = github_connection.search_repositories(
          f'language:{language.language} sort:stars fork:false')
        self.total_num_repos_on_github = self.query.totalCount
        break
      except github.RateLimitExceededException:
        logging.info('Pausing on GitHub rate limit')
        time.sleep(3)
    self.next_page_num = 0
    super(LanguageCloneWorker, self).__init__()

  def GetNumberOfClonedRepos(self) -> int:
    """Get the number of repositories which have been cloned.

    Returns:
      The number of repos which have been cloned.
    """
    if self.destination_directory.is_dir():
      return len(
        [x for x in self.destination_directory.iterdir()
         if x.suffix == '.pbtxt'])
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
        logging.debug('Page %d contains %d results',
                      self.next_page_num, len(page))
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
    num_cloned_repos = self.GetNumberOfClonedRepos()
    if num_cloned_repos >= self.language.num_repos_to_clone:
      return True
    if num_cloned_repos >= self.total_num_repos_on_github:
      logging.warning('Ran out of repositories to clone on GitHub')
      return True
    return False

  def run(self) -> None:
    """Execute the worker thread."""
    self.destination_directory.mkdir(parents=True, exist_ok=True)
    repos = self.GetNextBatchOfRepositories()
    while not self.IsDone(repos):
      num_remaining = (self.language.num_repos_to_clone -
                       self.GetNumberOfClonedRepos())
      repos = repos[:num_remaining]
      CloneRepositories(repos, self.destination_directory)
      repos = self.GetNextBatchOfRepositories()


def RunLanguageCloneWorker(worker: threading.Thread) -> None:
  """Run a language clone worker with an asynchronously updating progress bar.

  Args:
    worker: A language clone worker instance.
  """
  logging.info('Cloning %d of %d %s repos on GitHub ...',
               worker.language.num_repos_to_clone,
               worker.total_num_repos_on_github,
               worker.language.language)
  bar = progressbar.ProgressBar(max_value=worker.language.num_repos_to_clone,
                                redirect_stderr=True)
  worker.start()
  while worker.is_alive():
    bar.update(worker.GetNumberOfClonedRepos())
    worker.join(.5)
  bar.update(worker.GetNumberOfClonedRepos())
  print('', file=sys.stderr)
  sys.stderr.flush()


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
    if not ((clone_dir / '.git').is_dir() and
            pbutil.ProtoIsReadable(
              meta_path, scrape_repos_pb2.GitHubRepoMetadata())):
      fs.rm(clone_dir)
      fs.rm(meta_path)

      meta = scrape_repos_pb2.GitHubRepoMetadata()
      meta.owner = repo.owner.login
      meta.name = repo.name
      meta.num_watchers = repo.watchers_count
      meta.num_forks = repo.forks_count
      meta.num_stars = repo.stargazers_count
      meta.clone_from_url = repo.clone_url

      logging.debug('%s', meta)
      try:
        subprocess.check_call(
          ['timeout', '30m', '/usr/bin/git', 'clone', '--recursive',
           repo.clone_url, clone_dir],
          stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        pbutil.ToFile(meta, meta_path)
      except subprocess.CalledProcessError:
        logging.warning('\nClone failed %s', clone_dir)
        raise Exception()
      except:
        fs.rm(clone_dir)
        fs.rm(meta_path)


def main(argv) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  clone_list_path = pathlib.Path(FLAGS.clone_list or "")
  if not clone_list_path.is_file():
    raise app.UsageError('--clone_list is not a file.')

  clone_list = pbutil.FromFile(
    clone_list_path, scrape_repos_pb2.LanguageCloneList())

  for language in clone_list.language:
    RunLanguageCloneWorker(LanguageCloneWorker(language))


if __name__ == '__main__':
  app.run(main)
