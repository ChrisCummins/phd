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
import time
from absl import app
from absl import flags
from absl import logging
from github import Repository

from datasets.github.scrape_repos.proto import scrape_repos_pb2
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
    self.query = github_connection.search_repositories(
      f'language:{language.language} sort:stars fork:false')
    self.total_num_repos_on_github = self.query.totalCount
    self.next_page_num = 0
    super(LanguageCloneWorker, self).__init__()

  def GetNumberOfClonedRepos(self) -> int:
    """Get the number of repositories which have been cloned.

    Returns:
      The number of repos which have been cloned.
    """
    if self.destination_directory.is_dir():
      return len(
        [x for x in self.destination_directory.glob('*/.git') if x.is_dir()])
    else:
      return 0

  def GetNextBatchOfRepositories(self) -> typing.List[Repository.Repository]:
    """Get the next batch of repositories to clone from the query.

    Returns:
      A list of GitHub Repository instances.
    """
    while True:
      try:
        page = self.query.get_page(self.next_page_num)
        self.next_page_num += 1
        return list(page)
      except github.RateLimitExceededException:
        time.sleep(3)

  def IsDone(self) -> bool:
    num_cloned_repos = self.GetNumberOfClonedRepos()
    if num_cloned_repos >= self.language.num_repos_to_clone:
      return True
    if num_cloned_repos >= self.total_num_repos_on_github:
      logging.warning('Ran out of repositories to clone on GitHub')
      return True

  def run(self) -> None:
    """Execute the worker thread."""
    self.destination_directory.mkdir(parents=True, exist_ok=True)
    page = self.GetNextBatchOfRepositories()
    while not self.IsDone():
      num_remaining = (self.language.num_repos_to_clone -
                       self.GetNumberOfClonedRepos())
      repos = list(page)[:num_remaining]
      CloneRepositories(repos, self.destination_directory)
      page = self.GetNextBatchOfRepositories()


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


class RepositoryCloneWorker(object):
  """An object which clones a git repository from GitHub."""

  def __init__(self, destination_directory: pathlib.Path):
    """Instantiate a RepositoryCloneWorker.

    Args:
      destination_directory: The root directory to clone to.
    """
    self.destination_directory = destination_directory

  def __call__(self, repository: Repository.Repository) -> None:
    """Clone a git repository from GitHub.

    Args:
      repository: A Repository instance.
    """
    clone_dir = self.destination_directory / repository.name
    if not (clone_dir / '.git').is_dir():
      try:
        subprocess.check_call(
          ['timeout', '30m', '/usr/bin/git', 'clone', '--recursive',
           repository.clone_url, clone_dir],
          stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
      except subprocess.CalledProcessError:
        logging.warning('Clone failed %s', clone_dir)


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
