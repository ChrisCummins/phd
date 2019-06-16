"""This script combines the scrape, clone, and import stages.

This script is a pipelined implementation of the scraping, cloning, and
importing stages of the GitHub repo scraper. Instead of first scraping all
repos, then cloning them, then indexing them, this script performs all 3 stages
for each repository in turn. This reduces the required storage overhead.
"""
from datasets.github.scrape_repos import cloner
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import importer
from datasets.github.scrape_repos import scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2

import github
from github import Repository
from labm8 import app
from labm8 import fs

FLAGS = app.FLAGS
app.DEFINE_boolean(
    'pipelined_scraper_keep_files', False,
    'If set, keep temporary files: metadata protos and cloned repos.')


class PipelinedScraperError(EnvironmentError):
  """Raised in case scraping a repo fails."""
  pass


class PipelinedScraper(scraper.QueryScraper):

  def __init__(self, language: scrape_repos_pb2.LanguageToClone,
               query: scrape_repos_pb2.GitHubRepositoryQuery,
               github_connection: github.Github, db: contentfiles.ContentFiles):
    super(PipelinedScraper, self).__init__(language, query, github_connection)
    if not db:
      raise ValueError('contentfiles database not provided')
    self.contentfiles = db

  def DoProcessRepo(self, repo: Repository.Repository) -> None:
    # Scrape the repo, producing <dir>/<repo>.pbtxt
    super(PipelinedScraper, self).ProcessRepo(repo)
    meta_path = self.GetRepoMetaPath(repo)
    if not meta_path or not meta_path.is_file():
      raise PipelinedScraperError('Failed to scrape repo')

    # Clone the repo, producing directory <dir>/<repo>
    clone_dir = cloner.GetCloneDir(meta_path)
    app.Log(1, 'Cloning repo %s', repo.html_url)
    cloner.CloneFromMetafile(meta_path)
    if not clone_dir or not clone_dir.is_dir():
      raise PipelinedScraperError('Failed to clone repo')

    # Import to database.
    with self.contentfiles.Session() as session:
      if importer.ShouldImportRepo(session, meta_path):
        importer.ImportRepo(session, self.language, meta_path)
        session.commit()

  def ProcessRepo(self, repo: Repository.Repository) -> None:
    """Scrape, clone, and import a single repo."""
    try:
      self.DoProcessRepo(repo)
    except PipelinedScraperError as e:
      app.Error(str(e))
    finally:
      self.CleanupTemporaryFiles(repo)

  def CleanupTemporaryFiles(self, repo: Repository.Repository) -> None:
    """Delete temporary files."""
    if FLAGS.pipelined_scraper_keep_files:
      return

    meta_path = self.GetRepoMetaPath(repo)
    if not meta_path or not meta_path.is_file():
      return

    clone_dir = cloner.GetCloneDir(meta_path)
    meta_path.unlink()  # Must be done *after* resolving the clone dir.
    if clone_dir and clone_dir.is_dir():
      fs.rm(clone_dir)
