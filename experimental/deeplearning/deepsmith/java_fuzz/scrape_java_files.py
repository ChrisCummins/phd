"""Scrape Java files from GitHub and put them in a contentfiles database for
later processing into a corpus."""

import time

import pathlib
import random
import tempfile
import typing
from datasets.github import api as github_api
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos import pipelined_scraper
from datasets.github.scrape_repos.proto import scrape_repos_pb2

import github
import urllib3
from labm8 import app
from labm8 import humanize


FLAGS = app.FLAGS
app.DEFINE_integer('n', int(1e6),
                   'The number of repos to scrape before terminating.')
app.DEFINE_database(
    'db', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/java.db',
    'URL of the database to scrape Java files to.')

WORD_LIST_URLS = [
    # The 1,000 most commonly used words.
    "https://gist.githubusercontent.com/deekayen/4148741/raw/01c6252ccc5b5fb307c1bb899c95989a8a284616/1-1000.txt",
    # 479k English words
    "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
    # Unofficial Jargon File Word Lists
    "https://raw.githubusercontent.com/en-wl/wordlist/master/jargon-wl/word.lst",
    # List of Dirty, Naughty, Obscene, and Otherwise Bad Words
    "https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/raw/master/en",
]

# A list of GitHub repos to ignore.
BLACKLIST_GITHUB_REPOS = [
    "https://github.com/adoptopenjdk/openjdk-tests.git",
    "https://github.com/adoptopenjdk/openjdk-systemtest.git",
    "https://github.com/adoptopenjdk/stf.git",
    "https://github.com/eclipse/openj9.git",
    "https://github.com/eclipse/openj9-systemtest.git",
]


def GetJavaQuery(prefix: str):
  return scrape_repos_pb2.GitHubRepositoryQuery(
      string=f"{prefix} language:java sort:stars fork:false")


def GetLanguageToClone(query_prefix: str, destination_dir: str
                      ) -> scrape_repos_pb2.LanguageToClone:
  return scrape_repos_pb2.LanguageToClone(
      language="java",
      # See: https://help.github.com/en/articles/sorting-search-results
      sort_by = random.choice(['stars', 'forks', 'updated'])
      query=[
          scrape_repos_pb2.GitHubRepositoryQuery(
              string=f"{query_prefix} language:java sort:{sort_by} fork:false")
      ],
      destination_directory=destination_dir,
      importer=[
          scrape_repos_pb2.ContentFilesImporterConfig(
              source_code_pattern='.*\\.java')
      ],
      clone_from_url_blacklist=BLACKLIST_GITHUB_REPOS,
  )


class FuzzyGitHubJavaScraper(object):
  """Class to scrape Java files from GitHub."""

  def __init__(self, connection: github.Github, db: contentfiles.ContentFiles,
               word_list: typing.List[str]):
    self.connection = connection
    self.db = db
    self.word_list = word_list
    self.last_time_check = 0

  def DoRun(self, tempdir: str, n: int):
    i = 0
    random.shuffle(self.word_list)
    start_time = time.time()

    for word in self.word_list:
      language = GetLanguageToClone(word, tempdir)
      scraper = pipelined_scraper.PipelinedScraper(language, language.query[0],
                                                   self.connection, self.db)
      app.Log(1, "Query '%s' returned %s results. Processing first %s ...",
              scraper.repo_query.string,
              humanize.Commas(scraper.total_result_count),
              humanize.Commas(scraper.repo_query.max_results))
      scraper.start()
      while scraper.is_alive():
        query_i = scraper.GetNumberOfResultsProcessed()
        scraper.join(1)
        if i + query_i >= n:
          app.Log(1, "Scraped %s repositories. Done.",
                  humanize.Commas(i + query_i))
          scraper.Stop()
          scraper.join()
          break
        current_time = time.time()
        if current_time - self.last_time_check > 15:
          self.last_time_check = current_time
          app.Log(1, "Scraped %s of %s repos (%.2f %%) in %s",
                  humanize.Commas(i + query_i), humanize.Commas(n),
                  ((i + query_i) / n) * 100,
                  humanize.Duration(time.time() - start_time))

      i += query_i
      if i >= n:
        app.Log(1, "Done!")
        return

  def Run(self, n: int):
    """Scrape *.java files from 'n' GitHub repos."""
    with tempfile.TemporaryDirectory(prefix='phd_java_fuzz_') as tempdir:
      self.DoRun(tempdir, n)


def GetCertificateBundle() -> typing.Optional[str]:
  bundles = [
      '/etc/ssl/certs/ca-certificates.crt', '/usr/local/etc/openssl/cert.pem'
  ]
  for bundle in bundles:
    if pathlib.Path(bundle).is_file():
      return bundle


def main():
  """Main entry point."""
  http = urllib3.PoolManager(ca_certs=GetCertificateBundle())

  url = random.choice(WORD_LIST_URLS)

  app.Log(1, 'Downloading word list from %s', url)
  response = http.request('GET', url)
  words = response.data.decode('utf-8').strip().split('\n')
  app.Log(1, 'Beginning fuzzy GitHub scraping using %s words ...',
          humanize.Commas(len(words)))

  connection = github_api.GetGithubConectionFromFlagsOrDie()
  scraper = FuzzyGitHubJavaScraper(connection, FLAGS.db(), words)
  scraper.Run(FLAGS.n)


if __name__ == '__main__':
  app.Run(main)
