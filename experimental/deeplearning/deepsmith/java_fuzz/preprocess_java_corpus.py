"""Preprocess an exported database of Java methods.

From the input database, it exports contentfiles from repositories where
  repositories.active = 1
and
  repositories.exported = 0
it then sets this column to 1.

In the output database, it adds new contentfiles.
"""
import multiprocessing
import sys
import time

import hashlib
import pathlib
import subprocess
import threading
import typing
from concurrent import futures

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.preprocessors import secrets
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import humanize
from labm8 import pbutil

FLAGS = app.FLAGS
app.DEFINE_database(
    'input', contentfiles.ContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/export.db',
    'URL of the database of exported Java methods.')
app.DEFINE_database(
    'output', preprocessed.PreprocessedContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db',
    'URL of the database to add preprocessed files to.')
app.DEFINE_boolean('multithreaded_preprocess', True,
                   'Use multiple threads during preprocessing.')
app.DEFINE_integer('preprocess_worker_threads', multiprocessing.cpu_count(),
                   "The number of preprocessor threads.")
app.DEFINE_boolean(
    'reverse_order', False,
    'If set, pre-process repositories in a reverse order. Use '
    'this flag to have two instances of this process running '
    'concurrently - one in-order, the other in reverse-order.')

JAVA_PREPROCESSOR = bazelutil.DataPath(
    'phd/deeplearning/clgen/preprocessors/JavaPreprocessor')


def PreprocessStringList(
    srcs: typing.List[str]) -> internal_pb2.PreprocessorWorkerJobOutcomes:
  input_message = scrape_repos_pb2.ListOfStrings(string=srcs)
  output_message = internal_pb2.PreprocessorWorkerJobOutcomes()
  try:
    pbutil.RunProcessMessageToProto([JAVA_PREPROCESSOR],
                                    input_message,
                                    output_message,
                                    timeout_seconds=3600)
  except subprocess.CalledProcessError:
    # In case of preprocessor failure, dump the proto that it was working on.
    path = pathlib.Path('/tmp/preprocess_java_corpus_failed_job.pbtxt')
    pbutil.ToFile(input_message, path)
    app.FatalWithoutStackTrace(
        f'JavaPreprocessor failed processing message written to {path}')
  return output_message


def PreprocessContentfiles(
    texts: typing.List[str]
) -> typing.List[preprocessed.PreprocessedContentFile]:
  start_time = time.time()
  output_message = PreprocessStringList(texts)
  wall_time_ms = int((time.time() - start_time) * 1000)

  assert (len(texts) == len(output_message.outcome) == len(
      output_message.preprocess_time_ms))

  pp_cfs = [
      preprocessed.PreprocessedContentFile(
          input_relpath=f"{cf.clone_from_url}:{cf.relpath}:{cf.artifact_index}",
          input_sha256=cf.sha256,
          input_charcount=cf.charcount,
          input_linecount=cf.linecount,
          sha256=hashlib.sha256(outcome.contents.encode('utf-8')).hexdigest(),
          charcount=len(outcome.contents),
          linecount=len(outcome.contents.split('\n')),
          text=outcome.contents,
          preprocessing_succeeded=(
              outcome.status == internal_pb2.PreprocessorWorkerJobOutcome.OK),
          preprocess_time_ms=preprocess_time_ms,
          wall_time_ms=wall_time_ms,
      ) for cf, outcome, preprocess_time_ms in zip(
          cfs, output_message.outcome, output_message.preprocess_time_ms)
  ]

  # Scan for secrets.
  for pp_cf in pp_cfs:
    if pp_cf.preprocessing_succeeded:
      try:
        secrets.ScanForSecrets(pp_cf.text)
      except secrets.TextContainsSecret as e:
        pp_cf.preprocessing_succeeded = False
        pp_cf.text = f"Text contains secrets: {e}"

  return pp_cfs


class Preprocessor(threading.Thread):

  def __init__(self, input_db: contentfiles.ContentFiles,
               output_db: preprocessed.PreprocessedContentFiles):
    super(Preprocessor, self).__init__()
    self.input_db = input_db
    self.output_db = output_db
    self.max_workers = FLAGS.preprocess_worker_threads
    # Default to error, set to 0 upon completion.
    self.returncode = 1

  def GetABatchOfRepos(self, batch_size: int) -> bool:
    """Get a batch of repos that haven't yet been exported."""
    with self.input_db.Session() as input_session:
      query = input_session.query(
          contentfiles.GitHubRepository.clone_from_url) \
        .filter(contentfiles.GitHubRepository.active == True) \
        .filter(contentfiles.GitHubRepository.exported == False)

      if FLAGS.reverse_order:
        query = query.order_by(contentfiles.GitHubRepository.date_scraped)
      else:
        query = query.order_by(
            contentfiles.GitHubRepository.date_scraped.desc())

      query = query.limit(batch_size)

      clone_from_urls = [x[0] for x in query]

    return clone_from_urls

  def ProcessABatchOfRepos(self, batch_size: int) -> bool:
    """Process a batch of repos. Return True if one-or-more repos processed."""
    clone_from_urls = self.GetABatchOfRepos(batch_size)

    # Check if there are any repos left to export.
    if not len(clone_from_urls):
      return False

    with self.input_db.Session() as input_session:
      to_preprocess = input_session.query(contentfiles.ContentFile) \
        .filter(contentfiles.ContentFile.clone_from_url.in_(clone_from_urls))

    app.Log(1, "Preprocessing Java methods from a batch of %s files",
            humanize.Commas(len(to_preprocess)))

    preprocessed_contentfiles = PreprocessContentfiles(to_preprocess)

    with self.output_db.Session(commit=True) as output_session:
      output_session.add_all(preprocessed_contentfiles)

    return True

  def run(self):
    """Preprocess the contents of a database."""

    if FLAGS.multithreaded_preprocess:
      with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        f = lambda x: ProcessABatchOfRepos(batch_size=32)
        while executor.map(f, iter(int, 1)):
          pass
    else:
      while self.ProcessABatchOfRepos(batch_size=32):
        pass

    self.returncode = 0
    app.Log(1, "Done!")


def main():
  """Main entry point."""
  start_time = time.time()
  input_db = FLAGS.input()
  output_db = FLAGS.output()
  exporter = Preprocessor(input_db, output_db)
  exporter.start()

  while True:
    runtime = time.time() - start_time
    with input_db.Session() as s:
      all_repo_count = s.query(contentfiles.GitHubRepository) \
        .filter(contentfiles.GitHubRepository.active == True).count()
      processed_repo_count = s.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.exported == True).count()
    with output_db.Session() as s:
      preprocessed_file_count = s.query(
          preprocessed.PreprocessedContentFile).count()
    sys.stdout.write(
        f"\rRuntime: {humanize.Duration(runtime)}. "
        f"Processed repos: {humanize.Commas(processed_repo_count)} "
        f"of {humanize.Commas(all_repo_count)} "
        f"({processed_repo_count / all_repo_count:.2%}), "
        f"preprocessed methods: {humanize.Commas(preprocessed_file_count)}"
        "    ")
    sys.stdout.flush()

    if not exporter.is_alive():
      break
    time.sleep(1)
  exporter.join()

  sys.stdout.flush()
  sys.stderr.flush()
  print('Done!')
  sys.exit(exporter.returncode)


if __name__ == '__main__':
  app.Run(main)
