"""Preprocess an exported database of Java methods.

From the input database, it exports contentfiles from repositories where
  repositories.active = 1
and
  repositories.exported = 0
it then sets this column to 1.

In the output database, it adds new contentfiles.
"""
import hashlib
import multiprocessing
import pathlib
import subprocess
import sys
import threading
import time
import typing
from concurrent import futures

import sqlalchemy as sql

from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.preprocessors import secrets
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
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
                   'The number of preprocessor threads.')
app.DEFINE_boolean(
    'reverse_order', False,
    'If set, pre-process repositories in a reverse order. Use '
    'this flag to have two instances of this process running '
    'concurrently - one in-order, the other in reverse-order.')
app.DEFINE_integer('preprocessor_input_query_size', 1024,
                   'The number of repo URLs to fetch per query.')
app.DEFINE_integer(
    'preprocessor_worker_repo_count', 16,
    'The number of repos to process sequentially in each worker.')

JAVA_PREPROCESSOR = bazelutil.DataPath(
    'phd/deeplearning/clgen/preprocessors/JavaPreprocessor')


def BatchedReposIterator(input_db, batch_size):
  with input_db.Session() as input_session:
    if FLAGS.reverse_order:
      last_date = input_session.query(
          sql.func.max(contentfiles.GitHubRepository.date_scraped)).one()
    else:
      last_date = input_session.query(
          sql.func.min(contentfiles.GitHubRepository.date_scraped)).one()

  while True:
    with input_db.Session() as input_session:
      query = input_session.query(
          contentfiles.GitHubRepository.clone_from_url,
          contentfiles.GitHubRepository.date_scraped) \
        .filter(contentfiles.GitHubRepository.active == True) \
        .filter(contentfiles.GitHubRepository.exported == False)

      if FLAGS.reverse_order:
        query = query.filter(contentfiles.GitHubRepository.date_scraped <= last_date)\
            .order_by(contentfiles.GitHubRepository.date_scraped.desc())
      else:
        query = query.filter(contentfiles.GitHubRepository.date_scraped >= last_date)\
            .order_by(contentfiles.GitHubRepository.date_scraped)

      query = query.limit(batch_size)

      # Check that there is something in the query else min() will raise an
      # error.
      if query.first():
        if FLAGS.reverse_order:
          last_date = min([x[1] for x in query])
        else:
          last_date = max([x[1] for x in query])

      clone_from_urls = [x[0] for x in query]

    yield clone_from_urls


def Chunkify(iterator, chunk_size):
  for items in iterator:
    for i in range(0, len(items), chunk_size):
      yield items[i:i + chunk_size]


def MergePreprocessorWorkerJobOutcomes(
    left: internal_pb2.PreprocessorWorkerJobOutcomes,
    right: internal_pb2.PreprocessorWorkerJobOutcomes):
  left.outcome.extend(right.outcome)
  left.preprocess_time_ms.extend(right.preprocess_time_ms)
  return left


def PreprocessStringList(
    srcs: typing.List[str]) -> internal_pb2.PreprocessorWorkerJobOutcomes:
  input_message = scrape_repos_pb2.ListOfStrings(string=srcs)
  output_message = internal_pb2.PreprocessorWorkerJobOutcomes()
  try:
    pbutil.RunProcessMessageToProto([str(JAVA_PREPROCESSOR)],
                                    input_message,
                                    output_message,
                                    timeout_seconds=len(srcs) * 100)
  except subprocess.CalledProcessError as e:
    # When pre-processing fails, we use a dividie-and-conquer strategy to
    # isolate the one-or-more methods which cause pre-processing to fail.
    if len(srcs) > 1:
      # Divide and conquer.
      app.Log(1, 'JavaPreprocessor failed processing %d srcs, dividing',
              len(srcs))
      mid = int(len(srcs) / 2)
      left, right = srcs[:mid], srcs[mid:]

      left, right = PreprocessStringList(left), PreprocessStringList(right)
      output_message = MergePreprocessorWorkerJobOutcomes(left, right)
    else:
      # Base case: Log the error, dump the input, and move on.
      src = srcs[0]
      path = pathlib.Path('/tmp/preprocess_java_corpus_failed_job.java')
      fs.Write(path, src.encode('utf-8'))
      app.Error('JavaPreprocessor failed processing message written to %s',
                path)
      outcome = output_message.outcome.add()
      outcome.status = internal_pb2.PreprocessorWorkerJobOutcome.FAIL
      outcome.contents = 'internal error'
      output_message.preprocess_time_ms.append(0)
  return output_message


def PreprocessContentfiles(
    cfs: typing.List[contentfiles.ContentFile]
) -> typing.List[preprocessed.PreprocessedContentFile]:
  start_time = time.time()
  output_message = PreprocessStringList([cf.text for cf in cfs])
  wall_time_ms = int((time.time() - start_time) * 1000)

  assert (len(cfs) == len(output_message.outcome) == len(
      output_message.preprocess_time_ms))

  pp_cfs = [
      preprocessed.PreprocessedContentFile(
          input_relpath=f'{cf.clone_from_url}:{cf.relpath}:{cf.artifact_index}',
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
        pp_cf.text = f'Text contains secrets: {e}'

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

  def ProcessABatchOfRepos(self, clone_from_urls: typing.List[str]) -> bool:
    """Process a batch of repos. Return True if one-or-more repos processed."""
    # Check if there are any repos left to export.
    if not len(clone_from_urls):
      return False

    start_time = time.time()

    with self.input_db.Session() as input_session:
      # Select all of the contentfiles from the repository which are active.
      to_preprocess = input_session.query(contentfiles.ContentFile) \
        .filter(contentfiles.ContentFile.clone_from_url.in_(clone_from_urls)) \
        .filter(contentfiles.ContentFile.active == True).all()

    preprocessed_contentfiles = PreprocessContentfiles(to_preprocess)

    with self.output_db.Session(commit=True) as output_session:
      with self.input_db.Session(commit=True) as input_session:
        input_session.query(contentfiles.GitHubRepository)\
            .filter(contentfiles.GitHubRepository.clone_from_url.in_(clone_from_urls)) \
            .update({'exported': True}, synchronize_session=False)
        output_session.add_all(preprocessed_contentfiles)

    duration = time.time() - start_time
    app.Log(1, 'Preprocessed %s Java methods at a rate of %d ms per method',
            humanize.Commas(len(preprocessed_contentfiles)),
            (duration / len(preprocessed_contentfiles)) * 1000)

    return True

  def run(self):
    """Preprocess the contents of a database."""
    repos = Chunkify(
        BatchedReposIterator(self.input_db,
                             FLAGS.preprocessor_input_query_size),
        FLAGS.preprocessor_worker_repo_count)

    if FLAGS.multithreaded_preprocess:
      with futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        while executor.map(self.ProcessABatchOfRepos, repos):
          pass
    else:
      for clone_from_urls in repos:
        self.ProcessABatchOfRepos(clone_from_urls)

    self.returncode = 0
    app.Log(1, 'Done!')


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
    sys.stdout.write(
        f'\rRuntime: {humanize.Duration(runtime)}. '
        f'Processed repos: {humanize.Commas(processed_repo_count)} '
        f'of {humanize.Commas(all_repo_count)} '
        f'({processed_repo_count / all_repo_count:.2%})'
        '    ')
    sys.stdout.flush()

    if processed_repo_count == all_repo_count or not exporter.is_alive():
      break
    time.sleep(1)
  exporter.join()

  sys.stdout.flush()
  sys.stderr.flush()
  print('Done!')
  sys.exit(exporter.returncode)


if __name__ == '__main__':
  app.Run(main)
