"""Preprocess an exported database of Java methods."""
import multiprocessing
import sys
import time

import hashlib
import threading
import typing
from concurrent import futures
from datasets.github.scrape_repos import contentfiles
from datasets.github.scrape_repos.proto import scrape_repos_pb2
from deeplearning.clgen.proto import internal_pb2

from datasets.github.scrape_repos.preprocessors import secrets
from deeplearning.clgen.corpuses import preprocessed
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

JAVA_PREPROCESSOR = bazelutil.DataPath(
    'phd/experimental/deeplearning/deepsmith/java_fuzz/JavaPreprocessor')


def PreprocessStringList(
    srcs: typing.List[str]) -> internal_pb2.PreprocessorWorkerJobOutcomes:
  input_message = scrape_repos_pb2.ListOfStrings(string=srcs)
  output_message = internal_pb2.PreprocessorWorkerJobOutcomes()
  pbutil.RunProcessMessageToProto([JAVA_PREPROCESSOR],
                                  input_message,
                                  output_message,
                                  timeout_seconds=3600)
  return output_message


def PreprocessContentFiles(
    cfs: typing.List[contentfiles.ContentFile]
) -> typing.List[preprocessed.PreprocessedContentFile]:
  start_time = time.time()
  output_message = PreprocessStringList([cf.text for cf in cfs])
  wall_time_ms = int((time.time() - start_time) * 1000)

  assert (len(cfs) == len(output_message.outcome) == len(
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


def ProcessRepo(input_db: contentfiles.ContentFiles,
                output_db: preprocessed.PreprocessedContentFiles,
                clone_from_url: str):
  with input_db.Session(commit=True) as input_session:
    with output_db.Session(commit=True) as output_session:
      contentfiles_to_process = input_session.query(contentfiles.ContentFile) \
        .filter(contentfiles.ContentFile.clone_from_url == clone_from_url)
      app.Log(2, 'Processing %s content files from %s',
              humanize.Commas(contentfiles_to_process.count()), clone_from_url)

      processed = PreprocessContentFiles(contentfiles_to_process.all())
      output_session.add_all(processed)

    input_session.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.clone_from_url == clone_from_url) \
        .update({'exported': True})


class Preprocessor(threading.Thread):

  def __init__(self, input_db: contentfiles.ContentFiles,
               output_db: preprocessed.PreprocessedContentFiles):
    super(Preprocessor, self).__init__()
    self.input_db = input_db
    self.output_db = output_db

  def run(self):
    """Preprocess the contents of a database."""
    with self.input_db.Session() as input_session:
      repos_to_export = input_session.query(
          contentfiles.GitHubRepository.clone_from_url) \
        .filter(contentfiles.GitHubRepository.active == True) \
        .filter(contentfiles.GitHubRepository.exported == False)
      clone_from_urls = [x[0] for x in repos_to_export]

    max_workers = FLAGS.preprocess_worker_threads
    app.Log(1, "Preprocessing Java methods from %s repos in %s worker threads",
            humanize.Commas(len(clone_from_urls)), max_workers)
    if FLAGS.multithreaded_preprocess:
      with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        f = lambda x: ProcessRepo(self.input_db, self.output_db, x)
        for _ in executor.map(f, clone_from_urls):
          pass
    else:
      for clone_from_url in clone_from_urls:
        ProcessRepo(self.input_db, self.output_db, clone_from_url)


def main():
  """Main entry point."""
  start_time = time.time()
  exporter = Preprocessor(FLAGS.input(), FLAGS.output())
  exporter.start()

  with FLAGS.input().Session() as s:
    repo_count = s.query(contentfiles.GitHubRepository) \
      .filter(contentfiles.GitHubRepository.active == True).count()

  while True:
    runtime = time.time() - start_time
    with FLAGS.input().Session() as s:
      exported_repo_count = s.query(contentfiles.GitHubRepository)\
        .filter(contentfiles.GitHubRepository.exported == True).count()
    with FLAGS.output().Session() as s:
      exported_contentfile_count = s.query(
          preprocessed.PreprocessedContentFile).count()
    sys.stdout.write(
        f"\rRuntime: {humanize.Duration(runtime)}. "
        f"Processed repos: {humanize.Commas(exported_repo_count)} "
        f"of {humanize.Commas(repo_count)} "
        f"({exported_repo_count / repo_count:.2%}), "
        f"preprocessed methods: {humanize.Commas(exported_contentfile_count)}"
        "    ")
    sys.stdout.flush()

    if not exporter.is_alive():
      break
    time.sleep(1)
  exporter.join()

  sys.stdout.flush()
  sys.stderr.flush()
  print('Done!')


if __name__ == '__main__':
  app.Run(main)
