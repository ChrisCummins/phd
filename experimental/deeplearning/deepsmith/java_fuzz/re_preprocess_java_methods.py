"""Re-run Java methods pre-processing.

This is a debugging script for checking that the JavaPreprocessor behaves
as expected. We expect the contents of the re-preprocessed db to match the oroginal 
"""
import sys
import time

import hashlib
import threading
import typing
from concurrent import futures
from deeplearning.clgen.proto import internal_pb2

from deeplearning.clgen.corpuses import preprocessed
from experimental.deeplearning.deepsmith.java_fuzz import preprocess_java_corpus
from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS
app.DEFINE_database(
    'input', preprocessed.PreprocessedContentFile,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db',
    'URL of the database of exported Java methods.')
app.DEFINE_database(
    'output', preprocessed.PreprocessedContentFiles,
    'sqlite:////tmp/phd/experimental/deeplearning/deepsmith/java_fuzz/repreprocessed.db',
    'URL of the database to add preprocessed files to.')
app.DEFINE_boolean('multithreaded_preprocess', True,
                   'Use multiple threads during preprocessing.')
app.DEFINE_integer('preprocess_worker_chunk_size', 128,
                   'The number of methods to batch to the preprocessors.')


def PreprocessList(cfs: typing.List[preprocessed.PreprocessedContentFile]
                  ) -> typing.List[preprocessed.PreprocessedContentFile]:
  output_message = preprocess_java_corpus.PreprocessStringList(
      [cf.text for cf in cfs])

  assert (len(cfs) == len(output_message.outcome))
  pp_cfs = [
      preprocessed.PreprocessedContentFile(
          input_relpath=cf.input_relpath,
          input_sha256=cf.sha256,
          input_charcount=cf.charcount,
          input_linecount=cf.linecount,
          sha256=hashlib.sha256(outcome.contents.encode('utf-8')).hexdigest(),
          charcount=len(outcome.contents),
          linecount=len(outcome.contents.split('\n')),
          text=outcome.contents,
          preprocessing_succeeded=(
              outcome.status == internal_pb2.PreprocessorWorkerJobOutcome.OK),
          preprocess_time_ms=0,
          wall_time_ms=0,
      ) for cf, outcome in zip(cfs, output_message.outcome)
  ]

  return pp_cfs


def ProcessBatch(input_db: preprocessed.PreprocessedContentFile,
                 output_db: preprocessed.PreprocessedContentFiles,
                 ids: typing.List[int]):
  with input_db.Session(commit=True) as input_session:
    with output_db.Session(commit=True) as output_session:
      to_preprocess = input_session.query(preprocessed.PreprocessedContentFile) \
        .filter(preprocessed.PreprocessedContentFile.id.in_(ids))
      processed = PreprocessList(to_preprocess)
      output_session.add_all(processed)


def Chunk(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


class RePreprocessor(threading.Thread):

  def __init__(self, input_db: preprocessed.PreprocessedContentFiles,
               output_db: preprocessed.PreprocessedContentFiles):
    super(RePreprocessor, self).__init__()
    self.input_db = input_db
    self.output_db = output_db

  def run(self):
    """Preprocess the contents of a database."""
    with self.input_db.Session() as input_session:
      with self.output_db.Session() as output_session:
        already_preprocessed = output_session.query(
            preprocessed.PreprocessedContentFile.id)

        to_preprocess = input_session.query(
            preprocessed.PreprocessedContentFile.id) \
            .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True)\
            .filter(preprocessed.PreprocessedContentFile.id.in_(
                already_preprocessed))
        ids_to_preprocess = [x[0] for x in to_preprocess]

    max_workers = FLAGS.preprocess_worker_threads
    app.Log(1, "Preprocessing %s Java methods in %s worker threads",
            humanize.Commas(len(ids_to_preprocess)), max_workers)
    if FLAGS.multithreaded_preprocess:
      with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        f = lambda x: ProcessBatch(self.input_db, self.output_db, x)
        for _ in executor.map(
            f, Chunk(ids_to_preprocess, FLAGS.preprocess_worker_chunk_size)):
          pass
    else:
      for id_ in ids_to_preprocess:
        ProcessBatch(self.input_db, self.output_db, [id_])


def Repreprocess(input_db, output_db):
  start_time = time.time()
  thread = RePreprocessor(input_db, output_db)
  thread.start()

  with input_db.Session() as s:
    cf_count = s.query(preprocessed.PreprocessedContentFile) \
      .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True) \
      .count()

  while True:
    runtime = time.time() - start_time
    with output_db.Session() as s:
      exported_count = s.query(preprocessed.PreprocessedContentFile).count()
    sys.stdout.write(
        f"\rRuntime: {humanize.Duration(runtime)}. "
        f"Exported contentfiles: {humanize.Commas(exported_count)} "
        f"of {humanize.Commas(cf_count)} "
        f"({exported_count / cf_count:.2%})    ")
    sys.stdout.flush()

    if not thread.is_alive():
      break
    time.sleep(1)
  thread.join()

  sys.stdout.flush()
  sys.stderr.flush()
  print('Done!')


def main():
  """Main entry point."""
  Repreprocess(FLAGS.input(), FLAGS.output())


if __name__ == '__main__':
  app.Run(main)
