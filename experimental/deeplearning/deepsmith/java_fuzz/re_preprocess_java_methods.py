"""Re-run Java methods pre-processing.

This is a debugging script for checking that the JavaPreprocessor behaves
as expected. We expect the contents of the re-preprocessed db to match the oroginal
"""
import hashlib
import pathlib
import sys
import threading
import time
import typing
from concurrent import futures

import sqlalchemy.orm.exc

from datasets.github.scrape_repos import contentfiles
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import internal_pb2
from experimental.deeplearning.deepsmith.java_fuzz import preprocess_java_corpus
from labm8.py import app
from labm8.py import fs
from labm8.py import humanize

FLAGS = app.FLAGS
app.DEFINE_database(
    'input_pp', preprocessed.PreprocessedContentFiles,
    'sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db',
    'URL of the database of exported Java methods.')
app.DEFINE_output_path(
    'outdir',
    '/tmp/phd/experimental/deeplearning/deepsmith/java_fuzz/repreprocess_errors',
    'Directory to write re-preprocess failures to.',
    is_dir=True)
app.DEFINE_integer('preprocess_worker_chunk_size', 128,
                   'The number of methods to batch to the preprocessors.')


def GetOriginalContentFile(
    input_session,
    pp_cf: preprocessed.PreprocessedContentFile) -> contentfiles.ContentFile:
  components = pp_cf.input_relpath.split(':')
  clone_from_url = ':'.join(components[:-2])
  relpath = components[-2]
  artifact_index = int(components[-1])
  try:
    return input_session.query(contentfiles.ContentFile)\
      .filter(contentfiles.ContentFile.clone_from_url == clone_from_url) \
      .filter(contentfiles.ContentFile.relpath == relpath) \
      .filter(contentfiles.ContentFile.artifact_index == artifact_index).one()
  except sqlalchemy.orm.exc.NoResultFound:
    return None


def PreprocessList(input_session,
                   cfs: typing.List[preprocessed.PreprocessedContentFile],
                   outdir: pathlib.Path):
  strings = [cf.text for cf in cfs]
  output_message = preprocess_java_corpus.PreprocessStringList(strings)

  assert (len(strings) == len(output_message.outcome))
  for pp_cf, outcome in zip(cfs, output_message.outcome):
    cf = GetOriginalContentFile(input_session, pp_cf)
    if not cf:
      continue
    contents = f"""\
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ORIGINAL

{cf.text.rstrip()}

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> RE-WRITTEN

{pp_cf.text.rstrip()}
""".encode('utf-8')
    if outcome.status != internal_pb2.PreprocessorWorkerJobOutcome.OK:
      fs.Write(outdir / 'fail' / f'{pp_cf.sha256}.txt', contents)
    elif hashlib.sha256(
        outcome.contents.encode('utf-8')).hexdigest() != pp_cf.sha256:
      fs.Write(outdir / 'unstable' / f'{pp_cf.sha256}.txt', contents)
    else:
      fs.Write(outdir / 'pass' / f'{pp_cf.sha256}.txt', contents)


def ProcessBatch(input_db: contentfiles.ContentFiles,
                 pp_db: preprocessed.PreprocessedContentFile,
                 outdir: pathlib.Path, ids: typing.List[int]):
  with pp_db.Session(commit=True) as pp_session:
    with input_db.Session() as input_session:
      to_preprocess = pp_session.query(preprocessed.PreprocessedContentFile) \
        .filter(preprocessed.PreprocessedContentFile.id.in_(ids))
      PreprocessList(input_session, to_preprocess, outdir)


def Chunk(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


class RePreprocessor(threading.Thread):

  def __init__(self, input_db: contentfiles.ContentFiles,
               pp_db: preprocessed.PreprocessedContentFiles,
               outdir: pathlib.Path):
    super(RePreprocessor, self).__init__()
    self.input_db = input_db
    self.pp_db = pp_db
    self.outdir = outdir

  def run(self):
    """Preprocess the contents of a database."""
    with self.pp_db.Session() as pp_session:
      to_preprocess = pp_session.query(
          preprocessed.PreprocessedContentFile.id) \
        .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True)
      ids_to_preprocess = [x[0] for x in to_preprocess]

    max_workers = FLAGS.preprocess_worker_threads
    app.Log(1, "Preprocessing %s Java methods in %s worker threads",
            humanize.Commas(len(ids_to_preprocess)), max_workers)
    if FLAGS.multithreaded_preprocess:
      with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        f = lambda x: ProcessBatch(self.input_db, self.pp_db, self.outdir, x)
        for _ in executor.map(
            f, Chunk(ids_to_preprocess, FLAGS.preprocess_worker_chunk_size)):
          pass
    else:
      for id_ in ids_to_preprocess:
        ProcessBatch(self.input_db, self.pp_db, self.outdir, [id_])


def GetPreprocessedCount(outdir: pathlib.Path) -> int:
  return (len(list(
      (outdir / 'pass').iterdir())) + len(list((outdir / 'fail').iterdir())) +
          len(list((outdir / 'unstable').iterdir())))


def Repreprocess(input_db, pp_db, outdir: pathlib.Path):
  (outdir / 'pass').mkdir(parents=True, exist_ok=True)
  (outdir / 'fail').mkdir(parents=True, exist_ok=True)
  (outdir / 'unstable').mkdir(parents=True, exist_ok=True)

  start_time = time.time()
  thread = RePreprocessor(input_db, pp_db, outdir)
  thread.start()

  with pp_db.Session() as s:
    cf_count = s.query(preprocessed.PreprocessedContentFile) \
      .filter(preprocessed.PreprocessedContentFile.preprocessing_succeeded == True) \
      .count()

  while True:
    runtime = time.time() - start_time
    exported_count = GetPreprocessedCount(outdir)
    sys.stdout.write(
        f"\rRuntime: {humanize.Duration(runtime)}. "
        f"Exported contentfiles: {humanize.Commas(exported_count)} "
        f"of {humanize.Commas(cf_count)} "
        f"({exported_count / max(cf_count, 1):.2%})    ")
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
  Repreprocess(FLAGS.input(), FLAGS.input_pp(), FLAGS.outdir)


if __name__ == '__main__':
  app.Run(main)
