"""Preprocess OpenCL files for machine learning."""
import contextlib
import math
import multiprocessing
import pathlib
import queue
import random
import typing
from io import open
from multiprocessing import Pool, cpu_count
from threading import Thread
from typing import List

import progressbar
from absl import logging

from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen import languages
from deeplearning.clgen.preprocessors import common
from deeplearning.clgen.preprocessors import cxx
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import internal_pb2


def preprocess(src: str,
               preprocessors: typing.List[corpus_pb2.Preprocessor]) -> str:
  """
  Preprocess a file. There are three possible outcomes:

  1. Good. Code is preprocessed and ready to be put into a training set.
  2. Bad. Code can't be preprocessed (i.e. it's "bad" code).
  3. Ugly. Code can be preprocessed but isn't useful for training
     (e.g. it's an empty file).

  Args:
    src: The source code as a string.
    preprocessors: The list of preprocessors to run.
    id: An identifying name for the source code (used in exception messages).

  Returns
  -------
  str
      Preprocessed source code as a string.

  Raises
  ------
  BadCodeException
      If code is bad (see above).
  UglyCodeException
      If code is ugly (see above).
  errors.InternalException
      In case of some other error.
  """
  for preprocessor_proto in preprocessors:
    preprocessor = {"common:MinimumLineCount": common.MinimumLineCount,
                    "common:RemoveDuplicateEmptyLines":
                      common.RemoveDuplicateEmptyLines,
                    "cxx:ClangFormat": cxx.ClangFormat,
                    "cxx:ClangPreprocess": cxx.ClangPreprocess,
                    "cxx:StripComments": cxx.StripComments,
                    "opencl:ClangFormat": cxx.ClangFormat,
                    "opencl:ClangPreprocess": opencl.ClangPreprocess,
                    "opencl:MinimumLineCount": opencl.MinimumLineCount,
                    "opencl:NormalizeIdentifiers": opencl.NormalizeIdentifiers,
                    "opencl:SanitizeKernelPrototype":
                      opencl.SanitizeKernelPrototype}.get(
      preprocessor_proto.name)
    src = preprocessor(src, **preprocessor_proto.opts)
  return src


def preprocess_file(path: str, inplace: bool = False,
                    **preprocess_opts) -> None:
  """
  Preprocess a file.

  Prints output to stdout by default. If preprocessing fails, this function
  exits.

  Parameters
  ----------
  path : str
      String path to file.
  inplace : bool, optional
      If True, overwrite input file.
  """
  with open(path) as infile:
    contents = infile.read()
  try:
    out = preprocess(contents, **preprocess_opts)
    if inplace:
      with open(path, 'w') as outfile:
        outfile.write(out)
    else:
      print(out)
  except errors.BadCodeException as e:
    logging.fatal(e, ret=1)
  except errors.UglyCodeException as e:
    logging.fatal(e, ret=2)


def _preprocess_inplace_worker(path: str) -> None:
  """worker function for preprocess_inplace()"""
  logging.info('preprocess', path)
  preprocess_file(path, inplace=True)


@contextlib.contextmanager
def terminating(thing):
  """
  Context manager to terminate object at end of scope.
  """
  try:
    yield thing
  finally:
    thing.terminate()


def preprocess_inplace(paths: List[str], max_num_workers: int = cpu_count(),
                       max_attempts: int = 100, attempt: int = 1) -> None:
  """
  Preprocess a list of files in place.

  Parameters
  ----------
  paths : List[str]
      List of paths.
  max_num_workers : int, optional
      Number of processes to spawn.
  max_attempts : int, optional
      In case of an OSError or TimeoutError, this number of attempts will be
      made.
  """
  if attempt > max_attempts:
    raise errors.InternalError(
      f"Failed to process files after {max_attempts} attempts")
  elif attempt > 1:
    logging.warning("preprocess attempt #.", attempt)

  num_workers = min(len(paths), max_num_workers)

  try:
    logging.info('spawned', num_workers, 'worker threads to process',
                 len(paths), 'files ...')
    with terminating(Pool(num_workers)) as pool:
      pool.map(_preprocess_inplace_worker, paths)
  except (OSError, TimeoutError) as e:
    logging.error(e)

    # Try again with fewer threads.
    # See: https://github.com/ChrisCummins/clgen/issues/64
    max_num_workers = max(int(max_num_workers / 2), 1)
    preprocess_inplace(paths, max_num_workers=max_num_workers,
                       attempt=attempt + 1, max_attempts=max_attempts)


class PreprocessWorker(Thread):
  """A preprocessor worker thread."""

  def __init__(self, jobs: List[internal_pb2.PreprocessorWorkerJob],
               output_queue: queue.Queue):
    """Instantiate a PreprocessWorker.

    Args:
      jobs: A list of jobs to execute.
      output_queue: The output queue to write results to.
    """
    super(PreprocessWorker, self).__init__()
    self.jobs = jobs
    self.queue = output_queue

  def run(self):
    for job in self.jobs:
      result = internal_pb2.PreprocessorWorkerJobOutcome()
      result.contentfile_id = job.contentfile_id
      result.status = internal_pb2.PreprocessorWorkerJobOutcome.OK
      try:
        # TODO(cec): Implement preprocessors.
        preprocess = job.preprocessors
        result.contents = preprocess(job.src)
      except errors.BadCodeException as e:
        result.status = internal_pb2.PreprocessorWorkerJobOutcome.BAD
        result.contents = str(e)
      except errors.UglyCodeException as e:
        result.status = internal_pb2.PreprocessorWorkerJobOutcome.UGLY
        result.contents = str(e)
      self.queue.put(result)


def _DoPreprocessDatabase(db_path: pathlib.Path, language: languages.Language,
                          preprocessors: typing.List[corpus_pb2.Preprocessor],
                          attempt_num: int, max_attempts: int,
                          max_num_threads: int) -> None:
  """The private function to preprocess a database.

  Args:
    db_path: The path to the contentfiles database.
    language: The language of the contentfiles database.
    preprocessors: The list of preprocessors to run.
    attempt_num: The current attempt number.
    max_attempts: The maxmium number of attempts to try.
    max_num_threads: The maximum number of threads to spawn.
  """
  if attempt_num > max_attempts:
    logging.error('Failed to complete preprocessing after %d attempts. '
                  'Stopping now', max_attempts)

  # Determine the set of contentfiles which need preprocessing.
  contentfiles = set(dbutil.kernel_ids(str(db_path), "ContentFiles"))
  preprocessedfiles = set(dbutil.kernel_ids(str(db_path), "PreprocessedFiles"))
  todo = contentfiles - preprocessedfiles
  if not todo:
    logging.warning("Database preprocess requested, but there's nothing to do")
    return

  logging.info('%d of %d (%.1f%%) samples need preprocessing', len(todo),
               len(contentfiles), (len(todo) / len(contentfiles)) * 100)
  # Determine if we need to inline kernels when creating jobs.
  if dbutil.HasContentMetaTable(db_path):
    get_kernel = lambda kid: dbutil.get_inlined_kernel(str(db_path), kid,
                                                       lang=language)
  else:
    get_kernel = lambda kid: dbutil.get_kernel(str(db_path), kid,
                                               table="ContentFiles")
  # Create job protos and distribute the jobs.
  jobs = [
    internal_pb2.PreprocessorWorkerJob(contentfiles_id=kid, src=get_kernel(kid),
                                       preprocessors=preprocessors) for kid in
    todo]
  random.shuffle(jobs)
  num_threads = int(math.ceil(len(todo) / max_num_threads))
  output_queue = queue.Queue(maxsize=128)
  logging.debug('assigning %d jobs to %s threads', len(todo), num_threads)

  num_preprocessed = 0
  try:
    # Our worker threads. These busy little bees will do the heavy lifting
    # of preprocessing the contentfiles and pushing their results onto the
    # queue.
    producers = [PreprocessWorker(jobs[i:i + num_threads], output_queue) for i
                 in range(0, len(todo), num_threads)]
    # Fly, my pretties, fly!
    for producer in producers:
      producer.start()
    # Consume the results from the worker threads in the main thread.
    db = dbutil.connect(db_path)
    for _ in progressbar.ProgressBar()(range(len(todo))):
      num_preprocessed += 1
      # Block until another result comes in.
      result: internal_pb2.PreprocessorWorkerJobOutcome = output_queue.get(
        timeout=120)
      status = internal_pb2.PreprocessorWorkerJobOutcome.Status.Value(
        result.status)
      # Insert result into database.
      c = db.cursor()
      c.execute(
        "INSERT INTO PreprocessedFiles (id,status,contents) VALUES(?,?,?)",
        (result.contenfile_id, status, result.contents))
      c.close()
      db.commit()
    db.close()
    for producer in producers:
      producer.join()

  except (OSError, TimeoutError, output_queue.Empty) as e:
    logging.error(e)
    if attempt_num >= 3 and not num_preprocessed:
      logging.warning('No progress has been made since previous attempt. '
                      "I'm not going to try another attempt.")
      return
    # Try again with fewer threads.
    # See: https://github.com/ChrisCummins/clgen/issues/64
    new_max_threads = max(int(math.ceil(max_num_threads / 2)), 1)
    _DoPreprocessDatabase(db_path, language, preprocessors, attempt_num + 1,
                          max_attempts, new_max_threads)


def PreprocessDatabase(db_path: pathlib.Path, language: languages.Language,
                       preprocessors: typing.List[
                         corpus_pb2.Preprocessor]) -> bool:
  """Preprocess a contentfiles database.

  This function tries to do as little as possible. If no changes to the database
  have been made since the last time this function was called on it, then this
  function does nothing.

  Args:
    db_path: The path of the contentfiles database.
    language: The language of the contentfiles database.
    preprocessors: The list of preprocessors to run.

  Returns:
    True if the database was modified, else False.
  """
  db = dbutil.connect(db_path)
  is_modified = dbutil.is_modified(db)
  max_retries = 10
  if is_modified:
    _DoPreprocessDatabase(db_path, language, preprocessors, 1, max_retries,
                          multiprocessing.cpu_count())
    dbutil.set_modified_status(db, is_modified)
    return True
  else:
    return False
