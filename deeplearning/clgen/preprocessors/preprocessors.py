"""Preprocess OpenCL files for machine learning."""
import importlib
import math
import multiprocessing
import pathlib
import queue
import random
import threading
import typing
from io import open

import progressbar
from absl import logging

from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen import languages
from deeplearning.clgen.proto import internal_pb2


# Type hint for a preprocessor function. See @clgen_preprocess for details.
PreprocessorFunction = typing.Callable[[str], str]


def clgen_preprocessor(func: PreprocessorFunction) -> PreprocessorFunction:
  """A decorator which marks a function as a CLgen preprocessor.

  A CLgen preprocessor is accessible using GetPreprocessFunction(), and is a
  function which accepts a single parameter 'text', and returns a string.
  Type hinting is used to ensure that any function wrapped with this decorator
  has the appropriate argument and return type. If the function does not, an
  InternalError is raised at the time that the module containing the function
  is imported.

  Args:
    func: The preprocessor function to decorate.

  Returns:
    The decorated preprocessor function.

  Raises:
    InternalError: If the function being wrapped does not have the signature
      'def func(text: str) -> str:'.
  """
  type_hints = typing.get_type_hints(func)
  if not type_hints == {'text': str, 'return': str}:
    raise errors.InternalError(
      f'Preprocessor {func.__name__} does not have signature '
      f'"def {func.__name__}(text: str) -> str".')
  func.__dict__['is_clgen_preprocessor'] = True
  return func


def GetPreprocessorFunction(name: str) -> PreprocessorFunction:
  """Lookup a preprocess function by name.

  A preprocessor is a function which takes a single argument 'text' of type str,
  and returns a str. The name is the fully qualified name of the python
  function which implements it, in the form <module>:<name>. For example,
  the name 'deeplearning.clgen.preprocessors.cxx:Compile' will return the
  function 'Compile' in the module 'deeplearning.clgen.preprocessors.cxx'.

  Args:
    name: The name of the preprocessor to get.

  Returns:
    The python preprocessor function.

  Raises:
    UserError: If the requested name cannot be found or is not a
      @clgen_preprocessor decorated function.
  """
  components = name.split(':')
  if len(components) != 2:
    raise errors.UserError(f'Invalid preprocessor name {name}')
  module_name, function_name = components
  try:
    module = importlib.import_module(module_name)
    function_ = getattr(module, function_name)
  except (ModuleNotFoundError, AttributeError):
    raise errors.UserError(f'Preprocessor {name} not found.')
  if not function_.__dict__.get('is_clgen_preprocessor'):
    raise errors.UserError(
      f'Preprocessor {name} not decorated with @clgen_preprocessor')
  return function_


def Preprocess(text: str, preprocessors: typing.List[str]) -> str:
  """Preprocess a text using the given preprocessor pipeline.

  If preprocessing succeeds, the preprocessed text is returned. If preprocessing
  fails (in an expected way, for example by trying to compile incorrect code),
  a BadCodeException is raised. Any other error leads to an InternalError.


  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.

  Returns:
    Preprocessed source input as a string.

  Raises:
    UserError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  preprocessor_functions = [GetPreprocessorFunction(p) for p in preprocessors]
  for preprocessor in preprocessor_functions:
    text = preprocessor(text)
  return text


def PreprocessFile(path: str, preprocessors: typing.List[str],
                   inplace: bool) -> str:
  """Preprocess a file and optionally update it.

  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.
    inplace: If True, the input file is overwritten with the preprocessed code,
      unless the preprocessing fails. If the preprocessing fails, the input
      file is left unmodified.

  Returns:
    Preprocessed source input as a string.

  Raises:
    UserError: If the requested preprocessors cannot be loaded.
    BadCodeException: If one of the preprocessors rejects the input.
    InternalException: In case of some other error.
  """
  with open(path) as infile:
    contents = infile.read()
  preprocessed = Preprocess(contents, preprocessors)
  if inplace:
    with open(path, 'w') as outfile:
      outfile.write(preprocessed)
  return preprocessed


class PreprocessWorker(threading.Thread):
  """A preprocessor worker thread."""

  def __init__(self, jobs: typing.List[internal_pb2.PreprocessorWorkerJob],
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
      try:
        result.contents = Preprocess(job.src, job.preprocessors)
        result.status = internal_pb2.PreprocessorWorkerJobOutcome.OK
      except errors.BadCodeException as e:
        result.status = internal_pb2.PreprocessorWorkerJobOutcome.FAIL
        result.contents = str(e)
      # An intentionally broad exception catch here to catch whatever's left.
      except Exception as e:
        result.status = internal_pb2.PreprocessorWorkerJobOutcome.FAIL
        result.contents = f'!!INTERNAL ERROR!! {e}'
      self.queue.put(result)


def _DoPreprocessDatabase(db_path: pathlib.Path, language: languages.Language,
                          preprocessors: typing.List[str], attempt_num: int,
                          max_attempts: int, max_num_threads: int) -> None:
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
    internal_pb2.PreprocessorWorkerJob(contentfile_id=kid, src=get_kernel(kid),
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
      status = result.status
      # Insert result into database.
      c = db.cursor()
      c.execute(
        "INSERT INTO PreprocessedFiles (id,status,contents) VALUES(?,?,?)",
        (result.contentfile_id, status, result.contents))
      c.close()
      db.commit()
    db.close()
    for producer in producers:
      producer.join()

  except (OSError, TimeoutError, queue.Empty) as e:
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
                       preprocessors: typing.List[str]) -> bool:
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
