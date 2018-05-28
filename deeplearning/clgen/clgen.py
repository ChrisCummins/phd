"""CLgen: a deep learning program generator.

The core operations of CLgen are:

  1. Preprocess a corpus of handwritten example programs.
  2. Define and train a machine learning model on the corpus.
  3. Sample the trained model to generate new programs.

This program automates the execution of all three stages of the pipeline.
The pipeline can be interrupted and resumed at any time. Results are cached
across runs. Please note that many of the steps in the pipeline are extremely
compute intensive and highly parallelized. If configured with CUDA support,
any NVIDIA GPUs will be used to improve performance where possible.

Made with \033[1;31m♥\033[0;0m by Chris Cummins <chrisc.101@gmail.com>.
https://chriscummins.cc/clgen
"""
import argparse
import cProfile
import contextlib
import os
import pathlib
import traceback
import typing
from pathlib import Path

import sys
from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import clgen_pb2
from lib.labm8 import fs
from lib.labm8 import pbutil
from lib.labm8 import prof


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'config', None,
    'Path to a clgen.Instance proto file.')
flags.DEFINE_integer(
    'min_samples', -1,
    'The minimum number of samples to make.')
flags.DEFINE_string(
    'print_cache_dir', None,
    'Print the directory of a cache and exit. Valid options are: "corpus", '
    '"model", or "sampler".')
flags.DEFINE_bool(
    'clgen_debug', False,
    'Enable a debugging mode of CLgen python runtime. When enabled, errors '
    'which may otherwise be caught lead to program crashes and stack traces.')
flags.DEFINE_bool(
    'clgen_profiling', False,
    'Enable CLgen self profiling. Profiling results be logged.')


class Instance(object):
  """A CLgen instance."""

  def __init__(self, config: clgen_pb2.Instance):
    self.config = config
    self.model = None
    self.sampler = None
    with self.Session():
      if config.HasField('model'):
        self.model = models.Model(config.model)
      if config.HasField('sampler'):
        self.sampler = samplers.Sampler(config.sampler)

  @contextlib.contextmanager
  def Session(self) -> 'Instance':
    old_working_dir = os.environ.get('CLGEN_CACHE', '')
    working_dir = ''
    if self.config.HasField('working_dir'):
      working_dir = str(pathlib.Path(
          os.path.expandvars(self.config.working_dir)).expanduser())
    os.environ['CLGEN_CACHE'] = working_dir
    yield self
    os.environ['CLGEN_CACHE'] = old_working_dir


def getself(func):
  """ decorator to pass function as first argument to function """

  def wrapper(*args, **kwargs):
    return func(func, *args, **kwargs)

  return wrapper


class ReadableFilesOrDirectories(argparse.Action):
  """
  Adapted from @mgilson http://stackoverflow.com/a/11415816
  """

  def __call__(self, parser, namespace, values, option_string=None) -> None:
    for path in values:
      if not os.path.isdir(path) and not os.path.isfile(path):
        raise argparse.ArgumentTypeError(
            f"ReadableFilesOrDirectories:{path} not found")
      if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(
            f"ReadableFilesOrDirectories:{path} is not readable")

    setattr(namespace, self.dest, [Path(path) for path in values])


def run(function_to_run: typing.Callable, *args, **kwargs) -> typing.Any:
  """
  Runs the given method as the main entrypoint to a program.

  If an exception is thrown, print error message and exit. If FLAGS.debug is
  set, the exception is not caught.

  Args:
    function_to_run: The function to run.
    *args: Arguments to be passed to the function.
    **kwargs: Arguments to be passed to the function.

  Returns:
    The return value of the function when called with the given args.
  """

  def _user_message(exception: Exception):
    logging.error(f"""\
%s (%s)

Please report bugs at <https://github.com/ChrisCummins/phd/issues>\
""", exception, type(exception).__name__)
    sys.exit(1)

  def _user_message_with_stacktrace(exception: Exception):
    # get limited stack trace
    def _msg(i, x):
      n = i + 1
      filename, lineno, fnname, _ = x
      # TODO(cec): Report filename relative to PhD root.
      filename = fs.basename(filename)
      loc = f'{filename}:{lineno}'
      return f'      #{n}  {loc: <18} {fnname}()'

    _, _, tb = sys.exc_info()
    NUM_ROWS = 5  # number of rows in traceback
    trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS + 1)[1:])
    message = "\n".join(_msg(*r) for r in enumerate(trace))
    logging.error("""\
%s (%s)

  stacktrace:
%s

Please report bugs at <https://github.com/ChrisCummins/clgen/issues>\
""", exception, type(exception).__name__, message)
    sys.exit(1)

  if FLAGS.clgen_debug:
    # verbose stack traces. see: https://pymotw.com/2/cgitb/
    import cgitb
    cgitb.enable(format='text')
    return function_to_run(*args, **kwargs)

  try:
    def runctx():
      return function_to_run(*args, **kwargs)

    if prof.is_enabled() and logging.get_verbosity() == logging.DEBUG:
      return cProfile.runctx('runctx()', None, locals(), sort='tottime')
    else:
      return runctx()
  except errors.UserError as err:
    logging.error("%s (%s)", err, type(err).__name__)
    sys.exit(1)
  except KeyboardInterrupt:
    sys.stdout.flush()
    sys.stderr.flush()
    print("\nkeyboard interrupt, terminating", file=sys.stderr)
    sys.exit(1)
  except errors.File404 as e:
    _user_message(e)
  except Exception as e:
    _user_message_with_stacktrace(e)


def DoTheRest():
  config_path = pathlib.Path(FLAGS.config)
  config = pbutil.FromFile(config_path, clgen_pb2.Instance())
  os.environ['PWD'] = str(config_path.parent)

  instance = Instance(config)
  if FLAGS.print_cache_dir == 'corpus':
    print(instance.model.corpus.cache.path)
    return
  elif FLAGS.print_cache_dir == 'model':
    print(instance.model.cache.path)
    return
  elif FLAGS.print_cache_dir == 'sampler':
    print(instance.model.SamplerCache(instance.sampler))
    return
  elif FLAGS.print_cache_dir:
    raise app.UsageError(
        f"Invalid --print_cache_dir argument: '{FLAGS.print_cache_dir}'")

  instance.model.Sample(instance.sampler, FLAGS.min_samples)


def main(argv):
  """Main entrypoint."""
  if len(argv) > 1:
    raise app.UsageError(
        "Uncrecognized command line options: '{}'".format(', '.join(argv[1:])))

  if FLAGS.clgen_profiling:
    prof.enable()

  run(DoTheRest)


if __name__ == '__main__':
  app.run(main)
