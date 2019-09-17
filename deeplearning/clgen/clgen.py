# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""CLgen: a deep learning program generator.

The core operations of CLgen are:

  1. Preprocess and encode a corpus of handwritten example programs.
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
import contextlib
import cProfile
import os
import pathlib
import shutil
import sys
import traceback
import typing

from deeplearning.clgen import errors
from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen import samplers
from deeplearning.clgen.dashboard import dashboard
from deeplearning.clgen.models import models
from deeplearning.clgen.models import pretrained
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8 import app
from labm8 import pbutil
from labm8 import prof

FLAGS = app.FLAGS

app.DEFINE_string('config', '/clgen/config.pbtxt',
                  'Path to a clgen.Instance proto file.')
app.DEFINE_integer(
    'min_samples', 0,
    'The minimum number of samples to make. If <= 0, sampling continues '
    'indefinitely and never terminates.')
app.DEFINE_boolean('print_samples', True,
                   'If set, print the generated samples.')
app.DEFINE_boolean('cache_samples', False,
                   'If set, cache the generated sample protobufs.')
app.DEFINE_string('sample_text_dir', None,
                  'A directory to write plain text samples to.')
app.DEFINE_string('stop_after', None,
                  'Stop CLgen early. Valid options are: "corpus", or "train".')
app.DEFINE_string(
    'print_cache_path', None,
    'Print the directory of a cache and exit. Valid options are: "corpus", '
    '"model", or "sampler".')
app.DEFINE_string(
    'export_model', None,
    'Path to export a trained TensorFlow model to. This exports all of the '
    'files required for sampling to specified directory. The directory can '
    'then be used as the pretrained_model field of an Instance proto config.')

# TODO(github.com/ChrisCummins/clgen/issues/131): Remove these in favor of
# standard labm8.app methods for enabling extra debugging or profiling
# information.
app.DEFINE_boolean(
    'clgen_debug', False,
    'Enable a debugging mode of CLgen python runtime. When enabled, errors '
    'which may otherwise be caught lead to program crashes and stack traces.')
app.DEFINE_boolean('clgen_profiling', False,
                   'Enable CLgen self profiling. Profiling results be logged.')
app.DEFINE_boolean('clgen_dashboard_only', False,
                   'If true, launch dashboard only.')


class Instance(object):
  """A CLgen instance encapsulates a model, sampler, and working directory."""

  def __init__(self, config: clgen_pb2.Instance, dashboard_opts={}):
    """Instantiate an instance.

    Args:
      config: An Instance proto.

    Raises:
      UserError: If the instance proto contains invalid values, is missing
        a model or sampler fields.
    """
    try:
      pbutil.AssertFieldIsSet(config, 'model_specification')
      pbutil.AssertFieldIsSet(config, 'sampler')
    except pbutil.ProtoValueError as e:
      raise errors.UserError(e)

    self.config = config
    self.working_dir = None
    if config.HasField('working_dir'):
      self.working_dir: pathlib.Path = pathlib.Path(
          os.path.expandvars(config.working_dir)).expanduser().absolute()
    # Enter a session so that the cache paths are set relative to any requested
    # working directory.
    with self.Session():
      if config.HasField('model'):
        self.model: models.Model = models.Model(config.model)
      else:
        self.model: pretrained.PreTrainedModel = pretrained.PreTrainedModel(
            pathlib.Path(config.pretrained_model))
      self.sampler: samplers.Sampler = samplers.Sampler(config.sampler)

    self.dashboard = dashboard.Launch(**dashboard_opts)

  @contextlib.contextmanager
  def Session(self) -> 'Instance':
    """Scoped $CLGEN_CACHE value."""
    old_working_dir = os.environ.get('CLGEN_CACHE', '')
    if self.working_dir:
      os.environ['CLGEN_CACHE'] = str(self.working_dir)
    yield self
    if self.working_dir:
      os.environ['CLGEN_CACHE'] = old_working_dir

  def Create(self) -> None:
    with self.Session():
      self.model.Create()

  def Train(self, *args, **kwargs) -> None:
    with self.Session():
      test_sampler_config = sampler_pb2.Sampler()
      test_sampler_config.CopyFrom(self.sampler.config)
      del test_sampler_config.termination_criteria[:]
      test_sampler_config.termination_criteria.extend([
          sampler_pb2.SampleTerminationCriterion(
              maxlen=sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=512)),
      ])
      test_sampler = samplers.Sampler(test_sampler_config)

      # We inject the `test_sampler` argument so that we can create samples
      # during training.
      self.model.Train(*args, test_sampler=test_sampler_config, **kwargs)

  def Sample(self, *args, **kwargs) -> typing.List[model_pb2.Sample]:
    self.Train()
    with self.Session():
      return self.model.Sample(self.sampler, *args, **kwargs)

  def ExportPretrainedModel(self, export_dir: pathlib.Path) -> None:
    """Export a trained model."""
    if isinstance(self.model, pretrained.PreTrainedModel):
      shutil.copytree(self.config.pretrained_model, export_dir / 'model')
    else:
      self.Train()
      for path in self.model.InferenceManifest():
        relpath = pathlib.Path(os.path.relpath(path, self.model.cache.path))
        (export_dir / relpath.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, export_dir / relpath)

  def ToProto(self) -> clgen_pb2.Instance:
    """Get the proto config for the instance."""
    config = clgen_pb2.Instance()
    config.working_dir = str(self.working_dir)
    config.model.CopyFrom(self.model.config)
    config.sampler.CopyFrom(self.sampler.config)
    return config

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> 'Instance':
    return cls(pbutil.FromFile(path, clgen_pb2.Instance()))


def Flush():
  """Flush logging and stout/stderr outputs."""
  app.FlushLogs()
  sys.stdout.flush()
  sys.stderr.flush()


def LogException(exception: Exception):
  """Log an error."""
  app.Error(
      f"""\
%s (%s)

Please report bugs at <https://github.com/ChrisCummins/phd/issues>\
""", exception,
      type(exception).__name__)
  sys.exit(1)


def LogExceptionWithStackTrace(exception: Exception):
  """Log an error with a stack trace."""

  # get limited stack trace
  def _msg(i, x):
    n = i + 1
    filename, lineno, fnname, _ = x
    # TODO(github.com/ChrisCummins/clgen/issues/131): Report filename relative
    # to PhD root.
    loc = f'{filename}:{lineno}'
    return f'      #{n}  {loc: <18} {fnname}()'

  _, _, tb = sys.exc_info()
  NUM_ROWS = 5  # number of rows in traceback
  trace = reversed(traceback.extract_tb(tb, limit=NUM_ROWS + 1)[1:])
  message = '\n'.join(_msg(*r) for r in enumerate(trace))
  app.Error(
      """\
%s (%s)

  stacktrace:
%s

Please report bugs at <https://github.com/ChrisCummins/phd/issues>\
""", exception,
      type(exception).__name__, message)
  sys.exit(1)


def RunWithErrorHandling(function_to_run: typing.Callable, *args,
                         **kwargs) -> typing.Any:
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
  if FLAGS.clgen_debug:
    # Enable verbose stack traces. See: https://pymotw.com/2/cgitb/
    import cgitb
    cgitb.enable(format='text')
    return function_to_run(*args, **kwargs)

  try:

    def RunContext():
      """Run the function with arguments."""
      return function_to_run(*args, **kwargs)

    if prof.is_enabled():
      return cProfile.runctx('RunContext()', None, locals(), sort='tottime')
    else:
      return RunContext()
  except app.UsageError as err:
    # UsageError is handled by the call to app.RunWithArgs(), not here.
    raise err
  except errors.UserError as err:
    app.Error('%s (%s)', err, type(err).__name__)
    sys.exit(1)
  except KeyboardInterrupt:
    Flush()
    print('\nReceived keyboard interrupt, terminating', file=sys.stderr)
    sys.exit(1)
  except errors.File404 as e:
    Flush()
    LogException(e)
    sys.exit(1)
  except Exception as e:
    Flush()
    LogExceptionWithStackTrace(e)
    sys.exit(1)


def ConfigFromFlags() -> clgen_pb2.Instance:
  config_path = pathlib.Path(FLAGS.config)
  if not config_path.is_file():
    raise app.UsageError(f"CLgen --config file not found: '{config_path}'")
  config = pbutil.FromFile(config_path, clgen_pb2.Instance())
  os.environ['PWD'] = str(config_path.parent)
  return config


def SampleObserversFromFlags(
) -> typing.List[sample_observers_lib.SampleObserver]:
  """Instantiate sample observers from flag values."""
  sample_observers = []
  if FLAGS.min_samples <= 0:
    app.Warning(
        'Entering an infinite sample loop, this process will never end!')
  else:
    sample_observers.append(
        sample_observers_lib.MaxSampleCountObserver(FLAGS.min_samples))
  if FLAGS.print_samples:
    sample_observers.append(sample_observers_lib.PrintSampleObserver())
  if FLAGS.cache_samples:
    sample_observers.append(sample_observers_lib.LegacySampleCacheObserver())
  if FLAGS.sample_text_dir:
    sample_observers.append(
        sample_observers_lib.SaveSampleTextObserver(
            pathlib.Path(FLAGS.sample_text_dir)))
  return sample_observers


def DoFlagsAction(
    instance: Instance,
    sample_observers: typing.List[sample_observers_lib.SampleObserver]) -> None:
  """Do the action requested by the command line flags.

  By default, this method trains and samples the instance using the given
  sample observers. Flags which affect this behaviour are:

    --print_cache_path={corpus,model,sampler}: Prints the path and returns.
    --stop_after={corpus,train}: Stops after corpus creation or training,
        respectively
    --export_model=<path>: Train the model and export it to the requested path.

  Args:
    config: The CLgen instance to act on.
    sample_observer: A list of sample observers. Unused if no sampling occurs.
  """
  if FLAGS.clgen_profiling:
    prof.enable()

  with instance.Session():
    if FLAGS.clgen_dashboard_only:
      instance.Create()
      return
    if FLAGS.print_cache_path == 'corpus':
      print(instance.model.corpus.cache.path)
      return
    elif FLAGS.print_cache_path == 'model':
      print(instance.model.cache.path)
      return
    elif FLAGS.print_cache_path == 'sampler':
      print(instance.model.SamplerCache(instance.sampler))
      return
    elif FLAGS.print_cache_path:
      raise app.UsageError(
          f"Invalid --print_cache_path argument: '{FLAGS.print_cache_path}'")

    # The default action is to sample the model.
    if FLAGS.stop_after == 'corpus':
      instance.model.corpus.Create()
    elif FLAGS.stop_after == 'train':
      instance.Train()
      app.Log(1, 'Model: %s', instance.model.cache.path)
    elif FLAGS.stop_after:
      raise app.UsageError(
          f"Invalid --stop_after argument: '{FLAGS.stop_after}'")
    elif FLAGS.export_model:
      instance.ExportPretrainedModel(pathlib.Path(FLAGS.export_model))
    else:
      instance.Sample(sample_observers)


def main():
  """Main entry point."""
  instance = Instance(ConfigFromFlags(),
                      dashboard_opts={
                          'debug': FLAGS.clgen_dashboard_only,
                      })
  sample_observers = SampleObserversFromFlags()
  DoFlagsAction(instance, sample_observers)


if __name__ == '__main__':
  app.Run(lambda: RunWithErrorHandling(main))
