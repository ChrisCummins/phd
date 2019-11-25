"""Evaluate testcase results.

This program evaluates and difftests the results of DeepSmith testcases across
devices.
"""
import collections
import math
import pathlib
import typing

import pandas as pd
import progressbar

from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from deeplearning.deepsmith import result
from deeplearning.deepsmith import testbed
from deeplearning.deepsmith import testcase
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import labtypes
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string(
    'datastore',
    str(
        bazelutil.DataPath(
            'phd/docs/2018_07_issta/artifact_evaluation/data/datastore.pbtxt')),
    'Path to datastore configuration file.')
app.DEFINE_list('input_directories', [
    str(
        bazelutil.DataPath(
            'phd/docs/2018_07_issta/artifact_evaluation/data/our_results')),
    '/tmp/phd/docs/2018_07_issta/artifact_evaluation/results',
], 'Directories to read results from.')
app.DEFINE_string(
    'output_directory',
    '/tmp/phd/docs/2018_07_issta/artifact_evaluation/difftest_classifications',
    'Directory to write classified results to.')


def GetResultRuntimeMs(r: result.Result) -> int:
  for event in r.profiling_events:
    if str(event.type) == 'runtime':
      return event.duration_ms
  return 0


def GetResultOutputClass(r: result.Result) -> str:
  """Determine the output class of a testcase."""

  def RuntimeCrashOrBuildFailure():
    if "[cldrive] Kernel: " in r.outputs['stderr']:
      return 'Runtime crash'
    else:
      return 'Build failure'

  def RuntimeCrashOrBuildCrash():
    if "[cldrive] Kernel: " in r.outputs['stderr']:
      return 'Runtime crash'
    else:
      return 'Build crash'

  def RuntimeTimeoutOrBuildTimeout():
    if "[cldrive] Kernel: " in r.outputs['stderr']:
      return 'Runtime timeout'
    else:
      return 'Build timeout'

  runtime_ms = GetResultRuntimeMs(r)
  timeout_ms = int(r.testcase.harness.opts['timeout_seconds']) * 1000

  if r.returncode == 0:
    return 'Pass'
  # SIGSEV.
  elif r.returncode == 139 or r.returncode == -11:
    return RuntimeCrashOrBuildCrash()
  # SIGTRAP.
  elif r.returncode == -5:
    return RuntimeCrashOrBuildCrash()
  # SIGKILL.
  elif r.returncode == -9 and runtime_ms >= timeout_ms:
    return RuntimeTimeoutOrBuildTimeout()
  elif r.returncode == -9:
    return RuntimeCrashOrBuildCrash()
  # SIGILL.
  elif r.returncode == -4:
    return RuntimeCrashOrBuildCrash()
  # SIGFPE.
  elif r.returncode == -8:
    return RuntimeCrashOrBuildCrash()
  # SIGBUS.
  elif r.returncode == -7:
    return RuntimeCrashOrBuildCrash()
  # SIGABRT.
  elif r.returncode == -6:
    return RuntimeCrashOrBuildCrash()
  elif r.returncode == 1 and runtime_ms >= timeout_ms:
    return RuntimeTimeoutOrBuildTimeout()
  elif r.returncode == 1:
    return RuntimeCrashOrBuildFailure()
  elif r.returncode == 127:
    return RuntimeCrashOrBuildFailure()
  raise LookupError('Failed to output class of result.')


class Majority(typing.NamedTuple):
  majority_outcome: str
  outcome_majority_size: int
  majority_stdout: str
  stdout_majority_size: int


def GetMajorityOutput(results: typing.List[result.Result]) -> Majority:
  """Majority vote on testcase outcomes and outputs."""
  majority_outcome, outcome_majority_size = collections.Counter(
      [r.output_class for r in results]).most_common(1)[0]
  majority_stdout, stdout_majority_size = collections.Counter(
      [r.outputs['stdout'] for r in results]).most_common(1)[0]
  return Majority(majority_outcome, outcome_majority_size, majority_stdout,
                  stdout_majority_size)


def DifftestTestcase(s: db.session_t, t: testcase.Testcase,
                     outdir: pathlib.Path) -> None:
  """Difftest a testcase."""
  results = list(s.query(result.Result).filter(result.Result.testcase == t))
  for r in results:
    r.output_class = GetResultOutputClass(r)
  majority = GetMajorityOutput(results)

  def OutputPath(result_class: str) -> pathlib.Path:
    try:
      if r.testbed.opts['opencl_opt'] == 'enabled':
        opt = '+'
      elif r.testbed.opts['opencl_opt'] == 'disabled':
        opt = '-'
      else:
        raise KeyError
    except KeyError:
      raise LookupError(str(r.testbed))
    testbeds = sorted(x[0] for x in s.query(testbed.Testbed.name))
    dir = outdir / result_class / str(testbeds.index(r.testbed.name)) / opt
    dir.mkdir(parents=True, exist_ok=True)
    return dir / (str(r.id) + '.pbtxt')

  for r in results:
    if r.output_class == 'Build crash':
      pbutil.ToFile(r.ToProto(), OutputPath('bc'))
    elif r.output_class == 'Build timeout':
      pbutil.ToFile(r.ToProto(), OutputPath('bto'))
    elif (majority.majority_outcome == 'Pass' and
          r.output_class == 'Build failure'):
      pbutil.ToFile(r.ToProto(), OutputPath('abf'))
    elif (majority.majority_outcome == 'Pass' and
          r.output_class == 'Runtime crash'):
      pbutil.ToFile(r.ToProto(), OutputPath('arc'))
    elif (r.outputs['stdout'] != majority.majority_stdout and
          majority.majority_outcome == 'Pass' and majority.stdout_majority_size
          >= math.ceil(2 * majority.outcome_majority_size / 3)):
      pbutil.ToFile(r.ToProto(), OutputPath('awo'))
    else:
      pbutil.ToFile(r.ToProto(), OutputPath('pass'))


def ReadClassificationsToTable(output_dir: pathlib.Path) -> pd.DataFrame:
  rows = []
  counters = {}
  for f in fs.lsfiles(output_dir, recursive=True, abspaths=True):
    path = pathlib.Path(f)
    result_class, testbed_num, opt = path.parts[-4:-1]
    t = testbed_num + opt
    if t not in counters:
      counters[t] = collections.defaultdict(int)
    counters[t][result_class] += 1
  for t, result_classes in counters.items():
    rows.append([
        t,
        result_classes['bc'],
        result_classes['bto'],
        result_classes['abf'],
        result_classes['arc'],
        result_classes['awo'],
        sum(result_classes.values()),
    ])
  rows = sorted(rows, key=lambda x: (int(x[0][:-1]), x[0][-1]))
  rows.append([
      'Total',
      len(fs.lsfiles(output_dir / 'bc', recursive=True)),
      len(fs.lsfiles(output_dir / 'bto', recursive=True)),
      len(fs.lsfiles(output_dir / 'abf', recursive=True)),
      len(fs.lsfiles(output_dir / 'arc', recursive=True)),
      len(fs.lsfiles(output_dir / 'awo', recursive=True)),
      len(fs.lsfiles(output_dir / 'pass', recursive=True)),
  ])
  df = pd.DataFrame(
      rows, columns=['Testbed', 'bc', 'bto', 'abf', 'arc', 'awo', 'pass'])
  df['Total'] = df.sum(axis=1)
  return df


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    unknown_args = ', '.join(argv[1:])
    raise app.UsageError(f'Unknown arguments "{unknown_args}"')

  app.Log(1, 'Initializing datastore.')
  config = pathlib.Path(FLAGS.datastore)
  ds = datastore.DataStore.FromFile(config)

  output_dir = pathlib.Path(FLAGS.output_directory)
  # Make directories to write the classifications to. We use the same shorthand
  # classification names as in Table 2 of the paper:
  #
  #   http://chriscummins.cc/pub/2018-issta.pdf
  (output_dir / 'bc').mkdir(parents=True, exist_ok=True)
  (output_dir / 'bto').mkdir(exist_ok=True)
  (output_dir / 'abf').mkdir(exist_ok=True)
  (output_dir / 'arc').mkdir(exist_ok=True)
  (output_dir / 'awo').mkdir(exist_ok=True)
  (output_dir / 'pass').mkdir(exist_ok=True)
  result_dirs = [
      pathlib.Path(x)
      for x in FLAGS.input_directories
      if pathlib.Path(x).is_dir()
  ]
  results_paths = labtypes.flatten(
      [pathlib.Path(x)
       for x in fs.lsfiles(x, recursive=True, abspaths=True)]
      for x in result_dirs)
  app.Log(1, 'Importing %d results into datastore ...', len(results_paths))
  with ds.Session(commit=True) as s:
    for path in progressbar.ProgressBar()(results_paths):
      # Instantiating a result from file has the side effect of adding the
      # result object to the datastore's session.
      result.Result.FromFile(s, path)

  with ds.Session() as s:
    testcases = s.query(testcase.Testcase)
    app.Log(1, 'Difftesting the results of %d testcases ...', testcases.count())
    for t in progressbar.ProgressBar(max_value=testcases.count())(testcases):
      DifftestTestcase(s, t, output_dir)
  df = ReadClassificationsToTable(output_dir)
  print()
  print('Table of results. For each testbed, this shows the number of results')
  print('of each class, using the same shortand as in Table 2 of the paper:')
  print('http://chriscummins.cc/pub/2018-issta.pdf')
  print()
  print(df.to_string(index=False))
  print()
  print('Individual classified programs are written to: '
        f"'{output_dir}/<class>/<device>/'")


if __name__ == '__main__':
  app.RunWithArgs(main)
