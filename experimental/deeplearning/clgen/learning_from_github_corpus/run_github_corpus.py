"""Given a corpus of OpenCL kernels mined from GitHub, how many can be run?

This assembles a CLgen corpus from a directory of sample texts, then uses the
cldrive DeepSmith harness to attempt to run all of the successfully preprocessed
files.
"""
import math
import os
import pathlib
import pickle
import typing

import numpy as np
import pandas as pd

from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import service_pb2
from gpu.oclgrind import oclgrind
from labm8 import app
from labm8 import humanize
from labm8 import labtypes

FLAGS = app.FLAGS

app.DEFINE_string('github_kernels_dir',
                  '/var/phd/datasets/github/corpuses/opencl',
                  'Directory containing OpenCL github kernels.')
app.DEFINE_string(
    'result_cache_dir',
    '/tmp/phd/experimental/deeplearning/clgen/learning_from_github_corpus/run_github_corpus',
    'Path to cache experimental results in.')
app.DEFINE_string('opencl_env', oclgrind.CLINFO_DESCRIPTION.name,
                  'The OpenCL environment to execute programs on.')
app.DEFINE_boolean('opencl_opt', True, 'If true, enable OpenCL optimizations.')
app.DEFINE_boolean(
    'summarize_only', False,
    'If true, only summarize cached results, do not run new experiments.')

# All the combinations of local and global sizes used for synthetic kernels in
# the CGO'17 experiments. These are the first dimension values, the other two
# dimensions are ones. E.g. the first row (64, 64) means local (workgroup) size
# of (64, 1, 1), and a global size of (64, 1, 1).
LSIZE_GSIZE_PAIRS = [
    (64, 64),
    (128, 128),
    (256, 256),
    (256, 512),
    (256, 1024),
    (256, 2048),
    (256, 4096),
    (256, 8192),
    (256, 16384),
    (256, 65536),
    (256, 131072),
    (256, 262144),
    (256, 524288),
    (256, 1048576),
    (256, 2097152),
    (256, 4194304),
]


def OpenClSourceToTestCases(src: str) -> typing.List[deepsmith_pb2.Testcase]:
  """Generate DeepSmith testcases for each of the combination of gsize and
  lsize used in CGO'17 synthetic kernels."""
  return [
      deepsmith_pb2.Testcase(
          toolchain='opencl',
          harness=deepsmith_pb2.Harness(name='cldrive'),
          inputs={
              'src': src,
              'gsize': f'{gsize},1,1',
              'lsize': f'{lsize},1,1',
          }) for lsize, gsize in LSIZE_GSIZE_PAIRS
  ]


def RunTestCasesOrDie(driver: cldrive.CldriveHarness,
                      testcases: typing.List[deepsmith_pb2.Testcase]
                     ) -> typing.Iterable[deepsmith_pb2.Result]:
  """Run the test cases and return their results."""
  response = driver.RunTestcases(
      harness_pb2.RunTestcasesRequest(
          testbed=driver.testbeds[0],
          testcases=testcases,
      ), None)

  # Harness returned correct number of results without complaining.
  if response.status.returncode != service_pb2.ServiceStatus.SUCCESS:
    app.Fatal(
        'Driver failed with return code %s',
        service_pb2.ServiceStatus.ReturnCode.Name(response.status.returncode))
  assert len(response.results) == len(testcases)
  return response.results


def GetOutcomeWithDynamicChecks(result: deepsmith_pb2.Result,
                                driver: cldrive.CldriveHarness) -> str:
  """Get the outcome name of a result, including running additional dynamic
  checks as required."""
  outcome = deepsmith_pb2.Result.Outcome.Name(result.outcome)
  if outcome != 'PASS':
    return outcome

  # Check that the kernel was executed.
  if result.testcase.invariant_opts['driver_type'] != 'compile_and_run':
    return 'DRIVER_FAILED'

  # TODO(cec): Check that outputs are different from inputs, i.e. that the
  # kernel is input sensitive.

  # Run the kernel again to check that it's output is stable.
  repeat_result = RunTestCasesOrDie(driver, [result.testcase])[0]
  repeat_outcome = deepsmith_pb2.Result.Outcome.Name(result.outcome)
  if repeat_outcome != 'PASS':
    app.Log(1, 'Kernel failed when run a second time: %s', repeat_outcome)
    return 'DIFFTEST_FAIL'

  # The output should be the same when run twice with the same input.
  if len(set(r.outputs['stdout'] for r in [result, repeat_result])) != 1:
    app.Log(1, 'Kernel failed nondeterminism test on first input')
    return 'DIFFTEST_NONDETERMINISM_FAIL'

  # Run kernel twice more, with a pair of identical inputs.
  different_inputs_testcase = result.testcase
  different_inputs_testcase.inputs['data_generator'] = 'ones'
  different_input_results = RunTestCasesOrDie(
      driver, [different_inputs_testcase, different_inputs_testcase])

  outcomes = [
      deepsmith_pb2.Result.Outcome.Name(r.outcome)
      for r in different_input_results
  ]
  if not set(outcomes) == {'PASS'}:
    app.Log(1, 'Kernel failed when run on second inputs')
    return 'DIFFTEST_FAIL'

  # The output should be the same when run twice with the same input.
  if len(set(r.outputs['stdout'] for r in different_input_results)) != 1:
    app.Log(1, 'Kernel failed nondeterminism test on seocnd inputs')
    return 'DIFFTEST_NONDETERMINISM_FAIL'

  # The outputs must be different when run twice with the same inputs.
  if len(
      set(r.outputs['stdout']
          for r in [result, different_input_results[0]])) == 1:
    app.Log(1, 'Kernel produced identicial outputs with differnet inputs')
    return 'INPUT_INSENSITIVE'

  return 'PASS'


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  os.environ['CLGEN_CACHE'] = f'{FLAGS.result_cache_dir}/clgen'
  # An OpenCL corpus, configured as described in CGO'17.
  corpus = corpuses.Corpus(
      corpus_pb2.Corpus(
          local_directory=FLAGS.github_kernels_dir,
          ascii_character_atomizer=True,
          contentfile_separator="\n\n",
          preprocessor=[
              "deeplearning.clgen.preprocessors.opencl:ClangPreprocessWithShim",
              "deeplearning.clgen.preprocessors.opencl:Compile",
              "deeplearning.clgen.preprocessors.opencl:NormalizeIdentifiers",
              "deeplearning.clgen.preprocessors.opencl:StripDoubleUnderscorePrefixes",
              "deeplearning.clgen.preprocessors.common:StripDuplicateEmptyLines",
              "deeplearning.clgen.preprocessors.opencl:SanitizeKernelPrototype",
              "deeplearning.clgen.preprocessors.common:StripTrailingWhitespace",
              "deeplearning.clgen.preprocessors.opencl:ClangFormat",
              "deeplearning.clgen.preprocessors.common:MinimumLineCount3",
              "deeplearning.clgen.preprocessors.opencl:Compile",
          ]))
  corpus.Create()

  cache_dir = pathlib.Path(FLAGS.result_cache_dir) / corpus.hash
  cache_dir.mkdir(parents=True, exist_ok=True)

  driver = cldrive.CldriveHarness(
      harness_pb2.CldriveHarness(
          opencl_env=[FLAGS.opencl_env],
          opencl_opt=[FLAGS.opencl_opt],
      ))

  with corpus.preprocessed.Session() as session:
    # Query to return all successfully preprocessed OpenCL kernels in a stable
    # order.
    q = session.query(preprocessed.PreprocessedContentFile.text) \
      .filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True) \
      .order_by(preprocessed.PreprocessedContentFile.id)

    num_good_files = q.count()
    num_files = session.query(preprocessed.PreprocessedContentFile).count()
    app.Log(1, 'Corpus of %s files (%.1f%% of %s)',
             humanize.Commas(num_good_files),
             (num_good_files / num_files) * 100, humanize.Commas(num_files))

    srcs = [x[0] for x in q]
    batch_size = 8
    max_batch = math.ceil(len(srcs) / batch_size)

    all_outcomes = []
    for i, start_idx in enumerate(range(0, len(srcs), batch_size)):
      cached_results_path = cache_dir / f'{i}.pkl'

      if cached_results_path.is_file():
        app.Log(1, 'batch %d of %d', i + 1, max_batch)
        # Read cached results.
        with open(cached_results_path, 'rb') as f:
          outcomes = pickle.load(f)
      elif FLAGS.summarize_only:
        continue
      else:
        app.Log(1, 'batch %d of %d', i + 1, max_batch)
        # Evaluate OpenCL kernels and cache results.
        batch = srcs[start_idx:start_idx + batch_size]
        testcases = labtypes.flatten(
            [OpenClSourceToTestCases(src) for src in batch])
        results = RunTestCasesOrDie(driver, testcases)

        outcomes = [
            GetOutcomeWithDynamicChecks(result, driver) for result in results
        ]
        with open(cached_results_path, 'wb') as f:
          pickle.dump(outcomes, f)

      all_outcomes += outcomes
      df = pd.DataFrame(
          list(zip(all_outcomes, np.ones(len(all_outcomes)))) +
          [('Total', len(all_outcomes))],
          columns=['outcome', 'count'])
      summary = df.groupby('outcome').sum().reset_index()
      summary['ratio'] = [
          f'{x:.2%}' for x in
          # Double the "ratio" values because the 'count' column contains a
          # grand total row.
          2 * summary['count'].values / summary['count'].sum()
      ]
      summary['count'] = [humanize.Commas(int(x)) for x in summary['count']]
      print(summary)
      del df
      del summary


if __name__ == '__main__':
  app.RunWithArgs(main)
