"""Given a corpus of OpenCL kernels mined from GitHub, how many can be run?

This assembles a CLgen corpus from a directory of sample texts, then uses the
cldrive DeepSmith harness to attempt to run all of the successfully preprocessed
files.
"""
import math
import pathlib
import pickle
import typing

import humanize
import numpy as np
import pandas as pd
from absl import app
from absl import flags
from absl import logging

from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import service_pb2
from gpu.oclgrind import oclgrind
from labm8 import labtypes


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'github_kernels_dir', '/var/phd/datasets/github/corpuses/opencl',
    'Directory containing OpenCL github kernels.')
flags.DEFINE_string(
    'result_cache_dir',
    '/tmp/phd/experimental/deeplearning/clgen/learning_from_github_corpus/run_github_corpus',
    'Path to cache experimental results in.')
flags.DEFINE_string(
    'opencl_env', oclgrind.CLINFO_DESCRIPTION.name,
    'The OpenCL environment to execute programs on.')
flags.DEFINE_boolean(
    'opencl_opt', True,
    'If true, enable OpenCL optimizations.')
flags.DEFINE_boolean(
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
  return [
    deepsmith_pb2.Testcase(
        toolchain='opencl',
        harness=deepsmith_pb2.Harness(name='cldrive'),
        inputs={
          'src': src,
          'gsize': f'{lsize},1,1',
          'lsize': f'{gsize},1,1',
        }) for lsize, gsize in LSIZE_GSIZE_PAIRS
  ]


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  # An OpenCL corpus, configured as described in CGO'17.
  corpus = corpuses.Corpus(corpus_pb2.Corpus(
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
      ]
  ))
  corpus.Create()

  cache_dir = pathlib.Path(FLAGS.result_cache_dir) / corpus.hash
  cache_dir.mkdir(parents=True, exist_ok=True)

  # An OpenCL corpus, configured as described in CGO'17.
  driver = cldrive.CldriveHarness(harness_pb2.CldriveHarness(
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
    logging.info('Corpus of %s files (%.1f%% of %s)',
                 humanize.intcomma(num_good_files),
                 (num_good_files / num_files) * 100,
                 humanize.intcomma(num_files))

    srcs = [x[0] for x in q]
    batch_size = 16
    max_batch = math.ceil(len(srcs) / batch_size)

    all_outcomes = []
    for i, start_idx in enumerate(range(0, len(srcs), batch_size)):
      cached_results_path = cache_dir / f'{i}.pkl'
      logging.info('batch %d of %d', i + 1, max_batch)

      if cached_results_path.is_file():
        # Read cached results.
        with open(cached_results_path, 'rb') as f:
          outcomes = pickle.load(f)
      elif FLAGS.summarize_only:
        continue
      else:
        # Evaluate OpenCL kernels and cache results.
        batch = srcs[start_idx:start_idx + batch_size]
        testcases = labtypes.flatten(
            [OpenClSourceToTestCases(src) for src in batch])
        response = driver.RunTestcases(harness_pb2.RunTestcasesRequest(
            testbed=driver.testbeds[0],
            testcases=testcases,
        ), None)

        # Harness returned correct number of results without complaining.
        if response.status.returncode != service_pb2.ServiceStatus.SUCCESS:
          raise OSError(service_pb2.ServiceStatus.ReturnCode.Name(
              response.status.returncode))
        assert len(response.results) == len(testcases)

        outcomes = [
          deepsmith_pb2.Result.Outcome.Name(result.outcome)
          for result in response.results
        ]
        with open(cached_results_path, 'wb') as f:
          pickle.dump(outcomes, f)

      all_outcomes += outcomes

    df = pd.DataFrame(list(zip(all_outcomes, np.ones(len(all_outcomes)))),
                      columns=['outcome', 'count'])
    print(df.groupby('outcome').count())


if __name__ == '__main__':
  app.run(main)
