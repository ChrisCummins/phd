"""Benchmarks in the LLVM test suite.

See: https://llvm.org/docs/TestingGuide.html#test-suite-overview
"""

from absl import flags

from lib.labm8 import bazelutil


FLAGS = flags.FLAGS

BENCHMARKS = {
  'SingleSource': {
    'McGill': {
      'queens': {
        'srcs': [str(bazelutil.DataPath(
            'llvm_test_suite/SingleSource/Benchmarks/McGill/queens.c'))],
        'binary': [bazelutil.DataPath('llvm_test_suite/queens')],
      }
    }
  }
}
