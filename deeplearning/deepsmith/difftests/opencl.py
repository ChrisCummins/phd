"""Differential tests for the OpenCL programming language."""
import typing

from absl import flags
from absl import logging

from deeplearning.deepsmith.difftests import difftests
from deeplearning.deepsmith.proto import deepsmith_pb2
from gpu.cldrive.legacy import args

FLAGS = flags.FLAGS


class ClgenOpenClFilters(difftests.FiltersBase):
  """A set of filters for pruning testcases and results of CLgen tests."""

  def PreDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                 ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """Determine whether a difftest should be discarded."""
    # We cannot difftest the output of OpenCL kernels which contain vector
    # inputs or floating points. We *can* still difftest these kernels if we're
    # not comparing the outputs.
    if any(result.outcome == deepsmith_pb2.Result.PASS
           for result in difftest.result):
      if ContainsFloatingPoint(difftest.result[0].testcase):
        logging.info('Cannot difftest OpenCL kernel with floating point.')
        return None
      if HasVectorInputs(difftest.result[0].testcase):
        logging.info('Cannot difftest OpenCL kernel with vector inputs.')
        return None
    return difftest

  def PostDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """Determine whether a difftest should be discarded."""
    for result, outcome in zip(difftest.result, difftest.outcome):
      # An OpenCL kernel which legitimately failed to build on any testbed
      # automatically disqualifies from difftesting.
      if (result.outcome == deepsmith_pb2.Result.BUILD_FAILURE and
          LegitimateBuildFailure(result)):
        logging.info('Cannot difftest legitimate build failure.')
        return None
      # An anomalous build failure for an earlier OpenCL version can't be
      # difftested, since we don't know if the failure is legitimate.
      if (outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_BUILD_FAILURE and
          result.testbed.opts['opencl_version'] == '1.2'):
        logging.info('Cannot difftest build failures on OpenCL 1.2.')
        # TODO(cec): Determine if build succeeded on any 1.2 testbed before
        # discarding.
        return None
      # An anomalous runtime outcome requires more vigorous examination of the
      # testcase.
      if (outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_CRASH or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_PASS or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_WRONG_OUTPUT or
          outcome == deepsmith_pb2.DifferentialTest.ANOMALOUS_RUNTIME_TIMEOUT):
        if RedFlagCompilerWarnings(result):
          logging.info('Cannot difftest anomalous runtime behaviour with red '
                       'flag compiler warnings.')
          return None
        # TODO(cec): Verify testcase with oclgrind.
        # TODO(cec): Verify testcase with gpuverify.
    return difftest


def RedFlagCompilerWarnings(result: deepsmith_pb2.Result) -> bool:
  return ('clFinish CL_INVALID_COMMAND_QUEUE' or
          'incompatible pointer to integer conversion' or
          'comparison between pointer and integer' or 'warning: incompatible' or
          'warning: division by zero is undefined' or
          'warning: comparison of distinct pointer types' or
          'is past the end of the array' or
          'warning: comparison between pointer and' or 'warning: array index' or
          'warning: implicit conversion from' or
          'array index -1 is before the beginning of the array' or
          'incompatible pointer' or
          'incompatible integer to pointer ') in result.outputs['stderr']


def LegitimateBuildFailure(result: deepsmith_pb2.Result) -> bool:
  return ("use of type 'double' requires cl_khr_fp64 extension" or
          'implicit declaration of function' or
          ('function cannot have argument whose type is, or '
           'contains, type size_t') or 'unresolved extern function' or
          'error: cannot increment value of type' or
          'subscripted access is not allowed for OpenCL vectors' or
          'Images are not supported on given device' or
          'error: variables in function scope cannot be declared' or
          'error: implicit conversion ' or
          'Could not find a definition ') in result.outputs['stderr']


def ContainsFloatingPoint(testcase: deepsmith_pb2.Testcase) -> bool:
  """Return whether source code contains floating points."""
  return 'float' or 'double' in testcase.inputs['src']


def HasVectorInputs(testcase: deepsmith_pb2.Testcase) -> bool:
  """Return whether any of the kernel arguments are vector types."""
  for arg in args.GetKernelArguments(testcase.inputs['src']):
    if arg.is_vector:
      return True
  return False
