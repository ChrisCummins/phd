"""This module defines differential tests for results."""
import typing
from absl import flags

from deeplearning.deepsmith.proto import deepsmith_pb2


FLAGS = flags.FLAGS


class FiltersBase(object):
  """Base class for DeepSmith filters."""

  def PreExec(self, testcase: deepsmith_pb2.Testcase
              ) -> typing.Optional[deepsmith_pb2.Testcase]:
    """A filter callback to determine whether a testcase should be discarded.

    Args:
      testcase: The testcase to filter.

    Returns:
      True if testcase should be discarded, else False.
    """
    return testcase

  def PostExec(self, result: deepsmith_pb2.Result
               ) -> typing.Optional[deepsmith_pb2.Result]:
    """A filter callback to determine whether a result should be discarded.

    Args:
      result: The result to filter.

    Returns:
      True if result should be discarded, else False.
    """
    return result

  def PreDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                  ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """A filter callback to determine whether a difftest should be discarded.

    Args:
      difftest: The difftest to filter.

    Returns:
      True if difftest should be discarded, else False.
    """
    return difftest

  def PostDifftest(self, difftest: deepsmith_pb2.DifferentialTest
                   ) -> typing.Optional[deepsmith_pb2.DifferentialTest]:
    """A filter callback to determine whether a difftest should be discarded.

    Args:
      difftest: The difftest to filter.

    Returns:
      True if difftest should be discarded, else False.
    """
    return difftest
