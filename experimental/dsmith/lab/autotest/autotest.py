import json
import logging
import math
from collections import Counter, namedtuple
from pathlib import Path
from typing import List, NewType

testcase_t = NewType('testcase_t', object)
output_t = NewType('output_t', object)
reduced_t = namedtuple('reduced_t', ['reduced', 'expected', 'actual'])
outbox_t = namedtuple('outbox_x', ['dut', 'testcase'])
majority_t = namedtuple('majority_t', ['majority_value', 'majority_size'])


class GeneratorError(Exception):
  pass


class DeviceUnderTestError(Exception):
  pass


class StaticAnalyzerError(Exception):
  pass


class ComparatorError(Exception):
  pass


class DynamicAnalyzerError(Exception):
  pass


class Generator(object):

  def next_batch(self) -> List[testcase_t]:
    raise NotImplementedError("abstract class")


class DeviceUnderTest(object):

  def run(self, testcase: testcase_t) -> output_t:
    raise NotImplementedError("abstract class")

  def to_json(self):
    raise NotImplementedError("abstract class")


class StaticAnalyzer(object):

  def is_valid(self, testcase: testcase_t) -> bool:
    raise NotImplementedError("abstract class")


class Comparator(object):

  def majority(self, outputs: List[output_t]) -> majority_t:
    return Counter(outputs).most_common(1)[0]


class DynamicAnalyzer(object):

  def is_valid(self, testcase: testcase_t, duts: List[DeviceUnderTest],
               outputs: List[output_t]) -> bool:
    raise NotImplementedError("abstract class")


class Reducer(object):

  def reduce(self, testcase: testcase_t, dut: DeviceUnderTest) -> reduced_t:
    raise NotImplementedError("abstract class")


def export_outbox(outbox: List[reduced_t], path: Path):
  """
  Write interesting reduced testcases to file.

  Arguments:
      outbox: The list of interesting reduced testcases.
      path: Path to write file to.
  """
  blob = [{
      'dut': o['dut'].to_json(),
      'testcase': o['testcase']['reduced'],
      'expected_output': o['testcase']['expected'],
      'actual_output': o['testcase']['actual']
  } for o in outbox]

  with open(path, "w") as outfile:
    json.dump(blob, outfile)


def autotest(num_batches: int,
             generator: Generator,
             preflight_checks: List[StaticAnalyzer],
             duts: List[DeviceUnderTest],
             comparator: Comparator,
             postflight_checks: List[DynamicAnalyzer],
             reducer: Reducer,
             batch_size=1) -> None:
  num_devices = len(duts)
  assert num_devices > 2
  outbox = []

  for i in range(1, num_batches + 1):
    logging.info(
        f"generating {batch_size} testcases, batch {i} of {num_batches}")
    testcases = generator.next_batch(batch_size)

    assert len(testcases)

    for testcase in testcases:
      if len(preflight_checks):
        # Do all the pre-flight checks before running:
        logging.info("running static analysis on testcase")
        if not all(checker.is_valid(testcase) for checker in preflight_checks):
          logging.info("-> testcase failed static analysis")
          continue

      logging.info(f"running testcase on {num_devices} devices")
      outputs = [dut.run(testcase) for dut in duts]

      assert len(outputs) == num_devices

      # Check if outputs are interesting:
      logging.info("comparing outputs of tests")
      majority_output, majority_size = comparator.majority(outputs)
      if majority_size == num_devices:
        logging.info("-> testcase outcomes are all equal")
        continue
      # TODO: take config size as constructor argument
      elif majority_size < math.ceil(2 * num_devices / 3):
        logging.info("-> majority size of {majority_size}, no consensus")
        continue

      if len(postflight_checks):
        # Do all the post-flight checks to validate testcase:
        logging.info("running dynamic analysis on testcase")
        if not all(
            checker.is_valid(testcase, duts, outputs)
            for checker in postflight_checks):
          logging.info("-> testcase failed dynamic analysis")
          continue

      logging.info("identifying outputs of interest")
      for j in range(len(outputs)):
        if outputs[j] != majority_output:
          logging.info("reducing testcase for device")
          reduced, expected, actual = reducer.reduce(testcase, duts[j],
                                                     outputs[j])
          logging.info("-> reduced testcase")
          outbox.append(outbox_t(duts[j], reduced_t))

  export_outbox(outbox, "outbox.json")
