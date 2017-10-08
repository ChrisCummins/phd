import logging

from collections import namedtuple
from typing import List, NewType

testcase_t = NewType('testcase_t', object)
output_t = NewType('output_t', object)
outbox_t = namedtuple('outbox_x', ['testcase', 'dut', 'output'])

class GeneratorError(Exception): pass
class DeviceUnderTestError(Exception): pass
class StaticAnalyzerError(Exception): pass
class ComparatorError(Exception): pass
class DynamicAnalyzerError(Exception): pass


class Generator(object):
    def next_batch(self) -> List[testcase_t]:
        raise NotImplementedError("abstract class")


class DeviceUnderTest(object):
    def run(self, testcase: testcase_t) -> output_t:
        raise NotImplementedError("abstract class")


class StaticAnalyzer(object):
    def is_valid(self, testcase: testcase_t) -> bool:
        raise NotImplementedError("abstract class")


class Comparator(object):
    def all_equal(self, outputs: List[output_t]) -> bool:
        raise NotImplementedError("abstract class")


class DynamicAnalyzer(object):
    def is_valid(self, testcase: testcase_t,
                 duts: List[DeviceUnderTest],
                 outputs: List[output_t]) -> bool:
        raise NotImplementedError("abstract class")


class Reducer(object):
    def reduce(self, testcase: testcase_t, dut: DeviceUnderTest) -> output_t:
        raise NotImplementedError("abstract class")


def autotest(num_batches: int, generator: Generator,
             preflight_checks: List[StaticAnalyzer],
             duts: List[DeviceUnderTest],
             comparator: Comparator,
             postflight_checks: List[DynamicAnalyzer],
             reducer: Reducer) -> None:
    outbox = []

    for i in range(1, num_batches + 1):
        logging.info(f"generating batch {i} of {num_batches}")
        testcases = generator.next_batch()

        for testcase in testcases:
            # Do all the pre-flight checks before running:
            logging.info("running static analysis on testcase")
            if not all(checker.is_valid(testcase) for checker in preflight_checks):
                logging.info("-> testcase failed static analysis")
                continue

            logging.info("running testcases on devices")
            outputs = [dut.run(testcase) for dut in duts]

            # Check if outputs are interesting:
            logging.info("comparing outputs of tests")
            if not comparator.all_equal(outputs):
                logging.info("-> testcase outcomes are all equal")
                continue

            # Do all the post-flight checks to validate testcase:
            logging.info("running dynamic analysis on testcase")
            if not all(checker.is_valid(testcase, duts, outputs)
                       for checker in postflight_checks):
                logging.info("-> testcase failed dynamic analysis")
                continue

            logging.info("identifying outputs of interest")
            for j in range(len(outputs)):
                if comparator.is_interesting(outputs, j):
                    logging.info("reducing output")
                    reduced = reducer.reduce(testcase, duts[j], outputs[j])
                    logging.info("-> reduced output from device")
                    outbox.append(outbox_t(testcase, duts[j], reduced))

    for item in outbox:
        print(item)
