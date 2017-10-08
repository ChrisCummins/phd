#!/usr/bin/env python3
import logging
import json
import sys

from typing import List, NewType
from pathlib import Path
from labm8 import crypto

import autotest


class CLSmithGenerator(autotest.Generator):
    def __init__(self, exec: Path):
        self.clsmith = exec

        exec_checksum = crypto.sha1_file(self.clsmith)

        logging.debug(f"using CLSmith '{self.clsmith}' {exec_checksum}")

    def next_batch(self) -> List[autotest.testcase_t]:
        pass


class DeviceUnderTest(object):
    def __init__(self):
        pass

    def run(self, testcase: autotest.testcase_t) -> autotest.output_t:
        pass


class StaticAnalyzer(object):
    def __init__(self):
        pass

    def is_valid(self, testcase: autotest.testcase_t) -> bool:
        pass


class Comparator(object):
    def __init__(self):
        pass

    def all_equal(self, outputs: List[autotest.output_t]) -> bool:
        pass


class DynamicAnalyzer(object):
    def __init__(self):
        pass

    def is_valid(self, testcase: autotest.testcase_t,
                 duts: List[autotest.DeviceUnderTest],
                 outputs: List[autotest.output_t]) -> bool:
        pass


class Reducer(object):
    def __init__(self):
        pass

    def reduce(self, testcase: autotest.testcase_t,
               dut: autotest.DeviceUnderTest) -> autotest.output_t:
        pass


def main(args):
    assert len(args) == 2

    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.DEBUG)

    with open(args[0]) as infile:
        json_config = json.loads(infile.read())
        logging.debug(f"parsed config file '{args[0]}'")
    num_batches = int(args[1])

    generator = CLSmithGenerator(**json_config["generator"])
    preflight_checks = [
        StaticAnalyzer(**x) for x in json_config["preflight_checks"]]
    duts = [DeviceUnderTest(**x) for x in json_config["duts"]]
    comparator = Comparator(**json_config["comparator"])
    postflight_checks = [
        DynamicAnalyzer(**x) for x in json_config["postflight_checks"]]
    reducer = Reducer(**json_config["reducer"])

    autotest.autotest(num_batches, generator, preflight_checks, duts,
                      comparator, postflight_checks, reducer)

if __name__ == "__main__":
    main(sys.argv[1:])
