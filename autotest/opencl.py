#!/usr/bin/env python3
import logging
import json
import sys

from typing import List, NewType
from pathlib import Path
from labm8 import crypto

import autotest


class Testcase(object):
    def __init__(self, path: Path):
        self.path = path

    @property
    def src(self):
        with open(self.path) as infile:
            return infile.read()

    def __repr__(self):
        return self.src


class CLSmithGenerator(autotest.Generator):
    def __init__(self, exec: Path):
        self.cmd = exec

        exec_checksum = crypto.sha1_file(self.cmd[0])
        logging.debug(f"CLSmith binary '{self.cmd[0]}' {exec_checksum}")

    def next_batch(self, batch_size: int) -> List[autotest.testcase_t]:
        for _ in range(batch_size):
            logging.debug(" ".join(self.cmd))


class DeviceUnderTest(object):
    def __init__(self, device, platform):
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
