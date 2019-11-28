#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path
from typing import List

import autotest
import cldrive
from dsmith.opencl import clsmith

from labm8.py import crypto


class OpenCLTestcase(object):

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
    self.exec = exec
    exec_checksum = crypto.sha1_file(self.exec)
    app.Log(2, f"CLSmith binary '{self.exec}' {exec_checksum}")

  def _clsmith(self, path: Path, *flags, attempt_num=1) -> Path:
    """ Generate a program using CLSmith """
    if attempt_num >= 1000:
      raise autotest.GeneratorError(
          f"failed to generate a program using CLSmith after {attempt_num} attempts"
      )

    flags = ['-o', path, *flags]
    app.Log(2, " ".join([self.exec] + flags))

    _, returncode, stdout, stderr = clsmith.clsmith(*flags, exec_path=self.exec)

    # A non-zero returncode of clsmith implies that no program was
    # generated. Try again
    if returncode:
      app.Log(2, f"CLSmith call failed with returncode {returncode}:")
      app.Log(2, stdout)
      self._clsmith(path, *flags, attempt_num=attempt_num + 1)

    return path

  def next_batch(self, batch_size: int) -> List[OpenCLTestcase]:
    outbox = []

    for i in range(batch_size):
      generated_kernel = self._clsmith(f"clsmith-{i}.cl")
      outbox.append(OpenCLTestcase(generated_kernel))

    return outbox


class DeviceUnderTest(object):

  def __init__(self, platform: str, device: str, flags: List[str]):
    self.device = device
    self.platform = platform
    self.flags = flags
    self.env = cldrive.make_env(self.platform, self.device)
    self.ids = self.env.ids()

  def run(self, testcase: autotest.testcase_t) -> autotest.output_t:
    runtime, returncode, stdout, stderr = clsmith.cl_launcher(
        testcase.path, *self.ids, *self.flags)

    print(runtime)
    print(returncode)
    print(stdout[:200])
    print(stderr[:200])


class StaticAnalyzer(object):

  def __init__(self):
    pass

  def is_valid(self, testcase: autotest.testcase_t) -> bool:
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
    app.Log(2, f"parsed config file '{args[0]}'")
  num_batches = int(args[1])

  generator = CLSmithGenerator(clsmith.exec_path)
  preflight_checks = [
      StaticAnalyzer(**x) for x in json_config["preflight_checks"]
  ]
  duts = [DeviceUnderTest(**x) for x in json_config["duts"]]
  comparator = autotest.Comparator(**json_config["comparator"])
  postflight_checks = [
      DynamicAnalyzer(**x) for x in json_config["postflight_checks"]
  ]
  reducer = Reducer(**json_config["reducer"])

  autotest.autotest(num_batches, generator, preflight_checks, duts, comparator,
                    postflight_checks, reducer)


if __name__ == "__main__":
  main(sys.argv[1:])
