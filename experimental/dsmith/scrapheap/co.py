"""
Code for compile-only experiments.
"""
import re
from collections import namedtuple
from subprocess import PIPE
from subprocess import Popen
from tempfile import NamedTemporaryFile
from time import time
from typing import List
from typing import NewType

from dsmith.db import *
from dsmith.lib import *

from labm8.py import fs

status_t = NewType("status_t", int)
return_t = namedtuple("return_t", ["runtime", "status", "stdout", "stderr"])


def verify_params(
  platform: str, device: str, optimizations: str, stderr: str
) -> None:
  """ verify that expected params match actual as reported by cldrive """
  optimizations = "on" if optimizations else "off"

  actual_platform = None
  actual_device = None
  actual_optimizations = None

  for line in stderr.split("\n"):
    if line.startswith("[cldrive] Platform: "):
      actual_platform_name = re.sub(
        r"^\[cldrive\] Platform: ", "", line
      ).rstrip()
    elif line.startswith("[cldrive] Device: "):
      actual_device_name = re.sub(r"^\[cldrive\] Device: ", "", line).rstrip()
    elif line.startswith("[cldrive] OpenCL optimizations: "):
      actual_optimizations = re.sub(
        r"^\[cldrive\] OpenCL optimizations: ", "", line
      ).rstrip()

    # check if we've collected everything:
    if actual_platform and actual_device and actual_optimizations:
      assert actual_platform == platform
      assert actual_device == device
      assert actual_optimizations == optimizations
      return


def drive(command: List[str], src: str) -> return_t:
  """ invoke cldrive on source """
  start_time = time()

  with NamedTemporaryFile() as tmp:
    tmp_path = tmp.name

  cli = ["timeout", "-s9", "60", "./libexec/co.sh", tmp_path] + command
  process = Popen(cli, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  stdout, stderr = process.communicate(src.encode("utf-8"))
  fs.rm(tmp_path)
  stdout, stderr = stdout.decode("utf-8"), stderr.decode("utf-8")

  runtime = time() - start_time

  return return_t(
    runtime=runtime,
    status=status_t(process.returncode),
    stdout=stdout,
    stderr=stderr,
  )
