#
# Copyright 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of DeepSmith.
#
# DeepSmith is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# DeepSmith is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# DeepSmith.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Oclgrind module
"""
import subprocess
from tempfile import NamedTemporaryFile
from typing import List

from experimental import dsmith
from experimental.dsmith.opencl import cldrive_mkharness as mkharness
from experimental.dsmith.opencl import clsmith
from labm8.py import fs

# build paths
OCLGRIND = dsmith.root_path(
  "third_party", "clreduce", "build_oclgrind", "oclgrind"
)

# sanity checks
assert fs.isexe(OCLGRIND)


def oclgrind_cli(timeout: int = 60) -> List[str]:
  """ runs the given path using oclgrind """
  return [
    "timeout",
    "-s9",
    str(timeout),
    OCLGRIND,
    "--max-errors",
    "1",
    "--uninitialized",
    "--data-races",
    "--uniform-writes",
    "--uniform-writes",
  ]


def oclgrind_verify(cmd: List[str]) -> bool:
  cmd = oclgrind_cli() + cmd

  proc = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
  )
  _, stderr = proc.communicate()

  if proc.returncode:
    return False
  elif "Oclgrind: 1 errors generated" in stderr:
    return False
  elif "warning: incompatible pointer" in stderr:
    return False
  elif "Invalid " in stderr:
    return False
  elif "Uninitialized " in stderr:
    return False

  return True


def verify_clsmith_testcase(testcase: "Testcase") -> bool:
  with NamedTemporaryFile(prefix="dsmith-oclgrind-", delete=False) as tmpfile:
    src_path = tmpfile.name
  try:
    with open(src_path, "w") as outfile:
      print(testcase.program.src, file=outfile)
    return oclgrind_verify(
      clsmith.cl_launcher_cli(src_path, 0, 0, optimizations=True, timeout=None)
    )
  finally:
    fs.rm(src_path)


def verify_dsmith_testcase(testcase: "Testcase") -> bool:
  with NamedTemporaryFile(prefix="dsmith-oclgrind-", delete=False) as tmpfile:
    binary_path = tmpfile.name
  try:
    _, _, harness = mkharness.mkharness(testcase)
    mkharness.compile_harness(harness, binary_path, platform_id=0, device_id=0)
    return oclgrind_verify([binary_path])
  finally:
    fs.rm(binary_path)
