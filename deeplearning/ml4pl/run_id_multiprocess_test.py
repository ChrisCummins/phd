# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-processing tests for //deeplearning/ml4pl:run_id."""
import multiprocessing
import subprocess
import sys

from deeplearning.ml4pl import run_id
from labm8.py import test


FLAGS = test.FLAGS


def _MakeARunId(*args):
  """Generate a run ID."""
  return run_id.RunId.GenerateGlobalUnique()


def test_GenerateGlobalUnique_subprocess():
  """Run IDs should be unique when running as subprocesses."""
  unique_run_ids = set()

  for _ in range(30):
    run_id_ = subprocess.check_output(
      [
        sys.executable,
        "-c",
        "from deeplearning.ml4pl import run_id; "
        "print(run_id.RunId.GenerateGlobalUnique())",
      ],
      universal_newlines=True,
    )

    if run_id_ in unique_run_ids:
      test.Fail("Non-unique run ID generated")
    unique_run_ids.add(run_id_)


def test_GenerateGlobalUnique_multiprocessed():
  """Generate a bunch of run IDs concurrently and check that they are unique."""
  unique_run_ids = set()

  with multiprocessing.Pool() as p:
    for run_id_ in p.imap_unordered(_MakeARunId, range(30)):
      print("Generated run ID:", run_id)
      if run_id_ in unique_run_ids:
        test.Fail("Non-unique run ID generated")
      unique_run_ids.add(run_id_)


if __name__ == "__main__":
  test.Main()
