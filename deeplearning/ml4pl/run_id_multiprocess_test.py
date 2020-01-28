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
import time

from deeplearning.ml4pl import run_id
from labm8.py import bazelutil
from labm8.py import test

RUN_ID_PATH = bazelutil.DataPath("phd/deeplearning/ml4pl/run_id")


FLAGS = test.FLAGS


def GetRunId(*args):
  """Get a run ID by running //deeplearning/ml4pl:run_id."""
  return subprocess.check_output([str(RUN_ID_PATH)], universal_newlines=True)


def test_GenerateGlobalUnique_multiprocessed():
  """Generate a bunch of run IDs concurrently and check that they are unique.

  Do this by spawning a bunch of //deeplearning/ml4pl:run_id processes, which
  each print a global run ID. When spanwing multiple procsses simultaneously,
  they will block on the system-wide ID lockfile to produce unique run IDs.
  """
  # The number of run IDs to produce in this test.
  run_id_count = 30

  unique_run_ids = set()

  start_time = time.time()

  with multiprocessing.Pool(processes=8) as p:
    for run_id_ in p.imap_unordered(GetRunId, range(run_id_count)):
      print("Generated run ID:", run_id_)

      # Check that the process produced a valid run ID:
      run_id.RunId.FromString(run_id_)

      if run_id_ in unique_run_ids:
        test.Fail("Non-unique run ID generated")
      unique_run_ids.add(run_id_)

  elapsed_time = time.time() - start_time
  # Because the locking ensures that a single run ID is produced per second,
  # we know that it must take at least 'n-2' seconds to produce 'n' unique run
  # IDs. Any additional time is spent in the overhead of spawning processes,
  # which shouldn't be too great.
  assert elapsed_time >= run_id_count - 2
  assert elapsed_time < run_id_count * 2


if __name__ == "__main__":
  test.Main()
