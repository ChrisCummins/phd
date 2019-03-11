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
OpenCL test harnesses.
"""
from tempfile import NamedTemporaryFile
from time import time

from experimental.dsmith.langs import Generator, Harness
from experimental.dsmith.sol import generators
from experimental.dsmith.sol.db import *


def _log_outcome(outcome: Outcomes, runtime: float):
  """ verbose logging output """
  outcome_name = Outcomes.to_str(outcome)
  return_color = Colors.GREEN if outcome == Outcomes.PASS else Colors.RED
  app.Log(1, f"↳  {Colors.BOLD}{return_color}{outcome_name}{Colors.END} "
           f"after {Colors.BOLD}{runtime:.2f}{Colors.END} seconds")


class SolidityHarness(Harness):
  """ Common superclass for test harnesses """

  def run(self, session: session_t, testcase: Testcase,
          testbed: Testbed) -> ResultProxy:
    """ execute a testcase """
    raise NotImplementedError

  def make_testcases(self, generator: Generator):
    """ Generate testcases, optionally for a specific generator """
    # Sanity check
    if generator.__name__ not in self.__generators__:
      raise ValueError(f"incompatible combination {self}:{generator}")

    with Session() as s:
      already_exists = s.query(Program.id) \
        .join(Testcase) \
        .filter(Program.generator == generator.id,
                Testcase.harness == self.id)

      # The list of testcases to make is the compliment of the above:
      todo = s.query(Program.id) \
        .filter(Program.generator == generator.id,
                ~Program.id.in_(already_exists))

      # Determine how many, if any, testcases need to be made:
      ndone = already_exists.count()
      ntodo = todo.count()
      ntotal = ndone + ntodo
      app.Log(2, f"{self}:{generator} testcases = {ndone} / {ntotal}")

      # Break early if there's nothing to do:
      if not ntodo:
        return

      print(f"Generating {Colors.BOLD}{ntodo}{Colors.END} "
            f"{self}:{generator} testcases")

      # Bulk insert new testcases:
      s.add_all(
          Testcase(
              program_id=program.id,
              harness=self.id,
              timeout=self.default_timeout,
          ) for program in todo)
      s.commit()

  def testbeds(self, session: session_t = None) -> List[TestbedProxy]:
    with ReuseSession(session) as s:
      q = s.query(Testbed) \
        .join(Platform)
      return sorted(TestbedProxy(testbed) for testbed in q)

  def available_testbeds(self, session: session_t = None) -> List[TestbedProxy]:
    with ReuseSession(session) as s:
      testbeds = []
      # TODO: determine if solc / solcjs is installed in path
      testbeds += Testbed.from_bin("solc", session=s)
      testbeds += Testbed.from_bin("solcjs", session=s)
      return sorted(TestbedProxy(testbed) for testbed in testbeds)

  def num_results(self,
                  generator: Generator,
                  testbed: str,
                  session: session_t = None):
    with ReuseSession(session) as s:
      testbed_ = Testbed.from_str(testbed, session=s)[0]
      n = s.query(func.count(Result.id)) \
        .join(Testcase) \
        .join(Program) \
        .filter(Result.testbed_id == testbed_.id,
                Program.generator == generator,
                Testcase.harness == self.id) \
        .scalar()
      return n


class Solc(SolidityHarness):
  """
  The solc compiler
  """
  __name__ = "solc"
  __generators__ = {
      "randchar": generators.RandChar,
      "github": generators.GitHub,
      "dsmith": generators.DSmith,
  }

  id = Harnesses.SOLC
  default_timeout = 60

  def run(self, session: session_t, testcase: Testcase,
          testbed: Testbed) -> ResultProxy:
    """ execute a testcase using cl_launcher """

    with NamedTemporaryFile(
        prefix='dsmith-solc-', suffix='.sol', delete=False) as tmp:
      tmp.write(testcase.program.src.encode("utf-8"))
      tmp.flush()
      path = tmp.name

    cmd = [testbed.platform.platform, '--bin', path]
    if testbed.optimizations:
      cmd.append('--optimize')
    app.Log(2, f"{Colors.BOLD}${Colors.END} " + " ".join(cmd))

    try:
      start_time = time()
      process = subprocess.Popen(
          cmd,
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          universal_newlines=True)
      stdout, stderr = process.communicate()
      runtime = time() - start_time
      returncode = process.returncode
    finally:
      fs.rm(path)

    outcome = SolcResult.get_outcome(returncode, stderr, runtime,
                                     testcase.timeout)

    _log_outcome(outcome, runtime)

    return ResultProxy(
        testbed_id=testbed.id,
        testcase_id=testcase.id,
        returncode=returncode,
        outcome=outcome,
        runtime=runtime,
        stdout=stdout,
        stderr=stderr)
