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
Differential test OpenCL results.
"""
from experimental.dsmith.opencl.db import *


def difftest():
  with Session() as s:
    create_results_metas(s)
    create_majorities(s)
    create_classifications(s)
    prune_abf_classifications(s)
    prune_arc_classifications(s)
    prune_awo_classifications(s)


def create_results_metas(s: session_t):
  """
  Create total time and cumulative time for each test case evaluated on each
  testbed using each harness.
  """

  class Worker(threading.Thread):
    """ worker thread to run testcases asynchronously """

    def __init__(
      self,
      testbeds_harnesses: List[Tuple["Testbed.id_t", "Harnesses.column_t"]],
    ):
      self.ndone = 0
      self.testbeds_harnesses = testbeds_harnesses
      super(Worker, self).__init__()

    def run(self):
      """ main loop"""
      with Session() as s:
        for testbed_id, harness in self.testbeds_harnesses:
          self.ndone += 1
          testbed = s.query(Testbed).filter(Testbed.id == testbed_id).scalar()

          # FIXME: @cumtime variable is not supported by SQLite.
          s.execute(
            f"""
INSERT INTO {ResultMeta.__tablename__} (id, total_time, cumtime)
SELECT  results.id,
        results.runtime + programs.generation_time AS total_time,
        @cumtime := @cumtime + results.runtime + programs.generation_time AS cumtime
FROM {Result.__tablename__} results
INNER JOIN {Testcase.__tablename__} testcases ON results.testcase_id = testcases.id
INNER JOIN {Program.__tablename__} programs ON testcases.program_id = programs.id
JOIN (SELECT @cumtime := 0) r
WHERE results.testbed_id = {testbed.id}
AND testcases.harness = {harness}
ORDER BY programs.date, testcases.threads_id"""
          )
          s.commit()

  # break early if we can
  num_results = s.query(func.count(Result.id)).scalar()
  num_metas = s.query(func.count(ResultMeta.id)).scalar()
  if num_results == num_metas:
    return

  print("creating results metas ...")
  s.execute(f"DELETE FROM {ResultMeta.__tablename__}")
  testbeds_harnesses = (
    s.query(Result.testbed_id, Testcase.harness)
    .join(Testcase)
    .group_by(Result.testbed_id, Testcase.harness)
    .order_by(Testcase.harness, Result.testbed_id)
    .all()
  )

  bar = progressbar.ProgressBar(
    initial_value=0, max_value=len(testbeds_harnesses), redirect_stdout=True
  )
  worker = Worker(testbeds_harnesses)
  worker.start()
  while worker.is_alive():
    bar.update(min(worker.ndone, len(testbeds_harnesses)))
    worker.join(0.5)


def create_majorities(s: session_t) -> None:
  """
  Majority vote on testcase outcomes and outputs.
  """
  # We require at least this many results in order for there to be a majority:
  min_results_for_majority = 3

  print("voting on test case majorities ...")
  s.execute(f"DELETE FROM {Majority.__tablename__}")

  # Note we have to insert ignore here because there may be ties in the
  # majority outcome or output. E.g. there could be a test case with an even
  # split of 5 '1' outcomes and 5 '3' outcomes. Since there is only a single
  # majority outcome, we order results by outcome number, so that '1' (build
  # failure) will over-rule '6' (pass).
  insert_ignore = (
    "INSERT IGNORE" if dsmith.DB_ENGINE == "mysql" else "INSERT OR IGNORE"
  )
  s.execute(
    f"""
{insert_ignore} INTO {Majority.__tablename__}
    (id, num_results, maj_outcome, outcome_majsize, maj_stdout_id, stdout_majsize)
SELECT  result_counts.testcase_id,
        result_counts.num_results,
        outcome_majs.maj_outcome,
        outcome_majs.outcome_majsize,
        stdout_majs.maj_stdout_id,
        stdout_majs.stdout_majsize
FROM (
    SELECT testcase_id, num_results FROM (
        SELECT testcase_id, COUNT(*) AS num_results
        FROM {Result.__tablename__}
        GROUP BY testcase_id
    ) s
    WHERE num_results >= {min_results_for_majority}
) result_counts
JOIN (
    SELECT l.testcase_id, s.outcome as maj_outcome, s.outcome_count AS outcome_majsize
    FROM (
        SELECT testcase_id, MAX(outcome_count) as max_count
        FROM (
            SELECT testcase_id,COUNT(*) as outcome_count
            FROM {Result.__tablename__}
            GROUP BY testcase_id, outcome
        ) r
        GROUP BY testcase_id
    ) l
    INNER JOIN (
        SELECT testcase_id, outcome, COUNT(*) as outcome_count
        FROM {Result.__tablename__}
        GROUP BY testcase_id, outcome
    ) s ON l.testcase_id = s.testcase_id AND l.max_count = s.outcome_count
) outcome_majs ON result_counts.testcase_id = outcome_majs.testcase_id
JOIN (
    SELECT l.testcase_id, s.stdout_id as maj_stdout_id, s.stdout_count AS stdout_majsize
    FROM (
        SELECT testcase_id, MAX(stdout_count) as max_count
        FROM (
            SELECT testcase_id, COUNT(*) as stdout_count
            FROM {Result.__tablename__}
            GROUP BY testcase_id, stdout_id
        ) r
        GROUP BY testcase_id
    ) l
    INNER JOIN (
        SELECT testcase_id, stdout_id, COUNT(*) as stdout_count
        FROM {Result.__tablename__}
        GROUP BY testcase_id, stdout_id
    ) s ON l.testcase_id = s.testcase_id AND l.max_count = s.stdout_count
) stdout_majs ON outcome_majs.testcase_id = stdout_majs.testcase_id
ORDER BY outcome_majs.maj_outcome DESC
"""
  )
  s.commit()


def create_classifications(s: session_t) -> None:
  """
  Determine anomalous results.
  """
  s.execute(f"DELETE FROM {Classification.__tablename__}")

  min_majsize = 7

  print("creating {bc,bto} classifications ...")
  s.execute(
    f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.BC}
FROM {Result.__tablename__} results
WHERE outcome = {Outcomes.BC}
"""
  )
  s.execute(
    f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.BTO}
FROM {Result.__tablename__} results
WHERE outcome = {Outcomes.BTO}
"""
  )

  print("determining anomalous build-failures ...")
  s.execute(
    f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.ABF}
FROM {Result.__tablename__} results
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.BF}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {Outcomes.PASS}
"""
  )

  print("determining anomalous runtime crashes ...")
  s.execute(
    f"""
INSERT INTO {Classification.__tablename__}
SELECT {Result.__tablename__}.id, {Classifications.ARC}
FROM {Result.__tablename__}
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.RC}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {Outcomes.PASS}
"""
  )

  print("determining anomylous wrong output classifications ...")
  s.execute(
    f"""
INSERT INTO {Classification.__tablename__}
SELECT results.id, {Classifications.AWO}
FROM {Result.__tablename__} results
INNER JOIN {Majority.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {Outcomes.PASS}
AND maj_outcome = {Outcomes.PASS}
AND outcome_majsize >= {min_majsize}
AND stdout_majsize >= CEILING(2 * outcome_majsize / 3)
AND stdout_id <> maj_stdout_id
"""
  )
  s.commit()


def prune_awo_classifications(s: session_t) -> None:
  def testcases_to_verify(session: session_t) -> query_t:
    q = (
      session.query(Result.testcase_id)
      .join(Classification)
      .filter(Classification.classification == Classifications.AWO)
      .distinct()
    )
    return session.query(Testcase).filter(Testcase.id.in_(q)).distinct()

  class Worker(threading.Thread):
    """ worker thread to run testcases asynchronously """

    def __init__(self):
      self.ndone = 0
      super(Worker, self).__init__()

    def run(self):
      """ main loop"""
      with Session() as s:
        for testcase in testcases_to_verify(s):
          self.ndone += 1
          if not testcase.verify_awo(s):
            testcase.retract_classifications(s, Classifications.AWO)
        s.commit()

  print("Verifying awo-classified testcases ...")
  ntodo = testcases_to_verify(s).count()
  bar = progressbar.ProgressBar(
    initial_value=0, max_value=ntodo, redirect_stdout=True
  )
  worker = Worker()
  worker.start()
  while worker.is_alive():
    bar.update(min(worker.ndone, ntodo))
    worker.join(0.5)


def verify_opencl_version(s: session_t, testcase: Testcase) -> None:
  """
  OpenCL 2.0
  """
  opencl_2_0_platforms = s.query(Platform.id).filter(Platform.opencl == "2.0")

  passes_2_0 = (
    s.query(sql.sql.func.count(Result.id))
    .join(Testbed)
    .filter(
      Result.testcase_id == testcase.id,
      Testbed.platform_id.in_(opencl_2_0_platforms),
      Result.outcome != Outcomes.BF,
    )
    .scalar()
  )

  if not passes_2_0:
    # If it didn't build on OpenCL 2.0, we're done.
    return

  passes_1_2 = (
    s.query(sql.sql.func.count(Result.id))
    .join(Testbed)
    .filter(
      Result.testcase_id == testcase.id,
      ~Testbed.platform_id.in_(opencl_2_0_platforms),
      Result.outcome == Outcomes.PASS,
    )
    .scalar()
  )

  if passes_1_2:
    # If it *did* build on OpenCL 1.2, we're done.
    return

  testcase.retract_classifications(s, Classifications.ABF)


def prune_abf_classifications(s: session_t) -> None:
  def prune_stderr_like(like):
    q = (
      s.query(Result.id)
      .join(Classification)
      .join(Stderr)
      .filter(
        Classification.classification == Classifications.ABF,
        Stderr.stderr.like(f"%{like}%"),
      )
    )
    ids_to_delete = [x[0] for x in q]

    n = len(ids_to_delete)
    if n:
      print(f'retracting {n} bf-classified results with msg "{like[:40]}"')
      s.query(Classification).filter(
        Classification.id.in_(ids_to_delete)
      ).delete(synchronize_session=False)

  prune_stderr_like("use of type 'double' requires cl_khr_fp64 extension")
  prune_stderr_like("implicit declaration of function")
  prune_stderr_like(
    "function cannot have argument whose type is, or contains, type size_t"
  )
  prune_stderr_like("unresolved extern function")
  prune_stderr_like("error: cannot increment value of type%")
  prune_stderr_like("subscripted access is not allowed for OpenCL vectors")
  prune_stderr_like("Images are not supported on given device")
  prune_stderr_like("error: variables in function scope cannot be declared")
  prune_stderr_like("error: implicit conversion ")
  prune_stderr_like("Could not find a definition ")

  def testcases_to_verify(session: session_t) -> query_t:
    q = (
      session.query(Result.testcase_id)
      .join(Classification)
      .join(Testbed)
      .join(Platform)
      .filter(
        Classification.classification == Classifications.ABF,
        Platform.opencl == "1.2",
      )
      .distinct()
    )
    return session.query(Testcase).filter(Testcase.id.in_(q)).distinct()

  class Worker(threading.Thread):
    """ worker thread to run testcases asynchronously """

    def __init__(self):
      self.ndone = 0
      super(Worker, self).__init__()

    def run(self):
      """ main loop"""
      with Session() as s:
        for testcase in testcases_to_verify(s):
          self.ndone += 1
          verify_opencl_version(s, testcase)
        s.commit()

  # Verify results
  print("Verifying abf-classified testcases ...")
  ntodo = testcases_to_verify(s).count()
  bar = progressbar.ProgressBar(
    initial_value=0, max_value=ntodo, redirect_stdout=True
  )
  worker = Worker()
  worker.start()
  while worker.is_alive():
    bar.update(min(worker.ndone, ntodo))
    worker.join(0.5)


def prune_arc_classifications(s: session_t) -> None:
  def prune_stderr_like(like):
    q = (
      s.query(Result.id)
      .join(Classification)
      .join(Stderr)
      .filter(
        Classification.classification == Classifications.ARC,
        Stderr.stderr.like(f"%{like}%"),
      )
    )
    ids_to_delete = [x[0] for x in q]

    n = len(ids_to_delete)
    if n:
      print(f"retracting {n} arc classified results with msg {like[:30]}")
      s.query(Classification).filter(
        Classification.id.in_(ids_to_delete)
      ).delete(synchronize_session=False)

  prune_stderr_like("clFinish CL_INVALID_COMMAND_QUEUE")

  def testcases_to_verify(session: session_t) -> query_t:
    q = (
      session.query(Result.testcase_id)
      .join(Classification)
      .filter(Classification.classification == Classifications.ARC)
      .distinct()
    )
    return session.query(Testcase).filter(Testcase.id.in_(q)).distinct()

  class Worker(threading.Thread):
    """ worker thread to run testcases asynchronously """

    def __init__(self):
      self.ndone = 0
      super(Worker, self).__init__()

    def run(self):
      """ main loop"""
      with Session() as s:
        for testcase in testcases_to_verify(s):
          self.ndone += 1
          if not testcase.verify_arc(s):
            testcase.retract_classifications(s, Classifications.ARC)
        s.commit()

  # Verify testcases
  print("Verifying arc-classified testcases ...")
  ntodo = testcases_to_verify(s).count()
  bar = progressbar.ProgressBar(
    initial_value=0, max_value=ntodo, redirect_stdout=True
  )
  worker = Worker()
  worker.start()
  while worker.is_alive():
    bar.update(min(worker.ndone, ntodo))
    worker.join(0.5)

  s.commit()
