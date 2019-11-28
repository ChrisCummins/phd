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
Differential test soldity results.
"""
from experimental.dsmith.sol.db import *


def difftest():
  with Session() as s:
    create_results_metas(s)
    create_majorities(s)
    create_classifications(s)


def create_results_metas(s: session_t):
  """
  Create total time and cumulative time for each test case evaluated on each
  testbed using each harness.
  """
  # break early if we can
  num_results = s.query(func.count(Result.id)).scalar()
  num_metas = s.query(func.count(ResultMeta.id)).scalar()
  if num_results == num_metas:
    return

  print("creating results metas ...")
  s.execute(f"DELETE FROM {ResultMeta.__tablename__}")
  app.Log(2, "deleted existing result metas")
  testbeds_harnesses = (
    s.query(Result.testbed_id, Testcase.harness)
    .join(Testcase)
    .group_by(Result.testbed_id, Testcase.harness)
    .order_by(Testcase.harness, Result.testbed_id)
    .all()
  )

  bar = progressbar.ProgressBar(redirect_stdout=True)
  for testbed_id, harness in bar(testbeds_harnesses):
    # FIXME: @cumtime variable is not supported by SQLite.
    s.execute(
      sql_query(
        f"""
INSERT INTO {ResultMeta.__tablename__} (id, total_time, cumtime)
SELECT  results.id,
        results.runtime + programs.generation_time AS total_time,
        @cumtime := @cumtime + results.runtime + programs.generation_time AS cumtime
FROM {Result.__tablename__} results
INNER JOIN {Testcase.__tablename__} testcases ON results.testcase_id = testcases.id
INNER JOIN {Program.__tablename__} programs ON testcases.program_id = programs.id
JOIN (SELECT @cumtime := 0) r
WHERE results.testbed_id = {testbed_id}
AND testcases.harness = {harness}
ORDER BY programs.date"""
      )
    )
    s.commit()


def create_majorities(s: session_t) -> None:
  """
  Majority vote on testcase outcomes and outputs.
  """
  # We require at least this many results in order for there to be a majority:
  min_results_for_majority = 1

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
    (id, num_results, maj_outcome, outcome_majsize, maj_stderr_id, stderr_majsize)
SELECT  result_counts.testcase_id,
        result_counts.num_results,
        outcome_majs.maj_outcome,
        outcome_majs.outcome_majsize,
        stderr_majs.maj_stderr_id,
        stderr_majs.stderr_majsize
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
    SELECT l.testcase_id, s.stderr_id as maj_stderr_id, s.stderr_count AS stderr_majsize
    FROM (
        SELECT testcase_id, MAX(stderr_count) as max_count
        FROM (
            SELECT testcase_id, COUNT(*) as stderr_count
            FROM {Result.__tablename__}
            GROUP BY testcase_id, stderr_id
        ) r
        GROUP BY testcase_id
    ) l
    INNER JOIN (
        SELECT testcase_id, stderr_id, COUNT(*) as stderr_count
        FROM {Result.__tablename__}
        GROUP BY testcase_id, stderr_id
    ) s ON l.testcase_id = s.testcase_id AND l.max_count = s.stderr_count
) stderr_majs ON outcome_majs.testcase_id = stderr_majs.testcase_id
ORDER BY outcome_majs.maj_outcome DESC
"""
  )
  s.commit()


def create_classifications(s: session_t) -> None:
  """
  Determine anomalous results.
  """
  s.execute(f"DELETE FROM {Classification.__tablename__}")

  # We require at least this many results in order to vote on the majority:
  min_majsize = 1

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
