#!/usr/bin/env python
import sqlalchemy as sql
import sys
from argparse import ArgumentParser
from labm8 import crypto
from progressbar import ProgressBar

import db
from db import *


def set_results_outcomes(s: session_t) -> None:
    """
    Determine the `outcome` column of results which have been marked with the
    "todo" outcome value.
    """
    total_todo = s.query(sql.sql.func.count(Result.id))\
        .filter(Result.outcome == Outcomes.TODO).scalar()

    if not total_todo:
        return

    for harness in [Harnesses.CL_LAUNCHER, Harnesses.DSMITH]:
        result_t = Harnesses.result_t(harness)

        # Determine if there are any results for which we haven't yet determined
        # the outcome.
        num_todo = s.query(sql.sql.func.count(Result.id))\
            .join(Testcase)\
            .filter(Testcase.harness == harness,
                    Result.outcome == Outcomes.TODO).scalar()
        if not num_todo:
            continue

        print("determining outcomes of", Harnesses.to_str(harness), "results ...")
        q = s.query(Result, Testcase.timeout, Stderr.stderr)\
            .join(Testcase)\
            .join(Stderr)\
            .filter(Testcase.harness == harness,
                    Result.outcome == Outcomes.TODO)

        for i, (result, timeout, stderr) in enumerate(ProgressBar(max_value=q.count())(q)):
            result.outcome = result_t.get_outcome(
                result.returncode, stderr, result.runtime, timeout)

        s.commit()

        # Keep going if there are still more harnesses to process.
        total_todo -= num_todo
        if not total_todo:
            continue


def set_stderr_assertions(s: session_t) -> None:
    stderrs = s.query(Stderr)\
        .join(Result)\
        .filter(Result.returncode != 0,
                Stderr.assertion_id == None,
                Stderr.stderr.like("%assertion%"))\
        .distinct()\
        .all()

    if not len(stderrs):
        return

    print("extracting compiler assertions ...")
    for stderr in ProgressBar()(stderrs):
        if not stderr.get_assertion(s):
            raise LookupError("no assertion found in stderr #{stderr.id}")
    s.commit()


def set_stderr_unreachables(s: session_t) -> None:
    stderrs = s.query(Stderr)\
        .join(Result)\
        .filter(Result.returncode != 0,
                Stderr.unreachable_id == None,
                Stderr.stderr.like("%unreachable%"))\
        .distinct()\
        .all()

    if not len(stderrs):
        return

    print("extracting unreachables ...")
    for stderr in ProgressBar()(stderrs):
        if not stderr.get_unreachable(s):
            raise LookupError("no unreachable found in stderr #{stderr.id}")
    s.commit()


def set_stderr_stackdumps(s: session_t) -> None:
    stderrs = s.query(Stderr)\
        .join(Result)\
        .filter(Result.returncode != 0,
                Stderr.stackdump_id == None,
                Stderr.stderr.like("%stack dump%"))\
        .distinct()\
        .all()

    if not len(stderrs):
        return

    print("extracting stack dumps ...")
    for stderr in ProgressBar()(stderrs):
        if not stderr.get_stackdump(s):
            raise LookupError("no stackdump found in stderr #{stderr.id}")
    s.commit()


def set_results_metas(s: session_t):
    """
    Create total time and cumulative time for each test case evaluated on each
    testbed using each harness.
    """
    num_results = s.query(sql.sql.func.count(Result.id)).scalar()
    num_metas = s.query(sql.sql.func.count(ResultMeta.id)).scalar()

    if num_results == num_metas:
        return

    print("creating results metas ...")
    s.execute(f"DELETE FROM {ResultMeta.__tablename__}")
    testbeds_harnesses = s.query(Result.testbed_id, Testcase.harness)\
        .join(Testcase)\
        .group_by(Result.testbed_id, Testcase.harness)\
        .order_by(Testcase.harness, Result.testbed_id).all()

    for testbed_id, harness in ProgressBar()(testbeds_harnesses):
        testbed = s.query(Testbed).filter(Testbed.id == testbed_id).scalar()

        s.execute(f"""
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
ORDER BY programs.date, testcases.threads_id""")
        s.commit()


def set_majorities(s: session_t) -> None:
    """
    Majority vote on testcase outcomes and outputs.
    """
    # We require at least this many results in order for there to be a majority:
    min_results_for_majority = 3

    print(f"voting on test case majorities ...")
    s.execute(f"DELETE FROM {Majority.__tablename__}")

    # Note we have to insert ignore here because there may be ties in the
    # majority outcome or output. E.g. there could be a test case with an even
    # split of 5 '1' outcomes and 5 '3' outcomes. Since there is only a single
    # majority outcome, we order results by outcome number, so that '1' (build
    # failure) will over-rule '6' (pass).
    s.execute(f"""
INSERT IGNORE INTO {Majority.__tablename__}
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
""")
    s.commit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    args = parser.parse_args()

    db_hostname = args.hostname
    print("connected to", db.init(db_hostname))

    with Session() as s:
        set_results_outcomes(s)
        set_stderr_assertions(s)
        set_stderr_unreachables(s)
        set_stderr_stackdumps(s)
        set_results_metas(s)
        set_majorities(s)
    print("done.")
