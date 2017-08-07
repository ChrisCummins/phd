#!/usr/bin/env python
import sqlalchemy as sql
import sys
from argparse import ArgumentParser
from labm8 import crypto
from progressbar import ProgressBar

import util
import db
from db import *


def set_metas_for_device(session: session_t, tables: Tableset, testbed: Testbed, optimizations: int):
    devname = util.device_str(testbed.device)
    print(f"{tables.name} Metas for {devname}", "opt" if optimizations else "no-opt", "...")

    # Check if there's anything to do:
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    if tables.name == "CLSmith":
        session.execute(f"""
    INSERT INTO {tables.meta.__tablename__} (id, total_time, cumtime)
    SELECT CLSmithResults.id,
           CLSmithResults.runtime + CLSmithPrograms.runtime AS total_time,
           @cumtime := @cumtime + CLSmithResults.runtime + CLSmithPrograms.runtime AS cumtime
       FROM CLSmithResults
       LEFT JOIN CLSmithTestCases ON CLSmithResults.testcase_id=CLSmithTestCases.id
       LEFT JOIN CLSmithPrograms ON CLSmithTestCases.program_id=CLSmithPrograms.id
       JOIN (SELECT @cumtime := 0) r
            WHERE CLSmithResults.testbed_id={testbed.id}
            AND CLSmithTestCases.params_id IN (SELECT id FROM cl_launcherParams WHERE optimizations = {optimizations})
    ORDER BY CLSmithResults.date""")
    else:
        # CLgen has harness creation time too
        session.execute(f"""
    INSERT INTO {tables.meta.__tablename__} (id, total_time, cumtime)
    SELECT CLgenResults.id,
           CLgenResults.runtime + CLgenPrograms.runtime + CLgenHarnesses.generation_time AS total_time,
           @cumtime := @cumtime + CLgenResults.runtime + CLgenPrograms.runtime + CLgenHarnesses.generation_time AS cumtime
       FROM CLgenResults
       LEFT JOIN CLgenTestCases ON CLgenResults.testcase_id=CLgenTestCases.id
       LEFT JOIN CLgenHarnesses ON CLgenResults.testcase_id=CLgenHarnesses.id
       LEFT JOIN CLgenPrograms ON CLgenTestCases.program_id=CLgenPrograms.id
       JOIN (SELECT @cumtime := 0) r
            WHERE CLgenResults.testbed_id={testbed.id}
            AND CLgenTestCases.params_id IN (SELECT id FROM cldriveParams WHERE optimizations = {optimizations})
    ORDER BY CLgenResults.date""")

    session.commit()


def set_metas(session:session_t, tables: Tableset):
    """
    """
    num_results = session.query(sql.sql.func.count(tables.results.id)).scalar()
    num_metas = session.query(sql.sql.func.count(tables.meta.id)).scalar()

    if num_results == num_metas:
        return

    # start from scratch
    print(f"Resetting {tables.name} metas ...")
    session.execute(f"DELETE FROM {tables.meta.__tablename__}")
    for testbed in session.query(Testbed):
        set_metas_for_device(session, tables, testbed, 0)
        set_metas_for_device(session, tables, testbed, 1)


def set_majorities(session, tables: Tableset) -> None:
    """
    Majority vote on testcase outcomes and outputs.
    """
    print(f"Resetting {tables.name} test case majorities ...")
    session.execute(f"DELETE FROM {tables.majorities.__tablename__}")

    print(f"Voting on {tables.name} test case majorities ...")
    # Note we have to insert ignore here because there may be ties in the
    # majority outcome or output. E.g. there could be a test case with an even
    # split of 5 '1' outcomes and 5 '3' outcomes. Since there is only a single
    # majority outcome, we order results by outcome number, so that '1' (build
    # failure) will over-rule '6' (pass).
    session.execute(f"""
INSERT IGNORE INTO {tables.majorities.__tablename__}
SELECT t1.testcase_id, t1.maj_outcome, t1.outcome_majsize, t2.maj_stdout_id, t2.stdout_majsize
FROM (
    SELECT l.testcase_id,s.outcome as maj_outcome,s.outcome_count AS outcome_majsize
    FROM (
        SELECT testcase_id,MAX(outcome_count) as max_count FROM (
            SELECT testcase_id,COUNT(*) as outcome_count
            FROM {tables.results.__tablename__}
            GROUP BY testcase_id, outcome
        ) r
        GROUP BY testcase_id
    ) l INNER JOIN (
        SELECT testcase_id,outcome,COUNT(*) as outcome_count
        FROM {tables.results.__tablename__}
        GROUP BY testcase_id, outcome
    ) s ON l.testcase_id = s.testcase_id AND l.max_count = s.outcome_count
) t1 JOIN (
    SELECT l.testcase_id, s.stdout_id as maj_stdout_id, s.stdout_count AS stdout_majsize
    FROM (
        SELECT testcase_id,MAX(stdout_count) as max_count FROM (
            SELECT testcase_id,COUNT(*) as stdout_count
            FROM {tables.results.__tablename__}
            GROUP BY testcase_id, stdout_id
        ) r
        GROUP BY testcase_id
    ) l INNER JOIN (
        SELECT testcase_id,stdout_id,COUNT(*) as stdout_count
        FROM {tables.results.__tablename__}
        GROUP BY testcase_id, stdout_id
    ) s ON l.testcase_id = s.testcase_id AND l.max_count = s.stdout_count
) t2 ON t1.testcase_id = t2.testcase_id
ORDER BY t1.maj_outcome DESC
""")
    session.commit()


def get_assertions(session: session_t, tables: Tableset) -> None:
    print(f"Recording {tables.name} compiler assertions ...")
    stderrs = session.query(tables.stderrs)\
        .join(tables.results)\
        .filter(tables.results.status != 0,
                tables.stderrs.stderr.like("%assertion%"))\
        .distinct()

    for stderr in ProgressBar(max_value=stderrs.count())(stderrs):
        assertion = util.get_assertion(session, tables.assertions, stderr.stderr,
                                       clang_assertion=False)
        stderr.assertion = assertion


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--clsmith", action="store_true",
                        help="Only run CLSmith test cases")
    parser.add_argument("--clgen", action="store_true",
                        help="Only run CLgen test cases")
    args = parser.parse_args()

    # Connect to database
    db_hostname = args.hostname
    print("connected to", db.init(db_hostname))

    tables = []
    if not args.clgen:
        tables.append(CLSMITH_TABLES)
    if not args.clsmith:
        tables.append(CLGEN_TABLES)

    with Session(commit=True) as s:
        for tableset in tables:
            set_metas(s, tableset)
            set_majorities(s, tableset)
            get_assertions(s, tableset)
