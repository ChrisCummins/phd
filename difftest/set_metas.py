#!/usr/bin/env python
from argparse import ArgumentParser
from progressbar import ProgressBar

import util
import db
from db import *


def results_iter(session: session_t, tables: Tableset, testbed: Testbed,
                 no_opt: bool, count: bool=False):
    """
    Returns a query which iterates over results without a Meta table entry.

    Arguments:
        session (session_t): Database session.
        tables (Tableset): CLSmith / CLgen tableset.
        testbed (Testbed): Device to iterate over.
        no_opt (bool): Optimizations disabled?
        count (bool): If true, return count, not iterator.
    """
    optimizations = not no_opt
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    if count:
        retvalue = sql.func.count(tables.results.id),
    else:
        retvalue = (
            tables.results.id,
            tables.results.testcase_id,
            tables.results.runtime + tables.programs.runtime
        )

    # Existing meta entries:
    done = session.query(tables.meta.id)

    q = session.query(*retvalue)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.testcases.params_id.in_(param_ids),
                ~tables.results.id.in_(done))\
        .join(tables.testcases)\
        .order_by(tables.results.date)

    return q


def current_cumtime(session: session_t, tables: Tableset, testbed: Testbed,
                    no_opt: bool) -> float:
    """
    Get the current cumulative time of existing results on the device.
    """
    optimizations = not no_opt
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    last_meta = session.query(tables.meta)\
        .join(tables.results)\
        .join(tables.testcases)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.testcases.params_id.in_(param_ids))\
        .order_by(tables.results.date.desc()).first()

    return last_meta.cumtime if last_meta else 0


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

    # Commit whatever's left over:
    session.commit()

def set_metas(session:session_t, tables: Tableset):
    num_results = session.query(sql.sql.func.count(tables.results.id)).scalar()
    num_metas = session.query(sql.sql.func.count(tables.meta.id)).scalar()

    if num_results == num_metas:
        return

    # start from scratch
    print("Resetting {tables.name} metas ...")
    session.execute(f"DELETE FROM {tables.meta.__tablename__}")
    for testbed in session.query(Testbed):
        set_metas_for_device(session, tables, testbed, 0)
        set_metas_for_device(session, tables, testbed, 1)


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

    with Session() as s:
        if not args.clgen:
            set_metas(s, CLSMITH_TABLES)
        if not args.clsmith:
            set_metas(s, CLGEN_TABLES)
