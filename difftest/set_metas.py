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
            tables.results.program_id,
            tables.results.params_id,
            tables.results.runtime + tables.programs.runtime
        )

    # Existing meta entries:
    done = session.query(tables.meta.id)

    q = session.query(*retvalue)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.results.params_id.in_(param_ids),
                ~tables.results.id.in_(done))\
        .join(tables.programs)\
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
        .filter(tables.results.testbed_id == testbed.id,
                tables.results.params_id.in_(param_ids))\
        .order_by(tables.results.date.desc()).first()

    return last_meta.cumtime if last_meta else 0


def set_metas(session: session_t, tables: Tableset, testbed: Testbed, no_opt: bool):
    devname = util.device_str(testbed.device)
    print(f"{tables.name} Metas for {devname}", "no-opt" if no_opt else "opt", "...")

    # Check if there's anything to do:
    todo = results_iter(session, tables, testbed, no_opt, count=True).scalar()
    if not todo:
        return

    # Get current elapsed cumulative time:
    cumtime = current_cumtime(session, tables, testbed, no_opt)

    # Iterate over results without meta entries:
    results = results_iter(session, tables, testbed, no_opt).all()
    for i, (result_id, program_id, params_id, total_time) in enumerate(ProgressBar()(results)):
        # Add harness generation time, if applicable:
        if tables.harnesses:
            total_time += session.query(tables.harnesses.generation_time)\
                .filter(tables.harnesses.program_id == program_id,
                        tables.harnesses.params_id == params_id).first()[0]
        cumtime += total_time

        # Create meta entry:
        m = tables.meta(id=result_id, total_time=total_time, cumtime=cumtime)
        session.add(m)
        if not i % 1000:
            session.commit()

    # Commit whatever's left over:
    session.commit()


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

    with Session(commit=True) as s:
        for testbed in s.query(Testbed):
            if not args.clgen:
                set_metas(s, CLSMITH_TABLES, testbed, True)
                set_metas(s, CLSMITH_TABLES, testbed, False)
            if not args.clsmith:
                set_metas(s, CLGEN_TABLES, testbed, True)
                set_metas(s, CLGEN_TABLES, testbed, False)
