#!/usr/bin/env python
from argparse import ArgumentParser
from progressbar import ProgressBar

import util
import db
from db import *


def results_iter(session, tables: Tableset, testbed: Testbed, no_opt: bool,
                 count=False):
    optimizations = not no_opt
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    done = session.query(tables.meta.id)

    if count:
        retvalue = sql.func.count(tables.results.id),
    else:
        retvalue = (
            tables.results.id,
            tables.results.program_id,
            tables.results.params_id,
            tables.results.runtime + tables.programs.runtime
        )

    q = session.query(*retvalue)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.results.params_id.in_(param_ids),
                ~tables.results.id.in_(done))\
        .join(tables.programs)\

    q = q.order_by(tables.results.date)

    return q


def set_metas(session: session_t, tables: Tableset, testbed: Testbed, no_opt: bool):
    devname = util.device_str(testbed.device)
    print(f"{tables.name} Metas for {devname} {no_opt} ...")

    optimizations = not no_opt
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == optimizations)

    last_meta = session.query(tables.meta)\
        .join(tables.results)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.results.params_id.in_(param_ids))\
        .order_by(tables.results.date.desc()).first()

    if last_meta:
        cumtime = last_meta.cumtime
    else:
        cumtime = 0

    todo = results_iter(session, tables, testbed, no_opt, count=True).scalar()

    # check if there's something to do
    if not todo:
        return

    # iterate over results without meta entries
    results = results_iter(session, tables, testbed, no_opt).all()
    for i, (result_id, program_id, params_id, total_time) in enumerate(ProgressBar(max_value=todo)(results)):
        # Add harness generation time, if applicable
        if tables.harnesses:
            total_time += session.query(tables.harnesses.generation_time)\
                .filter(tables.harnesses.program_id == program_id,
                        tables.harnesses.params_id == params_id).first()[0]
        cumtime += total_time

        # Create meta entry
        m = tables.meta(id=result_id, total_time=total_time, cumtime=cumtime)
        session.add(m)
        if not i % 1000:
            session.commit()
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
