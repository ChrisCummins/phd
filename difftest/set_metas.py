#!/usr/bin/env python
from argparse import ArgumentParser
from progressbar import ProgressBar

import util
import db
from db import *


def results_iter(session, tables: Tableset, testbed: Testbed, no_opt: bool):
    param_ids = session.query(tables.params.id)\
        .filter(tables.params.optimizations == no_opt)

    done = session.query(tables.meta.id)

    q = session.query(tables.results)\
        .filter(tables.results.testbed_id == testbed.id,
                tables.results.params_id.in_(param_ids),
                ~tables.results.id.in_(done))\
        .order_by(tables.results.date)

    return q


def set_metas(session: session_t, tables: Tableset, testbed: Testbed):
    devname = util.device_str(testbed.device)

    for noopt in [True, False]:
        print(f"{tables.name} Metas for {devname} {noopt} ...")
        results = results_iter(session, tables, testbed, noopt)
        for i, result in enumerate(ProgressBar(max_value=results.count())(results)):
            result.get_meta(session)
            if not i % 100:
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
                set_metas(s, CLSMITH_TABLES, testbed)
            if not args.clsmith:
                set_metas(s, CLGEN_TABLES, testbed)
