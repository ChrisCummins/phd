#!/usr/bin/env python
from argparse import ArgumentParser

import analyze
import db
from db import *

if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("-t", "--time-limit", type=int, default=48,
                        help="time limit in hours (default: 48)")
    parser.add_argument("--no-commit", action="store_true")
    args = parser.parse_args()

    # Connect to database
    db_hostname = args.hostname
    print("connected to", db.init(db_hostname))

    with Session(commit=not args.no_commit) as s:
        if args.prune:
            analyze.prune_w_classifications(s, CLGEN_TABLES, args.time_limit * 3600)
        else:
            analyze.set_classifications(s, CLGEN_TABLES, rerun=True)
