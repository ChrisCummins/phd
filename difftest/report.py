#!/usr/bin/env python
"""
Generate bug reports.
"""
import sqlalchemy as sql
import cldrive
import sys

from argparse import ArgumentParser
from labm8 import fs, crypto

import analyze
import db
import util
from db import *


def comment(*msg, prefix=''):
    return '\n'.join(f'// {prefix}{line}' for line in " ".join(msg).strip().split('\n'))


def get_bug_report(session: session_t, tables: Tableset, result_id: int, report_type: str="bf"):
    with Session(commit=False) as s:
        result = s.query(tables.results).filter(tables.results.id == result_id).first()

        if not result:
            raise KeyError(f"no result with ID {result_id}")

        # generate bug report
        now = datetime.datetime.utcnow().isoformat()
        report_id = crypto.md5_str(tables.name) + "-" + str(result.id)
        bug_type = {
            "bf": "compilation failure",
            "bto": "compiler hangs",
            "bc": "compiler crash",
            "c": "runtime crash",
            "w": "wrong-code",
        }[report_type]

        header = f"""\
// {bug_type} bug report {report_id}.c
//
// Metadata:
//   OpenCL platform:        {result.testbed.platform}
//   OpenCL device:          {result.testbed.device}
//   Driver version:         {result.testbed.driver}
//   OpenCL version:         {result.testbed.opencl}
//   Host operating system:  {result.testbed.host}
//   OpenCL optimizations:   {result.params.optimizations_on_off}
"""
        if report_type == "bf" or report_type == "bc":
            result_output = comment(result.stderr, prefix='  ')
            header += f"""\
//
// Output:
{result_output}
//   [Return code {result.status}]
//
"""
        elif report_type == "w":
            stderr = comment(result.stderr, prefix='  ')
            result_output = comment(result.stdout, prefix='  ')
            majority_output = comment(analyze.get_majority_output(session, tables, result), prefix='  ')
            assert majority_output != result_output
            header += f"""\
//
// Expected output:
{majority_output}
// Actual output:
{result_output}
//
// stderr:
{stderr}
//
"""
        elif report_type == "c":
            stdout = comment(result.stderr, prefix='  ')
            stderr = comment(result.stderr, prefix='  ')
            header += f"""\
//
// stdout:
{stdout}
//
// stderr:
{stderr}
//   [Return code {result.status}]
//
"""

        if isinstance(result.program, CLgenProgram):
            src = s.query(CLgenHarness).filter(
                CLgenHarness.program_id == result.program.id,
                CLgenHarness.params_id == result.params.id).first().src
        else:
            src = result.program.src
        return (header + src).strip()


def generate_bc_reports(tables, time_limit):
    outbox = fs.path(f"../data/bug-reports/{tables.name}/bc")
    fs.mkdir(outbox)
    with Session(commit=True) as s:
        q = s.query(tables.results.id)\
            .join(tables.meta)\
            .filter(tables.results.outcome == "bc")

        if time_limit > 0:
            q = q.filter(tables.meta.cumtime < time_limit)

        dupes = 0
        errs = set()
        for result_id, in q:
            result = s.query(tables.results)\
                .filter(tables.results.id == result_id)\
                .first()

            key = result.testbed_id, result.program_id
            if key in errs:
                dupes += 1
                continue
            errs.add(key)

            vendor = util.vendor_str(result.testbed.platform)
            outpath = fs.path(outbox, f"bug-report-{vendor}-{result.id}.c")

            if not fs.exists(outpath):
                report = get_bug_report(**{
                    "session": s,
                    "tables": tables,
                    "result_id": result.id,
                    "report_type": "bc",
                })
                with open(outpath, "w") as outfile:
                    print(report, file=outfile)
                print(outpath)
    print(f"{dupes} duplicate {tables.name} bc results flagged")


def generate_reports(tables, time_limit, type_field, type_name):
    outbox = fs.path(f"../data/bug-reports/{tables.name}/{type_name}")
    fs.mkdir(outbox)
    with Session(commit=True) as s:
        q = s.query(tables.results.id)\
            .join(tables.meta)\
            .filter(type_field == type_name)

        if time_limit > 0:
            q = q.filter(tables.meta.cumtime < time_limit)

        dupes = 0
        errs = set()
        for result_id, in q:
            result = s.query(tables.results)\
                .filter(tables.results.id == result_id)\
                .first()

            key = result.testbed_id, result.program_id
            if key in errs:
                dupes += 1
                continue
            errs.add(key)

            vendor = util.vendor_str(result.testbed.platform)
            outpath = fs.path(outbox, f"bug-report-{vendor}-{result.id}.c")

            if not fs.exists(outpath):
                report = get_bug_report(**{
                    "session": s,
                    "tables": tables,
                    "result_id": result.id,
                    "report_type": type_name,
                })
                with open(outpath, "w") as outfile:
                    print(report, file=outfile)
                print(outpath)
    print(f"{dupes} duplicate {tables.name} {type_name} results flagged")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("-t", "--time-limit", type=int, default=48,
                        help="Number of hours to limit results to (default: 48)")
    parser.add_argument("--bf", action="store_true")
    parser.add_argument("--w", action="store_true")
    parser.add_argument("--bc", action="store_true")
    parser.add_argument("--bto", action="store_true")
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--to", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--clsmith", action="store_true")
    parser.add_argument("--clgen", action="store_true")
    args = parser.parse_args()

    # get testbed information
    db_hostname = args.hostname
    db_url = db.init(db_hostname)

    time_limit = args.time_limit * 3600

    tablesets = []
    if not args.clgen:
        tablesets.append(CLSMITH_TABLES)
    if not args.clsmith:
        tablesets.append(CLGEN_TABLES)

    try:
        for tables in tablesets:
            if args.all or args.bc:
                generate_reports(tables, time_limit, tables.results.outcome, "bc")
            if args.all or args.bto:
                generate_reports(tables, time_limit, tables.results.outcome, "bto")
            if args.all or args.w:
                generate_reports(tables, time_limit, tables.results.classification, "w")
    except KeyboardInterrupt:
        print("stop")


if __name__ == "__main__":
    main()
