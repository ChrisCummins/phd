#!/usr/bin/env python

import clgen
import math
import sys
import sqlalchemy as sql

from argparse import ArgumentParser
from collections import Counter
from signal import Signals
from progressbar import ProgressBar

import db
import oclgrind
import util
from db import *


def get_majority(lst):
    """ get the majority value of the list elements, and the majority count """
    return Counter(lst).most_common(1)[0]


def get_majority_output(session, tables: Tableset, result):
    q = session.query(tables.results.stdout)\
        .filter(tables.results.program_id == result.program.id,
                tables.results.params_id == result.params.id)
    outputs = [r[0] for r in q]
    majority_output, majority_count = get_majority(outputs)
    return majority_output, majority_count, len(outputs)


def get_cl_launcher_outcome(result) -> None:
    """
    Given a cl_launcher result, determine and set it's outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return "c" if "Compilation terminated successfully..." in result.stderr else "bf"
    def crash_or_build_crash():
        return "c" if "Compilation terminated successfully..." in result.stderr else "bc"
    def timeout_or_build_timeout():
        return "to" if "Compilation terminated successfully..." in result.stderr else "bto"

    if result.status == 0:
        return "pass"
    # 139 is SIGSEV
    elif result.status == 139 or result.status == -11:
        result.status = 139
        return crash_or_build_crash()
    # Preproccessor or Unicode error
    elif result.status == 1024 or result.status == 1025:
        return "fail"
    # SIGTRAP
    elif result.status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif result.status == -9 and result.runtime >= 60:
        return timeout_or_build_timeout()
    elif result.status == -9:
        print(f"SIGKILL, but only ran for {result.runtime:.2f}s")
        return crash_or_build_crash()
    # SIGILL
    elif result.status == -4:
        return crash_or_build_crash()
    # SIGABRT
    elif result.status == -6:
        return crash_or_build_crash()
    # SIGFPE
    elif result.status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif result.status == -7:
        return crash_or_build_crash()
    # cl_launcher error
    elif result.status == 1:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-result.status).name)
        except ValueError:
            print(result.status)
        raise LookupError(f"failed to determine outcome of cl_launcher result #{result.id}")


def set_cl_launcher_outcomes(session, tables: Tableset, rerun: bool=False) -> None:
    """ Set all cl_launcher outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLSmith outcomes ...")
    q = session.query(tables.results)
    if not rerun:
        q = q.filter(tables.results.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cl_launcher outcomes')(q, max_value=ntodo):
        result.outcome = get_cl_launcher_outcome(result)


def get_cldrive_outcome(result):
    """
    Given a cldrive result, determine its outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return "c" if "[cldrive] Kernel: " in result.stderr else "bf"
    def crash_or_build_crash():
        return "c" if "[cldrive] Kernel: " in result.stderr else "bc"
    def timeout_or_build_timeout():
        return "to" if "[cldrive] Kernel: " in result.stderr else "bto"

    if result.status == 0:
        return "pass"
    # 401 is bad harness
    elif result.status == 401:
        return "fail"
    # 139 is SIGSEV
    elif result.status == 139 or result.status == -11:
        result.status = 139
        return crash_or_build_crash()
    # SIGTRAP
    elif result.status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif result.status == -9 and result.runtime >= 60:
        return timeout_or_build_timeout()
    elif result.status == -9:
        print(f"SIGKILL, but only ran for {result.runtime:.2f}s")
        return crash_or_build_crash()
    # SIGILL
    elif result.status == -4:
        return crash_or_build_crash()
    # SIGFPE
    elif result.status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif result.status == -7:
        return crash_or_build_crash()
    # SIGABRT
    elif result.status == -6:
        return crash_or_build_crash()
    # cl_launcher error
    elif result.status == 1:
        if 'cldrive.driver.Timeout: 60' in result.stderr:
            return timeout_or_build_timeout()
        else:
            return crash_or_build_failure()
    # file not found (check the stderr on this one):
    elif result.status == 127:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-result.status).name)
        except ValueError:
            print(result.status)
        raise LookupError(f"failed to determine outcome of cldrive result #{result.id}")


def set_cldrive_outcomes(session, tables: Tableset, rerun: bool=False) -> None:
    """ Set all cldrive outcomes. Set `rerun' to recompute outcomes for all results """
    print("Determining CLgen outcomes ...")
    q = session.query(tables.results)
    if not rerun:
        q = q.filter(tables.results.outcome == None)
    ntodo = q.count()
    for result in util.NamedProgressBar('cldrive outcomes')(q, max_value=ntodo):
        result.outcome = get_cldrive_outcome(result)


def set_classifications(session, tables: Tableset) -> None:
    """
    Apply voting heuristics to expose anomalous results.
    """
    # Check that there's something to do:
    num_results = session.query(sql.sql.func.count(tables.results.id)).scalar()
    num_classifications = session.query(sql.sql.func.count(tables.classifications.id)).scalar()

    # Nothing to do
    if num_classifications == num_results:
        return

    print(f"Resetting {tables.name} classifications ...")
    session.execute(f"DELETE FROM {tables.classifications.__tablename__}")
    session.commit()

    min_majority_outcome = 6
    print(f"Classifying testcases with majority outcomes smaller than {min_majority_outcome} as passes ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__} (id, classification)
SELECT id, {CLASSIFICATIONS_TO_INT['pass']}
FROM {tables.results.__tablename__}
WHERE testcase_id IN
    (SELECT testcase_id
     FROM (
        SELECT testcase_id, COUNT(*) as majority_outcome
        FROM {tables.results.__tablename__}
        WHERE outcome <> {OUTCOMES_TO_INT['bc']}
        GROUP BY testcase_id) r
     WHERE majority_outcome < {min_majority_outcome})
""")

    session.commit()
    # Go testcase-by-testcase
    testcases = session.query(tables.testcases)\
        .filter(tables.testcases.id.in_(
            session.query(tables.results.testcase_id)\
                .outerjoin(tables.classifications)\
                .filter(tables.classifications.id == None))).all()

    # We count compiler crashes and timeouts as passes, since there's no way of
    # voting on crashes/timeouts:
    print("Classifying {bc,bto}-outcomes as passes ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__} (id, classification)
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT['pass']}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.classifications.__tablename__} ON {tables.results.__tablename__}.id = {tables.classifications.__tablename__}.id
WHERE {tables.classifications.__tablename__}.id IS NULL
AND outcome IN ({OUTCOMES_TO_INT['bc']}, {OUTCOMES_TO_INT['bto']})
""")
    session.commit()

    to_add = []

    print("Applying voting heuristics ...")
    for i, testcase in enumerate(ProgressBar()(testcases)):

        if i and not i % 1000:
            session.bulk_save_objects(to_add)
            to_add = []
            session.commit()

        testcase_id = testcase.id

        results = session.query(tables.results)\
            .outerjoin(tables.classifications)\
            .filter(tables.results.testcase_id == testcase_id,
                    tables.classifications.id == None).all()
        n = len(results)

        if n < min_majority_outcome:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT['pass'])
                       for r in results]
            continue

        # Determine majority outcome:
        min_majority_count = math.ceil(n / 2)
        majority_outcome, majority_count = get_majority([r.outcome for r in results])

        # If majority outcome resulted in binaries, mark anomalous build
        # failures:
        minority_outcomes = set([OUTCOMES_TO_INT["bf"], OUTCOMES_TO_INT["c"], OUTCOMES_TO_INT["to"]])
        if majority_outcome not in minority_outcomes:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT[OUTCOMES[r.outcome]])
                       for r in results if r.outcome in minority_outcomes]
            results[:] = [r for r in results if r.outcome not in minority_outcomes]

        # If the majority did not produce outputs, then we're done:
        if majority_outcome != OUTCOMES_TO_INT["pass"]:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT['pass'])
                       for r in results]
            continue

        # Look for wrong-code bugs:
        majority_output, output_majority_count = get_majority([r.stdout_id for r in results])

        # Ensure that the majority of configurations agree on the output:
        min_output_majority_count = math.ceil(len(results) / 2)
        # min_output_majority_count = len(results) - 1
        if output_majority_count == len(results):
            # Everyone agreed on the output:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"])
                       for r in results]
        elif output_majority_count < min_output_majority_count:
            # No majority:
            print("skipping output_majority_count <", min_output_majority_count, " = ", output_majority_count)
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"])
                       for r in results]
        else:
            # At least one result disagreed:
            to_add += [tables.classifications(id=r.id, classification=CLASSIFICATIONS_TO_INT["pass"] if r.stdout_id == majority_output else CLASSIFICATIONS_TO_INT["w"])
                       for r in results]
    session.commit()


def verify_testcase(session: session_t, tables: Tableset, testcase) -> None:
    """
    Verify

    This is time consuming. It should only be run on supicious testcases.
    """

    def fail():
        ids_to_update = [
            x[0] for x in
            session.query(tables.results.id)\
                .join(tables.classifications)\
                .filter(tables.results.testcase_id == testcase.id,
                        tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"]).all()
        ]
        n = len(ids_to_update)
        assert n > 0
        ids_str = ",".join(str(x) for x in ids_to_update)
        print(f"retracting w-classification on {n} results: {ids_str}")
        session.query(tables.classifications)\
            .filter(tables.classifications.id.in_(ids_to_update))\
            .update({"classification": CLASSIFICATIONS_TO_INT["pass"]},
                    synchronize_session=False)
        session.commit()

    # CLgen-specific analysis. We can omit these checks for CLSmith, as they
    # will always pass.
    if tables.name == "CLgen":

        # Check for red-flag compiler warnings. We can't do this for CLSmith
        # because cl_launcher doesn't print build logs.
        if testcase.compiler_warnings == None:
            stderrs = [
                x[0] for x in
                session.query(tables.stderrs.stderr)\
                    .join(tables.results)\
                    .filter(tables.results.testcase_id == testcase.id)
            ]
            # print("checking", len(stderrs), "stderrs. first:", stderrs[0])
            testcase.compiler_warnings = False
            for stderr in stderrs:
                if "incompatible pointer to integer conversion" in stderr:
                    print(f"testcase {testcase.id}: incompatible pointer to to integer conversion")
                    testcase.compiler_warnings = True
                    break
                elif "ordered comparison between pointer and integer" in stderr:
                    print(f"testcase {testcase.id}: ordered comparison between pointer and integer")
                    testcase.compiler_warnings = True
                    break
                elif "warning: incompatible" in stderr:
                    print(f"testcase {testcase.id}: incompatible warning")
                    testcase.compiler_warnings = True
                    break
                elif "warning: division by zero is undefined" in stderr:
                    print(f"testcase {testcase.id}: division by zero is undefined")
                    testcase.compiler_warnings = True
                    break
                elif "warning: comparison of distinct pointer types" in stderr:
                    print(f"testcase {testcase.id}: comparison of distinct pointer types")
                    testcase.compiler_warnings = True
                    break
                elif "is past the end of the array" in stderr:
                    print(f"testcase {testcase.id}: is past the end of the array")
                    testcase.compiler_warnings = True
                    break
                elif "warning: comparison between pointer and" in stderr:
                    print(f"testcase {testcase.id}: comparison between pointer and")
                    testcase.compiler_warnings = True
                    break
                elif "warning" in stderr:
                    print("\n UNRECOGNIZED WARNINGS in testcase {testcase.id}:")
                    print("\n".join(f">> {line}" for line in stderr.split("\n")))


        if testcase.compiler_warnings:
            print(f"testcase {testcase.id}: redflag compiler warnings")
            return fail()

        # Determine if
        if testcase.contains_floats == None:
            testcase.contains_floats = "float" in testcase.program.src

        if testcase.contains_floats:
            print(f"testcase {testcase.id}: contains floats")
            return fail()

        # Run GPUverify on kernel
        if testcase.gpuverified == None:
            try:
                clgen.gpuverify(testcase.program.src, ["--local_size=64", "--num_groups=128"])
                testcase.gpuverified = 1
            except clgen.GPUVerifyException:
                testcase.gpuverified = 0

        if not testcase.gpuverified:
            print(f"testcase {testcase.id}: failed GPUVerify check")
            return fail()

    # Check that program runs with Oclgrind without error:
    if testcase.oclverified == None:
        if tables.name == "CLSmith":
            testcase.oclverified = oclgrind.oclgrind_verify_clsmith(testcase)
        else:
            testcase.oclverified = oclgrind.oclgrind_verify_clgen(testcase)

    if not testcase.oclverified:
        print(f"testcase {testcase.id}: failed OCLgrind verification")
        return fail()

    session.commit()


def verify_w_classifications(session: session_t, tables: Tableset) -> None:
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
                            .filter(tables.testcases.id.in_(q))\
                            .distinct().all()

    for testcase in ProgressBar()(testcases_to_verify):
        verify_testcase(session, tables, testcase)


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--clsmith", action="store_true",
                        help="analyze only clsmith results")
    parser.add_argument("--clgen", action="store_true",
                        help="analyze only clgen results")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("-t", "--time-limit", type=int, default=48,
                        help="time limit in hours (default: 48)")
    args = parser.parse_args()

    tables = []
    if not args.clgen:
        tables.append(CLSMITH_TABLES)
    if not args.clsmith:
        tables.append(CLGEN_TABLES)

    # Connect to database
    db_hostname = args.hostname
    print("connected to", db.init(db_hostname))

    with Session(commit=True) as s:
        for tableset in tables:
            if args.prune:
                verify_w_classifications(s, tableset)
            else:
                set_classifications(s, tableset)
