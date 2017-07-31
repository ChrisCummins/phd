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
    # cldrive error
    elif result.status == 1 and result.runtime >= 60:
        return timeout_or_build_timeout()
    elif result.status == 1:
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
    Determine anomalous results.
    """
    print(f"Resetting {tables.name} classifications ...")
    session.execute(f"DELETE FROM {tables.classifications.__tablename__}")

    min_majsize = 7

    print(f"Determining {tables.name} wrong-code classifications ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT["w"]}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.testcases.__tablename__} ON {tables.results.__tablename__}.testcase_id={tables.testcases.__tablename__}.id
LEFT JOIN {tables.majorities.__tablename__} ON {tables.testcases.__tablename__}.id={tables.majorities.__tablename__}.id
WHERE outcome = {OUTCOMES_TO_INT["pass"]}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
AND outcome_majsize >= {min_majsize}
AND stdout_majsize >= CEILING(2 * outcome_majsize / 3)
AND stdout_id <> maj_stdout_id
AND oclverified = 1
""")

    print(f"Determining {tables.name} anomalous build-failures ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT["bf"]}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.testcases.__tablename__} ON {tables.results.__tablename__}.testcase_id={tables.testcases.__tablename__}.id
LEFT JOIN {tables.majorities.__tablename__} ON {tables.testcases.__tablename__}.id={tables.majorities.__tablename__}.id
WHERE outcome = {OUTCOMES_TO_INT["bf"]}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
""")

    print(f"Determining {tables.name} anomalous crashes ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT["c"]}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.testcases.__tablename__} ON {tables.results.__tablename__}.testcase_id={tables.testcases.__tablename__}.id
LEFT JOIN {tables.majorities.__tablename__} ON {tables.testcases.__tablename__}.id={tables.majorities.__tablename__}.id
WHERE outcome = {OUTCOMES_TO_INT["c"]}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
""")

    print(f"Determining {tables.name} anomalous timeouts ...")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT {tables.results.__tablename__}.id, {CLASSIFICATIONS_TO_INT["to"]}
FROM {tables.results.__tablename__}
LEFT JOIN {tables.majorities.__tablename__} ON {tables.results.__tablename__}.id={tables.majorities.__tablename__}.id
WHERE outcome = {OUTCOMES_TO_INT["to"]}
AND outcome_majsize = {min_majsize}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
""")
    session.commit()


def testcase_raises_compiler_warnings(session: session_t, tables: Tableset, testcase) -> bool:
    # Check for red-flag compiler warnings. We can't do this for CLSmith
    # because cl_launcher doesn't print build logs.
    if testcase.compiler_warnings == None:
        stderrs = [
            x[0] for x in
            session.query(tables.stderrs.stderr)\
                .join(tables.results)\
                .filter(tables.results.testcase_id == testcase.id)
        ]
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

    return testcase.compiler_warnings


def verify_w_testcase(session: session_t, tables: Tableset, testcase) -> None:
    """
    Verify that a testcase is sensible.

    On first run, this is time consuming, though results are cached for later
    re-use.
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
            .delete()

    # CLgen-specific analysis. We can omit these checks for CLSmith, as they
    # will always pass.
    if tables.name == "CLgen":
        if testcase_raises_compiler_warnings(session, tables, testcase):
            print(f"testcase {testcase.id}: redflag compiler warnings")
            return fail()

        # Determine if kernel contains floats
        if testcase.contains_floats == None:
            testcase.contains_floats = "float" in testcase.program.src

        if testcase.contains_floats:
            print(f"testcase {testcase.id}: contains floats")
            # TODO: return fail()

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
        session.commit()

    if not testcase.oclverified:
        print(f"testcase {testcase.id}: failed OCLgrind verification")
        return fail()


def verify_optimization_sensitive(session: session_t, tables: Tableset, result) -> None:
    """
    Check if a wrong-code result is optimization-sensitive, and ignore it
    if not.
    """
    params = result.testcase.params
    complement_params_id = session.query(tables.params.id)\
        .filter(tables.params.gsize_x == params.gsize_x,
                tables.params.gsize_y == params.gsize_y,
                tables.params.gsize_z == params.gsize_z,
                tables.params.optimizations == (not params.optimizations))\
        .scalar()

    complement_result = session.query(tables.results)\
                            .join(tables.testcases)\
                            .filter(tables.testcases.program_id == result.testcase.program_id,
                                    tables.testcases.params_id == complement_params_id)\
                            .first()
    if complement_result:
        if complement_result.stdout_id == result.stdout_id:
            print(f"retracting w-classification on {tables.name} result {result.id} - optimization insensitive")
            session.query(tables.classifications)\
                .filter(tables.classifications.id == result.id).delete()
    else:
        print(f"no complement result for {tables.name} result {result.id}")
        session.query(tables.classifications)\
            .filter(tables.classifications.id == result.id).delete()


def prune_w_classifications(session: session_t, tables: Tableset) -> None:

    # Verify testcases
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
            .filter(tables.testcases.id.in_(q))\
            .distinct()\
            .all()

    print(f"Verifying {tables.name} w-classified testcases ...")
    for testcase in ProgressBar()(testcases_to_verify):
        verify_w_testcase(session, tables, testcase)

    # Verify results
    results_to_verify = session.query(tables.results)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
            .all()

    print(f"Verifying {tables.name} w-classified optimization sensitivity ...")
    for result in ProgressBar()(results_to_verify):
        verify_optimization_sensitive(session, tables, result)

    # TODO: Do any of the other results for this testcase crash?


def verify_opencl_version(session: session_t, tables: Tableset, testcase) -> None:
    """
    OpenCL 2.0
    """
    opencl_2_0_testbeds = session.query(Testbed.id).filter(Testbed.opencl == "2.0")

    passes_2_0 = session.query(sql.sql.func.count(tables.results.id))\
        .filter(tables.results.testcase_id == testcase.id,
                tables.results.testbed_id.in_(opencl_2_0_testbeds),
                tables.results.outcome != OUTCOMES_TO_INT["bf"])\
        .scalar()

    if not passes_2_0:
        # If it didn't build on OpenCL 2.0, we're done.
        return

    passes_1_2 = session.query(sql.sql.func.count(tables.results.id))\
        .filter(tables.results.testcase_id == testcase.id,
                ~tables.results.testbed_id.in_(opencl_2_0_testbeds),
                tables.results.outcome == OUTCOMES_TO_INT["pass"])\
        .scalar()

    if passes_1_2:
        # If it *did* build on OpenCL 1.2, we're done.
        return

    ids_to_update = [
        x[0] for x in
        session.query(tables.results.id)\
            .join(tables.classifications)\
            .filter(tables.results.testcase_id == testcase.id,
                    ~tables.results.testbed_id.in_(opencl_2_0_testbeds),
                    tables.classifications.classification == CLASSIFICATIONS_TO_INT["bf"]).all()
    ]
    n = len(ids_to_update)
    if n:
        ids_str = ",".join(str(x) for x in ids_to_update)
        print(f"retracting bf-classification for testcase {testcase.id} - only passes on OpenCL 2.0. {n} results: {ids_str}")
        session.query(tables.classifications)\
            .filter(tables.classifications.id.in_(ids_to_update))\
            .delete(synchronize_session=False)


def prune_bf_classifications(session: session_t, tables: Tableset) -> None:
    # Retract results where 'double' is not supported
    ids_to_delete = [x[0] for x in session.query(tables.results.id)\
        .join(tables.classifications)\
        .join(tables.stderrs)\
        .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["bf"],
                tables.stderrs.stderr.like("%use of type 'double' requires cl_khr_fp64 extension%"))]

    n = len(ids_to_delete)
    if n:
        print(f"retracting {n} bf-classified {tables.name} results in which 'double' is not supported ...")
        session.query(tables.classifications)\
            .filter(tables.classifications.id.in_(ids_to_delete))\
            .delete(synchronize_session=False)

    # Retract results in which 'implicit function declaration'
    ids_to_delete = [x[0] for x in session.query(tables.results.id)\
        .join(tables.classifications)\
        .join(tables.stderrs)\
        .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["bf"],
                tables.stderrs.stderr.like("%implicit declaration of function%"))]

    n = len(ids_to_delete)
    if n:
        print(f"retracting {n} bf-classified {tables.name} results with implicit function declarations ...")
        session.query(tables.classifications)\
            .filter(tables.classifications.id.in_(ids_to_delete))\
            .delete(synchronize_session=False)
    session.commit()

    # Verify results
    q = session.query(tables.results.testcase_id)\
        .join(tables.classifications)\
        .join(Testbed)\
        .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["bf"],
                Testbed.opencl == "1.2")\
        .distinct()
    testcases_to_verify = session.query(tables.testcases)\
        .filter(tables.testcases.id.in_(q))\
        .distinct()\
        .all()

    print(f"Verifying {tables.name} bf-classified testcases ...")
    for testcase in ProgressBar()(testcases_to_verify):
        verify_opencl_version(session, tables, testcase)


def verify_c_testcase(session: session_t, tables: Tableset, testcase) -> None:
    """
    Verify that a testcase is sensible.
    """

    def fail():
        ids_to_update = [
            x[0] for x in
            session.query(tables.results.id)\
                .join(tables.classifications)\
                .filter(tables.results.testcase_id == testcase.id,
                        tables.classifications.classification == CLASSIFICATIONS_TO_INT["c"]).all()
        ]
        n = len(ids_to_update)
        assert n > 0
        ids_str = ",".join(str(x) for x in ids_to_update)
        print(f"retracting c-classification on {n} results: {ids_str}")
        session.query(tables.classifications)\
            .filter(tables.classifications.id.in_(ids_to_update))\
            .delete(synchronize_session=False)

    # CLgen-specific analysis. We can omit these checks for CLSmith, as they
    # will always pass.
    if tables.name == "CLgen":
        if testcase_raises_compiler_warnings(session, tables, testcase):
            print(f"testcase {testcase.id}: redflag compiler warnings")
            return fail()


def prune_c_classifications(session: session_t, tables: Tableset) -> None:
    # Verify testcases
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["c"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
            .filter(tables.testcases.id.in_(q))\
            .distinct()\
            .all()

    print(f"Verifying {tables.name} w-classified testcases ...")
    for testcase in ProgressBar()(testcases_to_verify):
        verify_c_testcase(session, tables, testcase)

    session.commit()


if __name__ == "__main__":
    parser = ArgumentParser(description="Collect difftest results for a device")
    parser.add_argument("-H", "--hostname", type=str, default="cc1",
                        help="MySQL database hostname")
    parser.add_argument("--clsmith", action="store_true",
                        help="analyze only clsmith results")
    parser.add_argument("--clgen", action="store_true",
                        help="analyze only clgen results")
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
            set_classifications(s, tableset)
            prune_w_classifications(s, tableset)
            prune_bf_classifications(s, tableset)
            prune_c_classifications(s, tableset)
