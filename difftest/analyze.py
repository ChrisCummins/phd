#!/usr/bin/env python
import cldrive
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


def get_cl_launcher_outcome(status: int, runtime: float, stderr: str) -> None:
    """
    Given a cl_launcher result, determine and set it's outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return OUTCOMES_TO_INT["c"] if "Compilation terminated successfully..." in stderr else OUTCOMES_TO_INT["bf"]
    def crash_or_build_crash():
        return OUTCOMES_TO_INT["c"] if "Compilation terminated successfully..." in stderr else OUTCOMES_TO_INT["bc"]
    def timeout_or_build_timeout():
        return OUTCOMES_TO_INT["to"] if "Compilation terminated successfully..." in stderr else OUTCOMES_TO_INT["bto"]

    if status == 0:
        return OUTCOMES_TO_INT["pass"]
    # 139 is SIGSEV
    elif status == 139 or status == -11:
        status = 139
        return crash_or_build_crash()
    # SIGTRAP
    elif status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif status == -9 and runtime >= 60:
        return timeout_or_build_timeout()
    elif status == -9:
        print(f"SIGKILL, but only ran for {runtime:.2f}s")
        return crash_or_build_crash()
    # SIGILL
    elif status == -4:
        return crash_or_build_crash()
    # SIGABRT
    elif status == -6:
        return crash_or_build_crash()
    # SIGFPE
    elif status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif status == -7:
        return crash_or_build_crash()
    # cl_launcher error
    elif status == 1:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-status).name)
        except ValueError:
            print(status)
        raise LookupError(f"failed to determine outcome of cl_launcher status {status} with stderr: {stderr}")


def get_cldrive_outcome(status: int, runtime: float, stderr: str) -> int:
    """
    Given a cldrive result, determine its outcome.

    See OUTCOMES for list of possible outcomes.
    """
    def crash_or_build_failure():
        return OUTCOMES_TO_INT["c"] if "[cldrive] Kernel: " in stderr else OUTCOMES_TO_INT["bf"]
    def crash_or_build_crash():
        return OUTCOMES_TO_INT["c"] if "[cldrive] Kernel: " in stderr else OUTCOMES_TO_INT["bc"]
    def timeout_or_build_timeout():
        return OUTCOMES_TO_INT["to"] if "[cldrive] Kernel: " in stderr else OUTCOMES_TO_INT["bto"]

    if status == 0:
        return OUTCOMES_TO_INT["pass"]
    # 139 is SIGSEV
    elif status == 139 or status == -11:
        status = 139
        return crash_or_build_crash()
    # SIGTRAP
    elif status == -5:
        return crash_or_build_crash()
    # SIGKILL
    elif status == -9 and runtime >= 60:
        return timeout_or_build_timeout()
    elif status == -9:
        print(f"SIGKILL, but only ran for {runtime:.2f}s")
        return crash_or_build_crash()
    # SIGILL
    elif status == -4:
        return crash_or_build_crash()
    # SIGFPE
    elif status == -8:
        return crash_or_build_crash()
    # SIGBUS
    elif status == -7:
        return crash_or_build_crash()
    # SIGABRT
    elif status == -6:
        return crash_or_build_crash()
    # cldrive error
    elif status == 1 and runtime >= 60:
        return timeout_or_build_timeout()
    elif status == 1:
        return crash_or_build_failure()
    # file not found (check the stderr on this one):
    elif status == 127:
        return crash_or_build_failure()
    else:
        print(result)
        try:
            print(Signals(-status).name)
        except ValueError:
            print(status)
        raise LookupError(f"failed to determine outcome of cldrive status {status} with stderr: {stderr}")


def set_classifications(session, tables: Tableset) -> None:
    """
    Determine anomalous results.
    """
    print(f"Resetting {tables.name} classifications ...", file=sys.stderr)
    session.execute(f"DELETE FROM {tables.classifications.__tablename__}")

    min_majsize = 7

    print(f"Setting {tables.name} {{bc,bto}} classifications ...", file=sys.stderr)
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT results.id, {CLASSIFICATIONS_TO_INT["bc"]}
FROM {tables.results.__tablename__} results
WHERE outcome = {OUTCOMES_TO_INT["bc"]}
""")
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT results.id, {CLASSIFICATIONS_TO_INT["bto"]}
FROM {tables.results.__tablename__} results
WHERE outcome = {OUTCOMES_TO_INT["bto"]}
""")

    print(f"Determining {tables.name} wrong-code classifications ...", file=sys.stderr)
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT results.id, {CLASSIFICATIONS_TO_INT["w"]}
FROM {tables.results.__tablename__} results
LEFT JOIN {tables.testcases.__tablename__} testcases ON results.testcase_id = testcases.id
LEFT JOIN {tables.majorities.__tablename__} majorities ON results.testcase_id = majorities.id
WHERE outcome = {OUTCOMES_TO_INT["pass"]}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
AND outcome_majsize >= {min_majsize}
AND stdout_majsize >= CEILING((2 * outcome_majsize) / 3)
AND stdout_id <> maj_stdout_id
""")

    print(f"Determining {tables.name} anomalous build-failures ...", file=sys.stderr)
    session.execute(f"""
INSERT INTO {tables.classifications.__tablename__}
SELECT results.id, {CLASSIFICATIONS_TO_INT["bf"]}
FROM {tables.results.__tablename__} results
LEFT JOIN {tables.testcases.__tablename__} ON results.testcase_id = {tables.testcases.__tablename__}.id
LEFT JOIN {tables.majorities.__tablename__} ON results.testcase_id = {tables.majorities.__tablename__}.id
WHERE outcome = {OUTCOMES_TO_INT["bf"]}
AND outcome_majsize >= {min_majsize}
AND maj_outcome = {OUTCOMES_TO_INT["pass"]}
""")

    print(f"Determining {tables.name} anomalous crashes ...", file=sys.stderr)
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

    print(f"Determining {tables.name} anomalous timeouts ...", file=sys.stderr)
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
            elif "comparison between pointer and integer" in stderr:
                print(f"testcase {testcase.id}: comparison between pointer and integer")
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
            elif "warning: array index" in stderr:
                print(f"testcase {testcase.id}: array index")
                testcase.compiler_warnings = True
                break
            elif "warning: implicit conversion from" in stderr:
                print(f"testcase {testcase.id}: implicit conversion")
                testcase.compiler_warnings = True
                break
            elif "array index -1 is before the beginning of the array" in stderr:
                print(f"testcase {testcase.id}: negative array index")
                testcase.compiler_warnings = True
                break
            elif "incompatible pointer" in stderr:
                print(f"testcase {testcase.id}: negative array index")
                testcase.compiler_warnings = True
                break
            elif "incompatible integer to pointer " in stderr:
                print(f"testcase {testcase.id}: incompatible integer to pointer")
                testcase.compiler_warnings = True
                break
            elif "warning" in stderr:
                print(f"\n UNRECOGNIZED WARNINGS in testcase {testcase.id}:")
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
            .delete(synchronize_session=False)

    # CLgen-specific analysis. We can omit these checks for CLSmith, as they
    # will always pass.
    if tables.name == "CLgen":
        if testcase.contains_floats == None:
            testcase.contains_floats = "float" in testcase.program.src

        if testcase.contains_floats:
            print(f"testcase {testcase.id}: contains floats")
            return fail()

        if testcase_raises_compiler_warnings(session, tables, testcase):
            print(f"testcase {testcase.id}: redflag compiler warnings")
            return fail()

        for arg in cldrive.extract_args(testcase.program.src):
            if arg.is_vector:
                # An error in my implementation of vector types:
                print(f"testcase {testcase.id}: contains vector types")
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
    if not oclgrind.verify_testcase(session, tables, testcase):
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
                tables.params.optimizations == (not params.optimizations))

    complement_testcase = session.query(tables.testcases.id)\
        .filter(tables.testcases.program_id == result.testcase.program_id,
                tables.testcases.params_id == complement_params_id)

    q = session.query(
            tables.results.id,
            tables.classifications.classification)\
        .join(tables.testcases)\
        .filter(tables.results.testbed_id == result.testbed_id,
                tables.results.testcase_id == complement_testcase)\
        .first()

    if q:
        complement_id, complement_classification = q
        if complement_classification == CLASSIFICATIONS_TO_INT["w"]:
            print(f"retracting w-classification on 2 {tables.name} results {{{result.id},{complement_id}}} - optimization insensitive")
            session.query(tables.classifications)\
                .filter(tables.classifications.id.in_([result.id, complement_id]))\
                .delete(synchronize_session=False)
    else:
        print(f"no complement result for {tables.name} result {result.id}")


def prune_w_classifications(session: session_t, tables: Tableset) -> None:

    print(f"Verifying {tables.name} w-classified testcases ...", file=sys.stderr)
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
            .filter(tables.testcases.id.in_(q))\
            .distinct()\
            .all()

    for testcase in ProgressBar()(testcases_to_verify):
        verify_w_testcase(session, tables, testcase)

    # print(f"Verifying {tables.name} w-classified optimization sensitivity ...", file=sys.stderr)
    # results_to_verify = session.query(tables.results)\
    #         .join(tables.classifications)\
    #         .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["w"])\
    #         .all()

    # for result in ProgressBar()(results_to_verify):
    #     verify_optimization_sensitive(session, tables, result)

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

    def prune_stderr_like(like):
        ids_to_delete = [x[0] for x in session.query(tables.results.id)\
            .join(tables.classifications)\
            .join(tables.stderrs)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["bf"],
                    tables.stderrs.stderr.like(f"%{like}%"))]

        n = len(ids_to_delete)
        if n:
            print(f"retracting {n} bf-classified {tables.name} results with msg {like[:30]}")
            session.query(tables.classifications)\
                .filter(tables.classifications.id.in_(ids_to_delete))\
                .delete(synchronize_session=False)

    prune_stderr_like("use of type 'double' requires cl_khr_fp64 extension")
    prune_stderr_like("implicit declaration of function")
    prune_stderr_like("function cannot have argument whose type is, or contains, type size_t")
    prune_stderr_like("unresolved extern function")
    # prune_stderr_like("error: declaration does not declare anything")
    prune_stderr_like("error: cannot increment value of type%")
    prune_stderr_like("subscripted access is not allowed for OpenCL vectors")
    prune_stderr_like("Images are not supported on given device")
    prune_stderr_like("error: variables in function scope cannot be declared")
    prune_stderr_like("error: implicit conversion ")
    # This is fine: prune_stderr_like("error: automatic variable qualified with an address space ")
    prune_stderr_like("Could not find a definition ")

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

    print(f"Verifying {tables.name} bf-classified testcases ...", file=sys.stderr)
    for testcase in ProgressBar()(testcases_to_verify):
        verify_opencl_version(session, tables, testcase)

    session.commit()


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
                        tables.classifications.classification == CLASSIFICATIONS_TO_INT["c"])\
                .all()
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
    if not oclgrind.verify_testcase(session, tables, testcase):
        print(f"testcase {testcase.id}: failed OCLgrind verification")
        return fail()


def prune_c_classifications(session: session_t, tables: Tableset) -> None:

    def prune_stderr_like(like):
        ids_to_delete = [x[0] for x in session.query(tables.results.id)\
            .join(tables.classifications)\
            .join(tables.stderrs)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["c"],
                    tables.stderrs.stderr.like(f"%{like}%"))]

        n = len(ids_to_delete)
        if n:
            print(f"retracting {n} c-classified {tables.name} results with msg {like[:30]}")
            session.query(tables.classifications)\
                .filter(tables.classifications.id.in_(ids_to_delete))\
                .delete(synchronize_session=False)

    prune_stderr_like("clFinish CL_INVALID_COMMAND_QUEUE")

    # Verify testcases
    q = session.query(tables.results.testcase_id)\
            .join(tables.classifications)\
            .filter(tables.classifications.classification == CLASSIFICATIONS_TO_INT["c"])\
            .distinct()
    testcases_to_verify = session.query(tables.testcases)\
            .filter(tables.testcases.id.in_(q))\
            .distinct()\
            .all()

    print(f"Verifying {tables.name} c-classified testcases ...", file=sys.stderr)
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
    print("connected to", db.init(db_hostname), file=sys.stderr)

    with Session(commit=True) as s:
        for tableset in tables:
            set_classifications(s, tableset)
            prune_w_classifications(s, tableset)
            prune_bf_classifications(s, tableset)
            prune_c_classifications(s, tableset)
